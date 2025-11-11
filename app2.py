#!/usr/bin/env python3
"""
app2.py

Usage:
    python app2.py --folder /path/to/videos [--weights /abs/path/to/Meso4_DF.h5] [--out out_dir] [--frames 50] [--simulate]

Examples:
    python app2.py --folder ./videos
    python app2.py --folder "C:\Users\Hardik\Videos\clips" --weights "C:\...Meso4_DF.h5"
    python app2.py --folder ./videos --simulate   # generate random scores (no TF needed)
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple
import csv

import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---- Config defaults ----
DEFAULT_IMG_SIZE = 256
DEFAULT_NUM_FRAMES = 50
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".mpg", ".mpeg", ".wmv"}


# ---- Model loader (lazy, cached) ----
_model = None


def load_meso4(weights_path: Optional[Path] = None):
    """
    Lazy load and cache Meso4 classifier from classifiers.py.
    Raises FileNotFoundError if weights_path provided and not found.
    """
    global _model
    if _model is not None:
        return _model

    if weights_path is not None:
        weights_path = Path(weights_path)

    try:
        # import here so TensorFlow isn't required until you actually load
        from classifiers import Meso4
    except Exception as e:
        raise RuntimeError(f"failed to import classifiers.Meso4: {e}")

    model = Meso4()
    if weights_path:
        if not weights_path.exists():
            raise FileNotFoundError(f"Meso4 weights not found at: {weights_path}")
        # many classifier implementations provide .load(path) or .model.load_weights
        try:
            model.load(str(weights_path))
        except AttributeError:
            # fallback to keras model attribute
            try:
                model.model.load_weights(str(weights_path))
            except Exception as e:
                raise RuntimeError(f"failed to load weights: {e}")
        except Exception as e:
            raise RuntimeError(f"failed to load weights: {e}")

    _model = model
    return _model


# ---- Frame sampling / preprocessing ----
def sample_frame_indices(total_frames: int, k: int) -> List[int]:
    if total_frames <= 0:
        return []
    k = min(k, total_frames)
    # Use random.sample for unique indices and shuffle them sorted for nicer visuals
    inds = random.sample(range(total_frames), k)
    inds.sort()
    return inds


def preprocess_frame(frame_bgr, img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    # BGR -> RGB, resize, scale to [0,1], dtype float32
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


# ---- Video processing ----
def read_sampled_frames(video_path: Path, num_frames: int, img_size: int) -> List[np.ndarray]:
    """
    Read up to num_frames sampled frames from video_path.
    Returns list of raw BGR frames (not preprocessed).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames = []
    if total_frames <= 0:
        # fallback: read all frames into memory (costly)
        tmp = []
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            tmp.append(frame)
        total_frames = len(tmp)
        if total_frames == 0:
            cap.release()
            raise IOError(f"video contains no frames: {video_path}")
        indices = sample_frame_indices(total_frames, num_frames)
        frames = [tmp[i] for i in indices]
    else:
        indices = sample_frame_indices(total_frames, num_frames)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                # skip if failed
                continue
            frames.append(frame)

    cap.release()
    if len(frames) == 0:
        raise IOError(f"no frames could be read from: {video_path}")
    return frames


def predict_on_frames(model, batch: np.ndarray) -> np.ndarray:
    """
    Run model.predict on a batch of preprocessed frames.
    Try to handle common output shapes: (N,), (N,1), (N,2).
    If (N,2) - take column 1 as the 'real' probability if available; else compute softmax's second column.
    """
    preds = model.predict(batch)
    preds = np.asarray(preds)

    if preds.ndim == 1:
        return preds.astype(np.float32)
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds.reshape(-1).astype(np.float32)
    if preds.ndim == 2 and preds.shape[1] == 2:
        # Often models output [prob_fake, prob_real] or [prob_real, prob_fake].
        # Heuristic: take column with larger average as "real" prob? That's risky.
        # Safer: if values appear like logits, apply softmax and take second column.
        try:
            from scipy.special import softmax
            probs = softmax(preds, axis=1)[:, 1]
            return probs.astype(np.float32)
        except Exception:
            # fallback - take second column
            return preds[:, 1].astype(np.float32)
    # other shapes: try flattening first column
    return preds.reshape(-1).astype(np.float32)


# ---- Plotting utilities ----
def save_frame_scores_plot(scores: List[float], out_path: Path, title: str):
    plt.figure(figsize=(8, 4))
    plt.plot(scores)          # do not set explicit colors
    plt.xlabel("sample index")
    plt.ylabel("score")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


def save_avg_scores_bar(names: List[str], avgs: List[float], out_path: Path):
    plt.figure(figsize=(max(6, len(names) * 0.6), 5))
    x = np.arange(len(names))
    plt.bar(x, avgs)          # no explicit colors
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("average score")
    plt.title("Average scores per video")
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


# ---- Main runner ----
def process_folder(
    folder: Path,
    out_dir: Path,
    weights_path: Optional[Path] = None,
    num_frames: int = DEFAULT_NUM_FRAMES,
    img_size: int = DEFAULT_IMG_SIZE,
    simulate: bool = False,
):
    folder = Path(folder)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    if not video_files:
        raise SystemExit(f"No video files found in: {folder}")

    print(f"found {len(video_files)} video files, output -> {out_dir}")

    model = None
    if not simulate:
        try:
            model = load_meso4(weights_path)
            print("Loaded Meso4 model.")
        except Exception as e:
            raise SystemExit(f"Failed to load model: {e}\nUse --simulate to run without model.")

    summary = []
    avg_scores = []
    names = []

    for vid in video_files:
        print(f"\nprocessing: {vid.name}")
        try:
            frames_bgr = read_sampled_frames(vid, num_frames, img_size)
        except Exception as e:
            print(f"  skipped (read error): {e}")
            continue

        # preprocess
        batch = np.stack([preprocess_frame(f, img_size) for f in frames_bgr], axis=0).astype(np.float32)

        if simulate:
            # random values in [0,1]
            preds = np.random.rand(batch.shape[0]).astype(np.float32)
            print(f"  simulated {len(preds)} scores")
        else:
            try:
                preds = predict_on_frames(model, batch)
            except Exception as e:
                print(f"  model prediction failed: {e}")
                continue

        # per-frame graph
        stem = vid.stem
        frame_plot = out_dir / f"{stem}_frames.png"
        save_frame_scores_plot(preds.tolist(), frame_plot, title=f"{stem} - frame scores")
        print(f"  saved per-frame plot -> {frame_plot.name}")

        avg = float(np.mean(preds))
        avg_scores.append(avg)
        names.append(stem)
        is_fake = avg < 0.33  # same threshold you used
        summary.append({"file": vid.name, "avg_score": avg, "num_samples": len(preds), "is_fake": is_fake})

    # overall avg bar chart
    if names:
        avg_plot = out_dir / "avg_scores.png"
        save_avg_scores_bar(names, avg_scores, avg_plot)
        print(f"\nsaved average-scores bar chart -> {avg_plot.name}")

        # write CSV summary
        csv_path = out_dir / "summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["file", "avg_score", "num_samples", "is_fake"])
            writer.writeheader()
            for r in summary:
                writer.writerow(r)
        print(f"wrote summary -> {csv_path.name}")
    else:
        print("No videos processed successfully; nothing to plot.")


# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser(description="Batch run Meso4 on a folder of videos and produce plots.")
    p.add_argument("--folder", "-f", required=True, help="Folder containing only video clips.")
    p.add_argument("--weights", "-w", default=None, help="Path to Meso4 weights (.h5). Optional if --simulate.")
    p.add_argument("--out", "-o", default="out_graphs", help="Output directory for graphs and CSV.")
    p.add_argument("--frames", type=int, default=DEFAULT_NUM_FRAMES, help="Number of frames to sample per video.")
    p.add_argument("--imgsize", type=int, default=DEFAULT_IMG_SIZE, help="Size to resize frames to before predict.")
    p.add_argument("--simulate", action="store_true", help="Simulate predictions with random scores (no TF).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    folder = Path(args.folder).resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"folder does not exist: {folder}")

    weights = Path(args.weights).resolve() if args.weights else None

    try:
        process_folder(
            folder=folder,
            out_dir=Path(args.out).resolve(),
            weights_path=weights,
            num_frames=args.frames,
            img_size=args.imgsize,
            simulate=args.simulate,
        )
        print("\ndone.")
    except Exception as e:
        print("ERROR:", e)
        raise
