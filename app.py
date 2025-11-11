import os
import tempfile
import random
from typing import List, Optional
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="ascraa deepfake check")

script_dir = Path(__file__).resolve().parent
default_weights = script_dir / "weights" / "Meso4_DF.h5"
WEIGHTS_PATH = Path(os.environ.get("MESO_WEIGHTS_PATH", str(default_weights))).resolve()

IMG_SIZE = 256
NUM_FRAMES = 50
FAKE_THRESHOLD = 0.33  # avg score below this -> fake

_model_instance = None


def _sample_frame_indices(total_frames: int, k: int) -> List[int]:
    if total_frames <= 0:
        return []
    k = min(k, total_frames)
    # if k == total_frames, random.sample still works
    indices = random.sample(range(total_frames), k)
    indices.sort()
    return indices


def _preprocess_frame(frame_bgr) -> np.ndarray:
    # Convert BGR -> RGB, resize to IMG_SIZE x IMG_SIZE and scale to [0,1]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return frame_resized.astype(np.float32) / 255.0


def load_model(weights_path: Optional[Path] = None):
    """
    Lazy import and load the Meso4 model once. Returns cached instance on subsequent calls.
    Raises FileNotFoundError if weights_path doesn't exist.
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    # Lazy import so TensorFlow/protobuf errors don't happen at module import time
    try:
        from classifiers import Meso4  # your classifiers.py must define Meso4
    except Exception as e:
        raise RuntimeError(f"failed to import classifiers.Meso4: {e}")

    weights_path = Path(weights_path or WEIGHTS_PATH)
    if not weights_path.exists():
        raise FileNotFoundError(f"Meso4 weights not found at: {weights_path}")

    model = Meso4()
    # Attempt to call common load APIs
    try:
        # preferred: classifier provides load(path)
        model.load(str(weights_path))
    except AttributeError:
        # fallback: Keras model inside classifier
        try:
            model.model.load_weights(str(weights_path))
        except Exception as e:
            raise RuntimeError(f"failed to load weights via fallback: {e}")
    except Exception as e:
        raise RuntimeError(f"failed to load model weights: {e}")

    _model_instance = model
    return _model_instance

@app.on_event("startup")
def _startup():
    print("@@@ server started @@@")
    print("app.py loaded from:", __file__)
    print("WEIGHTS_PATH:", WEIGHTS_PATH)
    print("does weights exists?:", WEIGHTS_PATH.exists())
    print("IMG_SIZE, NUM_FRAMES, FAKE_THRESHOLD:", IMG_SIZE, NUM_FRAMES, FAKE_THRESHOLD)
    # Attempt to preload model so errors appear early (optional)
    try:
        load_model(WEIGHTS_PATH)
        print("Meso4 model loaded at startup (cached).")
    except Exception as e:
        print("Warning: model not loaded at startup:", repr(e))


@app.get("/")
def root():
    return {"status": "ok", "routes": ["/predict (POST)"]}


@app.get("/health")
def health():
    ok = WEIGHTS_PATH.exists()
    return {"ok": ok, "weights_path": str(WEIGHTS_PATH), "weights_exists": ok}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts multipart file upload with key 'file'.
    Returns per-sampled-frame scores, average, and final `is_fake` boolean.
    """
    # save upload to temp file (OpenCV needs a path)
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not save uploaded file: {e}")

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sampled_frames = []

        if total_frames <= 0:
            # fallback: read all frames
            frames_tmp = []
            cap.release()
            cap = cv2.VideoCapture(tmp_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_tmp.append(frame)
            total_frames = len(frames_tmp)
            if total_frames == 0:
                raise HTTPException(status_code=400, detail="video contains no frames")
            indices = _sample_frame_indices(total_frames, NUM_FRAMES)
            sampled_frames = [frames_tmp[i] for i in indices]
        else:
            indices = _sample_frame_indices(total_frames, NUM_FRAMES)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                sampled_frames.append(frame)

        cap.release()

        if len(sampled_frames) == 0:
            raise HTTPException(status_code=400, detail="no frames could be read from the video")

        # Preprocess and batch
        batch = np.stack([_preprocess_frame(f) for f in sampled_frames], axis=0)  # (N, H, W, 3)

        # Load model (cached)
        try:
            model = load_model(WEIGHTS_PATH)
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"model load error: {e}")

        # Prediction - handle different possible return shapes
        try:
            preds = model.predict(batch)
        except Exception as e:
            # try passing batch.astype(np.float32)
            try:
                preds = model.predict(batch.astype(np.float32))
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"model prediction failed: {e}; fallback failed: {e2}")

        preds = np.asarray(preds).reshape(-1).tolist()
        avg_score = float(np.mean(preds))
        is_true = bool(avg_score > FAKE_THRESHOLD)

        return JSONResponse(
            content={
                "file_name": file.filename,
                "num_sampled_frames": len(preds),
                "per_frame_scores": preds,
                "avg_score": avg_score,
                "threshold": FAKE_THRESHOLD,
                "is_true": is_true,
            }
        )
    finally:
        # cleanup temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass
