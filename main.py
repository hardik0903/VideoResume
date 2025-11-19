import os
import tempfile
import random
import shutil
from typing import List, Optional
from pathlib import Path

import numpy as np
import cv2
import tensorflow as tf
import librosa

# MoviePy v2 import syntax
from moviepy import VideoFileClip 

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

# --- 1. CONFIGURATION ---

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI(title="Ascraa - Combined Deepfake Detection",
              description="Checks both Visual and Audio authenticity with user-defined thresholds.")

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
# Looks for weights/Meso4_DF.h5 relative to this script
VIDEO_WEIGHTS_PATH = Path(os.environ.get("MESO_WEIGHTS_PATH", str(SCRIPT_DIR / "weights" / "Meso4_DF.h5"))).resolve()
AUDIO_MODEL_PATH = "audio_classifier.h5"

# Audio Processing Constants (Architecture specific, unlikely to change per request)
AUDIO_SR = 22050
N_MELS = 128
TARGET_FRAMES = 109
HOP_LENGTH = 512
N_FFT = 2048

# Global Model Holders
_video_model_instance = None
_audio_model_instance = None


# --- 2. HELPER FUNCTIONS ---

def _sample_frame_indices(total_frames: int, k: int) -> List[int]:
    if total_frames <= 0:
        return []
    k = min(k, total_frames)
    indices = random.sample(range(total_frames), k)
    indices.sort()
    return indices

def _preprocess_frame(frame_bgr, img_size=256) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return frame_resized.astype(np.float32) / 255.0

def load_video_model():
    global _video_model_instance
    if _video_model_instance is not None:
        return _video_model_instance

    try:
        from classifiers import Meso4
    except ImportError:
        print("Error: 'classifiers.py' not found.")
        return None
    
    if not VIDEO_WEIGHTS_PATH.exists():
        print(f"Error: Video weights not found at {VIDEO_WEIGHTS_PATH}")
        return None

    try:
        model = Meso4()
        model.load(str(VIDEO_WEIGHTS_PATH))
        _video_model_instance = model
        print("Video Model loaded.")
    except Exception as e:
        print(f"Failed to load Meso4 weights: {e}")
        return None
        
    return _video_model_instance

def load_audio_model():
    global _audio_model_instance
    if _audio_model_instance is not None:
        return _audio_model_instance

    try:
        _audio_model_instance = tf.keras.models.load_model(AUDIO_MODEL_PATH, compile=False)
        print("Audio Model loaded.")
    except Exception as e:
        print(f"Failed to load Audio model: {e}")
        _audio_model_instance = None
    
    return _audio_model_instance

def extract_audio_from_video(video_path: str, out_audio_path: str, sr: int = AUDIO_SR):
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return False
        clip.audio.write_audiofile(out_audio_path, fps=sr, nbytes=2, codec="pcm_s16le", logger=None)
        clip.close()
        return True
    except Exception:
        return False

def audio_to_log_mel(path_wav: str):
    y, sr_loaded = librosa.load(path_wav, sr=AUDIO_SR, mono=True)
    if y.size == 0: raise RuntimeError("Empty audio")
    
    S = librosa.feature.melspectrogram(y=y, sr=AUDIO_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    T = S_db.shape[1]
    if T < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - T
        min_val = np.min(S_db)
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=min_val)
    elif T > TARGET_FRAMES:
        start = max(0, (T - TARGET_FRAMES) // 2)
        S_db = S_db[:, start:start + TARGET_FRAMES]
        
    mean = S_db.mean()
    std = S_db.std() + 1e-9
    S_db = (S_db - mean) / std
    return S_db

# --- 3. ENDPOINT ---

@app.post("/check_deepfake")
async def check_deepfake(
    file: UploadFile = File(...),
    video_num_frames: int = Query(50, description="Number of frames to sample"),
    video_threshold: float = Query(0.5, description="Threshold for video realness (Score > Threshold = Real)"),
    audio_threshold: float = Query(0.5, description="Threshold for audio fakeness (Score > Threshold = Fake)")
):
    
    tmp_video_path = None
    tmp_audio_path = None
    
    # Initialize response structure
    response_data = {
        "filename": file.filename,
        "video_analysis": {
            "per_frame_scores": [],
            "avg_score": 0.0,
            "is_true": False
        },
        "audio_analysis": {
            "score": 0.0,
            "is_true": False
        }
    }
    
    try:
        # Save Temp File
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_vid:
            tmp_video_path = tmp_vid.name
            content = await file.read()
            tmp_vid.write(content)

        # --- VIDEO ANALYSIS ---
        video_model = load_video_model()
        if video_model:
            cap = cv2.VideoCapture(tmp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            sampled_frames = []

            # Extract Frames
            indices = _sample_frame_indices(total_frames, video_num_frames)
            if not indices and total_frames == 0:
                # Fallback if frame count extraction fails
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    sampled_frames.append(frame)
                    if len(sampled_frames) >= video_num_frames: break
            else:
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret: sampled_frames.append(frame)
            cap.release()

            if sampled_frames:
                batch = np.stack([_preprocess_frame(f) for f in sampled_frames], axis=0)
                
                # Predict
                try:
                    preds = video_model.predict(batch)
                except:
                    preds = video_model.predict(batch.astype(np.float32))
                
                # Flatten scores
                scores_list = np.asarray(preds).reshape(-1).tolist()
                avg_score = float(np.mean(scores_list))
                
                # LOGIC: Based on your previous code "is_true = avg_score > FAKE_THRESHOLD"
                # This assumes High Score = Real.
                is_video_true = avg_score > video_threshold

                response_data["video_analysis"] = {
                    "per_frame_scores": scores_list,
                    "avg_score": avg_score,
                    "is_true": is_video_true
                }
            else:
                # No frames found
                 response_data["video_analysis"] = {"error": "No frames extracted", "is_true": False}
        else:
            response_data["video_analysis"] = {"error": "Video Model failed to load", "is_true": False}

        # --- AUDIO ANALYSIS ---
        audio_model = load_audio_model()
        if audio_model:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_aud:
                tmp_audio_path = tmp_aud.name
            
            has_audio = extract_audio_from_video(tmp_video_path, tmp_audio_path)
            
            if has_audio:
                try:
                    S_db = audio_to_log_mel(tmp_audio_path)
                    x = np.expand_dims(S_db, axis=(0, -1)).astype(np.float32)
                    audio_preds = audio_model.predict(x)
                    
                    # Extract single score
                    aud_score = 0.0
                    if audio_preds.ndim == 2 and audio_preds.shape[1] == 1:
                        aud_score = float(audio_preds[0, 0])
                    elif audio_preds.ndim == 2 and audio_preds.shape[1] == 2:
                        aud_score = float(audio_preds[0, 1]) 
                    else:
                        aud_score = float(audio_preds.flatten()[0])
                    
                    # LOGIC: Usually Audio models output "Probability of Fake".
                    # If Score > Threshold => Fake.
                    # Therefore is_true => Score < Threshold.
                    is_audio_true = aud_score < audio_threshold
                    
                    response_data["audio_analysis"] = {
                        "score": aud_score,
                        "is_true": is_audio_true
                    }
                except Exception as e:
                    response_data["audio_analysis"] = {"error": str(e), "is_true": False}
            else:
                response_data["audio_analysis"] = {"message": "No audio detected", "is_true": True, "score": 0.0}
        else:
            response_data["audio_analysis"] = {"error": "Audio Model failed to load", "is_true": False}

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        if tmp_video_path and os.path.exists(tmp_video_path):
            try: os.remove(tmp_video_path)
            except: pass
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            try: os.remove(tmp_audio_path)
            except: pass