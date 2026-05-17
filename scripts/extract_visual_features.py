"""Extract per-frame face features from an AMI Closeup video.

Phase 3 smoke test: feed one Closeup{N}.avi through MediaPipe Face Landmarker,
emit a (T, D) numpy array of features that mirror the MM-VAP feature set
(FAU-like blendshapes + head pose + gaze + mouth openness).

MediaPipe Face Landmarker model is auto-downloaded to ``models/`` on first run.

Usage::

    python scripts/extract_visual_features.py \\
        --video data/raw/ami/ES2002a/video/ES2002a.Closeup1.avi \\
        --out data/processed/visual/ES2002a.Closeup1.npy
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# Feature layout: 52 blendshapes + 3 head Euler + 1 mouth openness = 56 dims
FEATURE_DIM = 56


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face_landmarker model to {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def make_landmarker():
    base = mp_python.BaseOptions(model_asset_path=str(ensure_model()))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def matrix_to_euler(m: np.ndarray) -> tuple[float, float, float]:
    """Extract pitch/yaw/roll (radians) from a 4x4 facial transformation matrix."""
    r = m[:3, :3]
    sy = float(np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    if sy > 1e-6:
        pitch = float(np.arctan2(r[2, 1], r[2, 2]))
        yaw = float(np.arctan2(-r[2, 0], sy))
        roll = float(np.arctan2(r[1, 0], r[0, 0]))
    else:
        pitch = float(np.arctan2(-r[1, 2], r[1, 1]))
        yaw = float(np.arctan2(-r[2, 0], sy))
        roll = 0.0
    return pitch, yaw, roll


def mouth_openness(landmarks) -> float:
    """Vertical distance between upper / lower inner lip landmarks (normalized)."""
    # MediaPipe Face Mesh indices: upper inner lip = 13, lower inner lip = 14
    if not landmarks:
        return 0.0
    upper = landmarks[13]
    lower = landmarks[14]
    return float(abs(upper.y - lower.y))


def extract_features(video_path: Path) -> np.ndarray:
    landmarker = make_landmarker()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {video_path.name}: {n_total} frames @ {fps:.1f} fps")

    features: list[np.ndarray] = []
    no_face_count = 0
    t_start = time.time()
    frame_idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(frame_idx * 1000 / fps)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        feat = np.zeros(FEATURE_DIM, dtype=np.float32)
        if result.face_blendshapes:
            bs = result.face_blendshapes[0]
            for i, b in enumerate(bs[:52]):
                feat[i] = b.score
            if result.facial_transformation_matrixes:
                pitch, yaw, roll = matrix_to_euler(result.facial_transformation_matrixes[0])
                feat[52] = pitch
                feat[53] = yaw
                feat[54] = roll
            if result.face_landmarks:
                feat[55] = mouth_openness(result.face_landmarks[0])
        else:
            no_face_count += 1
        features.append(feat)
        frame_idx += 1

    cap.release()
    landmarker.close()

    arr = np.stack(features, axis=0)
    elapsed = time.time() - t_start
    print(
        f"  done: {arr.shape}, {elapsed:.1f}s ({frame_idx / elapsed:.1f} fps), "
        f"no-face on {no_face_count}/{frame_idx} frames"
    )
    return arr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    feats = extract_features(args.video)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, feats)
    print(f"Saved {feats.shape} → {args.out}")


if __name__ == "__main__":
    main()
