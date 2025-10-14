# src/pixels_to_players/webcam/processors.py
from __future__ import annotations
import cv2
import numpy as np

# Keep these pure and fast; easy to unit-test and compose.

def flip_horizontal(frame: np.ndarray) -> np.ndarray:
    """Mirror the frame (selfie style)."""
    return cv2.flip(frame, 1)

def to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def resize(frame: np.ndarray, width: int | None = None, height: int | None = None) -> np.ndarray:
    """Resize while preserving aspect if one dimension is None."""
    h, w = frame.shape[:2]
    if width is None and height is None:
        return frame
    if width is None:
        scale = height / h
        width = int(w * scale)
    elif height is None:
        scale = width / w
        height = int(h * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay FPS text on the frame (utility for quick debugging)."""
    out = frame.copy()
    cv2.putText(out, f"{fps:.1f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return out

def get_iris_center(frame: np.ndarray) -> tuple[float, float] | None:
    """Return average iris center (x, y) in pixel coordinates, or None if not detected."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    h, w, _ = frame.shape
    left_ids = [468, 469, 470, 471]
    right_ids = [473, 474, 475, 476]

    for face_landmarks in results.multi_face_landmarks:
        lx = np.mean([face_landmarks.landmark[i].x for i in left_ids]) * w
        ly = np.mean([face_landmarks.landmark[i].y for i in left_ids]) * h
        rx = np.mean([face_landmarks.landmark[i].x for i in right_ids]) * w
        ry = np.mean([face_landmarks.landmark[i].y for i in right_ids]) * h

        cx = (lx + rx) / 2
        cy = (ly + ry) / 2
        return (cx, cy)