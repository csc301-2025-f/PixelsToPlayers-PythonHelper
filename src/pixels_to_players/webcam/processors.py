"""
Frame Processing Utilities

This module defines lightweight, composable image processors for use with
`WebcamClient` and other computer vision pipelines. Each processor is a pure
function that takes a frame (`np.ndarray`) and returns a modified frame,
allowing flexible chaining and real-time transformations.

Functions include:
    - Landmark visualization using Mediapipe FaceMesh
    - Iris center detection
    - Frame flipping, resizing, and grayscale conversion
    - FPS overlay for debugging

These processors are intentionally fast and stateless, making them suitable for
unit testing, live preview pipelines, and data logging in gaze tracking or
facial analysis workflows.

Example:
    from pixels_to_players.webcam.processors import flip_horizontal, draw_facemesh
    from pixels_to_players.webcam.client import WebcamClient

    client = WebcamClient()
    client.record(duration=5.0, processors=[flip_horizontal, draw_facemesh])

Dependencies:
    - opencv-python: Image conversion, drawing, and transformation utilities
    - mediapipe: FaceMesh and iris landmark detection
    - numpy: Vectorized math and coordinate operations
"""

from __future__ import annotations
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global Mediapipe FaceMesh instance for lightweight frame processing
_face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def draw_facemesh(frame: np.ndarray) -> np.ndarray:
    """
    Draw Mediapipe FaceMesh landmarks, contours, and iris connections.

    Args:
        frame (np.ndarray): Input BGR image.

    Returns:
        np.ndarray: Output frame with face mesh annotations drawn in place.

    Notes:
        - Performs internal RGB conversion for Mediapipe.
        - Preserves input frame shape.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
    return frame


def flip_horizontal(frame: np.ndarray) -> np.ndarray:
    """
    Mirror the frame horizontally (selfie-style).

    Args:
        frame (np.ndarray): Input image.

    Returns:
        np.ndarray: Horizontally flipped frame.
    """
    return cv2.flip(frame, 1)


def to_gray(frame: np.ndarray) -> np.ndarray:
    """
    Convert a color frame (BGR) to grayscale.

    Args:
        frame (np.ndarray): Input BGR image.

    Returns:
        np.ndarray: Grayscale version of the frame.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def resize(frame: np.ndarray, width: int | None = None, height: int | None = None) -> np.ndarray:
    """
    Resize a frame to the given width and/or height while preserving aspect ratio.

    Args:
        frame (np.ndarray): Input image.
        width (int | None): Desired width in pixels, or None to infer from height.
        height (int | None): Desired height in pixels, or None to infer from width.

    Returns:
        np.ndarray: Resized image.
    """
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
    """
    Overlay current frames-per-second (FPS) value on the frame.

    Args:
        frame (np.ndarray): Input image.
        fps (float): Measured FPS value to display.

    Returns:
        np.ndarray: Frame with FPS text drawn.
    """
    out = frame.copy()
    cv2.putText(out, f"{fps:.1f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def get_iris_center(frame: np.ndarray) -> tuple[float, float] | None:
    """
    Estimate the average iris center coordinates from the detected face mesh.

    Args:
        frame (np.ndarray): Input BGR image.

    Returns:
        tuple[float, float] | None: Pixel coordinates (x, y) of the iris center,
        or None if no face or irises are detected.

    Notes:
        - Uses Mediapipe landmark indices 468–471 (left eye) and 473–476 (right eye).
        - Returns the midpoint between the two iris centers.
    """
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
