# src/pixels_to_players/webcam/processors.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import cv2, time, json
import mediapipe as mp
import numpy as np

class FaceMeshLogger:
    """ 
    Processor that runs Mediapipe FaceMesh with iris tracking on webcam frames and logs iris coordinates (x, y, timestamp)
    """
    def __init__(self, output_dir=Path(__file__).parent / "recordings", skip:int=1):
        """
        output_dir (Path): Directory to save iris tracking data.
        skip (int): Process every Nth frame (default=1 means every frame).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = None
        self.data = []
        self.frame_count = 0
        self.skip = skip

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, frame):
        try:
            if frame is None or frame.size == 0:
                print("[Warning] Empty frame received, skipping...")
                return frame
            
            # skip some frames for performance
            self.frame_count += 1
            if self.frame_count % self.skip != 0:
                return frame
            
            # reset time in case time diff between instantiation and recording
            if not hasattr(self, "start_time") or self.start_time is None:
                self.start_time = time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                for face_landmarks in results.multi_face_landmarks:
                    iris_left = face_landmarks.landmark[474]
                    iris_right = face_landmarks.landmark[469]
                    self.data.append({
                        "timestamp": round(time.time() - self.start_time, 3),
                        "left_iris": [iris_left.x * w, iris_left.y * h],
                        "right_iris": [iris_right.x * w, iris_right.y * h],
                    })

                    # draw iris model 
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )
        except Exception as e:
            print(f"[Error] FaceMeshLogger failed on frame {self.frame_count}: {e}")

        return frame

    def save(self):
        filename = self.output_dir / f"iris_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"Saved iris log: {filename}")

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