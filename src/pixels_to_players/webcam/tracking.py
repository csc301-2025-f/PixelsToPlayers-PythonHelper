import cv2
from .client import WebcamClient
from .processors import get_iris_center
import json
import numpy as np
from typing import Callable, Dict, List, Tuple


def fit_mapping(calibration_data: List[Dict[str, float]]) -> Callable[[Tuple[float, float]], Tuple[float, float]]:
    """
    Fit a linear mapping from eye coordinates to screen coordinates based on calibration data
    (JSON list of dicts with keys: screen_x, screen_y, eye_x, eye_y).

    Returns a function, <map_eye_to_screen>, that maps (eye_x, eye_y) to (screen_x, screen_y)
    """
    eye_coords = np.array([[d["eye_x"], d["eye_y"], 1] for d in calibration_data])
    screen_x = np.array([d["screen_x"] for d in calibration_data])
    screen_y = np.array([d["screen_y"] for d in calibration_data])
    
    # Solve linear regression
    a_x, b_x, c_x = np.linalg.lstsq(eye_coords, screen_x, rcond=None)[0]
    a_y, b_y, c_y = np.linalg.lstsq(eye_coords, screen_y, rcond=None)[0]
    
    def map_eye_to_screen(eye_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Map eye position (eye_x, eye_y) to screen position (screen_x, screen_y) using the fitted linear model
        Returns estimated screen coordinates (sx, sy).
        """
        ex, ey = eye_pos
        sx = a_x * ex + b_x * ey + c_x
        sy = a_y * ex + b_y * ey + c_y
        return sx, sy
    
    return map_eye_to_screen

def run_eye_tracking(input_json="demo_recordings/demo_calibration_data.json") -> None:
    """
    Run webcam and eye tracking with gaze overlay based on calibration data from input_json and linear mapping.
    Eye tracking is displayed as a red circle on the webcam feed. 
    Press 'q' to quit.

    * Default input_json is "demo_recordings/demo_calibration_data.json" for now.
    
    Returns None   
    """
    
    with open(input_json) as f:
        cali_data = json.load(f)
    
    map_eye_to_screen = fit_mapping(cali_data)

    with WebcamClient() as cam:
        width, height, _ = cam._actual_props()

        while True:
            frame = cam.snapshot()
            iris_center = get_iris_center(frame)

            if iris_center:
                screen_pos = map_eye_to_screen(iris_center)

                # Convert normalized coordinates to pixel coordinates
                x = int(screen_pos[0] * width)
                y = int(screen_pos[1] * height)

                # Draw gaze point (red circle) on frame
                cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)

            cv2.imshow("Gaze Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

# run_eye_tracking()