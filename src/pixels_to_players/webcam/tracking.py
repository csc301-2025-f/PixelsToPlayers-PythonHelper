import cv2
from .client import WebcamClient
from .processors import get_iris_center
import json
import numpy as np
from typing import Callable, Dict, List, Tuple


def fit_mapping(calibration_data: List[Dict[str, float]]) -> Callable[[Tuple[float, float]], Tuple[float, float]]:
    """
    Fit a linear mapping from eye coordinates (camera space) to screen coordinates (screen space).

    This function performs a least-squares linear regression using calibration data collected
    during the eye-tracking calibration process. It learns a mapping that predicts where on the
    screen the user is looking based on detected iris coordinates.

    Parameters
    ----------
    calibration_data : list of dict
        A list of dictionaries where each entry corresponds to a calibration point.
        Each dictionary must contain the following float keys:
            - "screen_x": normalized screen x-coordinate of the calibration point (0.0–1.0)
            - "screen_y": normalized screen y-coordinate of the calibration point (0.0–1.0)
            - "eye_x": x-coordinate of the detected iris position in the webcam frame
            - "eye_y": y-coordinate of the detected iris position in the webcam frame

    Returns
    -------
    map_eye_to_screen : Callable[[Tuple[float, float]], Tuple[float, float]]
        A function that takes an (eye_x, eye_y) tuple and returns the corresponding
        predicted (screen_x, screen_y) position using the fitted linear model.
    """

    eye_coords = np.array([[d["eye_x"], d["eye_y"], 1] for d in calibration_data])
    screen_x = np.array([d["screen_x"] for d in calibration_data])
    screen_y = np.array([d["screen_y"] for d in calibration_data])

    # Solve linear regression
    a_x, b_x, c_x = np.linalg.lstsq(eye_coords, screen_x, rcond=None)[0]
    a_y, b_y, c_y = np.linalg.lstsq(eye_coords, screen_y, rcond=None)[0]

    def map_eye_to_screen(eye_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Map eye position (eye_x, eye_y) to screen position (screen_x, screen_y)
        using the fitted linear regression model.

        Parameters
        ----------
        eye_pos : tuple of float
            The detected iris position in webcam coordinates, given as (eye_x, eye_y).

        Returns
        -------
        tuple of float
            The estimated screen coordinates (screen_x, screen_y), typically normalized to [0, 1].
        """
        ex, ey = eye_pos
        sx = a_x * ex + b_x * ey + c_x
        sy = a_y * ex + b_y * ey + c_y
        return sx, sy

    return map_eye_to_screen

def run_eye_tracking(input_json="demo_recordings/demo_calibration_data.json") -> None:
    """
    Perform real-time eye tracking using a webcam and a pre-trained linear gaze-mapping model.

    This function loads calibration data, fits a linear model that maps eye positions to
    screen coordinates, and then uses the webcam feed to track the user's gaze in real time.
    The predicted gaze point is displayed as a red circle on the webcam video feed.
    This function runs continuously until the user presses the 'q' key.

    Parameters
    ----------
    input_json : str, optional
        Path to the JSON file containing calibration data, by default
        "demo_recordings/demo_calibration_data.json".

    Returns
    -------
    None
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
