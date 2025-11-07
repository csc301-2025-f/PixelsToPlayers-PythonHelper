"""
Webcam Calibration Module

This module provides functionality for calibrating eye tracking systems by
collecting gaze data at known screen positions. It displays calibration points
on screen, tracks iris positions via webcam, and saves the mapping between
screen coordinates and eye positions for later use in gaze estimation.

The calibration process displays a series of points on screen (typically at
corners and center), prompts the user to look at each point, and records the
corresponding iris center positions. The collected data is averaged and saved
to a JSON file for use in gaze mapping algorithms.

Example:
    Basic usage:
        from pixels_to_players.webcam.calibration import run_calibration
        
        # Run calibration with default settings
        run_calibration()
        
        # Custom duration and output file
        run_calibration(duration_per_point=3.0, output_file="my_calibration.json")

Dependencies:
    - opencv-python: For video capture, frame display, and drawing operations
    - numpy: For array operations and averaging iris center positions
    - pixels_to_players.webcam.client: For webcam access and frame capture
    - pixels_to_players.webcam.processors: For iris detection and center calculation
"""

import cv2
import time
import json
import numpy as np
from .client import WebcamClient
from .processors import get_iris_center

DOT_RADIUS = 20
IRIS_RADIUS = 5

# Normalized calibration points (x, y)
CALIB_POINTS = [
    (0.1, 0.1),
    (0.9, 0.1),
    (0.1, 0.9),
    (0.9, 0.9),
    (0.5, 0.5)
]

def show_dot(frame: np.ndarray, screen_width: int, screen_height: int, normalized_x: float, normalized_y: float) -> np.ndarray:
    """
    Draw a red calibration dot on the frame at the specified normalized coordinates.
    
    Converts normalized screen coordinates (0.0-1.0) to pixel coordinates and
    draws a filled red circle at that position. The dot is used as a visual
    target during calibration to guide the user's gaze.
    
    Args:
        frame (np.ndarray): The video frame to draw on (modified in place).
        screen_width (int): Width of the screen/display in pixels.
        screen_height (int): Height of the screen/display in pixels.
        normalized_x (float): X coordinate normalized to [0.0, 1.0] range.
        normalized_y (float): Y coordinate normalized to [0.0, 1.0] range.
    
    Returns:
        np.ndarray: The frame with the calibration dot drawn (same reference as input).
    
    Example:
        frame = cam.snapshot()
        frame = show_dot(frame, 1920, 1080, 0.5, 0.5)  # Center dot
    """
    x = int(normalized_x * screen_width)
    y = int(normalized_y * screen_height)
    cv2.circle(frame, (x, y), DOT_RADIUS, (0, 0, 255), -1)
    return frame

def run_calibration(duration_per_point=2.0, output_file="calibration_data.json"):
    """
    Run the eye tracking calibration procedure.
    
    This function displays a series of calibration points on screen in fullscreen
    mode, tracks the user's iris position via webcam while they look at each point,
    and saves the collected data to a JSON file. The calibration window displays
    red dots at predefined screen positions, and the user clicks when they are
    looking at each point. Iris center positions are collected and averaged for
    each calibration point.
    
    The calibration points are displayed in the following order:
    - Top-left corner (0.1, 0.1)
    - Top-right corner (0.9, 0.1)
    - Bottom-left corner (0.1, 0.9)
    - Bottom-right corner (0.9, 0.9)
    - Center (0.5, 0.5)
    
    The output JSON file contains an array of objects, each with:
    - screen_x: Normalized X screen coordinate (0.0-1.0)
    - screen_y: Normalized Y screen coordinate (0.0-1.0)
    - eye_x: Average iris center X coordinate in pixels
    - eye_y: Average iris center Y coordinate in pixels
    
    Args:
        duration_per_point (float): Duration in seconds to collect samples at each
            point. Note: Currently unused; calibration proceeds when user clicks.
            Defaults to 2.0.
        output_file (str): Path to the output JSON file where calibration data
            will be saved. Defaults to "calibration_data.json".
    
    Raises:
        IOError: If the output file cannot be written.
        RuntimeError: If webcam access fails.
    
    Example:
        # Run with default settings
        run_calibration()
        
        # Custom output location
        run_calibration(output_file="my_calibration.json")
        
        # The output file can be loaded later:
        import json
        with open("calibration_data.json") as f:
            data = json.load(f)
    """
    data = []
    clicked = [False]
    click_pos = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)
            clicked[0] = True

    with WebcamClient() as cam:
        width, height = 640, 480  # fixed for calibration display
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Calibration", on_mouse)
        cv2.resizeWindow("Calibration", width, height)
        cv2.moveWindow("Calibration", 0, 0)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



        for (normalized_x, normalized_y) in CALIB_POINTS:
            print(f"Look at point ({normalized_x:.1f}, {normalized_y:.1f})")
            # start = time.time()
            samples = []
            clicked[0] = False

            while not clicked[0]:   # while time.time() - start < duration_per_point:
                frame = cam.snapshot()
                iris_center = get_iris_center(frame)
                frame = show_dot(frame, width, height, normalized_x, normalized_y)

                if iris_center:
                    samples.append(iris_center)
                    cv2.circle(frame, (int(iris_center[0]), int(iris_center[1])), IRIS_RADIUS, (0, 255, 255), -1)

                cv2.imshow("Calibration", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if samples:
                avg = np.mean(samples, axis=0).tolist()
                data.append({
                    "screen_x": normalized_x,
                    "screen_y": normalized_y,
                    "eye_x": avg[0],
                    "eye_y": avg[1]
                })
                print(f"Captured point: ({normalized_x}, {normalized_y}) -> eye {avg}")

        cv2.destroyAllWindows()

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Calibration is complete. Data saved to {output_file}")

if __name__ == "__main__":
    run_calibration()

