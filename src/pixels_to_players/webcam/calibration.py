#src/pixels_to_players/webcam/calibration.py
import cv2
import time
import json
import numpy as np
from pixels_to_players.webcam.client import WebcamClient
from pixels_to_players.webcam.processors import get_iris_center

# Normalized calibration points (x, y)
CALIB_POINTS = [
    (0.1, 0.1),
    (0.9, 0.1),
    (0.1, 0.9),
    (0.9, 0.9),
    (0.5, 0.5)
]

def show_dot(frame, screen_w, screen_h, px, py):
    """Draw a red calibration dot on the frame."""
    x = int(px * screen_w)
    y = int(py * screen_h)
    cv2.circle(frame, (x, y), 20, (0, 0, 255), -1)
    return frame

def run_calibration(duration_per_point=2.0, output_file="calibration_data.json"):
    """Show calibration points, record iris centers, and save averaged data."""
    data = []

    with WebcamClient() as cam:
        w, h, _ = cam._actual_props()

        for (px, py) in CALIB_POINTS:
            print(f"Look at point ({px:.1f}, {py:.1f})")
            start = time.time()
            samples = []

            while time.time() - start < duration_per_point:
                frame = cam.snapshot()
                iris_center = get_iris_center(frame)
                frame = show_dot(frame, w, h, px, py)

                if iris_center:
                    samples.append(iris_center)
                    cv2.circle(frame, (int(iris_center[0]), int(iris_center[1])), 5, (0, 255, 255), -1)

                cv2.imshow("Calibration", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if samples:
                avg = np.mean(samples, axis=0).tolist()
                data.append({
                    "screen_x": px,
                    "screen_y": py,
                    "eye_x": avg[0],
                    "eye_y": avg[1]
                })
                print(f"Captured point: ({px}, {py}) -> eye {avg}")

        cv2.destroyAllWindows()

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Calibration is complete. Data saved to {output_file}")

if __name__ == "__main__":
    run_calibration()

