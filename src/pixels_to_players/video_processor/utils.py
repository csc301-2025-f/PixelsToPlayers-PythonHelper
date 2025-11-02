import cv2
import os
from pathlib import Path

def read_video(video_path):
    """
    Open a video file with OpenCV and return the capture object, FPS, and frame count.
    Automatically resolves relative paths based on the current working directory.
    """
    # Resolve to absolute path
    video_path = Path(video_path).expanduser().resolve()

    # Try opening
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Can't open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, frame_count


def save_image(image, path):
    """
    Save an image to a given path, automatically creating directories as needed.
    Works with both relative and absolute paths.
    """
    path = Path(path).expanduser().resolve()
    os.makedirs(path.parent, exist_ok=True)
    cv2.imwrite(str(path), image)
