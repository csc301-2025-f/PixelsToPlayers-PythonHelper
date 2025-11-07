import cv2
import os
import json
import numpy as np
from pathlib import Path


def read_json_features(json_path):
    """
    Load pre-computed features from JSON file.

    Args:
        json_path (str): Path to JSON features file

    Returns:
        tuple: (embeddings, timestamps, frame_indices, metadata, recording_config)

    Raises:
        FileNotFoundError: If JSON file does not exist
        ValueError: If JSON file has invalid format
    """
    json_path = Path(json_path).expanduser().resolve()

    if not json_path.exists():
        raise FileNotFoundError(f"JSON features file not found: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate required fields
        required_fields = ['frame_features', 'frame_timestamps',
                           'frame_indices']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field in JSON: {field}")

        # Convert to numpy arrays
        embeddings = np.array(data['frame_features'])
        timestamps = np.array(data['frame_timestamps'])
        frame_indices = np.array(data['frame_indices'])

        # Get optional fields with defaults
        recording_config = data.get('recording_config', {})
        metadata = data.get('metadata', {})

        # Validate data consistency
        if len(embeddings) != len(timestamps) or len(embeddings) != len(
                frame_indices):
            raise ValueError("Inconsistent data lengths in JSON features file")

        return embeddings, timestamps, frame_indices, metadata, recording_config

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_path}: {e}")


def read_video(video_path):
    """
    Open a video file with OpenCV and return the capture object, FPS, and frame count.
    Automatically resolves relative paths based on the current working directory.

    Args:
        video_path (str): Path to the video file

    Returns:
        tuple: (cap, fps, frame_count) where:
            - cap: OpenCV VideoCapture object
            - fps: Frames per second
            - frame_count: Total number of frames in video

    Raises:
        FileNotFoundError: If video file cannot be opened
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

    Args:
        image (numpy.ndarray): Image array to save
        path (str): Path where to save the image
    """
    path = Path(path).expanduser().resolve()
    os.makedirs(path.parent, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise ValueError(f"Failed to save image to: {path}")
