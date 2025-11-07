"""
Webcam Client Module

This module provides an abstraction layer for managing webcam input streams.
It handles camera lifecycle (open/close), frame capture, and recording sessions,
optionally applying custom frame processors such as filters, overlays, or
landmark detectors.

It is designed to integrate seamlessly with other gaze tracking and face
analysis utilities (e.g., `processors.py`), where frames can be processed in
real time or logged for later analysis.

Example:
    Basic snapshot:
        from pixels_to_players.webcam.client import WebcamClient
        frame = WebcamClient().snapshot()

    Record 5 seconds with face mesh overlay:
        from pixels_to_players.webcam.client import WebcamClient
        from pixels_to_players.webcam.processors import draw_facemesh
        client = WebcamClient()
        video_path = client.record(duration=5.0, processors=[draw_facemesh])
        print(f"Saved to {video_path}")

Dependencies:
    - opencv-python: For webcam capture, display, and video encoding
    - numpy: For array and frame operations
    - mediapipe (optional): For landmark processing when used with processors
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional, List, Tuple
import time
import cv2
import numpy as np


FrameProcessor = Callable[[np.ndarray], np.ndarray]


@dataclass
class WebcamConfig:
    """
    Configuration for webcam capture.

    Controls which camera device is used, capture dimensions, frame rate,
    preview settings, and output codec/format for recordings.

    Attributes:
        device_index (int): Index of the camera to open (default: 0).
        width (int): Desired frame width in pixels.
        height (int): Desired frame height in pixels.
        fps (int): Target frames per second for capture.
        show_preview (bool): Whether to show a live OpenCV preview window.
        save_dir (Path): Directory for saving recordings.
        fourcc (str): Video codec FourCC (e.g., 'mp4v' for .mp4).
        backend (int): OpenCV capture backend (e.g., CAP_MSMF, CAP_DSHOW).
    """
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    show_preview: bool = True
    save_dir: Path = Path(__file__).parent / "recordings"
    fourcc: str = "mp4v"
    backend: int = cv2.CAP_ANY


class WebcamClient:
    """
    Provides high-level webcam interaction and lifecycle management.

    Handles camera opening, frame retrieval, and recording with optional
    frame processors for real-time effects such as face mesh overlays,
    mirroring, grayscale conversion, or annotation.

    Example:
        client = WebcamClient()
        frame = client.snapshot()
        client.record(duration=5.0)
    """

    def __init__(self, cfg: WebcamConfig | None = None) -> None:
        """
        Initialize the webcam client with an optional configuration.

        Args:
            cfg (WebcamConfig, optional): Camera configuration. Defaults to
                standard webcam parameters if None.
        """
        self.cfg = cfg or WebcamConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        self.cfg.save_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "WebcamClient":
        """Open the webcam for use in a context manager (`with` block`)."""
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the webcam and release resources when exiting a context."""
        self.close()

    # ---- lifecycle ----
    def open(self) -> None:
        """
        Open the webcam and configure capture properties.

        Raises:
            RuntimeError: If the specified camera cannot be opened.
        """
        self.cap = cv2.VideoCapture(self.cfg.device_index, self.cfg.backend)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.cfg.device_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

    def close(self) -> None:
        """
        Close the webcam and clean up OpenCV resources.

        Always safe to call even if the webcam is already closed.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    # ---- helpers ----
    def _actual_props(self) -> Tuple[int, int, int]:
        """
        Retrieve the actual width, height, and FPS of the active camera stream.

        Returns:
            Tuple[int, int, int]: The actual (width, height, fps) values.
        """
        assert self.cap and self.cap.isOpened(), "Camera not opened"
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.cfg.width
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.cfg.height
        fps_val = self.cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps_val) if fps_val and fps_val > 0 else self.cfg.fps
        return w, h, fps

    # ---- public API ----
    def snapshot(
        self,
        processors: Optional[Iterable[FrameProcessor]] = None,
    ) -> np.ndarray:
        """
        Capture a single frame from the webcam, optionally processing it.

        Args:
            processors (Iterable[FrameProcessor], optional): A sequence of
                frame-processing functions applied sequentially.

        Returns:
            np.ndarray: The captured (and optionally processed) image frame.

        Raises:
            RuntimeError: If frame capture fails.
        """
        if not self.cap or not self.cap.isOpened():
            self.open()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera")
        for p in processors or []:
            frame = p(frame)
        return frame

    def record(
        self,
        duration: Optional[float] = 10.0,
        processors: Optional[Iterable[FrameProcessor]] = None,
    ) -> Path:
        """
        Record webcam video for a specified duration.

        Each frame is optionally passed through the provided processors and
        written to a video file in the configured save directory.

        Args:
            duration (float | None): Duration in seconds. If None, recording
                continues indefinitely until ESC or 'q' is pressed.
            processors (Iterable[FrameProcessor], optional): Frame-processing
                pipeline applied to each captured frame.

        Returns:
            Path: Path to the saved output video file.

        Raises:
            RuntimeError: If the video writer cannot be initialized.
        """
        if not self.cap or not self.cap.isOpened():
            self.open()

        w, h, fps = self._actual_props()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.cfg.save_dir / f"recording_{ts}.mp4"

        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter.fourcc(*self.cfg.fourcc),
            fps,
            (w, h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {out_path}")

        start = time.time()
        try:
            while duration is None or (time.time() - start) < duration:
                ok, frame = self.cap.read()
                if not ok:
                    break
                for p in processors or []:
                    frame = p(frame)
                if self.cfg.show_preview:
                    cv2.imshow("webcam", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                writer.write(frame)
        finally:
            writer.release()
            if self.cfg.show_preview:
                cv2.destroyWindow("webcam")

        return out_path
