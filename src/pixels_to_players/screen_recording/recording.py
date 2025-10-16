"""
Screen Recording Module

This module provides functionality for recording screen content to video files.
It uses pyautogui for screen capture and OpenCV for video encoding.

The module is designed to work alongside gaze tracking systems, where both
screen recording and gaze data collection can be started simultaneously
for later analysis and alignment.

Example:
    Basic usage:
        from pixels_to_players.screen_recording import ScreenRecorder, ScreenRecordingConfig
        
        # Simple recording
        recorder = ScreenRecorder()
        video_path = recorder.record(duration_seconds=10.0)
        
        # Custom configuration
        config = ScreenRecordingConfig(fps=30, width=1920, height=1080, show_preview=True)
        recorder = ScreenRecorder(config)
        video_path = recorder.record(duration_seconds=5.0)

Dependencies:
    - pyautogui: For screen capture functionality
    - opencv-python: For video encoding and processing
    - numpy: For array operations on captured frames

Author: PixelsToPlayers Team
Version: 1.0.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pyautogui
import cv2
import numpy as np


@dataclass
class ScreenRecordingConfig:
    """
    Configuration class for screen recording parameters.
    
    This dataclass holds all the configuration options needed to customize
    the screen recording behavior, including resolution, frame rate, and
    output settings.
    
    Attributes:
        width (Optional[int]): Target video width in pixels. If None, uses full screen width.
        height (Optional[int]): Target video height in pixels. If None, uses full screen height.
        fps (int): Target frame rate for the recording. Default is 15 FPS.
        show_preview (bool): Whether to show a live preview window during recording.
        save_dir (Path): Directory where recorded videos will be saved.
        fourcc (str): Video codec to use for encoding. Default is 'mp4v' for MP4 format.
    
    Example:
        # High quality recording with preview
        config = ScreenRecordingConfig(
            width=1920,
            height=1080,
            fps=30,
            show_preview=True,
            fourcc="mp4v"
        )
        
        # Low quality recording without preview
        config = ScreenRecordingConfig(
            width=640,
            height=480,
            fps=10,
            show_preview=False
        )
    """
    width: Optional[int] = None  # None → use full screen width
    height: Optional[int] = None # None → use full screen height
    fps: int = 15                # target frame rate
    show_preview: bool = False   # show live preview window
    save_dir: Path = Path(__file__).parent / "recordings"
    fourcc: str = "mp4v"         # video codec


class ScreenRecorder:
    """
    Main class for recording screen content to video files.
    
    The ScreenRecorder class provides a simple interface for capturing screen
    content and saving it as a video file. It supports configurable resolution,
    frame rate, and output format.
    
    The recorder uses pyautogui for screen capture and OpenCV for video encoding,
    providing a cross-platform solution for screen recording functionality.
    
    Attributes:
        cfg (ScreenRecordingConfig): Configuration object containing recording parameters.
        _writer (Optional[cv2.VideoWriter]): OpenCV video writer instance.
        _running (bool): Flag indicating if recording is currently active.
        _frame_index (int): Current frame number being recorded.
        video_path (Optional[Path]): Path to the output video file.
        _window_name (str): Name of the preview window (if enabled).
    
    Example:
        # Basic usage
        recorder = ScreenRecorder()
        video_path = recorder.record(duration_seconds=10.0)
        
        # Advanced usage with manual control
        recorder = ScreenRecorder(ScreenRecordingConfig(fps=30, show_preview=True))
        video_path = recorder.start()
        recorder.record_for(duration_seconds=5.0)
        recorder.stop()
    """

    def __init__(self, cfg: ScreenRecordingConfig | None = None) -> None:
        """
        Initialize the ScreenRecorder with optional configuration.
        
        Args:
            cfg (ScreenRecordingConfig, optional): Configuration object. If None,
                uses default configuration. Defaults to None.
        
        Raises:
            OSError: If the save directory cannot be created.
        """
        self.cfg = cfg or ScreenRecordingConfig()
        self.cfg.save_dir.mkdir(parents=True, exist_ok=True)
        self._writer: Optional[cv2.VideoWriter] = None
        self._running: bool = False
        self._frame_index: int = 0
        self.video_path: Optional[Path] = None
        self._window_name = "screen"

    def start(self) -> Path:
        """
        Start the screen recording session.
        
        This method initializes the video writer, creates the output file,
        and prepares the recorder for capturing frames. The output filename
        is automatically generated with a timestamp.
        
        Returns:
            Path: Path to the output video file that will be created.
        
        Raises:
            RuntimeError: If the video writer cannot be initialized.
            OSError: If the output file cannot be created.
        
        Example:
            recorder = ScreenRecorder()
            video_path = recorder.start()
            print(f"Recording to: {video_path}")
        """
        screen_w, screen_h = pyautogui.size()
        w = self.cfg.width or screen_w
        h = self.cfg.height or screen_h

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = self.cfg.save_dir / f"screen_{ts}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*self.cfg.fourcc)
        self._writer = cv2.VideoWriter(str(self.video_path), fourcc, self.cfg.fps, (w, h))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.video_path}")

        self._running = True
        self._frame_index = 0

        return self.video_path

    def record_for(self, duration_seconds: Optional[float] = None) -> None:
        """
        Record screen content for a specified duration.
        
        This method captures screen frames at the configured frame rate and
        writes them to the video file. If the recording hasn't been started
        yet, it will automatically call start() first.
        
        The method uses time-based pacing to maintain the target frame rate
        and includes optional preview window functionality.
        
        Args:
            duration_seconds (Optional[float]): Duration to record in seconds.
                If None, records indefinitely until stop() is called or 'q' is pressed.
                Defaults to None.
        
        Raises:
            RuntimeError: If the video writer is not properly initialized.
            KeyboardInterrupt: If recording is interrupted by user (ESC or 'q' key).
        
        Example:
            # Record for 10 seconds
            recorder = ScreenRecorder()
            recorder.record_for(duration_seconds=10.0)
            
            # Record indefinitely (until manually stopped)
            recorder.record_for()
        """
        if not self._running:
            self.start()

        assert self._writer is not None

        frame_interval = 1.0 / max(1, self.cfg.fps)
        next_frame_time = time.monotonic()
        end_time = None if duration_seconds is None else (time.monotonic() + duration_seconds)

        try:
            while self._running:
                now = time.monotonic()
                if end_time is not None and now >= end_time:
                    break

                # pace capture near target FPS
                if now < next_frame_time:
                    time.sleep(max(0.0, next_frame_time - now))
                    now = time.monotonic()

                # capture screenshot (RGB)
                screenshot = pyautogui.screenshot()
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

                # resize if requested
                if self.cfg.width or self.cfg.height:
                    target_w = self.cfg.width or frame.shape[1]
                    target_h = self.cfg.height or frame.shape[0]
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

                # write frame
                self._writer.write(frame)

                if self.cfg.show_preview:
                    cv2.imshow(self._window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        break

                self._frame_index += 1
                next_frame_time += frame_interval
        finally:
            self.stop()

    def stop(self) -> None:
        """
        Stop the screen recording session.
        
        This method safely stops the recording, releases the video writer,
        and cleans up any resources. It's safe to call multiple times.
        
        The method will:
        - Stop the recording loop
        - Release the OpenCV video writer
        - Close the preview window (if enabled)
        - Reset internal state
        
        Example:
            recorder = ScreenRecorder()
            recorder.start()
            recorder.record_for(duration_seconds=5.0)
            recorder.stop()  # Clean shutdown
        """
        if not self._running:
            return
        self._running = False

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        if self.cfg.show_preview:
            cv2.destroyWindow(self._window_name)

    def record(self, duration_seconds: Optional[float] = None) -> Path:
        """
        Convenience method for one-shot recording.
        
        This method combines start(), record_for(), and stop() into a single
        call for simple recording scenarios. It's the easiest way to record
        screen content without manual session management.
        
        Args:
            duration_seconds (Optional[float]): Duration to record in seconds.
                If None, records indefinitely until manually stopped.
                Defaults to None.
        
        Returns:
            Path: Path to the recorded video file.
        
        Raises:
            RuntimeError: If video writer initialization fails.
            KeyboardInterrupt: If recording is interrupted by user.
        
        Example:
            # Simple 10-second recording
            recorder = ScreenRecorder()
            video_path = recorder.record(duration_seconds=10.0)
            print(f"Video saved to: {video_path}")
            
            # High-quality recording with custom config
            config = ScreenRecordingConfig(fps=30, width=1920, height=1080)
            recorder = ScreenRecorder(config)
            video_path = recorder.record(duration_seconds=5.0)
        """
        video_path = self.start()
        self.record_for(duration_seconds)
        return video_path