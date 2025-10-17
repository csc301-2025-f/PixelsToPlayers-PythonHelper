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
        fourcc (str): Video codec to use for encoding. Default is 'avc1' for MP4 (H.264) format.
        enable_key_stop (bool): Whether to allow stopping recording with ESC or 'q' keys.
            Disabled by default for gaming scenarios. Defaults to False.
    
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
        
        # Gaming-friendly recording (no key stops)
        config = ScreenRecordingConfig(
            show_preview=True,
            enable_key_stop=False  # Won't stop on ESC or 'q'
        )
    """
    width: Optional[int] = None  # None → use full screen width
    height: Optional[int] = None # None → use full screen height
    fps: int = 15                # target frame rate
    show_preview: bool = False   # show live preview window
    save_dir: Path = Path(__file__).parent / "recordings"
    fourcc: str = "avc1"         # video codec
    enable_key_stop: bool = False # allow stopping with ESC or 'q' keys


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
        self._window_name = "Screen Recording Preview"

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
        
        # Use a smaller, more standard resolution for better compatibility
        if w > 1280:
            w = 1280
        if h > 720:
            h = 720
            
        # Ensure dimensions are even numbers (required by some codecs)
        w = w - (w % 2)
        h = h - (h % 2)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = self.cfg.save_dir / f"screen_{ts}.mp4"

        # Try different codec approaches
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.cfg.fourcc)
            self._writer = cv2.VideoWriter(str(self.video_path), fourcc, self.cfg.fps, (w, h), True)
        except:
            # Fallback to a more basic approach
            print("Primary codec failed, trying fallback...")
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self._writer = cv2.VideoWriter(str(self.video_path), fourcc, self.cfg.fps, (w, h), True)
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.video_path}")
        
        print(f"Video writer initialized: {self.video_path}")
        print(f"Codec: {self.cfg.fourcc}, FPS: {self.cfg.fps}, Size: {w}x{h}")

        # Store target size to enforce consistent frame dimensions
        self._target_size = (w, h)
        
        # Test write a dummy frame to verify the writer works
        test_frame = np.zeros((h, w, 3), dtype=np.uint8)
        self._writer.write(test_frame)
        print("Test frame write attempted")

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
                If None, records indefinitely until stop() is called or key stop is triggered
                (only if enable_key_stop is True in config).
                Defaults to None.
        
        Raises:
            RuntimeError: If the video writer is not properly initialized.
            KeyboardInterrupt: If recording is interrupted by user (ESC or 'q' key, only if enable_key_stop is True).
        
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
        target_w, target_h = getattr(self, "_target_size", (None, None))

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
                
                # Ensure frame is the right size and format
                # Note: We'll resize later if needed based on config

                # Enforce exact frame size for VideoWriter
                if target_w is not None and target_h is not None:
                    src_h, src_w = frame.shape[0], frame.shape[1]
                    if (src_w, src_h) != (target_w, target_h):
                        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

                # write frame to video
                self._writer.write(frame)
                if self._frame_index % 10 == 0:  # Print every 10th frame
                    print(f"Wrote frame {self._frame_index}")

                # show preview after writing to avoid capturing the preview window
                if self.cfg.show_preview:
                    # Create a smaller preview to avoid interfering with screen capture
                    preview_frame = cv2.resize(frame, (640, 360))
                    cv2.imshow(self._window_name, preview_frame)
                    # Position the window in top-right corner to minimize interference
                    cv2.moveWindow(self._window_name, 100, 100)
                    if self.cfg.enable_key_stop:
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord('q')):
                            break
                    else:
                        cv2.waitKey(1)  # Still need to process window events

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
            # Ensure all frames are written
            self._writer.release()
            self._writer = None
            print(f"Video writer released. Final file: {self.video_path}")
            if self.video_path and self.video_path.exists():
                file_size = self.video_path.stat().st_size
                print(f"Final file size: {file_size} bytes")
                
                # Try to verify the video can be opened
                test_cap = cv2.VideoCapture(str(self.video_path))
                if test_cap.isOpened():
                    frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = test_cap.get(cv2.CAP_PROP_FPS)
                    print(f"Video verification: {frame_count} frames at {fps} FPS")
                    test_cap.release()
                else:
                    print("WARNING: Video file cannot be opened for verification")

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
