"""
Screen Recording Module with Real-time Feature Extraction

This module provides functionality for recording screen content to video files
while simultaneously extracting histogram features for each frame.
It uses pyautogui for screen capture and OpenCV for video encoding and feature extraction.

The module is designed to work alongside gaze tracking systems, where both
screen recording and gaze data collection can be started simultaneously
for later analysis and alignment.

Example:
    Basic usage with feature extraction:
        from pixels_to_players.screen_recording import ScreenRecorder, ScreenRecordingConfig

        # Simple recording with feature collection
        recorder = ScreenRecorder()
        video_path, features_data = recorder.record(duration_seconds=10.0)

        # Custom configuration with specific feature settings
        config = ScreenRecordingConfig(
            fps=30,
            width=1920,
            height=1080,
            show_preview=True,
            feature_type='hsv_histogram'  # Extract HSV histogram features
        )
        recorder = ScreenRecorder(config)
        video_path, features_data = recorder.record(duration_seconds=5.0)

Dependencies:
    - pyautogui: For screen capture functionality
    - opencv-python: For video encoding and processing
    - numpy: For array operations on captured frames
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import cv2
import numpy as np
import pyautogui
from cv2.typing import Size

from pixels_to_players.utils.os_version import *


@dataclass
class FeatureExtractionConfig:
    """
    Configuration for real-time feature extraction during recording.

    Attributes:
        feature_type (str): Type of feature to extract. Options:
            'grayscale_histogram', 'hsv_histogram', 'rgb_histogram', 'lab_histogram'
        hist_bins (int): Number of bins for histogram. Default: 32
        normalize_histogram (bool): Whether to normalize histograms. Default: True
        downsample_ratio (float): Ratio to downsample frames for feature extraction.
            Improves performance. Default: 0.25 (1/4 of original size)
        store_features (bool): Whether to store features in memory. Default: True
        auto_save_features (bool): Whether to automatically save features to disk after recording.
            Default: True
    """
    feature_type: str = 'hsv_histogram'
    hist_bins: int = 32
    normalize_histogram: bool = True
    downsample_ratio: float = 0.25
    store_features: bool = True
    auto_save_features: bool = True


@dataclass
class ScreenRecordingConfig:
    """
    Configuration for screen recording.

    Controls capture resolution, frame rate, preview window, codec, and output location.

    Attributes:
        res (Optional[Size]): Desired capture resolution (width, height). If None, uses the
            primary display size. Final resolution is capped by MAX_RES and forced to even
            dimensions for codec compatibility.
        MAX_RES (Size): Upper bound for the output resolution. Defaults to (1280, 720).
        fps (int): Target frames per second for capture and encoding. Default: 15.
        show_preview (bool): If True, shows a small live preview window during recording.
            Note: The preview processes UI events via cv2.waitKey to keep the window responsive.
        save_dir (Path): Directory where output videos are saved. Created if missing.
        fourcc (str): FourCC codec string (e.g., 'H264', 'mp4v', 'MJPG'). Default: 'H264'.
            The recorder attempts platform-specific backends (MSMF on Windows, AVFoundation on macOS)
            for H.264 when available, with an internal fallback if initialization fails.
        enable_key_stop (bool): If True, allows stopping with ESC or 'q' when preview is enabled.
            Default: False (use programmatic stop for uninterrupted sessions).
        feature_config (FeatureExtractionConfig): Configuration for real-time feature extraction.

    Examples:
        # Recording with feature extraction
        config = ScreenRecordingConfig(
            res=(1920, 1080),
            fps=30,
            show_preview=True,
            feature_config=FeatureExtractionConfig(
                feature_type='hsv_histogram',
                hist_bins=64,
                downsample_ratio=0.5
            )
        )
    """
    res: Optional[Size] = None
    MAX_RES: Size = (1280, 720)

    fps: int = 15  # target frame rate
    show_preview: bool = False  # show live preview window
    save_dir: Path = Path(__file__).parent / "recordings"
    fourcc: str = 'H264'  # video codec H.264
    enable_key_stop: bool = False  # allow stopping with ESC or 'q' keys
    feature_config: FeatureExtractionConfig = FeatureExtractionConfig()
    _target_size: Size = None, None


class ScreenRecorder:
    """
    Records the screen to a video file with real-time feature extraction.

    Features:
    - Auto-detects screen size (or uses config.res), caps to MAX_RES, and enforces even dimensions.
    - Attempts H.264 via platform-native backends (MSMF on Windows, AVFoundation on macOS),
      with a safe fallback to a broadly supported codec if initialization fails.
    - Time-based pacing to approximate the target FPS.
    - Optional live preview window with optional key-based stop.
    - Real-time histogram feature extraction for each frame.
    - Automatic saving of extracted features to JSON format.

    Attributes:
        cfg (ScreenRecordingConfig): Recording configuration.
        video_path (Optional[Path]): Output video path after start().
        _writer (Optional[cv2.VideoWriter]): Internal OpenCV writer.
        _running (bool): Recording loop state.
        _frame_index (int): Number of frames written in the current session.
        _window_name (str): Preview window title.
        _target_size (Size): Effective WxH used for encoding.
        _features_data (Dict): Collected feature data during recording.
        _features_save_path (Path): Path where features will be saved as JSON.

    Typical usage:
        # One-shot, 10 seconds with feature extraction
        recorder = ScreenRecorder()
        video_path, features_data = recorder.record(duration_seconds=10.0)

        # Manual control with custom config
        recorder = ScreenRecorder(ScreenRecordingConfig(
            fps=30,
            show_preview=True,
            fourcc="H264",
            feature_config=FeatureExtractionConfig(feature_type='rgb_histogram')
        ))
        video_path, features_data = recorder.start()
        recorder.record_for(duration_seconds=5.0)
        features_data = recorder.stop()
    """
    cfg: Optional[ScreenRecordingConfig]
    _writer: Optional[cv2.VideoWriter]
    _running: bool
    _frame_index: int
    video_path: Optional[Path]
    _window_name: str
    _target_size: Size
    _features_data: Dict[str, Any]
    _recording_start_time: float
    _features_save_path: Optional[Path]

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
        self._writer = None
        self._running = False
        self._frame_index = 0
        self.video_path = None
        self._window_name = 'Screen Recording Preview'
        self._target_size = 0, 0
        self._features_data = {}
        self._recording_start_time = 0.0
        self._features_save_path = None

    def _initialize_feature_storage(self):
        """Initialize data structures for feature storage."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._features_data = {
            'frame_features': [],
            'frame_timestamps': [],
            'frame_indices': [],
            'recording_config': {
                'fps': self.cfg.fps,
                'resolution': self._target_size,
                'feature_type': self.cfg.feature_config.feature_type,
                'hist_bins': self.cfg.feature_config.hist_bins,
                'downsample_ratio': self.cfg.feature_config.downsample_ratio,
                'normalize_histogram': self.cfg.feature_config.normalize_histogram
            },
            'metadata': {
                'total_frames': 0,
                'recording_duration': 0.0,
                'feature_extraction_time': 0.0,
                'video_filename': self.video_path.name if self.video_path else None,
                'recording_start_time': ts,
                'feature_dimension': 0
            }
        }

        # 设置特征保存路径（只保存JSON格式）
        if self.video_path:
            features_filename = f"features_{ts}.json"
            self._features_save_path = self.cfg.save_dir / features_filename

    def _extract_frame_features(self, frame: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        """
        Extract histogram features from a frame.

        Args:
            frame: Input frame in BGR format
            timestamp: Current timestamp in seconds

        Returns:
            Extracted feature vector or None if extraction failed
        """
        start_time = time.time()

        try:
            # Downsample frame for faster processing if configured
            if self.cfg.feature_config.downsample_ratio < 1.0:
                h, w = frame.shape[:2]
                new_w = int(w * self.cfg.feature_config.downsample_ratio)
                new_h = int(h * self.cfg.feature_config.downsample_ratio)
                frame_small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame

            feature_type = self.cfg.feature_config.feature_type
            bins = self.cfg.feature_config.hist_bins

            if feature_type == 'grayscale_histogram':
                # Convert to grayscale and compute histogram
                gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])

            elif feature_type == 'hsv_histogram':
                # Convert to HSV and compute 2D histogram (Hue, Saturation)
                hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])

            elif feature_type == 'rgb_histogram':
                # Compute RGB histogram
                hist_r = cv2.calcHist([frame_small], [0], None, [bins], [0, 256])
                hist_g = cv2.calcHist([frame_small], [1], None, [bins], [0, 256])
                hist_b = cv2.calcHist([frame_small], [2], None, [bins], [0, 256])
                hist = np.concatenate([hist_r, hist_g, hist_b])

            elif feature_type == 'lab_histogram':
                # Convert to LAB color space
                lab = cv2.cvtColor(frame_small, cv2.COLOR_BGR2LAB)
                hist = cv2.calcHist([lab], [0, 1], None, [bins, bins], [0, 256, 0, 256])

            else:
                # Default to grayscale
                gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])

            # Normalize histogram if configured
            if self.cfg.feature_config.normalize_histogram:
                hist = cv2.normalize(hist, hist).flatten()
            else:
                hist = hist.flatten()

            extraction_time = time.time() - start_time

            # Store feature data if configured
            if self.cfg.feature_config.store_features:
                self._features_data['frame_features'].append(hist)
                self._features_data['frame_timestamps'].append(timestamp)
                self._features_data['frame_indices'].append(self._frame_index)
                self._features_data['metadata']['feature_extraction_time'] += extraction_time

                # Update feature dimension in metadata
                if len(hist) > self._features_data['metadata']['feature_dimension']:
                    self._features_data['metadata']['feature_dimension'] = len(hist)

            return hist

        except Exception as e:
            print(f"Feature extraction failed for frame {self._frame_index}: {e}")
            return None

    def _add_feature_info_to_preview(self, frame: np.ndarray, feature: np.ndarray) -> np.ndarray:
        """
        Add feature extraction information to preview frame.

        Args:
            frame: Original frame
            feature: Extracted feature vector

        Returns:
            Frame with overlaid feature information
        """
        overlay = frame.copy()

        info_lines = [
            f"Frame: {self._frame_index}",
            f"Feature: {self.cfg.feature_config.feature_type}",
            f"Dim: {len(feature)}",
            f"Time: {time.time() - self._recording_start_time:.1f}s"
        ]

        for i, line in enumerate(info_lines):
            y_position = 30 + i * 25
            cv2.putText(overlay, line, (10, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return overlay

    def save_features(self) -> Optional[Path]:
        """
        Save extracted features to JSON file.

        Returns:
            Optional[Path]: Path to saved JSON features file, or None if failed
        """
        if not self._features_data or not self._features_data['frame_features']:
            print("No features data to save")
            return None

        if self._features_save_path is None:
            print("No features save path configured")
            return None

        try:
            # Convert numpy arrays to lists for JSON serialization
            json_data = self._features_data.copy()
            json_data['frame_features'] = [feat.tolist() for feat in json_data['frame_features']]
            json_data['frame_timestamps'] = json_data['frame_timestamps'].copy()
            json_data['frame_indices'] = json_data['frame_indices'].copy()

            with open(self._features_save_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            print(f"Features saved to JSON: {self._features_save_path}")
            return self._features_save_path

        except Exception as e:
            print(f"Failed to save features to JSON: {e}")
            return None

    def load_features(self, features_path: Path) -> Optional[Dict]:
        """
        Load previously saved features from JSON file.

        Args:
            features_path: Path to JSON features file

        Returns:
            Loaded features data or None if failed
        """
        try:
            if features_path.suffix != '.json':
                print("Only JSON format is supported")
                return None

            with open(features_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert lists back to numpy arrays for features
            data['frame_features'] = [np.array(feat) for feat in data['frame_features']]
            return data

        except Exception as e:
            print(f"Failed to load features from JSON: {e}")
            return None

    def start(self) -> Tuple[Path, Dict]:
        """
        Start the screen recording session with feature extraction.

        This method initializes the video writer, creates the output file,
        and prepares the recorder for capturing frames and extracting features.

        Returns:
            Tuple[Path, Dict]: Path to the output video file and empty features data structure

        Raises:
            RuntimeError: If the video writer cannot be initialized.
            OSError: If the output file cannot be created.
        """
        (w, h) = self.cfg.res or pyautogui.size()

        if w <= 0 or h <= 0:
            raise ValueError("Invalid screen resolution (Size)")

        w, h = (min(w, self.cfg.MAX_RES[0]),
                min(h, self.cfg.MAX_RES[1]))

        w -= w % 2
        h -= h % 2

        self._target_size = w, h

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = self.cfg.save_dir / f"screen_{ts}.mp4"

        # Initialize feature storage
        self._initialize_feature_storage()

        # Try different codec approaches
        try:
            fourcc = cv2.VideoWriter.fourcc(*self.cfg.fourcc)
            os_version: Optional[OSVersion] = OSVersion.get_os_version()

            if os_version is None:
                raise RuntimeError("Failed to get OS version")

            print('OS version:', os_version)

            if isinstance(os_version, WindowsVersion) and os_version.version_number >= 7:
                print("windows >= 7, using MSMF")
                self._writer = cv2.VideoWriter(str(self.video_path), cv2.CAP_MSMF, fourcc, self.cfg.fps,
                                               self._target_size)
            elif isinstance(os_version, MacOSVersion) and os_version.version_number >= (10, 13, 0):
                print("macOS >= 10.13, using AVFoundation")
                self._writer = cv2.VideoWriter(str(self.video_path), cv2.CAP_AVFOUNDATION, fourcc, self.cfg.fps,
                                               self._target_size)
            else:
                self._writer = cv2.VideoWriter(str(self.video_path), fourcc, self.cfg.fps, self._target_size)

        except Exception as e:
            print(e)
            print("Primary codec failed, trying fallback...")
            fourcc = cv2.VideoWriter.fourcc(*'MJPG')
            print(f"Using codec: {fourcc}")
            self._writer = cv2.VideoWriter(str(self.video_path), fourcc, self.cfg.fps, self._target_size)

        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.video_path}")

        print(f"Video writer initialized: {self.video_path}")
        print(f"Codec: {self.cfg.fourcc}, FPS: {self.cfg.fps}, Size: {self._target_size[0]}x{self._target_size[1]}")
        print(f"Feature extraction: {self.cfg.feature_config.feature_type}")
        print(f"Features will be saved as: {self._features_save_path}")

        # Test write a dummy frame to verify the writer works
        test_frame = np.zeros((self._target_size[1], self._target_size[0], 3), dtype=np.uint8)
        self._writer.write(test_frame)
        print("Test frame write attempted")

        self._running = True
        self._frame_index = 0
        self._recording_start_time = time.monotonic()

        return self.video_path, self._features_data

    def record_for(self, duration_seconds: Optional[float] = None) -> None:
        """
        Record screen content for a specified duration with real-time feature extraction.

        This method captures screen frames at the configured frame rate,
        writes them to the video file, and extracts histogram features for each frame.

        Args:
            duration_seconds (Optional[float]): Duration to record in seconds.
                If None, records indefinitely until stop() is called or key stop is triggered.
        """
        if not self._running:
            self.start()

        assert self._writer is not None, "writer is not initialized"
        assert self._target_size[0] > 0 and self._target_size[1] > 0, "Target size is invalid"

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

                # capture screenshot (RGB)
                screenshot = pyautogui.screenshot()
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

                # Enforce exact frame size for VideoWriter
                src_h, src_w = frame.shape[0], frame.shape[1]
                if (src_w, src_h) != self._target_size:
                    frame = cv2.resize(frame, self._target_size, interpolation=cv2.INTER_AREA)

                # write frame to video
                self._writer.write(frame)

                # Extract features from current frame
                current_time = now - self._recording_start_time
                current_feature = self._extract_frame_features(frame, current_time)

                # show preview with feature information
                if self.cfg.show_preview and current_feature is not None:
                    preview_frame = self._add_feature_info_to_preview(frame, current_feature)
                    preview_frame = cv2.resize(preview_frame, (640, 360))
                    cv2.imshow(self._window_name, preview_frame)
                    cv2.moveWindow(self._window_name, 100, 100)
                    if self.cfg.enable_key_stop:
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord('q')):
                            break
                    else:
                        cv2.waitKey(1)

                if self._frame_index % 30 == 0:  # Print every 30 frames
                    print(f"Frame {self._frame_index}: Feature extracted")

                self._frame_index += 1
                next_frame_time += frame_interval

        finally:
            self.stop()

    def stop(self) -> Dict:
        """
        Stop the screen recording session and return collected features.

        Returns:
            Dict: Collected feature data including frame features, timestamps, and metadata
        """
        if not self._running:
            return self._features_data

        self._running = False

        # Update metadata
        if self._features_data['frame_timestamps']:
            self._features_data['metadata']['recording_duration'] = \
                self._features_data['frame_timestamps'][-1]
            self._features_data['metadata']['total_frames'] = self._frame_index

        # Save features automatically if configured
        if (self.cfg.feature_config.auto_save_features and
            self.cfg.feature_config.store_features and
            self._features_data['frame_features']):
            saved_path = self.save_features()
            if saved_path:
                self._features_data['metadata']['features_save_path'] = str(saved_path)

        # Release video writer
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"Video writer released. Final file: {self.video_path}")

            # Video verification
            if self.video_path and self.video_path.exists():
                file_size = self.video_path.stat().st_size
                print(f"Final file size: {file_size} bytes")
                test_cap = cv2.VideoCapture(str(self.video_path))
                if test_cap.isOpened():
                    frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = test_cap.get(cv2.CAP_PROP_FPS)
                    print(f"Video verification: {frame_count} frames at {fps} FPS")
                    test_cap.release()
                else:
                    print("WARNING: Video file cannot be opened for verification")

        # Close preview window
        if self.cfg.show_preview:
            cv2.destroyWindow(self._window_name)

        # Print feature extraction summary
        if self._features_data['frame_features']:
            total_features = len(self._features_data['frame_features'])
            avg_extraction_time = self._features_data['metadata']['feature_extraction_time'] / total_features
            print(f"Feature extraction summary:")
            print(f"  Total frames processed: {self._frame_index}")
            print(f"  Features extracted: {total_features}")
            print(f"  Average extraction time: {avg_extraction_time*1000:.2f}ms per frame")
            print(f"  Feature dimension: {self._features_data['metadata']['feature_dimension']}")
            if 'features_save_path' in self._features_data['metadata']:
                print(f"  Features saved to: {self._features_data['metadata']['features_save_path']}")

        return self._features_data

    def record(self, duration_seconds: Optional[float] = None) -> Tuple[Path, Dict]:
        """
        Convenience method for one-shot recording with feature extraction.

        Args:
            duration_seconds (Optional[float]): Duration to record in seconds.

        Returns:
            Tuple[Path, Dict]: Path to the recorded video file and collected feature data
        """
        video_path, features_data = self.start()
        self.record_for(duration_seconds)
        final_features_data = self.stop()
        return video_path, final_features_data

    def get_features_data(self) -> Dict:
        """
        Get the currently collected feature data without stopping recording.

        Returns:
            Dict: Current feature data (useful for long recordings)
        """
        return self._features_data.copy()

    def get_features_array(self) -> Optional[np.ndarray]:
        """
        Get features as a numpy array for machine learning.

        Returns:
            Optional[np.ndarray]: Features array of shape (n_frames, feature_dim) or None
        """
        if not self._features_data or not self._features_data['frame_features']:
            return None
        return np.array(self._features_data['frame_features'])


# Example usage and testing
if __name__ == "__main__":
    def example_usage():
        """Demonstrate the screen recorder with feature extraction."""

        # Example 1: Basic recording with default features
        print("=== Example 1: Basic recording ===")
        recorder = ScreenRecorder()
        video_path, features_data = recorder.record(duration_seconds=5.0)
        print(f"Video saved: {video_path}")
        print(f"Features collected: {len(features_data['frame_features'])}")

        # Example 2: Custom configuration with HSV features
        print("\n=== Example 2: Custom configuration ===")
        config = ScreenRecordingConfig(
            fps=15,
            show_preview=True,
            feature_config=FeatureExtractionConfig(
                feature_type='hsv_histogram',
                hist_bins=64,
                downsample_ratio=0.3
            )
        )
        recorder = ScreenRecorder(config)
        video_path, features_data = recorder.record(duration_seconds=120.0)

        # Example 3: Load saved features
        print("\n=== Example 3: Load saved features ===")
        if 'features_save_path' in features_data['metadata']:
            features_path = Path(features_data['metadata']['features_save_path'])
            loaded_features = recorder.load_features(features_path)
            if loaded_features:
                print(f"Loaded {len(loaded_features['frame_features'])} features from JSON")
                print(f"First feature sample: {loaded_features['frame_features'][0][:5]}...")

    # Run the example
    example_usage()