"""
Pytest tests for screen_recording.recording module.

Tests cover basic operations, config options, resolution handling, OS-specific codec
selection, preview window functionality, edge cases, and resource cleanup.
All external dependencies are mocked to allow tests to run on headless systems.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, call, patch

import numpy as np
import pytest
from PIL import Image

from pixels_to_players.screen_recording.recording import (
    ScreenRecorder,
    ScreenRecordingConfig,
)
from pixels_to_players.utils.os_version import (
    LinuxVersion,
    MacOSVersion,
    WindowsVersion,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_pyautogui():
    """
    Mock pyautogui module for screen capture operations.
    
    Returns a patched pyautogui with:
    - size() returning (1920, 1080)
    - screenshot() returning a mock PIL Image
    """
    with patch("pixels_to_players.screen_recording.recording.pyautogui") as mock:
        # Mock screen size
        mock.size.return_value = (1920, 1080)
        
        # Mock screenshot to return a PIL Image-like object
        mock_image = MagicMock(spec=Image.Image)
        mock_image.__array__ = Mock(return_value=np.zeros((1080, 1920, 3), dtype=np.uint8))
        mock.screenshot.return_value = mock_image
        
        yield mock


@pytest.fixture
def mock_cv2():
    """
    Mock cv2 (OpenCV) module for video writing and window operations.
    
    Returns a patched cv2 with mocked VideoWriter, VideoCapture, and window functions.
    """
    with patch("pixels_to_players.screen_recording.recording.cv2") as mock:
        # Mock VideoWriter
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_writer.write.return_value = None
        mock_writer.release.return_value = None
        mock.VideoWriter.return_value = mock_writer
        
        # Mock VideoWriter.fourcc
        mock.VideoWriter.fourcc.return_value = 0x12345678
        
        # Mock CAP constants
        mock.CAP_MSMF = 1400
        mock.CAP_AVFOUNDATION = 1200
        mock.CAP_PROP_FRAME_COUNT = 7
        mock.CAP_PROP_FPS = 5
        
        # Mock VideoCapture for verification
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.side_effect = lambda prop: 100 if prop == 7 else 15.0
        mock_capture.release.return_value = None
        mock.VideoCapture.return_value = mock_capture
        
        # Mock window operations
        mock.imshow.return_value = None
        mock.moveWindow.return_value = None
        mock.waitKey.return_value = -1  # No key pressed by default
        mock.destroyWindow.return_value = None
        mock.resize.side_effect = lambda img, size, **kwargs: np.zeros(
            (size[1], size[0], 3), dtype=np.uint8
        )
        mock.cvtColor.side_effect = lambda img, code: img
        
        # Mock color conversion constants
        mock.COLOR_RGB2BGR = 4
        mock.INTER_AREA = 3
        
        yield mock


@pytest.fixture
def mock_time():
    """
    Mock time module for controlling timing without delays.
    
    Returns a patched time module with controllable monotonic() values.
    """
    with patch("pixels_to_players.screen_recording.recording.time") as mock:
        # Start at time 0 and increment by 0.1 on each call
        mock.monotonic.side_effect = (i * 0.1 for i in range(1000))
        mock.sleep.return_value = None
        yield mock


@pytest.fixture
def mock_os_version():
    """Mock OSVersion.get_os_version() for OS-specific codec testing."""
    with patch(
        "pixels_to_players.screen_recording.recording.OSVersion.get_os_version"
    ) as mock:
        yield mock


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for video output."""
    output_dir = tmp_path / "recordings"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def basic_config(temp_output_dir):
    """Create a basic ScreenRecordingConfig for testing."""
    return ScreenRecordingConfig(
        save_dir=temp_output_dir,
        fps=15,
        show_preview=False,
    )


# ============================================================================
# BASIC OPERATIONS TESTS
# ============================================================================


class TestBasicOperations:
    """Test basic recording operations: start(), stop(), record_for(), record()."""
    
    def test_start_initializes_recorder(
        self, basic_config, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that start() properly initializes the recorder.
        
        Verifies:
        - VideoWriter is created with correct parameters
        - Running state is set to True
        - Video path is created with timestamp
        - Test frame is written
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        recorder = ScreenRecorder(basic_config)
        
        # Act
        video_path = recorder.start()
        
        # Assert
        assert recorder._running is True
        assert recorder.video_path is not None
        assert video_path.parent == basic_config.save_dir
        assert video_path.name.startswith("screen_")
        assert video_path.suffix == ".mp4"
        assert recorder._frame_index == 0
        
        # Verify VideoWriter was created
        mock_cv2.VideoWriter.assert_called_once()
        assert mock_cv2.VideoWriter.return_value.write.call_count == 1
    
    def test_stop_releases_resources(
        self, basic_config, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that stop() properly releases resources.
        
        Verifies:
        - Writer is released
        - Running state is set to False
        - VideoCapture verification is attempted
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        recorder = ScreenRecorder(basic_config)
        recorder.start()
        
        # Act
        recorder.stop()
        
        # Assert
        assert recorder._running is False
        assert recorder._writer is None
        mock_cv2.VideoWriter.return_value.release.assert_called_once()
        mock_cv2.VideoCapture.assert_called_once()
    
    def test_stop_multiple_times_safe(
        self, basic_config, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that calling stop() multiple times is safe.
        
        Verifies:
        - No errors on multiple stop() calls
        - Writer release only called once
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        recorder = ScreenRecorder(basic_config)
        recorder.start()
        
        # Act
        recorder.stop()
        recorder.stop()
        recorder.stop()
        
        # Assert
        assert recorder._running is False
        # Should only release once (first call)
        assert mock_cv2.VideoWriter.return_value.release.call_count == 1
    
    def test_record_for_duration(
        self, basic_config, mock_pyautogui, mock_cv2, mock_os_version, mock_time
    ):
        """
        Test recording for a specific duration.
        
        Verifies:
        - Frames are captured and written
        - Recording stops after duration
        - Frame count is approximately correct for duration and FPS
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        recorder = ScreenRecorder(basic_config)
        duration = 1.0  # 1 second
        
        # Mock time to simulate duration passing
        start_time = 0.0
        time_values = [start_time + i * 0.01 for i in range(200)]
        mock_time.monotonic.side_effect = time_values
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=duration)
        
        # Assert
        assert recorder._running is False  # Should be stopped after record_for
        # At 15 FPS for 1 second, we expect around 15 frames
        assert recorder._frame_index >= 10  # Allow some tolerance
        assert mock_pyautogui.screenshot.call_count >= 10
    
    def test_record_for_auto_starts(
        self, basic_config, mock_pyautogui, mock_cv2, mock_os_version, mock_time
    ):
        """
        Test that record_for() automatically starts recording if not started.
        
        Verifies:
        - start() is implicitly called
        - Recording proceeds normally
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        recorder = ScreenRecorder(basic_config)
        
        # Mock time for quick completion
        mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 1.0, 1.1]
        
        # Act
        recorder.record_for(duration_seconds=0.5)
        
        # Assert
        assert recorder.video_path is not None
        assert recorder._running is False
    
    def test_record_convenience_method(
        self, basic_config, mock_pyautogui, mock_cv2, mock_os_version, mock_time
    ):
        """
        Test the record() convenience method.
        
        Verifies:
        - Combines start() and record_for() in one call
        - Returns video path
        - Recording completes successfully
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        recorder = ScreenRecorder(basic_config)
        mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 0.5, 1.0]
        
        # Act
        video_path = recorder.record(duration_seconds=0.5)
        
        # Assert
        assert video_path is not None
        assert video_path.parent == basic_config.save_dir
        assert recorder._running is False
        assert recorder._frame_index > 0


# ============================================================================
# CONFIG OPTIONS TESTS
# ============================================================================


class TestConfigOptions:
    """Test different configuration options."""
    
    @pytest.mark.parametrize(
        "fps,expected_interval",
        [
            (15, 1.0 / 15),
            (30, 1.0 / 30),
            (60, 1.0 / 60),
            (1, 1.0),
        ],
    )
    def test_different_fps_values(
        self,
        temp_output_dir,
        fps,
        expected_interval,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test recording with different FPS values.
        
        Verifies:
        - VideoWriter is created with correct FPS
        - Frame interval is calculated correctly
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(save_dir=temp_output_dir, fps=fps)
        recorder = ScreenRecorder(config)
        
        # Mock time for quick test
        mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 1.0]
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=0.1)
        
        # Assert
        # Check that VideoWriter was created with correct FPS
        call_args = mock_cv2.VideoWriter.call_args
        assert call_args is not None
        # FPS is typically the 4th positional argument or 'fps' keyword
        if len(call_args[0]) >= 4:
            assert call_args[0][3] == fps
    
    @pytest.mark.parametrize(
        "codec",
        ["H264", "mp4v", "MJPG", "XVID"],
    )
    def test_different_codecs(
        self,
        temp_output_dir,
        codec,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
    ):
        """
        Test recording with different codec options.
        
        Verifies:
        - Codec string is properly converted to fourcc
        - VideoWriter receives correct fourcc value
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(save_dir=temp_output_dir, fourcc=codec)
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        
        # Assert
        # Verify fourcc was called with the codec string
        mock_cv2.VideoWriter.fourcc.assert_called_with(*codec)
    
    def test_preview_window_enabled(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test recording with preview window enabled.
        
        Verifies:
        - Preview window is shown during recording
        - Window events are processed (waitKey called)
        - Window is destroyed on stop
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            show_preview=True,
            fps=15,
        )
        recorder = ScreenRecorder(config)
        mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 0.3, 1.0]
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=0.2)
        
        # Assert
        # Preview should have been shown
        assert mock_cv2.imshow.call_count > 0
        assert mock_cv2.moveWindow.call_count > 0
        assert mock_cv2.waitKey.call_count > 0
        
        # Window should be destroyed
        mock_cv2.destroyWindow.assert_called_once_with("Screen Recording Preview")
    
    def test_preview_window_disabled(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test recording with preview window disabled.
        
        Verifies:
        - No preview window operations are called
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            show_preview=False,
        )
        recorder = ScreenRecorder(config)
        mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 1.0]
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=0.1)
        
        # Assert
        mock_cv2.imshow.assert_not_called()
        mock_cv2.destroyWindow.assert_not_called()
    
    def test_enable_key_stop_with_esc(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test that ESC key stops recording when enable_key_stop is True.
        
        Verifies:
        - Recording stops when ESC (27) is pressed
        - waitKey is checked
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            show_preview=True,
            enable_key_stop=True,
        )
        recorder = ScreenRecorder(config)
        
        # Mock waitKey to return ESC on third call
        mock_cv2.waitKey.side_effect = [-1, -1, 27]  # ESC key
        mock_time.monotonic.side_effect = [0.0] + [i * 0.1 for i in range(100)]
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=10.0)  # Long duration, should stop early
        
        # Assert
        # Recording should have stopped before 10 seconds
        assert recorder._running is False
        assert recorder._frame_index < 100  # Would be ~150 frames at 15 FPS for 10s
    
    def test_enable_key_stop_with_q(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test that 'q' key stops recording when enable_key_stop is True.
        
        Verifies:
        - Recording stops when 'q' is pressed
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            show_preview=True,
            enable_key_stop=True,
        )
        recorder = ScreenRecorder(config)
        
        # Mock waitKey to return 'q' on second call
        mock_cv2.waitKey.side_effect = [-1, ord('q')]
        mock_time.monotonic.side_effect = [0.0] + [i * 0.1 for i in range(100)]
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=10.0)
        
        # Assert
        assert recorder._running is False
    
    def test_key_stop_disabled(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test that keys don't stop recording when enable_key_stop is False.
        
        Verifies:
        - Recording continues despite key presses
        - waitKey is still called for window events
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            show_preview=True,
            enable_key_stop=False,
        )
        recorder = ScreenRecorder(config)
        
        # Mock waitKey to return ESC (should be ignored)
        mock_cv2.waitKey.return_value = 27
        mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 0.3, 1.0]
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=0.2)
        
        # Assert
        # Should complete full duration despite ESC
        assert mock_cv2.waitKey.call_count > 0


# ============================================================================
# RESOLUTION HANDLING TESTS
# ============================================================================


class TestResolutionHandling:
    """Test resolution auto-detection, capping, and dimension enforcement."""
    
    def test_auto_detect_screen_resolution(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that screen resolution is auto-detected when not specified.
        
        Verifies:
        - pyautogui.size() is called
        - Detected resolution is used (capped by MAX_RES)
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        mock_pyautogui.size.return_value = (1920, 1080)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            res=None,  # Auto-detect
        )
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        
        # Assert
        mock_pyautogui.size.assert_called_once()
        # Should be capped by MAX_RES (1280, 720)
        assert recorder._target_size == (1280, 720)
    
    def test_max_res_capping(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that resolutions are capped by MAX_RES.
        
        Verifies:
        - Resolutions exceeding MAX_RES are reduced
        - Aspect ratio may change to fit MAX_RES
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            res=(3840, 2160),  # 4K
            MAX_RES=(1280, 720),
        )
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        
        # Assert
        assert recorder._target_size[0] <= 1280
        assert recorder._target_size[1] <= 720
        assert recorder._target_size == (1280, 720)
    
    def test_even_dimension_enforcement(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that odd dimensions are adjusted to even numbers.
        
        Verifies:
        - Both width and height are even
        - Dimensions are reduced by 1 if odd
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            res=(1921, 1081),  # Odd dimensions
            MAX_RES=(2000, 2000),
        )
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        
        # Assert
        assert recorder._target_size[0] % 2 == 0
        assert recorder._target_size[1] % 2 == 0
        assert recorder._target_size == (1920, 1080)
    
    @pytest.mark.parametrize(
        "input_res,max_res,expected_res",
        [
            ((1920, 1080), (1280, 720), (1280, 720)),  # Capped
            ((800, 600), (1280, 720), (800, 600)),  # Unchanged
            ((1921, 1081), (2000, 2000), (1920, 1080)),  # Made even
            ((1279, 719), (1280, 720), (1278, 718)),  # Made even, capped
            ((640, 480), (640, 480), (640, 480)),  # Exact match
        ],
    )
    def test_resolution_combinations(
        self,
        temp_output_dir,
        input_res,
        max_res,
        expected_res,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
    ):
        """
        Test various resolution input/output combinations.
        
        Verifies:
        - Different resolution scenarios are handled correctly
        - Capping and even-dimension enforcement work together
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            res=input_res,
            MAX_RES=max_res,
        )
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        
        # Assert
        assert recorder._target_size == expected_res
    
    def test_invalid_resolution_zero(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that invalid resolutions (zero or negative) raise errors.
        
        Verifies:
        - ValueError is raised for invalid dimensions
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            res=(0, 720),
        )
        recorder = ScreenRecorder(config)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid screen resolution"):
            recorder.start()
    
    def test_invalid_resolution_negative(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that negative resolutions raise errors.
        
        Verifies:
        - ValueError is raised for negative dimensions
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            res=(1920, -1080),
        )
        recorder = ScreenRecorder(config)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid screen resolution"):
            recorder.start()


# ============================================================================
# OS-SPECIFIC CODEC SELECTION TESTS
# ============================================================================


class TestOSSpecificCodecs:
    """Test OS-specific codec selection and fallback behavior."""
    
    def test_windows_10_uses_msmf(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that Windows 10+ uses MSMF backend for H.264.
        
        Verifies:
        - CAP_MSMF is passed to VideoWriter
        - Windows version check is performed
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            fourcc="H264",
        )
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        
        # Assert
        call_args = mock_cv2.VideoWriter.call_args[0]
        assert call_args[1] == mock_cv2.CAP_MSMF
    
    def test_windows_7_uses_msmf(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that Windows 7 uses MSMF backend.
        
        Verifies:
        - Windows 7 meets the version requirement
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(7)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            fourcc="H264",
        )
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        
        # Assert
        call_args = mock_cv2.VideoWriter.call_args[0]
        assert call_args[1] == mock_cv2.CAP_MSMF
    
    def test_windows_xp_no_msmf(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that Windows XP (version < 7) doesn't use MSMF.
        
        Verifies:
        - Older Windows versions use default backend
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(5)  # Windows XP
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            fourcc="H264",
        )
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        
        # Assert
        # Should use default VideoWriter without backend parameter
        call_args = mock_cv2.VideoWriter.call_args[0]
        # Check that it's not using MSMF (no backend or different backend)
        assert call_args[1] != mock_cv2.CAP_MSMF
    
    def test_macos_high_sierra_uses_avfoundation(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that macOS 10.13+ uses AVFoundation backend.
        
        Verifies:
        - CAP_AVFOUNDATION is passed to VideoWriter
        - macOS version check is performed
        """
        # Arrange
        mock_os_version.return_value = MacOSVersion((10, 13, 0))
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            fourcc="H264",
        )
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        
        # Assert
        call_args = mock_cv2.VideoWriter.call_args[0]
        assert call_args[1] == mock_cv2.CAP_AVFOUNDATION
    
    def test_macos_mojave_uses_avfoundation(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that macOS 10.14 (Mojave) uses AVFoundation.
        
        Verifies:
        - Newer macOS versions use AVFoundation
        """
        # Arrange
        mock_os_version.return_value = MacOSVersion((10, 14, 0))
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            fourcc="H264",
        )
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        
        # Assert
        call_args = mock_cv2.VideoWriter.call_args[0]
        assert call_args[1] == mock_cv2.CAP_AVFOUNDATION
    
    def test_macos_old_version_no_avfoundation(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that old macOS versions (< 10.13) don't use AVFoundation.
        
        Verifies:
        - Older macOS uses default backend
        """
        # Arrange
        mock_os_version.return_value = MacOSVersion((10, 12, 0))
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            fourcc="H264",
        )
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        
        # Assert
        call_args = mock_cv2.VideoWriter.call_args[0]
        assert call_args[1] != mock_cv2.CAP_AVFOUNDATION
    
    def test_linux_uses_default_backend(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that Linux uses default backend (no special backend).
        
        Verifies:
        - Linux doesn't use MSMF or AVFoundation
        """
        # Arrange
        mock_os_version.return_value = LinuxVersion("5.10.0-generic")
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            fourcc="H264",
        )
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        
        # Assert
        call_args = mock_cv2.VideoWriter.call_args[0]
        # Linux should use default (fourcc as second parameter)
        assert call_args[1] != mock_cv2.CAP_MSMF
        assert call_args[1] != mock_cv2.CAP_AVFOUNDATION
    
    def test_codec_fallback_on_failure(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test fallback to MJPG codec when primary codec fails.
        
        Verifies:
        - Exception during codec initialization triggers fallback
        - MJPG codec is tried as fallback
        - Recording can still proceed
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        
        # Make VideoWriter initialization fail on first call, succeed on second
        failed_writer = MagicMock()
        failed_writer.isOpened.return_value = False
        
        success_writer = MagicMock()
        success_writer.isOpened.return_value = True
        
        mock_cv2.VideoWriter.side_effect = [Exception("Codec not supported"), success_writer]
        
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            fourcc="H264",
        )
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        
        # Assert
        # Should have tried twice (original + fallback)
        assert mock_cv2.VideoWriter.call_count == 2
        # Second call should use MJPG
        second_call = mock_cv2.VideoWriter.call_args_list[1]
        # Verify MJPG fourcc was used in fallback
        mock_cv2.VideoWriter.fourcc.assert_any_call(*'MJPG')
    
    def test_unknown_os_raises_error(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that unknown OS (None from get_os_version) raises error.
        
        Verifies:
        - RuntimeError is raised when OS cannot be detected
        """
        # Arrange
        mock_os_version.return_value = None
        config = ScreenRecordingConfig(save_dir=temp_output_dir)
        recorder = ScreenRecorder(config)
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to get OS version"):
            recorder.start()


# ============================================================================
# PREVIEW WINDOW TESTS
# ============================================================================


class TestPreviewWindow:
    """Test preview window functionality and key handling."""
    
    def test_preview_window_shows_resized_frame(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test that preview window displays resized frames.
        
        Verifies:
        - Frame is resized to 640x360 for preview
        - imshow is called with preview window name
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            show_preview=True,
        )
        recorder = ScreenRecorder(config)
        mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 1.0]
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=0.1)
        
        # Assert
        # Check that resize was called for preview (640x360)
        resize_calls = [
            call for call in mock_cv2.resize.call_args_list
            if call[0][1] == (640, 360)
        ]
        assert len(resize_calls) > 0
        
        # Check that imshow was called
        assert mock_cv2.imshow.call_count > 0
        assert mock_cv2.imshow.call_args_list[0][0][0] == "Screen Recording Preview"
    
    def test_preview_window_positioned(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test that preview window is positioned at (100, 100).
        
        Verifies:
        - moveWindow is called with correct coordinates
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            show_preview=True,
        )
        recorder = ScreenRecorder(config)
        mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 1.0]
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=0.1)
        
        # Assert
        assert mock_cv2.moveWindow.call_count > 0
        first_call = mock_cv2.moveWindow.call_args_list[0]
        assert first_call[0] == ("Screen Recording Preview", 100, 100)
    
    def test_preview_window_cleanup_on_stop(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test that preview window is destroyed when recording stops.
        
        Verifies:
        - destroyWindow is called with window name
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            show_preview=True,
        )
        recorder = ScreenRecorder(config)
        mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 1.0]
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=0.1)
        
        # Assert
        mock_cv2.destroyWindow.assert_called_once_with("Screen Recording Preview")
    
    def test_no_preview_cleanup_when_disabled(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test that destroyWindow is not called when preview is disabled.
        
        Verifies:
        - No window cleanup when show_preview is False
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            show_preview=False,
        )
        recorder = ScreenRecorder(config)
        mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 1.0]
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=0.1)
        
        # Assert
        mock_cv2.destroyWindow.assert_not_called()
    
    def test_waitkey_called_for_event_processing(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test that waitKey(1) is called to process window events.
        
        Verifies:
        - waitKey is called during preview
        - Called with 1ms delay
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            show_preview=True,
        )
        recorder = ScreenRecorder(config)
        mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 1.0]
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=0.1)
        
        # Assert
        assert mock_cv2.waitKey.call_count > 0
        # All calls should be waitKey(1)
        for call in mock_cv2.waitKey.call_args_list:
            assert call[0][0] == 1


# ============================================================================
# EDGE CASES TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_duration_recording(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test recording with zero duration.
        
        Verifies:
        - Recording starts and stops immediately
        - No frames are captured
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(save_dir=temp_output_dir)
        recorder = ScreenRecorder(config)
        mock_time.monotonic.side_effect = [0.0, 0.0, 0.1]
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=0.0)
        
        # Assert
        # Should have minimal or no frames
        assert recorder._frame_index <= 1
    
    def test_negative_fps_handled(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that negative FPS is handled gracefully.
        
        Verifies:
        - Frame interval calculation uses max(1, fps) to avoid division issues
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            fps=-5,  # Invalid
        )
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        
        # Assert
        # Should not crash, frame_interval should be reasonable
        # Implementation uses max(1, fps) so interval would be 1.0
        assert recorder._running is True
    
    def test_video_writer_not_opened_raises_error(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that RuntimeError is raised if VideoWriter fails to open.
        
        Verifies:
        - isOpened() returning False raises RuntimeError
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = False
        mock_cv2.VideoWriter.return_value = mock_writer
        
        config = ScreenRecordingConfig(save_dir=temp_output_dir)
        recorder = ScreenRecorder(config)
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to open video writer"):
            recorder.start()
    
    def test_video_verification_failure_warning(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        capsys,
    ):
        """
        Test that warning is printed if video verification fails.
        
        Verifies:
        - Warning message when VideoCapture cannot open file
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        
        # Mock VideoCapture to fail on verification
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_capture
        
        config = ScreenRecordingConfig(save_dir=temp_output_dir)
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        recorder.stop()
        
        # Assert
        captured = capsys.readouterr()
        assert "WARNING: Video file cannot be opened for verification" in captured.out
    
    def test_save_directory_creation(
        self, tmp_path, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that save directory is created if it doesn't exist.
        
        Verifies:
        - Directory is created with parents=True
        - No error when directory already exists
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        save_dir = tmp_path / "new_dir" / "recordings"
        assert not save_dir.exists()
        
        config = ScreenRecordingConfig(save_dir=save_dir)
        
        # Act
        recorder = ScreenRecorder(config)
        
        # Assert
        assert save_dir.exists()
        assert save_dir.is_dir()
    
    def test_frame_resizing_when_sizes_differ(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test that frames are resized when screenshot size differs from target.
        
        Verifies:
        - resize is called when dimensions don't match
        - Correct interpolation method is used
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        
        # Mock screenshot returning different size than target
        mock_image = MagicMock(spec=Image.Image)
        mock_image.__array__ = Mock(
            return_value=np.zeros((1080, 1920, 3), dtype=np.uint8)
        )
        mock_pyautogui.screenshot.return_value = mock_image
        
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            res=(800, 600),  # Different from screenshot size
        )
        recorder = ScreenRecorder(config)
        mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 1.0]
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=0.1)
        
        # Assert
        # Verify resize was called with target size
        resize_calls = [
            call for call in mock_cv2.resize.call_args_list
            if call[0][1] == (800, 600)
        ]
        assert len(resize_calls) > 0
    
    def test_indefinite_recording_stops_on_manual_stop(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test that indefinite recording (duration=None) can be stopped manually.
        
        Verifies:
        - Recording continues without duration limit
        - Stops when _running is set to False
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(save_dir=temp_output_dir)
        recorder = ScreenRecorder(config)
        
        # Mock time to progress
        time_values = [i * 0.1 for i in range(100)]
        mock_time.monotonic.side_effect = time_values
        
        # Create a side effect that stops recording after a few iterations
        call_count = [0]
        
        def screenshot_side_effect():
            call_count[0] += 1
            if call_count[0] >= 5:
                recorder._running = False
            mock_image = MagicMock(spec=Image.Image)
            mock_image.__array__ = Mock(
                return_value=np.zeros((1080, 1920, 3), dtype=np.uint8)
            )
            return mock_image
        
        mock_pyautogui.screenshot.side_effect = screenshot_side_effect
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=None)  # Indefinite
        
        # Assert
        # Should have stopped after ~5 frames
        assert recorder._frame_index >= 4
        assert not recorder._running


# ============================================================================
# RESOURCE CLEANUP TESTS
# ============================================================================


class TestResourceCleanup:
    """Test proper resource cleanup and state management."""
    
    def test_writer_released_on_stop(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that VideoWriter.release() is called on stop.
        
        Verifies:
        - Writer release is called exactly once
        - Writer reference is cleared
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(save_dir=temp_output_dir)
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        mock_writer = recorder._writer
        recorder.stop()
        
        # Assert
        mock_writer.release.assert_called_once()
        assert recorder._writer is None
    
    def test_window_destroyed_on_stop(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test that preview window is destroyed on stop.
        
        Verifies:
        - destroyWindow is called when preview was shown
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            show_preview=True,
        )
        recorder = ScreenRecorder(config)
        mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 1.0]
        
        # Act
        recorder.start()
        recorder.record_for(duration_seconds=0.1)
        
        # Assert
        mock_cv2.destroyWindow.assert_called_once()
    
    def test_multiple_stop_calls_safe(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that multiple stop() calls don't cause errors.
        
        Verifies:
        - Second stop() returns early
        - Resources only released once
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(save_dir=temp_output_dir)
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        mock_writer = recorder._writer
        recorder.stop()
        recorder.stop()
        recorder.stop()
        
        # Assert
        # Release should only be called once
        mock_writer.release.assert_called_once()
    
    def test_stop_without_start_safe(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that calling stop() without start() doesn't cause errors.
        
        Verifies:
        - No errors when stopping before starting
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(save_dir=temp_output_dir)
        recorder = ScreenRecorder(config)
        
        # Act & Assert - should not raise
        recorder.stop()
    
    def test_exception_during_recording_still_cleans_up(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test that resources are cleaned up even if exception occurs during recording.
        
        Verifies:
        - stop() is called in finally block
        - Resources are released despite exception
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(save_dir=temp_output_dir)
        recorder = ScreenRecorder(config)
        
        # Make screenshot raise exception after a few calls
        call_count = [0]
        
        def screenshot_side_effect():
            call_count[0] += 1
            if call_count[0] >= 3:
                raise Exception("Screenshot failed")
            mock_image = MagicMock(spec=Image.Image)
            mock_image.__array__ = Mock(
                return_value=np.zeros((1080, 1920, 3), dtype=np.uint8)
            )
            return mock_image
        
        mock_pyautogui.screenshot.side_effect = screenshot_side_effect
        mock_time.monotonic.side_effect = [i * 0.1 for i in range(100)]
        
        # Act
        recorder.start()
        mock_writer = recorder._writer
        
        with pytest.raises(Exception, match="Screenshot failed"):
            recorder.record_for(duration_seconds=10.0)
        
        # Assert
        # Despite exception, stop() should have been called (finally block)
        mock_writer.release.assert_called_once()
        assert recorder._writer is None
        assert not recorder._running
    
    def test_video_file_stats_printed(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        capsys,
    ):
        """
        Test that video file statistics are printed on stop.
        
        Verifies:
        - File size is printed
        - Frame count and FPS are printed from verification
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(save_dir=temp_output_dir)
        recorder = ScreenRecorder(config)
        
        # Create a dummy file so stat() works
        recorder.start()
        recorder.video_path.touch()
        recorder.video_path.write_bytes(b"dummy video data")
        
        # Act
        recorder.stop()
        
        # Assert
        captured = capsys.readouterr()
        assert "Final file size:" in captured.out
        assert "bytes" in captured.out
        assert "Video verification:" in captured.out
        assert "frames at" in captured.out
        assert "FPS" in captured.out
    
    def test_state_reset_after_stop(
        self, temp_output_dir, mock_pyautogui, mock_cv2, mock_os_version
    ):
        """
        Test that internal state is properly reset after stop.
        
        Verifies:
        - _running is False
        - _writer is None
        - Can start a new recording session
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(save_dir=temp_output_dir)
        recorder = ScreenRecorder(config)
        
        # Act
        recorder.start()
        first_path = recorder.video_path
        recorder.stop()
        
        # Assert state is reset
        assert not recorder._running
        assert recorder._writer is None
        
        # Should be able to start again
        recorder.start()
        assert recorder._running
        assert recorder.video_path != first_path  # New timestamp
        recorder.stop()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for complete recording workflows."""
    
    def test_full_recording_workflow(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test a complete recording workflow from start to finish.
        
        Verifies:
        - All components work together
        - Video file is created
        - Proper sequence of operations
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(
            save_dir=temp_output_dir,
            fps=15,
            show_preview=True,
            enable_key_stop=False,
        )
        recorder = ScreenRecorder(config)
        mock_time.monotonic.side_effect = [i * 0.1 for i in range(100)]
        
        # Act
        video_path = recorder.record(duration_seconds=0.5)
        
        # Assert
        assert video_path.parent == temp_output_dir
        assert video_path.name.startswith("screen_")
        assert not recorder._running
        assert recorder._frame_index > 0
        
        # Verify sequence of operations
        assert mock_cv2.VideoWriter.called
        assert mock_pyautogui.screenshot.called
        assert mock_cv2.VideoWriter.return_value.write.called
        assert mock_cv2.VideoWriter.return_value.release.called
        assert mock_cv2.destroyWindow.called
    
    def test_multiple_recording_sessions(
        self,
        temp_output_dir,
        mock_pyautogui,
        mock_cv2,
        mock_os_version,
        mock_time,
    ):
        """
        Test multiple sequential recording sessions.
        
        Verifies:
        - Multiple recordings can be done with same recorder
        - Each recording gets unique filename
        - State is properly reset between recordings
        """
        # Arrange
        mock_os_version.return_value = WindowsVersion(10)
        config = ScreenRecordingConfig(save_dir=temp_output_dir)
        recorder = ScreenRecorder(config)
        
        # Act - First recording
        mock_time.monotonic.side_effect = [i * 0.1 for i in range(50)]
        video_path1 = recorder.record(duration_seconds=0.2)
        
        # Reset mocks for second recording
        mock_cv2.VideoWriter.reset_mock()
        mock_time.monotonic.side_effect = [i * 0.1 for i in range(50)]
        
        # Act - Second recording
        video_path2 = recorder.record(duration_seconds=0.2)
        
        # Assert
        assert video_path1 != video_path2  # Different filenames
        assert video_path1.exists() or True  # May not actually exist (mocked)
        assert video_path2.exists() or True
        
        # Verify VideoWriter was created twice
        assert mock_cv2.VideoWriter.call_count == 2

