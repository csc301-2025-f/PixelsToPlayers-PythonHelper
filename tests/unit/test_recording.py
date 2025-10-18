import os
import time
import unittest
from pathlib import Path

from pixels_to_players.screen_recording.recording import (
    ScreenRecorder,
    ScreenRecordingConfig,
)


class TestScreenRecording(unittest.TestCase):
    """Unit tests for pixels_to_players screen recording."""

    def setUp(self):
        self._created_files: list[Path] = []

    def tearDown(self):

        if os.getenv("P2P_KEEP_RECORDINGS") == "1":
            print("P2P_KEEP_RECORDINGS=1 -> keeping recordings for inspection.")
            return

        print("automatically removing recordings. Set P2P_KEEP_RECORDINGS=1 to keep.")

        # Remove any files created by a test
        for p in self._created_files:
            try:
                p.unlink(missing_ok=True)
            except Exception as e:
                # Best-effort cleanup; don't fail the suite on cleanup errors
                print(f"Error cleaning up {p}: {e}")
                pass

    # ---- helpers ---------------------------------------------------------

    def _assert_video_created(self, path: Path) -> int:
        self.assertIsInstance(path, Path, "record() should return a pathlib.Path")
        self.assertTrue(path.exists(), f"Expected video file to exist: {path}")
        size = path.stat().st_size
        self.assertGreater(size, 0, "Expected non-empty video file")
        self._created_files.append(path)
        return size

    # ---- tests -----------------------------------------------------------

    def test_basic_recording(self):
        """Records for a short duration without preview or key-stop."""
        config = ScreenRecordingConfig(fps=15, show_preview=False, enable_key_stop=False)
        recorder = ScreenRecorder(config)
        video_path = recorder.record(duration_seconds=5.0)
        self._assert_video_created(video_path)

    @unittest.skipUnless(
        os.environ.get("P2P_INTERACTIVE") == "1",
        "Requires manual key presses to validate behavior (ESC/q should NOT stop)."
        " Set P2P_INTERACTIVE=1 to enable."
    )
    def test_gaming_mode(self):
        """With preview, key-stop disabled: ESC/q should not stop recording."""
        config = ScreenRecordingConfig(fps=10, show_preview=True, enable_key_stop=False)
        recorder = ScreenRecorder(config)
        video_path = recorder.record(duration_seconds=5.0)
        self._assert_video_created(video_path)

    @unittest.skipUnless(
        os.environ.get("P2P_INTERACTIVE") == "1",
        "Requires manual key presses to validate behavior (ESC/q SHOULD stop)."
        " Set P2P_INTERACTIVE=1 to enable."
    )
    def test_key_stop_mode(self):
        """With preview, key-stop enabled: ESC/q should stop recording early."""
        config = ScreenRecordingConfig(fps=10, show_preview=True, enable_key_stop=True)
        recorder = ScreenRecorder(config)
        video_path = recorder.record(duration_seconds=8.0)
        self._assert_video_created(video_path)

    def test_manual_control(self):
        """Manual start/stop API produces a file."""
        config = ScreenRecordingConfig(fps=10, show_preview=False, enable_key_stop=False)
        recorder = ScreenRecorder(config)

        video_path = recorder.start()
        self.assertIsInstance(video_path, Path, "start() should return the target Path")
        time.sleep(3.0)
        recorder.stop()

        self._assert_video_created(video_path)


if __name__ == "__main__":
    unittest.main()
