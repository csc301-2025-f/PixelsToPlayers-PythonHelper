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
    width: Optional[int] = None  # None → use full screen width
    height: Optional[int] = None # None → use full screen height
    fps: int = 15                # target frame rate
    show_preview: bool = False   # show live preview window
    save_dir: Path = Path(__file__).parent / "recordings"
    fourcc: str = "mp4v"         # video codec


class ScreenRecorder:
    """Record the screen to a video file."""

    def __init__(self, cfg: ScreenRecordingConfig | None = None) -> None:
        self.cfg = cfg or ScreenRecordingConfig()
        self.cfg.save_dir.mkdir(parents=True, exist_ok=True)
        self._writer: Optional[cv2.VideoWriter] = None
        self._running: bool = False
        self._frame_index: int = 0
        self.video_path: Optional[Path] = None
        self._window_name = "screen"

    def start(self) -> Path:
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
        if not self._running:
            return
        self._running = False

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        if self.cfg.show_preview:
            cv2.destroyWindow(self._window_name)

    # convenience: one-shot API
    def record(self, duration_seconds: Optional[float] = None) -> Path:
        video_path = self.start()
        self.record_for(duration_seconds)
        return video_path