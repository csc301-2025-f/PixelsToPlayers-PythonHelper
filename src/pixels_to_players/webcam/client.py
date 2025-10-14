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
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    show_preview: bool = True
    save_dir: Path = Path(__file__).parent / "recordings"
    fourcc: str = "mp4v"         # 'mp4v' for .mp4, 'XVID' for .avi
    backend: int = cv2.CAP_ANY   # CAP_DSHOW/CAP_MSMF on Windows, V4L2 on Linux


class WebcamClient:
    """
    Owns the camera lifecycle and provides simple verbs:
    - open()/close()
    - snapshot(processors=...)
    - record(duration, processors=...)
    """

    def __init__(self, cfg: WebcamConfig | None = None) -> None:
        self.cfg = cfg or WebcamConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        self.cfg.save_dir.mkdir(parents=True, exist_ok=True)

    # ---- context manager support ----
    def __enter__(self) -> "WebcamClient":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- lifecycle ----
    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.cfg.device_index, self.cfg.backend)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.cfg.device_index}")

        # Request properties; some cameras may not honor them exactly
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        # Safe even if no windows were created
        cv2.destroyAllWindows()

    # ---- helpers ----
    def _actual_props(self) -> Tuple[int, int, int]:
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
        Record for `duration` seconds (or indefinitely if None).
        Returns the output file path.
        """
        if not self.cap or not self.cap.isOpened():
            self.open()

        w, h, fps = self._actual_props()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.cfg.save_dir / f"recording_{ts}.mp4"

        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*self.cfg.fourcc),
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
                    # End of stream / device error
                    break

                for p in processors or []:
                    frame = p(frame)

                if self.cfg.show_preview:
                    cv2.imshow("webcam", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):  # ESC or 'q'
                        break

                writer.write(frame)
        finally:
            writer.release()
            if self.cfg.show_preview:
                # Destroy just in case; no-op if not shown
                cv2.destroyWindow("webcam")

        return out_path
