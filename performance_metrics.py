"""Lightweight runtime performance metrics."""

from collections import deque
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class FpsSnapshot:
    frame_count: int
    cumulative_fps: float
    window_25_fps: float
    window_100_fps: float


class FpsTracker:
    """Track cumulative and recent FPS from the first received frame."""

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._start_time = None
        self._frame_count = 0
        self._samples = deque(maxlen=101)

    def start(self, now=None):
        if self._start_time is not None:
            return
        start_time = self._clock() if now is None else float(now)
        self._start_time = start_time
        self._samples.append((0, start_time))

    def _window_fps(self, window_size, now):
        target_count = self._frame_count - window_size
        if target_count < 0:
            return 0.0
        for frame_count, timestamp in self._samples:
            if frame_count == target_count:
                elapsed = now - timestamp
                return window_size / elapsed if elapsed > 0 else 0.0
        return 0.0

    def tick(self, now=None):
        timestamp = self._clock() if now is None else float(now)
        if self._start_time is None:
            self.start(now=timestamp)
        self._frame_count += 1
        self._samples.append((self._frame_count, timestamp))
        elapsed = timestamp - self._start_time
        cumulative_fps = self._frame_count / elapsed if elapsed > 0 else 0.0
        return FpsSnapshot(
            frame_count=self._frame_count,
            cumulative_fps=cumulative_fps,
            window_25_fps=self._window_fps(25, timestamp),
            window_100_fps=self._window_fps(100, timestamp),
        )
