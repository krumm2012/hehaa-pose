"""Reader loop timing and source-frame bookkeeping helpers."""

import time


class DeadlinePacer:
    """Keep a stable frame cadence by scheduling against one absolute timeline."""

    def __init__(
        self,
        fps,
        start_time=None,
        clock=time.perf_counter,
        max_lag_intervals=None,
    ):
        self.interval = 1.0 / float(fps) if fps and fps > 0 else 0.0
        self._clock = clock
        self._deadline = self._clock() if start_time is None else float(start_time)
        self.max_lag_intervals = (
            None if max_lag_intervals is None else max(0.0, float(max_lag_intervals))
        )

    def next_delay(self, now=None):
        if self.interval <= 0:
            return 0.0
        current = self._clock() if now is None else float(now)
        next_deadline = self._deadline + self.interval
        if (
            self.max_lag_intervals is not None
            and current - next_deadline > self.interval * self.max_lag_intervals
        ):
            self._deadline = current
            return 0.0
        self._deadline = next_deadline
        return max(0.0, self._deadline - current)


class SourceFrameClock:
    """Track source time independently from the number of processed frames."""

    def __init__(self):
        self.source_count = 0
        self.processed_count = 0

    def accepted(self):
        frame_id = self.source_count
        self.source_count += 1
        self.processed_count += 1
        return frame_id

    def dropped(self):
        frame_id = self.source_count
        self.source_count += 1
        return frame_id
