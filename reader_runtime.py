"""Reader loop timing and source-frame bookkeeping helpers."""

import time
import math

from analysis_data_contracts import SOURCE_TIME_SCHEMA_VERSION


class SourceMediaClock:
    """Describe media time without confusing decode/receipt time with exposure.

    OpenCV POS_MSEC is a backend-reported presentation timestamp for the input
    file, not proof of sensor exposure or a mapping to a pre-transcode original.
    Stream device timestamps are unavailable through the current reader.
    """

    def __init__(self, fps, source_kind="video_file"):
        self.fps = float(fps or 0)
        self.source_kind = source_kind
        self.previous_pts = None

    def observe(self, frame_id, position_ms=None):
        record = {"schema_version": SOURCE_TIME_SCHEMA_VERSION,
                  "source_kind": self.source_kind, "source_frame_id": int(frame_id),
                  "timestamp_seconds": None, "basis": "unavailable",
                  "quality": "unavailable", "exposure_time_verified": False}
        if self.source_kind == "stream":
            record["reason"] = "device_media_time_unavailable"
            return record
        try:
            pts = float(position_ms) / 1000
        except (TypeError, ValueError):
            pts = float("nan")
        if math.isfinite(pts) and pts >= 0:
            quality = "reported"
            if self.previous_pts is not None:
                if pts == self.previous_pts:
                    quality = "duplicate"
                elif pts < self.previous_pts:
                    quality = "discontinuous"
            self.previous_pts = pts
            record.update(timestamp_seconds=pts, basis="media_pts", quality=quality,
                          provider="opencv_pos_msec")
        elif math.isfinite(self.fps) and self.fps > 0:
            record.update(timestamp_seconds=frame_id / self.fps, basis="nominal_fps",
                          quality="estimated", reason="media_pts_unavailable", nominal_fps=self.fps)
        else:
            record["reason"] = "media_pts_and_fps_unavailable"
        return record


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
