"""Incremental Swing event analysis shared by live and recorded pipelines."""

from __future__ import annotations

import html
import json
import os
import queue
import re
import threading
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from analysis_data_contracts import (
    EVENT_LOG_SCHEMA_VERSION,
    document_contract,
    stamp_swing_event,
    utc_iso_from_ns,
)
from swing_event_analyzer import analyze_frame_records
from swing_session_quality import build_session_quality_dashboard
from video_writer_backend import create_video_writer


class RealtimeFrameJournal:
    """Persist processed frame records without blocking the analysis loop."""

    def __init__(
        self,
        jsonl_path: str,
        snapshot_path: str,
        snapshot_size: int = 200,
        flush_interval: int = 5,
        queue_size: int = 1024,
        session_metadata: Optional[Dict] = None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.snapshot_path = Path(snapshot_path)
        self.snapshot_size = max(1, int(snapshot_size))
        self.flush_interval = max(1, int(flush_interval))
        self._recent: Deque[Dict] = deque(maxlen=self.snapshot_size)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._sentinel = object()
        self._closed = False
        self._frame_count = 0
        self._dropped_records = 0
        self._worker_error: Optional[BaseException] = None
        self.session_metadata = deepcopy(session_metadata or {})
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._write_loop,
            name="realtime-frame-journal",
            daemon=True,
        )
        self._thread.start()

    def record(self, frame_record: Dict) -> None:
        """Queue one processed frame record for durable and snapshot outputs."""
        if self._closed:
            raise RuntimeError("Cannot record a frame after the realtime frame journal is closed")
        if self._worker_error is not None:
            raise RuntimeError("Realtime frame journal writer failed") from self._worker_error
        item = deepcopy(frame_record)
        self._queue.put(item)

    def close(self) -> None:
        """Drain queued records and publish the final recent-frame snapshot."""
        if self._closed:
            return
        self._closed = True
        self._queue.join()
        self._queue.put(self._sentinel)
        self._queue.join()
        self._thread.join()
        if self._worker_error is not None:
            raise RuntimeError("Realtime frame journal writer failed") from self._worker_error

    def _write_loop(self) -> None:
        stream = None
        try:
            try:
                stream = self.jsonl_path.open("w", encoding="utf-8")
            except Exception as exc:
                self._worker_error = exc

            while True:
                item = self._queue.get()
                try:
                    if item is self._sentinel:
                        if self._worker_error is None and stream is not None:
                            stream.flush()
                            self._write_snapshot()
                        return
                    if self._worker_error is not None or stream is None:
                        continue
                    stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
                    stream.write("\n")
                    self._recent.append(item)
                    self._frame_count += 1
                    if self._frame_count % self.flush_interval == 0:
                        stream.flush()
                        self._write_snapshot()
                except Exception as exc:
                    self._worker_error = exc
                    if item is self._sentinel:
                        return
                finally:
                    self._queue.task_done()
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception as exc:
                    if self._worker_error is None:
                        self._worker_error = exc

    def _write_snapshot(self) -> None:
        latest_frame = (
            int(self._recent[-1].get("frame_id", -1))
            if self._recent
            else -1
        )
        document = {
            **document_contract("frames", self.session_metadata),
            "summary": {
                "frame_count": self._frame_count,
                "latest_frame": latest_frame,
                "snapshot_size": self.snapshot_size,
                "dropped_records": self._dropped_records,
                "realtime": True,
            },
            "frames": list(self._recent),
        }
        temporary = self.snapshot_path.with_name(f".{self.snapshot_path.name}.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.snapshot_path)


class RealtimeEventJournal:
    """Append every event creation and asynchronous patch to a durable JSONL log."""

    def __init__(
        self,
        path: str,
        session_metadata: Optional[Dict] = None,
        queue_size: int = 1024,
    ):
        self.path = Path(path)
        self.session_metadata = deepcopy(session_metadata or {})
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._sentinel = object()
        self._closed = False
        self._worker_error: Optional[BaseException] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._write_loop,
            name="realtime-event-journal",
            daemon=True,
        )
        self._thread.start()

    def append(self, operation: str, event_id: int, payload: Dict) -> None:
        if self._closed:
            raise RuntimeError("Cannot append after the realtime event journal is closed")
        if self._worker_error is not None:
            raise RuntimeError("Realtime event journal writer failed") from self._worker_error
        self._queue.put(
            {
                "operation": str(operation),
                "event_id": int(event_id),
                "payload": deepcopy(payload),
            }
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.join()
        self._queue.put(self._sentinel)
        self._queue.join()
        self._thread.join()
        if self._worker_error is not None:
            raise RuntimeError("Realtime event journal writer failed") from self._worker_error

    def _write_loop(self) -> None:
        stream = None
        sequence = 0
        try:
            try:
                sequence = self._existing_sequence()
                needs_separator = self._needs_separator()
                stream = self.path.open("a", encoding="utf-8")
                if needs_separator:
                    stream.write("\n")
                    stream.flush()
            except Exception as exc:
                self._worker_error = exc
            while True:
                item = self._queue.get()
                try:
                    if item is self._sentinel:
                        if stream is not None:
                            stream.flush()
                        return
                    if self._worker_error is not None or stream is None:
                        continue
                    sequence += 1
                    recorded_ns = time.time_ns()
                    row = {
                        "schema_version": EVENT_LOG_SCHEMA_VERSION,
                        "sequence": sequence,
                        "recorded_at": utc_iso_from_ns(recorded_ns),
                        "recorded_at_unix_ns": recorded_ns,
                        "session_id": str(
                            self.session_metadata.get("session_id") or ""
                        ),
                        **item,
                    }
                    stream.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    )
                    stream.write("\n")
                    stream.flush()
                except Exception as exc:
                    self._worker_error = exc
                    if item is self._sentinel:
                        return
                finally:
                    self._queue.task_done()
        finally:
            if stream is not None:
                stream.close()

    def _existing_sequence(self) -> int:
        """Resume after the greatest valid sequence already persisted."""
        if not self.path.exists():
            return 0
        greatest = 0
        with self.path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except (TypeError, ValueError):
                    continue
                value = row.get("sequence") if isinstance(row, dict) else None
                if isinstance(value, int) and value > greatest:
                    greatest = value
        return greatest

    def _needs_separator(self) -> bool:
        """Keep a crash-truncated final row separate from the next JSON record."""
        try:
            if self.path.stat().st_size == 0:
                return False
            with self.path.open("rb") as stream:
                stream.seek(-1, os.SEEK_END)
                return stream.read(1) != b"\n"
        except (FileNotFoundError, OSError):
            return False


class RealtimeSwingEventEngine:
    """Emit stable, completed Swing events from an incoming frame-record stream.

    The engine deliberately reuses ``analyze_frame_records`` as the single
    analysis implementation. Its interface only adds rolling-window state,
    completion latency, stable event IDs, and duplicate suppression.
    """

    def __init__(
        self,
        fps: float,
        analysis_interval_frames: int = 5,
        settle_frames: Optional[int] = None,
        window_frames: Optional[int] = None,
        dominant_hand: str = "right",
        min_peak_energy: float = 9.0,
        active_energy: float = 5.5,
        min_event_frames: int = 8,
        max_internal_gap: int = 3,
        min_event_gap: int = 18,
        coach=None,
        session_metadata: Optional[Dict] = None,
    ):
        self.fps = max(1.0, float(fps or 25.0))
        self.analysis_interval_frames = max(1, int(analysis_interval_frames))
        self.settle_frames = max(
            0,
            int(round(self.fps * 0.6)) if settle_frames is None else int(settle_frames),
        )
        self.window_frames = max(
            32,
            int(round(self.fps * 8.0)) if window_frames is None else int(window_frames),
        )
        self.options = {
            "dominant_hand": dominant_hand,
            "min_peak_energy": float(min_peak_energy),
            "active_energy": float(active_energy),
            "min_event_frames": max(1, int(min_event_frames)),
            "max_internal_gap": max(0, int(max_internal_gap)),
            "min_event_gap": max(0, int(min_event_gap)),
        }
        # Keep realtime duplicate suppression aligned with the segmenter's
        # quality-ordered peak NMS.  The segmenter enforces at least 1.6s
        # between event peaks even when min_event_gap is configured lower.
        self.peak_dedup_frames = max(
            self.options["min_event_gap"],
            int(round(self.fps * 1.6)),
        )
        self.coach = coach
        self.session_metadata = deepcopy(session_metadata or {})
        self._frames: Deque[Dict] = deque(maxlen=self.window_frames)
        self._events: List[Dict] = []
        self._frame_trace: List[Dict] = []
        self._emitted_peaks: List[int] = []
        self._total_frames = 0
        self._latest_frame_id = -1

    def push_frame(self, frame_record: Dict) -> List[Dict]:
        """Consume one frame record and return newly completed events."""
        record = deepcopy(frame_record)
        self._frames.append(record)
        self._total_frames += 1
        self._latest_frame_id = int(record.get("frame_id", self._latest_frame_id + 1))
        if self._total_frames % self.analysis_interval_frames:
            return []
        return self._analyze(force=False)

    def flush(self) -> List[Dict]:
        """Finalize any valid Swing remaining when the source stream ends."""
        return self._analyze(force=True)

    def snapshot(self) -> Dict:
        """Return the complete live event document written to JSON/frontends."""
        type_counts = Counter(event["stroke_type"] for event in self._events)
        return {
            **document_contract("swing_events", self.session_metadata),
            "summary": {
                "total_frames": self._total_frames,
                "latest_frame": self._latest_frame_id,
                "swing_event_count": len(self._events),
                "swing_event_type_counts": dict(sorted(type_counts.items())),
                "thresholds": dict(self.options),
                "realtime": True,
                "settle_frames": self.settle_frames,
                "analysis_interval_frames": self.analysis_interval_frames,
                "window_frames": self.window_frames,
                "peak_dedup_frames": self.peak_dedup_frames,
                "session_quality": build_session_quality_dashboard(self._events),
            },
            "events": deepcopy(self._events),
            "frame_trace": deepcopy(self._frame_trace),
        }

    def frame_records_for_event(self, event: Dict) -> List[Dict]:
        """Return retained per-frame analysis records for one Swing event."""
        start_frame = int(event["start_frame"])
        end_frame = int(event["end_frame"])
        return [
            deepcopy(record)
            for record in self._frames
            if start_frame <= int(record.get("frame_id", -1)) <= end_frame
        ]

    def _analyze(self, force: bool) -> List[Dict]:
        analysis_frames = list(self._frames)
        if self._events:
            # Published event ranges are immutable.  Re-analyzing their tail
            # lets a weaker secondary peak become dominant after the original
            # peak leaves the rolling window, which previously emitted
            # overlapping duplicate events.  Only analyze the uncommitted
            # timeline after the latest published event.
            committed_through = max(int(event["end_frame"]) for event in self._events)
            analysis_frames = [
                record
                for record in analysis_frames
                if int(record.get("frame_id", -1)) > committed_through
            ]
        if len(analysis_frames) < self.options["min_event_frames"]:
            return []

        analysis = analyze_frame_records(
            analysis_frames,
            session_metadata=self.session_metadata,
            **self.options,
        )
        emitted = []
        for candidate in analysis.get("events", []):
            end_frame = int(candidate["end_frame"])
            if not force and self._latest_frame_id - end_frame < self.settle_frames:
                continue
            if self._is_duplicate(candidate):
                continue

            event = deepcopy(candidate)
            event["event_id"] = len(self._events) + 1
            event["emitted_at_frame"] = self._latest_frame_id
            event["latency_frames"] = max(0, self._latest_frame_id - end_frame)
            contact_frame = int(event["contact_frame"])
            contact_record = next(
                (
                    record
                    for record in self._frames
                    if int(record.get("frame_id", -1)) == contact_frame
                ),
                None,
            )
            stamp_swing_event(
                event,
                session=self.session_metadata,
                emitted_at_unix_ns=time.time_ns(),
                contact_frame_record=contact_record,
            )
            if self.coach is not None:
                if hasattr(self.coach, "advise_all"):
                    coach_advices = self.coach.advise_all(event)
                else:
                    coach_advices = [self.coach.advise(event)]
                event["coach_advices"] = coach_advices
                event["coach_advice"] = coach_advices[0]
                coach_generated_ns = time.time_ns()
                timing = dict(event.get("timing") or {})
                timing["coach_generated_at"] = utc_iso_from_ns(coach_generated_ns)
                timing["coach_generated_at_unix_ns"] = coach_generated_ns
                emitted_ns = timing.get("event_emitted_at_unix_ns")
                if isinstance(emitted_ns, int) and emitted_ns > 0:
                    timing["event_to_coach_ms"] = round(
                        max(0, coach_generated_ns - emitted_ns) / 1_000_000,
                        3,
                    )
                contact_capture_ns = (
                    ((contact_record or {}).get("timing") or {}).get(
                        "captured_at_unix_ns"
                    )
                )
                if isinstance(contact_capture_ns, int) and contact_capture_ns > 0:
                    timing["contact_capture_to_coach_ms"] = round(
                        max(0, coach_generated_ns - contact_capture_ns) / 1_000_000,
                        3,
                    )
                event["timing"] = timing
            self._events.append(event)
            self._emitted_peaks.append(int(event["peak_frame"]))
            self._append_event_trace(analysis.get("frame_trace", []), candidate, event)
            emitted.append(deepcopy(event))
        return emitted

    def _is_duplicate(self, candidate: Dict) -> bool:
        peak_frame = int(candidate["peak_frame"])
        tolerance = max(1, int(self.peak_dedup_frames))
        if any(
            abs(peak_frame - emitted_peak) <= tolerance
            for emitted_peak in self._emitted_peaks
        ):
            return True

        # A rolling window can later promote a weaker secondary peak after the
        # original dominant peak leaves the window edge. If that new peak lies
        # inside an already-published Swing range, it is the same action rather
        # than a newly completed Swing.
        candidate_start = int(candidate["start_frame"])
        candidate_end = int(candidate["end_frame"])
        return any(
            max(candidate_start, int(event["start_frame"]))
            <= min(candidate_end, int(event["end_frame"]))
            for event in self._events
        )

    def _append_event_trace(self, traces: List[Dict], candidate: Dict, event: Dict) -> None:
        start_frame = int(candidate["start_frame"])
        end_frame = int(candidate["end_frame"])
        existing_frames = {int(row["frame"]) for row in self._frame_trace}
        for row in traces:
            frame_id = int(row["frame"])
            if frame_id in existing_frames or not start_frame <= frame_id <= end_frame:
                continue
            trace = deepcopy(row)
            trace["event_id"] = event["event_id"]
            self._frame_trace.append(trace)


class RealtimeSwingOutputManager:
    """Publish live event state and encode independent Swing clips off-thread."""

    def __init__(
        self,
        output_json: str,
        output_html: str,
        clips_dir: str,
        fps: float,
        frame_size: Tuple[int, int],
        buffer_frames: int,
        clip_workers: int = 1,
        video_backend: str = "auto",
        video_bitrate: str = "8M",
        clip_padding_frames: int = 0,
        clip_max_width: int = 1280,
        jpeg_quality: int = 85,
        preview_path: Optional[str] = None,
        roi_metadata: Optional[Dict] = None,
        preview_interval_frames: int = 25,
        event_log_path: Optional[str] = None,
        session_metadata: Optional[Dict] = None,
    ):
        self.output_json = Path(output_json)
        self.output_html = Path(output_html)
        self.clips_dir = Path(clips_dir)
        self.fps = max(1.0, float(fps or 25.0))
        source_width = int(frame_size[0])
        source_height = int(frame_size[1])
        max_width = max(160, int(clip_max_width))
        scale = min(1.0, max_width / max(1, source_width))
        self.width = max(2, int(round(source_width * scale)) // 2 * 2)
        self.height = max(2, int(round(source_height * scale)) // 2 * 2)
        self.jpeg_quality = max(40, min(100, int(jpeg_quality)))
        self.video_backend = video_backend
        self.video_bitrate = video_bitrate
        self.clip_padding_frames = max(0, int(clip_padding_frames))
        self.preview_path = Path(preview_path) if preview_path else None
        self.roi_metadata = deepcopy(roi_metadata or {})
        self.preview_interval_frames = max(1, int(preview_interval_frames))
        self.session_metadata = deepcopy(session_metadata or {})
        self.event_log_path = Path(event_log_path) if event_log_path else self.output_json.with_suffix(".jsonl")
        self._last_preview_frame: Optional[int] = None
        self._frames: Deque[Tuple[int, object]] = deque(maxlen=max(1, int(buffer_frames)))
        self._events: List[Dict] = []
        self._buffer_condition = threading.Condition()
        self._output_lock = threading.Lock()
        self._output_queue: queue.Queue = queue.Queue()
        self._output_sentinel = object()
        self._output_worker_error: Optional[BaseException] = None
        raw_frame_bytes = max(1, source_width * source_height * 3)
        memory_bounded_queue_frames = max(
            4,
            (96 * 1024 * 1024) // raw_frame_bytes,
        )
        compression_queue_frames = min(
            16,
            max(4, int(buffer_frames)),
            memory_bounded_queue_frames,
        )
        self._frame_queue: queue.Queue = queue.Queue(maxsize=compression_queue_frames)
        self._frame_sentinel = object()
        self._compressor_stopped = False
        self._closed = False
        self._latest_recorded_frame = -1
        self._latest_buffered_frame = -1
        self._dropped_compression_frames = 0
        self._compression_errors: Dict[int, str] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(clip_workers)),
            thread_name_prefix="swing-clip",
        )
        self._futures: List[Future] = []
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        self.output_html.parent.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        if self.preview_path is not None:
            self.preview_path.parent.mkdir(parents=True, exist_ok=True)
        self._event_journal = RealtimeEventJournal(
            str(self.event_log_path),
            session_metadata=self.session_metadata,
        )
        self._document = {
            **document_contract("swing_events", self.session_metadata),
            "summary": {
                "total_frames": 0,
                "latest_frame": -1,
                "swing_event_count": 0,
                "swing_event_type_counts": {},
                "realtime": True,
                "roi": deepcopy(self.roi_metadata),
            },
            "events": [],
            "frame_trace": [],
        }
        self._write_live_outputs(self._document)
        self._output_thread = threading.Thread(
            target=self._write_outputs_loop,
            name="swing-live-output",
            daemon=True,
        )
        self._output_thread.start()
        self._compressor = threading.Thread(
            target=self._compress_frames,
            name="swing-frame-compressor",
            daemon=True,
        )
        self._compressor.start()

    def record_frame(self, frame_id: int, frame) -> None:
        """Queue a frame without making resize/JPEG work block the analysis loop."""
        if self._closed:
            raise RuntimeError("Cannot record a frame after the live Swing output manager is closed")

        frame_id = int(frame_id)
        with self._buffer_condition:
            self._latest_recorded_frame = max(self._latest_recorded_frame, frame_id)
        item = (frame_id, frame.copy())
        while True:
            try:
                self._frame_queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.task_done()
                    self._dropped_compression_frames += 1
                except queue.Empty:
                    continue

    def publish_events(self, new_events: List[Dict], snapshot: Dict) -> None:
        """Update JSON/HTML immediately and queue clip encoding in the background."""
        pending_jobs = []
        expected_frames_by_event: Dict[int, List[int]] = {}
        for trace in snapshot.get("frame_trace") or []:
            event_id = trace.get("event_id")
            if event_id is None:
                continue
            expected_frames_by_event.setdefault(int(event_id), []).append(
                int(trace["frame"])
            )
        with self._output_lock:
            for event in new_events:
                published = deepcopy(event)
                stamp_swing_event(
                    published,
                    session=self.session_metadata,
                    emitted_at_unix_ns=(
                        ((published.get("timing") or {}).get(
                            "event_emitted_at_unix_ns"
                        ))
                        or time.time_ns()
                    ),
                )
                output_queued_ns = time.time_ns()
                timing = dict(published.get("timing") or {})
                timing["output_queued_at"] = utc_iso_from_ns(output_queued_ns)
                timing["output_queued_at_unix_ns"] = output_queued_ns
                coach_generated_ns = timing.get("coach_generated_at_unix_ns")
                if isinstance(coach_generated_ns, int) and coach_generated_ns > 0:
                    timing["coach_to_output_queue_ms"] = round(
                        max(0, output_queued_ns - coach_generated_ns) / 1_000_000,
                        3,
                    )
                published["timing"] = timing
                clip_path = self.clips_dir / self._clip_filename(published)
                published["clip_path"] = os.path.relpath(
                    clip_path,
                    self.output_json.parent,
                ).replace(os.sep, "/")
                published["clip_status"] = "pending"
                published["clip_frame_count"] = 0
                published["clip_missing_frame_count"] = 0
                published["clip_missing_frames"] = []
                self._events.append(published)
                self._event_journal.append(
                    "event_created",
                    int(published["event_id"]),
                    {"event": published},
                )
                with self._buffer_condition:
                    target_frame = min(
                        int(published["end_frame"]) + self.clip_padding_frames,
                        self._latest_recorded_frame,
                    )
                pending_jobs.append(
                    (
                        clip_path,
                        deepcopy(published),
                        target_frame,
                        expected_frames_by_event.get(int(published["event_id"]), []),
                    )
                )

            self._document = deepcopy(snapshot)
            self._document["events"] = deepcopy(self._events)
            summary = self._document.setdefault("summary", {})
            summary["swing_event_count"] = len(self._events)
            summary["realtime"] = True
            summary["roi"] = deepcopy(self.roi_metadata)
            summary["clip_buffer_dropped_frames"] = self._dropped_compression_frames
            self._queue_live_outputs(self._document)

        for clip_path, event, target_frame, expected_frame_ids in pending_jobs:
            future = self._executor.submit(
                self._encode_event_clip,
                clip_path,
                event,
                target_frame,
                expected_frame_ids,
            )
            future.add_done_callback(
                lambda completed, event_id=int(event["event_id"]): self._finish_clip(
                    event_id,
                    completed,
                )
            )
            self._futures.append(
                future
            )

    def update_event(self, event_id: int, patch: Dict) -> bool:
        """Merge an asynchronous sidecar result into one published event."""
        with self._output_lock:
            for event in self._events:
                if int(event["event_id"]) != int(event_id):
                    continue
                event.update(deepcopy(patch))
                self._event_journal.append(
                    "event_updated",
                    int(event_id),
                    {"patch": patch},
                )
                self._document["events"] = deepcopy(self._events)
                self._queue_live_outputs(self._document)
                return True
        return False

    def close(self) -> None:
        """Drain frame compression and wait for every queued clip status update."""
        if self._closed:
            return
        self._closed = True
        self._frame_queue.join()
        self._frame_queue.put(self._frame_sentinel)
        self._frame_queue.join()
        self._compressor.join()
        self._executor.shutdown(wait=True)
        self._event_journal.close()
        self._output_queue.put(self._output_sentinel)
        self._output_queue.join()
        self._output_thread.join()
        if self._output_worker_error is not None:
            raise RuntimeError("Realtime Swing JSON/HTML writer failed") from self._output_worker_error

    def _compress_frames(self) -> None:
        try:
            while True:
                item = self._frame_queue.get()
                try:
                    if item is self._frame_sentinel:
                        return
                    frame_id, frame = item
                    if (
                        self.preview_path is not None
                        and (
                            self._last_preview_frame is None
                            or frame_id - self._last_preview_frame
                            >= self.preview_interval_frames
                        )
                    ):
                        preview_frame = self._draw_roi_preview(frame, frame_id)
                        if (
                            preview_frame.shape[1] != self.width
                            or preview_frame.shape[0] != self.height
                        ):
                            preview_frame = cv2.resize(
                                preview_frame,
                                (self.width, self.height),
                                interpolation=cv2.INTER_AREA,
                            )
                        preview_ok, preview_encoded = cv2.imencode(
                            ".jpg",
                            preview_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                        )
                        if preview_ok:
                            self._atomic_write_bytes(
                                self.preview_path,
                                preview_encoded.tobytes(),
                            )
                            self._last_preview_frame = frame_id
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(
                            frame,
                            (self.width, self.height),
                            interpolation=cv2.INTER_AREA,
                        )
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    )
                    with self._buffer_condition:
                        if ok:
                            self._frames.append((frame_id, encoded.tobytes()))
                        else:
                            self._compression_errors[frame_id] = "JPEG encoding failed"
                        self._latest_buffered_frame = max(self._latest_buffered_frame, frame_id)
                        self._buffer_condition.notify_all()
                except Exception as exc:
                    with self._buffer_condition:
                        frame_id = int(item[0])
                        self._compression_errors[frame_id] = str(exc)
                        self._latest_buffered_frame = max(self._latest_buffered_frame, frame_id)
                        self._buffer_condition.notify_all()
                finally:
                    self._frame_queue.task_done()
        finally:
            with self._buffer_condition:
                self._compressor_stopped = True
                self._buffer_condition.notify_all()

    def _frames_for_event_locked(self, event: Dict) -> List[Tuple[int, object]]:
        start_frame = int(event["start_frame"]) - self.clip_padding_frames
        end_frame = int(event["end_frame"]) + self.clip_padding_frames
        return [
            (frame_id, frame)
            for frame_id, frame in self._frames
            if start_frame <= frame_id <= end_frame
        ]

    @staticmethod
    def _clip_filename(event: Dict) -> str:
        stroke = re.sub(r"[^a-z0-9]+", "_", str(event.get("stroke_type") or "unknown").lower()).strip("_")
        return (
            f"event_{int(event['event_id']):04d}_{stroke or 'unknown'}_"
            f"f{int(event['start_frame']):06d}-{int(event['end_frame']):06d}.mp4"
        )

    def _encode_event_clip(
        self,
        output_path: Path,
        event: Dict,
        target_frame: int,
        expected_frame_ids: List[int],
    ) -> Dict:
        with self._buffer_condition:
            self._buffer_condition.wait_for(
                lambda: (
                    self._latest_buffered_frame >= target_frame
                    or self._compressor_stopped
                )
            )
            frames = self._frames_for_event_locked(event)
        if not frames:
            raise RuntimeError(f"No buffered frames available for Swing event {event['event_id']}")
        writer = create_video_writer(
            output_path=str(output_path),
            width=self.width,
            height=self.height,
            fps=self.fps,
            backend=self.video_backend,
            bitrate=self.video_bitrate,
        )
        try:
            for frame_id, encoded_frame in frames:
                frame = cv2.imdecode(
                    np.frombuffer(encoded_frame, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is None:
                    raise RuntimeError(
                        f"Unable to decode buffered frame {frame_id} for Swing event {event['event_id']}"
                    )
                self._draw_event_badge(frame, frame_id, event)
                writer.write(frame)
        finally:
            writer.release()
        buffered_frame_ids = {int(frame_id) for frame_id, _ in frames}
        missing_frame_ids = sorted(
            frame_id
            for frame_id in set(expected_frame_ids)
            if frame_id not in buffered_frame_ids
        )
        return {
            "status": "partial" if missing_frame_ids else "ready",
            "frame_count": len(frames),
            "missing_frames": missing_frame_ids,
        }

    def _finish_clip(self, event_id: int, future: Future) -> None:
        try:
            result = future.result()
            missing_frames = list(result.get("missing_frames") or [])
            updates = {
                "clip_status": str(result.get("status") or "ready"),
                "clip_frame_count": int(result.get("frame_count") or 0),
                "clip_missing_frame_count": len(missing_frames),
                "clip_missing_frames": missing_frames,
            }
        except Exception as exc:
            updates = {
                "clip_status": "failed",
                "clip_frame_count": 0,
                "clip_error": str(exc),
            }

        with self._output_lock:
            for event in self._events:
                if int(event["event_id"]) == event_id:
                    event.update(updates)
                    break
            self._event_journal.append(
                "event_updated",
                int(event_id),
                {"patch": updates},
            )
            self._document["events"] = deepcopy(self._events)
            self._document.setdefault("summary", {})[
                "clip_buffer_dropped_frames"
            ] = self._dropped_compression_frames
            self._queue_live_outputs(self._document)

    @staticmethod
    def _draw_event_badge(frame, frame_id: int, event: Dict) -> None:
        label = (
            f"Swing #{event['event_id']} {event.get('stroke_type', 'Unknown')} "
            f"| Frame {frame_id}"
        )
        cv2.rectangle(frame, (16, 16), (min(frame.shape[1] - 16, 620), 58), (0, 0, 0), -1)
        cv2.putText(
            frame,
            label,
            (28, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _write_live_outputs(self, document: Dict) -> None:
        document = deepcopy(document)
        document.setdefault("summary", {})[
            "session_quality"
        ] = build_session_quality_dashboard(document.get("events") or [])
        self._atomic_write(
            self.output_json,
            json.dumps(document, ensure_ascii=False, indent=2),
        )
        self._atomic_write(
            self.output_html,
            self._render_live_html(document),
        )

    def _queue_live_outputs(self, document: Dict) -> None:
        self._output_queue.put(deepcopy(document))

    def _write_outputs_loop(self) -> None:
        while True:
            document = self._output_queue.get()
            try:
                if document is self._output_sentinel:
                    return
                if self._output_worker_error is None:
                    self._write_live_outputs(document)
            except Exception as exc:
                self._output_worker_error = exc
            finally:
                self._output_queue.task_done()

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _atomic_write_bytes(path: Path, content: bytes) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)

    def _draw_roi_preview(self, frame, frame_id: int):
        preview = frame.copy()
        points = self.roi_metadata.get("points") or []
        configured_size = self.roi_metadata.get("frame_size") or [
            frame.shape[1],
            frame.shape[0],
        ]
        if len(points) == 4 and len(configured_size) >= 2:
            scale_x = frame.shape[1] / max(1, int(configured_size[0]))
            scale_y = frame.shape[0] / max(1, int(configured_size[1]))
            polygon = np.array(
                [
                    [
                        int(round(float(point[0]) * scale_x)),
                        int(round(float(point[1]) * scale_y)),
                    ]
                    for point in points
                ],
                dtype=np.int32,
            )
            overlay = preview.copy()
            cv2.fillPoly(overlay, [polygon], (0, 255, 255))
            cv2.addWeighted(overlay, 0.10, preview, 0.90, 0, preview)
            cv2.polylines(preview, [polygon], True, (0, 255, 255), 4, cv2.LINE_AA)
            for index, point in enumerate(polygon):
                location = (int(point[0]), int(point[1]))
                cv2.circle(preview, location, 8, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(
                    preview,
                    f"P{index + 1}",
                    (location[0] + 10, location[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        label = str(self.roi_metadata.get("label") or "Camera")
        source = str(self.roi_metadata.get("source") or "")
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 92), (0, 0, 0), -1)
        cv2.putText(
            preview,
            f"ROI ACTIVE | {label} | Frame {frame_id}",
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            source,
            (24, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        return preview

    @staticmethod
    def _render_live_session_dashboard(document: Dict) -> str:
        dashboard = (document.get("summary") or {}).get("session_quality") or {}
        quality = dashboard.get("quality") or {}
        drift = dashboard.get("drift") or {}

        def number(value, digits=0, suffix=""):
            if value is None:
                return "-"
            return f"{float(value):.{digits}f}{suffix}"

        kpis = [
            (
                "证据质量",
                number(quality.get("evidence_quality_score_100"), 0, "/100"),
            ),
            (
                "可见动作",
                number(quality.get("visible_technique_mean_9"), 1, "/9"),
            ),
            (
                "校准覆盖",
                number((quality.get("calibrated_event_ratio") or 0.0) * 100, 0, "%"),
            ),
            (
                "触球证据",
                number((quality.get("contact_supported_ratio") or 0.0) * 100, 0, "%"),
            ),
            (
                "复核比例",
                number((quality.get("review_recommended_ratio") or 0.0) * 100, 0, "%"),
            ),
        ]
        kpi_html = "".join(
            f'<div><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
            for label, value in kpis
        )
        trend_rows = []
        for point in dashboard.get("series") or []:
            score = point.get("visible_score_9")
            evidence = point.get("evidence_quality_100")
            score_width = max(0.0, min(100.0, float(score or 0.0) / 9.0 * 100.0))
            evidence_width = max(0.0, min(100.0, float(evidence or 0.0)))
            trend_rows.append(
                '<div class="live-trend-row">'
                f'<b>#{int(point.get("event_id") or 0)}</b>'
                '<div class="live-trend-bars">'
                f'<i class="technique" style="width:{score_width:.1f}%"></i>'
                f'<i class="evidence" style="width:{evidence_width:.1f}%"></i>'
                '</div>'
                f'<span>{number(score, 1, "/9")}</span>'
                '</div>'
            )
        alerts = dashboard.get("alerts") or []
        alert_html = (
            '<ul>'
            + "".join(
                f'<li>{html.escape(str(alert.get("message") or alert.get("code")))}</li>'
                for alert in alerts
            )
            + '</ul>'
            if alerts
            else '<p>当前没有会话级报警。</p>'
        )
        state_labels = {
            "warming_up": "预热中",
            "stable": "稳定",
            "improving": "改善",
            "attention": "需关注",
            "integrity_blocked": "事件重叠",
        }
        state = str(drift.get("status") or "warming_up")
        if state == "integrity_blocked":
            note = "相邻事件范围重叠，暂停漂移结论"
        elif drift.get("ready"):
            note = (
                f'前 {int(drift.get("window_size") or 1)} 次与最近 '
                f'{int(drift.get("window_size") or 1)} 次对比'
            )
        else:
            note = (
                f'至少 {int(drift.get("minimum_event_count") or 6)} 次挥拍后判断漂移'
            )
        return f"""
        <section class="session-monitor">
          <div class="session-monitor-head"><div><h2>会话质量与漂移</h2><p>{html.escape(note)}</p></div><strong data-state="{html.escape(state)}">{html.escape(state_labels.get(state, state))}</strong></div>
          <div class="session-monitor-kpis">{kpi_html}</div>
          <div class="live-trends">{''.join(trend_rows) or '<p>等待挥拍事件…</p>'}</div>
          <div class="session-monitor-alerts">{alert_html}</div>
        </section>
        """

    def _render_live_html(self, document: Dict) -> str:
        summary = document.get("summary") or {}
        session_dashboard = self._render_live_session_dashboard(document)
        roi = summary.get("roi") or self.roi_metadata
        stream_content = ""
        if self.preview_path is not None and roi.get("enabled"):
            preview_href = os.path.relpath(
                self.preview_path,
                self.output_html.parent,
            ).replace(os.sep, "/")
            stream_content = f"""
            <section class="stream-card">
              <div class="stream-heading">
                <div><h2>{html.escape(str(roi.get("label") or "Live camera"))}</h2>
                <p>{html.escape(str(roi.get("source") or ""))}</p></div>
                <strong>ROI ACTIVE</strong>
              </div>
              <img id="roi-preview" src="{html.escape(preview_href)}" alt="Live stream ROI preview">
              <p class="stream-note">实时截图 · 黄色区域为推理 ROI · P1–P4 为配置点</p>
            </section>
            """
        cards = []
        for event in reversed(document.get("events") or []):
            clip_status = str(event.get("clip_status") or "pending")
            if clip_status in {"ready", "partial"}:
                clip_source = self.output_json.parent / str(event.get("clip_path") or "")
                clip_href = os.path.relpath(clip_source, self.output_html.parent).replace(os.sep, "/")
                clip_content = (
                    f'<video controls preload="metadata" '
                    f'src="{html.escape(clip_href)}"></video>'
                )
                if clip_status == "partial":
                    clip_content += (
                        '<p class="clip-state clip-partial">片段缺少 '
                        f'{int(event.get("clip_missing_frame_count") or 0)} 个分析帧</p>'
                    )
            elif clip_status == "failed":
                clip_content = (
                    '<p class="clip-state clip-failed">Clip encoding failed: '
                    f'{html.escape(str(event.get("clip_error") or "unknown error"))}</p>'
                )
            else:
                clip_content = (
                    '<p class="clip-state">Encoding this Swing clip in the background…</p>'
                )
            coach_advices = event.get("coach_advices") or []
            if not coach_advices and event.get("coach_advice"):
                coach_advices = [event["coach_advice"]]
            coach_content = ""
            if coach_advices:
                advice_rows = []
                for index, advice in enumerate(coach_advices[:3], start=1):
                    if not advice.get("message"):
                        continue
                    confidence = max(
                        0.0,
                        min(1.0, float(advice.get("confidence") or 0.0)),
                    )
                    advice_rows.append(
                        '<li>'
                        f'<span>{index}</span>'
                        f'<strong>{html.escape(str(advice["message"]))}</strong>'
                        f'<small>{confidence:.0%}</small>'
                        '</li>'
                    )
                if advice_rows:
                    coach_content = (
                        '<div class="coach-advice"><div class="coach-title">'
                        '<span>实时动作纠错</span><small>单摄像头2D估计</small></div>'
                        f'<ol>{"".join(advice_rows)}</ol></div>'
                    )
            metric_labels = {
                "shoulder_turn_change": "转肩变化",
                "preparation_knee_flexion": "准备屈膝",
                "arm_extension": "挥拍舒展",
                "contact_lateral_distance": "击球点距离",
            }
            metric_rows = []
            for key, label in metric_labels.items():
                metric = (
                    ((event.get("biomechanics") or {}).get("metrics") or {}).get(key)
                    or {}
                )
                if (
                    metric.get("value") is None
                    or metric.get("coach_eligible") is False
                ):
                    continue
                raw_unit = str(metric.get("unit") or "")
                if "deg" in raw_unit:
                    unit = "°"
                elif raw_unit == "body_width":
                    unit = "×身宽"
                else:
                    unit = raw_unit
                metric_rows.append(
                    '<div class="bio-metric">'
                    f'<span>{html.escape(label)}</span>'
                    f'<strong>{float(metric["value"]):.2f}{unit}</strong>'
                    f'<small>{float(metric.get("confidence") or 0.0):.0%}</small>'
                    '</div>'
                )
            biomechanics_content = (
                f'<div class="biomechanics">{"".join(metric_rows)}</div>'
                if metric_rows
                else ""
            )
            deepseek_advice = event.get("deepseek_advice") or {}
            deepseek_status = str(deepseek_advice.get("status") or "")
            deepseek_content = ""
            if deepseek_status == "ready" and deepseek_advice.get("message"):
                deepseek_content = (
                    '<div class="deepseek-advice"><span>DeepSeek旁路</span>'
                    f'<strong>{html.escape(str(deepseek_advice["message"]))}</strong>'
                    f'<small>{int(deepseek_advice.get("latency_ms") or 0)} ms</small></div>'
                )
            elif deepseek_status == "pending":
                deepseek_content = (
                    '<div class="deepseek-advice pending"><span>DeepSeek旁路</span>'
                    '<strong>分析中…</strong></div>'
                )
            elif deepseek_status in {"failed", "unavailable"}:
                deepseek_content = (
                    '<div class="deepseek-advice unavailable"><span>DeepSeek旁路</span>'
                    '<strong>本地建议已生效</strong></div>'
                )
            warnings = ", ".join((event.get("quality_flags") or {}).get("warnings") or []) or "none"
            start_boundary = (event.get("evidence") or {}).get("start_boundary") or {}
            classification_context = (event.get("evidence") or {}).get("classification_context") or {}
            player_context = classification_context.get("player") or {}
            camera_context = classification_context.get("camera") or {}
            swing_context = classification_context.get("swing") or {}
            hand_text = {"right": "右手", "left": "左手"}.get(
                player_context.get("dominant_hand"),
                "未知",
            )
            camera_text = {
                "facing_player": "球员面向相机",
                "behind_player": "相机位于球员后方",
                "side_or_uncertain": "侧向/不确定",
                "unknown": "未知",
            }.get(camera_context.get("view"), "未知")
            swing_side_text = {
                "forehand": "正手侧",
                "backhand": "反手侧",
                "uncertain": "不确定",
                "unknown": "未知",
            }.get(swing_context.get("side"), "未知")
            coach_calibration = event.get("coach_calibration") or {}
            visible_score = coach_calibration.get("visible_technique_score_9")
            visible_uncertainty = coach_calibration.get("uncertainty_9")
            calibration_text = (
                f"{float(visible_score):.1f}/9"
                + (
                    f" ±{float(visible_uncertainty):.1f}"
                    if visible_uncertainty is not None
                    else ""
                )
                if visible_score is not None
                else "证据不足"
            )
            boundary_text = " · ".join(
                str(value)
                for value in (
                    start_boundary.get("mode"),
                    start_boundary.get("confidence"),
                )
                if value
            ) or "legacy"
            cards.append(
                f"""
                <article class="event-card" data-annotation-card data-annotation-id="model-{int(event['event_id'])}" data-source-event-id="{int(event['event_id'])}" data-predicted-stroke-type="{html.escape(str(event.get('stroke_type') or 'Unknown'))}" data-peak-frame="{html.escape(str('' if event.get('peak_frame') is None else event.get('peak_frame')))}">
                  <div class="event-heading">
                    <h2>Swing #{int(event['event_id'])} · {html.escape(str(event.get('stroke_type') or 'Unknown'))}</h2>
                    <span>{float(event.get('confidence') or 0.0):.1%}</span>
                  </div>
                  {clip_content}
                  {coach_content}
                  {biomechanics_content}
                  {deepseek_content}
                  <dl>
                    <div><dt>Frames</dt><dd>{int(event['start_frame'])}–{int(event['end_frame'])}</dd></div>
                    <div><dt>Contact</dt><dd>{html.escape(str(event.get('contact_frame', '-')))}</dd></div>
                    <div><dt>Peak</dt><dd>{html.escape(str(event.get('peak_frame', '-')))}</dd></div>
                    <div><dt>Start boundary</dt><dd>{html.escape(boundary_text)}</dd></div>
                    <div><dt>球员 / 机位 / 挥拍侧</dt><dd>{html.escape(hand_text)} · {html.escape(camera_text)} · {html.escape(swing_side_text)}</dd></div>
                    <div><dt>可见动作校准</dt><dd>{html.escape(calibration_text)}</dd></div>
                    <div><dt>Warnings</dt><dd>{html.escape(warnings)}</dd></div>
                  </dl>
                  <div class="annotation-box">
                    <div class="annotation-frames">
                      <label>人工开始帧<input type="number" min="0" step="1" data-field="start_frame" value="{int(event['start_frame'])}"></label>
                      <label>人工触球帧<input type="number" min="0" step="1" data-field="contact_frame" value="{html.escape(str('' if event.get('contact_frame') is None else event.get('contact_frame')))}"></label>
                      <label>人工结束帧<input type="number" min="0" step="1" data-field="end_frame" value="{int(event['end_frame'])}"></label>
                    </div>
                    <label>人工类型
                      <select data-field="actual_stroke_type">
                        <option value="Forehand"{' selected' if event.get('stroke_type') == 'Forehand' else ''}>Forehand</option>
                        <option value="Backhand"{' selected' if event.get('stroke_type') == 'Backhand' else ''}>Backhand</option>
                        <option value="Two-Handed Backhand"{' selected' if event.get('stroke_type') == 'Two-Handed Backhand' else ''}>Two-Handed Backhand</option>
                        <option value="Serve"{' selected' if event.get('stroke_type') == 'Serve' else ''}>Serve</option>
                        <option value="Volley"{' selected' if event.get('stroke_type') == 'Volley' else ''}>Volley</option>
                        <option value="Unclear"{' selected' if event.get('stroke_type') not in {'Forehand', 'Backhand', 'Two-Handed Backhand', 'Serve', 'Volley'} else ''}>Unclear</option>
                      </select>
                    </label>
                    <div class="annotation-checks">
                      <label><input type="checkbox" data-field="valid_hit" checked> 有效击球</label>
                      <label><input type="checkbox" data-field="count_correct" checked> 计数正确</label>
                      <label><input type="checkbox" data-field="needs_review" checked> 待人工确认（确认后取消）</label>
                      <label><input type="checkbox" data-tag="wrong_type"> 类型错误</label>
                      <label><input type="checkbox" data-tag="contact_timing"> 触球帧偏差</label>
                      <label><input type="checkbox" data-tag="event_boundary"> 边界偏差</label>
                    </div>
                    <label>备注<textarea rows="2" data-field="note"></textarea></label>
                  </div>
                </article>
                """
            )
        content = "\n".join(cards) or '<p class="waiting">Waiting for the first completed Swing event…</p>'
        event_json_href = os.path.relpath(
            self.output_json,
            self.output_html.parent,
        ).replace(os.sep, "/")
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Live Swing Events</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0a0f1a; --panel:#151d2b; --line:#2b3a51; --accent:#f4c95d; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:#edf3fb; font:15px/1.5 system-ui,sans-serif; }}
    header,main {{ width:min(1180px,calc(100% - 32px)); margin:auto; }}
    header {{ padding:28px 0 18px; display:flex; justify-content:space-between; gap:20px; align-items:end; }}
    h1,h2,p {{ margin:0; }}
    .summary {{ color:#aab8cc; }}
    .header-tools {{ display:flex; align-items:center; justify-content:flex-end; flex-wrap:wrap; gap:10px; }}
    #report-summary {{ white-space:nowrap; }}
    .refresh-controls {{ display:flex; align-items:center; flex-wrap:wrap; gap:8px; }}
    .refresh-status {{ border:1px solid #46643c; border-radius:999px; background:#182317; color:#baf1c8; padding:5px 10px; white-space:nowrap; }}
    .refresh-status[data-state="paused"] {{ border-color:#725f2d; background:#1e1b13; color:#f4d995; }}
    .refresh-button {{ border:1px solid var(--line); border-radius:8px; background:#17253a; color:#dceaff; padding:7px 11px; font:inherit; font-weight:700; cursor:pointer; }}
    .refresh-button:hover:not(:disabled),.refresh-button:focus-visible {{ border-color:#6f8fb6; outline:none; }}
    .refresh-button-stop {{ background:#2a2112; border-color:#725f2d; color:#f4d995; }}
    .refresh-button:disabled {{ opacity:.42; cursor:not-allowed; }}
    main {{ display:grid; gap:18px; padding-bottom:40px; }}
    .session-monitor {{ border:1px solid #365d8c; border-radius:14px; background:#111d2d; padding:16px; }}
    .session-monitor-head {{ display:flex; justify-content:space-between; gap:16px; align-items:start; }}
    .session-monitor-head p {{ margin-top:4px; color:#93a4bb; }}
    .session-monitor-head > strong {{ border-radius:999px; background:#27364c; color:#b8c8dc; padding:5px 10px; }}
    .session-monitor-head > strong[data-state="stable"],.session-monitor-head > strong[data-state="improving"] {{ background:#23432f; color:#baf1c8; }}
    .session-monitor-head > strong[data-state="attention"],.session-monitor-head > strong[data-state="integrity_blocked"] {{ background:#512c2c; color:#ffb3b3; }}
    .session-monitor-kpis {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; margin-top:14px; }}
    .session-monitor-kpis div {{ display:grid; gap:2px; border:1px solid var(--line); border-radius:8px; padding:9px 11px; }}
    .session-monitor-kpis span {{ color:#93a4bb; font-size:12px; }}
    .session-monitor-kpis strong {{ font-size:20px; }}
    .live-trends {{ display:grid; gap:6px; margin-top:14px; }}
    .live-trend-row {{ display:grid; grid-template-columns:32px minmax(100px,1fr) 58px; gap:8px; align-items:center; font-size:12px; }}
    .live-trend-bars {{ position:relative; height:17px; border-radius:5px; background:#243247; overflow:hidden; }}
    .live-trend-bars i {{ position:absolute; left:0; height:8px; }}
    .live-trend-bars .technique {{ top:0; background:#9b86ff; }}
    .live-trend-bars .evidence {{ bottom:0; background:#44c4a1; }}
    .session-monitor-alerts {{ margin-top:12px; color:#ffbf91; }}
    .session-monitor-alerts ul {{ margin:0; padding-left:20px; }}
    .stream-card {{ border:1px solid var(--line); border-radius:14px; background:var(--panel); padding:16px; }}
    .stream-heading {{ display:flex; justify-content:space-between; gap:20px; align-items:center; margin-bottom:12px; }}
    .stream-heading p,.stream-note {{ color:#93a4bb; overflow-wrap:anywhere; }}
    .stream-heading strong {{ color:#0a0f1a; background:var(--accent); border-radius:999px; padding:5px 10px; white-space:nowrap; }}
    #roi-preview {{ display:block; width:100%; max-height:680px; object-fit:contain; background:#000; border-radius:9px; }}
    .stream-note {{ margin-top:10px; }}
    .live-coach-feed {{ border:1px solid #46643c; border-radius:14px; background:#182317; padding:16px; }}
    .live-coach-feed h2 {{ color:#dff7d4; }}
    .live-coach-feed p {{ margin-top:6px; color:#a9c99e; }}
    .live-coach-feed ol {{ margin:12px 0 0; padding-left:22px; display:grid; gap:6px; }}
    .live-coach-feed strong {{ color:#e8ffd8; font-size:18px; }}
    .event-card {{ border:1px solid var(--line); border-radius:14px; background:var(--panel); padding:16px; }}
    .event-heading {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }}
    .event-heading span {{ color:var(--accent); font-weight:700; }}
    video {{ display:block; width:100%; max-height:620px; background:#000; border-radius:9px; }}
    .clip-state {{ padding:48px 20px; text-align:center; color:#93a4bb; border:1px dashed var(--line); border-radius:9px; }}
    .clip-failed {{ color:#ff9d9d; }}
    .clip-partial {{ margin-top:8px; padding:10px; color:#ffd28a; }}
    .coach-advice {{ margin-top:14px; padding:14px 16px; border-radius:9px; background:#202b1d; border:1px solid #46643c; }}
    .coach-title {{ display:flex; justify-content:space-between; color:#a9c99e; }}
    .coach-title small {{ color:#78906f; }}
    .coach-advice ol {{ list-style:none; display:grid; gap:8px; margin:10px 0 0; padding:0; }}
    .coach-advice li {{ display:grid; grid-template-columns:28px 1fr auto; align-items:center; gap:10px; }}
    .coach-advice li span {{ display:grid; place-items:center; width:24px; height:24px; border-radius:50%; background:#46643c; color:#fff; }}
    .coach-advice li strong {{ color:#e8ffd8; font-size:18px; }}
    .coach-advice li small {{ color:#b9dcae; font-variant-numeric:tabular-nums; }}
    .biomechanics {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; margin-top:10px; }}
    .bio-metric {{ display:grid; grid-template-columns:1fr auto; gap:2px 8px; padding:9px 11px; border:1px solid var(--line); border-radius:8px; }}
    .bio-metric span {{ color:#93a4bb; font-size:12px; }} .bio-metric strong {{ grid-column:1; }} .bio-metric small {{ grid-column:2; grid-row:1/3; align-self:center; color:#7890ad; }}
    .deepseek-advice {{ display:flex; justify-content:space-between; align-items:center; gap:16px; margin-top:10px; padding:12px 16px; border-radius:9px; background:#17253a; border:1px solid #365d8c; }}
    .deepseek-advice span,.deepseek-advice small {{ color:#9bbce2; }} .deepseek-advice strong {{ color:#e3f1ff; font-size:18px; }}
    .deepseek-advice.pending,.deepseek-advice.unavailable {{ opacity:.72; }}
    .annotation-workspace {{ border:1px solid #725f2d; border-radius:14px; background:#1e1b13; padding:16px; }}
    .annotation-workspace h2 {{ color:#ffe39a; }}
    .annotation-actions {{ display:flex; align-items:center; flex-wrap:wrap; gap:10px; margin-top:12px; }}
    .annotation-actions button {{ border:1px solid var(--accent); border-radius:8px; background:var(--accent); color:#17130a; padding:8px 12px; font-weight:700; cursor:pointer; }}
    .timeline-review {{ display:block; margin-top:10px; color:#d8c99f; }}
    .annotation-readiness {{ margin-top:8px; color:#ffb68c; }}
    .annotation-readiness[data-state="ready"] {{ color:#a9e99a; }}
    .annotation-box {{ display:grid; gap:9px; margin-top:14px; padding-top:14px; border-top:1px solid var(--line); }}
    .annotation-box label {{ color:#c6d2e2; font-size:13px; }}
    .annotation-box input[type="number"],.annotation-box select,.annotation-box textarea {{ width:100%; margin-top:4px; border:1px solid var(--line); border-radius:7px; background:#0e1521; color:#edf3fb; padding:8px; }}
    .annotation-frames {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; }}
    .annotation-checks {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:7px; }}
    .manual-events {{ display:grid; gap:10px; margin-top:12px; }}
    .manual-event-card {{ border:1px solid #927438; border-radius:10px; background:#272115; padding:12px; }}
    .manual-event-heading {{ display:flex; justify-content:space-between; align-items:center; gap:10px; }}
    .manual-event-heading button {{ border:1px solid #d98c8c; border-radius:7px; background:transparent; color:#ffb1b1; padding:5px 8px; cursor:pointer; }}
    .review-workflow {{ border:1px solid #8a6d2f; border-radius:14px; background:#17170f; padding:16px; }}
    .review-workflow-head {{ display:flex; justify-content:space-between; align-items:start; gap:18px; }}
    .review-workflow-head h2 {{ color:#f4d995; }}
    .review-workflow-head p {{ margin-top:5px; color:#b9ad8d; }}
    .review-workflow-state {{ border:1px solid #5d5134; border-radius:999px; padding:5px 10px; color:#d8c99f; white-space:nowrap; }}
    .review-workflow-state[data-state="finalized"] {{ border-color:#507a49; color:#bce7b2; background:#1b2a18; }}
    .review-workflow-state[data-state="error"],.review-workflow-state[data-state="needs_review"] {{ border-color:#9a5b43; color:#ffc0a8; background:#2b1913; }}
    .workflow-track {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:7px; margin-top:14px; counter-reset:stage; }}
    .workflow-track span {{ counter-increment:stage; border-top:3px solid #4a4534; padding:7px 2px 0; color:#887f68; font-size:12px; }}
    .workflow-track span::before {{ content:counter(stage) '. '; }}
    .workflow-track span[data-active="true"] {{ border-color:#f4c76b; color:#f4d995; }}
    .review-actions {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-top:15px; }}
    .review-file {{ position:relative; overflow:hidden; display:inline-flex; border:1px solid #8a6d2f; border-radius:8px; padding:8px 12px; color:#f4d995; cursor:pointer; }}
    .review-file input {{ position:absolute; inset:0; opacity:0; cursor:pointer; }}
    .review-actions button {{ border:1px solid #f4c76b; border-radius:8px; background:#f4c76b; color:#17130a; padding:8px 12px; font-weight:700; cursor:pointer; }}
    .review-actions button.secondary {{ background:transparent; color:#f4d995; }}
    .review-actions button:disabled {{ opacity:.45; cursor:not-allowed; }}
    .review-message {{ margin-top:10px; color:#d5c9a9; }}
    .review-message[data-state="error"] {{ color:#f6a487; }}
    .review-metrics {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; margin-top:14px; }}
    .review-metrics div {{ border:1px solid #4a4534; border-radius:8px; padding:9px 11px; }}
    .review-metrics span {{ display:block; color:#9e957d; font-size:12px; }}
    .review-metrics strong {{ font-size:20px; }}
    .coach-comparisons {{ display:grid; gap:10px; margin-top:14px; }}
    .coach-comparison {{ border:1px solid #4a4534; border-radius:10px; padding:12px; }}
    .coach-comparison-head {{ display:flex; justify-content:space-between; gap:12px; color:#d8c99f; }}
    .coach-columns {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px; }}
    .coach-column {{ border:1px solid #3c4d3a; border-radius:8px; background:#172017; padding:10px; }}
    .coach-column.manual {{ border-color:#8a6d2f; background:#241f12; }}
    .coach-column h3 {{ margin:0; color:#a8d5a2; font-size:13px; }}
    .coach-column.manual h3 {{ color:#f4c76b; }}
    .coach-column ol {{ margin:7px 0 0; padding-left:20px; }}
    .coach-column small {{ color:#9e957d; }}
    dl {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:14px 0 0; }}
    dl div {{ border:1px solid var(--line); border-radius:8px; padding:9px 11px; }}
    dt {{ color:#93a4bb; font-size:12px; text-transform:uppercase; }} dd {{ margin:2px 0 0; }}
    .waiting {{ padding:50px; text-align:center; border:1px dashed var(--line); border-radius:14px; color:#93a4bb; }}
    @media(max-width:720px) {{ header {{ align-items:start; flex-direction:column; }} .header-tools {{ justify-content:flex-start; }} .session-monitor-kpis,.review-metrics {{ grid-template-columns:1fr 1fr; }} dl,.biomechanics,.annotation-frames,.annotation-checks,.coach-columns {{ grid-template-columns:1fr; }} .workflow-track {{ grid-template-columns:1fr 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div><h1>Live Swing Events</h1><p class="summary">Coach 建议每 200 ms 增量更新。</p></div>
    <div class="header-tools">
      <strong id="report-summary">{int(summary.get('swing_event_count') or 0)} events · frame {int(summary.get('latest_frame') or -1)}</strong>
      <div class="refresh-controls" role="group" aria-label="页面刷新控制">
        <span id="refresh-status" class="refresh-status" data-state="running" aria-live="polite">自动刷新中</span>
        <button id="refresh-start" class="refresh-button" type="button" aria-pressed="true">自动刷新</button>
        <button id="refresh-stop" class="refresh-button refresh-button-stop" type="button" aria-pressed="false">停止刷新</button>
      </div>
    </div>
  </header>
  <main>
    {session_dashboard}
    <section id="live-coach-feed" class="live-coach-feed" aria-live="polite"><h2>实时 Coach</h2><p>等待第一条建议…</p></section>
    <section id="annotation-workspace" class="annotation-workspace">
      <h2>人工真值标注 V2</h2>
      <p class="summary">可修正事件边界；系统漏检的挥拍请单独补录。标注会保存在当前浏览器，页面刷新后仍保留。</p>
      <label class="timeline-review"><input id="timeline-review-complete" type="checkbox"> 已完整检查整段视频（完成后才计算 Precision / Recall / F1）</label>
      <p id="annotation-readiness" class="annotation-readiness"></p>
      <div class="annotation-actions">
        <button id="add-missed-event" type="button">新增漏检挥拍</button>
        <button id="download-annotations" type="button">下载标注 JSON</button>
        <span id="annotation-status" class="summary"></span>
      </div>
      <div id="manual-events" class="manual-events"></div>
    </section>
    <section id="manual-review-workflow" class="review-workflow">
      <div class="review-workflow-head">
        <div><h2>人工校准闭环</h2><p>导入人工标注，生成评估，并对比实时 Coach 与人工边界重算结果。</p></div>
        <strong id="review-workflow-state" class="review-workflow-state" data-state="waiting">等待标注</strong>
      </div>
      <div class="workflow-track" aria-label="人工校准流程">
        <span id="review-stage-import" data-active="true">导入标注</span>
        <span id="review-stage-validate">完整性校验</span>
        <span id="review-stage-evaluate">生成评估</span>
        <span id="review-stage-coach">重算 Coach</span>
      </div>
      <div class="review-actions">
        <label class="review-file">选择人工标注 JSON<input id="manual-review-file" type="file" accept="application/json,.json"></label>
        <button id="evaluate-imported-review" type="button" disabled>评估导入标注</button>
        <button id="evaluate-current-review" class="secondary" type="button">评估页面当前标注</button>
      </div>
      <p id="manual-review-message" class="review-message">使用本地 workflow 服务打开本页后，可生成持久化评估与人工 Coach。</p>
      <div id="manual-review-metrics" class="review-metrics" hidden></div>
      <div id="coach-comparisons" class="coach-comparisons"></div>
    </section>
    {stream_content}{content}
  </main>
  <script>
    const eventJsonUrl = {json.dumps(event_json_href)};
    const coachFeed = document.getElementById('live-coach-feed');
    const reportSummary = document.getElementById('report-summary');
    const refreshStatus = document.getElementById('refresh-status');
    const refreshStart = document.getElementById('refresh-start');
    const refreshStop = document.getElementById('refresh-stop');
    const annotationWorkspace = document.getElementById('annotation-workspace');
    const manualEvents = document.getElementById('manual-events');
    const timelineReviewComplete = document.getElementById('timeline-review-complete');
    const annotationStatus = document.getElementById('annotation-status');
    const annotationReadiness = document.getElementById('annotation-readiness');
    const manualReviewWorkspace = document.getElementById('manual-review-workflow');
    const manualReviewFile = document.getElementById('manual-review-file');
    const manualReviewMessage = document.getElementById('manual-review-message');
    const manualReviewState = document.getElementById('review-workflow-state');
    const manualReviewMetrics = document.getElementById('manual-review-metrics');
    const coachComparisons = document.getElementById('coach-comparisons');
    const evaluateImportedReview = document.getElementById('evaluate-imported-review');
    const evaluateCurrentReview = document.getElementById('evaluate-current-review');
    const annotationStorageKey = `tennis.swing.annotations.v2:${{location.pathname}}:${{eventJsonUrl}}`;
    const refreshStorageKey = `tennis.swing.auto-refresh.v1:${{location.pathname}}`;
    let autoRefreshEnabled = true;
    let coachFeedPending = false;
    let manualCounter = 0;
    let lastAnnotationInteraction = 0;
    let importedReviewPayload = null;

    try {{
      autoRefreshEnabled = sessionStorage.getItem(refreshStorageKey) !== 'false';
    }} catch (_error) {{
      // Keep auto-refresh enabled when storage is unavailable (for example, restricted file:// pages).
    }}

    function updateRefreshControls() {{
      refreshStatus.dataset.state = autoRefreshEnabled ? 'running' : 'paused';
      refreshStatus.textContent = autoRefreshEnabled ? '自动刷新中' : '页面刷新已暂停';
      refreshStart.disabled = autoRefreshEnabled;
      refreshStop.disabled = !autoRefreshEnabled;
      refreshStart.setAttribute('aria-pressed', String(autoRefreshEnabled));
      refreshStop.setAttribute('aria-pressed', String(!autoRefreshEnabled));
    }}

    function setAutoRefreshEnabled(enabled) {{
      autoRefreshEnabled = Boolean(enabled);
      try {{
        sessionStorage.setItem(refreshStorageKey, String(autoRefreshEnabled));
      }} catch (_error) {{
        // The controls still work for the current document when storage is unavailable.
      }}
      updateRefreshControls();
      if (autoRefreshEnabled) {{
        refreshCoachFeed();
        refreshPreview();
      }}
    }}

    function integerField(card, field) {{
      const input = card.querySelector(`[data-field="${{field}}"]`);
      if (!input || input.value.trim() === '') return null;
      const value = Number(input.value);
      return Number.isFinite(value) ? Math.round(value) : null;
    }}

    function annotationFromCard(card) {{
      const sourceText = card.dataset.sourceEventId || '';
      const sourceNumber = Number(sourceText);
      const sourceEventId = sourceText === '' ? null : (Number.isFinite(sourceNumber) ? sourceNumber : sourceText);
      const peakNumber = Number(card.dataset.peakFrame);
      return {{
        annotation_id: card.dataset.annotationId,
        source_event_id: sourceEventId,
        predicted_stroke_type: card.dataset.predictedStrokeType || null,
        actual_stroke_type: card.querySelector('[data-field="actual_stroke_type"]').value,
        count_correct: card.querySelector('[data-field="count_correct"]').checked,
        valid_hit: card.querySelector('[data-field="valid_hit"]').checked,
        needs_review: card.querySelector('[data-field="needs_review"]').checked,
        issue_tags: [...card.querySelectorAll('[data-tag]:checked')].map(input => input.dataset.tag),
        note: card.querySelector('[data-field="note"]').value.trim(),
        frames: {{
          start: integerField(card, 'start_frame'),
          contact: integerField(card, 'contact_frame'),
          peak: Number.isFinite(peakNumber) ? peakNumber : null,
          end: integerField(card, 'end_frame')
        }}
      }};
    }}

    function collectAnnotations() {{
      return {{
        schema_version: 'swing_manual_annotations_v2',
        timeline_review_complete: timelineReviewComplete.checked,
        source: {{ event_json: eventJsonUrl }},
        events: [...document.querySelectorAll('[data-annotation-card]')].map(annotationFromCard)
      }};
    }}

    function updateAnnotationReadiness(payload) {{
      const pending = payload.events.filter(event => event.needs_review).length;
      if (pending > 0) {{
        annotationReadiness.dataset.state = 'blocked';
        annotationReadiness.textContent = `还有 ${{pending}} 条“需要复核”，评估指标将保持 provisional。`;
      }} else if (!payload.timeline_review_complete) {{
        annotationReadiness.dataset.state = 'blocked';
        annotationReadiness.textContent = '请完整检查整段视频后勾选确认项。';
      }} else {{
        annotationReadiness.dataset.state = 'ready';
        annotationReadiness.textContent = '已满足正式评估条件，可以下载标注 JSON。';
      }}
    }}

    function saveAnnotations() {{
      lastAnnotationInteraction = Date.now();
      const payload = collectAnnotations();
      localStorage.setItem(annotationStorageKey, JSON.stringify(payload));
      updateAnnotationReadiness(payload);
      annotationStatus.textContent = `已在浏览器保存 ${{payload.events.length}} 条`;
      return payload;
    }}

    function setAnnotationField(card, field, value) {{
      const input = card.querySelector(`[data-field="${{field}}"]`);
      if (!input || value === undefined || value === null) return;
      if (input.type === 'checkbox') input.checked = Boolean(value);
      else input.value = String(value);
    }}

    function applyAnnotation(card, imported) {{
      setAnnotationField(card, 'actual_stroke_type', imported.actual_stroke_type);
      setAnnotationField(card, 'count_correct', imported.count_correct);
      setAnnotationField(card, 'valid_hit', imported.valid_hit);
      setAnnotationField(card, 'needs_review', imported.needs_review);
      setAnnotationField(card, 'note', imported.note || '');
      const frames = imported.frames || {{}};
      setAnnotationField(card, 'start_frame', frames.start ?? imported.start_frame);
      setAnnotationField(card, 'contact_frame', frames.contact ?? imported.contact_frame);
      setAnnotationField(card, 'end_frame', frames.end ?? imported.end_frame);
      const tags = new Set(imported.issue_tags || []);
      card.querySelectorAll('[data-tag]').forEach(input => input.checked = tags.has(input.dataset.tag));
    }}

    function addMissedEvent(imported = null, persist = true) {{
      manualCounter += 1;
      const card = document.createElement('article');
      card.className = 'manual-event-card';
      card.dataset.annotationCard = '';
      card.dataset.annotationId = String(imported?.annotation_id ?? imported?.event_id ?? `manual-${{manualCounter}}`);
      card.dataset.sourceEventId = imported?.source_event_id == null ? '' : String(imported.source_event_id);
      card.dataset.predictedStrokeType = '';
      card.dataset.peakFrame = String(imported?.frames?.peak ?? '');
      card.innerHTML = `
        <div class="manual-event-heading"><h2>人工补充挥拍</h2><button type="button">删除</button></div>
        <div class="annotation-box">
          <div class="annotation-frames">
            <label>人工开始帧<input type="number" min="0" step="1" data-field="start_frame"></label>
            <label>人工触球帧<input type="number" min="0" step="1" data-field="contact_frame"></label>
            <label>人工结束帧<input type="number" min="0" step="1" data-field="end_frame"></label>
          </div>
          <label>人工类型<select data-field="actual_stroke_type"><option>Forehand</option><option>Backhand</option><option>Two-Handed Backhand</option><option>Serve</option><option>Volley</option><option selected>Unclear</option></select></label>
          <div class="annotation-checks">
            <label><input type="checkbox" data-field="valid_hit" checked> 有效击球</label>
            <label><input type="checkbox" data-field="count_correct"> 计数正确</label>
            <label><input type="checkbox" data-field="needs_review"> 需要复核</label>
            <label><input type="checkbox" data-tag="missed_event" checked> 漏检事件</label>
            <label><input type="checkbox" data-tag="contact_timing"> 触球帧偏差</label>
            <label><input type="checkbox" data-tag="event_boundary"> 边界偏差</label>
          </div>
          <label>备注<textarea rows="2" data-field="note"></textarea></label>
        </div>`;
      card.querySelector('button').addEventListener('click', () => {{ card.remove(); saveAnnotations(); }});
      if (imported) applyAnnotation(card, imported);
      manualEvents.append(card);
      if (persist) saveAnnotations();
      return card;
    }}

    function restoreAnnotations() {{
      let payload;
      try {{ payload = JSON.parse(localStorage.getItem(annotationStorageKey) || 'null'); }}
      catch (_error) {{ payload = null; }}
      if (!payload || !Array.isArray(payload.events)) return;
      timelineReviewComplete.checked = Boolean(payload.timeline_review_complete);
      for (const imported of payload.events) {{
        const sourceId = imported.source_event_id ?? imported.event_id ?? null;
        const card = sourceId == null ? null : document.querySelector(`[data-annotation-card][data-source-event-id="${{String(sourceId)}}"]`);
        if (card) applyAnnotation(card, imported);
        else addMissedEvent(imported, false);
      }}
      annotationStatus.textContent = `已恢复 ${{payload.events.length}} 条浏览器标注`;
    }}

    function downloadAnnotations() {{
      const payload = saveAnnotations();
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: 'application/json' }});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'swing_manual_annotations_v2.json';
      link.click();
      URL.revokeObjectURL(link.href);
    }}

    annotationWorkspace.addEventListener('input', saveAnnotations);
    annotationWorkspace.addEventListener('change', saveAnnotations);
    document.getElementById('add-missed-event').addEventListener('click', () => addMissedEvent());
    document.getElementById('download-annotations').addEventListener('click', downloadAnnotations);
    restoreAnnotations();
    updateAnnotationReadiness(collectAnnotations());

    function reviewPercent(value) {{
      return value == null ? '待确认' : `${{(Number(value) * 100).toFixed(1)}}%`;
    }}

    function setReviewStage(stage) {{
      const order = ['import', 'validate', 'evaluate', 'coach'];
      const activeIndex = Math.max(0, order.indexOf(stage));
      order.forEach((name, index) => {{
        document.getElementById(`review-stage-${{name}}`).dataset.active = String(index <= activeIndex);
      }});
    }}

    function setReviewStatus(label, state, message) {{
      manualReviewState.textContent = label;
      manualReviewState.dataset.state = state;
      manualReviewMessage.textContent = message || '';
      manualReviewMessage.dataset.state = state === 'error' ? 'error' : '';
    }}

    function coachColumn(title, advices, manual = false) {{
      const column = document.createElement('div');
      column.className = `coach-column${{manual ? ' manual' : ''}}`;
      const heading = document.createElement('h3');
      heading.textContent = title;
      column.append(heading);
      const list = document.createElement('ol');
      for (const advice of advices || []) {{
        const row = document.createElement('li');
        const message = document.createElement('span');
        message.textContent = advice.message || advice.code || '无建议';
        const confidence = document.createElement('small');
        confidence.textContent = ` ${{reviewPercent(advice.confidence)}}`;
        row.append(message, confidence);
        list.append(row);
      }}
      if (!list.children.length) {{
        const row = document.createElement('li');
        row.textContent = '等待正式人工复核';
        list.append(row);
      }}
      column.append(list);
      return column;
    }}

    function renderReviewState(state) {{
      const status = state?.status || 'waiting_for_annotations';
      const validation = state?.validation || null;
      const summary = state?.evaluation?.summary || null;
      if (status === 'finalized') {{
        setReviewStatus('已完成人工校准', 'finalized', '评估已定稿，人工 Coach 已按确认边界重新计算。');
        setReviewStage('coach');
      }} else if (status === 'needs_review') {{
        const pending = validation?.pending_count ?? 0;
        setReviewStatus('仍需人工复核', 'needs_review', `已生成 provisional 评估；还有 ${{pending}} 条需要复核，暂不生成正式人工 Coach。`);
        setReviewStage('evaluate');
      }} else {{
        setReviewStatus('等待标注', 'waiting', '请选择下载的 swing_manual_annotations_v2.json，或直接评估页面当前标注。');
        setReviewStage('import');
      }}

      manualReviewMetrics.replaceChildren();
      if (summary) {{
        const metrics = [
          ['Precision', reviewPercent(summary.precision)],
          ['Recall', reviewPercent(summary.recall)],
          ['F1', reviewPercent(summary.f1)],
          ['类型准确率', reviewPercent(summary.stroke_type_accuracy)],
          ['触球准确率', reviewPercent(summary.contact_accuracy)],
        ];
        for (const [label, value] of metrics) {{
          const item = document.createElement('div');
          const caption = document.createElement('span');
          caption.textContent = label;
          const number = document.createElement('strong');
          number.textContent = value;
          item.append(caption, number);
          manualReviewMetrics.append(item);
        }}
        manualReviewMetrics.hidden = false;
      }} else {{
        manualReviewMetrics.hidden = true;
      }}

      coachComparisons.replaceChildren();
      for (const comparison of state?.comparisons || []) {{
        const card = document.createElement('article');
        card.className = 'coach-comparison';
        const head = document.createElement('div');
        head.className = 'coach-comparison-head';
        const title = document.createElement('strong');
        title.textContent = `Swing #${{comparison.source_event_id ?? comparison.event_id}} · ${{comparison.stroke_type || 'Unknown'}}`;
        const changes = document.createElement('span');
        changes.textContent = comparison.changed_fields?.length
          ? `变化：${{comparison.changed_fields.join('、')}}`
          : '人工边界与 Coach 未变化';
        head.append(title, changes);
        const columns = document.createElement('div');
        columns.className = 'coach-columns';
        columns.append(
          coachColumn('实时 Coach（原始）', comparison.original_coach, false),
          coachColumn('人工校准 Coach', comparison.manual_coach, true),
        );
        card.append(head, columns);
        coachComparisons.append(card);
      }}
    }}

    function applyImportedReviewToEditor(payload) {{
      if (!payload || !Array.isArray(payload.events)) return;
      timelineReviewComplete.checked = Boolean(payload.timeline_review_complete);
      for (const imported of payload.events) {{
        const sourceId = imported.source_event_id ?? imported.event_id ?? null;
        const card = sourceId == null ? null : document.querySelector(`[data-annotation-card][data-source-event-id="${{String(sourceId)}}"]`);
        if (card) applyAnnotation(card, imported);
        else addMissedEvent(imported, false);
      }}
      saveAnnotations();
    }}

    function manualReviewReportHeaders() {{
      const headers = {{}};
      const artifactPrefix = '/artifacts/';
      if (location.pathname.startsWith(artifactPrefix)) {{
        const reportPath = decodeURIComponent(location.pathname.slice(artifactPrefix.length));
        if (reportPath) headers['X-Manual-Review-Report'] = reportPath;
      }}
      return headers;
    }}

    async function manualReviewRequestHeaders() {{
      const headers = manualReviewReportHeaders();
      headers['Content-Type'] = 'application/json';
      try {{
        const response = await fetch('/api/config', {{ cache: 'no-store' }});
        if (!response.ok) return headers;
        const config = await response.json();
        if (config && config.token) headers['X-Control-Token'] = String(config.token);
      }} catch (_error) {{
        // The standalone manual-review server does not require a control token.
      }}
      return headers;
    }}

    async function submitManualReview(payload) {{
      if (location.protocol === 'file:') {{
        setReviewStatus('需要本地服务', 'error', '请运行 manual_review_workflow.py 后从 http://127.0.0.1 打开本页。');
        return;
      }}
      setReviewStatus('处理中', 'waiting', '正在校验标注并重新聚合逐帧证据…');
      setReviewStage('validate');
      evaluateImportedReview.disabled = true;
      evaluateCurrentReview.disabled = true;
      try {{
        const response = await fetch('/api/manual-review/evaluate', {{
          method: 'POST',
          headers: await manualReviewRequestHeaders(),
          body: JSON.stringify(payload),
        }});
        const state = await response.json();
        if (!response.ok) throw new Error(state.message || state.error || '人工校准失败');
        renderReviewState(state);
      }} catch (error) {{
        setReviewStatus('校准失败', 'error', error.message || String(error));
        setReviewStage('validate');
      }} finally {{
        evaluateImportedReview.disabled = !importedReviewPayload;
        evaluateCurrentReview.disabled = false;
      }}
    }}

    manualReviewFile.addEventListener('change', async event => {{
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      try {{
        const payload = JSON.parse(await file.text());
        if (payload.schema_version !== 'swing_manual_annotations_v2' || !Array.isArray(payload.events)) {{
          throw new Error('请选择 swing_manual_annotations_v2 JSON');
        }}
        importedReviewPayload = payload;
        applyImportedReviewToEditor(payload);
        const pending = payload.events.filter(item => item.needs_review).length;
        evaluateImportedReview.disabled = false;
        setReviewStatus(
          pending ? '导入完成，仍需复核' : '导入完成',
          pending ? 'needs_review' : 'waiting',
          `已读取 ${{payload.events.length}} 条标注；${{pending ? `其中 ${{pending}} 条仍需复核。` : '可以生成正式评估。'}}`,
        );
        setReviewStage('import');
      }} catch (error) {{
        importedReviewPayload = null;
        evaluateImportedReview.disabled = true;
        setReviewStatus('导入失败', 'error', error.message || String(error));
      }}
    }});
    evaluateImportedReview.addEventListener('click', () => submitManualReview(importedReviewPayload));
    evaluateCurrentReview.addEventListener('click', () => submitManualReview(collectAnnotations()));

    async function loadManualReviewState() {{
      if (location.protocol === 'file:') {{
        renderReviewState({{ status: 'waiting_for_annotations' }});
        manualReviewMessage.textContent = '当前是 file:// 页面；运行 manual_review_workflow.py 后可生成评估和人工 Coach。';
        return;
      }}
      try {{
        const response = await fetch('/api/manual-review/state', {{
          cache: 'no-store',
          headers: manualReviewReportHeaders(),
        }});
        const state = await response.json();
        if (!response.ok) throw new Error(state.message || state.error || '人工校准服务不可用');
        renderReviewState(state);
      }} catch (error) {{
        setReviewStatus('服务不可用', 'error', error.message || String(error));
      }}
    }}
    loadManualReviewState();

    function renderCoachFeed(documentPayload) {{
      const events = Array.isArray(documentPayload.events) ? documentPayload.events : [];
      const summary = documentPayload.summary || {{}};
      reportSummary.textContent = `${{events.length}} events · frame ${{summary.latest_frame ?? -1}}`;
      coachFeed.replaceChildren();
      const title = document.createElement('h2');
      title.textContent = '实时 Coach';
      coachFeed.append(title);
      if (!events.length) {{
        const waiting = document.createElement('p');
        waiting.textContent = '等待第一条建议…';
        coachFeed.append(waiting);
        return;
      }}
      const latest = events[events.length - 1];
      const meta = document.createElement('p');
      meta.textContent = `Swing #${{latest.event_id}} · ${{latest.stroke_type || 'Unknown'}}`;
      coachFeed.append(meta);
      const advices = Array.isArray(latest.coach_advices) && latest.coach_advices.length
        ? latest.coach_advices
        : (latest.coach_advice ? [latest.coach_advice] : []);
      const list = document.createElement('ol');
      for (const advice of advices.slice(0, 3)) {{
        if (!advice || !advice.message) continue;
        const row = document.createElement('li');
        const message = document.createElement('strong');
        message.textContent = advice.message;
        row.append(message);
        list.append(row);
      }}
      if (list.children.length) coachFeed.append(list);
    }}

    async function refreshCoachFeed() {{
      if (!autoRefreshEnabled || coachFeedPending) return;
      coachFeedPending = true;
      try {{
        const response = await fetch(`${{eventJsonUrl}}?t=${{Date.now()}}`, {{ cache: 'no-store' }});
        if (autoRefreshEnabled && response.ok) renderCoachFeed(await response.json());
      }} catch (_error) {{
        // file:// reports cannot fetch siblings; the slower HTML refresh below remains available.
      }} finally {{
        coachFeedPending = false;
      }}
    }}

    const preview = document.getElementById('roi-preview');
    const previewSource = preview ? preview.getAttribute('src').split('?')[0] : '';
    function refreshPreview() {{
      if (autoRefreshEnabled && preview) preview.src = previewSource + '?t=' + Date.now();
    }}

    refreshStart.addEventListener('click', () => setAutoRefreshEnabled(true));
    refreshStop.addEventListener('click', () => setAutoRefreshEnabled(false));
    updateRefreshControls();
    if (autoRefreshEnabled) {{
      refreshCoachFeed();
      refreshPreview();
    }}
    setInterval(refreshCoachFeed, 200);
    setInterval(refreshPreview, 1000);
    setInterval(() => {{
      if (!autoRefreshEnabled) return;
      const playing = [...document.querySelectorAll('video')].some(video => !video.paused && !video.ended);
      const editingAnnotations = Date.now() - lastAnnotationInteraction < 15000;
      const annotationFocused = annotationWorkspace.contains(document.activeElement);
      const workflowFocused = manualReviewWorkspace.contains(document.activeElement);
      if (!playing && !editingAnnotations && !annotationFocused && !workflowFocused) location.reload();
    }}, 3000);
  </script>
</body>
</html>
"""
