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
                stream = self.path.open("w", encoding="utf-8")
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
        if len(self._frames) < self.options["min_event_frames"]:
            return []

        analysis = analyze_frame_records(
            list(self._frames),
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
            self._events.append(event)
            self._emitted_peaks.append(int(event["peak_frame"]))
            self._append_event_trace(analysis.get("frame_trace", []), candidate, event)
            emitted.append(deepcopy(event))
        return emitted

    def _is_duplicate(self, candidate: Dict) -> bool:
        peak_frame = int(candidate["peak_frame"])
        tolerance = max(1, int(self.options["min_event_gap"]))
        return any(abs(peak_frame - emitted_peak) <= tolerance for emitted_peak in self._emitted_peaks)

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

    def _render_live_html(self, document: Dict) -> str:
        summary = document.get("summary") or {}
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
                "hip_shoulder_separation": "肩髋分离",
                "shoulder_turn": "肩部转动",
                "arm_extension": "手臂伸展",
                "contact_lateral_distance": "击球点距离",
                "weight_transfer": "重心转移",
                "balance_drift": "平衡漂移",
            }
            metric_rows = []
            for key, label in metric_labels.items():
                metric = (
                    ((event.get("biomechanics") or {}).get("metrics") or {}).get(key)
                    or {}
                )
                if metric.get("value") is None:
                    continue
                unit = "°" if metric.get("unit") == "deg" else "×身宽"
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
            cards.append(
                f"""
                <article class="event-card">
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
                    <div><dt>Warnings</dt><dd>{html.escape(warnings)}</dd></div>
                  </dl>
                </article>
                """
            )
        content = "\n".join(cards) or '<p class="waiting">Waiting for the first completed Swing event…</p>'
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
    main {{ display:grid; gap:18px; padding-bottom:40px; }}
    .stream-card {{ border:1px solid var(--line); border-radius:14px; background:var(--panel); padding:16px; }}
    .stream-heading {{ display:flex; justify-content:space-between; gap:20px; align-items:center; margin-bottom:12px; }}
    .stream-heading p,.stream-note {{ color:#93a4bb; overflow-wrap:anywhere; }}
    .stream-heading strong {{ color:#0a0f1a; background:var(--accent); border-radius:999px; padding:5px 10px; white-space:nowrap; }}
    #roi-preview {{ display:block; width:100%; max-height:680px; object-fit:contain; background:#000; border-radius:9px; }}
    .stream-note {{ margin-top:10px; }}
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
    dl {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:14px 0 0; }}
    dl div {{ border:1px solid var(--line); border-radius:8px; padding:9px 11px; }}
    dt {{ color:#93a4bb; font-size:12px; text-transform:uppercase; }} dd {{ margin:2px 0 0; }}
    .waiting {{ padding:50px; text-align:center; border:1px dashed var(--line); border-radius:14px; color:#93a4bb; }}
    @media(max-width:720px) {{ header {{ align-items:start; flex-direction:column; }} dl,.biomechanics {{ grid-template-columns:1fr 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div><h1>Live Swing Events</h1><p class="summary">Auto-refresh pauses while a clip is playing.</p></div>
    <strong>{int(summary.get('swing_event_count') or 0)} events · frame {int(summary.get('latest_frame') or -1)}</strong>
  </header>
  <main>{stream_content}{content}</main>
  <script>
    const preview = document.getElementById('roi-preview');
    if (preview) {{
      const previewSource = preview.getAttribute('src').split('?')[0];
      setInterval(() => {{
        preview.src = previewSource + '?t=' + Date.now();
      }}, 1000);
    }}
    setInterval(() => {{
      const playing = [...document.querySelectorAll('video')].some(video => !video.paused && !video.ended);
      if (!playing) location.reload();
    }}, 3000);
  </script>
</body>
</html>
"""
