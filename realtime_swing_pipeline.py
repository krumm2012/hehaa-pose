"""Incremental Swing event analysis shared by live and recorded pipelines."""

from __future__ import annotations

import html
from manual_annotation_contract import annotation_contract_script
from evaluation_reference_policy import reference_metric_script
import json
import math
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
from swing_report_builder import (_build_radar_svg, _build_kinematic_sequence_html,
    _build_event_source_timing_html, _impact_freeze_label, _evidence_quality_label,
    _advice_evidence_label, EVIDENCE_QUALITY_NOTE, extract_biomechanical_sub_scores)
from swing_session_quality import build_session_quality_dashboard
from video_writer_backend import create_video_writer
from image_motion_measurements import source_timestamp
from motion_time_contract import REFERENCE_HZ, POLICY_VERSION as MOTION_TIME_POLICY


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
        refractory_frames: Optional[int] = None,
        min_wrist_sweep: float = 0.0,
        min_arm_extension_range: float = 0.0,
        coach=None,
        session_metadata: Optional[Dict] = None,
        execution_mode: str = 'live',
        window_seconds: float = 8.,
    ):
        if execution_mode not in ('live', 'replay'):
            raise ValueError('Unsupported event engine execution mode')
        self.execution_mode = execution_mode
        self.fps = max(1.0, float(fps or 25.0))
        self.analysis_interval_frames = max(1, int(analysis_interval_frames))
        self.settle_frames = max(
            0,
            int(round(REFERENCE_HZ * 0.6)) if settle_frames is None else int(settle_frames),
        )
        self.window_frames = max(
            32,
            int(round(REFERENCE_HZ * 8.0)) if window_frames is None else int(window_frames),
        )
        self.window_seconds = max(0., float(window_seconds))
        self.settle_seconds = self.settle_frames / REFERENCE_HZ
        self.options = {
            "dominant_hand": dominant_hand,
            "min_peak_energy": float(min_peak_energy),
            "active_energy": float(active_energy),
            "min_event_frames": max(1, int(min_event_frames)),
            "max_internal_gap": max(0, int(max_internal_gap)),
            "min_event_gap": max(0, int(min_event_gap)),
            "min_wrist_sweep": float(min_wrist_sweep),
            "min_arm_extension_range": float(min_arm_extension_range),
        }
        if refractory_frames is not None:
            self.refractory_frames = max(0, int(refractory_frames))
        else:
            self.refractory_frames = (
                max(self.options["min_event_gap"], int(round(REFERENCE_HZ * 0.8)))
                if self.options["min_event_gap"] >= 15
                else self.options["min_event_gap"]
            )
        # Keep realtime duplicate suppression aligned with the segmenter's
        # quality-ordered peak NMS.  The segmenter enforces at least 1.6s
        # between event peaks even when min_event_gap is configured lower.
        self.peak_dedup_frames = max(
            self.options["min_event_gap"],
            int(round(REFERENCE_HZ * 1.6)),
        )
        self.peak_dedup_seconds = self.peak_dedup_frames / REFERENCE_HZ
        self.refractory_seconds = self.refractory_frames / REFERENCE_HZ
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
        latest_time, latest_basis = source_timestamp(record)
        if latest_time is not None and latest_basis == 'media_pts':
            while len(self._frames) > 1:
                first_time, first_basis = source_timestamp(self._frames[0])
                if (first_time is None or first_basis != latest_basis
                        or latest_time-first_time <= self.window_seconds+1e-9):
                    break
                self._frames.popleft()
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
            "analysis_build": __import__('analysis_provenance').analysis_build_info(),
            "generated_at_unix_ns": time.time_ns(),
            "summary": {
                "total_frames": self._total_frames,
                "latest_frame": self._latest_frame_id,
                "swing_event_count": len(self._events),
                "swing_event_type_counts": dict(sorted(type_counts.items())),
                "thresholds": dict(self.options),
                "coach_configuration": (
                    self.coach.configuration() if self.coach is not None else None
                ),
                "realtime": self.execution_mode == 'live',
                "execution_mode": self.execution_mode,
                "settle_frames": self.settle_frames,
                "analysis_interval_frames": self.analysis_interval_frames,
                "window_frames": self.window_frames,
                "peak_dedup_frames": self.peak_dedup_frames,
                "refractory_frames": self.refractory_frames,
                "candidate_runtime_time_policy": MOTION_TIME_POLICY,
                "window_seconds": self.window_seconds,
                "window_frame_limit_semantics": 'independent_storage_capacity_not_duration_or_FPS',
                "retained_frame_count": len(self._frames),
                "frame_capacity_reached": len(self._frames) == self.window_frames,
                "settle_seconds": self.settle_seconds,
                "peak_dedup_seconds": self.peak_dedup_seconds,
                "refractory_seconds": self.refractory_seconds,
                "threshold_reference_hz": REFERENCE_HZ,
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
            # timeline after the latest published event plus refractory cooldown.
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
            measurement_frames=list(self._frames) if self._events else None,
            session_metadata=self.session_metadata,
            execution_mode=self.execution_mode,
            **self.options,
        )
        emitted = []
        for candidate in analysis.get("events", []):
            end_frame = int(candidate["end_frame"])
            end_time = self._event_source_time(candidate, 'end_frame')
            latest_time, latest_basis = source_timestamp(self._frames[-1])
            source_wait = (end_time is not None and latest_time is not None
                           and latest_basis == 'media_pts' and latest_time >= end_time)
            wait_pending = (latest_time-end_time+1e-9 < self.settle_seconds if source_wait
                            else self._latest_frame_id-end_frame < self.settle_frames)
            if not force and wait_pending:
                continue
            if self._is_duplicate(candidate):
                continue

            event = deepcopy(candidate)
            event["event_id"] = len(self._events) + 1
            event["emitted_at_frame"] = self._latest_frame_id
            event["latency_frames"] = max(0, self._latest_frame_id - end_frame)
            event['candidate_runtime_timing'] = {
                'policy_version': MOTION_TIME_POLICY,
                'settle_basis': 'media_pts' if source_wait else 'frame_identity_heuristic_unverified',
                'end_to_emission_media_seconds': latest_time-end_time if source_wait else None,
                'forced_at_source_end': bool(force),
                'completion_status': 'source_end_unsettled_candidate' if force and wait_pending else 'settled_candidate',
                'latency_frames_semantics': 'source_frame_id_distance_not_observed_count_or_seconds',
                'accuracy_validated': False}
            if force and wait_pending:
                quality = event.setdefault('quality_flags', {})
                quality['warnings'] = sorted(set(quality.get('warnings') or []) | {'source_end_without_settle_confirmation'})
                quality['review_recommended'] = True
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
                contact_frame_record=contact_record if self.execution_mode == 'live' else None,
            )
            event['timing']['latency_scope'] = 'live_receipt_to_publication' if self.execution_mode == 'live' else 'offline_replay'
            if self.coach is not None:
                if hasattr(self.coach, "advise_all"):
                    coach_advices = self.coach.advise_all(event) or []
                else:
                    adv = self.coach.advise(event)
                    coach_advices = [adv] if adv else []
                coach_advices = [a for a in coach_advices if a]
                event["coach_advices"] = coach_advices
                event["coach_advice"] = coach_advices[0] if coach_advices else None
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
                if self.execution_mode == 'live' and isinstance(contact_capture_ns, int) and contact_capture_ns > 0:
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

    @staticmethod
    def _event_source_time(event, anchor):
        timing = event.get('phase_timing') or {}
        record = (timing.get('anchors') or {}).get(anchor) or {}
        value = record.get('timestamp_seconds')
        if (timing.get('status') == 'reported_media_time' and timing.get('basis') == 'media_pts'
                and type(record.get('source_frame_id')) is int
                and record['source_frame_id'] == event.get(anchor)
                and type(value) in (float,int) and math.isfinite(value) and value >= 0):
            return float(value)
        return None

    def _is_duplicate(self, candidate: Dict) -> bool:
        peak_frame = int(candidate["peak_frame"])
        tolerance = max(1, int(self.peak_dedup_frames))
        peak_time = self._event_source_time(candidate, 'peak_frame')
        events_by_peak = {int(e['peak_frame']):e for e in self._events}
        for emitted_peak in self._emitted_peaks:
            old_peak_time = self._event_source_time(events_by_peak.get(emitted_peak, {}), 'peak_frame')
            duplicate = (abs(peak_time-old_peak_time) <= self.peak_dedup_seconds+1e-9
                         if peak_time is not None and old_peak_time is not None
                         else abs(peak_frame-emitted_peak) <= tolerance)
            if duplicate:
                return True

        # A rolling window can later promote a weaker secondary peak after the
        # original dominant peak leaves the window edge. If that new peak lies
        # inside an already-published Swing range, or within the refractory cooldown
        # after an already-published Swing, it is a follow-through tail or recovery
        # gesture rather than a newly completed Swing.
        candidate_start = int(candidate["start_frame"])
        candidate_end = int(candidate["end_frame"])
        for event in self._events:
            event_start = int(event["start_frame"])
            event_end = int(event["end_frame"])
            if max(candidate_start, event_start) <= min(candidate_end, event_end):
                return True
            if self.refractory_frames > 0:
                start_time = self._event_source_time(candidate, 'start_frame')
                old_end_time = self._event_source_time(event, 'end_frame')
                if start_time is not None and peak_time is not None and old_end_time is not None:
                    in_cooldown = (start_time <= old_end_time+.2+1e-9
                                   and peak_time < old_end_time+self.refractory_seconds)
                else:
                    in_cooldown = candidate_start <= event_end+5 and peak_frame < event_end+self.refractory_frames
                if in_cooldown:
                    return True
        return False

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
        is_dual_view: bool = False,
    ):
        self.output_json = Path(output_json)
        self.output_html = Path(output_html)
        self.clips_dir = Path(clips_dir)
        self.is_dual_view = bool(is_dual_view)
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
        from latest_snapshot_queue import LatestSnapshotQueue
        self._output_queue = LatestSnapshotQueue()
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

    def update_event(self, event_id: int, patch: Dict) -> bool:
        """Merge an asynchronous sidecar result into one published event."""
        with self._output_lock:
            for event in self._events:
                if int(event["event_id"]) != int(event_id):
                    continue
                for key, value in deepcopy(patch).items():
                    if isinstance(event.get(key), dict) and isinstance(value, dict):
                        event[key] = {**event[key], **value}
                    else:
                        event[key] = value
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
        # Buffered realtime overlays may belong to the previous confirmed swing.
        # Replace the card in the background encoder, using this clip's finalized event.
        renderer = None
        if self.is_dual_view:
            from dual_view_renderer import DualViewRenderer
            renderer = DualViewRenderer()
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
                frame = self._annotate_event_frame(frame, frame_id, event, renderer)
                writer.write(frame)
        finally:
            writer.release()

        # Extract and save contact frame freeze snapshot if available
        contact_fid = int(event.get('contact_frame') if event.get('contact_frame') is not None else event.get('start_frame', 0))
        freeze_fid = contact_fid
        target_bytes = None
        for fid, fbytes in frames:
            if fid == contact_fid:
                target_bytes = fbytes
                break
        if target_bytes is None and frames:
            freeze_fid, target_bytes = min(frames, key=lambda x: abs(x[0] - contact_fid))

        freeze_rel = None
        if target_bytes:
            freeze_filename = f"event_{int(event['event_id']):04d}_impact_freeze.jpg"
            freeze_path = output_path.parent / freeze_filename
            try:
                freeze = cv2.imdecode(np.frombuffer(target_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if freeze is None:
                    raise RuntimeError("Unable to decode contact freeze frame")
                freeze = self._annotate_event_frame(freeze, freeze_fid, event, renderer)
                if not cv2.imwrite(str(freeze_path), freeze):
                    raise RuntimeError("Unable to write contact freeze frame")
                freeze_rel = str(freeze_path.relative_to(self.output_json.parent)).replace(os.sep, "/")
            except Exception:
                freeze_rel = None

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
            "impact_freeze_path": freeze_rel,
            "impact_freeze_source_frame_id": freeze_fid if freeze_rel else None,
            "clip_telemetry_event_id": int(event["event_id"]) if renderer is not None else None,
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
                "clip_telemetry_event_id": result.get("clip_telemetry_event_id"),
            }
            if result.get("impact_freeze_path"):
                updates["impact_freeze_path"] = result["impact_freeze_path"]
                updates['impact_freeze_source_frame_id'] = result.get('impact_freeze_source_frame_id')
                updates["snapshots"] = {"impact_freeze": result["impact_freeze_path"]}
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

    def _annotate_event_frame(self, frame, frame_id, event, renderer):
        if renderer is not None:
            from realtime_swing_runtime import build_impact_telemetry_card
            frame = renderer.draw_impact_telemetry_card(
                frame, build_impact_telemetry_card(event), opacity=1.0)
        self._draw_event_badge(frame, frame_id, event)
        return frame

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
        self.check_health()
        self._output_queue.publish(deepcopy(document))

    def check_health(self) -> None:
        if self._output_worker_error is not None:
            raise RuntimeError("Realtime Swing JSON/HTML writer failed") from self._output_worker_error

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
        # 算法 2.0 虚拟双机位画面（Side-by-Side 1080x720）已自带 Front/Back View 标题、
        # 镜面翻转视点、姿态骨架及击球遥测 HUD，直接输出高清实时推流画面，
        # 不应将单机位 2560x1440 全景球场四角点强行错位叠加到局部人像特写上。
        if self.is_dual_view:
            return frame.copy()

        configured_size = self.roi_metadata.get("frame_size") or [
            frame.shape[1],
            frame.shape[0],
        ]
        # 防御性检测：如果画面宽高比与配置的 ROI 全景宽高比差异过大（例如局部双视角裁剪），跳过错位绘制
        aspect_frame = frame.shape[1] / max(1, frame.shape[0])
        aspect_conf = (
            configured_size[0] / max(1, configured_size[1])
            if len(configured_size) >= 2
            else aspect_frame
        )
        if abs(aspect_frame - aspect_conf) > 0.15:
            return frame.copy()

        preview = frame.copy()
        points = self.roi_metadata.get("points") or []
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
                "练习评分",
                number(quality.get("practice_score_mean_100"), 1, "/100"),
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
            score = point.get("practice_score_100")
            evidence = point.get("evidence_quality_100")
            score_width = max(0.0, min(100.0, float(score or 0.0)))
            evidence_width = max(0.0, min(100.0, float(evidence or 0.0)))
            trend_rows.append(
                '<div class="live-trend-row">'
                f'<b>#{int(point.get("event_id") or 0)}</b>'
                '<div class="live-trend-bars">'
                f'<i class="technique" style="width:{score_width:.1f}%"></i>'
                f'<i class="evidence" style="width:{evidence_width:.1f}%"></i>'
                '</div>'
                f'<span>{number(score, 1, "/100")}</span>'
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
        """Render realtime live report to HTML.

        Delegates to report_rendering.render_live_report_html for template-driven rendering.
        """
        from report_rendering import render_live_report_html
        return render_live_report_html(document, self)
