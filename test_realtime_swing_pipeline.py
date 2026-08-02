import unittest
import json
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from analysis_data_contracts import (
    EVENT_LOG_SCHEMA_VERSION,
    FRAME_DOCUMENT_SCHEMA_VERSION,
    SWING_EVENT_SCHEMA_VERSION,
)
from local_realtime_coach import LocalRealtimeCoach
from realtime_swing_pipeline import (
    RealtimeEventJournal,
    RealtimeFrameJournal,
    RealtimeSwingEventEngine,
    RealtimeSwingOutputManager,
)


def frame_record(frame_id, wrist_x, label="Forehand"):
    return {
        "frame_id": frame_id,
        "timestamp": frame_id / 25.0,
        "swing_type": label,
        "ball": [wrist_x + 300, 100],
        "rackets": [{"box": [wrist_x, 90, wrist_x + 12, 112], "confidence": 0.9}],
        "pose": {
            "right_wrist": [wrist_x, 100],
            "left_wrist": [wrist_x - 100, 100],
            "right_shoulder": [20, 80],
            "right_elbow": [wrist_x - 10, 92],
            "left_shoulder": [0, 80],
            "left_hip": [0, 140],
            "right_hip": [20, 140],
        },
    }


class RealtimeSwingEventEngineTests(unittest.TestCase):
    def make_engine(self):
        return RealtimeSwingEventEngine(
            fps=25.0,
            analysis_interval_frames=1,
            settle_frames=3,
            window_frames=80,
            min_peak_energy=8.0,
            active_energy=6.0,
            min_event_frames=4,
            max_internal_gap=1,
            min_event_gap=3,
        )

    def test_emits_completed_swing_once_after_settle_window(self):
        engine = self.make_engine()
        positions = [0, 0, 0, 10, 25, 45, 65, 80, 90, 95, 95, 95, 95, 95, 95, 95]
        emitted = []

        for frame_id, wrist_x in enumerate(positions):
            emitted.extend(engine.push_frame(frame_record(frame_id, wrist_x)))

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["event_id"], 1)
        self.assertEqual(emitted[0]["stroke_type"], "Forehand")
        self.assertEqual(
            emitted[0]["schema_version"],
            SWING_EVENT_SCHEMA_VERSION,
        )
        self.assertIn(
            "event_emitted_at_unix_ns",
            emitted[0]["timing"],
        )
        self.assertLess(emitted[0]["start_frame"], emitted[0]["end_frame"])

        for frame_id in range(len(positions), len(positions) + 8):
            self.assertEqual(engine.push_frame(frame_record(frame_id, 95)), [])
        self.assertEqual(engine.snapshot()["summary"]["swing_event_count"], 1)

    def test_flush_emits_last_unsettled_swing(self):
        engine = self.make_engine()
        positions = [0, 0, 10, 25, 45, 65, 80, 90]

        for frame_id, wrist_x in enumerate(positions):
            engine.push_frame(frame_record(frame_id, wrist_x))

        emitted = engine.flush()

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["event_id"], 1)

    def test_suppresses_late_secondary_peak_inside_an_emitted_event(self):
        engine = self.make_engine()
        engine._events.append(
            {
                "event_id": 1,
                "start_frame": 20,
                "end_frame": 72,
                "peak_frame": 41,
            }
        )
        engine._emitted_peaks.append(41)

        self.assertTrue(
            engine._is_duplicate(
                {
                    "start_frame": 48,
                    "end_frame": 95,
                    "peak_frame": 69,
                }
            )
        )

    def test_suppresses_rolling_secondary_peak_from_overlapping_published_event(self):
        """Regression for RTSP event #1 33-98 followed by #2 62-131."""
        engine = RealtimeSwingEventEngine(
            fps=25.0,
            min_event_gap=18,
        )
        engine._events.append(
            {
                "event_id": 1,
                "start_frame": 33,
                "end_frame": 98,
                "peak_frame": 67,
            }
        )
        engine._emitted_peaks.append(67)

        self.assertTrue(
            engine._is_duplicate(
                {
                    "start_frame": 62,
                    "end_frame": 131,
                    "peak_frame": 100,
                }
            )
        )

    def test_keeps_non_overlapping_event_beyond_effective_peak_gap(self):
        engine = RealtimeSwingEventEngine(
            fps=25.0,
            min_event_gap=18,
        )
        engine._events.append(
            {
                "event_id": 1,
                "start_frame": 33,
                "end_frame": 98,
                "peak_frame": 67,
            }
        )
        engine._emitted_peaks.append(67)

        self.assertFalse(
            engine._is_duplicate(
                {
                    "start_frame": 105,
                    "end_frame": 155,
                    "peak_frame": 115,
                }
            )
        )

    def test_returns_original_frame_records_for_emitted_event_range(self):
        engine = self.make_engine()
        positions = [0, 0, 0, 10, 25, 45, 65, 80, 90, 95, 95, 95, 95, 95, 95, 95]
        emitted = []

        for frame_id, wrist_x in enumerate(positions):
            record = frame_record(frame_id, wrist_x)
            record["detection_diagnostics"] = {"final_decision": f"frame_{frame_id}"}
            emitted.extend(engine.push_frame(record))

        self.assertEqual(len(emitted), 1)
        event = emitted[0]
        event_frames = engine.frame_records_for_event(event)

        self.assertEqual(
            [row["frame_id"] for row in event_frames],
            list(range(event["start_frame"], event["end_frame"] + 1)),
        )
        self.assertEqual(
            event_frames[0]["detection_diagnostics"]["final_decision"],
            f"frame_{event['start_frame']}",
        )
        event_frames[0]["pose"]["right_wrist"][0] = -1
        self.assertNotEqual(
            engine.frame_records_for_event(event)[0]["pose"]["right_wrist"][0],
            -1,
        )

    def test_emitted_event_contains_local_coach_guidance_when_enabled(self):
        engine = RealtimeSwingEventEngine(
            fps=25.0,
            analysis_interval_frames=1,
            settle_frames=3,
            window_frames=80,
            min_peak_energy=8.0,
            active_energy=6.0,
            min_event_frames=4,
            max_internal_gap=1,
            min_event_gap=3,
            coach=LocalRealtimeCoach(max_chars=15),
        )
        positions = [0, 0, 0, 10, 25, 45, 65, 80, 90, 95, 95, 95, 95, 95, 95]
        emitted = []

        for frame_id, wrist_x in enumerate(positions):
            emitted.extend(engine.push_frame(frame_record(frame_id, wrist_x)))

        self.assertEqual(len(emitted), 1)
        self.assertIn("coach_advice", emitted[0])
        self.assertIn("coach_advices", emitted[0])
        self.assertIn("biomechanics", emitted[0])
        self.assertGreaterEqual(len(emitted[0]["coach_advices"]), 1)
        self.assertLessEqual(len(emitted[0]["coach_advices"]), 3)
        self.assertEqual(
            emitted[0]["coach_advice"],
            emitted[0]["coach_advices"][0],
        )
        self.assertLessEqual(len(emitted[0]["coach_advice"]["message"]), 15)
        self.assertIn(
            "coach_generated_at_unix_ns",
            emitted[0]["timing"],
        )
        self.assertGreaterEqual(
            emitted[0]["timing"]["coach_generated_at_unix_ns"],
            emitted[0]["timing"]["event_emitted_at_unix_ns"],
        )
        self.assertGreaterEqual(
            emitted[0]["timing"]["event_to_coach_ms"],
            0.0,
        )
        self.assertEqual(
            engine.snapshot()["events"][0]["coach_advice"],
            emitted[0]["coach_advice"],
        )


class RealtimeFrameJournalTests(unittest.TestCase):
    def test_writes_all_processed_frames_and_an_atomic_recent_snapshot(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            journal = RealtimeFrameJournal(
                jsonl_path=str(root / "live_frames.jsonl"),
                snapshot_path=str(root / "live_frames_latest.json"),
                snapshot_size=3,
                flush_interval=2,
            )

            for frame_id in range(7):
                journal.record({"frame_id": frame_id, "timestamp": frame_id / 25.0})
            journal.close()

            jsonl_rows = [
                json.loads(line)
                for line in (root / "live_frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            snapshot = json.loads(
                (root / "live_frames_latest.json").read_text(encoding="utf-8")
            )

        self.assertEqual([row["frame_id"] for row in jsonl_rows], list(range(7)))
        self.assertEqual(
            [row["frame_id"] for row in snapshot["frames"]],
            [4, 5, 6],
        )
        self.assertEqual(snapshot["summary"]["frame_count"], 7)
        self.assertEqual(snapshot["summary"]["dropped_records"], 0)
        self.assertEqual(snapshot["schema_version"], FRAME_DOCUMENT_SCHEMA_VERSION)

    def test_small_queue_still_preserves_every_processed_frame(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            journal = RealtimeFrameJournal(
                jsonl_path=str(root / "frames.jsonl"),
                snapshot_path=str(root / "latest.json"),
                snapshot_size=10,
                flush_interval=10,
                queue_size=1,
            )

            for frame_id in range(200):
                journal.record(
                    {
                        "frame_id": frame_id,
                        "metrics": {"payload": "x" * 2000},
                    }
                )
            journal.close()

            rows = (root / "frames.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(rows), 200)

    def test_writer_error_is_raised_instead_of_hanging_close(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            journal = RealtimeFrameJournal(
                jsonl_path=str(root / "frames.jsonl"),
                snapshot_path=str(root / "latest.json"),
                snapshot_size=3,
                flush_interval=1,
            )
            journal.record({"frame_id": 1, "not_json": {1, 2, 3}})
            errors = []

            def close_journal():
                try:
                    journal.close()
                except Exception as exc:
                    errors.append(exc)

            closer = threading.Thread(target=close_journal, daemon=True)
            closer.start()
            closer.join(timeout=1.0)

        self.assertFalse(closer.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)


class RealtimeEventJournalTests(unittest.TestCase):
    def test_reopening_log_appends_and_continues_sequence(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            for session_id in ("session-one", "session-two"):
                journal = RealtimeEventJournal(
                    str(path),
                    session_metadata={"session_id": session_id},
                )
                journal.append(
                    "event_created",
                    1,
                    {"event": {"event_id": 1}},
                )
                journal.close()

            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([row["sequence"] for row in rows], [1, 2])
        self.assertEqual(
            [row["session_id"] for row in rows],
            ["session-one", "session-two"],
        )


class RealtimeSwingOutputManagerTests(unittest.TestCase):
    def test_frontend_preview_draws_camera_bound_roi_without_credentials(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            preview_path = root / "live_roi_preview.jpg"
            manager = RealtimeSwingOutputManager(
                output_json=str(root / "events.json"),
                output_html=str(root / "report.html"),
                clips_dir=str(root / "clips"),
                fps=25.0,
                frame_size=(320, 180),
                buffer_frames=8,
                clip_workers=1,
                video_backend="opencv",
                preview_path=str(preview_path),
                roi_metadata={
                    "enabled": True,
                    "matched": True,
                    "stream_id": "court01-main",
                    "label": "Court 01",
                    "source": "rtsp://192.168.1.191:554/camera/main",
                    "points": [[20, 30], [300, 30], [300, 160], [20, 160]],
                    "frame_size": [320, 180],
                },
                preview_interval_frames=1,
            )
            manager.record_frame(
                1,
                np.zeros((180, 320, 3), dtype=np.uint8),
            )
            manager.close()

            html = (root / "report.html").read_text(encoding="utf-8")
            payload = json.loads((root / "events.json").read_text(encoding="utf-8"))
            preview = cv2.imread(str(preview_path))

        self.assertIsNotNone(preview)
        self.assertIn("Court 01", html)
        self.assertIn("rtsp://192.168.1.191:554/camera/main", html)
        self.assertNotIn("admin:", html)
        self.assertIn("roi-preview", html)
        self.assertEqual(payload["summary"]["roi"]["stream_id"], "court01-main")
        self.assertGreater(int(preview[100, 20, 1]), 100)

    def test_publishes_json_html_and_async_event_clip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager = RealtimeSwingOutputManager(
                output_json=str(root / "live_swing_events.json"),
                output_html=str(root / "live_swing_report.html"),
                clips_dir=str(root / "clips"),
                fps=25.0,
                frame_size=(64, 48),
                buffer_frames=20,
                clip_workers=1,
                video_backend="opencv",
            )
            for frame_id in range(10):
                manager.record_frame(
                    frame_id,
                    np.full((48, 64, 3), frame_id * 10, dtype=np.uint8),
                )

            event = {
                "event_id": 1,
                "start_frame": 2,
                "end_frame": 6,
                "peak_frame": 4,
                "contact_frame": 5,
                "stroke_type": "Forehand",
                "confidence": 0.88,
                "quality_flags": {"warnings": []},
                "coach_advice": {
                    "code": "short_follow_through",
                    "message": "击球后完成随挥",
                    "category": "technique",
                    "confidence": 0.88,
                    "source": "local_rules_v1",
                    "evidence": {"follow_through_frames": 2},
                },
                "coach_advices": [
                    {
                        "code": "short_follow_through",
                        "message": "击球后完成随挥",
                        "category": "technique",
                        "confidence": 0.88,
                        "source": "local_rules_v1",
                        "evidence": {"follow_through_frames": 2},
                    },
                    {
                        "code": "limited_separation",
                        "message": "加大肩髋分离",
                        "category": "technique",
                        "confidence": 0.76,
                        "source": "local_biomechanics_v2",
                        "focus": "hip_shoulder_separation",
                        "evidence": {"value": 8.0, "unit": "deg"},
                    },
                ],
            }
            snapshot = {
                "summary": {
                    "total_frames": 10,
                    "latest_frame": 9,
                    "swing_event_count": 1,
                    "swing_event_type_counts": {"Forehand": 1},
                },
                "events": [event],
                "frame_trace": [],
            }

            manager.publish_events([event], snapshot)
            updated = manager.update_event(
                1,
                {
                    "deepseek_advice": {
                        "status": "ready",
                        "message": "提前转肩充分引拍",
                        "model": "deepseek-v4-flash",
                        "latency_ms": 420,
                    }
                },
            )
            manager.close()

            payload = json.loads((root / "live_swing_events.json").read_text(encoding="utf-8"))
            event_log = [
                json.loads(line)
                for line in (root / "live_swing_events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            html = (root / "live_swing_report.html").read_text(encoding="utf-8")
            clip_path = root / payload["events"][0]["clip_path"]
            capture = cv2.VideoCapture(str(clip_path))
            clip_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            clip_exists = clip_path.exists()

        self.assertEqual(payload["summary"]["swing_event_count"], 1)
        self.assertEqual(payload["summary"]["session_quality"]["event_count"], 1)
        self.assertEqual(
            payload["events"][0]["schema_version"],
            SWING_EVENT_SCHEMA_VERSION,
        )
        self.assertEqual(
            [row["operation"] for row in event_log],
            ["event_created", "event_updated", "event_updated"],
        )
        self.assertTrue(
            all(row["schema_version"] == EVENT_LOG_SCHEMA_VERSION for row in event_log)
        )
        self.assertEqual(payload["events"][0]["clip_status"], "ready")
        self.assertEqual(payload["events"][0]["clip_frame_count"], 5)
        self.assertTrue(updated)
        self.assertEqual(
            payload["events"][0]["deepseek_advice"]["message"],
            "提前转肩充分引拍",
        )
        self.assertIn("Forehand", html)
        self.assertIn("击球后完成随挥", html)
        self.assertIn("加大肩髋分离", html)
        self.assertIn("88%", html)
        self.assertIn("76%", html)
        self.assertIn("提前转肩充分引拍", html)
        self.assertIn("live-coach-feed", html)
        self.assertIn("会话质量与漂移", html)
        self.assertIn("session-monitor", html)
        self.assertIn("setInterval(refreshCoachFeed, 200)", html)
        self.assertIn("live_swing_events.json", html)
        self.assertIn("<video", html)
        self.assertIn("swing_manual_annotations_v2", html)
        self.assertIn('id="timeline-review-complete"', html)
        self.assertIn('id="annotation-readiness"', html)
        self.assertIn("updateAnnotationReadiness", html)
        self.assertIn('id="add-missed-event"', html)
        self.assertIn('data-field="start_frame"', html)
        self.assertIn('data-field="contact_frame"', html)
        self.assertIn('data-field="end_frame"', html)
        self.assertIn('data-field="needs_review" checked', html)
        self.assertIn("localStorage.setItem", html)
        self.assertIn("addMissedEvent", html)
        self.assertIn('id="manual-review-workflow"', html)
        self.assertIn('id="manual-review-file"', html)
        self.assertIn("/api/manual-review/evaluate", html)
        self.assertIn("实时 Coach（原始）", html)
        self.assertIn("人工校准 Coach", html)
        self.assertTrue(clip_exists)
        self.assertEqual(clip_frames, 5)

    def test_clip_with_missing_analyzed_frames_is_marked_partial(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager = RealtimeSwingOutputManager(
                output_json=str(root / "events.json"),
                output_html=str(root / "report.html"),
                clips_dir=str(root / "clips"),
                fps=25.0,
                frame_size=(64, 48),
                buffer_frames=20,
                clip_workers=1,
                video_backend="opencv",
            )
            for frame_id in [2, 4, 6]:
                manager.record_frame(
                    frame_id,
                    np.full((48, 64, 3), frame_id * 10, dtype=np.uint8),
                )
            event = {
                "event_id": 1,
                "start_frame": 2,
                "end_frame": 6,
                "peak_frame": 4,
                "contact_frame": 4,
                "stroke_type": "Forehand",
                "confidence": 0.8,
                "quality_flags": {"warnings": []},
            }
            snapshot = {
                "summary": {"latest_frame": 6, "swing_event_count": 1},
                "events": [event],
                "frame_trace": [
                    {"frame": frame_id, "event_id": 1}
                    for frame_id in range(2, 7)
                ],
            }

            manager.publish_events([event], snapshot)
            manager.close()
            payload = json.loads((root / "events.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["events"][0]["clip_status"], "partial")
        self.assertEqual(payload["events"][0]["clip_missing_frame_count"], 2)
        self.assertEqual(payload["events"][0]["clip_missing_frames"], [3, 5])


if __name__ == "__main__":
    unittest.main()
