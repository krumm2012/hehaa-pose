import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from manual_review_workflow import (
    discover_session_paths,
    process_manual_review,
    validate_manual_annotations,
)


def _frame(frame_id: int):
    wrist_x = 100 + frame_id * 4
    return {
        "frame_id": frame_id,
        "timestamp": frame_id / 25.0,
        "swing_type": "Forehand",
        "ball": [wrist_x + 20, 100] if frame_id % 3 else None,
        "rackets": [
            {
                "box": [wrist_x - 5, 90, wrist_x + 8, 116],
                "confidence": 0.9,
            }
        ],
        "pose": {
            "right_wrist": [wrist_x, 100],
            "left_wrist": [80, 102],
            "right_shoulder": [120, 80],
            "left_shoulder": [80, 80],
            "right_elbow": [wrist_x - 8, 92],
            "left_elbow": [78, 92],
            "right_hip": [116, 140],
            "left_hip": [84, 140],
            "right_knee": [118, 180],
            "left_knee": [82, 180],
            "right_ankle": [120, 220],
            "left_ankle": [80, 220],
        },
        "metrics": {},
        "detection_diagnostics": {"rejections": {}},
    }


def _event_document():
    return {
        "schema_version": "tennis.swing-events.v1",
        "session": {"session_id": "session-review-test"},
        "summary": {"latest_frame": 19},
        "events": [
            {
                "event_id": 1,
                "start_frame": 2,
                "contact_frame": 8,
                "peak_frame": 8,
                "end_frame": 14,
                "duration_frames": 13,
                "stroke_type": "Forehand",
                "confidence": 0.8,
                "evidence": {},
                "quality_flags": {
                    "pose_frame_ratio": 1.0,
                    "ball_frame_ratio": 0.5,
                    "racket_frame_ratio": 1.0,
                    "warnings": ["ball_track_gaps"],
                },
                "phase_counts": {"backswing": 4, "follow_through": 4},
                "coach_advice": {
                    "code": "ball_track_gaps",
                    "message": "确保来球完整入镜",
                    "confidence": 0.5,
                },
            }
        ],
    }


def _annotations(needs_review: bool):
    return {
        "schema_version": "swing_manual_annotations_v2",
        "timeline_review_complete": True,
        "source": {
            "event_json": "final_events.json",
            "session_id": "session-review-test",
        },
        "events": [
            {
                "annotation_id": "model-1",
                "source_event_id": 1,
                "actual_stroke_type": "Backhand",
                "count_correct": True,
                "valid_hit": True,
                "needs_review": needs_review,
                "issue_tags": ["wrong_type", "event_boundary"],
                "note": "人工修正",
                "frames": {"start": 1, "contact": 9, "peak": 8, "end": 16},
            }
        ],
    }


class ManualReviewWorkflowTests(unittest.TestCase):
    def _write_session(self, root: Path):
        events = _event_document()
        event_path = root / "final_events.json"
        event_path.write_text(json.dumps(events), encoding="utf-8")
        frame_path = root / "final_frames.jsonl"
        frame_path.write_text(
            "\n".join(json.dumps(_frame(index)) for index in range(20)) + "\n",
            encoding="utf-8",
        )
        (root / "final_report.html").write_text("original report", encoding="utf-8")
        return events, [_frame(index) for index in range(20)]

    def test_pending_review_generates_provisional_evaluation_without_manual_coach(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_session(root)
            paths = discover_session_paths(root)
            state = process_manual_review(paths, _annotations(needs_review=True))

            self.assertEqual(state["status"], "needs_review")
            self.assertEqual(state["validation"]["pending_count"], 1)
            self.assertTrue(state["evaluation"]["summary"]["provisional"])
            self.assertEqual(state["comparisons"], [])
            self.assertFalse(paths["manual_events"].exists())

    def test_final_review_recomputes_coach_and_preserves_original_events(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original, _ = self._write_session(root)
            paths = discover_session_paths(root)
            before_hash = hashlib.sha256(paths["events"].read_bytes()).hexdigest()

            state = process_manual_review(paths, _annotations(needs_review=False))

            after_hash = hashlib.sha256(paths["events"].read_bytes()).hexdigest()
            self.assertEqual(before_hash, after_hash)
            self.assertEqual(state["status"], "finalized")
            self.assertFalse(state["evaluation"]["summary"]["provisional"])
            manual = json.loads(paths["manual_events"].read_text(encoding="utf-8"))
            event = manual["events"][0]
            self.assertEqual(event["stroke_type"], "Backhand")
            self.assertEqual(
                [event["start_frame"], event["contact_frame"], event["end_frame"]],
                [1, 9, 16],
            )
            self.assertTrue(event["coach_advices"])
            self.assertIn("stroke_type", event["review_provenance"]["changed_fields"])
            self.assertIn("start_frame", event["review_provenance"]["changed_fields"])
            self.assertEqual(original["events"][0]["stroke_type"], "Forehand")

    def test_rejects_annotation_from_another_event_file(self):
        event_document = _event_document()
        frames = [_frame(index) for index in range(20)]
        annotations = _annotations(needs_review=False)
        annotations["source"]["event_json"] = "other_events.json"

        with self.assertRaisesRegex(ValueError, "不匹配"):
            validate_manual_annotations(
                event_document,
                annotations,
                frames,
                event_path=Path("final_events.json"),
            )


if __name__ == "__main__":
    unittest.main()
