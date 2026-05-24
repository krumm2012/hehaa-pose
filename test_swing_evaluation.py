import json
import tempfile
import unittest
from pathlib import Path

from swing_evaluation import (
    default_evaluation_output_path,
    evaluate_swing_events,
    write_evaluation_report,
)


class SwingEvaluationTests(unittest.TestCase):
    def test_evaluates_model_events_against_manual_annotations(self):
        event_data = {
            "summary": {"swing_event_count": 2},
            "events": [
                {
                    "event_id": 1,
                    "stroke_type": "Two-Handed Backhand",
                    "contact_frame": 62,
                    "quality_flags": {"review_recommended": True, "warnings": ["ball_track_gaps"]},
                },
                {
                    "event_id": 2,
                    "stroke_type": "Forehand",
                    "contact_frame": 158,
                    "quality_flags": {"review_recommended": False, "warnings": []},
                },
            ],
        }
        annotation_data = {
            "schema_version": "swing_manual_annotations_v1",
            "events": [
                {
                    "event_id": 1,
                    "actual_stroke_type": "Forehand",
                    "valid_hit": True,
                    "count_correct": True,
                    "needs_review": False,
                    "issue_tags": ["wrong_stroke_type"],
                    "frames": {"contact": 64},
                },
                {
                    "event_id": 2,
                    "actual_stroke_type": "Forehand",
                    "valid_hit": True,
                    "count_correct": True,
                    "needs_review": True,
                    "issue_tags": [],
                    "frames": {"contact": 163},
                },
            ],
        }

        report = evaluate_swing_events(event_data, annotation_data, contact_tolerance_frames=3)

        self.assertEqual(report["summary"]["predicted_event_count"], 2)
        self.assertEqual(report["summary"]["manual_valid_event_count"], 2)
        self.assertEqual(report["summary"]["stroke_type_correct"], 1)
        self.assertEqual(report["summary"]["stroke_type_accuracy"], 0.5)
        self.assertEqual(report["summary"]["contact_within_tolerance"], 1)
        self.assertEqual(report["summary"]["manual_review_count"], 1)
        self.assertEqual(report["events"][0]["stroke_type_correct"], False)
        self.assertEqual(report["events"][0]["contact_delta_frames"], -2)
        self.assertEqual(report["events"][1]["contact_within_tolerance"], False)

    def test_writes_default_evaluation_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            event_path = Path(tmp) / "clip_swing_events.json"
            annotation_path = Path(tmp) / "clip_manual_annotations.json"
            event_path.write_text(
                json.dumps(
                    {
                        "events": [
                            {"event_id": 1, "stroke_type": "Forehand", "contact_frame": 10},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            annotation_path.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "event_id": 1,
                                "actual_stroke_type": "Forehand",
                                "valid_hit": True,
                                "count_correct": True,
                                "frames": {"contact": 11},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output_path = write_evaluation_report(str(event_path), str(annotation_path))

            self.assertEqual(output_path, default_evaluation_output_path(str(event_path)))
            saved = json.loads(Path(output_path).read_text(encoding="utf-8"))
            self.assertEqual(saved["summary"]["stroke_type_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
