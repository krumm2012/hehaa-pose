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

    def test_v2_matches_independent_timeline_annotations_and_measures_recall(self):
        event_data = {
            "events": [
                {
                    "event_id": 1,
                    "start_frame": 10,
                    "contact_frame": 20,
                    "end_frame": 30,
                    "stroke_type": "Forehand",
                    "quality_flags": {},
                },
                {
                    "event_id": 2,
                    "start_frame": 50,
                    "contact_frame": 60,
                    "end_frame": 70,
                    "stroke_type": "Backhand",
                    "quality_flags": {},
                },
                {
                    "event_id": 3,
                    "start_frame": 90,
                    "contact_frame": 100,
                    "end_frame": 110,
                    "stroke_type": "Forehand",
                    "quality_flags": {},
                },
            ]
        }
        annotation_data = {
            "schema_version": "swing_manual_annotations_v2",
            "timeline_review_complete": True,
            "events": [
                {
                    "annotation_id": "manual-a",
                    "source_event_id": 99,
                    "actual_stroke_type": "Forehand",
                    "valid_hit": True,
                    "frames": {"start": 11, "contact": 22, "end": 31},
                },
                {
                    "annotation_id": "manual-b",
                    "source_event_id": None,
                    "actual_stroke_type": "Two-Handed Backhand",
                    "valid_hit": True,
                    "frames": {"start": 52, "contact": 61, "end": 72},
                },
                {
                    "annotation_id": "manual-c",
                    "source_event_id": None,
                    "actual_stroke_type": "Forehand",
                    "valid_hit": True,
                    "frames": {"start": 130, "contact": 140, "end": 150},
                },
            ],
        }

        report = evaluate_swing_events(
            event_data,
            annotation_data,
            contact_tolerance_frames=3,
            match_contact_tolerance_frames=12,
        )

        summary = report["summary"]
        self.assertEqual(report["schema_version"], "swing_evaluation_v2")
        self.assertEqual(summary["true_positive_count"], 2)
        self.assertEqual(summary["false_positive_count"], 1)
        self.assertEqual(summary["false_negative_count"], 1)
        self.assertEqual(summary["precision"], 0.6667)
        self.assertEqual(summary["recall"], 0.6667)
        self.assertEqual(summary["f1"], 0.6667)
        self.assertEqual(summary["stroke_type_accuracy"], 0.5)
        self.assertEqual(summary["contact_accuracy"], 1.0)
        self.assertEqual(summary["contact_mean_abs_error_frames"], 1.5)
        self.assertEqual(summary["start_mean_abs_error_frames"], 1.5)
        self.assertEqual(summary["end_mean_abs_error_frames"], 1.5)
        self.assertEqual(report["review"]["unmatched_model_event_ids"], [3])
        self.assertEqual(
            report["review"]["unmatched_annotation_ids"],
            ["manual-c"],
        )
        self.assertEqual(report["events"][0]["annotation_id"], "manual-a")
        self.assertEqual(report["events"][1]["annotation_id"], "manual-b")

    def test_v2_marks_precision_and_recall_provisional_until_timeline_reviewed(self):
        report = evaluate_swing_events(
            {
                "events": [
                    {
                        "event_id": 1,
                        "start_frame": 10,
                        "contact_frame": 20,
                        "end_frame": 30,
                        "stroke_type": "Forehand",
                    }
                ]
            },
            {
                "schema_version": "swing_manual_annotations_v2",
                "timeline_review_complete": False,
                "events": [
                    {
                        "annotation_id": "model-1",
                        "source_event_id": 1,
                        "actual_stroke_type": "Forehand",
                        "valid_hit": True,
                        "frames": {"start": 10, "contact": 20, "end": 30},
                    }
                ],
            },
        )

        self.assertTrue(report["summary"]["provisional"])
        self.assertIsNone(report["summary"]["precision"])
        self.assertIsNone(report["summary"]["recall"])
        self.assertIsNone(report["summary"]["f1"])
        self.assertEqual(
            report["summary"]["provisional_reasons"],
            ["timeline_review_incomplete"],
        )

    def test_v2_keeps_metrics_provisional_while_manual_review_is_pending(self):
        report = evaluate_swing_events(
            {
                "events": [
                    {
                        "event_id": 1,
                        "start_frame": 10,
                        "contact_frame": 20,
                        "end_frame": 30,
                        "stroke_type": "Forehand",
                    }
                ]
            },
            {
                "schema_version": "swing_manual_annotations_v2",
                "timeline_review_complete": True,
                "events": [
                    {
                        "annotation_id": "model-1",
                        "actual_stroke_type": "Forehand",
                        "valid_hit": True,
                        "needs_review": True,
                        "frames": {"start": 10, "contact": 20, "end": 30},
                    }
                ],
            },
        )

        summary = report["summary"]
        self.assertTrue(summary["timeline_review_complete"])
        self.assertTrue(summary["provisional"])
        self.assertFalse(summary["metrics_finalized"])
        self.assertEqual(summary["provisional_reasons"], ["manual_review_pending"])
        self.assertEqual(summary["manual_review_count"], 1)
        self.assertIsNone(summary["precision"])
        self.assertIsNone(summary["recall"])
        self.assertIsNone(summary["f1"])

    def test_v2_reports_zero_f1_when_all_predictions_and_truth_are_unmatched(self):
        report = evaluate_swing_events(
            {
                "events": [
                    {
                        "event_id": 1,
                        "start_frame": 10,
                        "contact_frame": 20,
                        "end_frame": 30,
                        "stroke_type": "Forehand",
                    }
                ]
            },
            {
                "schema_version": "swing_manual_annotations_v2",
                "timeline_review_complete": True,
                "events": [
                    {
                        "annotation_id": "manual-1",
                        "actual_stroke_type": "Forehand",
                        "valid_hit": True,
                        "frames": {"start": 100, "contact": 110, "end": 120},
                    }
                ],
            },
        )

        self.assertEqual(report["summary"]["precision"], 0.0)
        self.assertEqual(report["summary"]["recall"], 0.0)
        self.assertEqual(report["summary"]["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
