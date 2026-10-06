import unittest
from kinematic_attribution import (
    attribute_event_failures,
    attribute_all_events,
    explore_temporal_parameters,
    build_independent_benchmark_draft,
    build_structured_coach_rubric,
)


class KinematicAttributionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "frame_id": i,
                "source_time": {"timestamp_seconds": i * 0.04},
                "kinematic_views": {
                    "front": {
                        "left_hip": {"x": 100.0, "y": 200.0, "confidence": 0.9, "observed": True},
                        "right_hip": {"x": 130.0, "y": 200.0, "confidence": 0.9, "observed": True},
                        "left_shoulder": {"x": 95.0, "y": 150.0, "confidence": 0.9, "observed": True},
                        "right_shoulder": {"x": 135.0, "y": 150.0, "confidence": 0.9, "observed": True},
                    },
                    "back": {
                        "left_hip": {"x": 100.0, "y": 200.0, "confidence": 0.9, "observed": True},
                        "right_hip": {"x": 130.0, "y": 200.0, "confidence": 0.9, "observed": True},
                        "left_shoulder": {"x": 95.0, "y": 150.0, "confidence": 0.9, "observed": True},
                        "right_shoulder": {"x": 135.0, "y": 150.0, "confidence": 0.9, "observed": True},
                    },
                },
                "rackets": [{"box": [50.0, 50.0, 100.0, 100.0], "confidence": 0.8, "observed": True}],
            }
            for i in range(25)
        ]
        self.masked = [r for r in self.rows]
        self.review = {
            "labels": {
                "5:front:left_hip": {"visible": False, "review_actor": "human", "reason": "obscured"},
                "6:front:right_hip": {
                    "visible": False,
                    "review_actor": "automatic",
                    "abstention_reasons": ["projected_joint_pair_overlap"],
                },
            }
        }
        self.events = [{"event_id": 1, "contact_frame": 15}]

    def test_attribute_event_failures(self):
        attr = attribute_event_failures(self.rows, self.masked, self.review, 15, 1)
        self.assertEqual(attr["event_id"], 1)
        self.assertEqual(attr["contact_frame"], 15)
        self.assertIn("front", attr["views"])
        front_hip = attr["views"]["front"]["hip"]
        self.assertIn("missing_masked_frames", front_hip)
        self.assertEqual(front_hip["human_unknown_count"], 1)
        self.assertEqual(front_hip["automatic_abstention_count"], 1)
        self.assertIn("projected_joint_pair_overlap", front_hip["automatic_reasons_tally"])
        self.assertIn("contact_frame_racket", attr["racket"])
        self.assertIn("candidate_peak", attr["racket"])

    def test_attribute_all_events(self):
        res = attribute_all_events(self.rows, self.masked, self.review, self.events)
        self.assertEqual(res["schema"], "tennis.kinematic-failure-attribution.v1")
        self.assertEqual(res["event_count"], 1)
        self.assertIn("conclusions", res)

    def test_explore_temporal_parameters(self):
        res = explore_temporal_parameters(self.rows, self.events)
        self.assertEqual(res["schema"], "tennis.temporal-parameter-exploration.v1")
        self.assertFalse(res["parameters_approved"])
        self.assertEqual(len(res["parameter_candidates"]), 4)
        self.assertEqual(res["split_definition"]["tuning_event_ids"], [1])
        self.assertEqual(res["parameter_candidates"][1]["candidate"]["id"], "candidate_1_extended_followthrough")
        self.assertEqual(res["parameter_candidates"][2]["candidate"]["id"], "candidate_2_cadence_regularized")

    def test_build_independent_benchmark_draft(self):
        res = build_independent_benchmark_draft("a" * 64, "test_session", self.rows)
        self.assertEqual(res["schema"], "tennis.independent-joint-labels.v1")
        self.assertTrue(res["independent_reference"])
        self.assertFalse(res["confirmed"])
        self.assertEqual(len(res["frames"]), 9)
        self.assertEqual(len(res["labels"]), 9 * 2 * 4)

    def test_build_structured_coach_rubric(self):
        res = build_structured_coach_rubric("a" * 64, "test_session", "b" * 64)
        self.assertEqual(res["schema"], "tennis.independent-coach-reference-draft.v2")
        self.assertEqual(len(res["rules"]), 3)
        self.assertIn("expected_interval_ms", res["rules"][0])
        self.assertEqual(res["tolerance_score"], 5.0)


if __name__ == "__main__":
    unittest.main()
