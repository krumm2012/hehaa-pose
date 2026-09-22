import unittest

from swing_motion_features import extract_motion_features
from swing_event_classifier import classify_swing_event
from swing_biomechanics import aggregate_event_biomechanics
from local_realtime_coach import LocalRealtimeCoach


class Algo2PipelineIntegrationTests(unittest.TestCase):
    def test_swing_motion_features_extracts_dual_view_and_healed_pose(self):
        frames = [
            {
                "frame_id": 10,
                "timestamp": 1.0,
                "pose": {"right_wrist": [100, 200], "left_shoulder": [80, 100], "right_shoulder": [120, 100]},
                "healed_pose": {"right_wrist": [105, 205], "left_shoulder": [80, 100], "right_shoulder": [120, 100]},
                "dual_view_biomechanics": {
                    "shoulder_turn": {"shoulder_turn_deg": 88.5},
                    "takeback_depth": {"takeback_depth_ratio": 0.42},
                    "scapular_retraction": {"scapular_retraction_ratio": 0.28},
                    "shot_classification": {"stroke_type": "Forehand", "is_two_handed": False},
                    "contact_distance_gate": {"is_valid_contact": True},
                },
            }
        ]
        features = extract_motion_features(frames, dominant_hand="right")
        self.assertEqual(len(features), 1)
        feat = features[0]
        # Should use healed_pose coordinates for wrist
        self.assertEqual(feat["wrist"], (105.0, 205.0))
        self.assertAlmostEqual(feat["robust_shoulder_turn_deg"], 88.5)
        self.assertAlmostEqual(feat["takeback_depth_ratio"], 0.42)
        self.assertAlmostEqual(feat["scapular_retraction_ratio"], 0.28)
        self.assertEqual(feat["dual_view_stroke_type"], "Forehand")
        self.assertFalse(feat["dual_view_is_two_handed"])
        self.assertTrue(feat["dual_view_contact_valid"])

    def test_swing_event_classifier_prioritizes_dual_view(self):
        event_features = [
            {
                "dual_view_stroke_type": "Two-Handed Backhand",
                "dual_view_is_two_handed": True,
                "active_wrist_x_offset_body_width": -0.25,
                "two_hand_distance_body_width": 0.35,
                "raw_swing_type": "Backhand",
                "dominant_hand": "right",
            }
            for _ in range(5)
        ]
        result = classify_swing_event(event_features)
        self.assertEqual(result["stroke_type"], "Two-Handed Backhand")
        self.assertGreaterEqual(result["confidence"], 0.90)
        self.assertEqual(
            result["evidence"]["classification_context"]["decision_rule"],
            "dual_view_two_handed_backhand",
        )
        self.assertIn("dual_view", result["evidence"]["classification_context"])
        self.assertEqual(
            result["evidence"]["classification_context"]["dual_view"]["evidence_frames"],
            5,
        )

    def test_swing_biomechanics_dual_view_aggregation(self):
        frames = [
            {
                "frame_id": i,
                "pose": {
                    "left_shoulder": [90, 100],
                    "right_shoulder": [150, 100],
                    "left_hip": [95, 200],
                    "right_hip": [145, 200],
                },
            }
            for i in range(5)
        ]
        features = [
            {
                "frame_id": i,
                "has_pose": True,
                "contact_score": 0.8 if i == 3 else 0.1,
                "robust_shoulder_turn_deg": 92.0 + i,
                "takeback_depth_ratio": 0.30 + i * 0.05,
                "scapular_retraction_ratio": 0.15 + i * 0.03,
                "arm_extension_deg": 140.0,
            }
            for i in range(5)
        ]
        event = {
            "start_frame": 0,
            "contact_frame": 3,
            "peak_frame": 3,
            "end_frame": 4,
            "quality_flags": {"pose_frame_ratio": 1.0},
        }

        result = aggregate_event_biomechanics(event, frames, features)
        self.assertEqual(result["schema_version"], "dual_view_2d_v1")
        metrics = result["metrics"]

        # Shoulder turn should use robust dual-view anti-collapse metric
        self.assertTrue(metrics["shoulder_turn"]["coach_eligible"])
        self.assertEqual(metrics["shoulder_turn"]["observability"], "dual_view_anti_collapse")
        self.assertEqual(metrics["shoulder_turn"]["unit"], "deg_360")

        # Takeback depth and scapular retraction should be present
        self.assertIn("takeback_depth", metrics)
        self.assertIn("scapular_retraction", metrics)
        self.assertTrue(metrics["takeback_depth"]["coach_eligible"])
        self.assertTrue(metrics["scapular_retraction"]["coach_eligible"])
        self.assertEqual(metrics["takeback_depth"]["observability"], "dual_view_mirror_projection")
        # Peak value between frame 0 and frame 3: 0.30 + 3 * 0.05 = 0.45
        self.assertAlmostEqual(metrics["takeback_depth"]["value"], 0.45)

    def test_local_realtime_coach_gives_dual_view_guidance(self):
        coach = LocalRealtimeCoach(max_suggestions=3, min_confidence=0.45)
        # Event with shallow takeback and low scapular retraction
        event = {
            "event_id": 99,
            "confidence": 0.92,
            "quality_flags": {"warnings": [], "pose_frame_ratio": 1.0},
            "phase_counts": {"backswing": 10, "follow_through": 10},
            "biomechanics": {
                "schema_version": "dual_view_2d_v1",
                "metrics": {
                    "takeback_depth": {
                        "value": 0.22,  # < threshold 0.35
                        "unit": "ratio",
                        "confidence": 0.85,
                        "coach_eligible": True,
                    },
                    "scapular_retraction": {
                        "value": 0.12,  # < threshold 0.20
                        "unit": "ratio",
                        "confidence": 0.82,
                        "coach_eligible": True,
                    },
                    "arm_extension": {
                        "value": 155.0,  # OK
                        "unit": "deg_2d",
                        "confidence": 0.90,
                        "coach_eligible": True,
                    },
                },
            },
        }

        advices = coach.advise_all(event)
        codes = [a["code"] for a in advices]
        self.assertIn("limited_takeback_depth", codes)
        self.assertIn("limited_scapular_retraction", codes)

        for advice in advices:
            self.assertLessEqual(len(advice["message"]), 15)
            if advice["code"] == "limited_takeback_depth":
                self.assertEqual(advice["message"], "充分展开后背引拍")
            elif advice["code"] == "limited_scapular_retraction":
                self.assertEqual(advice["message"], "转肩蓄力拉开后背")


if __name__ == "__main__":
    unittest.main()
