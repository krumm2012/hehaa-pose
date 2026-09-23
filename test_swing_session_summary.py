"""Unit tests for swing_session_summary module."""

import unittest
from swing_session_summary import (
    build_session_coaching_summary,
    calculate_radar_dimensions,
)


class SwingSessionSummaryTests(unittest.TestCase):
    def test_empty_events(self):
        summary = build_session_coaching_summary([])
        self.assertEqual(summary["total_swings"], 0)
        self.assertEqual(summary["valid_shots_count"], 0)
        self.assertIsNone(summary["quality_metrics"]["average_score"])
        self.assertEqual(summary["distribution"]["forehand_count"], 0)

    def test_only_shadow_swings(self):
        events = [
            {
                "event_id": 1,
                "stroke_type": "Forehand",
                "is_shadow_swing": True,
                "swing_score": 55.0,
            }
        ]
        summary = build_session_coaching_summary(events)
        self.assertEqual(summary["total_swings"], 1)
        self.assertEqual(summary["valid_shots_count"], 0)
        self.assertEqual(summary["distribution"]["shadow_count"], 1)
        self.assertEqual(summary["distribution"]["shadow_ratio"], 100.0)
        self.assertIsNone(summary["quality_metrics"]["average_score"])
        self.assertIn("空挥", summary["macro_diagnosis"])

    def test_multi_ball_session_aggregation(self):
        events = [
            {
                "event_id": 1,
                "stroke_type": "Forehand",
                "is_shadow_swing": True,
                "swing_score": 50.0,
                "coach_advices": [],
            },
            {
                "event_id": 2,
                "stroke_type": "Forehand",
                "is_shadow_swing": False,
                "swing_score": 85.0,
                "swing_grade": "ADVANCED",
                "extended_biomechanics": {
                    "racket_head_speed": {"contact_kmh": 60.0},
                    "brush_angle": {"low_to_high_angle_deg": 45.0, "drop_depth_ratio": 0.3},
                    "kinematic_sequence": {"sequence_quality": "OPTIMAL"},
                    "leg_drive": {"drive_ratio": 1.4},
                },
                "coach_advices": [
                    {"code": "knee_flexion_low", "message": "准备时适当降低重心"}
                ],
            },
            {
                "event_id": 3,
                "stroke_type": "Forehand",
                "is_shadow_swing": False,
                "swing_score": 88.0,
                "swing_grade": "ADVANCED",
                "extended_biomechanics": {
                    "racket_head_speed": {"contact_kmh": 65.0},
                    "brush_angle": {"low_to_high_angle_deg": 35.0, "drop_depth_ratio": 0.15},
                    "kinematic_sequence": {"sequence_quality": "OPTIMAL"},
                    "leg_drive": {"drive_ratio": 1.3},
                },
                "coach_advices": [
                    {"code": "knee_flexion_low", "message": "准备时适当降低重心"},
                    {"code": "brush_drop_low", "message": "击球前拍头下潜刷球"},
                ],
            },
            {
                "event_id": 4,
                "stroke_type": "Backhand",
                "is_shadow_swing": False,
                "swing_score": 82.0,
                "swing_grade": "ADVANCED",
                "extended_biomechanics": {
                    "racket_head_speed": {"contact_kmh": 55.0},
                    "brush_angle": {"low_to_high_angle_deg": 30.0, "drop_depth_ratio": 0.2},
                    "kinematic_sequence": {"sequence_quality": "DISCONNECTED"},
                    "leg_drive": {"drive_ratio": 1.1},
                },
                "coach_advices": [
                    {"code": "kinematic_sequence_disconnected", "message": "用身体核心带动球拍发力"},
                ],
            },
        ]

        summary = build_session_coaching_summary(events)
        self.assertEqual(summary["total_swings"], 4)
        self.assertEqual(summary["valid_shots_count"], 3)

        # Distribution
        dist = summary["distribution"]
        self.assertEqual(dist["forehand_count"], 2)
        self.assertEqual(dist["backhand_count"], 1)
        self.assertEqual(dist["shadow_count"], 1)
        self.assertEqual(dist["forehand_ratio"], 50.0)
        self.assertEqual(dist["backhand_ratio"], 25.0)
        self.assertEqual(dist["shadow_ratio"], 25.0)

        # Quality metrics (scores: 85, 88, 82 -> mean: 85.0)
        qm = summary["quality_metrics"]
        self.assertEqual(qm["average_score"], 85.0)
        self.assertEqual(qm["min_score"], 82.0)
        self.assertEqual(qm["max_score"], 88.0)
        self.assertAlmostEqual(qm["score_std"], 3.0, places=1)
        self.assertEqual(qm["stability_rating"], "HIGH_CONSISTENCY")

        # Common deficiencies
        # knee_flexion_low: 2 / 3 = 66.7%
        # brush_drop_low: 1 / 3 = 33.3%
        # kinematic_sequence_disconnected: 1 / 3 = 33.3%
        defs = summary["common_deficiencies"]
        self.assertEqual(len(defs), 3)
        self.assertEqual(defs[0]["code"], "knee_flexion_low")
        self.assertEqual(defs[0]["count"], 2)
        self.assertEqual(defs[0]["occurrence_rate_percent"], 66.7)
        self.assertEqual(defs[0]["severity"], "HIGH")

        # Macro diagnosis text
        macro = summary["macro_diagnosis"]
        self.assertIn("完成 4 次挥拍", macro)
        self.assertIn("正手 2 球", macro)
        self.assertIn("平均技术质量得分为 85.0 分", macro)
        self.assertIn("准备时适当降低重心（出现率 66.7%）", macro)
        self.assertIn("训练处方建议", macro)

        # Radar dimensions
        radar = summary["radar_averages"]
        self.assertGreater(radar["speed"], 50.0)
        self.assertGreater(radar["kinematics"], 60.0)

    def test_radar_dimension_bounds(self):
        ev = {
            "extended_biomechanics": {
                "racket_head_speed": {"contact_kmh": 70.0},
                "brush_angle": {"low_to_high_angle_deg": 40.0, "drop_depth_ratio": 0.25},
                "kinematic_sequence": {"sequence_quality": "OPTIMAL"},
                "leg_drive": {"drive_ratio": 1.4},
            },
            "swing_score": 85.0,
        }
        dims = calculate_radar_dimensions(ev)
        for k in ["speed", "brush", "kinematics", "leg_drive", "preparation"]:
            self.assertIn(k, dims)
            self.assertGreaterEqual(dims[k], 0.0)
            self.assertLessEqual(dims[k], 100.0)


if __name__ == "__main__":
    unittest.main()
