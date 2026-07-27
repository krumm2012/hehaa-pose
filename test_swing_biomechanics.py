import unittest

from swing_biomechanics import aggregate_event_biomechanics


def frame(frame_id, center_x, ball_x=None):
    pose = {
        "left_shoulder": [center_x - 50, 100],
        "right_shoulder": [center_x + 50, 110],
        "left_hip": [center_x - 40, 200],
        "right_hip": [center_x + 40, 200],
    }
    return {
        "frame_id": frame_id,
        "pose": pose,
        "ball": [ball_x, 150] if ball_x is not None else None,
    }


class SwingBiomechanicsTests(unittest.TestCase):
    def test_aggregates_normalized_contact_rotation_and_balance_metrics(self):
        frames = [
            frame(0, 100),
            frame(1, 105),
            frame(2, 110, ball_x=140),
            frame(3, 125),
            frame(4, 145),
        ]
        features = [
            {
                "frame_id": row["frame_id"],
                "has_pose": True,
                "ball": row["ball"],
                "contact_score": 0.9 if row["frame_id"] == 2 else 0.0,
                "arm_extension_deg": 130 + row["frame_id"],
                "shoulder_turn_deg": 70 + row["frame_id"],
                "hip_shoulder_sep_deg": 10 + row["frame_id"],
            }
            for row in frames
        ]
        event = {
            "start_frame": 0,
            "contact_frame": 2,
            "peak_frame": 2,
            "end_frame": 4,
            "quality_flags": {"pose_frame_ratio": 1.0},
        }

        result = aggregate_event_biomechanics(event, frames, features)
        metrics = result["metrics"]

        self.assertEqual(result["schema_version"], "single_view_2d_v1")
        self.assertAlmostEqual(metrics["arm_extension"]["value"], 132.0)
        self.assertAlmostEqual(
            metrics["contact_lateral_distance"]["value"],
            30.0 / 90.2493781056,
            places=3,
        )
        self.assertGreater(metrics["weight_transfer"]["value"], 0.0)
        self.assertGreater(metrics["balance_drift"]["value"], 0.0)
        self.assertLessEqual(metrics["arm_extension"]["confidence"], 0.92)
        self.assertLessEqual(metrics["weight_transfer"]["confidence"], 0.8)
        self.assertLessEqual(metrics["balance_drift"]["confidence"], 0.8)

    def test_missing_pose_returns_unknown_metrics_instead_of_claims(self):
        event = {
            "start_frame": 0,
            "contact_frame": 0,
            "peak_frame": 0,
            "end_frame": 0,
            "quality_flags": {"pose_frame_ratio": 0.0},
        }
        result = aggregate_event_biomechanics(
            event,
            [{"frame_id": 0, "pose": {}}],
            [{"frame_id": 0, "has_pose": False}],
        )

        self.assertTrue(
            all(
                metric["value"] is None and metric["confidence"] == 0.0
                for metric in result["metrics"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
