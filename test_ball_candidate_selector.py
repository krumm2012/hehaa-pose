import unittest

from ball_candidate_selector import select_ball_candidate


class BallCandidateSelectorTests(unittest.TestCase):
    def test_prefers_continuous_real_ball_over_slightly_higher_confidence_mirror_ball(self):
        candidates = [
            {"position": [1291.0, 26.2], "confidence": 0.6553},
            {"position": [1229.0, 558.4], "confidence": 0.6401},
        ]

        selected = select_ball_candidate(
            candidates,
            previous_position=[1228.5, 575.2],
            racket_detections=[{"box": [1233, 460, 1397, 534], "confidence": 0.562}],
            config={"frame_height": 1440},
        )

        self.assertEqual(selected["position"], [1229.0, 558.4])

    def test_penalizes_top_mirror_region_when_no_previous_track_exists(self):
        candidates = [
            {"position": [1297.5, 18.6], "confidence": 0.55},
            {"position": [1229.5, 596.8], "confidence": 0.54},
        ]

        selected = select_ball_candidate(candidates, config={"frame_height": 1440})

        self.assertEqual(selected["position"], [1229.5, 596.8])

    def test_prefers_velocity_continuation_over_static_ball_jump(self):
        candidates = [
            {"position": [820.0, 839.2], "confidence": 0.72},   # static floor ball
            {"position": [896.0, 725.0], "confidence": 0.66},   # moving trajectory continuation
        ]
        selected = select_ball_candidate(
            candidates,
            previous_position=[898.0, 779.6],
            previous_velocity=[-1.0, -53.4],
            config={
                "ball_continuity_weight": 0.45,
                "ball_continuity_distance_px": 180.0,
                "ball_velocity_prediction_weight": 0.55,
                "ball_velocity_prediction_distance_px": 120.0,
            },
        )
        self.assertEqual(selected["position"], [896.0, 725.0])


if __name__ == "__main__":
    unittest.main()
