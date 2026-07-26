from collections import deque
import unittest

import numpy as np

from ball_track_selector import BallTrackSelector
from static_ball_filter import StaticBallFilter
from yolo26n_unified_detector import YOLO26nUnifiedDetector


class BallTrackSelectorTests(unittest.TestCase):
    """Verify Active Ball selection through the public track-selection seams."""

    def test_continuous_active_ball_beats_a_persistent_static_candidate(self) -> None:
        """Prefer a continuous Active Ball over a persistent static candidate."""
        selector = BallTrackSelector(
            {
                "static_ball_suppression_enabled": True,
                "static_ball_min_seen_frames": 3,
                "static_ball_hard_mask_enabled": True,
                "static_ball_hard_mask_min_seen_frames": 3,
                "static_ball_hard_mask_allow_near_prev": True,
                "ball_continuity_weight": 0.45,
                "ball_continuity_distance_px": 180.0,
            }
        )

        for x in (100.0, 112.0, 124.0):
            result = selector.select(
                [
                    {"position": [x, 300.0], "confidence": 0.95},
                    {"position": [700.0, 650.0], "confidence": 0.80},
                ],
                frame_height=720,
            )
            self.assertEqual(result.active_ball["position"], [x, 300.0])

        result = selector.select(
            [
                {"position": [136.0, 300.0], "confidence": 0.65},
                {"position": [700.0, 650.0], "confidence": 0.99},
            ],
            frame_height=720,
        )

        self.assertEqual(result.active_ball["position"], [136.0, 300.0])
        self.assertEqual(result.diagnostics["rejections"]["static_hard_mask"], 1)

    def test_detector_keeps_a_supported_active_ball_out_of_static_anchors(self) -> None:
        """Preserve detector diagnostics while a supported Active Ball is stationary."""
        config = {
            "static_ball_suppression_enabled": True,
            "static_ball_min_seen_frames": 3,
            "static_ball_hard_mask_enabled": True,
            "static_ball_hard_mask_min_seen_frames": 3,
            "static_ball_hard_mask_allow_near_prev": True,
        }
        detector = YOLO26nUnifiedDetector.__new__(YOLO26nUnifiedDetector)
        detector.model = _FakeModel()
        detector._coreml_input_names = set()
        detector._preprocess = lambda frame: frame
        detector._parse_predictions = lambda _predictions: (
            [{"position": [320.0, 420.0], "confidence": 0.90}],
            [],
        )
        detector.detection_times = deque(maxlen=100)
        detector.config = config
        detector.ball_history = deque(maxlen=10)
        detector.racket_history = deque(maxlen=10)
        detector.static_threshold = 6
        detector.static_ball_filter = StaticBallFilter(config)
        detector.ball_track_selector = BallTrackSelector(config)
        detector.last_ball_diagnostics = {}

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(4):
            balls, _rackets, _inference_time = detector.detect_unified(frame)
            self.assertEqual(balls[0]["position"], [320.0, 420.0])
            diagnostics = detector.get_last_ball_diagnostics()
            self.assertEqual(diagnostics["final_decision"], "selected")
            self.assertTrue(
                {
                    "raw_candidates",
                    "kept_candidates",
                    "continuity_enabled",
                    "continuity_disabled_reason",
                    "rejections",
                    "selected",
                    "final_decision",
                    "top_candidates",
                }.issubset(diagnostics)
            )

    def test_reacquires_only_after_the_eight_frame_window_expires(self) -> None:
        """Keep trajectory support for eight misses, then reacquire."""
        retained = BallTrackSelector({"active_ball_reacquisition_frames": 8})
        retained.select([{"position": [100.0, 300.0], "confidence": 0.90}])
        for _ in range(8):
            retained.select([])
        within_window = retained.select(
            [
                {"position": [110.0, 300.0], "confidence": 0.30},
                {"position": [650.0, 300.0], "confidence": 0.50},
            ]
        )

        expired = BallTrackSelector({"active_ball_reacquisition_frames": 8})
        expired.select([{"position": [100.0, 300.0], "confidence": 0.90}])
        for _ in range(9):
            expired.select([])
        after_window = expired.select(
            [
                {"position": [110.0, 300.0], "confidence": 0.30},
                {"position": [650.0, 300.0], "confidence": 0.50},
            ]
        )

        self.assertEqual(within_window.active_ball["position"], [110.0, 300.0])
        self.assertTrue(within_window.diagnostics["continuity_enabled"])
        self.assertEqual(after_window.active_ball["position"], [650.0, 300.0])
        self.assertFalse(after_window.diagnostics["continuity_enabled"])

    def test_trajectory_support_bypasses_a_static_hard_mask(self) -> None:
        """Never hard-mask a candidate with Trajectory Support."""
        selector = BallTrackSelector(
            {
                "static_ball_min_seen_frames": 3,
                "static_ball_hard_mask_min_seen_frames": 3,
                "static_ball_hard_mask_allow_near_prev": True,
            }
        )
        for _ in range(3):
            selector.static_ball_filter.update([[140.0, 300.0]])
        selector.select([{"position": [100.0, 300.0], "confidence": 0.9}])
        selector.select([{"position": [105.0, 300.0], "confidence": 0.9}])

        result = selector.select([{"position": [140.0, 300.0], "confidence": 0.3}])

        self.assertEqual(result.active_ball["position"], [140.0, 300.0])
        self.assertEqual(result.diagnostics["rejections"]["static_hard_mask"], 0)

    def test_stationary_active_ball_is_not_rejected_or_learned_as_static(self) -> None:
        """Keep a stationary Active Ball out of static-anchor learning."""
        selector = BallTrackSelector({"static_ball_movement_threshold_px": 6})

        for _ in range(11):
            result = selector.select([{"position": [320.0, 420.0], "confidence": 0.9}])
            self.assertEqual(result.active_ball["position"], [320.0, 420.0])

        self.assertEqual(result.diagnostics["final_decision"], "selected")

    def test_fast_motion_keeps_trajectory_support_inside_reacquisition_window(self) -> None:
        """Retain support after a fast prior motion until the window expires."""
        selector = BallTrackSelector(
            {
                "active_ball_reacquisition_frames": 8,
                "ball_max_motion_for_continuity_px": 140,
            }
        )
        selector.select([{"position": [0.0, 300.0], "confidence": 0.9}])
        selector.select([{"position": [200.0, 300.0], "confidence": 0.9}])
        selector.select([])

        result = selector.select(
            [
                {"position": [210.0, 300.0], "confidence": 0.1},
                {"position": [600.0, 300.0], "confidence": 0.9},
            ]
        )

        self.assertEqual(result.active_ball["position"], [210.0, 300.0])
        self.assertTrue(result.diagnostics["continuity_enabled"])


class _FakeModel:
    """Minimal Core ML model adapter for public detector behavior tests."""

    def predict(self, _inputs: dict) -> dict:
        """Return an empty prediction because parsing is supplied by the test."""
        return {}


if __name__ == "__main__":
    unittest.main()
