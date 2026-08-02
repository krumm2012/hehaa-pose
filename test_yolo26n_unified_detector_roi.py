import unittest

import numpy as np

from yolo26n_unified_detector import YOLO26nUnifiedDetector


class UnifiedDetectorROITests(unittest.TestCase):
    def test_offsets_ball_and_racket_before_candidate_selection(self):
        detections = [
            {
                "position": [10.5, 20.5],
                "box": [8, 18, 13, 23],
                "confidence": 0.8,
            }
        ]

        adjusted = YOLO26nUnifiedDetector._offset_detections(
            detections,
            (700, 150),
        )

        self.assertEqual(adjusted[0]["position"], [710.5, 170.5])
        self.assertEqual(adjusted[0]["box"], [708, 168, 713, 173])
        self.assertEqual(detections[0]["position"], [10.5, 20.5])

    def test_records_subthreshold_racket_confidence_without_selecting_it(self):
        detector = YOLO26nUnifiedDetector.__new__(YOLO26nUnifiedDetector)
        detector.original_width = 1280
        detector.original_height = 720
        detector.ball_class_id = 32
        detector.racket_class_id = 38
        detector.ball_conf_threshold = 0.02
        detector.racket_conf_threshold = 0.35
        detector.last_parse_diagnostics = {}

        coordinates = np.array([[0.5, 0.5, 0.2, 0.3]], dtype=float)
        confidence = np.zeros((1, 80), dtype=float)
        confidence[0, 32] = 0.019
        confidence[0, 38] = 0.32

        balls, rackets = detector._parse_predictions(
            {"coordinates": coordinates, "confidence": confidence}
        )

        self.assertEqual(balls, [])
        self.assertEqual(rackets, [])
        self.assertEqual(
            detector.last_parse_diagnostics["racket"]["max_confidence"],
            0.32,
        )
        self.assertEqual(
            detector.last_parse_diagnostics["racket"]["above_threshold_candidates"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
