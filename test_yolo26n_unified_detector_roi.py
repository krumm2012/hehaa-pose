import unittest

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


if __name__ == "__main__":
    unittest.main()
