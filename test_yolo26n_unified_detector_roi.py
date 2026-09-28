import unittest

import numpy as np

from yolo26n_unified_detector import YOLO26nUnifiedDetector


class UnifiedDetectorROITests(unittest.TestCase):
    def test_parses_two_class_end_to_end_output_with_configured_name(self):
        detector = YOLO26nUnifiedDetector.__new__(YOLO26nUnifiedDetector)
        detector.original_width = 1920
        detector.original_height = 1080
        detector.input_width = 960
        detector.input_height = 960
        detector.ball_class_id = 0
        detector.racket_class_id = 1
        detector.coreml_detection_output = "var_1440"
        detector.ball_conf_threshold = 0.692
        detector.racket_conf_threshold = 0.524
        detector.last_parse_diagnostics = {}

        output = np.array([[[100, 120, 112, 132, 0.80, 0],
                            [300, 220, 380, 320, 0.70, 1],
                            [400, 300, 410, 310, 0.60, 0]]], dtype=float)

        balls, rackets = detector._parse_predictions({"var_1440": output})

        self.assertEqual(len(balls), 1)
        self.assertEqual(len(rackets), 1)
        self.assertEqual(detector.last_parse_diagnostics["output_format"], "var_1440")
        self.assertEqual(detector.last_parse_diagnostics["ball"]["class_candidates"], 2)
        self.assertEqual(detector.last_parse_diagnostics["ball"]["above_threshold_candidates"], 1)

    def test_letterbox_restores_original_coordinates(self):
        detector = YOLO26nUnifiedDetector.__new__(YOLO26nUnifiedDetector)
        detector.original_width = 1920
        detector.original_height = 1080
        detector.input_width = 960
        detector.input_height = 960
        detector.preprocess_mode = "letterbox"
        detector.ball_class_id = 0
        detector.racket_class_id = 1
        detector.coreml_detection_output = "var_1440"
        detector.ball_conf_threshold = 0.5
        detector.racket_conf_threshold = 0.5
        detector.last_parse_diagnostics = {}

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        prepared = detector._preprocess(frame)
        balls, _ = detector._parse_predictions({
            "var_1440": np.array([[[100, 300, 200, 400, 0.8, 0]]], dtype=float)
        })

        self.assertEqual(prepared.size, (960, 960))
        self.assertEqual(detector._preprocess_pad, (0, 210))
        self.assertEqual(balls[0]["box"], [200.0, 180.0, 400.0, 380.0])

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

    def test_filters_balls_in_mirror_exclusion_zone_via_roi_manager(self):
        from roi_manager import ROIManager
        detector = YOLO26nUnifiedDetector.__new__(YOLO26nUnifiedDetector)
        detector.roi_manager = None
        detector._coreml_input_names = set()
        detector.detection_times = []
        detector.config = {}
        detector.last_ball_diagnostics = {}
        detector.last_racket_diagnostics = {}
        detector._filter_static_balls = lambda balls, rackets: balls
        detector._select_primary_racket = lambda rackets, balls: rackets

        roi_manager = ROIManager({"roi_settings": {"enabled": True}})
        roi_manager.set_roi_points([(100, 100), (900, 100), (900, 900), (100, 900)])
        # Mirror exclusion polygon
        roi_manager.add_exclusion_polygon([(400, 300), (600, 300), (600, 500), (400, 500)], label="mirror")
        detector.set_roi_manager(roi_manager)

        # Mock _preprocess, model, _parse_predictions
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        detector._preprocess = lambda img: None
        detector.model = type("MockModel", (), {"predict": lambda self, feed: {}})()

        # Candidate ball 1: inside court (250, 400)
        # Candidate ball 2: inside mirror (500, 400)
        # Candidate ball 3: outside court (50, 50)
        detector._parse_predictions = lambda preds: (
            [
                {"position": [250, 400], "box": [245, 395, 255, 405], "confidence": 0.9},
                {"position": [500, 400], "box": [495, 395, 505, 405], "confidence": 0.95},
                {"position": [50, 50], "box": [45, 45, 55, 55], "confidence": 0.85},
            ],
            [],
        )

        balls, rackets, _ = detector.detect_unified(frame)
        self.assertEqual(len(balls), 1)
        self.assertEqual(balls[0]["position"], [250, 400])


if __name__ == "__main__":
    unittest.main()
