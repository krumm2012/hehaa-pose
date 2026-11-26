"""Unit tests for vectorized detector/pose parsing and DetWorkerPool."""

from __future__ import annotations

import queue
import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from pipeline.det_worker_pool import DetWorkerPool
from pose_estimator_yolo26 import PoseEstimatorYOLO26
from static_ball_filter import StaticBallFilter
from yolo26n_unified_detector import YOLO26nUnifiedDetector


class _TrackingBuffer(np.ndarray):
    fill_calls = 0

    def fill(self, value):
        type(self).fill_calls += 1
        return super().fill(value)


class TestVectorizedParsersAndPool(unittest.TestCase):
    def _make_stub_detector(self, load_model=False):
        cfg = {
            "ball_class_id": 0,
            "racket_class_id": 1,
            "coreml_detection_output": "var_1440",
            "preprocess_mode": "letterbox",
            "ball_confidence_threshold": 0.65,
            "racket_confidence_threshold": 0.50,
            "racket_temporal_recovery_enabled": True,
            "racket_temporal_recovery_min_confidence": 0.25,
            "input_size": 960,
        }
        if load_model:
            return YOLO26nUnifiedDetector("unused.mlpackage", cfg, load_model=False)
        det = object.__new__(YOLO26nUnifiedDetector)
        det.config = cfg
        det.roi_manager = None
        det.model = None
        det._coreml_input_names = set()
        det._coreml_output_names = set()
        det.input_width = 960
        det.input_height = 960
        det.original_width = 1920
        det.original_height = 1080
        det.tracking_frame_width = 1920
        det.tracking_frame_height = 1080
        det.ball_class_id = 0
        det.racket_class_id = 1
        det.coreml_detection_output = "var_1440"
        det.preprocess_mode = "letterbox"
        det._preprocess_gain = 0.5
        det._preprocess_pad = (0, 210)
        det.ball_conf_threshold = 0.65
        det.racket_conf_threshold = 0.50
        det.ball_history = deque(maxlen=10)
        det.racket_history = deque(maxlen=10)
        det.static_ball_filter = StaticBallFilter(cfg)
        det.detection_times = deque(maxlen=100)
        det.last_ball_diagnostics = {}
        det.last_racket_diagnostics = {}
        det.last_parse_diagnostics = {}
        return det

    def test_vectorized_parse_predictions_matches_counts_and_geometry(self):
        det = self._make_stub_detector()
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        # Ball above threshold
        raw[0, 0] = [100, 310, 120, 330, 0.88, 0]
        # Ball below threshold
        raw[0, 1] = [140, 310, 160, 330, 0.40, 0]
        # Valid racket above threshold (aspect ratio 2.0, area well under 1/10 frame)
        raw[0, 2] = [200, 310, 230, 370, 0.72, 1]
        # Recoverable racket below main threshold (>= 0.25)
        raw[0, 3] = [300, 310, 330, 370, 0.35, 1]
        # Geometry-rejected tiny racket (< 500 px^2)
        raw[0, 4] = [400, 310, 405, 315, 0.90, 1]

        balls, rackets = det._parse_predictions({"var_1440": raw})
        self.assertEqual(len(balls), 1)
        self.assertAlmostEqual(balls[0]["position"][0], 220.0, places=3)
        self.assertAlmostEqual(balls[0]["position"][1], 220.0, places=3)
        self.assertEqual(len(rackets), 2)

        diag = det.last_parse_diagnostics
        self.assertEqual(diag["ball"]["class_candidates"], 297)  # 0s in trailing rows count as class 0 with 0 conf
        self.assertAlmostEqual(diag["ball"]["max_confidence"], 0.88, places=4)
        self.assertEqual(diag["ball"]["above_threshold_candidates"], 1)
        self.assertEqual(diag["racket"]["class_candidates"], 3)
        self.assertAlmostEqual(diag["racket"]["max_confidence"], 0.90, places=4)
        self.assertEqual(diag["racket"]["above_threshold_candidates"], 2)
        self.assertEqual(diag["racket"]["geometry_rejections"], 1)

    def test_load_model_false_and_adopt_without_deepcopy(self):
        det = self._make_stub_detector(load_model=True)
        self.assertIsNone(det.model)
        self.assertEqual(det.input_width, 960)
        packet = {
            "frame_id": 5,
            "balls": [{"position": [500.0, 600.0], "box": [490.0, 590.0, 510.0, 610.0], "confidence": 0.9}],
            "rackets": [{"box": [480, 550, 520, 620], "confidence": 0.8}],
            "raw_legacy": True,
            "parse_diagnostics": {"ball": {"class_candidates": 1}, "racket": {"class_candidates": 1}},
            "racket_diagnostics": {},
            "dimensions": (1440, 2560, 2560, 1440),
        }
        balls, rackets = det.adopt_external_packet(
            packet, frame_id=5, return_rackets=True, copy_diagnostics=False
        )
        self.assertEqual(len(balls), 1)
        self.assertEqual(len(rackets), 1)
        self.assertIs(det.last_parse_diagnostics, packet["parse_diagnostics"])

    def test_stitched_buffer_fill_cached_by_geometry(self):
        est = object.__new__(PoseEstimatorYOLO26)
        est.input_size = (640, 640)
        est.confidence_threshold = 0.4
        est.keypoint_confidence = 0.3
        est.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]
        _TrackingBuffer.fill_calls = 0
        est._stitched_rgb_buf = np.zeros((640, 640, 3), dtype=np.uint8).view(_TrackingBuffer)
        est._stitched_geom = None

        out = np.zeros((1, 10, 6 + 17 * 3), dtype=np.float32)
        out[0, 3, 0:5] = [50.0, 100.0, 150.0, 300.0, 0.85]
        out[0, 3, 6:9] = [100.0, 120.0, 0.9]
        est.model = SimpleNamespace(predict=lambda _inp: {"out": out})

        front = np.zeros((400, 300, 3), dtype=np.uint8)
        back = np.zeros((400, 300, 3), dtype=np.uint8)
        f1, b1 = est.get_keypoints_from_stitched_views(front, (300, 400), back, (300, 400))
        self.assertEqual(_TrackingBuffer.fill_calls, 1)
        self.assertEqual(len(f1), 1)
        self.assertEqual(len(b1), 0)
        self.assertIn("nose", f1[0])

        # Same dimensions: must NOT re-fill buffer
        f2, _ = est.get_keypoints_from_stitched_views(front, (300, 400), back, (300, 400))
        self.assertEqual(_TrackingBuffer.fill_calls, 1)
        self.assertEqual(len(f2), 1)

        # Changed dimensions: must re-fill buffer
        front_resized = np.zeros((420, 300, 3), dtype=np.uint8)
        est.get_keypoints_from_stitched_views(front_resized, (300, 420), back, (300, 400))
        self.assertEqual(_TrackingBuffer.fill_calls, 2)

    def test_det_worker_pool_reorders_out_of_order_and_propagates_errors(self):
        pool = object.__new__(DetWorkerPool)
        pool._out_q = queue.Queue()
        pool._ready = {}
        pool._procs = [SimpleNamespace(is_alive=lambda: True, exitcode=0)]

        pool._out_q.put(("ok", 2, {"frame_id": 2}, 4.5, 200))
        pool._out_q.put(("ok", 0, {"frame_id": 0}, 5.1, 100))
        pool._out_q.put(("ok", 1, {"frame_id": 1}, 4.9, 150))

        p0, ms0, ts0 = pool.get_result(0)
        p1, ms1, ts1 = pool.get_result(1)
        p2, ms2, ts2 = pool.get_result(2)
        self.assertEqual((p0["frame_id"], p1["frame_id"], p2["frame_id"]), (0, 1, 2))
        self.assertAlmostEqual((ms0 + ms1 + ms2), 14.5)
        self.assertEqual((ts0, ts1, ts2), (100, 150, 200))

        pool._out_q.put(("error", 3, "boom"))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            pool.get_result(3)

    def test_launch_and_wait_ready_are_idempotent_and_propagate_errors(self):
        started = []

        class FakeProc:
            def __init__(self, target=None, args=(), name=None):
                self.name = name
                self.alive = False

            def start(self):
                self.alive = True
                started.append(self.name)

            def is_alive(self):
                return self.alive

            def join(self, timeout=None):
                self.alive = False

        pool = object.__new__(DetWorkerPool)
        pool.num_workers = 2
        pool._ctx = SimpleNamespace(Process=FakeProc)
        pool._in_q = queue.Queue()
        pool._out_q = queue.Queue()
        pool._ready_q = queue.Queue()
        pool._ready = {}
        pool._procs = []
        pool._launched = False
        pool._closed = False
        pool._worker_args = ({}, None, (640, 480), True, ["s0"], False, True, None)

        pool.launch()
        pool.launch()  # idempotent second call
        self.assertEqual(started, ["tennis-det-0", "tennis-det-1"])

        pool._ready_q.put(("ready", 0))
        pool._ready_q.put(("ready", 1))
        pool.wait_ready(timeout=2.0)
        pool.close()

        # Error from a worker aborts wait_ready cleanly
        pool_err = object.__new__(DetWorkerPool)
        pool_err.num_workers = 1
        pool_err._ctx = SimpleNamespace(Process=FakeProc)
        pool_err._in_q = queue.Queue()
        pool_err._out_q = queue.Queue()
        pool_err._ready_q = queue.Queue()
        pool_err._ready = {}
        pool_err._procs = []
        pool_err._launched = False
        pool_err._closed = False
        pool_err._worker_args = ({}, None, (640, 480), True, ["s0"], False, True, None)
        pool_err._ready_q.put(("error", 0, "init_boom"))
        with self.assertRaisesRegex(RuntimeError, "init_boom"):
            pool_err.start(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
