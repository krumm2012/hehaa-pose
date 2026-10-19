import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from racket_pose_detector import RacketPoseDetector, decode_candidates, letterbox, pose_config, verify_acceptance
from racket_pose_runtime import accepted_runtime_config


class RacketResolutionTests(unittest.TestCase):
    def test_640_decode_restores_full_source_coordinates_and_confidence(self):
        config = pose_config({'mode': 'shadow', 'imgsz': 640, 'compute_units': 'ALL'})
        rgb, transform = letterbox(np.zeros((1440, 2560, 3), np.uint8), 640)
        self.assertEqual(rgb.shape, (640, 640, 3))
        self.assertEqual((transform['gain'], transform['top']), (.25, 140))
        raw = np.zeros((1, 26, 8400), np.float32)
        raw[0, :5, 0] = [320, 320, 80, 80, .9]
        raw[0, 5:, 0] = np.tile([320, 320, .9], 7)
        rows, diag = decode_candidates(raw, transform, config)
        self.assertEqual(diag['candidates'], 8400)
        self.assertEqual(diag['nonfinite_candidates'], 0)
        np.testing.assert_allclose(rows[0]['box'], [1120, 560, 1440, 880])
        self.assertEqual(len(rows[0]['keypoints']), 7)
        for p in rows[0]['keypoints'].values():
            self.assertEqual((p['x'], p['y'], p['observed']), (1280., 720., True))

    def test_resolution_output_mismatch_is_rejected_in_both_directions(self):
        for size, count in ((640, 18900), (960, 8400)):
            config = pose_config({'mode': 'shadow', 'imgsz': size})
            _, transform = letterbox(np.zeros((1440, 2560, 3), np.uint8), size)
            with self.assertRaisesRegex(ValueError, 'output shape'):
                decode_candidates(np.zeros((1, 26, count), np.float32), transform, config)

    def test_any_nonfinite_640_candidate_blocks_measurement(self):
        config = pose_config({'mode': 'shadow', 'imgsz': 640})
        _, transform = letterbox(np.zeros((640, 640, 3), np.uint8), 640)
        raw = np.zeros((1, 26, 8400), np.float32)
        raw[0, :5, 0] = [320, 320, 100, 100, .9]
        raw[0, 5:, 0] = np.tile([320, 320, .9], 7)
        raw[0, 0, 1] = np.nan
        rows, diag = decode_candidates(raw, transform, config)
        self.assertEqual(diag['nonfinite_candidates'], 1)
        self.assertFalse(rows[0]['observed'])
        self.assertFalse(rows[0]['measurement_eligible'])

    def test_640_cannot_borrow_a_960_acceptance_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'receipt.json'
            report = {'model_sha256': 'test', 'backend': 'coreml', 'compute_units': 'ALL',
                      'numerical_valid': True, 'quality_gate_passed': True}
            path.write_text(json.dumps(report))
            config = pose_config({'mode': 'active', 'imgsz': 640, 'compute_units': 'ALL', 'acceptance_path': str(path)})
            with self.assertRaises(ValueError):
                verify_acceptance(config, 'test')
            report['imgsz'] = 640
            path.write_text(json.dumps(report))
            verify_acceptance(config, 'test')
            report['quality_gate_passed'] = False
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                verify_acceptance(config, 'test')

    def test_640_still_processes_every_frame_in_continuous_modes(self):
        for mode in ('buffered', 'offline'):
            config = accepted_runtime_config({'mode': 'shadow', 'imgsz': 640, 'inference_stride': 3}, mode)
            self.assertEqual(config['inference_stride'], 1)
            self.assertEqual(config['imgsz'], 640)

    def test_loaded_package_must_match_configured_input_and_output(self):
        for package_size, count, error in ((960, 18900, 'configured RGB'), (640, 18900, 'output')):
            image = SimpleNamespace(name='image', type=SimpleNamespace(imageType=SimpleNamespace(
                width=package_size, height=package_size, colorSpace=20)))
            output = SimpleNamespace(name='output', type=SimpleNamespace(multiArrayType=SimpleNamespace(shape=[1, 26, count])))
            model = SimpleNamespace(get_spec=lambda: SimpleNamespace(description=SimpleNamespace(input=[image], output=[output])))
            coreml = SimpleNamespace(ComputeUnit=SimpleNamespace(CPU_AND_NE=1, ALL=2, CPU_ONLY=3),
                                     models=SimpleNamespace(MLModel=lambda *a, **k: model),
                                     proto=SimpleNamespace(FeatureTypes_pb2=SimpleNamespace(ImageFeatureType=SimpleNamespace(RGB=20))))
            with tempfile.TemporaryDirectory() as d:
                package = Path(d) / 'dummy'; package.write_bytes(b'package identity')
                with patch.dict('sys.modules', {'coremltools': coreml}):
                    with self.assertRaisesRegex(ValueError, error):
                        RacketPoseDetector({'mode': 'shadow', 'model_path': str(package), 'imgsz': 640})

    def test_benchmark_cache_cannot_relabel_960_results_as_640(self):
        from scripts.validate_racket_pose_pipeline import run_case
        with tempfile.TemporaryDirectory() as d:
            session = Path(d) / 'buffered-shadow-s1-49.43-r0'; session.mkdir()
            cached = {'roi_policy': 'project', 'roi_stream_id': 'court02-main',
                      'sidecar': {'config': {'imgsz': 960}}}
            (session / 'measurement.json').write_text(json.dumps(cached))
            args = SimpleNamespace(output=d, roi_policy='project', roi_stream_id='court02-main', imgsz=640)
            with self.assertRaisesRegex(ValueError, 'resolution differs'):
                run_case(args, Path('49.43.mp4'), 'shadow', 'buffered', 1, 0)

    def test_sidecar_applies_explicit_opencv_threads_before_model_loading(self):
        import cv2
        import queue
        from racket_pose_runtime import _worker
        previous = cv2.getNumThreads()
        seen = []
        def factory(config):
            seen.append(cv2.getNumThreads())
            return SimpleNamespace(digest='test', detect=lambda *a: {'status': 'observed'})
        incoming, outgoing = queue.Queue(), queue.Queue()
        incoming.put(None)
        try:
            cv2.setNumThreads(4)
            with patch('racket_pose_detector.RacketPoseDetector', side_effect=factory), patch('signal.signal'):
                _worker({'mode': 'shadow', 'imgsz': 640, 'opencv_threads': 0}, incoming, outgoing, None)
            self.assertEqual(seen, [1])
            self.assertEqual(outgoing.get_nowait()['kind'], 'ready')
        finally:
            cv2.setNumThreads(previous)


if __name__ == '__main__':
    unittest.main()
