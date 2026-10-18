"""Regression checks at the actual multi-model construction boundary."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from dual_pose_estimator import DualPoseEstimator
from pipeline_startup import warm_coreml_wrappers


class DualPoseComputeUnitTests(unittest.TestCase):
    def test_both_coreml_instances_receive_requested_compute_units(self):
        calls=[]
        def fake(path, config):
            calls.append(dict(config))
            return SimpleNamespace()
        with patch.dict(sys.modules, {'pose_estimator_yolo26':SimpleNamespace(PoseEstimatorYOLO26=fake)}), patch('dual_pose_estimator.Path.exists',return_value=True):
            estimator=DualPoseEstimator(model_path='fixture.mlpackage',backend='coreml',compute_units='ANE')
            self.addCleanup(estimator.close)
        self.assertEqual([c['pose_compute_units'] for c in calls],['ANE','ANE'])

    def test_default_keeps_existing_all_policy(self):
        calls=[]
        with patch.dict(sys.modules, {'pose_estimator_yolo26':SimpleNamespace(PoseEstimatorYOLO26=lambda p,c:calls.append(dict(c)) or SimpleNamespace())}), patch('dual_pose_estimator.Path.exists',return_value=True):
            estimator=DualPoseEstimator(model_path='fixture.mlpackage',backend='coreml')
            self.addCleanup(estimator.close)
        self.assertEqual([c['pose_compute_units'] for c in calls],['ALL','ALL'])

    def test_warmup_only_calls_coreml_predict_once_per_model(self):
        from unittest.mock import Mock
        desc=SimpleNamespace(name='image',type=SimpleNamespace(
            WhichOneof=lambda _: 'imageType',imageType=SimpleNamespace(width=960,height=960)))
        model=Mock();model.get_spec.return_value=SimpleNamespace(description=SimpleNamespace(input=[desc]))
        wrapper=Mock();wrapper.model=model
        self.assertEqual(len(warm_coreml_wrappers([wrapper,wrapper])),1)
        model.predict.assert_called_once()
        self.assertEqual(model.predict.call_args.args[0]['image'].size,(960,960))
        wrapper.detect.assert_not_called()

    def test_continuous_mode_overrides_legacy_drop_flags_at_construction(self):
        import cv2
        from unittest.mock import Mock
        from main_pipe import MultiprocessPipeline
        cap=Mock();cap.isOpened.return_value=True
        cap.get.side_effect=lambda key: {cv2.CAP_PROP_FRAME_WIDTH:64,cv2.CAP_PROP_FRAME_HEIGHT:48,cv2.CAP_PROP_FPS:25}.get(key,0)
        with patch.object(MultiprocessPipeline,'_open_capture',return_value=cap):
            pipeline=MultiprocessPipeline(config_path='configs/yolo26_tennis_config.yaml',
                processing_mode='buffered',live_mode=True,drop_stale_frames=True,no_save_video=True)
        try:
            self.assertFalse(pipeline.drop_stale_frames)
            self.assertTrue(pipeline.lossless_realtime)
            self.assertEqual(pipeline.processing_mode,'buffered')
            self.assertEqual(pipeline.q_inference._maxsize,pipeline.shm_num)
            self.assertEqual(pipeline.racket_pose_config['mode'],'legacy')
        finally:
            for q in (pipeline.q_free,pipeline.q_inference,pipeline.q_analyzer,pipeline.q_failures):
                q.cancel_join_thread();q.close()
