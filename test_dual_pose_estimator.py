"""
test_dual_pose_estimator.py
───────────────────────────
单元测试：验证 DualPoseEstimator 的双视角姿态估计、坐标反向映射与生物力学聚合流程。
"""
from pathlib import Path
import unittest
import cv2
import numpy as np

from dual_pose_estimator import DualPoseEstimator
from dual_view_manager import DualViewManager


class DualPoseEstimatorTests(unittest.TestCase):
    def test_mock_backend_flow(self):
        estimator = DualPoseEstimator(backend="mock")
        self.assertEqual(estimator.backend, "mock")

        mgr = DualViewManager()
        dummy_frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        dual_frame = mgr.split_frame(dummy_frame, frame_id=42)

        res = estimator.estimate_dual_pose(dual_frame)
        self.assertEqual(res.frame_id, 42)
        self.assertIsNotNone(res.biomechanics)

    def test_real_video_dual_pose(self):
        video_path = "/Users/krum5539/Desktop/Camera/49.35.mp4"
        if not Path(video_path).exists():
            self.skipTest("Video 49.35.mp4 not found")

        mgr = DualViewManager(config_path=str(Path(__file__).parent / "tests/fixtures/dual_view_49_35.yaml"))
        estimator = DualPoseEstimator(backend="auto")
        self.addCleanup(estimator.close)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 20) # 准备挥拍帧
        ret, frame = cap.read()
        cap.release()
        self.assertTrue(ret)

        dual_frame = mgr.split_frame(frame, frame_id=20)
        res = estimator.estimate_dual_pose(dual_frame)

        # 检查是否成功在双视角检测到了关键点（如果模型可用）
        if estimator.backend != "mock":
            self.assertGreater(len(res.front_pose_local), 0, "Front pose keypoints should be detected")
            self.assertGreater(len(res.back_pose_local), 0, "Back pose keypoints should be detected")
            # 检查坐标反向映射是否在原图合理范围内
            for kp in res.front_pose_orig.values():
                self.assertGreaterEqual(kp.x, 0.0)
                self.assertLessEqual(kp.x, 2560.0)
                self.assertGreaterEqual(kp.y, 0.0)
                self.assertLessEqual(kp.y, 1440.0)

            for kp in res.back_pose_orig.values():
                self.assertGreaterEqual(kp.x, 0.0)
                self.assertLessEqual(kp.x, 2560.0)
                self.assertGreaterEqual(kp.y, 0.0)
                self.assertLessEqual(kp.y, 1440.0)


    def test_concurrent_configuration_and_instances(self):
        # 1. Mock backend
        est_mock = DualPoseEstimator(backend="mock", concurrent=True)
        self.assertTrue(est_mock.concurrent)
        self.assertIsNotNone(est_mock._pool)
        est_mock.close()
        self.assertIsNone(est_mock._pool)

        # 2. Non-concurrent mock
        est_seq = DualPoseEstimator(backend="mock", concurrent=False)
        self.assertFalse(est_seq.concurrent)
        self.assertIsNone(est_seq._pool)
        est_seq.close()

        # 3. Auto backend dual instances
        est_auto = DualPoseEstimator(backend="auto", concurrent=True)
        if est_auto.backend != "mock":
            self.assertIsNotNone(est_auto.model_front)
            self.assertIsNotNone(est_auto.model_back)
            self.assertIs(est_auto.model, est_auto.model_front)
        est_auto.close()
        self.assertIsNone(est_auto._pool)

    def test_concurrent_vs_sequential_pose_consistency(self):
        mgr = DualViewManager()
        dummy_frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        dual_frame = mgr.split_frame(dummy_frame, frame_id=10)

        est_conc = DualPoseEstimator(backend="auto", concurrent=True)
        res_conc = est_conc.estimate_dual_pose(dual_frame)
        est_conc.close()

        est_seq = DualPoseEstimator(backend="auto", concurrent=False)
        res_seq = est_seq.estimate_dual_pose(dual_frame)
        est_seq.close()

        self.assertEqual(res_conc.frame_id, res_seq.frame_id)
        self.assertEqual(len(res_conc.front_pose_local), len(res_seq.front_pose_local))
        self.assertEqual(len(res_conc.back_pose_local), len(res_seq.back_pose_local))

    def test_front_view_candidate_selection_and_mirror_filtering(self):
        estimator = DualPoseEstimator(backend="mock")
        # 构造两个候选人数据：
        # Person 0 (镜面虚影): y 处于顶部 [20, 280]，h=260
        # Person 1 (真实前景选手): y 处于下半部 [220, 600]，h=380
        cand_mirror = np.zeros((17, 3), dtype=np.float32)
        for i in range(17):
            cand_mirror[i] = [270.0, 20.0 + i * 15.0, 0.90]

        cand_real = np.zeros((17, 3), dtype=np.float32)
        for i in range(17):
            cand_real[i] = [270.0, 220.0 + i * 22.0, 0.90]

        kp_data = np.stack([cand_mirror, cand_real], axis=0)
        h, w = 720, 540

        # 模拟镜面检测函数 (将顶部判定为镜面)
        is_mirror_fn = lambda x, y: y <= 200.0

        class MockCropInfo:
            def map_to_original(self, x, y):
                return x, y

        chosen = estimator._select_front_candidate_ultralytics(
            kp_data, h, w, crop_info=MockCropInfo(), is_point_in_mirror_fn=is_mirror_fn
        )
        self.assertIsNotNone(chosen)
        # 应选择下半部的真实选手 Person 1
        self.assertGreater(chosen[0, 1], 200.0)

    def test_front_view_temporal_smoothing_and_reset(self):
        estimator = DualPoseEstimator(backend="mock")
        from dual_view_biomechanics import Keypoint
        estimator._last_valid_front_pose = {"nose": Keypoint(x=100.0, y=200.0, conf=0.8)}
        estimator._front_missing_count = 1
        self.assertEqual(len(estimator._last_valid_front_pose), 1)

        estimator.reset()
        self.assertEqual(len(estimator._last_valid_front_pose), 0)
        self.assertEqual(estimator._front_missing_count, 0)
        self.assertEqual(len(estimator._last_valid_back_pose), 0)
        self.assertEqual(estimator._back_missing_count, 0)


if __name__ == "__main__":
    unittest.main()


