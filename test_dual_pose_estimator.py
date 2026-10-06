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
        self.assertFalse(estimator._has_back_orientation_anchor)

    def test_backview_temporal_whole_body_swap_filter(self):
        """验证时序二分图滤波器成功拦截并纠正类似 Frame 223 的全身左右颠倒。"""
        estimator = DualPoseEstimator(backend="mock")
        from dual_view_biomechanics import Keypoint

        # 帧 222 (基准正常帧，人体解剖左侧在局部画面右侧较大坐标)
        f222 = {
            "left_shoulder": Keypoint(x=280.0, y=100.0, conf=0.95),
            "right_shoulder": Keypoint(x=200.0, y=95.0, conf=0.95),
            "left_elbow": Keypoint(x=290.0, y=150.0, conf=0.92),
            "right_elbow": Keypoint(x=190.0, y=145.0, conf=0.92),
            "left_wrist": Keypoint(x=300.0, y=120.0, conf=0.70),
            "right_wrist": Keypoint(x=210.0, y=130.0, conf=0.65),
            "left_hip": Keypoint(x=260.0, y=200.0, conf=0.95),
            "right_hip": Keypoint(x=220.0, y=195.0, conf=0.95),
            "left_knee": Keypoint(x=270.0, y=280.0, conf=0.95),
            "right_knee": Keypoint(x=210.0, y=280.0, conf=0.95),
            "left_ankle": Keypoint(x=280.0, y=360.0, conf=0.95),
            "right_ankle": Keypoint(x=190.0, y=365.0, conf=0.95),
        }
        estimator._last_valid_back_pose = f222

        # 帧 223 注入全身 180° 翻转（左右标签互换，坐标颠倒）
        f223_flipped = {
            "left_shoulder": Keypoint(x=202.0, y=96.0, conf=0.95),
            "right_shoulder": Keypoint(x=278.0, y=101.0, conf=0.95),
            "left_elbow": Keypoint(x=188.0, y=147.0, conf=0.92),
            "right_elbow": Keypoint(x=288.0, y=152.0, conf=0.92),
            "left_wrist": Keypoint(x=208.0, y=132.0, conf=0.65),
            "right_wrist": Keypoint(x=298.0, y=122.0, conf=0.70),
            "left_hip": Keypoint(x=218.0, y=197.0, conf=0.95),
            "right_hip": Keypoint(x=258.0, y=202.0, conf=0.95),
            "left_knee": Keypoint(x=208.0, y=282.0, conf=0.95),
            "right_knee": Keypoint(x=268.0, y=282.0, conf=0.95),
            "left_ankle": Keypoint(x=188.0, y=367.0, conf=0.95),
            "right_ankle": Keypoint(x=278.0, y=362.0, conf=0.95),
        }

        action = estimator._filter_backview_temporal_swap(f223_flipped)
        self.assertEqual(action, "whole_body")
        # 验证修正后：左肩恢复在 x~278 附近，右肩恢复在 x~202 附近
        self.assertAlmostEqual(f223_flipped["left_shoulder"].x, 278.0, delta=1.0)
        self.assertAlmostEqual(f223_flipped["right_shoulder"].x, 202.0, delta=1.0)
        self.assertAlmostEqual(f223_flipped["left_ankle"].x, 278.0, delta=1.0)
        self.assertAlmostEqual(f223_flipped["right_ankle"].x, 188.0, delta=1.0)

        # 帧 224 (模型自然恢复正常)
        f224_normal = {
            "left_shoulder": Keypoint(x=279.0, y=102.0, conf=0.95),
            "right_shoulder": Keypoint(x=201.0, y=97.0, conf=0.95),
            "left_elbow": Keypoint(x=289.0, y=153.0, conf=0.92),
            "right_elbow": Keypoint(x=189.0, y=148.0, conf=0.92),
            "left_hip": Keypoint(x=259.0, y=203.0, conf=0.95),
            "right_hip": Keypoint(x=219.0, y=198.0, conf=0.95),
            "left_knee": Keypoint(x=269.0, y=283.0, conf=0.95),
            "right_knee": Keypoint(x=209.0, y=283.0, conf=0.95),
            "left_ankle": Keypoint(x=279.0, y=368.0, conf=0.95),
            "right_ankle": Keypoint(x=189.0, y=363.0, conf=0.95),
        }
        # 以修正后的 f223 作为历史参考，断言 f224 不会被误判翻转
        estimator._last_valid_back_pose = f223_flipped
        action224 = estimator._filter_backview_temporal_swap(f224_normal)
        self.assertIsNone(action224)
        self.assertAlmostEqual(f224_normal["left_shoulder"].x, 279.0, delta=1.0)

    def test_backview_temporal_leg_swap_filter(self):
        """验证时序二分图滤波器仅对下肢单独翻转进行局部矫正，不误伤正常躯干。"""
        estimator = DualPoseEstimator(backend="mock")
        from dual_view_biomechanics import Keypoint

        f_prev = {
            "left_shoulder": Keypoint(x=280.0, y=100.0, conf=0.95),
            "right_shoulder": Keypoint(x=200.0, y=95.0, conf=0.95),
            "left_hip": Keypoint(x=260.0, y=200.0, conf=0.95),
            "right_hip": Keypoint(x=220.0, y=195.0, conf=0.95),
            "left_knee": Keypoint(x=270.0, y=280.0, conf=0.95),
            "right_knee": Keypoint(x=210.0, y=280.0, conf=0.95),
            "left_ankle": Keypoint(x=280.0, y=360.0, conf=0.95),
            "right_ankle": Keypoint(x=190.0, y=365.0, conf=0.95),
        }
        estimator._last_valid_back_pose = f_prev

        # 下一帧躯干正常微动，但双腿发生交叉误识别翻转
        f_curr = {
            "left_shoulder": Keypoint(x=278.0, y=101.0, conf=0.95),
            "right_shoulder": Keypoint(x=202.0, y=96.0, conf=0.95),
            "left_hip": Keypoint(x=258.0, y=201.0, conf=0.95),
            "right_hip": Keypoint(x=222.0, y=196.0, conf=0.95),
            "left_knee": Keypoint(x=209.0, y=281.0, conf=0.95),
            "right_knee": Keypoint(x=269.0, y=281.0, conf=0.95),
            "left_ankle": Keypoint(x=189.0, y=366.0, conf=0.95),
            "right_ankle": Keypoint(x=279.0, y=361.0, conf=0.95),
        }

        action = estimator._filter_backview_temporal_swap(f_curr)
        self.assertEqual(action, "legs")
        # 躯干未被改动
        self.assertAlmostEqual(f_curr["left_shoulder"].x, 278.0, delta=1.0)
        self.assertAlmostEqual(f_curr["right_shoulder"].x, 202.0, delta=1.0)
        # 双腿被成功矫正回正确位置
        self.assertAlmostEqual(f_curr["left_ankle"].x, 279.0, delta=1.0)
        self.assertAlmostEqual(f_curr["right_ankle"].x, 189.0, delta=1.0)

    def test_backview_natural_body_movement_no_false_swap(self):
        """测试正常人体奔跑平移及旋转不会被误判为翻转。"""
        estimator = DualPoseEstimator(backend="mock")
        from dual_view_biomechanics import Keypoint

        f_prev = {
            "left_shoulder": Keypoint(x=280.0, y=100.0, conf=0.95),
            "right_shoulder": Keypoint(x=200.0, y=95.0, conf=0.95),
            "left_hip": Keypoint(x=260.0, y=200.0, conf=0.95),
            "right_hip": Keypoint(x=220.0, y=195.0, conf=0.95),
            "left_knee": Keypoint(x=270.0, y=280.0, conf=0.95),
            "right_knee": Keypoint(x=210.0, y=280.0, conf=0.95),
            "left_ankle": Keypoint(x=280.0, y=360.0, conf=0.95),
            "right_ankle": Keypoint(x=190.0, y=365.0, conf=0.95),
        }
        estimator._last_valid_back_pose = f_prev

        # 选手整体向左平移 15px，并伴随转体（肩宽由 80px 缩减至 65px）
        f_curr = {
            "left_shoulder": Keypoint(x=260.0, y=100.0, conf=0.95),
            "right_shoulder": Keypoint(x=195.0, y=95.0, conf=0.95),
            "left_hip": Keypoint(x=245.0, y=200.0, conf=0.95),
            "right_hip": Keypoint(x=210.0, y=195.0, conf=0.95),
            "left_knee": Keypoint(x=255.0, y=280.0, conf=0.95),
            "right_knee": Keypoint(x=200.0, y=280.0, conf=0.95),
            "left_ankle": Keypoint(x=265.0, y=360.0, conf=0.95),
            "right_ankle": Keypoint(x=180.0, y=365.0, conf=0.95),
        }

        action = estimator._filter_backview_temporal_swap(f_curr)
        self.assertIsNone(action)
        self.assertAlmostEqual(f_curr["left_shoulder"].x, 260.0)
        self.assertAlmostEqual(f_curr["right_shoulder"].x, 195.0)

    def test_cross_view_cold_start_orientation_alignment(self):
        """测试冷启动时背面视口若初始颠倒，能被正面机位解剖矢量符号成功拉回对齐。"""
        estimator = DualPoseEstimator(backend="mock")
        from dual_view_biomechanics import Keypoint

        # 正面原图坐标系：左肩 x=1650, 右肩 x=1550 (dx = +100)
        front_orig = {
            "left_shoulder": Keypoint(x=1650.0, y=400.0, conf=0.98),
            "right_shoulder": Keypoint(x=1550.0, y=390.0, conf=0.98),
            "left_hip": Keypoint(x=1600.0, y=530.0, conf=0.98),
            "right_hip": Keypoint(x=1530.0, y=525.0, conf=0.98),
        }
        # 背面原图坐标系初始冷启动输出颠倒：左肩 x=1450, 右肩 x=1520 (dx = -70)
        back_orig = {
            "left_shoulder": Keypoint(x=1450.0, y=85.0, conf=0.95),
            "right_shoulder": Keypoint(x=1520.0, y=80.0, conf=0.95),
            "left_hip": Keypoint(x=1460.0, y=185.0, conf=0.95),
            "right_hip": Keypoint(x=1510.0, y=180.0, conf=0.95),
        }
        back_local = {
            "left_shoulder": Keypoint(x=200.0, y=90.0, conf=0.95),
            "right_shoulder": Keypoint(x=270.0, y=85.0, conf=0.95),
            "left_hip": Keypoint(x=210.0, y=190.0, conf=0.95),
            "right_hip": Keypoint(x=260.0, y=185.0, conf=0.95),
        }

        res = estimator._verify_cross_view_cold_start(front_orig, back_orig, back_local)
        self.assertEqual(res, "cold_start_whole_body")
        # 断言 back_orig 与 back_local 均被对调为正确朝向
        self.assertAlmostEqual(back_orig["left_shoulder"].x, 1520.0)
        self.assertAlmostEqual(back_orig["right_shoulder"].x, 1450.0)
        self.assertAlmostEqual(back_local["left_shoulder"].x, 270.0)
        self.assertAlmostEqual(back_local["right_shoulder"].x, 200.0)


if __name__ == "__main__":
    unittest.main()



