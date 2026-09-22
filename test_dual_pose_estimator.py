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

        mgr = DualViewManager()
        estimator = DualPoseEstimator(backend="auto")

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


if __name__ == "__main__":
    unittest.main()
