"""
test_dual_view_renderer.py
──────────────────────────
单元测试与端到端渲染验证：验证 DualViewRenderer 的骨骼绘制、HUD 叠加与双视角视频输出。
"""
from pathlib import Path
import unittest
import cv2
import numpy as np

from dual_pose_estimator import DualPoseEstimator
from dual_view_manager import DualViewManager
from dual_view_renderer import DualViewRenderer


class DualViewRendererTests(unittest.TestCase):
    def test_render_synthetic_dual_frame(self):
        mgr = DualViewManager()
        dummy_frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        dual_frame = mgr.split_frame(dummy_frame, frame_id=1)

        estimator = DualPoseEstimator(backend="mock")
        pose_res = estimator.estimate_dual_pose(dual_frame)

        renderer = DualViewRenderer(show_hud=True, show_skeleton=True)
        rendered = renderer.render_dual_frame(dual_frame, pose_res)

        self.assertEqual(rendered.shape, (720, 1080, 3))

    def test_end_to_end_real_video_clip(self):
        video_path = "/Users/krum5539/Desktop/Camera/49.35.mp4"
        if not Path(video_path).exists():
            self.skipTest("Sample video 49.35.mp4 not found")

        mgr = DualViewManager()
        estimator = DualPoseEstimator(backend="auto")
        renderer = DualViewRenderer(show_hud=True, show_skeleton=True)

        cap = cv2.VideoCapture(video_path)
        out_frames = []

        # 抽检连续 15 帧挥拍过程 (帧 15 ~ 30)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 15)
        for i in range(15):
            ret, frame = cap.read()
            self.assertTrue(ret)
            dual_frame = mgr.split_frame(frame, frame_id=15 + i)
            pose_res = estimator.estimate_dual_pose(dual_frame)
            rendered = renderer.render_dual_frame(dual_frame, pose_res)
            self.assertEqual(rendered.shape, (720, 1080, 3))
            out_frames.append(rendered)

        cap.release()
        self.assertEqual(len(out_frames), 15)


if __name__ == "__main__":
    unittest.main()
