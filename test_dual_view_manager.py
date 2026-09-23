"""
test_dual_view_manager.py
─────────────────────────
单元测试：验证 DualViewManager 的机位解耦、0.45 padding 外扩、水平镜像翻转及双向坐标映射精度。
使用标准库 unittest 实现。
"""
from pathlib import Path
import unittest
import cv2
import numpy as np

from dual_view_manager import DualViewCropInfo, DualViewManager


class DualViewManagerTests(unittest.TestCase):
    def setUp(self):
        self.mgr = DualViewManager()

    def test_initialization(self):
        self.assertTrue(self.mgr.mirror_flip)
        self.assertEqual(self.mgr.front_bbox_padding, 0.45)
        self.assertGreaterEqual(len(self.mgr.mirror_polygon_norm), 4)

    def test_pad_and_clamp_bbox(self):
        fw, fh = 1000, 1000
        box = (100.0, 100.0, 200.0, 200.0)  # 100x100 box
        # padding 0.45 -> pad_x = 45, pad_y = 45
        px1, py1, px2, py2 = self.mgr._pad_and_clamp_bbox(box, 0.45, fw, fh)
        self.assertEqual(px1, 55)
        self.assertEqual(py1, 55)
        self.assertEqual(px2, 245)
        self.assertEqual(py2, 245)

        # Test clamping near image boundary
        edge_box = (10.0, 10.0, 100.0, 100.0)
        ex1, ey1, ex2, ey2 = self.mgr._pad_and_clamp_bbox(edge_box, 0.5, fw, fh)
        self.assertEqual(ex1, 0)
        self.assertEqual(ey1, 0)

    def test_split_frame_synthetic(self):
        # 2560x1440 synthetic frame
        dummy_frame = np.zeros((1440, 2560, 3), dtype=np.uint8)

        # Draw a marker in mirror region to verify flip
        cv2.circle(dummy_frame, (300, 300), 20, (0, 0, 255), -1)

        dual = self.mgr.split_frame(dummy_frame, player_bbox=(1000, 500, 1300, 1100), frame_id=1)

        self.assertEqual(dual.front_frame.shape, (720, 540, 3))
        self.assertEqual(dual.back_frame.shape, (720, 540, 3))
        self.assertFalse(dual.front_info.is_horizontally_flipped)
        self.assertTrue(dual.back_info.is_horizontally_flipped)

    def test_coordinate_mapping_front(self):
        dummy_frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        dual = self.mgr.split_frame(dummy_frame, player_bbox=(1000, 500, 1300, 1100))

        # Test round-trip mapping for points across front view
        test_pts = [(0.0, 0.0), (270.0, 360.0), (539.0, 719.0), (123.4, 567.8)]
        for vx, vy in test_pts:
            orig_x, orig_y = dual.front_info.map_to_original(vx, vy)
            recv_x, recv_y = dual.front_info.map_from_original(orig_x, orig_y)
            self.assertAlmostEqual(recv_x, vx, delta=1e-3, msg=f"Front X mismatch: {recv_x} vs {vx}")
            self.assertAlmostEqual(recv_y, vy, delta=1e-3, msg=f"Front Y mismatch: {recv_y} vs {vy}")

    def test_coordinate_mapping_back_with_flip(self):
        dummy_frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        dual = self.mgr.split_frame(dummy_frame)

        info = dual.back_info
        self.assertTrue(info.is_horizontally_flipped)

        # 1. Round-trip consistency
        test_pts = [(0.0, 0.0), (270.0, 360.0), (539.0, 719.0), (105.5, 432.1)]
        for vx, vy in test_pts:
            orig_x, orig_y = info.map_to_original(vx, vy)
            recv_x, recv_y = info.map_from_original(orig_x, orig_y)
            self.assertAlmostEqual(recv_x, vx, delta=1e-3, msg=f"Back X mismatch: {recv_x} vs {vx}")
            self.assertAlmostEqual(recv_y, vy, delta=1e-3, msg=f"Back Y mismatch: {recv_y} vs {vy}")

        # 2. Verify physical flip property:
        # A point at x_view = 0 (left edge of flipped view) must map to the right edge of original crop!
        x1, y1, x2, y2 = info.bbox_orig
        left_edge_orig_x, _ = info.map_to_original(0.0, 100.0)
        right_edge_orig_x, _ = info.map_to_original(info.view_size[0] - 1.0, 100.0)

        # Left in flipped view should be close to x2 (right in original)
        # Because 1 pixel in view space equals 1 / scale_x pixels in original space:
        max_px_delta = max(1.0, 1.0 / info.scale_x) + 0.1
        self.assertGreater(left_edge_orig_x, right_edge_orig_x)
        self.assertLess(abs(left_edge_orig_x - (x2 - 1.0)), max_px_delta)
        self.assertLess(abs(right_edge_orig_x - x1), max_px_delta)

    def test_render_side_by_side(self):
        dummy_frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        dual = self.mgr.split_frame(dummy_frame)
        sbs = self.mgr.render_side_by_side(dual, draw_labels=True)
        self.assertEqual(sbs.shape, (720, 1080, 3))  # 540 + 540 = 1080 width

    def test_mask_polygon_application(self):
        # Create a white dummy frame
        white_frame = np.ones((1440, 2560, 3), dtype=np.uint8) * 255
        
        # Test without mask
        self.mgr.mask_polygon_norm = None
        dual_no_mask = self.mgr.split_frame(white_frame)
        self.assertEqual(np.mean(dual_no_mask.back_frame), 255.0)

        # Test with mask covering reflection region
        # Mirror polygon in default config is roughly [0.08, 0.15] to [0.35, 0.85]
        self.mgr.mask_polygon_norm = [
            [0.10, 0.20],
            [0.30, 0.20],
            [0.30, 0.80],
            [0.10, 0.80],
        ]
        dual_masked = self.mgr.split_frame(white_frame)
        # Masked area should darken the back_frame pixels
        mean_masked = np.mean(dual_masked.back_frame)
        self.assertLess(mean_masked, 250.0)

        # Ensure front frame is untouched (remains pure white)
        self.assertEqual(np.mean(dual_masked.front_frame), 255.0)


if __name__ == "__main__":
    unittest.main()
