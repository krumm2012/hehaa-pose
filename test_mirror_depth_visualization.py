"""
test_mirror_depth_visualization.py
───────────────────────────────────
算法 2.0 镜面深度立体空间几何与光路可视化单元测试套件：
涵盖物理空间几何解算、SVG 矢量渲染、报告卡片生成与边界防御校验。
"""

import unittest
from pathlib import Path

from mirror_depth_visualization import (
    compute_mirror_depth_geometry,
    render_mirror_depth_isometric_svg,
    render_mirror_depth_card_html,
)


class MirrorDepthVisualizationTests(unittest.TestCase):
    def test_compute_standard_court02_geometry(self):
        """测试 Court 02 标称实测物理参数的光学解算。"""
        res = compute_mirror_depth_geometry(
            player_y=4.68,
            wall_y=6.20,
            cam_h=3.30,
            cam_pitch_deg=35.0,
            player_target_h=1.20,
        )
        self.assertEqual(res["player_y"], 4.68)
        self.assertEqual(res["wall_y"], 6.20)
        self.assertEqual(res["virtual_y"], 7.72)
        self.assertEqual(res["delta_y"], 3.04)
        self.assertEqual(res["cam_h"], 3.30)
        self.assertEqual(res["cam_pitch_deg"], 35.0)
        self.assertAlmostEqual(res["scale_x"], 7.72 / 4.68, places=2)
        self.assertGreater(res["scale_x"], res["scale_y"])
        # 各向异性比率约为 1.30
        self.assertAlmostEqual(res["anisotropy_ratio"], 1.30, places=1)
        # 反射点位于合理镜面高度 (0.5m ~ 2.5m)
        self.assertGreater(res["reflect_z"], 1.0)
        self.assertLess(res["reflect_z"], 2.5)

    def test_compute_boundary_and_degenerate_protection(self):
        """测试异常与贴近边界情况下的物理防崩与安全裁剪。"""
        # 选手站位超过或贴近平镜
        res = compute_mirror_depth_geometry(player_y=7.0, wall_y=6.0)
        self.assertLess(res["player_y"], res["wall_y"])
        self.assertGreater(res["virtual_y"], res["player_y"])
        self.assertGreater(res["delta_y"], 0)

        # 极端相机俯角与极端高度
        res_steep = compute_mirror_depth_geometry(cam_pitch_deg=89.0, cam_h=0.5)
        self.assertGreater(res_steep["cam_h"], 1.0)
        self.assertLessEqual(res_steep["cam_pitch_deg"], 80.0)

    def test_render_mirror_depth_isometric_svg(self):
        """测试 2.5D Isometric SVG 矢量代码生成完整性。"""
        geom = compute_mirror_depth_geometry(player_y=4.68, wall_y=6.20)
        svg = render_mirror_depth_isometric_svg(geom, width=860, height=380, show_hud=True)

        self.assertIn("<svg", svg)
        self.assertIn("</svg>", svg)
        self.assertIn("viewBox=\"0 0 860 380\"", svg)
        self.assertIn("optical-hud-panel", svg)
        self.assertIn("ray-direct", svg)
        self.assertIn("ray-incident", svg)
        self.assertIn("ray-reflected", svg)
        self.assertIn("player-real", svg)
        self.assertIn("player-virtual", svg)
        self.assertIn("camera-rig", svg)
        self.assertIn("7.72", svg)  # 虚像深度包含在文本或标签中

    def test_render_mirror_depth_card_html(self):
        """测试 HTML 报告卡片结构。"""
        card_html = render_mirror_depth_card_html(
            measured_relative_z=1.652,
            ground_calibration={"views": {"front": {"H": []}, "back": {"H": []}}},
            card_id="test-mirror-card",
        )
        self.assertIn("mirror-depth-card", card_html)
        self.assertIn("id=\"test-mirror-card\"", card_html)
        self.assertIn("室内光路立体几何与镜面深度剖面", card_html)
        self.assertIn("×1.652", card_html)
        self.assertIn("双视角地面单应性网格", card_html)

    def test_standalone_report_incorporates_mirror_depth(self):
        """测试在具有双视角/镜面标定的 payload 中，standalone 报告能成功集成卡片。"""
        from report_rendering import render_standalone_report_html

        payload = {
            "paths": {"video": "/tmp/test.mp4", "frame_json": "events.json"},
            "events": [
                {
                    "event_id": 1,
                    "stroke_type": "Forehand",
                    "start_frame": 10,
                    "contact_frame": 25,
                    "peak_frame": 30,
                    "end_frame": 45,
                    "confidence": 0.9,
                    "dual_view": {"is_valid": True},
                    "relative_depth_z": 1.648,
                }
            ],
            "ground_calibration": {"views": {"front": {"H": []}, "back": {"H": []}}},
            "summary": {"swing_event_count": 1, "frames": 100},
            "timeline": {},
        }

        html_out = render_standalone_report_html(payload, "/tmp/test_report.html")
        self.assertIn("session-mirror-depth-card", html_out)
        self.assertIn("室内光路立体几何与镜面深度剖面", html_out)


if __name__ == "__main__":
    unittest.main()
