"""
test_algo2_three_tiers.py
─────────────────────────
针对第一、第二、第三梯队网球生物力学拓展指标与特写遥测卡片的系统单元测试。
"""
import unittest
import numpy as np

from dual_view_biomechanics import (
    DualViewBiomechanicsEngine,
    Keypoint,
)
from dual_view_renderer import DualViewRenderer
from dual_view_manager import DualViewFrame, DualViewCropInfo
from dual_pose_estimator import DualPoseResult
from swing_biomechanics import (
    _calculate_extended_tier_biomechanics,
    aggregate_event_biomechanics,
)
from swing_motion_features import extract_motion_features


class TestAlgo2ThreeTiers(unittest.TestCase):
    def setUp(self):
        self.engine = DualViewBiomechanicsEngine(dominant_hand="right")
        self.renderer = DualViewRenderer(show_hud=True, show_skeleton=True)

    def test_extended_tier_biomechanics_computation(self):
        """测试第一、第二、第三梯队拓展指标计算与评分体系。"""
        features_in_event = []
        for f in range(20):
            # 模拟引拍到随挥的过程
            # 掉拍头下潜至帧 8，击球点在帧 10
            racket_y = 500.0 + (50.0 if f == 8 else (0.0 if f < 8 else -20.0 * (f - 8)))
            racket_x = 300.0 + f * 10.0
            hip_y = 600.0 + (20.0 if f <= 8 else 0.0) # 下蹲到蓄力，击球时升起
            features_in_event.append({
                "frame_id": f,
                "has_pose": True,
                "racket_head_speed_kmh": 60.0 + f * 2.5 if f <= 10 else 75.0,
                "racket_center": (racket_x, racket_y),
                "hip_vertical_pos": hip_y,
                "stance_type": "Semi-Open Stance",
                "hip_rotation_speed": 100.0 if f == 7 else 50.0,
                "shoulder_rotation_speed": 150.0 if f == 9 else 60.0,
                "racket_speed": 200.0 if f == 10 else 80.0,
                "robust_shoulder_turn_deg": 38.0,
                "takeback_depth_ratio": 1.35,
                "arm_extension_deg": 155.0,
            })

        ext = _calculate_extended_tier_biomechanics(
            features_in_event=features_in_event,
            start_frame=0,
            contact_frame=10,
            end_frame=19,
            body_width=120.0,
            fps=25.0,
        )

        # Tier 1: 拍头动力学与刷球角
        self.assertIn("racket_head_speed", ext)
        self.assertGreater(ext["racket_head_speed"]["max_kmh"], 70.0)
        self.assertGreater(ext["racket_head_speed"]["contact_kmh"], 70.0)
        self.assertIn("brush_angle", ext)
        self.assertGreater(ext["brush_angle"]["low_to_high_angle_deg"], 0.0)
        self.assertIsNotNone(ext["brush_angle"]["drop_depth_ratio"])

        # Tier 2: 站位类型与蹬地发力与技术评分
        self.assertIn("stance", ext)
        self.assertEqual(ext["stance"]["stance_type"], "Semi-Open Stance")
        self.assertIn("leg_drive", ext)
        self.assertGreater(ext["leg_drive"]["drive_px"], 0.0)
        self.assertIn("swing_quality_score", ext)
        self.assertGreaterEqual(ext["swing_quality_score"]["overall_score"], 60.0)
        self.assertIn(ext["swing_quality_score"]["grade"], ["PRO", "ADVANCED", "INTERMEDIATE"])

        # Tier 3: 动力学链时序
        self.assertIn("kinematic_sequence", ext)
        seq = ext["kinematic_sequence"]
        self.assertEqual(seq["hip_peak_frame"], 7)
        self.assertEqual(seq["shoulder_peak_frame"], 9)
        self.assertEqual(seq["racket_peak_frame"], 10)
        self.assertTrue(seq["is_sequential"])
        self.assertEqual(seq["sequence_quality"], "OPTIMAL")

    def test_dual_view_relative_depth_z(self):
        """测试双机位前后尺度视差拟合的相对 3D 深度推算。"""
        # 正面双肩宽 140px，背面双肩宽 70px (视差比 = 2.0)
        front_pose = {
            "left_shoulder": Keypoint(100.0, 200.0, 0.9),
            "right_shoulder": Keypoint(240.0, 200.0, 0.9),
            "left_wrist": Keypoint(150.0, 300.0, 0.8),
            "right_wrist": Keypoint(260.0, 300.0, 0.8),
        }
        back_pose = {
            "left_shoulder": Keypoint(100.0, 200.0, 0.9),
            "right_shoulder": Keypoint(170.0, 200.0, 0.9),
            "right_wrist": Keypoint(180.0, 280.0, 0.8),
        }

        res = self.engine.calculate_dual_biomechanics(front_pose, back_pose)
        self.assertIsNotNone(res.relative_depth_z)
        self.assertAlmostEqual(res.relative_depth_z, 2.0, places=2)

    def test_impact_telemetry_card_render(self):
        """测试特写遥测卡片渲染与单帧双视角复合渲染。"""
        canvas = np.zeros((720, 1080, 3), dtype=np.uint8)
        card_data = {
            "stroke_type": "FOREHAND",
            "swing_score": 86.5,
            "swing_grade": "PRO",
            "racket_speed_kmh": 84.2,
            "racket_max_speed_kmh": 89.0,
            "brush_angle_deg": 38.5,
            "drop_depth_ratio": 0.45,
            "stance_type": "Semi-Open Stance",
            "leg_drive_ratio": 0.18,
            "kinematic_sequence_text": "腿➔髋➔肩➔拍 (OPTIMAL)",
        }

        rendered = self.renderer.draw_impact_telemetry_card(canvas, card_data)
        self.assertEqual(rendered.shape, (720, 1080, 3))
        # 确保中央卡片区域已被有效着色绘制（不是全黑）
        card_roi = rendered[100:200, 500:600]
        self.assertGreater(np.mean(card_roi), 5.0)

        # 测试 render_dual_frame 携带 telemetry_card 参数调用
        dummy_front = np.zeros((720, 540, 3), dtype=np.uint8)
        dummy_back = np.zeros((720, 540, 3), dtype=np.uint8)
        dummy_orig = np.zeros((1440, 2560, 3), dtype=np.uint8)
        crop_info_f = DualViewCropInfo(bbox_orig=(768, 432, 1536, 1152), crop_size=(768, 720), view_size=(540, 720))
        crop_info_b = DualViewCropInfo(bbox_orig=(256, 144, 1024, 576), crop_size=(768, 432), view_size=(540, 720), is_horizontally_flipped=True)
        dummy_dual_frame = DualViewFrame(
            frame_id=0,
            original_frame=dummy_orig,
            front_frame=dummy_front,
            back_frame=dummy_back,
            front_info=crop_info_f,
            back_info=crop_info_b,
        )
        bio_res = self.engine.calculate_dual_biomechanics(
            {"left_shoulder": Keypoint(100.0, 200.0, 0.9), "right_shoulder": Keypoint(200.0, 200.0, 0.9)},
            {"left_shoulder": Keypoint(100.0, 200.0, 0.9), "right_shoulder": Keypoint(200.0, 200.0, 0.9)},
        )
        pose_res = DualPoseResult(
            frame_id=0,
            front_pose_orig={},
            back_pose_orig={},
            front_pose_local={},
            back_pose_local={},
            fused_pose_local={},
            fused_pose_orig={},
            biomechanics=bio_res,
        )
        out = self.renderer.render_dual_frame(
            dummy_dual_frame,
            pose_res,
            telemetry_card=card_data,
        )
        self.assertEqual(out.shape, (720, 1080, 3))


if __name__ == "__main__":
    unittest.main()
