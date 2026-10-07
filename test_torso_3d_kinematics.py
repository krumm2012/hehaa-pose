import math
import unittest
import numpy as np

from mirror_geometry import Torso3DKinematics, estimate_torso_3d_kinematics
from dual_view_biomechanics import DualViewBiomechanicsEngine, Keypoint


class TestTorso3DKinematics(unittest.TestCase):
    """测试基于平面反射光学模型的 3D 躯干绝对旋转与立体力学解算。"""

    def setUp(self):
        self.engine = DualViewBiomechanicsEngine()

    def test_front_facing_torso_yaw(self):
        """测试正对相机的站姿 (Yaw ~ 0度)。"""
        # 正面: 双肩展开宽约 100px, y 坐标水平
        front_pose = {
            "left_shoulder": (100.0, 200.0, 0.9),
            "right_shoulder": (200.0, 200.0, 0.9),
            "left_hip": (110.0, 350.0, 0.9),
            "right_hip": (190.0, 350.0, 0.9),
        }
        # 背面镜中: 虚像深度更远，正常缩放后宽度与正面一致
        back_pose = {
            "left_shoulder": (120.0, 150.0, 0.9),
            "right_shoulder": (180.6, 150.0, 0.9),  # 60.6 * 1.65 ≈ 100px
            "left_hip": (125.0, 240.0, 0.9),
            "right_hip": (173.5, 240.0, 0.9),
        }

        kin = estimate_torso_3d_kinematics(front_pose, back_pose)
        self.assertEqual(kin.status, "optimal")
        self.assertIsNotNone(kin.shoulder_yaw_deg)
        # 正对相机，转角应接近 0 度 (允许 ±10度误差)
        self.assertAlmostEqual(kin.shoulder_yaw_deg, 0.0, delta=10.0)
        self.assertAlmostEqual(kin.shoulder_roll_deg, 0.0, delta=5.0)
        self.assertIsNotNone(kin.hip_yaw_deg)
        self.assertAlmostEqual(kin.hip_yaw_deg, 0.0, delta=10.0)
        self.assertIsNotNone(kin.x_factor_3d_deg)
        self.assertLess(kin.x_factor_3d_deg, 15.0)

    def test_side_on_preparation_turn(self):
        """测试正手引拍 90 度完全侧身转肩姿态 (Yaw ~ 90度)。"""
        # 正面: 侧身后投影肩宽急剧变窄 (右肩拉到后方与左肩几乎重叠在 X 轴)
        front_pose = {
            "left_shoulder": (150.0, 200.0, 0.9),
            "right_shoulder": (155.0, 200.0, 0.9),  # 仅 5px 宽
            "left_hip": (130.0, 350.0, 0.9),
            "right_hip": (170.0, 350.0, 0.9),       # 髋部转角较小，40px 宽
        }
        # 背面镜中: 右肩靠近后墙镜面，在镜中呈现极大横向展开
        back_pose = {
            "left_shoulder": (110.0, 150.0, 0.9),
            "right_shoulder": (175.0, 150.0, 0.9),  # 65px 在镜中
            "left_hip": (120.0, 240.0, 0.9),
            "right_hip": (160.0, 240.0, 0.9),
        }

        kin = estimate_torso_3d_kinematics(front_pose, back_pose)
        self.assertEqual(kin.status, "optimal")
        self.assertIsNotNone(kin.shoulder_yaw_deg)
        # 侧身引拍，转角应在 75° ~ 105° 之间
        self.assertGreaterEqual(kin.shoulder_yaw_deg, 75.0)
        self.assertLessEqual(kin.shoulder_yaw_deg, 105.0)

        # 髋部转角应明显小于肩部转角，形成显著的 3D X-Factor
        self.assertIsNotNone(kin.hip_yaw_deg)
        self.assertLess(kin.hip_yaw_deg, kin.shoulder_yaw_deg)
        self.assertGreater(kin.x_factor_3d_deg, 20.0)

    def test_shoulder_roll_tilt(self):
        """测试右肩下沉/抬高倾斜角。"""
        front_pose = {
            "left_shoulder": (100.0, 180.0, 0.9),
            "right_shoulder": (200.0, 220.0, 0.9),  # 右肩下沉 40px
            "left_hip": (110.0, 350.0, 0.9),
            "right_hip": (190.0, 350.0, 0.9),
        }
        back_pose = {
            "left_shoulder": (120.0, 150.0, 0.9),
            "right_shoulder": (180.0, 150.0, 0.9),
            "left_hip": (125.0, 240.0, 0.9),
            "right_hip": (175.0, 240.0, 0.9),
        }
        kin = estimate_torso_3d_kinematics(front_pose, back_pose)
        self.assertIsNotNone(kin.shoulder_roll_deg)
        # atan2(40, 100) ≈ 21.8 度
        self.assertAlmostEqual(kin.shoulder_roll_deg, 21.8, delta=2.0)

    def test_fallback_when_back_view_missing(self):
        """测试背面机位缺失时的平滑降级 (fallback 模式)。"""
        front_pose = {
            "left_shoulder": (100.0, 200.0, 0.9),
            "right_shoulder": (200.0, 200.0, 0.9),
            "left_hip": (110.0, 350.0, 0.9),
            "right_hip": (190.0, 350.0, 0.9),
        }
        back_pose = {}  # 镜面遮挡全空

        kin = estimate_torso_3d_kinematics(front_pose, back_pose)
        self.assertEqual(kin.status, "invalid")

        # 若背面只给低置信度噪点
        back_low_conf = {
            "left_shoulder": (120.0, 150.0, 0.05),
            "right_shoulder": (180.0, 150.0, 0.05),
        }
        kin2 = estimate_torso_3d_kinematics(front_pose, back_low_conf)
        self.assertEqual(kin2.status, "fallback")
        self.assertIsNotNone(kin2.shoulder_yaw_deg)

    def test_dual_view_engine_integration(self):
        """测试 DualViewBiomechanicsEngine 全量集成输出 3D 动力学字段。"""
        front_pose = {
            "left_shoulder": [100.0, 200.0, 0.9],
            "right_shoulder": [200.0, 200.0, 0.9],
            "left_hip": [110.0, 350.0, 0.9],
            "right_hip": [190.0, 350.0, 0.9],
            "left_wrist": [115.0, 230.0, 0.8],
            "right_wrist": [195.0, 230.0, 0.8],
        }
        back_pose = {
            "left_shoulder": [120.0, 150.0, 0.9],
            "right_shoulder": [180.0, 150.0, 0.9],
            "left_hip": [125.0, 240.0, 0.9],
            "right_hip": [175.0, 240.0, 0.9],
            "right_wrist": [190.0, 210.0, 0.85],
        }
        res = self.engine.calculate_dual_biomechanics(front_pose, back_pose)
        self.assertIsNotNone(res.shoulder_yaw_3d_deg)
        self.assertIsNotNone(res.shoulder_roll_3d_deg)
        self.assertIsNotNone(res.hip_yaw_3d_deg)
        self.assertIsNotNone(res.x_factor_3d_deg)
        self.assertIsNotNone(res.relative_depth_z)


if __name__ == "__main__":
    unittest.main()
