"""
test_dual_view_biomechanics.py
──────────────────────────────
单元测试：验证 DualViewBiomechanicsEngine 的正反手分类、双手握拍检测、触球距离门控、姿态自愈与后背动力链指标。
使用标准 unittest。
"""
import unittest
from dual_view_biomechanics import (
    BACKHAND,
    FOREHAND,
    TWO_HANDED_BACKHAND,
    DualViewBiomechanicsEngine,
    Keypoint,
)


class DualViewBiomechanicsTests(unittest.TestCase):
    def setUp(self):
        self.engine = DualViewBiomechanicsEngine(dominant_hand="right", max_contact_distance=150.0)

    def test_forehand_classification(self):
        # 模拟右手正手击球动作：
        # 右肩位于 (100, 200)，左肩位于 (200, 200)，肩宽 = 100
        # 躯干中心 = (150, 200)
        # 右手腕位于 (60, 220)（在身体右侧，未跨越中线）
        pose = {
            "left_shoulder": (200.0, 200.0, 0.9),
            "right_shoulder": (100.0, 200.0, 0.9),
            "right_wrist": (60.0, 220.0, 0.85),
            "left_wrist": (220.0, 250.0, 0.8), # 左手自然放松
        }
        res = self.engine.classify_shot(self.engine.parse_pose_dict(pose), ball_pos=(55.0, 225.0))
        self.assertEqual(res.shot_type, FOREHAND)
        self.assertFalse(res.is_two_handed)
        self.assertTrue(res.is_valid_contact)
        self.assertLess(res.midline_side_projection, 0) # 位于右手侧

    def test_single_handed_backhand_classification(self):
        # 模拟单手反手击球动作：
        # 右肩 (100, 200)，左肩 (200, 200)
        # 右手腕跨越中线击球，位于 (240, 190)（跨越中线至左侧）
        pose = {
            "left_shoulder": (200.0, 200.0, 0.9),
            "right_shoulder": (100.0, 200.0, 0.9),
            "right_wrist": (240.0, 190.0, 0.85),
            "left_wrist": (120.0, 260.0, 0.8), # 左手向后平衡
        }
        res = self.engine.classify_shot(self.engine.parse_pose_dict(pose), ball_pos=(245.0, 195.0))
        self.assertEqual(res.shot_type, BACKHAND)
        self.assertFalse(res.is_two_handed)
        self.assertTrue(res.is_valid_contact)
        self.assertGreater(res.midline_side_projection, 0) # 跨越中线至左侧

    def test_two_handed_backhand_classification(self):
        # 模拟双手反拍击球动作 (Tennis-Vision 黄金判定)：
        # 肩宽 = 100
        # 左右手腕紧密靠近（握在拍柄上）：左手腕 (220, 200)，右手腕 (235, 205) -> gap = 15.8px < 45px
        pose = {
            "left_shoulder": (200.0, 200.0, 0.9),
            "right_shoulder": (100.0, 200.0, 0.9),
            "left_wrist": (220.0, 200.0, 0.9),
            "right_wrist": (235.0, 205.0, 0.9),
        }
        res = self.engine.classify_shot(self.engine.parse_pose_dict(pose), ball_pos=(240.0, 200.0))
        self.assertEqual(res.shot_type, TWO_HANDED_BACKHAND)
        self.assertTrue(res.is_two_handed)
        self.assertEqual(res.hitting_hand, "both")
        self.assertTrue(res.is_valid_contact)

    def test_contact_distance_validation_rejects_empty_swing(self):
        # 动作像正手，但球距离手腕超过 150px (空挥或假动作)
        pose = {
            "left_shoulder": (200.0, 200.0, 0.9),
            "right_shoulder": (100.0, 200.0, 0.9),
            "right_wrist": (60.0, 220.0, 0.9),
        }
        # 球在 (400, 500)，距离手腕 ~440px
        res = self.engine.classify_shot(self.engine.parse_pose_dict(pose), ball_pos=(400.0, 500.0))
        self.assertFalse(res.is_valid_contact)
        self.assertIn("exceeds_max", str(res.rejection_reason))

    def test_occlusion_healing(self):
        # 正面手腕被遮挡（置信度 0.1），背面镜中手腕清晰（置信度 0.88）
        front_pose = {
            "left_shoulder": (200.0, 200.0, 0.9),
            "right_shoulder": (100.0, 200.0, 0.9),
            "right_wrist": (60.0, 220.0, 0.1), # 低置信度遮挡
        }
        back_pose = {
            "left_shoulder": (200.0, 200.0, 0.85),
            "right_shoulder": (100.0, 200.0, 0.85),
            "right_wrist": (55.0, 215.0, 0.88), # 背面清晰可见
        }
        healed, healed_names = self.engine.heal_occluded_pose(
            self.engine.parse_pose_dict(front_pose),
            self.engine.parse_pose_dict(back_pose),
        )
        self.assertIn("right_wrist", healed_names)
        self.assertTrue(healed["right_wrist"].recovered_from_mirror)
        self.assertAlmostEqual(healed["right_wrist"].x, 55.0)

    def test_full_dual_biomechanics_calculation(self):
        front_pose = {
            "left_shoulder": (160.0, 200.0, 0.9),
            "right_shoulder": (140.0, 200.0, 0.9), # 侧身，前肩宽仅 20px
            "left_hip": (160.0, 300.0, 0.9),
            "right_hip": (140.0, 300.0, 0.9),
            "right_wrist": (80.0, 220.0, 0.85),
        }
        back_pose = {
            "left_shoulder": (220.0, 200.0, 0.85),
            "right_shoulder": (120.0, 200.0, 0.85), # 后背视角肩宽 100px
            "right_wrist": (50.0, 210.0, 0.85),
        }
        result = self.engine.calculate_dual_biomechanics(front_pose, back_pose, ball_pos=(75.0, 222.0))

        # 验证抗侧身塌陷转角未崩溃
        self.assertGreater(result.robust_shoulder_turn_deg, 0.0)
        self.assertEqual(result.front_shoulder_width, 20.0)
        self.assertEqual(result.back_shoulder_width, 100.0)
        # 验证引拍深度
        self.assertGreater(result.takeback_depth_ratio, 0.0)
        # 验证正手判定有效
        self.assertEqual(result.shot_classification.shot_type, FOREHAND)
        self.assertTrue(result.shot_classification.is_valid_contact)


if __name__ == "__main__":
    unittest.main()
