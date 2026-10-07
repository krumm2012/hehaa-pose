"""
Unit tests for RacketResolver and RacketResolutionConfig.
"""

import unittest
from dual_pose_estimator import Keypoint
from racket_resolution import RacketResolver, RacketResolutionConfig, RacketResolutionResult


class TestRacketResolution(unittest.TestCase):
    def setUp(self):
        self.config = RacketResolutionConfig(
            mirror_fallback_zone=(680.0, 0.0, 1580.0, 620.0),
            back_racket_dist_threshold=250.0,
            wrist_proximity_threshold=420.0,
            mirror_penalty_dist=200.0,
            max_missing_smooth_frames=2,
            decay_confidence=0.5,
            recovered_confidence_scale=0.85,
        )
        self.resolver = RacketResolver(self.config)

    def test_observed_front_racket_selected(self):
        """正面机位清晰检出手腕与球拍，正常作为 observed 选中。"""
        front_pose = {
            "right_wrist": Keypoint(1500.0, 450.0, 0.95),
            "right_elbow": Keypoint(1450.0, 420.0, 0.95),
        }
        raw_rackets = [
            {"box": [1480.0, 400.0, 1540.0, 520.0], "confidence": 0.88},
            {"box": [200.0, 300.0, 240.0, 350.0], "confidence": 0.30},
        ]

        res = self.resolver.resolve(
            raw_rackets=raw_rackets,
            front_pose=front_pose,
            back_pose={},
            frame_idx=10,
        )

        self.assertEqual(res.status, "observed")
        self.assertFalse(res.is_racket_recovered)
        self.assertEqual(res.missing_count, 0)
        self.assertIsNotNone(res.racket_box)
        self.assertEqual(len(res.rackets), 2)
        # 离手腕最近的球拍排在首位
        self.assertEqual(res.racket_box, [1480.0, 400.0, 1540.0, 520.0])
        self.assertTrue(res.rackets[0]["observed"])
        self.assertEqual(res.rackets[0]["source_frame_id"], 10)

    def test_back_mirror_racket_recovered_when_front_missing(self):
        """正面引拍遮挡时，利用镜中背面球拍进行空间自愈补偿。"""
        front_pose = {
            "left_shoulder": Keypoint(1545.3, 421.2, 0.99),
            "right_shoulder": Keypoint(1451.0, 387.7, 0.99),
            "left_hip": Keypoint(1508.1, 546.5, 0.99),
            "right_hip": Keypoint(1447.3, 527.9, 0.99),
            "right_wrist": Keypoint(1541.6, 438.6, 0.95),
        }
        back_pose = {
            "left_shoulder": Keypoint(1467.2, 69.1, 0.99),
            "right_shoulder": Keypoint(1413.4, 52.3, 0.96),
            "left_hip": Keypoint(1458.1, 166.0, 0.99),
            "right_hip": Keypoint(1420.5, 162.8, 0.99),
            "right_wrist": Keypoint(1480.0, 110.0, 0.90),
        }
        # 镜中检出球拍框 (x=1444~1526, 中心 1485.0，落在镜面区域内部，且靠近 back_pose)
        raw_rackets = [
            {"box": [1444.0, 89.0, 1526.0, 150.0], "confidence": 0.80}
        ]

        res = self.resolver.resolve(
            raw_rackets=raw_rackets,
            front_pose=front_pose,
            back_pose=back_pose,
            frame_idx=37,
        )

        self.assertEqual(res.status, "recovered")
        self.assertTrue(res.is_racket_recovered)
        self.assertEqual(res.missing_count, 0)
        self.assertIsNotNone(res.racket_box)
        self.assertEqual(res.back_racket_box, [1444.0, 89.0, 1526.0, 150.0])
        # 验证标记为 recovered_from_mirror，且 observed 为 False
        self.assertTrue(res.rackets[0]["recovered_from_mirror"])
        self.assertFalse(res.rackets[0]["observed"])
        # 验证置信度按比例缩放 (0.80 * 0.85 = 0.68)
        self.assertAlmostEqual(res.rackets[0]["confidence"], 0.68, delta=0.01)

    def test_temporal_decay_lifecycle(self):
        """验证球拍丢失时的平滑衰减与最终置空状态机。"""
        front_pose = {
            "right_wrist": Keypoint(1500.0, 450.0, 0.95),
        }
        # Frame 1: 正常检出
        res1 = self.resolver.resolve(
            raw_rackets=[{"box": [1480.0, 400.0, 1540.0, 520.0], "confidence": 0.9}],
            front_pose=front_pose,
            frame_idx=1,
        )
        self.assertEqual(res1.status, "observed")
        self.assertEqual(res1.missing_count, 0)

        # Frame 2: 突然丢失，进入第 1 帧衰减
        res2 = self.resolver.resolve(raw_rackets=[], front_pose=front_pose, frame_idx=2)
        self.assertEqual(res2.status, "decayed")
        self.assertEqual(res2.missing_count, 1)
        self.assertFalse(res2.rackets[0]["observed"])
        self.assertEqual(res2.racket_box, [1480.0, 400.0, 1540.0, 520.0])

        # Frame 3: 继续丢失，进入第 2 帧衰减
        res3 = self.resolver.resolve(raw_rackets=[], front_pose=front_pose, frame_idx=3)
        self.assertEqual(res3.status, "decayed")
        self.assertEqual(res3.missing_count, 2)

        # Frame 4: 超过最大衰减帧数(2帧)，彻底置空 missing
        res4 = self.resolver.resolve(raw_rackets=[], front_pose=front_pose, frame_idx=4)
        self.assertEqual(res4.status, "missing")
        self.assertEqual(res4.missing_count, 3)
        self.assertIsNone(res4.racket_box)
        self.assertEqual(res4.rackets, [])

    def test_mirror_ghost_without_body_rejected(self):
        """镜中反光杂物（远离镜中人体的假球拍）不会被误当成真实持拍。"""
        back_pose = {
            "right_wrist": Keypoint(800.0, 100.0, 0.90),
        }
        # 远在 x=1500 的杂物框 (距离 > 250px)
        raw_rackets = [
            {"box": [1480.0, 100.0, 1520.0, 150.0], "confidence": 0.70}
        ]
        res = self.resolver.resolve(
            raw_rackets=raw_rackets,
            front_pose={},
            back_pose=back_pose,
            frame_idx=5,
        )
        # 没有被当成镜中球拍，且无手腕，被安全过滤
        self.assertIsNone(res.back_racket_box)
        self.assertEqual(res.status, "missing")

    def test_config_from_yaml_and_dict(self):
        """验证配置项解析与回退机制。"""
        cfg_dict = {
            "racket_resolution": {
                "back_racket_dist_threshold": 310.0,
                "wrist_proximity_threshold": 480.0,
                "max_missing_smooth_frames": 3,
            }
        }
        cfg = RacketResolutionConfig.from_dict(cfg_dict)
        self.assertEqual(cfg.back_racket_dist_threshold, 310.0)
        self.assertEqual(cfg.wrist_proximity_threshold, 480.0)
        self.assertEqual(cfg.max_missing_smooth_frames, 3)
        self.assertEqual(cfg.decay_confidence, 0.5)  # 保持默认

        # 从现有的 configs/dual_view_config.yaml 加载
        cfg_yaml = RacketResolutionConfig.from_yaml("configs/dual_view_config.yaml")
        self.assertEqual(cfg_yaml.back_racket_dist_threshold, 250.0)
        self.assertEqual(cfg_yaml.wrist_proximity_threshold, 420.0)


if __name__ == "__main__":
    unittest.main()
