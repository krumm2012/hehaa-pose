import unittest
from swing_session_quality import build_session_quality_dashboard


class TestSessionFatigueConsistency(unittest.TestCase):
    """测试整场训练疲劳衰减与动作一致性方差统计 (Fatigue and Consistency Analytics)"""

    def _make_event(self, event_id: int, speed_kmh: float, latency_ms: float = 35.0, shoulder_turn: float = 85.0):
        return {
            "event_id": event_id,
            "start_frame": event_id * 100,
            "contact_frame": event_id * 100 + 40,
            "end_frame": event_id * 100 + 80,
            "stroke_type": "Forehand",
            "confidence": 0.90,
            "quality_flags": {"warnings": [], "pose_frame_ratio": 0.95},
            "practice_score": 85.0,
            "racket_speed": speed_kmh,
            "extended_biomechanics": {
                "racket_head_speed": {"contact_kmh": speed_kmh},
                "kinematic_sequence": {"latency_hip_to_shoulder_ms": latency_ms},
            },
            "biomechanics": {
                "metrics": {
                    "shoulder_turn": {"value": shoulder_turn, "coach_eligible": True},
                }
            },
        }

    def test_warming_up_few_events(self):
        """测试事件数少于 4 个时处于 WARMING_UP 状态。"""
        events = [self._make_event(i, 80.0) for i in range(1, 3)]
        dashboard = build_session_quality_dashboard(events)
        fc = dashboard.get("fatigue_and_consistency") or {}
        self.assertEqual(fc.get("fatigue_status"), "WARMING_UP")
        self.assertIsNone(fc.get("speed_decay_percent"))

    def test_consistent_session(self):
        """测试击球速度与时序稳定的会话 (CONSISTENT)。"""
        # 6次击球，速度保持在 80 ~ 82 km/h
        speeds = [80.0, 81.0, 82.0, 81.5, 80.5, 81.0]
        events = [self._make_event(i + 1, s, latency_ms=35.0 + (i % 2) * 2.0) for i, s in enumerate(speeds)]
        dashboard = build_session_quality_dashboard(events)
        fc = dashboard.get("fatigue_and_consistency") or {}
        self.assertEqual(fc.get("fatigue_status"), "CONSISTENT")
        self.assertIsNotNone(fc.get("speed_std"))
        self.assertLess(fc.get("speed_std"), 2.0)
        self.assertIsNotNone(fc.get("latency_jitter_std_ms"))

    def test_fatigue_observed_when_speed_drops(self):
        """测试击球速度明显衰减 (> 8%) 时被判定为 FATIGUE_OBSERVED。"""
        # 前期 Baseline 85 km/h，后期衰减到 72 km/h (衰减约 -15%)
        speeds = [85.0, 86.0, 84.0, 78.0, 73.0, 71.0]
        events = [self._make_event(i + 1, s, latency_ms=30.0 + i * 5.0) for i, s in enumerate(speeds)]
        dashboard = build_session_quality_dashboard(events)
        fc = dashboard.get("fatigue_and_consistency") or {}
        self.assertEqual(fc.get("fatigue_status"), "FATIGUE_OBSERVED")
        self.assertIsNotNone(fc.get("speed_decay_percent"))
        self.assertLess(fc.get("speed_decay_percent"), -8.0)
        self.assertGreater(fc.get("speed_std"), 5.0)

    def test_warmed_up_when_speed_increases(self):
        """测试随着训练深入速度提升 (> 5%) 时被判定为 WARMED_UP。"""
        # 前期 70 km/h，后期提升到 82 km/h (+17%)
        speeds = [70.0, 71.0, 72.0, 78.0, 81.0, 83.0]
        events = [self._make_event(i + 1, s) for i, s in enumerate(speeds)]
        dashboard = build_session_quality_dashboard(events)
        fc = dashboard.get("fatigue_and_consistency") or {}
        self.assertEqual(fc.get("fatigue_status"), "WARMED_UP")
        self.assertGreater(fc.get("speed_decay_percent"), 5.0)


if __name__ == "__main__":
    unittest.main()
