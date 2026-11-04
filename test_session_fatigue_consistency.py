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
                "racket_head_speed": {"contact_kmh": speed_kmh,
                    "measurement_policy": "image_motion_v2_source_identity",
                    "contact_time_basis": "media_pts", "status": "homography_height_debiased",
                    "ground_calibration_sha256": "test-calibration",
                    "geometry_assumptions": {"camera_height_m": 3.1, "racket_height_m": 1.2}},
                "kinematic_sequence": {"latency_hip_to_shoulder_ms": latency_ms},
            },
            "biomechanics": {
                "reference_body_width_px": 100.0,
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
        self.assertEqual(fc.get("fatigue_status"), "SPEED_STABLE")
        self.assertIsNotNone(fc.get("speed_std"))
        self.assertLess(fc.get("speed_std"), 2.0)
        self.assertIsNone(fc.get("latency_jitter_std_ms"))

    def test_fatigue_observed_when_speed_drops(self):
        """测试击球速度明显衰减 (> 8%) 时被判定为 FATIGUE_OBSERVED。"""
        # 前期 Baseline 85 km/h，后期衰减到 72 km/h (衰减约 -15%)
        speeds = [85.0, 86.0, 84.0, 78.0, 73.0, 71.0]
        events = [self._make_event(i + 1, s, latency_ms=30.0 + i * 5.0) for i, s in enumerate(speeds)]
        dashboard = build_session_quality_dashboard(events)
        fc = dashboard.get("fatigue_and_consistency") or {}
        self.assertEqual(fc.get("fatigue_status"), "SPEED_DECREASED")
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
        self.assertEqual(fc.get("fatigue_status"), "SPEED_INCREASED")
        self.assertGreater(fc.get("speed_decay_percent"), 5.0)

    def test_mixed_units_never_form_a_decay(self):
        events = [self._make_event(i, 80) for i in range(1, 7)]
        for event in events[3:]:
            event.pop('racket_speed')
            speed = event['extended_biomechanics']['racket_head_speed']
            speed.pop('contact_kmh'); speed['contact_px_s'] = 800
        fc = build_session_quality_dashboard(events)['fatigue_and_consistency']
        self.assertEqual(fc['fatigue_status'], 'BLOCKED')
        self.assertIsNone(fc['speed_decay_percent'])
        self.assertEqual({g['unit'] for g in fc['measurement_groups']}, {'km/h', 'px/s'})

    def test_complete_pixel_series_is_not_mixed_with_partial_metric(self):
        events = [self._make_event(i, 80 if i > 3 else None) for i in range(1, 7)]
        for event in events:
            event['extended_biomechanics']['racket_head_speed']['contact_px_s'] = 1000
        fc = build_session_quality_dashboard(events)['fatigue_and_consistency']
        self.assertEqual(fc['speed_unit'], 'px/s')
        self.assertEqual(fc['speed_decay_percent'], 0)

    def test_comparison_gates_and_missing_data(self):
        for reason in ('stroke', 'overlap', 'scale', 'geometry', 'method', 'time', 'missing'):
            with self.subTest(reason=reason):
                events = [self._make_event(i, 80 if i <= 3 else 60) for i in range(1, 7)]
                e = events[-1]
                speed = e['extended_biomechanics']['racket_head_speed']
                if reason == 'stroke': e['stroke_type'] = 'Backhand'
                if reason == 'overlap': e['start_frame'] = events[-2]['end_frame']
                if reason == 'scale':
                    for item in events[3:]: item['biomechanics']['reference_body_width_px'] = 200
                if reason == 'geometry': speed['geometry_assumptions']['camera_height_m'] = 2.4
                if reason == 'method': speed['measurement_policy'] = 'other'
                if reason == 'time': speed['contact_time_basis'] = 'nominal_fps'
                if reason == 'missing': speed.pop('contact_kmh'); e.pop('racket_speed')
                fc = build_session_quality_dashboard(events)['fatigue_and_consistency']
                self.assertEqual(fc['fatigue_status'], 'BLOCKED')
                self.assertIsNone(fc['speed_decay_percent'])

    def test_no_measurement_never_claims_consistency(self):
        events = [self._make_event(i, None) for i in range(1, 7)]
        fc = build_session_quality_dashboard(events)['fatigue_and_consistency']
        self.assertEqual(fc['samples_evaluated'], 0)
        self.assertFalse(fc['comparison_ready'])
        self.assertEqual(fc['fatigue_status'], 'BLOCKED')

    def test_zero_speed_is_retained_but_zero_baseline_blocks_percentage(self):
        events = [self._make_event(i, 0) for i in range(1, 7)]
        for e in events: e['extended_biomechanics']['racket_head_speed']['contact_px_s'] = 800
        fc = build_session_quality_dashboard(events)['fatigue_and_consistency']
        self.assertEqual(fc['speed_unit'], 'km/h')
        self.assertEqual(fc['samples_evaluated'], 6)
        self.assertIn('zero_baseline_speed', fc['blocked_reasons'])
        self.assertIsNone(fc['speed_decay_percent'])


if __name__ == "__main__":
    unittest.main()
