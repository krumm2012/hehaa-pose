import unittest

from swing_presence_detector import detect_swing_presence


class SwingPresenceDetectorLogicTests(unittest.TestCase):
    def test_returns_false_when_no_events(self):
        has_swing, stats = detect_swing_presence([])
        self.assertFalse(has_swing)
        self.assertEqual(stats["swing_frame_count"], 0)

    def test_requires_motion_threshold(self):
        events = [
            {"label": "Forehand", "motion_px": 2.0},
            {"label": "Backhand", "motion_px": 3.0},
            {"label": "Forehand", "motion_px": 2.5},
            {"label": "Forehand", "motion_px": 1.0},
        ]
        has_swing, stats = detect_swing_presence(
            events,
            min_swing_frames=3,
            min_consecutive=2,
            min_motion_px=6.0,
        )
        self.assertFalse(has_swing)
        self.assertEqual(stats["qualified_swing_frames"], 0)

    def test_detects_swing_when_thresholds_met(self):
        events = [
            {"label": "No Pose", "motion_px": 0.0},
            {"label": "Forehand", "motion_px": 8.0},
            {"label": "Forehand", "motion_px": 12.0},
            {"label": "Backhand", "motion_px": 10.5},
            {"label": "Forehand", "motion_px": 9.2},
            {"label": "Incomplete Pose", "motion_px": 0.0},
        ]
        has_swing, stats = detect_swing_presence(
            events,
            min_swing_frames=3,
            min_consecutive=2,
            min_motion_px=7.0,
        )
        self.assertTrue(has_swing)
        self.assertGreaterEqual(stats["qualified_swing_frames"], 3)
        self.assertGreaterEqual(stats["max_consecutive_qualified"], 2)

    def test_counts_swing_events_instead_of_frames(self):
        events = [
            {"frame": 0, "label": "Forehand", "motion_px": 9.0},
            {"frame": 1, "label": "Forehand", "motion_px": 8.5},
            {"frame": 2, "label": "No Pose", "motion_px": 0.0},
            {"frame": 3, "label": "No Pose", "motion_px": 0.0},
            {"frame": 4, "label": "Backhand", "motion_px": 10.0},
            {"frame": 5, "label": "Backhand", "motion_px": 9.8},
        ]
        has_swing, stats = detect_swing_presence(
            events,
            min_swing_frames=2,
            min_consecutive=2,
            min_motion_px=6.0,
            min_event_frames=2,
            max_break_frames=0,
            min_core_frames=1,
            min_rearm_frames=1,
            post_event_cooldown_frames=0,
            entry_confirm_frames=1,
            exit_confirm_frames=1,
            motion_smooth_window=1,
            label_vote_window=1,
        )
        self.assertTrue(has_swing)
        self.assertEqual(stats["swing_event_count"], 2)
        self.assertEqual(stats["swing_event_type_counts"]["Forehand"], 1)
        self.assertEqual(stats["swing_event_type_counts"]["Backhand"], 1)

    def test_merges_event_with_small_break(self):
        events = [
            {"frame": 0, "label": "Forehand", "motion_px": 9.0},
            {"frame": 1, "label": "No Pose", "motion_px": 0.0},
            {"frame": 2, "label": "Forehand", "motion_px": 8.5},
        ]
        has_swing, stats = detect_swing_presence(
            events,
            min_swing_frames=2,
            min_consecutive=1,
            min_motion_px=6.0,
            min_event_frames=2,
            max_break_frames=1,
        )
        self.assertTrue(has_swing)
        self.assertEqual(stats["swing_event_count"], 1)

    def test_filters_outlier_motion_by_max_motion_threshold(self):
        events = [
            {"frame": 0, "label": "Forehand", "motion_px": 10.0},
            {"frame": 1, "label": "Forehand", "motion_px": 180.0},  # 跳变异常
            {"frame": 2, "label": "Forehand", "motion_px": 11.0},
        ]
        has_swing, stats = detect_swing_presence(
            events,
            min_swing_frames=2,
            min_consecutive=1,
            min_motion_px=6.0,
            max_motion_px=120.0,
            min_event_frames=2,
            max_break_frames=1,
            entry_confirm_frames=1,
            exit_confirm_frames=1,
            motion_smooth_window=1,
        )
        self.assertTrue(has_swing)
        self.assertEqual(stats["qualified_swing_frames"], 2)
        self.assertEqual(stats["swing_event_count"], 1)

    def test_can_return_frame_trace(self):
        events = [{"frame": 0, "label": "Forehand", "motion_px": 8.0}]
        _, stats = detect_swing_presence(
            events,
            min_swing_frames=1,
            min_consecutive=1,
            min_motion_px=6.0,
            max_motion_px=120.0,
            min_event_frames=1,
            max_break_frames=0,
            return_frame_trace=True,
        )
        self.assertIn("frame_trace", stats)
        self.assertEqual(len(stats["frame_trace"]), 1)
        self.assertTrue(stats["frame_trace"][0]["qualified"])

    def test_does_not_split_event_on_single_frame_label_jitter(self):
        events = [
            {"frame": 0, "label": "Forehand", "motion_px": 9.0},
            {"frame": 1, "label": "Forehand", "motion_px": 9.0},
            {"frame": 2, "label": "Backhand", "motion_px": 9.0},  # 抖动
            {"frame": 3, "label": "Forehand", "motion_px": 9.0},
            {"frame": 4, "label": "Forehand", "motion_px": 9.0},
        ]
        has_swing, stats = detect_swing_presence(
            events,
            min_swing_frames=3,
            min_consecutive=2,
            min_motion_px=6.0,
            min_event_frames=2,
            max_break_frames=0,
            type_switch_min_frames=2,
        )
        self.assertTrue(has_swing)
        self.assertEqual(stats["swing_event_count"], 1)
        self.assertEqual(stats["swing_event_type_counts"]["Forehand"], 1)

    def test_splits_events_after_reset_window(self):
        events = [
            {"frame": 0, "label": "Forehand", "motion_px": 9.0},
            {"frame": 1, "label": "Forehand", "motion_px": 9.0},
            {"frame": 2, "label": "No Pose", "motion_px": 0.0},
            {"frame": 3, "label": "No Pose", "motion_px": 0.0},
            {"frame": 4, "label": "No Pose", "motion_px": 0.0},
            {"frame": 5, "label": "Backhand", "motion_px": 9.0},
            {"frame": 6, "label": "Backhand", "motion_px": 9.0},
            {"frame": 7, "label": "Backhand", "motion_px": 9.0},
        ]
        has_swing, stats = detect_swing_presence(
            events,
            min_swing_frames=3,
            min_consecutive=2,
            min_motion_px=6.0,
            min_event_frames=2,
            max_break_frames=0,
            type_switch_min_frames=2,
            min_core_frames=1,
            min_rearm_frames=2,
            post_event_cooldown_frames=0,
            entry_confirm_frames=1,
            exit_confirm_frames=1,
        )
        self.assertTrue(has_swing)
        self.assertEqual(stats["swing_event_count"], 2)
        self.assertEqual(stats["swing_event_type_counts"]["Forehand"], 1)
        self.assertEqual(stats["swing_event_type_counts"]["Backhand"], 1)

    def test_suppresses_follow_through_retrigger(self):
        events = [
            {"frame": 0, "label": "Forehand", "motion_px": 9.5},
            {"frame": 1, "label": "Forehand", "motion_px": 10.5},
            {"frame": 2, "label": "Forehand", "motion_px": 12.0},  # 峰值
            {"frame": 3, "label": "Forehand", "motion_px": 3.0},
            {"frame": 4, "label": "Forehand", "motion_px": 3.2},
            {"frame": 5, "label": "Forehand", "motion_px": 3.5},
            {"frame": 6, "label": "Forehand", "motion_px": 9.0},   # 随挥抬手，不应二次计数
            {"frame": 7, "label": "Forehand", "motion_px": 8.7},
        ]
        has_swing, stats = detect_swing_presence(
            events,
            min_swing_frames=3,
            min_consecutive=2,
            min_motion_px=6.0,
            min_event_frames=2,
            max_break_frames=1,
            min_core_frames=1,
            min_rearm_frames=4,
            post_event_cooldown_frames=6,
        )
        self.assertTrue(has_swing)
        self.assertEqual(stats["swing_event_count"], 1)


if __name__ == "__main__":
    unittest.main()
