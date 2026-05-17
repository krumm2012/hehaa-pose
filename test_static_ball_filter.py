import unittest

from static_ball_filter import StaticBallFilter


class StaticBallFilterTests(unittest.TestCase):
    def test_penalizes_persistent_static_anchor(self):
        flt = StaticBallFilter(
            {
                "static_ball_suppression_enabled": True,
                "static_ball_min_seen_frames": 4,
                "static_ball_penalty_max": 0.6,
            }
        )
        for _ in range(6):
            flt.update([[100.0, 100.0]])
        self.assertGreater(flt.penalty([101.0, 99.0]), 0.0)

    def test_reduces_penalty_near_racket_or_previous_track(self):
        flt = StaticBallFilter(
            {
                "static_ball_suppression_enabled": True,
                "static_ball_min_seen_frames": 4,
                "static_ball_penalty_max": 0.6,
            }
        )
        for _ in range(6):
            flt.update([[240.0, 410.0]])
        base = flt.penalty([240.0, 410.0])
        near_track = flt.penalty([240.0, 410.0], near_previous_track=True)
        near_racket = flt.penalty([240.0, 410.0], near_racket=True)
        self.assertGreater(base, near_track)
        self.assertGreater(base, near_racket)

    def test_expires_old_anchors(self):
        flt = StaticBallFilter(
            {
                "static_ball_suppression_enabled": True,
                "static_ball_min_seen_frames": 3,
                "static_ball_decay_frames": 5,
                "static_ball_penalty_max": 0.5,
            }
        )
        for _ in range(4):
            flt.update([[80.0, 80.0]])
        self.assertGreater(flt.penalty([80.0, 80.0]), 0.0)
        for _ in range(7):
            flt.update([])
        self.assertEqual(flt.penalty([80.0, 80.0]), 0.0)

    def test_hard_mask_blocks_persistent_static_anchor(self):
        flt = StaticBallFilter(
            {
                "static_ball_suppression_enabled": True,
                "static_ball_hard_mask_enabled": True,
                "static_ball_min_seen_frames": 4,
                "static_ball_hard_mask_min_seen_frames": 6,
            }
        )
        for _ in range(6):
            flt.update([[120.0, 240.0]])
        self.assertTrue(flt.should_mask([121.0, 239.0]))

    def test_hard_mask_keeps_continuous_track_when_enabled(self):
        flt = StaticBallFilter(
            {
                "static_ball_suppression_enabled": True,
                "static_ball_hard_mask_enabled": True,
                "static_ball_min_seen_frames": 4,
                "static_ball_hard_mask_min_seen_frames": 6,
                "static_ball_hard_mask_allow_near_prev": True,
            }
        )
        for _ in range(6):
            flt.update([[300.0, 360.0]])
        self.assertFalse(flt.should_mask([300.0, 360.0], near_previous_track=True))

    def test_hard_mask_does_not_allow_near_racket_by_default(self):
        flt = StaticBallFilter(
            {
                "static_ball_suppression_enabled": True,
                "static_ball_hard_mask_enabled": True,
                "static_ball_min_seen_frames": 4,
                "static_ball_hard_mask_min_seen_frames": 6,
                "static_ball_hard_mask_allow_near_racket": False,
            }
        )
        for _ in range(6):
            flt.update([[420.0, 420.0]])
        self.assertTrue(flt.should_mask([420.0, 420.0], near_racket=True))


if __name__ == "__main__":
    unittest.main()
