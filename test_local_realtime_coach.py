import unittest

from local_realtime_coach import LocalRealtimeCoach


class LocalRealtimeCoachTests(unittest.TestCase):
    def test_pose_gap_gets_short_capture_guidance_before_technique_advice(self):
        coach = LocalRealtimeCoach(max_chars=15)
        event = {
            "event_id": 1,
            "confidence": 0.92,
            "quality_flags": {
                "warnings": ["pose_gaps"],
                "pose_frame_ratio": 0.55,
            },
            "phase_counts": {
                "backswing": 1,
                "follow_through": 1,
            },
        }

        advice = coach.advise(event)

        self.assertEqual(advice["code"], "pose_gaps")
        self.assertEqual(advice["message"], "保持全身清晰入镜")
        self.assertLessEqual(len(advice["message"]), 15)
        self.assertEqual(advice["evidence"], {"pose_frame_ratio": 0.55})

    def test_short_follow_through_gets_one_actionable_tip(self):
        coach = LocalRealtimeCoach(max_chars=15)
        event = {
            "event_id": 2,
            "confidence": 0.88,
            "quality_flags": {"warnings": []},
            "phase_counts": {
                "backswing": 8,
                "follow_through": 2,
            },
        }

        advice = coach.advise(event)

        self.assertEqual(advice["code"], "short_follow_through")
        self.assertEqual(advice["message"], "击球后完成随挥")
        self.assertLessEqual(len(advice["message"]), 15)
        self.assertEqual(advice["evidence"], {"follow_through_frames": 2})

    def test_intermittent_ball_gap_does_not_override_technique_feedback(self):
        coach = LocalRealtimeCoach(max_chars=15)
        event = {
            "event_id": 3,
            "confidence": 0.91,
            "quality_flags": {
                "warnings": ["ball_track_gaps"],
                "ball_frame_ratio": 0.42,
                "ball_contact_window_ratio": 0.67,
            },
            "phase_counts": {"backswing": 8, "follow_through": 1},
        }

        advice = coach.advise(event)

        self.assertEqual(
            (advice["code"], advice["message"], advice["category"]),
            ("short_follow_through", "击球后完成随挥", "technique"),
        )

    def test_severe_ball_gap_still_gets_capture_guidance(self):
        coach = LocalRealtimeCoach(max_chars=15)
        event = {
            "event_id": 13,
            "confidence": 0.91,
            "quality_flags": {
                "warnings": ["ball_track_gaps"],
                "ball_frame_ratio": 0.1,
                "ball_contact_window_ratio": 0.0,
            },
            "phase_counts": {"backswing": 8, "follow_through": 1},
        }

        advice = coach.advise(event)

        self.assertEqual(
            (advice["code"], advice["message"], advice["category"]),
            ("ball_track_gaps", "确保来球完整入镜", "capture"),
        )

    def test_well_formed_swing_gets_short_positive_guidance(self):
        coach = LocalRealtimeCoach(max_chars=15)
        event = {
            "event_id": 4,
            "confidence": 0.9,
            "quality_flags": {"warnings": []},
            "phase_counts": {
                "backswing": 7,
                "follow_through": 8,
            },
        }

        advice = coach.advise(event)

        self.assertEqual(
            (advice["code"], advice["message"]),
            ("maintain_form", "动作稳定继续保持"),
        )
        self.assertLessEqual(len(advice["message"]), 15)

    def test_low_confidence_swing_is_reviewed_before_technique_feedback(self):
        coach = LocalRealtimeCoach(max_chars=15)
        event = {
            "event_id": 5,
            "confidence": 0.42,
            "quality_flags": {"warnings": []},
            "phase_counts": {"backswing": 1, "follow_through": 1},
        }

        advice = coach.advise(event)

        self.assertEqual(
            (advice["code"], advice["message"], advice["category"]),
            ("low_confidence", "本次动作建议复核", "review"),
        )

    def test_racket_gap_gets_capture_guidance(self):
        coach = LocalRealtimeCoach(max_chars=15)
        event = {
            "event_id": 6,
            "confidence": 0.9,
            "quality_flags": {
                "warnings": ["racket_track_gaps"],
                "racket_frame_ratio": 0.48,
            },
            "phase_counts": {"backswing": 8, "follow_through": 8},
        }

        advice = coach.advise(event)

        self.assertEqual(
            (advice["code"], advice["message"]),
            ("racket_track_gaps", "减少球拍遮挡"),
        )

    def test_racket_gap_does_not_override_short_follow_through(self):
        coach = LocalRealtimeCoach(max_chars=15)
        event = {
            "event_id": 16,
            "confidence": 0.9,
            "quality_flags": {
                "warnings": ["racket_track_gaps"],
                "pose_frame_ratio": 1.0,
                "racket_frame_ratio": 0.48,
            },
            "phase_counts": {"backswing": 8, "follow_through": 2},
        }

        advice = coach.advise(event)

        self.assertEqual(
            (advice["code"], advice["message"], advice["category"]),
            ("short_follow_through", "击球后完成随挥", "technique"),
        )

    def test_short_backswing_gets_early_preparation_tip(self):
        coach = LocalRealtimeCoach(max_chars=15)
        event = {
            "event_id": 7,
            "confidence": 0.9,
            "quality_flags": {"warnings": []},
            "phase_counts": {"backswing": 1, "follow_through": 8},
        }

        advice = coach.advise(event)

        self.assertEqual(
            (advice["code"], advice["message"]),
            ("short_backswing", "提前准备充分引拍"),
        )

    def test_contact_review_warning_overrides_positive_feedback(self):
        coach = LocalRealtimeCoach(max_chars=15)
        event = {
            "event_id": 8,
            "confidence": 0.9,
            "quality_flags": {
                "warnings": ["contact_frame_needs_review"],
                "review_recommended": True,
            },
            "phase_counts": {"backswing": 8, "follow_through": 8},
        }

        advice = coach.advise(event)

        self.assertEqual(
            (advice["code"], advice["message"], advice["category"]),
            ("contact_frame_needs_review", "触球位置需复核", "review"),
        )

    def test_tracking_review_warnings_never_produce_technique_feedback(self):
        coach = LocalRealtimeCoach(max_chars=15)
        warnings = [
            "ball_continuity_disabled",
            "static_ball_mask_in_event",
            "mirror_ball_rejection_in_event",
        ]

        advice = [
            coach.advise(
                {
                    "event_id": index,
                    "confidence": 0.9,
                    "quality_flags": {
                        "warnings": [warning],
                        "review_recommended": True,
                    },
                    "phase_counts": {"backswing": 8, "follow_through": 8},
                }
            )
            for index, warning in enumerate(warnings, start=9)
        ]

        self.assertEqual(
            [(item["code"], item["category"]) for item in advice],
            [(warning, "review") for warning in warnings],
        )
        self.assertTrue(all(len(item["message"]) <= 15 for item in advice))


if __name__ == "__main__":
    unittest.main()
