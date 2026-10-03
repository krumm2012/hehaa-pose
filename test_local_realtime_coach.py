import unittest

from local_realtime_coach import LocalRealtimeCoach


class LocalRealtimeCoachTests(unittest.TestCase):
    def test_returns_up_to_three_ranked_biomechanical_corrections_with_confidence(self):
        coach = LocalRealtimeCoach(
            max_chars=15,
            max_suggestions=3,
            min_confidence=0.45,
        )
        event = {
            "event_id": 20,
            "confidence": 0.9,
            "quality_flags": {"warnings": [], "pose_frame_ratio": 0.94},
            "phase_counts": {"backswing": 8, "follow_through": 8},
            "biomechanics": {
                "schema_version": "single_view_2d_v1",
                "metrics": {
                    "contact_lateral_distance": {
                        "value": 0.31,
                        "unit": "body_width",
                        "confidence": 0.86,
                        "coach_eligible": True,
                    },
                    "arm_extension": {
                        "value": 121.0,
                        "unit": "deg",
                        "confidence": 0.91,
                        "coach_eligible": True,
                    },
                    "hip_shoulder_separation": {
                        "value": 7.0,
                        "unit": "deg",
                        "confidence": 0.88,
                        "coach_eligible": True,
                    },
                    "balance_drift": {
                        "value": 0.9,
                        "unit": "body_width",
                        "confidence": 0.8,
                        "coach_eligible": True,
                    },
                },
            },
        }

        advices = coach.advise_all(event)

        self.assertEqual(len(advices), 3)
        self.assertEqual(
            {item["focus"] for item in advices},
            {"contact_position", "arm_extension", "hip_shoulder_separation"},
        )
        self.assertTrue(all(item["category"] == "technique" for item in advices))
        self.assertTrue(all(0.0 <= item["confidence"] <= 1.0 for item in advices))
        self.assertTrue(all(len(item["message"]) <= 15 for item in advices))
        self.assertTrue(all("metric_confidence" in item["evidence"] for item in advices))
        self.assertEqual(coach.advise(event), advices[0])

    def test_low_confidence_metric_is_not_used_for_technique_claim(self):
        coach = LocalRealtimeCoach(max_suggestions=3, min_confidence=0.6)
        event = {
            "event_id": 21,
            "confidence": 0.9,
            "quality_flags": {"warnings": []},
            "phase_counts": {"backswing": 8, "follow_through": 8},
            "biomechanics": {
                "metrics": {
                    "arm_extension": {
                        "value": 95.0,
                        "unit": "deg",
                        "confidence": 0.3,
                    }
                }
            },
        }

        advices = coach.advise_all(event)

        self.assertEqual(len(advices), 1)
        self.assertEqual(advices[0]["code"], "insufficient_technique_evidence")

    def test_moderate_pose_gap_does_not_block_reliable_biomechanics(self):
        coach = LocalRealtimeCoach(max_suggestions=3, min_confidence=0.45)
        event = {
            "event_id": 23,
            "confidence": 0.62,
            "quality_flags": {
                "warnings": ["pose_gaps"],
                "pose_frame_ratio": 0.8,
            },
            "phase_counts": {"backswing": 8, "follow_through": 8},
            "biomechanics": {
                "metrics": {
                    "arm_extension": {
                        "value": 120.0,
                        "unit": "deg",
                        "confidence": 0.75,
                        "coach_eligible": True,
                    }
                }
            },
        }

        advices = coach.advise_all(event)

        self.assertEqual(advices[0]["code"], "limited_arm_extension")
        self.assertEqual(advices[0]["category"], "technique")

    def test_gravity_candidates_are_deduplicated_to_one_focus(self):
        coach = LocalRealtimeCoach(max_suggestions=3)
        event = {
            "event_id": 22,
            "confidence": 0.92,
            "quality_flags": {"warnings": []},
            "phase_counts": {"backswing": 8, "follow_through": 8},
            "biomechanics": {
                "metrics": {
                    "weight_transfer": {
                        "value": 0.02,
                        "unit": "body_width",
                        "confidence": 0.9,
                        "coach_eligible": True,
                    },
                    "balance_drift": {
                        "value": 1.0,
                        "unit": "body_width",
                        "confidence": 0.9,
                        "coach_eligible": True,
                    },
                }
            },
        }

        advices = coach.advise_all(event)

        self.assertEqual(len(advices), 1)
        self.assertEqual(advices[0]["focus"], "balance")

    def test_unobservable_single_view_proxies_never_create_technique_claims(self):
        coach = LocalRealtimeCoach(max_suggestions=3)
        event = {
            "event_id": 24,
            "confidence": 0.95,
            "quality_flags": {"warnings": []},
            "phase_counts": {"backswing": 8, "follow_through": 8},
            "biomechanics": {
                "quality": {"contact_evidence_confidence": 0.0},
                "metrics": {
                    "hip_shoulder_separation": {
                        "value": 2.0,
                        "confidence": 0.9,
                        "coach_eligible": False,
                    },
                    "balance_drift": {
                        "value": 3.0,
                        "confidence": 0.9,
                        "coach_eligible": False,
                    },
                },
            },
        }

        advices = coach.advise_all(event)

        self.assertEqual(advices[0]["code"], "insufficient_technique_evidence")
        self.assertNotIn("limited_separation", {item["code"] for item in advices})
        self.assertNotIn("unstable_balance", {item["code"] for item in advices})

    def test_visible_knee_flexion_can_drive_lower_body_cue(self):
        coach = LocalRealtimeCoach(max_suggestions=3)
        event = {
            "event_id": 25,
            "confidence": 0.95,
            "quality_flags": {"warnings": []},
            "phase_counts": {"backswing": 8, "follow_through": 8},
            "biomechanics": {
                "quality": {"contact_evidence_confidence": 0.0},
                "metrics": {
                    "preparation_knee_flexion": {
                        "value": 5.0,
                        "unit": "deg_2d",
                        "confidence": 0.8,
                        "coach_eligible": True,
                        "coach_eligible": True,
                        "observability": "image_plane_joint_angle",
                    }
                },
            },
        }

        advice = coach.advise(event)

        self.assertEqual(advice["code"], "limited_knee_flexion")
        self.assertEqual(advice["message"], "准备时适当降低重心")

    def test_low_boundary_and_contact_evidence_suppress_phase_claims(self):
        coach = LocalRealtimeCoach(max_suggestions=3)
        event = {
            "event_id": 26,
            "confidence": 0.95,
            "evidence": {"start_boundary": {"confidence": "low"}},
            "quality_flags": {"warnings": []},
            "phase_counts": {"backswing": 0, "follow_through": 0},
            "biomechanics": {
                "quality": {"contact_evidence_confidence": 0.0},
                "metrics": {},
            },
        }

        advice = coach.advise(event)

        self.assertEqual(advice["code"], "insufficient_technique_evidence")

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
                        "coach_eligible": True,
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
                        "coach_eligible": True,
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
                        "coach_eligible": True,
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
            ("insufficient_technique_evidence", "动作证据不足需复核"),
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

    def test_valid_contact_with_benign_filters_returns_maintain_form(self):
        coach = LocalRealtimeCoach(max_chars=15)
        # Event with valid contact where static mask and mirror rejection were benign
        event = {
            "event_id": 12,
            "confidence": 0.95,
            "quality_flags": {
                "warnings": ["static_ball_mask_in_event", "mirror_ball_rejection_in_event"],
                "review_recommended": False,
            },
            "evidence": {
                "classification_context": {
                    "contact_analysis": {
                        "has_ball": True,
                        "is_valid_contact": True,
                        "has_trajectory_rebound": True,
                    }
                }
            },
            "phase_counts": {"backswing": 8, "follow_through": 8},
        }
        advices = coach.advise_all(event)
        self.assertEqual(len(advices), 1)
        self.assertEqual(advices[0]["code"], "insufficient_technique_evidence")
        self.assertEqual(advices[0]["message"], "动作证据不足需复核")
        self.assertEqual(advices[0]["category"], "review")

    def test_disconnected_kinetic_chain_produces_advice(self):
        coach = LocalRealtimeCoach(max_chars=15)
        event = {
            "event_id": 30,
            "confidence": 0.9,
            "quality_flags": {"warnings": []},
            "phase_counts": {"backswing": 8, "follow_through": 8},
            "biomechanics": {
                "metrics": {
                    "kinematic_sequence": {
                        "value": "DISCONNECTED",
                        "confidence": 0.85,
                        "coach_eligible": True,
                    }
                }
            },
        }
        advice = coach.advise(event)
        self.assertEqual(advice["code"], "disconnected_kinetic_chain")
        self.assertEqual(advice["message"], "用身体核心带动球拍发力")
        self.assertEqual(advice["category"], "technique")

    def test_limited_leg_drive_produces_advice(self):
        coach = LocalRealtimeCoach(max_chars=15)
        event = {
            "event_id": 31,
            "confidence": 0.9,
            "quality_flags": {"warnings": []},
            "phase_counts": {"backswing": 8, "follow_through": 8},
            "biomechanics": {
                "metrics": {
                    "leg_drive": {
                        "value": 0.04,
                        "confidence": 0.85,
                        "coach_eligible": True,
                    }
                }
            },
        }
        advice = coach.advise(event)
        self.assertEqual(advice["code"], "limited_leg_drive")
        self.assertEqual(advice["message"], "击球瞬间双腿蹬地发力")
        self.assertEqual(advice["category"], "technique")

    def test_limited_brush_drop_produces_advice(self):
        coach = LocalRealtimeCoach(max_chars=15)
        event = {
            "event_id": 32,
            "confidence": 0.9,
            "quality_flags": {"warnings": []},
            "phase_counts": {"backswing": 8, "follow_through": 8},
            "biomechanics": {
                "metrics": {
                    "brush_angle": {
                        "value": 5.0,
                        "drop_depth_ratio": 0.05,
                        "confidence": 0.85,
                        "coach_eligible": True,
                    }
                }
            },
        }
        advice = coach.advise(event)
        self.assertEqual(advice["code"], "limited_brush_drop")
        self.assertEqual(advice["message"], "击球前拍头下潜刷球")
        self.assertEqual(advice["category"], "technique")

    def test_capture_warning_does_not_suppress_technique_when_max_suggestions_gt_1(self):
        coach = LocalRealtimeCoach(max_chars=15, max_suggestions=3)
        event = {
            "event_id": 33,
            "confidence": 0.91,
                        "coach_eligible": True,
            "quality_flags": {
                "warnings": ["ball_track_gaps"],
                "ball_frame_ratio": 0.1,
                "ball_contact_window_ratio": 0.0,
            },
            "phase_counts": {"backswing": 8, "follow_through": 8},
            "biomechanics": {
                "metrics": {
                    "preparation_knee_flexion": {
                        "value": 5.0,
                        "confidence": 0.82,
                        "coach_eligible": True,
                    },
                    "kinematic_sequence": {
                        "value": "DISCONNECTED",
                        "confidence": 0.85,
                        "coach_eligible": True,
                    },
                }
            },
        }
        advices = coach.advise_all(event)
        self.assertGreaterEqual(len(advices), 2)
        self.assertEqual(advices[0]["code"], "ball_track_gaps")
        self.assertEqual(advices[0]["category"], "capture")
        self.assertTrue(any(a["category"] == "technique" for a in advices[1:]))

    def test_shadow_swing_silenced_no_advice(self):
        coach = LocalRealtimeCoach(max_chars=15, max_suggestions=3)
        event = {
            "event_id": 99,
            "confidence": 0.88,
                        "coach_eligible": True,
            "is_shadow_swing": True,
            "quality_flags": {
                "warnings": ["ball_track_gaps"],
                "ball_frame_ratio": 0.05,
            },
            "phase_counts": {"backswing": 8, "follow_through": 8},
        }
        advices = coach.advise_all(event)
        self.assertEqual(advices, [])
        self.assertIsNone(coach.advise(event))

    def test_shadow_swing_skips_calibration(self):
        from swing_coach_calibration import calibrate_coaching_event
        event = {
            "event_id": 100,
            "confidence": 0.88,
                        "coach_eligible": True,
            "is_shadow_swing": True,
            "biomechanics": {
                "metrics": {
                    "arm_extension": {"value": 160.0, "confidence": 0.9, "coach_eligible": True},
                    "shoulder_turn_change": {"value": 20.0, "confidence": 0.85, "coach_eligible": True},
                }
            },
        }
        result = calibrate_coaching_event(event)
        self.assertEqual(result["status"], "skipped_shadow_swing")
        self.assertIsNone(result["visible_technique_score_9"])


if __name__ == "__main__":
    unittest.main()

