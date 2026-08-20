import unittest

from swing_session_quality import build_session_quality_dashboard


def event(
    event_id,
    score,
    evidence_quality,
    body_width=100.0,
    warnings=None,
    advice_code=None,
):
    return {
        "event_id": event_id,
        "start_frame": event_id * 20,
        "contact_frame": event_id * 20 + 5,
        "end_frame": event_id * 20 + 10,
        "stroke_type": "Forehand",
        "quality_flags": {
            "pose_frame_ratio": evidence_quality,
            "ball_frame_ratio": evidence_quality,
            "racket_frame_ratio": evidence_quality,
            "review_recommended": evidence_quality < 0.5,
            "warnings": warnings or [],
        },
        "coach_calibration": {
            "status": "calibrated",
            "visible_technique_score_9": score,
            "uncertainty_9": 0.6,
            "confidence": 0.8,
        },
        "biomechanics": {
            "reference_body_width_px": body_width,
            "quality": {"contact_evidence_confidence": evidence_quality},
            "metrics": {
                "arm_extension": {
                    "value": 120.0,
                    "coach_eligible": True,
                },
                "shoulder_turn_change": {
                    "value": 20.0,
                    "coach_eligible": True,
                },
                "preparation_knee_flexion": {
                    "value": 15.0,
                    "coach_eligible": True,
                },
                "weight_transfer": {
                    "value": 0.4,
                    "coach_eligible": False,
                },
            },
        },
        "coach_advices": (
            [{"code": advice_code, "message": "建议"}] if advice_code else []
        ),
    }


class SwingSessionQualityTests(unittest.TestCase):
    def test_short_session_exposes_trend_without_claiming_drift(self):
        dashboard = build_session_quality_dashboard(
            [event(1, 6.0, 0.8), event(2, 5.5, 0.7), event(3, 5.0, 0.6)]
        )

        self.assertEqual(dashboard["monitoring_state"], "warming_up")
        self.assertFalse(dashboard["drift"]["ready"])
        self.assertEqual(dashboard["drift"]["status"], "warming_up")
        self.assertEqual(len(dashboard["series"]), 3)
        score_indicator = next(
            row
            for row in dashboard["drift"]["indicators"]
            if row["name"] == "visible_technique_score"
        )
        self.assertEqual(score_indicator["status"], "insufficient_events")
        self.assertLess(score_indicator["delta"], 0.0)

    def test_ready_session_separates_technique_and_capture_decline(self):
        dashboard = build_session_quality_dashboard(
            [
                event(1, 7.0, 0.9),
                event(2, 7.2, 0.9),
                event(3, 6.5, 0.7),
                event(4, 5.9, 0.55),
                event(5, 5.0, 0.4, warnings=["racket_track_gaps"]),
                event(6, 4.8, 0.35, warnings=["racket_track_gaps"]),
            ]
        )

        self.assertTrue(dashboard["drift"]["ready"])
        self.assertEqual(dashboard["drift"]["status"], "attention")
        indicators = {
            row["name"]: row for row in dashboard["drift"]["indicators"]
        }
        self.assertEqual(indicators["visible_technique_score"]["status"], "declining")
        self.assertEqual(indicators["evidence_quality"]["status"], "declining")
        domains = {alert["domain"] for alert in dashboard["alerts"]}
        self.assertIn("technique", domains)
        self.assertIn("capture", domains)

    def test_camera_scale_shift_confounds_technique_conclusion(self):
        dashboard = build_session_quality_dashboard(
            [
                event(1, 7.0, 0.9, body_width=100),
                event(2, 7.1, 0.9, body_width=102),
                event(3, 6.5, 0.9, body_width=115),
                event(4, 5.8, 0.9, body_width=125),
                event(5, 5.0, 0.9, body_width=140),
                event(6, 4.8, 0.9, body_width=142),
            ]
        )

        indicators = {
            row["name"]: row for row in dashboard["drift"]["indicators"]
        }
        self.assertEqual(indicators["camera_scale"]["status"], "shifted")
        self.assertEqual(
            indicators["visible_technique_score"]["status"],
            "camera_shift_confounded",
        )
        self.assertTrue(dashboard["drift"]["camera_confounded"])
        self.assertIn(
            "camera_scale_shift",
            [alert["code"] for alert in dashboard["alerts"]],
        )

    def test_recurring_warnings_and_advice_are_counted(self):
        dashboard = build_session_quality_dashboard(
            [
                event(
                    1,
                    6.0,
                    0.7,
                    warnings=["racket_track_gaps"],
                    advice_code="limited_arm_extension",
                ),
                event(
                    2,
                    6.1,
                    0.7,
                    warnings=["racket_track_gaps"],
                    advice_code="limited_arm_extension",
                ),
                event(3, 6.2, 0.7),
            ]
        )

        self.assertEqual(
            dashboard["recurring"]["warning_counts"]["racket_track_gaps"],
            2,
        )
        self.assertEqual(
            dashboard["recurring"]["advice_counts"]["limited_arm_extension"],
            2,
        )
        self.assertEqual(
            dashboard["recurring"]["warnings"][0]["code"],
            "racket_track_gaps",
        )

    def test_overlapping_event_ranges_block_drift_conclusions(self):
        events = [event(index, 6.0, 0.8) for index in range(1, 7)]
        events[3]["start_frame"] = events[2]["end_frame"] - 3

        dashboard = build_session_quality_dashboard(events)

        self.assertFalse(dashboard["drift"]["ready"])
        self.assertEqual(dashboard["drift"]["status"], "integrity_blocked")
        self.assertEqual(
            dashboard["integrity"]["overlapping_event_pair_count"],
            1,
        )
        self.assertIn(
            "overlapping_event_ranges",
            [alert["code"] for alert in dashboard["alerts"]],
        )


if __name__ == "__main__":
    unittest.main()
