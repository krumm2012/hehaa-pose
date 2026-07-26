import json
import unittest

from deepseek_evidence_view import build_deepseek_evidence_view
from swing_evidence_builder import build_swing_evidence_packet
from test_swing_evidence_builder import sample_documents


class DeepSeekEvidenceViewTests(unittest.TestCase):
    def test_compacts_packet_but_preserves_one_row_per_event_frame(self):
        frame_document, event_document, coach_document = sample_documents()
        for row in frame_document["frames"]:
            row["detection_diagnostics"]["top_candidates"] = [
                {"position": [100, 200], "confidence": 0.01}
                for _ in range(20)
            ]
        packet = build_swing_evidence_packet(
            frame_document,
            event_document,
            coach_document,
            event_id=1,
        )

        view = build_deepseek_evidence_view(packet)

        self.assertEqual(view["schema_version"], "deepseek_swing_evidence_v1")
        self.assertEqual(view["event_id"], 1)
        self.assertEqual(
            [row["frame_id"] for row in view["frame_sequence"]],
            [10, 11, 12],
        )
        self.assertEqual(view["frame_sequence"][1]["phase"], "contact_candidate")
        self.assertEqual(view["frame_sequence"][1]["motion"]["contact_score"], 0.5)
        self.assertNotIn("top_candidates", json.dumps(view, ensure_ascii=False))
        self.assertLess(
            len(json.dumps(view, ensure_ascii=False)),
            len(json.dumps(packet, ensure_ascii=False)),
        )

    def test_low_quality_event_only_allows_capture_or_review_advice(self):
        frame_document, event_document, coach_document = sample_documents()
        event_document["events"][0]["quality_flags"] = {
            "warnings": ["pose_gaps", "ball_track_gaps", "racket_track_gaps"],
            "review_recommended": True,
            "pose_frame_ratio": 0.4,
            "ball_frame_ratio": 0.4,
            "racket_frame_ratio": 0.5,
        }
        coach_event = coach_document["events"][0]
        coach_event["quality_flags"] = event_document["events"][0]["quality_flags"]
        coach_event["data_quality"]["pose_frame_ratio"] = 0.4
        coach_event["data_quality"]["missing_fields"] = [
            "ball.estimated_spin",
            "ball.landing_point",
            "racket.racket_face_angle_deg",
        ]
        packet = build_swing_evidence_packet(
            frame_document,
            event_document,
            coach_document,
            event_id=1,
        )

        view = build_deepseek_evidence_view(packet)
        policy = view["decision_policy"]

        self.assertTrue(policy["review_required"])
        self.assertEqual(
            policy["allowed_advice_categories"],
            ["capture", "review"],
        )
        self.assertIn("spin", policy["prohibited_claims"])
        self.assertIn("landing", policy["prohibited_claims"])
        self.assertIn("racket_face", policy["prohibited_claims"])
        self.assertIn("professional_speed_comparison", policy["prohibited_claims"])

    def test_object_tracking_warnings_do_not_block_body_coaching(self):
        frame_document, event_document, coach_document = sample_documents()
        quality = {
            "warnings": ["racket_track_gaps", "static_ball_mask_in_event"],
            "review_recommended": True,
            "pose_frame_ratio": 1.0,
            "ball_frame_ratio": 0.42,
            "racket_frame_ratio": 0.5,
        }
        event_document["events"][0]["quality_flags"] = quality
        coach_event = coach_document["events"][0]
        coach_event["quality_flags"] = quality
        coach_event["body"].update(
            {
                "hip_shoulder_separation_at_contact": 8.0,
                "unit_turn_quality": "adequate",
                "balance_state": "moving",
            }
        )
        coach_event["scores"]["power_transfer_score"] = 0.1
        coach_event["data_quality"]["pose_frame_ratio"] = 1.0
        coach_event["data_quality"]["event_quality_flags"] = quality
        packet = build_swing_evidence_packet(
            frame_document,
            event_document,
            coach_document,
            event_id=1,
        )

        policy = build_deepseek_evidence_view(packet)["decision_policy"]

        self.assertTrue(policy["coaching_allowed"])
        self.assertIn("technique", policy["allowed_advice_categories"])
        self.assertIn("power_transfer", policy["allowed_advice_topics"])
        self.assertIn("racket_face", policy["blocked_advice_topics"])
        self.assertIn("ball_trajectory", policy["blocked_advice_topics"])
        self.assertEqual(
            policy["advice_candidates"][0]["focus"],
            "power_transfer",
        )

    def test_intermittent_ball_detection_is_tolerated_when_contact_is_covered(self):
        frame_document, event_document, coach_document = sample_documents()
        quality = {
            "warnings": ["ball_track_gaps"],
            "review_recommended": True,
            "ball_frame_ratio": 0.42,
            "ball_contact_window_ratio": 0.67,
        }
        event_document["events"][0]["quality_flags"] = quality
        event_document["events"][0]["coach_advice"] = {
            "code": "ball_track_gaps",
            "message": "确保来球完整入镜",
        }
        event_document["events"][0]["deepseek_advice"] = {
            "focus": "ball_track_gaps",
            "message": "确保来球完整入镜",
        }
        coach_document["events"][0]["quality_flags"] = quality
        coach_document["events"][0]["diagnosis_tags"] = [
            "ball_track_gaps",
            "short_follow_through",
        ]
        coach_document["events"][0]["data_quality"]["event_quality_flags"] = quality
        packet = build_swing_evidence_packet(
            frame_document,
            event_document,
            coach_document,
            event_id=1,
        )

        view = build_deepseek_evidence_view(packet)
        policy = view["decision_policy"]

        self.assertFalse(policy["review_required"])
        self.assertEqual(
            policy["allowed_advice_categories"],
            ["technique", "positive", "review"],
        )
        self.assertNotIn("ball_track_gaps", policy["effective_warnings"])
        self.assertIn(
            "intermittent_ball_detection",
            policy["tolerated_conditions"],
        )
        self.assertNotIn("coach_advice", view["event"])
        self.assertNotIn("deepseek_advice", view["event"])
        self.assertNotIn("ball_track_gaps", view["coach_metrics"]["diagnosis_tags"])
        serialized = json.dumps(view, ensure_ascii=False)
        self.assertNotIn("确保来球完整入镜", serialized)


if __name__ == "__main__":
    unittest.main()
