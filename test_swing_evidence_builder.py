import unittest

from swing_evidence_builder import build_swing_evidence_packet


def sample_documents():
    frames = [
        {
            "frame_id": frame_id,
            "timestamp": frame_id / 25.0,
            "swing_type": "Forehand",
            "ball": [frame_id * 2, 100],
            "rackets": [],
            "pose": {"right_wrist": [frame_id, 100]},
            "metrics": {},
            "detection_diagnostics": {"final_decision": "selected"},
        }
        for frame_id in range(9, 14)
    ]
    features = [
        {
            "frame_id": frame_id,
            "timestamp": frame_id / 25.0,
            "wrist_speed": float(frame_id),
            "contact_score": 0.5 if frame_id == 11 else 0.0,
        }
        for frame_id in range(9, 14)
    ]
    traces = [
        {
            "frame": frame_id,
            "event_id": 1,
            "phase": "contact_candidate" if frame_id == 11 else "forward_swing",
            "motion_energy": float(frame_id * 2),
        }
        for frame_id in range(9, 14)
    ]
    event = {
        "event_id": 1,
        "start_frame": 10,
        "contact_frame": 11,
        "peak_frame": 11,
        "end_frame": 12,
        "duration_frames": 3,
        "stroke_type": "Forehand",
        "confidence": 0.88,
        "quality_flags": {"warnings": []},
        "phase_counts": {"forward_swing": 2, "contact_candidate": 1},
    }
    coach_event = {
        "event_id": 1,
        "stroke_type": "Forehand",
        "confidence": 0.88,
        "frames": {"start": 10, "contact": 11, "end": 12},
        "ball": {"contact_confidence": 0.5},
        "racket": {"max_racket_speed": 24.0},
        "body": {"arm_extension_at_contact": 140.0},
        "timing": {"duration_seconds": 0.12},
        "scores": {"overall_score": 0.7},
        "diagnosis_tags": [],
        "data_quality": {"missing_fields": []},
        "frame_trace": [{"frame_id": 10, "phase": "forward_swing"}],
    }
    frame_document = {
        "video_info": {
            "path": "sample.mp4",
            "fps": 25.0,
            "resolution": [1920, 1080],
        },
        "frames": frames,
        "summary": {"total_frames": 5},
    }
    event_document = {
        "summary": {
            "total_frames": 5,
            "thresholds": {"dominant_hand": "right"},
        },
        "events": [event],
        "features": features,
        "frame_trace": traces,
    }
    coach_document = {
        "metadata": {
            "schema_version": "coach_dataset_v1.1",
            "video_path": "sample.mp4",
            "fps": 25.0,
            "resolution": [1920, 1080],
            "total_frames": 5,
        },
        "events": [coach_event],
    }
    return frame_document, event_document, coach_document


class SwingEvidenceBuilderTests(unittest.TestCase):
    def test_builds_aligned_packet_for_one_event(self):
        frame_document, event_document, coach_document = sample_documents()

        packet = build_swing_evidence_packet(
            frame_document,
            event_document,
            coach_document,
            event_id=1,
            player_context={"level": "intermediate"},
        )

        self.assertEqual(packet["schema_version"], "swing_evidence_packet_v1")
        self.assertEqual(packet["event_id"], 1)
        self.assertEqual(packet["video_context"]["fps"], 25.0)
        self.assertEqual(packet["video_context"]["dominant_hand"], "right")
        self.assertEqual(packet["video_context"]["coordinate_space"], "screen_pixels")
        self.assertEqual(packet["player_context"], {"level": "intermediate"})
        self.assertEqual(
            [row["frame_id"] for row in packet["event_frame_records"]],
            [10, 11, 12],
        )
        self.assertEqual(
            [row["frame_id"] for row in packet["motion_features"]],
            [10, 11, 12],
        )
        self.assertEqual(
            [row["frame"] for row in packet["frame_trace"]],
            [10, 11, 12],
        )
        self.assertNotIn("frame_trace", packet["coach_metrics"])
        self.assertTrue(packet["integrity"]["aligned_to_event_range"])
        self.assertEqual(packet["integrity"]["missing_frame_record_ids"], [])
        self.assertEqual(packet["integrity"]["missing_motion_feature_ids"], [])
        self.assertEqual(packet["integrity"]["missing_frame_trace_ids"], [])

    def test_reports_missing_rows_without_hiding_partial_evidence(self):
        frame_document, event_document, coach_document = sample_documents()
        frame_document["frames"] = [
            row for row in frame_document["frames"] if row["frame_id"] != 11
        ]
        event_document["features"] = [
            row for row in event_document["features"] if row["frame_id"] != 12
        ]

        packet = build_swing_evidence_packet(
            frame_document,
            event_document,
            coach_document,
            event_id=1,
        )

        self.assertFalse(packet["integrity"]["aligned_to_event_range"])
        self.assertEqual(packet["integrity"]["missing_frame_record_ids"], [11])
        self.assertEqual(packet["integrity"]["missing_motion_feature_ids"], [12])
        self.assertEqual(packet["integrity"]["missing_frame_trace_ids"], [])

    def test_rejects_unknown_event_id(self):
        frame_document, event_document, coach_document = sample_documents()

        with self.assertRaisesRegex(ValueError, "event_id 99"):
            build_swing_evidence_packet(
                frame_document,
                event_document,
                coach_document,
                event_id=99,
            )

    def test_rejects_mismatched_coach_event_range(self):
        frame_document, event_document, coach_document = sample_documents()
        coach_document["events"][0]["frames"]["end"] = 13

        with self.assertRaisesRegex(ValueError, "frame range"):
            build_swing_evidence_packet(
                frame_document,
                event_document,
                coach_document,
                event_id=1,
            )


if __name__ == "__main__":
    unittest.main()
