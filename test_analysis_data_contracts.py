import unittest

from analysis_data_contracts import (
    EVENT_LOG_SCHEMA_VERSION,
    FRAME_SCHEMA_VERSION,
    SWING_EVENT_SCHEMA_VERSION,
    build_session_metadata,
    stamp_frame_record,
    stamp_swing_event,
)


class AnalysisDataContractTests(unittest.TestCase):
    def setUp(self):
        self.session = build_session_metadata(
            session_id="court01-test",
            source="rtsp://admin:secret@192.168.1.191:554/camera",
            stream_id="court01-main",
            started_at_unix_ns=1_700_000_000_000_000_000,
            config={"threshold": 0.5},
            models={"detector": "yolo26n.mlpackage"},
        )

    def test_session_metadata_removes_stream_credentials(self):
        self.assertEqual(
            self.session["source"],
            "rtsp://192.168.1.191:554/camera",
        )
        self.assertNotIn("secret", self.session["source"])

    def test_frame_contract_is_additive_and_includes_four_stage_timing(self):
        record = {
            "frame_id": 12,
            "timestamp": 0.48,
            "swing_type": "Forehand",
            "ball": None,
            "rackets": [],
            "pose": None,
            "metrics": {},
        }

        stamped = stamp_frame_record(
            record,
            session=self.session,
            captured_at_unix_ns=1_000_000_000,
            inference_started_at_unix_ns=1_010_000_000,
            inference_completed_at_unix_ns=1_025_000_000,
            analysis_started_at_unix_ns=1_026_000_000,
            analysis_completed_at_unix_ns=1_030_000_000,
        )

        self.assertIs(stamped, record)
        self.assertEqual(stamped["schema_version"], FRAME_SCHEMA_VERSION)
        self.assertEqual(stamped["session_id"], "court01-test")
        self.assertEqual(stamped["swing_type"], "Forehand")
        self.assertEqual(stamped["timing"]["inference_ms"], 15.0)
        self.assertEqual(stamped["timing"]["analysis_ms"], 4.0)
        self.assertEqual(stamped["timing"]["capture_to_analysis_ms"], 30.0)

    def test_event_contract_uses_contact_frame_for_end_to_end_latency(self):
        event = {"event_id": 1, "contact_frame": 12}
        contact = {
            "session_id": "court01-test",
            "timing": {"captured_at_unix_ns": 2_000_000_000},
        }

        stamped = stamp_swing_event(
            event,
            session=self.session,
            emitted_at_unix_ns=2_750_000_000,
            contact_frame_record=contact,
        )

        self.assertEqual(stamped["schema_version"], SWING_EVENT_SCHEMA_VERSION)
        self.assertEqual(stamped["session_id"], "court01-test")
        self.assertEqual(
            stamped["timing"]["contact_capture_to_event_ms"],
            750.0,
        )
        self.assertTrue(EVENT_LOG_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
