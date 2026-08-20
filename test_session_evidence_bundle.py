import json
import tempfile
import unittest
from pathlib import Path

from analysis_data_contracts import build_session_metadata
from session_evidence_bundle import (
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
    replay_evidence_manifest,
    verify_evidence_manifest,
    write_session_evidence_manifest,
)


class SessionEvidenceBundleTests(unittest.TestCase):
    def _write_bundle(self, root: Path) -> Path:
        frames = root / "session_frames.jsonl"
        frames.write_text(
            "".join(
                json.dumps(
                    {
                        "frame_id": frame_id,
                        "timestamp": frame_id / 25,
                        "swing_type": "Ready",
                        "ball": None,
                        "rackets": [],
                        "pose": None,
                        "metrics": {},
                    }
                )
                + "\n"
                for frame_id in range(12)
            ),
            encoding="utf-8",
        )
        events = root / "session_swing_events.json"
        events.write_text(
            json.dumps({"summary": {"swing_event_count": 0}, "events": []}),
            encoding="utf-8",
        )
        manifest = root / "session_evidence_manifest.json"
        session = build_session_metadata(
            session_id="court01-test",
            source="rtsp://admin:secret@192.168.1.191/camera",
        )
        write_session_evidence_manifest(
            str(manifest),
            session=session,
            status="completed",
            capture={"fps": 25.0, "resolution": [320, 180]},
            replay={
                "fps": 25.0,
                "analysis_interval_frames": 5,
                "settle_frames": 15,
                "window_frames": 200,
                "swing_options": {
                    "dominant_hand": "right",
                    "min_peak_energy": 9.0,
                    "active_energy": 5.5,
                    "min_event_frames": 8,
                    "max_internal_gap": 3,
                    "min_event_gap": 18,
                },
                "coach": {"enabled": False},
            },
            artifacts=[
                {"role": "frame_journal", "path": frames, "required_for_replay": True},
                {"role": "event_snapshot", "path": events, "required_for_replay": True},
            ],
        )
        return manifest

    def test_writes_sanitized_checksummed_replayable_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_bundle(Path(directory))
            document = json.loads(manifest.read_text(encoding="utf-8"))
            verification = verify_evidence_manifest(str(manifest))

        self.assertEqual(document["schema_version"], EVIDENCE_BUNDLE_SCHEMA_VERSION)
        self.assertEqual(document["session"]["source"], "rtsp://192.168.1.191/camera")
        self.assertNotIn("secret", json.dumps(document))
        self.assertTrue(document["completeness"]["replayable"])
        self.assertTrue(verification["valid"])
        self.assertTrue(verification["replayable"])

    def test_detects_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_bundle(root)
            (root / "session_frames.jsonl").write_text("tampered\n", encoding="utf-8")

            verification = verify_evidence_manifest(str(manifest))

        self.assertFalse(verification["valid"])
        self.assertFalse(verification["replayable"])

    def test_replays_frame_journal_and_matches_expected_event_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_bundle(Path(directory))

            result = replay_evidence_manifest(str(manifest))

        self.assertEqual(result["frame_record_count"], 12)
        self.assertTrue(result["semantic_match"])
        self.assertEqual(result["expected_event_signatures"], [])
        self.assertEqual(result["replayed_event_signatures"], [])


if __name__ == "__main__":
    unittest.main()
