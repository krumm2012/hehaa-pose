import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwen3_tts_sidecar import CoachTtsSidecar


class CoachTtsSidecarTests(unittest.TestCase):
    def test_writes_wav_and_reports_relative_audio_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            requests = []

            def synthesize(event_id, text, target):
                requests.append((event_id, text, target))
                target.write_bytes(b"RIFFfake-wav")
                return {"ok": True, "sample_rate": 24000}

            sidecar = CoachTtsSidecar(
                output_dir=str(root / "coach_audio"),
                audio_path_prefix="coach_audio",
                playback=False,
                synthesizer=synthesize,
                logger=lambda message: None,
            )
            done = threading.Event()
            results = []

            sidecar.submit(
                {
                    "event_id": 7,
                    "stroke_type": "Forehand",
                    "coach_advices": [{"message": "击球后完成随挥"}],
                },
                lambda payload: (results.append(payload), done.set()),
            )
            self.assertTrue(done.wait(2))
            sidecar.close()

            ready = next(item for item in results if item["status"] == "ready")
            self.assertEqual(ready["audio_path"], "coach_audio/swing_007_coach.wav")
            self.assertTrue((root / ready["audio_path"]).is_file())
            self.assertEqual(requests[0][0], 7)
            self.assertIn("第7次Forehand", requests[0][1])

    def test_missing_advice_is_skipped_without_loading_model(self):
        with TemporaryDirectory() as directory:
            sidecar = CoachTtsSidecar(
                output_dir=directory,
                playback=False,
                logger=lambda message: None,
            )
            done = threading.Event()
            payloads = []
            sidecar.submit(
                {"event_id": 1, "coach_advices": []},
                lambda payload: (payloads.append(payload), done.set()),
            )
            self.assertTrue(done.wait(1))
            sidecar.close()

        self.assertEqual(payloads[0]["status"], "skipped")
        self.assertEqual(payloads[0]["reason"], "no_local_coach_advice")


if __name__ == "__main__":
    unittest.main()
