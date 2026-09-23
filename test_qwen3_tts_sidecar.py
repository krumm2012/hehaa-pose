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

    def test_single_core_advice_selection(self):
        event = {
            "event_id": 2,
            "stroke_type": "Forehand",
            "coach_advices": [
                {"message": "准备时适当降低重心", "priority": 1},
                {"message": "击球前拍头下潜刷球", "priority": 2},
            ],
        }
        text_single = CoachTtsSidecar._speech_text(event, single_core_advice=True)
        self.assertEqual(text_single, "第2次Forehand：准备时适当降低重心")
        text_multi = CoachTtsSidecar._speech_text(event, single_core_advice=False)
        self.assertIn("准备时适当降低重心。击球前拍头下潜刷球", text_multi)

    def test_preemption_cancels_older_queued_task(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            synthesizing = threading.Event()
            block_synthesis = threading.Event()

            def slow_synthesize(event_id, text, target):
                synthesizing.set()
                block_synthesis.wait(1.0)
                target.write_bytes(b"RIFFfake-wav")
                return {"ok": True, "sample_rate": 24000}

            sidecar = CoachTtsSidecar(
                output_dir=str(root / "coach_audio"),
                playback=False,
                max_pending=5,
                preempt=True,
                synthesizer=slow_synthesize,
                logger=lambda message: None,
            )

            results_1 = []
            results_2 = []
            results_3 = []
            # Event 1 starts synthesizing
            sidecar.submit(
                {"event_id": 1, "stroke_type": "Forehand", "coach_advices": [{"message": "Advice 1"}]},
                lambda p: results_1.append(p),
            )
            synthesizing.wait(1.0)

            # Event 2 enters queue while 1 is busy
            sidecar.submit(
                {"event_id": 2, "stroke_type": "Forehand", "coach_advices": [{"message": "Advice 2"}]},
                lambda p: results_2.append(p),
            )

            # Event 3 arrives -> should preempt Event 2 immediately
            sidecar.submit(
                {"event_id": 3, "stroke_type": "Forehand", "coach_advices": [{"message": "Advice 3"}]},
                lambda p: results_3.append(p),
            )

            # Check that Event 2 was preempted
            self.assertEqual(len(results_2), 1)
            self.assertEqual(results_2[0]["status"], "preempted")
            self.assertEqual(results_2[0]["reason"], "superseded_by_newer_swing")

            block_synthesis.set()
            sidecar.close()


if __name__ == "__main__":
    unittest.main()

