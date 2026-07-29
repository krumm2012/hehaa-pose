import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from local_control_panel import (
    ControlSettings,
    LocalPipelineController,
    inject_rtsp_credentials,
)
from roi_stream_config import sanitize_stream_source


class LocalControlPanelTests(unittest.TestCase):
    def make_controller(self, root: Path) -> LocalPipelineController:
        config = root / "config.yaml"
        roi = root / "roi.yaml"
        frontend = root / "index.html"
        config.write_text(
            """
video_input_path: data/input.mp4
video_output_path: data/output.mp4
video_processing:
  output_fps: 25
pipeline_perf:
  inference_workers: 8
roi_settings:
  enabled: true
  crop_margin: 12
realtime_swing:
  analysis_interval_frames: 5
  coach_max_suggestions: 3
  coach_min_confidence: 0.45
""",
            encoding="utf-8",
        )
        roi.write_text(
            """
streams:
  - stream_id: court01-main
    stream_label: Court 01
    stream_source: rtsp://192.168.1.191:554/camera/main
    default: true
    roi_enabled: true
    frame_size: [320, 180]
    roi_points: [[20, 20], [300, 20], [300, 160], [20, 160]]
""",
            encoding="utf-8",
        )
        frontend.write_text("<!doctype html>", encoding="utf-8")
        return LocalPipelineController(
            workspace=root,
            config_path=config,
            roi_config_path=roi,
            frontend_path=frontend,
        )

    def test_injects_credentials_without_changing_camera_identity(self):
        source = inject_rtsp_credentials(
            "rtsp://192.168.1.191:554/camera/main",
            "admin",
            "p@ss word",
        )

        self.assertIn("admin:p%40ss%20word@", source)
        self.assertEqual(
            sanitize_stream_source(source),
            "rtsp://192.168.1.191:554/camera/main",
        )

    def test_settings_enforce_limits_and_coach_event_dependency(self):
        settings = ControlSettings.from_payload(
            {
                "stream_id": "court01-main",
                "session_name": "Court 01 / live",
                "realtime_swing_events": False,
                "realtime_coach": True,
                "max_suggestions": 2,
                "min_confidence": 0.55,
            }
        )

        self.assertEqual(settings.session_name, "Court_01_live")
        self.assertTrue(settings.realtime_swing_events)
        self.assertEqual(settings.max_suggestions, 2)
        self.assertEqual(settings.min_confidence, 0.55)
        with self.assertRaisesRegex(ValueError, "1–3"):
            ControlSettings.from_payload(
                {
                    "stream_id": "court01-main",
                    "max_suggestions": 4,
                }
            )

    def test_public_config_contains_roi_but_no_camera_password(self):
        with TemporaryDirectory() as directory:
            controller = self.make_controller(Path(directory))
            with patch.dict(
                os.environ,
                {
                    "TENNIS_RTSP_USERNAME": "admin",
                    "TENNIS_RTSP_PASSWORD": "private",
                },
            ):
                payload = controller.public_config()

        serialized = str(payload)
        self.assertTrue(payload["credentials_configured"])
        self.assertIn("court01-main", serialized)
        self.assertNotIn("private", serialized)
        self.assertNotIn("admin@", serialized)

    def test_command_has_runtime_controls_and_never_contains_rtsp_secret(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.make_controller(root)
            settings = ControlSettings.from_payload(
                {
                    "stream_id": "court01-main",
                    "output_dir": "outputs",
                    "session_name": "court01",
                    "realtime_coach": True,
                    "max_suggestions": 3,
                    "min_confidence": 0.45,
                    "hdmi_output": True,
                    "display_origin_x": 1512,
                    "display_origin_y": 0,
                }
            )
            runtime_config = root / "runtime.yaml"
            runtime_config.write_text(
                "video_input_path: rtsp://admin:private@192.168.1.191/camera",
                encoding="utf-8",
            )
            output = root / "outputs"
            output.mkdir()

            command, artifacts = controller._build_command(
                settings,
                runtime_config,
                output,
            )

        joined = " ".join(command)
        self.assertIn("--realtime-coach-max-suggestions 3", joined)
        self.assertIn("--hdmi-output", command)
        self.assertIn("--display-origin 1512 0", joined)
        self.assertNotIn("admin:private", joined)
        self.assertNotIn("rtsp://", joined)
        self.assertIn("--session-id", command)
        self.assertIn("--session-output-root", command)
        self.assertIn("--realtime-swing-event-log", command)
        self.assertEqual(artifacts["session_id"], Path(artifacts["session_dir"]).name)
        self.assertIn(artifacts["session_id"], artifacts["event_json"])
        self.assertTrue(artifacts["event_log"].endswith(".jsonl"))
        self.assertTrue(
            artifacts["preview_path"].endswith(
                "court01_swing_report_roi_preview.jpg"
            )
        )

    def test_preview_draws_scaled_roi_and_returns_jpeg(self):
        with TemporaryDirectory() as directory:
            controller = self.make_controller(Path(directory))
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            with patch(
                "local_control_panel.capture_calibration_frame",
                return_value=frame,
            ) as capture:
                image = controller.preview(
                    {
                        "stream_id": "court01-main",
                        "username": "admin",
                        "password": "private",
                    }
                )

            decoded = cv2.imdecode(
                np.frombuffer(image, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )

        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape[:2], (360, 640))
        called_source = capture.call_args.args[0]
        self.assertIn("admin:private@", called_source)
        self.assertGreater(int(decoded[320, 40, 1]), 80)

    def test_output_directory_cannot_escape_workspace(self):
        with TemporaryDirectory() as directory:
            controller = self.make_controller(Path(directory))
            with self.assertRaisesRegex(ValueError, "项目工作区"):
                controller._safe_output_directory("../outside")

    def test_start_and_stop_manage_one_local_process(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.make_controller(root)
            (root / "main_pipe.py").write_text(
                """
import signal
import time

stopped = False
def request_stop(signum, frame):
    global stopped
    stopped = True

signal.signal(signal.SIGINT, request_stop)
print("FAKE_PIPELINE_READY", flush=True)
while not stopped:
    time.sleep(0.02)
print("FAKE_PIPELINE_STOPPED", flush=True)
""",
                encoding="utf-8",
            )
            started = controller.start(
                {
                    "stream_id": "court01-main",
                    "output_dir": "outputs",
                    "session_name": "lifecycle",
                    "realtime_coach": False,
                    "realtime_swing_events": False,
                }
            )
            deadline = time.time() + 2.0
            while (
                "FAKE_PIPELINE_READY"
                not in "\n".join(controller.status()["logs"])
                and time.time() < deadline
            ):
                time.sleep(0.02)

            stopping = controller.stop()
            deadline = time.time() + 2.0
            final = controller.status()
            while (
                final["state"] in {"running", "stopping"}
                and time.time() < deadline
            ):
                time.sleep(0.02)
                final = controller.status()
            controller.shutdown()

        self.assertEqual(started["state"], "running")
        self.assertEqual(stopping["state"], "stopping")
        self.assertEqual(final["state"], "stopped")
        self.assertEqual(final["returncode"], 0)
        self.assertIn(
            "FAKE_PIPELINE_STOPPED",
            "\n".join(final["logs"]),
        )


if __name__ == "__main__":
    unittest.main()
