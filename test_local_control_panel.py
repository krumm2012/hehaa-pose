import json
import io
import os
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from local_control_panel import (
    ControlSettings,
    LocalPipelineController,
    create_handler,
    inject_rtsp_credentials,
    load_local_environment_variable,
)
from roi_stream_config import sanitize_stream_source


class LocalControlPanelTests(unittest.TestCase):
    def test_video_upload_http_requires_control_token(self):
        with TemporaryDirectory() as directory:
            controller = self.make_controller(Path(directory))
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(controller))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/api/video/upload"
                request = urllib.request.Request(url, data=b"video", method="POST")
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request)
                self.assertEqual(error.exception.code, 403)
                request.add_header("X-Control-Token", controller.token)
                request.add_header("X-Video-Suffix", ".avi")
                with patch.object(controller, "upload_video", return_value={"video_id": "uploaded"}) as upload:
                    with urllib.request.urlopen(request) as response:
                        self.assertEqual(json.load(response)["video_id"], "uploaded")
                    self.assertEqual(upload.call_args.args[1:], (5, ".avi"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_local_video_upload_preview_and_offline_settings(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.make_controller(root)
            sample = root / "sample.avi"
            writer = cv2.VideoWriter(str(sample), cv2.VideoWriter_fourcc(*"MJPG"), 15, (320, 180))
            for _ in range(10):
                writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
            writer.release()
            data = sample.read_bytes()
            result = controller.upload_video(io.BytesIO(data), len(data), ".avi")
            payload = {"stream_id": "local_video", "video_id": result["video_id"],
                       "live_mode": True, "roi_enabled": True}
            stream = controller._stream_from_payload(payload)
            self.assertTrue(Path(stream["source"]).is_file())
            self.assertFalse(stream["roi_enabled"])
            self.assertTrue(controller.preview(payload).startswith(b"\xff\xd8"))
            settings = ControlSettings.from_payload(payload)
            self.assertFalse(settings.live_mode)
            self.assertFalse(settings.roi_enabled)
            command, _ = controller._build_command(settings, root / "runtime.yaml", root / "out")
            self.assertNotIn("--drop-stale-frames", command)
            self.assertNotIn("--live-mode", command)

    def test_local_video_rejects_invalid_uploads_and_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.make_controller(root)
            for data, length, suffix in [(b"", 0, ".mp4"), (b"x", 1, ".py"),
                                         (b"x", 10, ".mp4"), (b"garbage", 7, ".mp4"),
                                         (b"", 3 * 1024 ** 3, ".mp4")]:
                with self.assertRaises(ValueError):
                    controller.upload_video(io.BytesIO(data), length, suffix)
            self.assertEqual(list((root / "data/control_uploads").glob("*")), [])
            for video_id in ["../../config.yaml", "", "a" * 32 + ".mp4"]:
                with self.assertRaises(ValueError):
                    controller._stream_from_payload({"stream_id": "local_video", "video_id": video_id})

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

    def test_loads_only_requested_secret_from_local_env_file(self):
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env.local"
            env_path.write_text(
                "# Local secrets\n"
                "BROKEN_UNRELATED='\n"
                "export DEEPSEEK_API_KEY='local test key' # comment\n"
                "UNRELATED_SECRET=do-not-load\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                loaded = load_local_environment_variable(
                    env_path,
                    "DEEPSEEK_API_KEY",
                )

                self.assertTrue(loaded)
                self.assertEqual(
                    os.environ["DEEPSEEK_API_KEY"],
                    "local test key",
                )
                self.assertNotIn("UNRELATED_SECRET", os.environ)

    def test_local_env_never_overrides_existing_secret(self):
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env.local"
            env_path.write_text(
                "DEEPSEEK_API_KEY=file-key\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "process-key"},
                clear=True,
            ):
                loaded = load_local_environment_variable(
                    env_path,
                    "DEEPSEEK_API_KEY",
                )

                self.assertFalse(loaded)
                self.assertEqual(
                    os.environ["DEEPSEEK_API_KEY"],
                    "process-key",
                )

    def test_settings_enforce_limits_and_coach_event_dependency(self):
        settings = ControlSettings.from_payload(
            {
                "stream_id": "court01-main",
                "session_name": "Court 01 / live",
                "realtime_swing_events": False,
                "realtime_coach": True,
                "coach_tts": True,
                "coach_tts_playback": False,
                "max_suggestions": 2,
                "min_confidence": 0.55,
            }
        )

        self.assertEqual(settings.session_name, "Court_01_live")
        self.assertTrue(settings.realtime_swing_events)
        self.assertTrue(settings.evidence_bundle)
        self.assertEqual(settings.max_suggestions, 2)
        self.assertEqual(settings.min_confidence, 0.55)
        self.assertTrue(settings.coach_tts)
        self.assertFalse(settings.coach_tts_playback)
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

    def test_local_camera_file_authenticates_without_browser_input(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            (root / ".camera-credentials.local.env").write_text(
                "TENNIS_RTSP_USERNAME='admin'\nTENNIS_RTSP_PASSWORD='test # pass'\n",
                encoding="utf-8",
            )
            controller = self.make_controller(root)
            source = controller._authenticated_source({"source": "rtsp://127.0.0.1/live"}, {})
            self.assertEqual(source, "rtsp://admin:test%20%23%20pass@127.0.0.1/live")
            config = controller.public_config()
            self.assertTrue(config["credentials_configured"])
            self.assertNotIn("test # pass", str(config))
            overridden = controller._authenticated_source(
                {"source": "rtsp://127.0.0.1/live"}, {"username": "other", "password": "manual"}
            )
            self.assertEqual(overridden, "rtsp://other:manual@127.0.0.1/live")

    def test_empty_camera_password_does_not_claim_ready(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            (root / ".camera-credentials.local.env").write_text(
                "TENNIS_RTSP_USERNAME='admin'\nTENNIS_RTSP_PASSWORD=''\n", encoding="utf-8"
            )
            self.assertFalse(self.make_controller(root).public_config()["credentials_configured"])

    def test_camera_specific_credentials_and_common_fallback(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            (root / ".camera-credentials.local.env").write_text(
                "TENNIS_RTSP_USERNAME='shared'\nTENNIS_RTSP_PASSWORD='shared-pass'\n"
                "TENNIS_CAMERA_COURT01_MAIN_USERNAME='first'\n"
                "TENNIS_CAMERA_COURT01_MAIN_PASSWORD='first-pass'\n",
                encoding="utf-8",
            )
            controller = self.make_controller(root)
            first = {"stream_id": "court01-main", "source": "rtsp://127.0.0.1/ch1"}
            second = {"stream_id": "court02-main", "source": "rtsp://127.0.0.1/ch2"}
            self.assertEqual(controller._authenticated_source(first, {}), "rtsp://first:first-pass@127.0.0.1/ch1")
            self.assertEqual(controller._authenticated_source(second, {}), "rtsp://shared:shared-pass@127.0.0.1/ch2")
            os.environ["TENNIS_CAMERA_COURT02_MAIN_USERNAME"] = "second"
            os.environ["TENNIS_CAMERA_COURT02_MAIN_PASSWORD"] = "second-pass"
            self.assertEqual(controller._authenticated_source(second, {}), "rtsp://second:second-pass@127.0.0.1/ch2")
            os.environ.pop("TENNIS_CAMERA_COURT02_MAIN_PASSWORD")
            self.assertEqual(controller._saved_credentials("court02-main"), ("second", ""))
            config = controller.public_config()
            self.assertTrue(config["camera_credentials_configured"]["court01-main"])
            self.assertNotIn("first-pass", str(config))

    def test_custom_stream_reuses_matching_camera_credentials(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {
            "TENNIS_CAMERA_COURT01_MAIN_USERNAME": "first",
            "TENNIS_CAMERA_COURT01_MAIN_PASSWORD": "secret",
        }, clear=True):
            controller = self.make_controller(Path(directory))
            source = controller.streams[0]["source"]
            stream = controller._stream_from_payload({"stream_id": "custom", "custom_stream_source": source})
            self.assertEqual(stream["credentials_stream_id"], "court01-main")
            self.assertIn("first:secret@", controller._authenticated_source(stream, {}))

    def test_serves_manual_review_api_for_current_session(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.make_controller(root)
            session = root / "outputs" / "session-001"
            session.mkdir(parents=True)
            (session / "live_swing_events.json").write_text(
                json.dumps({"summary": {}, "events": []}),
                encoding="utf-8",
            )
            (session / "live_swing_frames.jsonl").write_text(
                "",
                encoding="utf-8",
            )
            (session / "live_swing_report.html").write_text(
                "<!doctype html>",
                encoding="utf-8",
            )
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                create_handler(controller),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            origin = f"http://127.0.0.1:{server.server_port}"
            try:
                state_request = urllib.request.Request(
                    f"{origin}/api/manual-review/state",
                    headers={
                        "X-Manual-Review-Report": (
                            "outputs/session-001/live_swing_report.html"
                        )
                    },
                )
                with urllib.request.urlopen(state_request, timeout=2) as response:
                    state = json.load(response)

                unknown_request = urllib.request.Request(
                    f"{origin}/api/not-a-real-route",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as unknown_error:
                    urllib.request.urlopen(unknown_request, timeout=2)
                unknown_error.exception.close()

                request = urllib.request.Request(
                    f"{origin}/api/manual-review/evaluate",
                    data=json.dumps(
                        {
                            "schema_version": "swing_manual_annotations_v2",
                            "events": [],
                        }
                    ).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "X-Control-Token": controller.token,
                        "X-Manual-Review-Report": (
                            "outputs/session-001/live_swing_report.html"
                        ),
                    },
                    method="POST",
                )
                with patch(
                    "local_control_panel.process_manual_review",
                    return_value={"status": "finalized"},
                ):
                    with urllib.request.urlopen(request, timeout=2) as response:
                        evaluated = json.load(response)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

        self.assertEqual(state["status"], "waiting_for_annotations")
        self.assertEqual(unknown_error.exception.code, 404)
        self.assertEqual(evaluated["status"], "finalized")

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
                    "coach_tts": True,
                    "coach_tts_playback": False,
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
        self.assertIn("--realtime-coach-tts", command)
        self.assertIn("--realtime-coach-tts-no-playback", command)
        self.assertIn("--hdmi-output", command)
        self.assertIn("--display-origin 1512 0", joined)
        self.assertNotIn("admin:private", joined)
        self.assertNotIn("rtsp://", joined)
        self.assertIn("--session-id", command)
        self.assertIn("--session-output-root", command)
        self.assertIn("--realtime-swing-event-log", command)
        self.assertIn("--realtime-frame-output", command)
        self.assertIn("--evidence-manifest", command)
        self.assertEqual(artifacts["session_id"], Path(artifacts["session_dir"]).name)
        self.assertIn(artifacts["session_id"], artifacts["event_json"])
        self.assertTrue(artifacts["event_log"].endswith(".jsonl"))
        self.assertTrue(artifacts["frame_jsonl"].endswith("_frames.jsonl"))
        self.assertTrue(
            artifacts["evidence_manifest"].endswith(
                "_evidence_manifest.json"
            )
        )
        self.assertIn("/artifacts/", artifacts["evidence_manifest_url"])
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

    def test_custom_stream_is_ephemeral_and_reuses_matching_roi(self):
        with TemporaryDirectory() as directory:
            controller = self.make_controller(Path(directory))

            stream = controller._stream_from_payload(
                {
                    "stream_id": "custom",
                    "custom_stream_source": (
                        "rtsp://192.168.1.191:554/camera/main?transport=tcp"
                    ),
                }
            )
            authenticated = controller._authenticated_source(
                stream,
                {"username": "operator", "password": "private"},
            )

        self.assertEqual(stream["label"], "自定义码流")
        self.assertEqual(
            stream["source"],
            "rtsp://192.168.1.191:554/camera/main?transport=tcp",
        )
        self.assertEqual(stream["points"], [[20, 20], [300, 20], [300, 160], [20, 160]])
        self.assertIn("operator:private@", authenticated)
        self.assertIn("transport=tcp", authenticated)

    def test_custom_stream_requires_supported_credential_free_url(self):
        with TemporaryDirectory() as directory:
            controller = self.make_controller(Path(directory))

            for source in ("", "file:///tmp/video.mp4", "ftp://camera/live"):
                with self.subTest(source=source):
                    with self.assertRaisesRegex(ValueError, "自定义码流"):
                        controller._stream_from_payload(
                            {
                                "stream_id": "custom",
                                "custom_stream_source": source,
                            }
                        )
            with self.assertRaisesRegex(ValueError, "用户名和密码输入框"):
                controller._stream_from_payload(
                    {
                        "stream_id": "custom",
                        "custom_stream_source": "rtsp://admin:secret@camera/live",
                    }
                )

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

    def test_command_includes_algo2_dual_view_when_requested(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.make_controller(root)
            settings_on = ControlSettings.from_payload(
                {
                    "stream_id": "court01-main",
                    "output_dir": "outputs",
                    "session_name": "court01",
                    "algo2_dual_view": True,
                }
            )
            settings_off = ControlSettings.from_payload(
                {
                    "stream_id": "court01-main",
                    "output_dir": "outputs",
                    "session_name": "court01",
                    "algo2_dual_view": False,
                }
            )
            runtime_config = root / "runtime.yaml"
            runtime_config.write_text("video_input_path: rtsp://127.0.0.1/camera", encoding="utf-8")
            output = root / "outputs"
            output.mkdir()

            cmd_on, _ = controller._build_command(settings_on, runtime_config, output)
            cmd_off, _ = controller._build_command(settings_off, runtime_config, output)

            self.assertIn("--algo2-dual-view", cmd_on)
            self.assertNotIn("--no-dual-view", cmd_on)
            self.assertIn("--no-dual-view", cmd_off)
            self.assertNotIn("--algo2-dual-view", cmd_off)


if __name__ == "__main__":
    unittest.main()
