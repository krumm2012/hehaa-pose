import json
import hashlib
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
import yaml

from local_control_panel import (
    ControlSettings,
    LocalPipelineController,
    create_handler,
    inject_rtsp_credentials,
    load_local_environment_variable,
)
from roi_stream_config import sanitize_stream_source


class LocalControlPanelTests(unittest.TestCase):
    def test_status_http_identifies_loaded_service_without_rehashing_edited_sources(self):
        with TemporaryDirectory() as directory:
            controller = self.make_controller(Path(directory))
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(controller))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/api/status"
                with urllib.request.urlopen(url) as response:
                    status = json.load(response)
                build = status['service_build']
                source = Path(__file__).parent / 'local_control_panel.py'
                self.assertEqual(build['control_source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
                self.assertEqual(build['schema'], 'tennis.control-service-build.v1')
                self.assertEqual(build['pid'], os.getpid())
                self.assertGreater(build['loaded_at_unix_ns'], 0)
                self.assertFalse(build['analysis_build']['accuracy_validated'])
                # A running process must report its import-time identity, not files edited later.
                with patch.object(Path, 'read_bytes', side_effect=AssertionError('status must not rehash sources')):
                    with urllib.request.urlopen(url) as response:
                        self.assertEqual(json.load(response)['service_build'], build)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

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

    def test_session_events_discovery_and_url_resolution(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.make_controller(root)
            session_dir = root / "data" / "analysis_results" / "control_panel" / "test_session_123"
            clips_dir = session_dir / "test_session_swing_clips"
            clips_dir.mkdir(parents=True)
            clip_file = clips_dir / "event_0001.mp4"
            clip_file.write_bytes(b"dummy_mp4_bytes")
            freeze_file = clips_dir / "event_0001_freeze.jpg"
            freeze_file.write_bytes(b"dummy_jpg_bytes")
            events_data = {
                "events": [
                    {
                        "event_id": 1,
                        "stroke_type": "Forehand",
                        "clip_path": "test_session_swing_clips/event_0001.mp4",
                        "impact_freeze_path": "test_session_swing_clips/event_0001_freeze.jpg",
                    }
                ]
            }
            events_json = session_dir / "test_session_swing_events.json"
            events_json.write_text(json.dumps(events_data), encoding="utf-8")

            res = controller.session_events()
            self.assertEqual(res["session_id"], "test_session_123")
            self.assertEqual(len(res["events"]), 1)
            event = res["events"][0]
            self.assertEqual(event["event_id"], 1)
            self.assertIn("clip_url", event)
            self.assertIn("impact_freeze_url", event)
            self.assertTrue(event["clip_url"].startswith("/artifacts/"))
            self.assertTrue(event["impact_freeze_url"].startswith("/artifacts/"))

            sessions = controller.list_sessions()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]["session_id"], "test_session_123")
            self.assertEqual(sessions[0]["event_count"], 1)

            specific = controller.session_events(session_id="test_session_123")
            self.assertEqual(specific["session_id"], "test_session_123")
            self.assertEqual(len(specific["events"]), 1)

            unknown = controller.session_events(session_id="non_existent")
            self.assertEqual(unknown["session_id"], "non_existent")
            self.assertEqual(len(unknown["events"]), 0)

    def test_restart_discovers_newest_capture_even_if_old_report_was_modified(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.make_controller(root)
            base = root / "data" / "analysis_results" / "control_panel"
            old = base / "live_session_20260921T073126Z_aaaa"
            new = base / "live_session_20261004T064445Z_bbbb"
            for session in (old, new):
                session.mkdir(parents=True)
                (session / "test_swing_events.json").write_text('{"events":[]}')
            os.utime(old, (time.time() + 100, time.time() + 100))
            self.assertEqual(controller.session_events()['session_id'], new.name)

    def test_http_session_events_and_range_requests(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.make_controller(root)
            artifact_file = root / "sample.mp4"
            test_bytes = b"0123456789abcdefghijklmnopqrstuvwxyz"
            artifact_file.write_bytes(test_bytes)
            
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(controller))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                # Test /api/session/events
                req = urllib.request.Request(f"{base_url}/api/session/events")
                with urllib.request.urlopen(req) as resp:
                    self.assertEqual(resp.status, 200)
                    data = json.load(resp)
                    self.assertIn("events", data)
                    self.assertIn("summary", data)

                # Test /api/session/summary
                sum_req = urllib.request.Request(f"{base_url}/api/session/summary")
                with urllib.request.urlopen(sum_req) as resp:
                    self.assertEqual(resp.status, 200)
                    sum_data = json.load(resp)
                    self.assertIn("summary", sum_data)

                # Test /api/sessions
                sess_req = urllib.request.Request(f"{base_url}/api/sessions")
                with urllib.request.urlopen(sess_req) as resp:
                    self.assertEqual(resp.status, 200)
                    sess_data = json.load(resp)
                    self.assertIsInstance(sess_data, list)

                # Test HEAD request
                head_req = urllib.request.Request(f"{base_url}/artifacts/sample.mp4", method="HEAD")
                with urllib.request.urlopen(head_req) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(resp.headers.get("Accept-Ranges"), "bytes")
                    self.assertEqual(resp.headers.get("Content-Length"), str(len(test_bytes)))

                # Test Range request
                range_req = urllib.request.Request(f"{base_url}/artifacts/sample.mp4")
                range_req.add_header("Range", "bytes=5-15")
                with urllib.request.urlopen(range_req) as resp:
                    self.assertEqual(resp.status, 206)
                    self.assertEqual(resp.headers.get("Content-Range"), f"bytes 5-15/{len(test_bytes)}")
                    content = resp.read()
                    self.assertEqual(content, b"56789abcdef")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_local_video_mapped_to_stream_id(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.make_controller(root)
            uploads = root / "data" / "control_uploads"
            uploads.mkdir(parents=True, exist_ok=True)
            video_name = "0123456789abcdef0123456789abcdef.mp4"
            (uploads / video_name).write_bytes(b"dummy")

            # 1. 映射到 court01-main
            stream = controller._stream_from_payload({
                "stream_id": "local_video",
                "video_id": video_name,
                "mapped_stream_id": "court01-main",
            })
            self.assertEqual(stream["stream_id"], "local_video")
            self.assertEqual(stream["mapped_stream_id"], "court01-main")
            self.assertTrue(stream["roi_enabled"])
            self.assertEqual(stream["points"], [[20, 20], [300, 20], [300, 160], [20, 160]])

            # 2. ControlSettings & Command 包含 --stream-id court01-main
            settings = ControlSettings.from_payload({
                "stream_id": "local_video",
                "video_id": video_name,
                "mapped_stream_id": "court01-main",
                "roi_enabled": True,
            })
            self.assertEqual(settings.mapped_stream_id, "court01-main")
            self.assertTrue(settings.roi_enabled)

            cmd, _ = controller._build_command(
                settings=settings,
                runtime_config=root / "runtime.yaml",
                output_dir=root / "output",
            )
            self.assertIn("--stream-id", cmd)
            idx = cmd.index("--stream-id")
            self.assertEqual(cmd[idx + 1], "court01-main")

    def test_camera_stream_never_overridden_by_mapped_stream_id(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.make_controller(root)
            settings = ControlSettings.from_payload({
                "stream_id": "court01-main",
                "mapped_stream_id": "court02-main",
            })
            self.assertEqual(settings.stream_id, "court01-main")
            self.assertEqual(settings.mapped_stream_id, "")

            cmd, _ = controller._build_command(
                settings=settings,
                runtime_config=root / "runtime.yaml",
                output_dir=root / "output",
            )
            self.assertIn("--stream-id", cmd)
            idx = cmd.index("--stream-id")
            self.assertEqual(cmd[idx + 1], "court01-main")

    def test_sse_events_streaming(self):
        with TemporaryDirectory() as directory:
            controller = self.make_controller(Path(directory))
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(controller))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/api/events"
                request = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(request, timeout=3.0) as response:
                    self.assertIn("text/event-stream", response.headers.get("Content-Type", ""))
                    # Read the first event: should be event: status
                    line1 = response.readline().decode("utf-8").strip()
                    line2 = response.readline().decode("utf-8").strip()
                    self.assertEqual(line1, "event: status")
                    self.assertTrue(line2.startswith("data: {"))
                    data = json.loads(line2[len("data: "):])
                    self.assertEqual(data.get("state"), "stopped")
            finally:
                controller.stop_server()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_roi_boundary_api_get_and_post(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.make_controller(root)
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(controller))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                # 1. GET /api/roi/config
                url = f"http://127.0.0.1:{server.server_port}/api/roi/config?stream_id=court01-main"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    self.assertEqual(resp.status, 200)
                    data = json.load(resp)
                    self.assertEqual(data["stream_id"], "court01-main")
                    self.assertEqual(data["points"], [[20, 20], [300, 20], [300, 160], [20, 160]])

                # 2. POST /api/roi/config with updated points and mirror_polygon
                post_url = f"http://127.0.0.1:{server.server_port}/api/roi/config"
                new_points = [[25, 25], [295, 25], [295, 155], [25, 155]]
                mirror_poly = [[0.5, 0.1], [0.8, 0.1], [0.8, 0.3], [0.5, 0.3]]
                payload = json.dumps({
                    "stream_id": "court01-main",
                    "points": new_points,
                    "mirror_polygon": mirror_poly,
                    "mirror_exclusion": True,
                }).encode("utf-8")
                post_req = urllib.request.Request(
                    post_url,
                    data=payload,
                    headers={"Content-Type": "application/json", "X-Control-Token": controller.token},
                    method="POST",
                )
                with urllib.request.urlopen(post_req, timeout=3.0) as resp:
                    self.assertEqual(resp.status, 200)
                    result = json.load(resp)
                    self.assertTrue(result["success"])
                    self.assertEqual(result["points"], new_points)
                    self.assertIn("mirror_view", result)
                    self.assertEqual(result["mirror_view"]["polygon"], mirror_poly)

                # Verify file was updated
                saved_doc = yaml.safe_load(controller.roi_config_path.read_text(encoding="utf-8"))
                matched = next(s for s in saved_doc["streams"] if s["stream_id"] == "court01-main")
                self.assertEqual(matched["roi_points"], new_points)
                self.assertEqual(matched["mirror_view"]["polygon"], mirror_poly)
                self.assertTrue(matched["mirror_view"]["exclusion_enabled"])
            finally:
                controller.stop_server()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_roi_and_mirror_points_canonical_sorting(self):
        from local_control_panel import sort_points_tl_tr_br_bl
        # Out-of-order points: BL, TL, TR, BR
        raw_pts = [[100, 500], [120, 100], [600, 120], [580, 510]]
        sorted_pts = sort_points_tl_tr_br_bl(raw_pts)
        self.assertEqual(sorted_pts, [[120, 100], [600, 120], [580, 510], [100, 500]])

        with TemporaryDirectory() as directory:
            root = Path(directory)
            controller = self.make_controller(root)
            # Post unsorted points
            res = controller.save_roi_boundary({
                "stream_id": "court01-main",
                "points": [[100, 500], [120, 100], [600, 120], [580, 510]],
                "mirror_polygon": [[0.25, 0.48], [0.27, 0.10], [0.61, 0.12], [0.59, 0.48]],
            })
            self.assertEqual(res["points"], [[120, 100], [600, 120], [580, 510], [100, 500]])
            self.assertEqual(res["mirror_view"]["polygon"], [[0.27, 0.10], [0.61, 0.12], [0.59, 0.48], [0.25, 0.48]])

            # Check loaded streams also return sorted points
            loaded = controller.streams[0]
            self.assertEqual(loaded["points"], [[120, 100], [600, 120], [580, 510], [100, 500]])
            self.assertEqual(loaded["mirror_view"]["polygon"], [[0.27, 0.10], [0.61, 0.12], [0.59, 0.48], [0.25, 0.48]])

    def test_static_file_routing_and_traversal_prevention(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            static_dir = root / "static" / "css"
            static_dir.mkdir(parents=True)
            css_file = static_dir / "control_panel.css"
            css_file.write_text(":root { --test: 1; }\n", encoding="utf-8")

            controller = self.make_controller(root)
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(controller))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                # 1. Successful fetch
                url = f"http://127.0.0.1:{server.server_port}/static/css/control_panel.css"
                with urllib.request.urlopen(url) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers.get("Content-Type"), "text/css; charset=utf-8")
                    self.assertIn(b"--test: 1", response.read())

                # 2. Not found
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/static/css/missing.css")
                self.assertEqual(ctx.exception.code, 404)

                # 3. Path traversal attack attempt -> 403 or 404
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/static/../../config.yaml")
                self.assertIn(ctx.exception.code, (403, 404))
            finally:
                controller.stop_server()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_session_cache_invalidation_on_update(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            s_dir = root / "data" / "analysis_results" / "control_panel" / "session_20261117T100000Z_abc"
            s_dir.mkdir(parents=True)
            events_path = s_dir / "test_swing_events.json"
            events_path.write_text(json.dumps({"events": [{"event_id": 1, "stroke_type": "forehand"}]}), encoding="utf-8")

            controller = self.make_controller(root)
            listed_1 = controller.list_sessions()
            self.assertEqual(len(listed_1), 1)
            self.assertEqual(listed_1[0]["event_count"], 1)

            ev_1 = controller.session_events("session_20261117T100000Z_abc")
            self.assertEqual(ev_1["total_events"], 1)

            # Update file with a second event (ensure size & mtime change)
            events_path.write_text(
                json.dumps({
                    "events": [
                        {"event_id": 1, "stroke_type": "forehand"},
                        {"event_id": 2, "stroke_type": "backhand"},
                    ]
                }),
                encoding="utf-8",
            )

            listed_2 = controller.list_sessions()
            self.assertEqual(listed_2[0]["event_count"], 2)

            ev_2 = controller.session_events("session_20261117T100000Z_abc")
            self.assertEqual(ev_2["total_events"], 2)

    def test_sqlite_player_binding_and_cross_session_comparison(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "data" / "analysis_results" / "control_panel"
            s1_dir = base / "session_20261115T100000Z_s1"
            s2_dir = base / "session_20261117T100000Z_s2"
            s1_dir.mkdir(parents=True)
            s2_dir.mkdir(parents=True)

            (s1_dir / "s1_swing_events.json").write_text(
                json.dumps({
                    "events": [
                        {
                            "event_id": i,
                            "stroke_type": "Forehand",
                            "is_shadow_swing": False,
                            "biomechanics": {
                                "metrics": {
                                    "shoulder_turn_change": {"value": 30.0 + i, "unit": "deg_2d"},
                                    "arm_extension": {"value": 105.0 + i, "unit": "deg_2d"},
                                }
                            },
                            "extended_biomechanics": {
                                "racket_head_speed": {"contact_px_s": 1000.0, "max_px_s": 1100.0 + i * 10},
                                "brush_angle": {"low_to_high_angle_deg": 18.0},
                                "leg_drive": {"drive_px": 20.0},
                                "kinematic_sequence": {"sequence_quality": "ACCEPTABLE"},
                            },
                        }
                        for i in range(1, 5)
                    ]
                }),
                encoding="utf-8",
            )
            (s2_dir / "s2_swing_events.json").write_text(
                json.dumps({
                    "events": [
                        {
                            "event_id": i,
                            "stroke_type": "Forehand",
                            "is_shadow_swing": False,
                            "biomechanics": {
                                "metrics": {
                                    "shoulder_turn_change": {"value": 65.0 + i, "unit": "deg_2d"},
                                    "arm_extension": {"value": 145.0 + i, "unit": "deg_2d"},
                                }
                            },
                            "extended_biomechanics": {
                                "racket_head_speed": {"contact_px_s": 1250.0, "max_px_s": 1500.0 + i * 10},
                                "brush_angle": {"low_to_high_angle_deg": 25.0},
                                "leg_drive": {"drive_px": 32.0},
                                "kinematic_sequence": {"sequence_quality": "OPTIMAL"},
                            },
                        }
                        for i in range(1, 5)
                    ]
                }),
                encoding="utf-8",
            )

            controller = self.make_controller(root)
            sessions = controller.list_sessions()
            self.assertEqual(len(sessions), 2)

            created = controller.create_or_update_player({"display_name": "张三", "dominant_hand": "right"})
            pid = created["player_id"]
            self.assertEqual(created["display_name"], "张三")

            controller.assign_session_player({"session_id": "session_20261115T100000Z_s1", "player_id": pid})
            controller.assign_session_player({"session_id": "session_20261117T100000Z_s2", "player_id": pid})

            cmp_res = controller.compare_player_sessions(player_id=pid, stroke_type="Forehand")
            self.assertEqual(cmp_res["session_count"], 2)
            self.assertEqual(cmp_res["sessions"][0]["median_racket_speed_px_s"], 1000.0)
            self.assertEqual(cmp_res["sessions"][1]["median_racket_speed_px_s"], 1250.0)
            self.assertIsNotNone(cmp_res["sessions"][0]["representative_event"])
            self.assertIn("iqr", cmp_res["sessions"][0]["metrics"]["racket_head_speed"])
            self.assertIsNotNone(cmp_res["sessions"][0]["consistency_index"])
            self.assertTrue(cmp_res["sessions"][0]["fatigue_split"]["available"])
            self.assertTrue(len(cmp_res["sessions"][0]["coach_warnings"]) > 0)
            self.assertTrue(cmp_res["personal_attribution"]["available"])
            self.assertIn("转肩角度变化", cmp_res["personal_attribution"]["insight"])
            self.assertEqual(cmp_res["latest_delta"]["median_racket_speed_px_s"], 250.0)
            self.assertEqual(cmp_res["latest_delta"]["median_brush_angle_deg"], 7.0)
            self.assertEqual(cmp_res["latest_delta"]["median_leg_drive_px"], 12.0)

    def test_player_compare_page_route(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_html = Path(__file__).with_name("player_compare.html").read_text(encoding="utf-8")
            (root / "player_compare.html").write_text(source_html, encoding="utf-8")

            controller = self.make_controller(root)
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(controller))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/player-compare"
                with urllib.request.urlopen(url) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers.get("Content-Type"), "text/html; charset=utf-8")
                    body = response.read().decode("utf-8")
                    self.assertIn("/static/js/player_compare.js", body)
                    self.assertIn("player-compare-body", body)
            finally:
                controller.stop_server()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_backhand_filter_batch_assign_global_profile_and_review(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "data" / "analysis_results" / "control_panel"
            for idx, (sid, turn_fh, spd_fh, spd_bh) in enumerate([
                ("session_20261114T100000Z_g1", 32.0, 900.0, 780.0),
                ("session_20261116T100000Z_g2", 48.0, 1220.0, 960.0),
            ], start=1):
                s_dir = base / sid
                s_dir.mkdir(parents=True)
                (s_dir / "test_swing_events.json").write_text(
                    json.dumps({
                        "events": [
                            {
                                "event_id": 1,
                                "stroke_type": "Forehand",
                                "is_shadow_swing": False,
                                "start_frame": 10,
                                "end_frame": 30,
                                "contact_frame": 20,
                                "extended_biomechanics": {"racket_head_speed": {"contact_px_s": spd_fh, "max_px_s": spd_fh}},
                                "analysis_metrics": [
                                    {"key": "shoulder_turn_change", "label": "转肩角度变化", "unit": "deg_2d", "value": turn_fh},
                                    {"key": "hip_shoulder_separation", "label": "肩髋分离角", "unit": "image_plane_deg", "value": 25.0 + idx * 5},
                                    {"key": "arm_extension", "label": "持拍侧肘夹角", "unit": "deg_2d", "value": 150.0},
                                    {"key": "racket_max_speed", "label": "拍框峰值速度", "unit": "px/s", "value": spd_fh},
                                ],
                            },
                            {
                                "event_id": 2,
                                "stroke_type": "Two-Handed Backhand",
                                "is_shadow_swing": False,
                                "start_frame": 40,
                                "end_frame": 60,
                                "contact_frame": 50,
                                "extended_biomechanics": {"racket_head_speed": {"contact_px_s": spd_bh, "max_px_s": spd_bh}},
                                "analysis_metrics": [
                                    {"key": "shoulder_turn_change", "label": "转肩角度变化", "unit": "deg_2d", "value": 36.0 + idx * 4},
                                    {"key": "hip_shoulder_separation", "label": "肩髋分离角", "unit": "image_plane_deg", "value": 20.0 + idx * 3},
                                    {"key": "arm_extension", "label": "持拍侧肘夹角", "unit": "deg_2d", "value": 140.0},
                                    {"key": "racket_max_speed", "label": "拍框峰值速度", "unit": "px/s", "value": spd_bh},
                                ],
                            },
                        ]
                    }),
                    encoding="utf-8",
                )

            controller = self.make_controller(root)
            controller.list_sessions()
            created = controller.create_or_update_player({"display_name": "李四", "dominant_hand": "right"})
            pid = created["player_id"]

            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(controller))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                # 1. Batch assign both sessions via HTTP POST
                batch_req = urllib.request.Request(
                    f"{base_url}/api/sessions/assign-player-batch",
                    data=json.dumps({
                        "session_ids": ["session_20261114T100000Z_g1", "session_20261116T100000Z_g2"],
                        "player_id": pid,
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json", "X-Control-Token": controller.token},
                    method="POST",
                )
                with urllib.request.urlopen(batch_req) as resp:
                    batch_data = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(batch_data["updated_count"], 2)

                # 2. Verify Backhand filter matches Two-Handed Backhand
                bh_cmp = controller.compare_player_sessions(player_id=pid, stroke_type="Backhand")
                self.assertEqual(bh_cmp["session_count"], 2)
                self.assertEqual(bh_cmp["sessions"][0]["swings"][0]["stroke_type"], "Two-Handed Backhand")

                # 3. Verify GET /api/players/global-profile
                with urllib.request.urlopen(f"{base_url}/api/players/global-profile?player_id={pid}") as resp:
                    self.assertEqual(resp.status, 200)
                    gp = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(gp["total_sessions"], 2)
                    self.assertEqual(gp["forehand"]["total_swings"], 2)
                    self.assertEqual(gp["backhand"]["total_swings"], 2)
                    self.assertEqual(len(gp["forehand"]["video_presets"]["timeline_4"]), 2)
                    self.assertEqual(len(gp["forehand"]["video_presets"]["conflict_4"]), 2)
                    self.assertIsInstance(gp["forehand"]["commonalities"], list)
                    self.assertIsInstance(gp["forehand"]["contradictions"], list)
                    self.assertGreater(len(gp["forehand"]["contradictions"]), 0)
                    self.assertGreater(len(gp["symmetry_comparison"]), 0)
                    self.assertEqual(gp["forehand"]["video_presets"]["best"]["session_id"], "session_20261116T100000Z_g2")

                # 4. Verify POST /api/players/global-review (single stroke_side) with mocked transport
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), patch(
                    "deepseek_realtime_coach.UrllibDeepSeekTransport.complete",
                    return_value={"model": "deepseek-chat", "choices": [{"message": {"content": "【正手评估】测试总评"}}]},
                ):
                    rev_req = urllib.request.Request(
                        f"{base_url}/api/players/global-review",
                        data=json.dumps({"player_id": pid, "stroke_side": "forehand"}).encode("utf-8"),
                        headers={"Content-Type": "application/json", "X-Control-Token": controller.token},
                        method="POST",
                    )
                    with urllib.request.urlopen(rev_req) as resp:
                        self.assertEqual(resp.status, 200)
                        rev_data = json.loads(resp.read().decode("utf-8"))
                        self.assertIn("【正手评估】", rev_data["review_text"])
                        self.assertEqual(rev_data["stroke_side"], "forehand")
                        self.assertEqual(rev_data["provider"], "deepseek")
                        self.assertEqual(rev_data["model"], "deepseek-chat")
                        self.assertTrue(str(rev_data.get("review_id") or "").startswith("rev_"))
                        self.assertIsInstance(rev_data.get("session_ids"), list)
                        self.assertIsInstance(rev_data.get("video_paths"), list)

                # 5. Verify GET /api/players/coach-reviews returns the persisted review
                list_req = urllib.request.Request(
                    f"{base_url}/api/players/coach-reviews?player_id={pid}&stroke_side=forehand",
                    headers={"X-Control-Token": controller.token},
                )
                with urllib.request.urlopen(list_req) as resp:
                    self.assertEqual(resp.status, 200)
                    saved_revs = json.loads(resp.read().decode("utf-8"))
                    self.assertGreaterEqual(len(saved_revs), 1)
                    self.assertEqual(saved_revs[0]["player_id"], pid)
                    self.assertEqual(saved_revs[0]["stroke_side"], "forehand")
                    self.assertIn("【正手评估】", saved_revs[0]["review_text"])
                    self.assertIsInstance(saved_revs[0]["session_ids"], list)
                    self.assertIsInstance(saved_revs[0]["evidence_packet"], dict)

                # 6. Verify POST /api/players/global-review with OpenAI GPT provider
                with patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "sk-test-openai", "OPENAI_BASE_URL": "https://api.openai.com/v1"},
                ), patch(
                    "deepseek_realtime_coach.UrllibDeepSeekTransport.complete",
                    return_value={"model": "gpt-4o", "choices": [{"message": {"content": "【GPT正手总评】测试"}}]},
                ) as mock_gpt:
                    gpt_req = urllib.request.Request(
                        f"{base_url}/api/players/global-review",
                        data=json.dumps({
                            "player_id": pid,
                            "stroke_side": "forehand",
                            "provider": "openai",
                            "model": "gpt-4o",
                        }).encode("utf-8"),
                        headers={"Content-Type": "application/json", "X-Control-Token": controller.token},
                        method="POST",
                    )
                    with urllib.request.urlopen(gpt_req) as resp:
                        self.assertEqual(resp.status, 200)
                        gpt_data = json.loads(resp.read().decode("utf-8"))
                        self.assertEqual(gpt_data["provider"], "openai")
                        self.assertEqual(gpt_data["model"], "gpt-4o")
                        self.assertIn("【GPT正手总评】", gpt_data["review_text"])
                    self.assertEqual(mock_gpt.call_args.kwargs["base_url"], "https://api.openai.com/v1")
                    self.assertEqual(mock_gpt.call_args.kwargs["api_key"], "sk-test-openai")
            finally:
                controller.stop_server()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_duplicate_video_session_dedup_and_cleanup(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            uploads_dir = root / "data" / "control_uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            cp_dir = root / "data" / "analysis_results" / "control_panel"

            vid_same_1 = uploads_dir / "uuid_a111.mp4"
            vid_same_2 = uploads_dir / "uuid_b222.mp4"
            vid_other = uploads_dir / "uuid_c333.mp4"
            vid_same_1.write_bytes(b"same-tennis-video-payload-123456")
            vid_same_2.write_bytes(b"same-tennis-video-payload-123456")
            vid_other.write_bytes(b"other-tennis-video-payload-999999")

            specs = [
                ("session_20261114T100000Z_s1", str(vid_same_1), False, 2000.0),
                ("session_20261115T100000Z_s2", str(vid_same_2), True, 2150.0),
                ("session_20261116T100000Z_s3", str(vid_other), False, 2200.0),
                ("session_20261117T100000Z_rtsp1", "rtsp://192.168.1.50/stream1", False, 2050.0),
                ("session_20261118T100000Z_rtsp2", "rtsp://192.168.1.50/stream1", False, 2100.0),
            ]
            for sid, src_str, is_dual, spd in specs:
                sdir = cp_dir / sid
                sdir.mkdir(parents=True, exist_ok=True)
                (sdir / f"{sid}_evidence_manifest.json").write_text(
                    json.dumps({"capture": {"source": src_str}, "session": {"stream_id": "main"}}),
                    encoding="utf-8",
                )
                metrics = [
                    {"key": "shoulder_turn_change", "label": "转肩角度变化", "unit": "deg_2d", "value": 45.0},
                    {"key": "racket_max_speed", "label": "拍框峰值速度", "unit": "px/s", "value": spd},
                ]
                if is_dual:
                    metrics.append(
                        {"key": "takeback_depth", "label": "引拍幅度", "unit": "shoulder_width_ratio", "value": 1.6}
                    )
                (sdir / f"{sid}_swing_events.json").write_text(
                    json.dumps({
                        "events": [
                            {
                                "event_id": 1,
                                "stroke_type": "Forehand",
                                "start_frame": 10,
                                "end_frame": 30,
                                "contact_frame": 20,
                                "racket_max_speed_px_s": spd,
                                "analysis_metrics": metrics,
                            }
                        ]
                    }),
                    encoding="utf-8",
                )

            controller = self.make_controller(root)
            sessions = controller.list_sessions()
            by_sid = {s["session_id"]: s for s in sessions}
            self.assertEqual(by_sid["session_20261114T100000Z_s1"]["source_sha256"], by_sid["session_20261115T100000Z_s2"]["source_sha256"])
            self.assertTrue(by_sid["session_20261114T100000Z_s1"]["is_duplicate"])
            self.assertFalse(by_sid["session_20261115T100000Z_s2"]["is_duplicate"])
            self.assertEqual(by_sid["session_20261115T100000Z_s2"]["source_group_size"], 2)
            self.assertFalse(by_sid["session_20261117T100000Z_rtsp1"]["is_duplicate"])
            self.assertFalse(by_sid["session_20261118T100000Z_rtsp2"]["is_duplicate"])

            created = controller.create_or_update_player({"display_name": "王五"})
            pid = created["player_id"]
            controller.assign_sessions_player_batch({
                "session_ids": [sp[0] for sp in specs],
                "player_id": pid,
            })

            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(controller))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                # Default dedup=1 keeps dual-view canonical s2 + distinct s3 + 2 RTSP sessions = 4 sessions
                with urllib.request.urlopen(f"{base_url}/api/players/global-profile?player_id={pid}") as resp:
                    gp_dedup = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(gp_dedup["dedup"]["enabled"])
                    self.assertEqual(gp_dedup["dedup"]["raw_session_count"], 5)
                    self.assertEqual(gp_dedup["dedup"]["unique_session_count"], 4)
                    self.assertEqual(gp_dedup["dedup"]["duplicate_session_count"], 1)
                    self.assertEqual(gp_dedup["total_sessions"], 4)
                    kept_sids = {c["session_id"] for c in gp_dedup["sessions_catalog"]}
                    self.assertIn("session_20261115T100000Z_s2", kept_sids)
                    self.assertNotIn("session_20261114T100000Z_s1", kept_sids)
                    self.assertIn("session_20261117T100000Z_rtsp1", kept_sids)
                    self.assertIn("session_20261118T100000Z_rtsp2", kept_sids)

                # Explicit dedup=0 returns all 5 sessions
                with urllib.request.urlopen(f"{base_url}/api/players/global-profile?player_id={pid}&dedup=0") as resp:
                    gp_raw = json.loads(resp.read().decode("utf-8"))
                    self.assertFalse(gp_raw["dedup"]["enabled"])
                    self.assertEqual(gp_raw["total_sessions"], 5)

                # List duplicate groups endpoint
                with urllib.request.urlopen(f"{base_url}/api/sessions/duplicates?player_id={pid}") as resp:
                    dup_data = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(dup_data["duplicate_group_count"], 1)
                    self.assertEqual(dup_data["groups"][0]["canonical"]["session_id"], "session_20261115T100000Z_s2")
                    self.assertEqual(dup_data["recommended_cleanup_ids"], ["session_20261114T100000Z_s1"])

                # Active session guard returns 400
                controller._artifacts["session_id"] = "session_20261116T100000Z_s3"
                bad_req = urllib.request.Request(
                    f"{base_url}/api/sessions/dedup-cleanup",
                    data=json.dumps({"session_ids": ["session_20261116T100000Z_s3"], "mode": "archive"}).encode("utf-8"),
                    headers={"Content-Type": "application/json", "X-Control-Token": controller.token},
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(bad_req)
                self.assertEqual(ctx.exception.code, 400)

                # Archive duplicate s1
                clean_req = urllib.request.Request(
                    f"{base_url}/api/sessions/dedup-cleanup",
                    data=json.dumps({"session_ids": ["session_20261114T100000Z_s1"], "mode": "archive"}).encode("utf-8"),
                    headers={"Content-Type": "application/json", "X-Control-Token": controller.token},
                    method="POST",
                )
                with urllib.request.urlopen(clean_req) as resp:
                    clean_res = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(clean_res["ok"])
                    self.assertEqual(clean_res["cleaned_count"], 1)

                self.assertFalse((cp_dir / "session_20261114T100000Z_s1").exists())
                self.assertTrue((root / "data" / "analysis_results" / "control_panel_duplicates_archive" / "session_20261114T100000Z_s1").exists())
                with urllib.request.urlopen(f"{base_url}/api/sessions/duplicates?player_id={pid}") as resp:
                    after_dup = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(after_dup["duplicate_group_count"], 0)
            finally:
                controller.stop_server()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_swing_trajectory_fitting_and_summary_endpoints(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cp_dir = root / "data" / "analysis_results" / "control_panel"
            sid = "session_20261119T100000Z_traj1"
            sdir = cp_dir / sid
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / f"{sid}_evidence_manifest.json").write_text(
                json.dumps({"capture": {"source": "/tmp/demo.mp4"}, "session": {"stream_id": "main"}}),
                encoding="utf-8",
            )
            (sdir / f"{sid}_swing_events.json").write_text(
                json.dumps({
                    "events": [
                        {
                            "event_id": 1,
                            "stroke_type": "Forehand",
                            "contact_status": "candidate",
                            "start_frame": 5,
                            "end_frame": 30,
                            "contact_frame": 20,
                            "racket_max_speed_px_s": 24000.0,
                            "quality_flags": {"ball_contact_window_ratio": 0.8},
                            "analysis_metrics": [
                                {"key": "arm_extension", "label": "肘角", "unit": "deg_2d", "value": 165.0},
                                {"key": "preparation_knee_flexion", "label": "屈膝", "unit": "deg_2d", "value": 18.0},
                                {"key": "shoulder_turn_change", "label": "转肩", "unit": "deg_2d", "value": 30.0},
                                {"key": "racket_max_speed", "label": "拍速", "unit": "px/s", "value": 24000.0},
                            ],
                        }
                    ]
                }),
                encoding="utf-8",
            )
            # Generate 35 synthetic frames with 1 mirror-teleport racket outlier at frame 18 and 1 missing frame at 19
            lines = []
            for fid in range(36):
                rx = 500.0 + fid * 6.0
                ry = 400.0 - abs(fid - 20) * 1.5
                pose = {
                    "left_shoulder": [460.0, 320.0],
                    "right_shoulder": [540.0, 320.0 + (fid - 15) * 0.8],
                    "right_elbow": [560.0, 370.0],
                    "right_wrist": [rx, ry],
                    "left_hip": [470.0, 450.0],
                    "right_hip": [530.0, 450.0],
                    "left_knee": [465.0, 540.0],
                    "right_knee": [540.0, 540.0],
                    "left_ankle": [450.0, 640.0],
                    "right_ankle": [520.0, 640.0],
                }
                if fid == 18:
                    # Teleport outlier far away in mirror
                    rkts = [{"box": [50.0, 20.0, 120.0, 90.0]}]
                elif fid == 19:
                    rkts = []
                else:
                    rkts = [{"box": [rx - 30.0, ry - 30.0, rx + 30.0, ry + 30.0]}]
                ball = [rx + 5.0, ry] if abs(fid - 20) <= 3 else None
                lines.append(json.dumps({"frame_id": fid, "timestamp": fid * 0.04, "pose": pose, "rackets": rkts, "ball": ball}))
            (sdir / f"{sid}_frames.jsonl").write_text("\n".join(lines), encoding="utf-8")

            controller = self.make_controller(root)
            controller.list_sessions()
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(controller))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                with urllib.request.urlopen(f"{base_url}/api/session/swing-trajectory?session_id={sid}&event_id=1") as resp:
                    self.assertEqual(resp.status, 200)
                    d = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(d["available"])
                    self.assertGreaterEqual(d["rejected_count"], 1)
                    self.assertGreaterEqual(d["interpolated_count"], 1)
                    self.assertIn("smooth_racket_speed", d["series"])
                    self.assertLess(d["summary"]["racket_max_speed"]["refined"], 2000.0)

                with urllib.request.urlopen(f"{base_url}/api/players/trajectory-summary?player_id=default&stroke_side=forehand") as resp:
                    self.assertEqual(resp.status, 200)
                    s_data = json.loads(resp.read().decode("utf-8"))
                    self.assertEqual(s_data["count"], 1)
                    self.assertEqual(len(s_data["swings"]), 1)
                    self.assertEqual(s_data["swings"][0]["event_id"], 1)
            finally:
                controller.stop_server()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
