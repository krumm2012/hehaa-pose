#!/usr/bin/env python3
"""Local-only web control panel for ROI preview and main_pipe lifecycle."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import cv2
import numpy as np
import yaml

from analysis_data_contracts import generate_session_id
from calibrate_roi import capture_calibration_frame
from roi_stream_config import sanitize_stream_source


SESSION_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
MAX_REQUEST_BYTES = 64 * 1024


def _bool(payload: Dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _number(
    payload: Dict[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
    integer: bool = False,
):
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是数字") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} 必须在 {minimum:g}–{maximum:g} 之间")
    return int(value) if integer else value


def inject_rtsp_credentials(source: str, username: str, password: str) -> str:
    """Add credentials to a URL without changing its stable camera identity."""
    parsed = urlsplit(str(source or "").strip())
    if not parsed.scheme or not parsed.hostname:
        return str(source or "")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    host_port = f"{host}:{port}" if port is not None else host
    user = str(username or "").strip()
    secret = str(password or "")
    netloc = host_port
    if user:
        netloc = f"{quote(user, safe='')}:{quote(secret, safe='')}@{host_port}"
    return urlunsplit(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


@dataclass(frozen=True)
class ControlSettings:
    stream_id: str
    output_dir: str
    session_name: str
    output_fps: float
    inference_workers: int
    roi_enabled: bool
    crop_margin: int
    show_roi_boundary: bool
    show_roi_fill: bool
    show_roi_points: bool
    live_mode: bool
    save_video: bool
    realtime_swing_events: bool
    realtime_frame_output: bool
    realtime_coach: bool
    max_suggestions: int
    min_confidence: float
    analysis_interval: int
    settle_frames: int
    deepseek_coach: bool
    hdmi_output: bool
    display_origin_x: int
    display_origin_y: int

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ControlSettings":
        session_name = SESSION_NAME_PATTERN.sub(
            "_",
            str(payload.get("session_name") or "live_session").strip(),
        ).strip("_")[:64]
        if not session_name:
            session_name = "live_session"
        output_dir = str(
            payload.get("output_dir")
            or "data/analysis_results/control_panel"
        ).strip()
        if not output_dir:
            raise ValueError("output_dir 不能为空")
        realtime_coach = _bool(payload, "realtime_coach", True)
        deepseek_coach = _bool(payload, "deepseek_coach", False)
        realtime_swing_events = (
            _bool(payload, "realtime_swing_events", True)
            or realtime_coach
            or deepseek_coach
        )
        return cls(
            stream_id=str(payload.get("stream_id") or "").strip(),
            output_dir=output_dir,
            session_name=session_name,
            output_fps=_number(payload, "output_fps", 25, 1, 60),
            inference_workers=_number(
                payload,
                "inference_workers",
                8,
                1,
                32,
                integer=True,
            ),
            roi_enabled=_bool(payload, "roi_enabled", True),
            crop_margin=_number(
                payload,
                "crop_margin",
                12,
                0,
                256,
                integer=True,
            ),
            show_roi_boundary=_bool(payload, "show_roi_boundary", True),
            show_roi_fill=_bool(payload, "show_roi_fill", False),
            show_roi_points=_bool(payload, "show_roi_points", True),
            live_mode=_bool(payload, "live_mode", True),
            save_video=_bool(payload, "save_video", False),
            realtime_swing_events=realtime_swing_events,
            realtime_frame_output=_bool(
                payload,
                "realtime_frame_output",
                False,
            ),
            realtime_coach=realtime_coach,
            max_suggestions=_number(
                payload,
                "max_suggestions",
                3,
                1,
                3,
                integer=True,
            ),
            min_confidence=_number(
                payload,
                "min_confidence",
                0.45,
                0,
                1,
            ),
            analysis_interval=_number(
                payload,
                "analysis_interval",
                5,
                1,
                60,
                integer=True,
            ),
            settle_frames=_number(
                payload,
                "settle_frames",
                15,
                0,
                300,
                integer=True,
            ),
            deepseek_coach=deepseek_coach,
            hdmi_output=_bool(payload, "hdmi_output", False),
            display_origin_x=_number(
                payload,
                "display_origin_x",
                0,
                -10000,
                10000,
                integer=True,
            ),
            display_origin_y=_number(
                payload,
                "display_origin_y",
                0,
                -10000,
                10000,
                integer=True,
            ),
        )


def load_stream_profiles(path: Path) -> List[Dict[str, Any]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    streams = document.get("streams") or []
    if isinstance(streams, dict):
        streams = [
            {"stream_id": stream_id, **profile}
            for stream_id, profile in streams.items()
            if isinstance(profile, dict)
        ]
    result = []
    for index, profile in enumerate(streams):
        if not isinstance(profile, dict):
            continue
        source = str(
            profile.get("stream_source")
            or profile.get("source")
            or profile.get("rtsp_url")
            or ""
        )
        points = profile.get("roi_points") or profile.get("points") or []
        frame_size = profile.get("frame_size") or [0, 0]
        result.append(
            {
                "stream_id": str(
                    profile.get("stream_id")
                    or profile.get("id")
                    or f"stream-{index + 1}"
                ),
                "label": str(
                    profile.get("stream_label")
                    or profile.get("label")
                    or profile.get("name")
                    or f"Camera {index + 1}"
                ).strip(),
                "source": sanitize_stream_source(source),
                "roi_enabled": bool(
                    profile.get(
                        "roi_enabled",
                        profile.get("enabled", False),
                    )
                ),
                "points": [
                    [int(point[0]), int(point[1])]
                    for point in points
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                ],
                "frame_size": [
                    int(frame_size[0]),
                    int(frame_size[1]),
                ]
                if isinstance(frame_size, (list, tuple))
                and len(frame_size) >= 2
                else [0, 0],
                "default": bool(profile.get("default", False)),
            }
        )
    if not result:
        raise ValueError(f"ROI 配置中没有可用码流: {path}")
    return result


class LocalPipelineController:
    """Own one main_pipe subprocess and expose only sanitized state."""

    def __init__(
        self,
        workspace: Path,
        config_path: Path,
        roi_config_path: Path,
        frontend_path: Path,
        username_env: str = "TENNIS_RTSP_USERNAME",
        password_env: str = "TENNIS_RTSP_PASSWORD",
    ):
        self.workspace = workspace.resolve()
        self.config_path = config_path.resolve()
        self.roi_config_path = roi_config_path.resolve()
        self.frontend_path = frontend_path.resolve()
        self.username_env = str(username_env)
        self.password_env = str(password_env)
        self.token = secrets.token_urlsafe(24)
        self.streams = load_stream_profiles(self.roi_config_path)
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._runtime_directory: Optional[Path] = None
        self._state = "stopped"
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None
        self._returncode: Optional[int] = None
        self._logs = deque(maxlen=300)
        self._public_command: List[str] = []
        self._artifacts: Dict[str, str] = {}

    def public_config(self) -> Dict[str, Any]:
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        realtime = config.get("realtime_swing") or {}
        roi = config.get("roi_settings") or {}
        perf = config.get("pipeline_perf") or {}
        default_stream = next(
            (
                stream["stream_id"]
                for stream in self.streams
                if stream.get("default")
            ),
            self.streams[0]["stream_id"],
        )
        return {
            "token": self.token,
            "streams": self.streams,
            "credentials_configured": bool(
                os.environ.get(self.username_env)
            ),
            "defaults": {
                "stream_id": default_stream,
                "output_dir": "data/analysis_results/control_panel",
                "session_name": "live_session",
                "output_fps": float(
                    (config.get("video_processing") or {}).get(
                        "output_fps",
                        25,
                    )
                ),
                "inference_workers": int(
                    perf.get("inference_workers", 8)
                ),
                "roi_enabled": bool(roi.get("enabled", True)),
                "crop_margin": int(roi.get("crop_margin", 12)),
                "show_roi_boundary": True,
                "show_roi_fill": False,
                "show_roi_points": True,
                "live_mode": True,
                "save_video": False,
                "realtime_swing_events": True,
                "realtime_frame_output": bool(
                    realtime.get("frame_output_enabled", False)
                ),
                "realtime_coach": True,
                "max_suggestions": int(
                    realtime.get("coach_max_suggestions", 3)
                ),
                "min_confidence": float(
                    realtime.get("coach_min_confidence", 0.45)
                ),
                "analysis_interval": int(
                    realtime.get("analysis_interval_frames", 5)
                ),
                "settle_frames": int(
                    realtime.get("settle_frames") or 15
                ),
                "deepseek_coach": False,
                "hdmi_output": False,
                "display_origin_x": 0,
                "display_origin_y": 0,
            },
        }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            self._refresh_process_state()
            elapsed = (
                max(0.0, time.time() - self._started_at)
                if self._started_at is not None
                and self._state in {"running", "stopping"}
                else 0.0
            )
            return {
                "state": self._state,
                "pid": (
                    int(self._process.pid)
                    if self._process is not None
                    and self._process.poll() is None
                    else None
                ),
                "started_at": self._started_at,
                "stopped_at": self._stopped_at,
                "elapsed_seconds": round(elapsed, 1),
                "returncode": self._returncode,
                "command": list(self._public_command),
                "logs": list(self._logs),
                "artifacts": dict(self._artifacts),
            }

    def preview(self, payload: Dict[str, Any]) -> bytes:
        stream = self._stream(str(payload.get("stream_id") or ""))
        source = self._authenticated_source(stream, payload)
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        frame = capture_calibration_frame(
            source,
            config,
            warmup_frames=3,
        )
        preview = self._draw_roi_preview(frame, stream)
        ok, encoded = cv2.imencode(
            ".jpg",
            preview,
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
        if not ok:
            raise RuntimeError("ROI 预览图编码失败")
        return encoded.tobytes()

    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        settings = ControlSettings.from_payload(payload)
        stream = self._stream(settings.stream_id)
        with self._lock:
            self._refresh_process_state()
            if self._state in {"running", "stopping"}:
                raise RuntimeError("分析进程已在运行")
            output_dir = self._safe_output_directory(settings.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            source = self._authenticated_source(stream, payload)
            runtime_directory = Path(
                tempfile.mkdtemp(prefix="tennis-control-")
            )
            runtime_config = runtime_directory / "runtime_config.yaml"
            config = yaml.safe_load(
                self.config_path.read_text(encoding="utf-8")
            ) or {}
            config["video_input_path"] = source
            config["video_output_path"] = str(
                output_dir / f"{settings.session_name}_live.mp4"
            )
            roi = config.setdefault("roi_settings", {})
            roi["enabled"] = settings.roi_enabled
            roi["auto_load_config"] = True
            roi["roi_config_path"] = str(self.roi_config_path)
            roi["crop_margin"] = settings.crop_margin
            visualization = roi.setdefault("visualization", {})
            visualization["show_roi_boundary"] = settings.show_roi_boundary
            visualization["show_roi_fill"] = settings.show_roi_fill
            visualization["show_roi_points"] = settings.show_roi_points
            runtime_config.write_text(
                yaml.safe_dump(
                    config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            os.chmod(runtime_config, 0o600)
            command, artifacts = self._build_command(
                settings,
                runtime_config,
                output_dir,
            )
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            self._logs.clear()
            self._logs.append(
                f"[control] 启动 {stream['label']} · {stream['source']}"
            )
            self._process = subprocess.Popen(
                command,
                cwd=str(self.workspace),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            self._runtime_directory = runtime_directory
            self._state = "running"
            self._started_at = time.time()
            self._stopped_at = None
            self._returncode = None
            self._public_command = [
                sanitize_stream_source(item)
                if "://" in item
                else item
                for item in command
            ]
            self._artifacts = artifacts
            self._reader_thread = threading.Thread(
                target=self._read_output,
                args=(self._process,),
                daemon=True,
                name="tennis-control-log-reader",
            )
            self._reader_thread.start()
            return self.status()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            self._refresh_process_state()
            if (
                self._process is None
                or self._process.poll() is not None
            ):
                self._state = "stopped"
                return self.status()
            self._state = "stopping"
            self._logs.append("[control] 正在发送安全停止信号…")
            try:
                os.killpg(self._process.pid, signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                self._process.send_signal(signal.SIGINT)
            return self.status()

    def shutdown(self) -> None:
        self.stop()
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        self._cleanup_runtime_directory()

    def artifact_path(self, request_path: str) -> Path:
        relative = unquote(request_path).lstrip("/")
        candidate = (self.workspace / relative).resolve()
        if self.workspace not in candidate.parents:
            raise ValueError("非法文件路径")
        return candidate

    def live_preview_path(self) -> Optional[Path]:
        value = self._artifacts.get("preview_path")
        if not value:
            return None
        return self.artifact_path(value)

    def _stream(self, stream_id: str) -> Dict[str, Any]:
        selected = next(
            (
                stream
                for stream in self.streams
                if stream["stream_id"] == stream_id
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"未知球场: {stream_id}")
        return selected

    def _authenticated_source(
        self,
        stream: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> str:
        username = str(payload.get("username") or "").strip()
        password = str(payload.get("password") or "")
        if not username:
            username = os.environ.get(self.username_env, "")
            password = os.environ.get(self.password_env, "")
        return inject_rtsp_credentials(
            stream["source"],
            username,
            password,
        )

    def _safe_output_directory(self, value: str) -> Path:
        requested = Path(value).expanduser()
        candidate = (
            requested.resolve()
            if requested.is_absolute()
            else (self.workspace / requested).resolve()
        )
        if candidate != self.workspace and self.workspace not in candidate.parents:
            raise ValueError("输出目录必须位于项目工作区内")
        return candidate

    def _build_command(
        self,
        settings: ControlSettings,
        runtime_config: Path,
        output_dir: Path,
    ):
        stem = settings.session_name
        session_id = generate_session_id(stem)
        session_dir = output_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        relative = lambda path: str(
            path.resolve().relative_to(self.workspace)
        )
        output_video = session_dir / f"{stem}_live.mp4"
        event_json = session_dir / f"{stem}_swing_events.json"
        event_log = session_dir / f"{stem}_swing_events.jsonl"
        event_html = session_dir / f"{stem}_swing_report.html"
        clips_dir = session_dir / f"{stem}_swing_clips"
        frame_jsonl = session_dir / f"{stem}_frames.jsonl"
        frame_snapshot = session_dir / f"{stem}_frames_latest.json"
        preview = session_dir / f"{stem}_swing_report_roi_preview.jpg"
        python_executable = Path(sys.executable)
        preferred_python = self.workspace / "venv_yolo26/bin/python"
        if preferred_python.exists():
            python_executable = preferred_python
        command = [
            str(python_executable),
            str(self.workspace / "main_pipe.py"),
            "--config",
            str(runtime_config),
            "--session-id",
            session_id,
            "--session-output-root",
            str(output_dir),
            "--output",
            str(output_video),
            "--output-fps",
            str(settings.output_fps),
            "--inference-workers",
            str(settings.inference_workers),
            "--no-dual-view",
        ]
        if settings.live_mode:
            command.extend(["--live-mode", "--drop-stale-frames"])
        if not settings.save_video:
            command.append("--no-save-video")
        if settings.realtime_swing_events:
            command.extend(
                [
                    "--realtime-swing-events",
                    "--realtime-swing-json",
                    str(event_json),
                    "--realtime-swing-html",
                    str(event_html),
                    "--realtime-swing-clips-dir",
                    str(clips_dir),
                    "--realtime-swing-event-log",
                    str(event_log),
                    "--realtime-analysis-interval",
                    str(settings.analysis_interval),
                    "--realtime-settle-frames",
                    str(settings.settle_frames),
                ]
            )
        if settings.realtime_frame_output:
            command.extend(
                [
                    "--realtime-frame-output",
                    "--realtime-frame-jsonl",
                    str(frame_jsonl),
                    "--realtime-frame-snapshot-json",
                    str(frame_snapshot),
                ]
            )
        if settings.realtime_coach:
            command.extend(
                [
                    "--realtime-coach",
                    "--realtime-coach-max-suggestions",
                    str(settings.max_suggestions),
                    "--realtime-coach-min-confidence",
                    str(settings.min_confidence),
                ]
            )
        if settings.deepseek_coach:
            command.append("--deepseek-coach")
        if settings.hdmi_output:
            command.extend(
                [
                    "--hdmi-output",
                    "--display-origin",
                    str(settings.display_origin_x),
                    str(settings.display_origin_y),
                ]
            )
        artifacts = {
            "session_id": session_id,
            "session_dir": relative(session_dir),
            "output_video": relative(output_video),
            "event_json": relative(event_json),
            "event_log": relative(event_log),
            "report_path": relative(event_html),
            "report_url": f"/artifacts/{quote(relative(event_html))}",
            "clips_dir": relative(clips_dir),
            "preview_path": relative(preview),
        }
        return command, artifacts

    def _draw_roi_preview(
        self,
        frame,
        stream: Dict[str, Any],
    ):
        height, width = frame.shape[:2]
        preview = frame.copy()
        configured_width, configured_height = stream.get("frame_size") or [
            width,
            height,
        ]
        scale_x = width / max(1, int(configured_width or width))
        scale_y = height / max(1, int(configured_height or height))
        points = np.asarray(
            [
                [
                    int(round(point[0] * scale_x)),
                    int(round(point[1] * scale_y)),
                ]
                for point in stream.get("points") or []
            ],
            dtype=np.int32,
        )
        if len(points) == 4:
            overlay = preview.copy()
            cv2.fillPoly(overlay, [points], (0, 210, 255))
            preview = cv2.addWeighted(overlay, 0.13, preview, 0.87, 0)
            cv2.polylines(
                preview,
                [points],
                True,
                (0, 235, 255),
                max(2, int(round(width / 900))),
                cv2.LINE_AA,
            )
            for index, point in enumerate(points, start=1):
                x, y = int(point[0]), int(point[1])
                cv2.circle(preview, (x, y), 7, (30, 240, 70), -1)
                cv2.putText(
                    preview,
                    f"P{index}",
                    (x + 9, y - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (30, 240, 70),
                    2,
                    cv2.LINE_AA,
                )
        header_height = max(64, int(round(height * 0.075)))
        cv2.rectangle(
            preview,
            (0, 0),
            (width, header_height),
            (9, 15, 25),
            -1,
        )
        cv2.putText(
            preview,
            f"ROI PREVIEW | {stream['label']}",
            (24, int(header_height * 0.48)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 235, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            stream["source"],
            (24, int(header_height * 0.82)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (205, 214, 226),
            1,
            cv2.LINE_AA,
        )
        max_width = 1600
        if width > max_width:
            scale = max_width / width
            preview = cv2.resize(
                preview,
                (max_width, int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return preview

    def _read_output(self, process: subprocess.Popen) -> None:
        stream = process.stdout
        try:
            if stream is not None:
                for line in stream:
                    text = line.rstrip()
                    if text:
                        with self._lock:
                            self._logs.append(text)
            returncode = process.wait()
            with self._lock:
                self._returncode = int(returncode)
                self._stopped_at = time.time()
                self._state = "stopped" if returncode == 0 else "failed"
                self._logs.append(
                    f"[control] 进程结束，返回码 {returncode}"
                )
        finally:
            if stream is not None:
                stream.close()
            self._cleanup_runtime_directory()

    def _refresh_process_state(self) -> None:
        if self._process is None:
            return
        returncode = self._process.poll()
        if returncode is None:
            return
        self._returncode = int(returncode)
        if self._state in {"running", "stopping"}:
            self._state = "stopped" if returncode == 0 else "failed"
            self._stopped_at = time.time()

    def _cleanup_runtime_directory(self) -> None:
        with self._lock:
            directory = self._runtime_directory
            self._runtime_directory = None
        if directory is not None and directory.exists():
            shutil.rmtree(directory, ignore_errors=True)


def create_handler(controller: LocalPipelineController):
    class ControlPanelHandler(BaseHTTPRequestHandler):
        server_version = "TennisControlPanel/1.0"

        def log_message(self, format_string, *args):
            sys.stdout.write(
                "[web] "
                + (format_string % args)
                + "\n"
            )

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path == "/":
                    self._send_file(controller.frontend_path, "text/html")
                elif path == "/api/config":
                    self._send_json(controller.public_config())
                elif path == "/api/status":
                    self._send_json(controller.status())
                elif path == "/api/live-preview":
                    preview = controller.live_preview_path()
                    if preview is None or not preview.exists():
                        self.send_error(
                            HTTPStatus.NOT_FOUND,
                            "Live preview not ready",
                        )
                    else:
                        self._send_file(preview, "image/jpeg", no_cache=True)
                elif path.startswith("/artifacts/"):
                    artifact = controller.artifact_path(
                        path[len("/artifacts/") :]
                    )
                    self._send_file(artifact)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, OSError) as exc:
                self._send_json(
                    {"error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if self.headers.get("X-Control-Token") != controller.token:
                self._send_json(
                    {"error": "控制令牌无效，请刷新页面"},
                    status=HTTPStatus.FORBIDDEN,
                )
                return
            try:
                payload = self._read_json()
                if path == "/api/preview":
                    image = controller.preview(payload)
                    self._send_bytes(
                        image,
                        "image/jpeg",
                        no_cache=True,
                    )
                elif path == "/api/start":
                    self._send_json(controller.start(payload))
                elif path == "/api/stop":
                    self._send_json(controller.stop())
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, RuntimeError, OSError) as exc:
                self._send_json(
                    {"error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )

        def _read_json(self) -> Dict[str, Any]:
            if "application/json" not in self.headers.get(
                "Content-Type",
                "",
            ):
                raise ValueError("请求必须使用 application/json")
            length = int(self.headers.get("Content-Length") or 0)
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("请求内容过大")
            raw = self.rfile.read(length)
            value = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(value, dict):
                raise ValueError("请求 JSON 必须是对象")
            return value

        def _send_json(
            self,
            payload: Dict[str, Any],
            status: HTTPStatus = HTTPStatus.OK,
        ):
            self._send_bytes(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8"),
                "application/json; charset=utf-8",
                status=status,
                no_cache=True,
            )

        def _send_file(
            self,
            path: Path,
            content_type: Optional[str] = None,
            no_cache: bool = False,
        ):
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            resolved_type = (
                content_type
                or mimetypes.guess_type(str(path))[0]
                or "application/octet-stream"
            )
            self._send_bytes(
                path.read_bytes(),
                resolved_type,
                no_cache=no_cache,
            )

        def _send_bytes(
            self,
            data: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
            no_cache: bool = False,
        ):
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            if no_cache:
                self.send_header(
                    "Cache-Control",
                    "no-store, no-cache, must-revalidate",
                )
            self.end_headers()
            self.wfile.write(data)

    return ControlPanelHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="启动本地 ROI 与 Swing 分析控制面板",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--config",
        default="configs/yolo26_tennis_config.yaml",
    )
    parser.add_argument(
        "--roi-config",
        default="configs/roi_config.yaml",
    )
    parser.add_argument(
        "--frontend",
        default="local_control_panel.html",
    )
    parser.add_argument(
        "--username-env",
        default="TENNIS_RTSP_USERNAME",
    )
    parser.add_argument(
        "--password-env",
        default="TENNIS_RTSP_PASSWORD",
    )
    parser.add_argument("--open", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("控制面板仅允许绑定本机回环地址")
    workspace = Path(__file__).resolve().parent
    controller = LocalPipelineController(
        workspace=workspace,
        config_path=workspace / args.config,
        roi_config_path=workspace / args.roi_config,
        frontend_path=workspace / args.frontend,
        username_env=args.username_env,
        password_env=args.password_env,
    )
    server = ThreadingHTTPServer(
        (args.host, int(args.port)),
        create_handler(controller),
    )
    url = f"http://{args.host}:{int(args.port)}/"
    print(f"🎛️ 本地控制面板: {url}")
    print(
        f"🔐 摄像头凭据环境变量: "
        f"{args.username_env} / {args.password_env}"
    )
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n🛑 正在关闭本地控制面板…")
    finally:
        server.shutdown()
        server.server_close()
        controller.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
