#!/usr/bin/env python3
"""Local-only web control panel for ROI preview and main_pipe lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import secrets
import shlex
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
from datetime import datetime, timezone
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
from manual_review_workflow import (
    MAX_ANNOTATION_BYTES,
    discover_session_paths,
    load_review_state,
    process_manual_review,
)
from roi_stream_config import sanitize_stream_source
from swing_session_summary import build_session_coaching_summary
from analysis_metric_delivery import event_analysis_metrics, scoring_blockers
from analysis_provenance import analysis_build_info


from control_panel.settings import (
    CUSTOM_STREAM_ID,
    CUSTOM_STREAM_SCHEMES,
    ENVIRONMENT_NAME_PATTERN,
    LOCAL_VIDEO_ID,
    MAX_REQUEST_BYTES,
    MAX_VIDEO_BYTES,
    SESSION_NAME_PATTERN,
    VIDEO_SUFFIXES,
    ControlSettings,
    _bool,
    _number,
    _session_capture_time,
    inject_rtsp_credentials,
    load_local_environment_variable,
    load_stream_profiles,
    sort_points_tl_tr_br_bl,
    validate_custom_stream_source,
)
from control_panel.http_handler import create_handler
from control_panel.session_store import SessionStore

_CONTROL_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_CONTROL_IMPORTED_AT_UNIX_NS = time.time_ns()


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
        # Load only camera credentials, never expose file contents to the browser.
        credentials_path = self.workspace / ".camera-credentials.local.env"
        for name in (self.username_env, self.password_env):
            load_local_environment_variable(credentials_path, name)
        self.token = secrets.token_urlsafe(24)
        self.streams = load_stream_profiles(self.roi_config_path)
        for stream in self.streams:
            for name in self._camera_variable_names(stream["stream_id"]):
                load_local_environment_variable(credentials_path, name)
        self._lock = threading.RLock()
        self._change_condition = threading.Condition()
        self._server_stop_event = threading.Event()
        self._process: Optional[subprocess.Popen] = None
        self._active_racket_model = None
        self._reader_thread: Optional[threading.Thread] = None
        self._runtime_directory: Optional[Path] = None
        self._state = "stopped"
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None
        self._returncode: Optional[int] = None
        self._logs = deque(maxlen=300)
        self._public_command: List[str] = []
        self._artifacts: Dict[str, str] = {}
        self._ground_contexts = {}
        self._session_store = SessionStore(self.workspace, self._lock, lambda: self._artifacts)

    def _notify_change(self) -> None:
        with self._change_condition:
            self._change_condition.notify_all()

    def is_server_stopped(self) -> bool:
        return self._server_stop_event.is_set()

    def stop_server(self) -> None:
        self._server_stop_event.set()
        self._notify_change()

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
            "camera_credentials_configured": {
                stream["stream_id"]: all(self._saved_credentials(stream["stream_id"]))
                for stream in self.streams
            },
            "streams": self.streams,
            "racket_model_profiles": {
                "configured": {"available": True, "imgsz": (config.get('racket_pose') or {}).get('imgsz', 960)},
                "640_shadow": {"available": (self.workspace / 'models/racket_pose/candidates/bn-repaired-640-v1/racket_pose_bounded_fp16_640.mlpackage').is_dir(), "imgsz": 640},
            },
            "credentials_configured": bool(
                os.environ.get(self.username_env)
                and os.environ.get(self.password_env)
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
                "racket_pose_mode": "legacy",
                "processing_mode": "buffered",
                "inference_schedule": "barrier",
                "racket_ipc": "queue",
                "racket_model_profile": "configured",
                "save_video": False,
                "realtime_swing_events": True,
                "realtime_frame_output": bool(
                    realtime.get("frame_output_enabled", False)
                ),
                "evidence_bundle": True,
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
                "coach_tts": bool((realtime.get("coach_tts") or {}).get("enabled", False)),
                "coach_tts_playback": bool((realtime.get("coach_tts") or {}).get("playback", True)),
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
                "racket_model": self._active_racket_model,
                "service_build": {
                    "schema": "tennis.control-service-build.v1",
                    "pid": os.getpid(),
                    "loaded_at_unix_ns": _CONTROL_IMPORTED_AT_UNIX_NS,
                    "control_source_sha256": _CONTROL_SOURCE_SHA256,
                    "hash_semantics": "source files at module import; restart required after source edits",
                    "analysis_build": analysis_build_info(),
                },
            }

    def preview(self, payload: Dict[str, Any]) -> bytes:
        stream = self._stream_from_payload(payload)
        source = self._authenticated_source(stream, payload)
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        frame = capture_calibration_frame(
            source,
            config,
            warmup_frames=3,
        )
        clean = bool(payload.get("clean") or payload.get("raw"))
        if clean:
            height, width = frame.shape[:2]
            max_width = 1600
            if width > max_width:
                scale = max_width / width
                frame = cv2.resize(
                    frame,
                    (max_width, int(round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            preview = frame
        else:
            preview = self._draw_roi_preview(frame, stream)
        ok, encoded = cv2.imencode(
            ".jpg",
            preview,
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
        if not ok:
            raise RuntimeError("ROI 预览图编码失败")
        return encoded.tobytes()

    def _ground_binding(self, payload):
        from ground_calibration_store import source_binding
        stream = self._stream_from_payload(payload)
        source = self._authenticated_source(stream, payload)
        identity = stream.get('mapped_stream_id') or stream['stream_id']
        return source, source_binding(source, identity)

    def ground_info(self, payload):
        _, binding = self._ground_binding(payload)
        with self._lock:
            resolved = self._resolve_ground_calibration(binding)
        return {'binding': binding, **resolved}

    def _ground_camera_binding(self, stream_id):
        from ground_calibration_store import source_binding
        stream = self._stream(stream_id)
        return source_binding(stream['source'], stream_id)

    def _resolve_ground_calibration(self, binding):
        from ground_calibration_store import GroundCalibrationStore
        from ground_camera_profiles import GroundCameraProfiles
        profiles = GroundCameraProfiles(self.workspace)
        if profiles.directory.exists() and any(s['stream_id'] == binding['stream_id'] for s in self.streams):
            resolved = profiles.resolve(binding, self._ground_camera_binding(binding['stream_id']))
            if resolved is not None: return resolved
        return {'calibration': GroundCalibrationStore(self.workspace).load(binding), 'application': None}

    def import_ground_camera_profile(self, payload):
        from ground_camera_profiles import GroundCameraProfiles
        target = payload.get('target_stream_id')
        if not isinstance(target, str) or not target:
            raise ValueError('请选择共享标定的目标机位')
        with self._lock:
            binding = self._ground_camera_binding(target)
            profile = GroundCameraProfiles(self.workspace).save(payload.get('calibration'), binding)
        return {'profile': profile, 'message': '已应用为机位共享标定；下次分析使用。仅用于该机位实时输入及明确绑定该机位的同尺寸视频。'}

    def ground_preview(self, payload):
        import base64
        from ground_calibration_store import GroundCalibrationStore
        source, binding = self._ground_binding(payload)
        config = yaml.safe_load(self.config_path.read_text(encoding='utf-8')) or {}
        frame = capture_calibration_frame(source, config, warmup_frames=3)
        height, width = frame.shape[:2]
        if width > 1600:
            frame = cv2.resize(frame, (1600, round(height*1600/width)))
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError('地面标定预览编码失败')
        nonce = secrets.token_hex(16)
        with self._lock:
            self._ground_contexts[nonce] = {'binding': binding, 'image_size': [width, height]}
            while len(self._ground_contexts) > 20:
                self._ground_contexts.pop(next(iter(self._ground_contexts)))
            resolved = self._resolve_ground_calibration(binding)
            document = resolved['calibration']
            reason = None
            if document and document['image_size'] != [width, height]:
                document = None
                reason = '标定原图尺寸不同，不能自动缩放；请重新核对当前机位四角'
        return {'context_id': nonce, 'binding': binding, 'image_size': [width, height],
                'image_base64': base64.b64encode(encoded).decode(), 'calibration': document,
                'application': resolved['application'], 'calibration_unavailable_reason': reason}

    def save_ground_calibration(self, payload):
        from ground_calibration_store import GroundCalibrationStore
        from ground_reference import normalize_calibration
        document = normalize_calibration(payload.get('calibration'))
        with self._lock:
            context = self._ground_contexts.get(payload.get('context_id'))
            if context is None:
                raise ValueError('请先加载当前输入的地面标定原图')
            _, binding = self._ground_binding(payload)
            if document['binding'] != context['binding'] or binding != context['binding']:
                raise ValueError('输入已改变，请重新加载标定原图')
            if document['image_size'] != context['image_size']:
                raise ValueError('标定原图尺寸不匹配')
            result = GroundCalibrationStore(self.workspace).save(document, binding)
        return {'calibration': result, 'message': '已保存独立标定版本；下次分析加载。着地与准确性仍待核验。'}

    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        settings = ControlSettings.from_payload(payload)
        stream = self._stream_from_payload(payload)
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
            from racket_model_profiles import apply_model_profile
            apply_model_profile(config, settings.racket_model_profile, settings.racket_pose_mode, self.workspace)
            config["video_output_path"] = str(
                output_dir / f"{settings.session_name}_live.mp4"
            )
            roi = config.setdefault("roi_settings", {})
            roi["enabled"] = settings.roi_enabled
            roi["auto_load_config"] = True
            roi["roi_config_path"] = str(self.roi_config_path)
            roi["crop_margin"] = settings.crop_margin
            target_id = (
                settings.mapped_stream_id
                if settings.stream_id == LOCAL_VIDEO_ID
                else (
                    settings.stream_id
                    if settings.stream_id not in (LOCAL_VIDEO_ID, CUSTOM_STREAM_ID)
                    else ""
                )
            )
            if target_id:
                roi["target_stream_id"] = target_id
                config["target_stream_id"] = target_id
            from ground_calibration_store import GroundCalibrationStore, source_binding
            from ground_camera_profiles import GroundCameraProfiles
            ground_stream_id = target_id or stream['stream_id']
            store = GroundCalibrationStore(self.workspace)
            resolved = {'calibration': None, 'application': None}
            if store.directory.exists() or GroundCameraProfiles(self.workspace).directory.exists():
                binding = source_binding(source, ground_stream_id)
                resolved = self._resolve_ground_calibration(binding)
            config['ground_reference'] = ({**resolved,
                                         'context_stream_id': ground_stream_id}
                                        if resolved['calibration'] is not None else {})
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
                f"[control] 启动 {stream['label']} · "
                f"{stream.get('public_source') or stream['source']}"
            )
            model_config = config.get('racket_pose') or {}
            self._active_racket_model = {
                'profile': settings.racket_model_profile, 'mode': settings.racket_pose_mode,
                'path': model_config.get('model_path'), 'imgsz': model_config.get('imgsz', 960),
                'compute_units': model_config.get('compute_units', 'ANE'),
            }
            self._logs.append('[control] 球拍模型: {} | {}x{} | {} | {}'.format(
                model_config.get('model_path', '未配置'), model_config.get('imgsz', 960),
                model_config.get('imgsz', 960), model_config.get('compute_units', 'ANE'), settings.racket_pose_mode))
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
            self._artifacts.clear()
            self._artifacts.update(artifacts)
            self._reader_thread = threading.Thread(
                target=self._read_output,
                args=(self._process,),
                daemon=True,
                name="tennis-control-log-reader",
            )
            self._reader_thread.start()
            self._notify_change()
            return self.status()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            self._refresh_process_state()
            if (
                self._process is None
                or self._process.poll() is not None
            ):
                self._state = "stopped"
                self._notify_change()
                return self.status()
            self._state = "stopping"
            self._logs.append("[control] 正在发送安全停止信号…")
            self._notify_change()
            try:
                os.killpg(self._process.pid, signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                self._process.send_signal(signal.SIGINT)
            return self.status()

    def shutdown(self) -> None:
        self.stop_server()
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
        return self._session_store.artifact_path(request_path)

    def live_preview_path(self) -> Optional[Path]:
        return self._session_store.live_preview_path()

    def manual_review_paths(self, report_path: Optional[str] = None) -> Dict[str, Path]:
        return self._session_store.manual_review_paths(report_path)

    def manual_review_state(self, report_path: Optional[str] = None) -> Dict[str, Any]:
        return self._session_store.manual_review_state(report_path)

    def session_events(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        return self._session_store.session_events(session_id)

    def session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        return self._session_store.session_summary(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        return self._session_store.list_sessions()

    def evaluate_manual_review(
        self,
        payload: Dict[str, Any],
        report_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return process_manual_review(
            self.manual_review_paths(report_path),
            payload,
        )

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

    def upload_video(self, body, length: int, suffix: str) -> Dict[str, Any]:
        """Stream a bounded upload to private storage; expose only opaque IDs."""
        if not 0 < length <= MAX_VIDEO_BYTES:
            raise ValueError("视频大小必须在 0 到 2 GB 之间")
        suffix = suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise ValueError("不支持的视频格式")
        directory = self.workspace / "data" / "control_uploads"
        directory.mkdir(parents=True, exist_ok=True)
        video_id = secrets.token_hex(16) + suffix
        target = directory / video_id
        try:
            with target.open("xb") as output:
                target.chmod(0o600)
                remaining = length
                while remaining:
                    chunk = body.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("视频上传未完成，请重新选择文件")
                    output.write(chunk)
                    remaining -= len(chunk)
            capture = cv2.VideoCapture(str(target))
            try:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError("无法解码视频，请选择可播放的视频文件")
            finally:
                capture.release()
            self._last_uploaded_video = target
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return {"video_id": video_id, "size": length}

    def _get_mirror_calibrator(self, stream_id: Optional[str] = None) -> Any:
        from calibrate_mirror import MirrorCalibrationServer
        video_to_use = getattr(self, "_last_uploaded_video", None)
        if video_to_use is None or not video_to_use.exists():
            candidates = [
                Path("/Users/krum5539/Desktop/Camera/test/40.26.mp4"),
                Path("/Users/krum5539/Desktop/Camera/49.35.mp4"),
            ]
            for c in candidates:
                if c.exists():
                    video_to_use = c
                    break
        if video_to_use is None or not video_to_use.exists():
            video_to_use = Path("/Users/krum5539/Desktop/Camera/test/40.26.mp4")

        return MirrorCalibrationServer(
            video_path=video_to_use,
            config_path=self.workspace / "configs/dual_view_config.yaml",
            html_path=self.workspace / "mirror_calibration.html",
            stream_id=stream_id,
            roi_config_path=self.roi_config_path,
        )

    def mirror_info(self, stream_id: Optional[str] = None) -> Dict[str, Any]:
        calib = self._get_mirror_calibrator(stream_id=stream_id)
        return {
            "video_path": str(calib.video_path),
            "width": calib.width,
            "height": calib.height,
            "total_frames": calib.total_frames,
            "fps": calib.fps,
            "current_config": calib.get_current_config(stream_id=stream_id),
        }

    def mirror_frame(self, index: int, stream_id: Optional[str] = None) -> bytes:
        calib = self._get_mirror_calibrator(stream_id=stream_id)
        return calib.get_frame_jpeg(index)

    def mirror_save_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        stream_id = payload.get("stream_id")
        calib = self._get_mirror_calibrator(stream_id=stream_id)
        roi = payload.get("reflection_roi", [])
        poly = payload.get("polygon", [])
        mask_poly = payload.get("mask_polygon", [])
        return calib.save_config(roi, poly, mask_poly, stream_id=stream_id)

    def mirror_preview_backview(self, payload: Dict[str, Any]) -> bytes:
        calib = self._get_mirror_calibrator()
        f_idx = int(payload.get("frame_index", 0))
        roi = payload.get("reflection_roi", [0.27, 0.10, 0.61, 0.46])
        poly = payload.get("polygon", [])
        mask_poly = payload.get("mask_polygon", [])
        mode = payload.get("mode", "back")
        return calib.render_backview_preview(f_idx, roi, poly, mask_poly, preview_mode=mode)

    def roi_info(self, stream_id: Optional[str] = None) -> Dict[str, Any]:
        target_id = stream_id or (self.streams[0]["stream_id"] if self.streams else "")
        matched = next((s for s in self.streams if s["stream_id"] == target_id), None)
        if not matched:
            raise ValueError(f"未找到机位配置: {target_id}")
        return {
            "stream_id": matched["stream_id"],
            "label": matched["label"],
            "points": list(matched.get("points") or []),
            "frame_size": list(matched.get("frame_size") or [2560, 1440]),
            "mirror_view": dict(matched.get("mirror_view") or {}),
            "mirror_exclusion": True,
            "exclusion_polygons": list(matched.get("exclusion_polygons") or []),
        }

    def save_roi_boundary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        stream_id = str(payload.get("stream_id") or "").strip()
        points = payload.get("points") or payload.get("roi_points") or []
        if not stream_id:
            raise ValueError("缺少 stream_id")
        if not isinstance(points, (list, tuple)) or len(points) < 3:
            raise ValueError("ROI 边界至少需要 3 个顶点坐标")
        cleaned_points = []
        for p in points:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                raise ValueError(f"无效的顶点坐标: {p}")
            cleaned_points.append([int(round(float(p[0]))), int(round(float(p[1])))])
        if len(cleaned_points) == 4:
            cleaned_points = sort_points_tl_tr_br_bl(cleaned_points)

        if not self.roi_config_path.exists():
            raise ValueError("ROI 配置文件不存在")
        doc = yaml.safe_load(self.roi_config_path.read_text(encoding="utf-8")) or {}
        streams = doc.get("streams") or []
        matched = next((s for s in streams if s.get("stream_id") == stream_id), None)
        if not matched:
            raise ValueError(f"在配置文件中未找到机位: {stream_id}")

        matched["roi_points"] = cleaned_points
        matched["roi_description"] = f"{len(cleaned_points)}-point ROI: {cleaned_points}"
        matched["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        if "mirror_exclusion" in payload:
            mv = matched.setdefault("mirror_view", {})
            mv["exclusion_enabled"] = bool(payload["mirror_exclusion"])

        if "mirror_polygon" in payload and isinstance(payload["mirror_polygon"], list):
            mv = matched.setdefault("mirror_view", {})
            raw_mirror = payload["mirror_polygon"]
            fw, fh = matched.get("frame_size") or [2560, 1440]
            norm_mirror = []
            for p in raw_mirror:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    nx = float(p[0]) / fw if float(p[0]) > 1.0 else float(p[0])
                    ny = float(p[1]) / fh if float(p[1]) > 1.0 else float(p[1])
                    norm_mirror.append([round(max(0.0, min(1.0, nx)), 4), round(max(0.0, min(1.0, ny)), 4)])
            if len(norm_mirror) >= 3:
                if len(norm_mirror) == 4:
                    norm_mirror = sort_points_tl_tr_br_bl(norm_mirror)
                mv["polygon"] = norm_mirror
                xs = [p[0] for p in norm_mirror]
                ys = [p[1] for p in norm_mirror]
                mv["reflection_roi"] = [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)]
                mv["enabled"] = True
                dual_view_path = self.workspace / "configs" / "dual_view_config.yaml"
                if dual_view_path.exists():
                    try:
                        dv_doc = yaml.safe_load(dual_view_path.read_text(encoding="utf-8")) or {}
                        if "mirror_view" in dv_doc:
                            dv_doc["mirror_view"]["polygon"] = norm_mirror
                            dv_doc["mirror_view"]["reflection_roi"] = mv["reflection_roi"]
                            dual_view_path.write_text(yaml.safe_dump(dv_doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
                    except Exception as e:
                        self.logger.warning(f"Failed to sync mirror_view to dual_view_config: {e}")

        if "exclusion_polygons" in payload and isinstance(payload["exclusion_polygons"], list):
            matched["exclusion_polygons"] = payload["exclusion_polygons"]

        self.roi_config_path.write_text(
            yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        self.streams = load_stream_profiles(self.roi_config_path)
        return {
            "success": True,
            "stream_id": stream_id,
            "points": cleaned_points,
            "mirror_view": dict(matched.get("mirror_view") or {}),
            "message": "ROI 边界与镜面排除配置已成功保存",
        }

    def _stream_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        stream_id = str(payload.get("stream_id") or "").strip()
        if stream_id == LOCAL_VIDEO_ID:
            video_id = str(payload.get("video_id") or "")
            if not re.fullmatch(r"[0-9a-f]{32}\.(mp4|mov|mkv|avi|webm|m4v)", video_id):
                raise ValueError("请先选择并上传本地视频")
            directory = (self.workspace / "data" / "control_uploads").resolve()
            source = (directory / video_id).resolve()
            if source.parent != directory or not source.is_file():
                raise ValueError("本地视频不存在，请重新上传")

            mapped_id = str(payload.get("mapped_stream_id") or "").strip()
            matched = next((s for s in self.streams if s["stream_id"] == mapped_id), None) if mapped_id else None
            if matched:
                return {
                    "stream_id": LOCAL_VIDEO_ID,
                    "mapped_stream_id": mapped_id,
                    "label": f"本地视频 ({matched['label']})",
                    "source": str(source),
                    "public_source": video_id,
                    "roi_enabled": bool(matched.get("roi_enabled", True)),
                    "points": list(matched.get("points") or []),
                    "frame_size": list(matched.get("frame_size") or [0, 0]),
                    "mirror_view": dict(matched.get("mirror_view") or {}),
                    "default": False,
                }
            return {
                "stream_id": LOCAL_VIDEO_ID,
                "mapped_stream_id": "",
                "label": "本地视频",
                "source": str(source),
                "public_source": video_id,
                "roi_enabled": False,
                "points": [],
                "frame_size": [0, 0],
                "mirror_view": {},
                "default": False,
            }
        if stream_id != CUSTOM_STREAM_ID:
            return self._stream(stream_id)

        source = validate_custom_stream_source(
            str(payload.get("custom_stream_source") or "")
        )
        public_source = sanitize_stream_source(source)
        matched = next(
            (
                stream
                for stream in self.streams
                if stream["source"] == public_source
            ),
            None,
        )
        return {
            "stream_id": CUSTOM_STREAM_ID,
            "credentials_stream_id": (matched or {}).get("stream_id", CUSTOM_STREAM_ID),
            "label": "自定义码流",
            "source": source,
            "public_source": public_source,
            "roi_enabled": bool((matched or {}).get("roi_enabled", False)),
            "points": list((matched or {}).get("points") or []),
            "frame_size": list((matched or {}).get("frame_size") or [0, 0]),
            "default": False,
        }

    @staticmethod
    def _camera_variable_names(stream_id: str):
        key = re.sub(r"[^A-Za-z0-9_]", "_", stream_id).upper()
        return f"TENNIS_CAMERA_{key}_USERNAME", f"TENNIS_CAMERA_{key}_PASSWORD"

    def _saved_credentials(self, stream_id: str):
        username_key, password_key = self._camera_variable_names(stream_id)
        username = os.environ.get(username_key, "")
        if username:
            # Credentials are a pair: never combine one camera's user with a default password.
            return username, os.environ.get(password_key, "")
        return os.environ.get(self.username_env, ""), os.environ.get(self.password_env, "")

    def _authenticated_source(
        self,
        stream: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> str:
        username = str(payload.get("username") or "").strip()
        password = str(payload.get("password") or "")
        if not username:
            username, password = self._saved_credentials(
                stream.get("credentials_stream_id") or stream.get("stream_id", "")
            )
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
        evidence_manifest = session_dir / f"{stem}_evidence_manifest.json"
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
        ]
        command.extend(['--racket-pose-mode', settings.racket_pose_mode,
                        '--inference-schedule', settings.inference_schedule,'--racket-ipc',settings.racket_ipc])
        if settings.processing_mode != 'auto':
            command.extend(['--processing-mode', settings.processing_mode])
        target_stream_id = (
            settings.mapped_stream_id
            if settings.stream_id == LOCAL_VIDEO_ID
            else (
                settings.stream_id
                if settings.stream_id not in (LOCAL_VIDEO_ID, CUSTOM_STREAM_ID)
                else ""
            )
        )
        if target_stream_id:
            command.extend(["--stream-id", target_stream_id])
        if settings.algo2_dual_view:
            command.append("--algo2-dual-view")
        else:
            command.append("--no-dual-view")
        if settings.live_mode and settings.processing_mode in ('auto', 'realtime'):
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
        if settings.realtime_frame_output or settings.evidence_bundle:
            command.extend(
                [
                    "--realtime-frame-output",
                    "--realtime-frame-jsonl",
                    str(frame_jsonl),
                    "--realtime-frame-snapshot-json",
                    str(frame_snapshot),
                ]
            )
        if settings.evidence_bundle:
            command.extend(["--evidence-manifest", str(evidence_manifest)])
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
        if settings.coach_tts:
            command.append("--realtime-coach-tts")
            if not settings.coach_tts_playback:
                command.append("--realtime-coach-tts-no-playback")
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
            "frame_jsonl": relative(frame_jsonl),
            "frame_snapshot": relative(frame_snapshot),
            "evidence_manifest": relative(evidence_manifest),
            "evidence_manifest_url": (
                f"/artifacts/{quote(relative(evidence_manifest))}"
            ),
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
        if len(points) >= 3:
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

        # 绘制镜面反射排除区域（阻断镜中球与虚影）
        mirror_poly = stream.get("mirror_view", {}).get("polygon") or []
        if len(mirror_poly) >= 3:
            m_pts = np.asarray(
                [
                    [
                        int(round(float(p[0]) * width)),
                        int(round(float(p[1]) * height)),
                    ]
                    for p in mirror_poly
                ],
                dtype=np.int32,
            )
            m_overlay = preview.copy()
            cv2.fillPoly(m_overlay, [m_pts], (0, 0, 200))
            preview = cv2.addWeighted(m_overlay, 0.22, preview, 0.78, 0)
            cv2.polylines(preview, [m_pts], True, (0, 0, 255), max(2, int(round(width / 900))), cv2.LINE_AA)
            tx = int(np.min(m_pts[:, 0])) + 15
            ty = int(np.min(m_pts[:, 1])) + 28
            cv2.putText(
                preview,
                "MIRROR EXCLUSION (BALL/RACKET BLOCKED)",
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 60, 255),
                2,
                cv2.LINE_AA,
            )
            sorted_m_pts = sort_points_tl_tr_br_bl(m_pts.tolist()) if len(m_pts) == 4 else m_pts
            for index, point in enumerate(sorted_m_pts, start=1):
                x, y = int(point[0]), int(point[1])
                cv2.circle(preview, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(
                    preview,
                    f"M{index}",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
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
            stream.get("public_source") or stream["source"],
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
                        self._notify_change()
            returncode = process.wait()
            with self._lock:
                self._returncode = int(returncode)
                self._stopped_at = time.time()
                self._state = "stopped" if returncode == 0 else "failed"
                self._logs.append(
                    f"[control] 进程结束，返回码 {returncode}"
                )
            self._notify_change()
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
        old_state = self._state
        self._returncode = int(returncode)
        if self._state in {"running", "stopping"}:
            self._state = "stopped" if returncode == 0 else "failed"
            self._stopped_at = time.time()
            if self._state != old_state:
                self._notify_change()

    def _cleanup_runtime_directory(self) -> None:
        with self._lock:
            directory = self._runtime_directory
            self._runtime_directory = None
        if directory is not None and directory.exists():
            shutil.rmtree(directory, ignore_errors=True)


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
    config_path = (workspace / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    deepseek_config = (config.get("realtime_swing") or {}).get("deepseek") or {}
    deepseek_api_key_env = str(
        deepseek_config.get("api_key_env") or "DEEPSEEK_API_KEY"
    )
    load_local_environment_variable(
        workspace / ".env.local",
        deepseek_api_key_env,
    )
    controller = LocalPipelineController(
        workspace=workspace,
        config_path=config_path,
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
    print(
        f"🧠 DeepSeek密钥环境变量: {deepseek_api_key_env} | "
        f"{'已配置' if os.environ.get(deepseek_api_key_env) else '未配置'}"
    )
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n🛑 正在关闭本地控制面板…")
    finally:
        controller.stop_server()
        server.shutdown()
        server.server_close()
        controller.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
