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
        configured_pose_model = str(config.get("yolo_pose_model_path") or "yolo26m-pose.mlpackage")
        det_cfg = config.get("unified_detection") or {}
        det_model_path = str(det_cfg.get("model_path") or "yolo26n.mlpackage")
        det_path_obj = Path(det_model_path)
        det_raw_name = (
            det_path_obj.parent.name
            if det_path_obj.name == "model.mlpackage" and det_path_obj.parent.name
            else det_path_obj.stem
        )
        exp_match = re.search(r"(exp\d+-\d+)", det_raw_name)
        det_model_label = exp_match.group(1) if exp_match else det_raw_name
        model_640_available = (
            self.workspace
            / "models/racket_pose/candidates/bn-repaired-640-v1/racket_pose_bounded_fp16_640.mlpackage"
        ).is_dir()
        pose_n_available = (self.workspace / "yolo26n-pose.mlpackage").is_dir()
        return {
            "token": self.token,
            "camera_credentials_configured": {
                stream["stream_id"]: all(self._saved_credentials(stream["stream_id"]))
                for stream in self.streams
            },
            "streams": self.streams,
            "detector_model": {
                "path": det_model_path,
                "label": det_model_label,
                "compute_units": str(det_cfg.get("compute_units") or "ANE"),
            },
            "racket_model_profiles": {
                "configured": {"available": True, "imgsz": (config.get('racket_pose') or {}).get('imgsz', 960)},
                "640_shadow": {"available": model_640_available, "imgsz": 640},
            },
            "pose_models": {
                "configured": {"available": True, "path": configured_pose_model},
                "yolo26n-pose.mlpackage": {"available": pose_n_available},
                "yolo26m-pose.mlpackage": {
                    "available": (
                        (self.workspace / "yolo26m-pose.mlpackage").is_dir()
                        or (self.workspace / "models/yolo26m-pose.mlpackage").is_dir()
                    )
                },
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
                "analysis_mode": "standard" if model_640_available else "lite",
                "racket_pose_mode": "shadow" if model_640_available else "legacy",
                "processing_mode": "buffered",
                "inference_schedule": "barrier",
                "racket_ipc": "shared_memory" if model_640_available else "queue",
                "racket_model_profile": "640_shadow" if model_640_available else "configured",
                "pose_model": "yolo26n-pose.mlpackage" if pose_n_available else "configured",
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

    def _resolve_pose_model_path(self, pose_model: str, config: Dict[str, Any]) -> str:
        if pose_model == "configured":
            return str(config.get("yolo_pose_model_path") or "yolo26m-pose.mlpackage")
        if (
            pose_model == "yolo26m-pose.mlpackage"
            and not (self.workspace / "yolo26m-pose.mlpackage").is_dir()
            and (self.workspace / "models/yolo26m-pose.mlpackage").is_dir()
        ):
            return "models/yolo26m-pose.mlpackage"
        if not (self.workspace / pose_model).exists():
            raise ValueError(f"姿态模型不存在: {pose_model}")
        return pose_model

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
            effective_pose_model = self._resolve_pose_model_path(settings.pose_model, config)
            if settings.pose_model != "configured":
                config["yolo_pose_model_path"] = effective_pose_model
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
                effective_pose_model=effective_pose_model,
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
            req_player_id = str(payload.get("player_id") or "").strip()
            if req_player_id and artifacts.get("session_id"):
                try:
                    self._session_store.assign_session_player(
                        {"session_id": artifacts["session_id"], "player_id": req_player_id}
                    )
                except Exception:
                    pass
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

    def list_players(self) -> List[Dict[str, Any]]:
        return self._session_store.list_players()

    def create_or_update_player(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._session_store.create_or_update_player(payload)

    def assign_session_player(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._session_store.assign_session_player(payload)

    def assign_sessions_player_batch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._session_store.assign_sessions_player_batch(payload)

    def compare_player_sessions(
        self,
        player_id: Optional[str] = None,
        stroke_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._session_store.compare_player_sessions(player_id, stroke_type)

    def get_player_global_profile(
        self,
        player_id: Optional[str] = None,
        days: Optional[int] = None,
        dedup: bool = True,
    ) -> Dict[str, Any]:
        return self._session_store.get_player_global_profile(player_id, days, dedup=dedup)

    def list_duplicate_sessions(self, player_id: Optional[str] = None) -> Dict[str, Any]:
        return self._session_store.list_duplicate_sessions(player_id=player_id)

    def get_swing_trajectory(self, session_id: str, event_id: int) -> Dict[str, Any]:
        return self._session_store.get_swing_trajectory(session_id=session_id, event_id=event_id)

    def get_player_trajectory_summary(
        self,
        player_id: Optional[str] = None,
        stroke_side: str = "forehand",
        days: Optional[int] = None,
        dedup: bool = True,
    ) -> Dict[str, Any]:
        return self._session_store.get_player_trajectory_summary(
            player_id=player_id,
            stroke_side=stroke_side,
            days=days,
            dedup=dedup,
        )

    def cleanup_duplicate_sessions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_ids = payload.get("session_ids")
        if not isinstance(session_ids, list):
            raise ValueError("session_ids 必须是列表")
        mode = str(payload.get("mode") or "archive").strip().lower()
        return self._session_store.cleanup_duplicate_sessions(session_ids=session_ids, mode=mode)

    def generate_player_global_review(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        player_id = str(payload.get("player_id") or "").strip() or None
        days_raw = payload.get("days")
        days = int(days_raw) if days_raw not in (None, "", "all") else None
        dedup_raw = payload.get("dedup", True)
        dedup_val = (
            str(dedup_raw).strip().lower() not in ("0", "false", "no", "off")
            if isinstance(dedup_raw, str)
            else bool(dedup_raw)
        )
        requested_side = str(payload.get("stroke_side") or "").strip().lower()
        profile = self.get_player_global_profile(player_id=player_id, days=days, dedup=dedup_val)
        fh = profile.get("forehand") or {}
        bh = profile.get("backhand") or {}
        if requested_side not in ("forehand", "backhand"):
            requested_side = "forehand" if (fh.get("total_swings", 0) > 0 or bh.get("total_swings", 0) == 0) else "backhand"
        side_data = fh if requested_side == "forehand" else bh
        side_label = "正手 (Forehand)" if requested_side == "forehand" else "反手 (Backhand)"
        if side_data.get("total_swings", 0) < 1:
            raise ValueError(f"该学员在当前范围内暂无足够的{side_label}实战挥拍记录，无法生成分析总评")

        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        rt_cfg = config.get("realtime_swing") or {}
        raw_provider = str(payload.get("provider") or "deepseek").strip().lower()
        requested_model = str(payload.get("model") or "").strip()
        if raw_provider in ("gemini", "google") or requested_model.lower().startswith("gemini-"):
            provider_key = "gemini"
            provider_label = "Google Gemini"
            gemini_cfg = rt_cfg.get("gemini") or {}
            api_key_env = str(gemini_cfg.get("api_key_env") or "GEMINI_API_KEY")
            base_url_env = str(gemini_cfg.get("base_url_env") or "GEMINI_BASE_URL")
            model_env = str(gemini_cfg.get("model_env") or "GEMINI_MODEL")
            load_local_environment_variable(self.workspace / ".env.local", api_key_env)
            load_local_environment_variable(self.workspace / ".env.local", base_url_env)
            load_local_environment_variable(self.workspace / ".env.local", model_env)
            api_key = str(os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY") or "").strip()
            if not api_key:
                raise RuntimeError(
                    f"未配置 {api_key_env}，请在 .env.local 中添加 {api_key_env}=AIzaSy...（或配置 OPENAI_API_KEY 中转）"
                )
            base_url = str(
                os.environ.get(base_url_env)
                or gemini_cfg.get("base_url")
                or "https://generativelanguage.googleapis.com/v1beta/openai/"
            ).strip()
            configured_model = (
                requested_model
                or str(os.environ.get(model_env) or "").strip()
                or str(gemini_cfg.get("model") or "gemini-2.5-flash").strip()
            )
            candidate_models = list(dict.fromkeys([configured_model, "gemini-2.5-flash", "gemini-2.0-flash"]))
        elif raw_provider in ("openai", "gpt", "claude") or requested_model.lower().startswith(("gpt-", "o1", "o3", "o4", "claude-")):
            is_claude = "claude" in requested_model.lower() or raw_provider == "claude"
            provider_key = "openai"
            provider_label = "Claude" if is_claude else "OpenAI GPT"
            openai_cfg = rt_cfg.get("openai") or {}
            api_key_env = str(openai_cfg.get("api_key_env") or "OPENAI_API_KEY")
            base_url_env = str(openai_cfg.get("base_url_env") or "OPENAI_BASE_URL")
            model_env = str(openai_cfg.get("model_env") or "OPENAI_MODEL")
            load_local_environment_variable(self.workspace / ".env.local", api_key_env)
            load_local_environment_variable(self.workspace / ".env.local", base_url_env)
            load_local_environment_variable(self.workspace / ".env.local", model_env)
            api_key = str(os.environ.get(api_key_env) or "").strip()
            if not api_key:
                raise RuntimeError(
                    f"未配置 {api_key_env} 环境变量，请在 .env.local 中添加 {api_key_env}=sk-...（可选配置 {base_url_env}）"
                )
            base_url = str(
                os.environ.get(base_url_env)
                or openai_cfg.get("base_url")
                or "https://api.openai.com/v1"
            ).strip()
            configured_model = (
                requested_model
                or str(os.environ.get(model_env) or "").strip()
                or str(openai_cfg.get("model") or "o3-mini").strip()
            )
            candidate_models = list(dict.fromkeys([configured_model, "gpt-4o", "gpt-4o-mini"]))
        else:
            provider_key = "deepseek"
            provider_label = "DeepSeek"
            ds_cfg = rt_cfg.get("deepseek") or {}
            api_key_env = str(ds_cfg.get("api_key_env") or "DEEPSEEK_API_KEY")
            load_local_environment_variable(self.workspace / ".env.local", api_key_env)
            load_local_environment_variable(self.workspace / ".env.local", "DEEPSEEK_BASE_URL")
            load_local_environment_variable(self.workspace / ".env.local", "DEEPSEEK_MODEL")
            api_key = str(os.environ.get(api_key_env) or "").strip()
            if not api_key:
                raise RuntimeError(f"未配置 {api_key_env} 环境变量，无法调用 DeepSeek")
            base_url = (
                str(os.environ.get("DEEPSEEK_BASE_URL") or "").strip()
                or str(ds_cfg.get("base_url") or "https://api.deepseek.com").strip()
            )
            configured_model = (
                requested_model
                or str(os.environ.get("DEEPSEEK_MODEL") or "").strip()
                or str(ds_cfg.get("model") or "deepseek-flash").strip()
            )
            candidate_models = list(dict.fromkeys([configured_model, "deepseek-flash", "deepseek-chat"]))

        import statistics as _stats

        # Pull PCHIP+SG fitted per-swing trajectory diagnostics so AI sees the same clean data as the UI
        traj_summary = self.get_player_trajectory_summary(
            player_id=player_id,
            stroke_side=requested_side,
            days=days,
            dedup=dedup_val,
        )
        traj_map = {
            (str(it.get("session_id")), int(it.get("event_id"))): it
            for it in (traj_summary.get("swings") or [])
        }
        side_prefix = "反" if requested_side == "backhand" else "正"
        raw_swings = side_data.get("swings") or []

        def _val_of(sw: Dict[str, Any], dg: Optional[Dict[str, Any]], key: str) -> Optional[float]:
            if dg and isinstance(dg.get("summary"), dict):
                rv = (dg["summary"].get(key) or {}).get("refined")
                if rv is not None:
                    return float(rv)
            fb = sw.get("racket_max_speed_px_s") if key == "racket_max_speed" else sw.get(key)
            return float(fb) if fb is not None else None

        valid_swings_clean: List[Dict[str, Any]] = []
        tier_c_labels: List[str] = []
        for idx, sw in enumerate(raw_swings):
            sid = str(sw.get("session_id") or "")
            eid = int(sw.get("event_id") or 0)
            dg = traj_map.get((sid, eid))
            tier = str(dg.get("tier") or "A") if dg else "A"
            seq_label = f"{side_prefix}{idx + 1}(#{eid})"
            if tier == "C":
                tier_c_labels.append(seq_label)
                continue
            item = {
                "seq": seq_label,
                "session_time": str(sw.get("display_time") or "")[5:16],
                "tier": tier,
                "turn": _val_of(sw, dg, "shoulder_turn_change"),
                "sep": _val_of(sw, dg, "hip_shoulder_separation"),
                "ext": _val_of(sw, dg, "arm_extension"),
                "knee": _val_of(sw, dg, "preparation_knee_flexion"),
                "stance": _val_of(sw, dg, "stance"),
                "takeback": _val_of(sw, dg, "takeback_depth"),
                "weight": _val_of(sw, dg, "weight_transfer"),
                "drift": float(sw["balance_drift"]) if sw.get("balance_drift") is not None else None,
                "speed": _val_of(sw, dg, "racket_max_speed"),
            }
            valid_swings_clean.append(item)

        def _summarize_metric(key: str, label: str, unit: str) -> Optional[Dict[str, Any]]:
            vals = [x[key] for x in valid_swings_clean if x.get(key) is not None]
            if not vals:
                return None
            s_vals = sorted(vals)
            n_v = len(s_vals)
            med = float(_stats.median(s_vals))
            q1 = s_vals[int((n_v - 1) * 0.25)]
            q3 = s_vals[int((n_v - 1) * 0.75)]
            mid = n_v // 2
            early_v = [x[key] for x in valid_swings_clean[:mid] if x.get(key) is not None] if n_v >= 4 else []
            late_v = [x[key] for x in valid_swings_clean[mid:] if x.get(key) is not None] if n_v >= 4 else []
            early_med = round(float(_stats.median(early_v)), 1) if early_v else None
            late_med = round(float(_stats.median(late_v)), 1) if late_v else None
            return {
                "label": label,
                "unit": unit,
                "n": n_v,
                "median": round(med, 2),
                "iqr": round(q3 - q1, 2),
                "min": round(s_vals[0], 2),
                "max": round(s_vals[-1], 2),
                "early_half_median": early_med,
                "late_half_median": late_med,
                "half_delta": round(late_med - early_med, 2) if (early_med is not None and late_med is not None) else None,
            }

        fitted_metrics: Dict[str, Any] = {}
        for k_short, k_full, lbl, unit in (
            ("turn", "shoulder_turn_change", "转肩角度变化", "deg_2d"),
            ("sep", "hip_shoulder_separation", "肩髋分离角(2D像面投影)", "image_plane_deg"),
            ("ext", "arm_extension", "触球肘伸展角", "deg_2d"),
            ("knee", "preparation_knee_flexion", "准备期屈膝弯曲度(180-膝内角)", "deg_2d"),
            ("stance", "stance", "足部连线站位倾角", "deg_2d"),
            ("takeback", "takeback_depth", "镜面引拍深度比(双视角)", "ratio"),
            ("weight", "weight_transfer", "躯干重心横移", "body_width"),
            ("drift", "balance_drift", "回位期重心漂移", "body_width"),
            ("speed", "racket_max_speed", "拟合球拍峰值速度(相对值)", "px/s"),
        ):
            sm_m = _summarize_metric(k_short, lbl, unit)
            if sm_m is not None:
                fitted_metrics[k_full] = sm_m

        # Top-3 vs Bottom-3 clean swings by fitted speed (for concrete good-vs-weak attribution)
        ranked_by_spd = sorted(
            [x for x in valid_swings_clean if x.get("speed") is not None],
            key=lambda x: float(x["speed"]),
            reverse=True,
        )
        top_k = min(3, max(1, len(ranked_by_spd) // 4)) if len(ranked_by_spd) >= 4 else 0
        top_swings = ranked_by_spd[:top_k] if top_k else []
        weak_swings = ranked_by_spd[-top_k:] if top_k else []

        # Separate capture notices from biomechanical flaws
        capture_keywords = ("遮挡", "复核", "候选")
        capture_notices = [
            {"message": w.get("message"), "rate_pct": w.get("rate_pct"), "count": w.get("count")}
            for w in (side_data.get("coach_warnings") or [])
            if any(kw in str(w.get("message") or "") for kw in capture_keywords)
        ]
        n_valid = len(valid_swings_clean)
        bio_flaws = []
        if n_valid > 0:
            for lbl, pred in (
                ("转腰拧发条不足(肩髋未拉开差距，单靠手臂挥拍)", lambda x: x["sep"] is not None and x["sep"] < 10.0),
                ("侧身引拍转肩不充分(<30°，身体容易正对球网)", lambda x: x["turn"] is not None and x["turn"] < 30.0),
                ("击球前下蹲蓄力不足(<35°，站得太直)", lambda x: x["knee"] is not None and x["knee"] < 35.0),
                ("触球瞬间缩肘(<155°，手臂未充分迎前伸展)", lambda x: x["ext"] is not None and x["ext"] < 155.0),
                ("击球后身体重心回位不稳", lambda x: x["drift"] is not None and x["drift"] > 0.45),
                ("迎球重心向前传递不足", lambda x: x["weight"] is not None and x["weight"] < 0.40),
            ):
                hits = [x["seq"] for x in valid_swings_clean if pred(x)]
                if hits:
                    bio_flaws.append({
                        "issue": lbl,
                        "count": len(hits),
                        "rate_pct": round(len(hits) / n_valid * 100.0, 1),
                        "example_swings": hits[:5],
                    })
            bio_flaws.sort(key=lambda x: x["rate_pct"], reverse=True)

        # Compact chronological swing sample list (cap at 24 swings to keep prompt concise & sharp)
        compact_swings = [
            {
                "seq": x["seq"],
                "t": x["session_time"],
                "tier": x["tier"],
                "turn": round(x["turn"], 1) if x["turn"] is not None else None,
                "sep": round(x["sep"], 1) if x["sep"] is not None else None,
                "ext": round(x["ext"], 1) if x["ext"] is not None else None,
                "knee": round(x["knee"], 1) if x["knee"] is not None else None,
                "stance": round(x["stance"], 0) if x["stance"] is not None else None,
                "spd": round(x["speed"], 0) if x["speed"] is not None else None,
            }
            for x in valid_swings_clean[-24:]
        ]

        session_count = int(side_data.get("session_count") or 0)
        evidence_packet = {
            "player_name": (profile.get("player") or {}).get("display_name"),
            "dominant_hand": (profile.get("player") or {}).get("dominant_hand"),
            "stroke_side": side_label,
            "analysis_scope": "single_session_multi_swing" if session_count <= 1 else "multi_session_longitudinal",
            "window_days": days or "all",
            "session_count": session_count,
            "total_swings_raw": side_data.get("total_swings", 0),
            "valid_swings_fitted": n_valid,
            "tier_counts": traj_summary.get("tier_counts"),
            "excluded_tier_c_swings": tier_c_labels,
            "dual_view_swings": side_data.get("dual_view_swings", 0),
            "metric_semantics": {
                "shoulder_turn_change": "侧身引拍转肩幅度（≥40°表示侧身蓄力充分，<30°表示身体正对球网不侧身）",
                "hip_shoulder_separation": "引拍转体时身体拧紧发条程度（肩线与髋线夹角，越大说明核心弹性储能越强）",
                "preparation_knee_flexion": "击球前屈膝深蹲蓄力程度（180°-膝内角，数值越大代表下蹲踩实越深、像弹簧压下）",
                "arm_extension": "触球瞬间持拍手臂伸展程度（160°-175°为充分延展，<145°为严重缩肘）",
                "contact_lateral_distance": "击球点与身体的横向距离（≥0.55倍肩宽为身前舒适空间，过小代表球顶在身上缩手打）",
                "weight_transfer": "击球时身体向前迎球传递重心幅度（≥0.40倍肩宽表示身体主动迎前压上去打）",
                "balance_drift": "击球后身体回位漂移（<0.30倍肩宽为稳固复位，>0.45倍肩宽表示打完身体东倒西歪）",
                "takeback_depth": "双视角手腕拉后引拍深度比（≥1.8为充分拉开引拍，过小代表推球式短引拍）",
                "brush_angle": "击球时由下向上的刷球上升角（≥14°为充分制造上旋，过低容易平击出界）",
                "drop_depth_ratio": "触球前拍头下潜掉拍头深度比（≥0.25为拍头下沉充分，有由下向上的抽击杠杆）",
                "stance": "双脚连线站位倾角（0°-15°为开放式站位适合快速对攻，30°-60°为半开放/关闭式适合主动发力）",
                "leg_drive": "蹬地髋部纵向垂直抬升幅度（蹬地起伏发力程度）",
                "whip_amplification_ratio": "甜区与手腕鞭打加速比（≥2.2倍表示拍头如同鞭子末端猛烈甩出）",
                "kinematic_sequence": "动力链传导时序（标准顺序：下肢蹬地 -> 髋转 -> 肩转 -> 手臂 -> 拍头）",
                "racket_max_speed": "击球全行程峰值拍速（相对值，用于区分高质重炮球 vs 发软推球）",
                "racket_head_speed": "触球瞬间有效拍头速度",
                "swing_quality_score": "动作综合定型度评分（百分制，衡量动作一致性与发力完整性）",
                "iqr": "四分位距（衡量动作离散波动：数值过大代表动作时好时坏、极不稳定）",
                "half_delta": "后半程相对前半程中位数变化（正值代表越打越好，负值且拍数充足代表后程走样偷懒）",
            },
            "fitted_metrics": fitted_metrics,
            "biomechanical_flaws": bio_flaws,
            "top_swings_by_speed": [{"seq": x["seq"], "spd": round(x["speed"], 0), "turn": x["turn"], "knee": x["knee"], "ext": x["ext"], "weight": x["weight"]} for x in top_swings],
            "weak_swings_by_speed": [{"seq": x["seq"], "spd": round(x["speed"], 0), "turn": x["turn"], "knee": x["knee"], "ext": x["ext"], "weight": x["weight"]} for x in weak_swings],
            "per_swing_sequence": compact_swings,
            "capture_quality_notices": capture_notices,
        }

        from deepseek_realtime_coach import UrllibDeepSeekTransport
        import urllib.error

        transport = UrllibDeepSeekTransport()
        if session_count <= 1:
            if n_valid < 8:
                sample_confidence_tip = f"【样本量警示】：当前仅记录到 {n_valid} 拍有效击球，样本较少（属于摸底小样本）。切勿妄下'体能严重耗竭'等过度推论，重点只针对这几拍动作规范性与单拍发力手感进行点拨。"
            elif n_valid < 15:
                sample_confidence_tip = f"【样本量提示】：当前共有 {n_valid} 拍有效击球，具备初步统计参考。重点对比好球与弱球的发力差异，前后半程演变请用'后半段节奏波动/轻微起伏'客观描述，不要武断断言体能耗尽。"
            else:
                sample_confidence_tip = f"【样本量充足】：当前共有 {n_valid} 拍连贯击球，数据统计稳定可靠。请深入对比好坏球发力开关以及前后半程动作定型度演变。"
            scope_instruction = (
                f"当前为单场训练击球记录（共 {n_valid} 拍有效{side_label}）。{sample_confidence_tip}"
            )
        else:
            scope_instruction = (
                f"当前为跨 {session_count} 场训练的长期数据（共 {n_valid} 拍有效{side_label}），请重点对比【跨场次固化的发力共性】与【不同场次间反复出现的动作毛病】。"
            )
        system_prompt = (
            f"你是一位拥有15年一线执教经验的【职业网球主教练】。\n"
            f"你的核心使命：基于下方经过拟合清洗的学员真实击球数据，给学员写一份【句句听得懂、直击发力痛点、像在场边面对面指导】的《{side_label}主教练实战评语》（中文，约500-700字）。\n"
            f"{scope_instruction}\n\n"
            "【核心表达铁律 - 严禁出现算法与统计行话，必须翻译成通俗教练语言】：\n"
            "1. 绝对严禁向学员输出算法研发术语！文中绝对不能出现：'IQR'、'half_delta'、'px/s'、'Tier C'、'bw'、'PCHIP'、'SG平滑'、'2D投影' 等词汇！\n"
            "   - 'IQR 很大' 必须翻译为：'动作定型度不足，击球动作忽大忽小，发挥不够稳定'\n"
            "   - 'half_delta 为负'：若拍数充足且为转肩/深蹲/拍速等关键发力指标下滑，翻译为：'前半程体力充沛时很到位，后半程一累就开始走样偷懒'；若拍数较少，严禁断言体能耗尽，用'后半段击球节奏出现明显波动/下滑'客观表述。\n"
            "   - 'px/s 速度对比'：根据好球与弱球速度的实际落差客观描述（如'重炮好球 vs 偏软的推球，球速有明显落差'，切勿死板写死'差了一倍多'，尊重真实比例）。\n"
            "   - '屈膝度高/低' 必须翻译为：'击球前深蹲下压蓄力充分（像弹簧压下）' / '站得太直没下蹲'\n"
            "   - '转肩角度' 必须翻译为：'侧身引拍转肩充不充分'\n"
            "   - '肩髋分离' 必须翻译为：'身体像拧毛巾一样把发条拧紧'\n"
            "   - '触球肘伸展' 必须翻译为：'击球瞬间手臂充分迎前延展，没有缩着肘打'\n"
            "   - 'Tier C 剔除' 必须翻译为：'部分受拍摄角度遮挡的拍次已自动过滤，非学员技术问题'\n\n"
            "2. 严格按以下四个板块输出（口吻专业生动、有鼓励有鞭策、指导性极强）：\n"
            f"   【一、🎾 主教练一句话定性（学员画像）】\n"
            "   用大白话打个生动形象的比喻，一针见血概括学员这侧动作的根本特质（例如：“你的下盘蹲得很扎实，手也敢甩，但中间的‘转体发条’没拧紧！前半场很生猛，后半场一累就开始纯靠胳膊硬抡！”）。\n\n"
            f"   【二、🌟 你的标杆好球 vs ⚠️ 你的偷懒弱球（具体拍号对照）】\n"
            "   - 【标杆好球（记住这种感觉）】：具体点名打得最好的拍号（如正5、正15），告诉他当时身体感觉好在哪（深蹲踩实了、侧身转体拉满了、手臂甩得顺畅，打出了全场最爆的球速），让他强化肌肉记忆。\n"
            "   - 【偷懒弱球（抓出发力毛病）】：具体点名打得软、没威胁的拍号（如正1、正11，以及后半程偷懒的几拍），指出当时是怎么偷懒的（站直了没蹲、身体正对球网不侧身、单靠手去够球），并说明后半程体能下滑导致的动作变形。\n\n"
            f"   【三、🎯 场上即刻口诀与 1 个课后练习】\n"
            "   - 【球场实战口诀】：提炼一句学员在场上准备击球时能默念的 6~8 字口令（例如：“先侧身、深蹲下、转髋甩拍”）。\n"
            "   - 【课后专属练习】：给 1 项不拿球在家里或场边就能练的针对性动作（如“原地持拍拧毛巾转体练习”或“分腿垫步深蹲启动”），标明组数和练习感觉。\n\n"
            f"   【四、📊 好坏球关键动作对比（通俗对比表）】\n"
            "   用学员一目了然的简单列表，对比打得好的几拍与打得差的几拍的关键动作差异（如：好球时转肩侧身43°/下肢深蹲充分/打出全场最高速度；弱球时转肩仅17°/站得太直/球速下降近一半）。\n\n"
            "3. 严禁混入另一侧（反手/正手）数据，必须严格基于数据包中真实拍号（如 正5、正15、正1 等）。"
        )
        user_prompt = f"学员【{side_label}】拟合清洗后训练证据包（JSON）：\n" + json.dumps(
            evidence_packet, ensure_ascii=False, separators=(",", ":")
        )

        last_exc: Optional[Exception] = None
        used_model = configured_model
        raw_reply = ""
        fallback_model = candidate_models[-1]
        for candidate_model in candidate_models:
            try:
                m_lower = candidate_model.lower()
                is_openai_reasoning = m_lower.startswith(("o1", "o3", "o4", "gpt-5"))
                is_deepseek_reasoning = "reasoner" in m_lower
                is_reasoning = is_openai_reasoning or is_deepseek_reasoning

                system_role = "developer" if is_openai_reasoning else "system"
                req_payload: Dict[str, Any] = {
                    "model": candidate_model,
                    "messages": [
                        {"role": system_role, "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                if is_openai_reasoning:
                    req_payload["max_completion_tokens"] = 3000
                elif is_deepseek_reasoning:
                    req_payload["max_tokens"] = 3000
                else:
                    req_payload["temperature"] = 0.25
                    req_payload["max_tokens"] = 1600

                timeout_sec = 65.0 if is_reasoning else 35.0
                resp = transport.complete(
                    base_url=base_url,
                    api_key=api_key,
                    payload=req_payload,
                    timeout=timeout_sec,
                )
                choices = resp.get("choices") or []
                msg = (choices[0].get("message") or {}) if choices else {}
                raw_reply = str(msg.get("content") or "").strip()
                used_model = str(resp.get("model") or candidate_model)
                if raw_reply:
                    break
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code in (400, 404) and candidate_model != fallback_model:
                    continue
                raise RuntimeError(f"{provider_label} 请求失败 (HTTP {exc.code})") from exc
            except Exception as exc:
                last_exc = exc
                raise RuntimeError(f"{provider_label} 请求异常: {exc}") from exc

        if not raw_reply:
            raise RuntimeError(f"{provider_label} 返回为空: {last_exc or 'unknown'}")

        session_ids = list(dict.fromkeys(str(sw.get("session_id") or "") for sw in raw_swings if sw.get("session_id")))
        video_paths: List[str] = []
        for s in (profile.get("sessions_catalog") or []):
            if str(s.get("session_id") or "") in session_ids:
                vp = s.get("source_path") or s.get("source_name")
                if vp and str(vp) not in video_paths:
                    video_paths.append(str(vp))

        saved_rec = self._session_store.save_coach_review(
            player_id=(profile.get("player") or {}).get("player_id") or player_id or "default",
            stroke_side=requested_side,
            provider=provider_key,
            model=used_model,
            review_text=raw_reply,
            session_ids=session_ids,
            video_paths=video_paths,
            evidence_packet=evidence_packet,
            days_scope=str(days) if days is not None else "all",
            dedup=1 if dedup_val else 0,
        )

        return {
            "review_id": saved_rec.get("review_id"),
            "player": profile.get("player"),
            "stroke_side": requested_side,
            "days": days,
            "provider": provider_key,
            "model": used_model,
            "generated_at": saved_rec.get("created_at"),
            "review_text": raw_reply,
            "session_ids": session_ids,
            "video_paths": video_paths,
            "evidence_packet": evidence_packet,
        }

    def list_coach_reviews(
        self,
        player_id: Optional[str] = None,
        stroke_side: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        return self._session_store.list_coach_reviews(
            player_id=player_id or "default",
            stroke_side=stroke_side,
            limit=limit,
        )

    def evaluate_manual_review(
        self,
        payload: Dict[str, Any],
        report_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return process_manual_review(
            self.manual_review_paths(report_path),
            payload,
        )

    def auto_calibrate_manual_review(
        self,
        payload: Dict[str, Any],
        report_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        from manual_review_workflow import auto_calibrate_manual_review as _auto_calibrate
        return _auto_calibrate(
            self.manual_review_paths(report_path),
            payload,
            config_path=self.config_path,
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
        *,
        effective_pose_model: Optional[str] = None,
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
        if settings.pose_model != "configured":
            command.extend(['--pose-model', effective_pose_model or settings.pose_model])
        if settings.processing_mode != 'auto':
            command.extend(['--processing-mode', settings.processing_mode])
        if (
            settings.processing_mode == "offline"
            and settings.racket_pose_mode == "legacy"
            and settings.algo2_dual_view
            and settings.inference_schedule == "barrier"
        ):
            pose_name = str(effective_pose_model or settings.pose_model or "").lower()
            auto_det_workers = 2 if "yolo26n" in pose_name else 1
            command.extend(["--det-workers", str(auto_det_workers)])
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
