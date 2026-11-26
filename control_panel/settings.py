"""Settings, stream profile loading, and credential helpers for the local control panel."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote, urlsplit, urlunsplit

import yaml

from roi_stream_config import sanitize_stream_source


SESSION_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
MAX_REQUEST_BYTES = 64 * 1024
CUSTOM_STREAM_ID = "custom"
LOCAL_VIDEO_ID = "local_video"
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
CUSTOM_STREAM_SCHEMES = {"http", "https", "rtmp", "rtsp", "tcp", "udp"}
ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _session_capture_time(path: Path) -> float:
    """Report edits must not promote an old capture when the server restarts."""
    match = re.search(r"_(\d{8}T\d{6}Z)_", path.name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    return path.stat().st_mtime


def load_local_environment_variable(path: Path, name: str) -> bool:
    """Load one named value from a dotenv file without overriding the process."""
    variable_name = str(name or "").strip()
    if not ENVIRONMENT_NAME_PATTERN.fullmatch(variable_name):
        raise ValueError("环境变量名无效")
    if os.environ.get(variable_name):
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(f"无法读取本地环境文件: {path}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        candidate = raw_line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        if candidate.startswith("export "):
            candidate = candidate[7:].lstrip()
        raw_key, raw_separator, _raw_value = candidate.partition("=")
        if not raw_separator or raw_key.strip() != variable_name:
            continue
        try:
            tokens = shlex.split(candidate, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(
                f"{path.name} 第 {line_number} 行格式无效"
            ) from exc
        if len(tokens) != 1:
            continue
        _key, _separator, value = tokens[0].partition("=")
        if not value:
            return False
        os.environ[variable_name] = value
        return True
    return False


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


def sort_points_tl_tr_br_bl(points: List[List[Any]]) -> List[List[Any]]:
    """Sort 4 points canonically: TL, TR, BR, BL (clockwise starting from top-left)."""
    if len(points) != 4:
        return points
    by_y = sorted(points, key=lambda p: float(p[1]))
    top_two = sorted(by_y[:2], key=lambda p: float(p[0]))
    bottom_two = sorted(by_y[2:], key=lambda p: float(p[0]), reverse=True)
    return [top_two[0], top_two[1], bottom_two[0], bottom_two[1]]


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


def validate_custom_stream_source(source: str) -> str:
    """Validate an ephemeral stream URL while keeping its connection options."""
    value = str(source or "").strip()
    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    if scheme not in CUSTOM_STREAM_SCHEMES or not parsed.hostname:
        allowed = ", ".join(sorted(CUSTOM_STREAM_SCHEMES))
        raise ValueError(f"自定义码流必须是有效地址，支持: {allowed}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("请通过用户名和密码输入框提供码流凭据")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("自定义码流端口无效") from exc
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit(
        (scheme, netloc, parsed.path, parsed.query, parsed.fragment)
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
    evidence_bundle: bool
    realtime_coach: bool
    max_suggestions: int
    min_confidence: float
    analysis_interval: int
    settle_frames: int
    deepseek_coach: bool
    coach_tts: bool
    coach_tts_playback: bool
    hdmi_output: bool
    display_origin_x: int
    display_origin_y: int
    algo2_dual_view: bool = False
    mapped_stream_id: str = ""
    racket_pose_mode: str = "legacy"
    processing_mode: str = "buffered"
    inference_schedule: str = "barrier"
    racket_ipc: str = "queue"
    racket_model_profile: str = "configured"
    pose_model: str = "configured"
    analysis_mode: str = "standard"

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
        coach_tts = _bool(payload, "coach_tts", False)
        realtime_swing_events = (
            _bool(payload, "realtime_swing_events", True)
            or realtime_coach
            or deepseek_coach
            or coach_tts
        )
        stream_id = str(payload.get("stream_id") or "").strip()
        mapped_stream_id = (
            str(payload.get("mapped_stream_id") or "").strip()
            if stream_id == LOCAL_VIDEO_ID
            else ""
        )
        roi_enabled = _bool(payload, "roi_enabled", True)
        if stream_id == LOCAL_VIDEO_ID and not mapped_stream_id:
            roi_enabled = False
        raw_analysis_mode = str(payload.get("analysis_mode") or "").strip()
        if raw_analysis_mode:
            if raw_analysis_mode not in ("standard", "lite"):
                raise ValueError("无效的分析模式")
            analysis_mode = raw_analysis_mode
            explicit_proc = str(payload.get("processing_mode") or "").strip()
            processing_mode = explicit_proc if explicit_proc else (
                "offline" if stream_id == LOCAL_VIDEO_ID else "buffered"
            )
            inference_schedule = "barrier"
            if analysis_mode == "standard":
                racket_pose_mode = "shadow"
                racket_model_profile = "640_shadow"
                racket_ipc = "shared_memory"
            else:
                racket_pose_mode = "legacy"
                racket_model_profile = "configured"
                racket_ipc = "queue"
        else:
            racket_pose_mode = str(payload.get('racket_pose_mode') or 'legacy')
            racket_model_profile = str(payload.get('racket_model_profile') or 'configured')
            processing_mode = str(payload.get('processing_mode') or 'buffered')
            inference_schedule = str(payload.get('inference_schedule') or 'barrier')
            racket_ipc = str(payload.get('racket_ipc') or 'queue')
            analysis_mode = (
                "standard"
                if (racket_model_profile == "640_shadow" and racket_pose_mode == "shadow")
                else "lite"
            )

        if racket_model_profile not in ('configured', '640_shadow'):
            raise ValueError('无效的球拍模型选择')
        if racket_model_profile == '640_shadow' and racket_pose_mode != 'shadow':
            raise ValueError('640候选尚未通过检测质量验收，请选择影子性能预览')
        if racket_model_profile == '640_shadow' and processing_mode not in ('buffered', 'offline'):
            raise ValueError('640性能预览需逐帧连续处理或完整分析')
        if racket_pose_mode not in ('legacy', 'exclusive', 'shadow', 'active'):
            raise ValueError('无效的球拍模式')
        if processing_mode not in ('auto', 'realtime', 'buffered', 'offline'):
            raise ValueError('无效的处理模式')
        if racket_pose_mode == 'exclusive' and processing_mode == 'realtime':
            raise ValueError('新版七点球拍方案要求连续处理或完整分析，不支持丢帧预览')
        if inference_schedule not in ('barrier', 'interframe') or racket_ipc not in ('queue', 'shared_memory'):
            raise ValueError('无效的推理调度或图像传输方式')
        if inference_schedule == 'interframe' and (racket_pose_mode != 'exclusive' or processing_mode == 'realtime'):
            raise ValueError('跨帧流水线仅支持新版独占连续处理')
        if racket_ipc == 'shared_memory' and racket_pose_mode != 'legacy' and processing_mode == 'realtime':
            raise ValueError('共享源帧仅支持连续处理或完整分析')
        if processing_mode == 'offline' and stream_id != LOCAL_VIDEO_ID:
            raise ValueError('完整分析模式仅支持本地视频')
        pose_model = str(payload.get('pose_model') or 'configured').strip()
        if pose_model not in ('configured', 'yolo26n-pose.mlpackage', 'yolo26m-pose.mlpackage', 'models/yolo26m-pose.mlpackage'):
            raise ValueError('无效的姿态模型选择')

        return cls(
            stream_id=stream_id,
            mapped_stream_id=mapped_stream_id,
            analysis_mode=analysis_mode,
            racket_pose_mode=racket_pose_mode,
            racket_model_profile=racket_model_profile,
            pose_model=pose_model,
            processing_mode=processing_mode,
            inference_schedule=inference_schedule,
            racket_ipc=racket_ipc,
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
            roi_enabled=roi_enabled,
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
            live_mode=payload.get("stream_id") != LOCAL_VIDEO_ID and _bool(payload, "live_mode", True),
            save_video=_bool(payload, "save_video", False),
            realtime_swing_events=realtime_swing_events,
            realtime_frame_output=_bool(
                payload,
                "realtime_frame_output",
                False,
            ),
            evidence_bundle=_bool(payload, "evidence_bundle", True),
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
            coach_tts=coach_tts,
            coach_tts_playback=_bool(payload, "coach_tts_playback", True),
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
            algo2_dual_view=_bool(payload, "algo2_dual_view", False),
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
                "points": sort_points_tl_tr_br_bl([
                    [int(point[0]), int(point[1])]
                    for point in points
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                ]) if len(points) == 4 else [
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
                "mirror_view": {
                    **dict(profile.get("mirror_view") or {}),
                    **({"polygon": sort_points_tl_tr_br_bl(profile["mirror_view"]["polygon"])}
                       if profile.get("mirror_view") and isinstance(profile["mirror_view"].get("polygon"), list) and len(profile["mirror_view"]["polygon"]) == 4
                       else {})
                },
            }
        )
    if not result:
        raise ValueError(f"ROI 配置中没有可用码流: {path}")
    return result
