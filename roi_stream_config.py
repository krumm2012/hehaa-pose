"""Resolve a camera-bound ROI without exposing stream credentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import yaml


Point = Tuple[int, int]
FrameSize = Tuple[int, int]


def sanitize_stream_source(source: str) -> str:
    """Return a stable stream identifier with credentials and query removed."""
    source = str(source or "").strip()
    parsed = urlsplit(source)
    if not parsed.scheme or not parsed.hostname:
        return Path(source).name if source else "unknown"

    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "", "", ""))


@dataclass(frozen=True)
class ROIStreamProfile:
    enabled: bool
    matched: bool
    stream_id: str
    label: str
    source: str
    points: Tuple[Point, ...]
    configured_frame_size: FrameSize
    frame_size: FrameSize
    config_path: str
    reason: str = ""
    mirror_view: Dict[str, Any] = field(default_factory=dict)
    front_view: Dict[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "matched": self.matched,
            "stream_id": self.stream_id,
            "label": self.label,
            "source": self.source,
            "points": [list(point) for point in self.points],
            "configured_frame_size": list(self.configured_frame_size),
            "frame_size": list(self.frame_size),
            "config_path": self.config_path,
            "reason": self.reason,
            "mirror_view": dict(self.mirror_view) if self.mirror_view else {},
            "front_view": dict(self.front_view) if self.front_view else {},
        }

    @property
    def has_mirror_view(self) -> bool:
        return bool(self.mirror_view and self.mirror_view.get("enabled", True))

    @property
    def mirror_polygon(self) -> list:
        return list(self.mirror_view.get("polygon", []))

    @property
    def mirror_mask_polygon(self) -> list:
        return list(self.mirror_view.get("mask_polygon", []))

    @property
    def mirror_reflection_roi(self) -> list:
        return list(self.mirror_view.get("reflection_roi", []))

    @property
    def has_front_view(self) -> bool:
        return bool(self.front_view and self.front_view.get("enabled", True))

    @property
    def front_default_roi(self) -> list:
        return list(self.front_view.get("default_roi", []))


def _disabled_profile(
    source: str,
    frame_size: FrameSize,
    config_path: str = "",
    reason: str = "",
    stream_id: str = "",
    label: str = "Unmatched stream",
    mirror_view: Optional[Dict[str, Any]] = None,
    front_view: Optional[Dict[str, Any]] = None,
) -> ROIStreamProfile:
    return ROIStreamProfile(
        enabled=False,
        matched=False,
        stream_id=stream_id,
        label=label,
        source=sanitize_stream_source(source),
        points=(),
        configured_frame_size=frame_size,
        frame_size=frame_size,
        config_path=config_path,
        reason=reason,
        mirror_view=mirror_view or {},
        front_view=front_view or {},
    )


def _candidate_profiles(document: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    streams = document.get("streams")
    if isinstance(streams, list):
        yield from (item for item in streams if isinstance(item, dict))
        return
    if isinstance(streams, dict):
        for stream_id, item in streams.items():
            if isinstance(item, dict):
                yield {"stream_id": stream_id, **item}
        return
    yield document


def _profile_source(candidate: Dict[str, Any]) -> str:
    return str(
        candidate.get("source")
        or candidate.get("stream_source")
        or candidate.get("rtsp_url")
        or ""
    )


def _parse_frame_size(value: Any, fallback: FrameSize) -> FrameSize:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        width, height = int(value[0]), int(value[1])
        if width > 0 and height > 0:
            return width, height
    return fallback


def _scaled_points(
    points: Any,
    configured_size: FrameSize,
    frame_size: FrameSize,
) -> Tuple[Point, ...]:
    if not isinstance(points, list) or len(points) != 4:
        return ()
    source_width, source_height = configured_size
    target_width, target_height = frame_size
    scale_x = target_width / max(1, source_width)
    scale_y = target_height / max(1, source_height)
    scaled = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return ()
        x = max(0, min(target_width - 1, int(round(float(point[0]) * scale_x))))
        y = max(0, min(target_height - 1, int(round(float(point[1]) * scale_y))))
        scaled.append((x, y))
    return tuple(scaled)


def resolve_roi_stream_profile(
    config: Dict[str, Any],
    source: str,
    frame_size: FrameSize,
    target_stream_id: Optional[str] = None,
) -> ROIStreamProfile:
    """Load and match the configured ROI to the current input stream."""
    roi_settings = config.get("roi_settings") or {}
    config_path = str(roi_settings.get("roi_config_path") or "")
    if target_stream_id is None:
        target_stream_id = (
            str(
                roi_settings.get("target_stream_id")
                or config.get("target_stream_id")
                or config.get("stream_id")
                or ""
            ).strip()
            or None
        )

    if not roi_settings.get("enabled", False):
        return _disabled_profile(source, frame_size, config_path, "ROI disabled")
    if not roi_settings.get("auto_load_config", True):
        return _disabled_profile(source, frame_size, config_path, "ROI auto-load disabled")
    if not config_path:
        return _disabled_profile(source, frame_size, "", "ROI config path missing")

    path = Path(config_path).expanduser()
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return _disabled_profile(source, frame_size, str(path), f"ROI config load failed: {exc}")
    if not isinstance(document, dict):
        return _disabled_profile(source, frame_size, str(path), "ROI config must be a mapping")

    sanitized_source = sanitize_stream_source(source)
    candidates = list(_candidate_profiles(document))
    selected: Optional[Dict[str, Any]] = None

    # 1. 显式指定的 target_stream_id 优先匹配
    if target_stream_id:
        for candidate in candidates:
            cand_id = str(candidate.get("stream_id") or candidate.get("id") or "").strip()
            if cand_id and cand_id == target_stream_id:
                selected = candidate
                break

    # 2. 按视频流来源地址匹配
    if selected is None:
        for candidate in candidates:
            configured_source = _profile_source(candidate)
            if configured_source and sanitize_stream_source(configured_source) == sanitized_source:
                selected = candidate
                break

    # 3. 回退到默认码流配置
    if selected is None:
        selected = next(
            (
                candidate
                for candidate in candidates
                if not _profile_source(candidate) or candidate.get("default", False)
            ),
            None,
        )
    if selected is None:
        return _disabled_profile(
            source,
            frame_size,
            str(path),
            "No ROI profile matches this stream",
        )

    mirror_view = dict(selected.get("mirror_view") or {})
    front_view = dict(selected.get("front_view") or {})
    enabled = bool(selected.get("enabled", selected.get("roi_enabled", False)))
    configured_size = _parse_frame_size(
        selected.get("frame_size") or selected.get("resolution"),
        frame_size,
    )
    points = _scaled_points(
        selected.get("roi_points") or selected.get("points"),
        configured_size,
        frame_size,
    )
    stream_id = str(selected.get("stream_id") or selected.get("id") or "default")
    label = str(
        selected.get("stream_label")
        or selected.get("label")
        or selected.get("name")
        or stream_id
    )

    if not enabled or not points:
        return _disabled_profile(
            source,
            frame_size,
            str(path),
            "Matched ROI profile is disabled or invalid",
            stream_id=stream_id,
            label=label,
            mirror_view=mirror_view,
            front_view=front_view,
        )

    return ROIStreamProfile(
        enabled=True,
        matched=True,
        stream_id=stream_id,
        label=label,
        source=sanitized_source,
        points=points,
        configured_frame_size=configured_size,
        frame_size=frame_size,
        config_path=str(path),
        mirror_view=mirror_view,
        front_view=front_view,
    )
