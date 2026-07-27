"""Resolve a camera-bound ROI without exposing stream credentials."""

from __future__ import annotations

from dataclasses import dataclass
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
        }


def _disabled_profile(
    source: str,
    frame_size: FrameSize,
    config_path: str = "",
    reason: str = "",
) -> ROIStreamProfile:
    return ROIStreamProfile(
        enabled=False,
        matched=False,
        stream_id="",
        label="Unmatched stream",
        source=sanitize_stream_source(source),
        points=(),
        configured_frame_size=frame_size,
        frame_size=frame_size,
        config_path=config_path,
        reason=reason,
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
) -> ROIStreamProfile:
    """Load and match the configured ROI to the current input stream."""
    roi_settings = config.get("roi_settings") or {}
    config_path = str(roi_settings.get("roi_config_path") or "")
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
    for candidate in candidates:
        configured_source = _profile_source(candidate)
        if configured_source and sanitize_stream_source(configured_source) == sanitized_source:
            selected = candidate
            break
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
    if not enabled or not points:
        return _disabled_profile(
            source,
            frame_size,
            str(path),
            "Matched ROI profile is disabled or invalid",
        )

    stream_id = str(selected.get("stream_id") or selected.get("id") or "default")
    label = str(
        selected.get("stream_label")
        or selected.get("label")
        or selected.get("name")
        or stream_id
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
    )
