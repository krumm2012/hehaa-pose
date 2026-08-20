"""Versioned data contracts shared by frame, Swing, and report pipelines.

The existing JSON fields remain valid.  Contract metadata is additive so older
consumers can continue to read ``frames`` and ``events`` without migration.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, MutableMapping, Optional
from urllib.parse import urlsplit, urlunsplit


FRAME_SCHEMA_VERSION = "tennis.frame.v1"
SWING_EVENT_SCHEMA_VERSION = "tennis.swing-event.v1"
FRAME_DOCUMENT_SCHEMA_VERSION = "tennis.frame-document.v1"
SWING_DOCUMENT_SCHEMA_VERSION = "tennis.swing-document.v1"
EVENT_LOG_SCHEMA_VERSION = "tennis.swing-event-log.v1"
SESSION_SCHEMA_VERSION = "tennis.analysis-session.v1"

_SESSION_ID_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def utc_iso_from_ns(value: int) -> str:
    """Return a stable UTC ISO-8601 representation for a Unix nanosecond value."""
    return datetime.fromtimestamp(int(value) / 1_000_000_000, timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def generate_session_id(prefix: str = "session") -> str:
    """Create a sortable, filesystem-safe session identifier."""
    safe_prefix = _SESSION_ID_PATTERN.sub("-", str(prefix or "session")).strip("-_")
    safe_prefix = safe_prefix or "session"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_prefix}_{timestamp}_{secrets.token_hex(3)}"


def normalize_session_id(value: Optional[str], prefix: str = "session") -> str:
    """Validate an explicit ID or generate one when it is absent."""
    if value is None or not str(value).strip():
        return generate_session_id(prefix)
    normalized = _SESSION_ID_PATTERN.sub("-", str(value).strip()).strip("-_")
    if not normalized:
        raise ValueError("session_id must contain at least one letter or number")
    return normalized[:96]


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a short deterministic fingerprint without exposing config secrets."""
    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _sanitize_source(source: str) -> str:
    value = str(source or "")
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        return value
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


def build_session_metadata(
    *,
    session_id: str,
    source: str,
    stream_id: str = "",
    stream_label: str = "",
    started_at_unix_ns: Optional[int] = None,
    config: Optional[Mapping[str, Any]] = None,
    models: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the immutable metadata attached to every output document."""
    started_ns = int(started_at_unix_ns or time.time_ns())
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "session_id": normalize_session_id(session_id),
        "started_at": utc_iso_from_ns(started_ns),
        "started_at_unix_ns": started_ns,
        "source": _sanitize_source(source),
        "stream_id": str(stream_id or ""),
        "stream_label": str(stream_label or ""),
        "config_hash": config_fingerprint(config or {}),
        "models": dict(models or {}),
    }


def stamp_frame_record(
    record: MutableMapping[str, Any],
    *,
    session: Mapping[str, Any],
    captured_at_unix_ns: int,
    inference_started_at_unix_ns: int,
    inference_completed_at_unix_ns: int,
    analysis_started_at_unix_ns: int,
    analysis_completed_at_unix_ns: int,
) -> MutableMapping[str, Any]:
    """Attach the v1 frame contract and end-to-end timing measurements in place."""
    frame_id = record.get("frame_id")
    if not isinstance(frame_id, int):
        raise ValueError("FrameRecord.frame_id must be an integer")
    if record.get("pose") is not None and not isinstance(record.get("pose"), dict):
        raise ValueError("FrameRecord.pose must be a mapping or null")

    capture_ns = int(captured_at_unix_ns)
    inference_start_ns = int(inference_started_at_unix_ns)
    inference_end_ns = int(inference_completed_at_unix_ns)
    analysis_start_ns = int(analysis_started_at_unix_ns)
    analysis_end_ns = int(analysis_completed_at_unix_ns)
    ordered = [
        capture_ns,
        inference_start_ns,
        inference_end_ns,
        analysis_start_ns,
        analysis_end_ns,
    ]
    if any(value <= 0 for value in ordered):
        raise ValueError("FrameRecord timing values must be positive Unix nanoseconds")

    record["schema_version"] = FRAME_SCHEMA_VERSION
    record["session_id"] = str(session["session_id"])
    record["timing"] = {
        "captured_at": utc_iso_from_ns(capture_ns),
        "captured_at_unix_ns": capture_ns,
        "inference_started_at_unix_ns": inference_start_ns,
        "inference_completed_at_unix_ns": inference_end_ns,
        "analysis_started_at_unix_ns": analysis_start_ns,
        "analysis_completed_at_unix_ns": analysis_end_ns,
        "inference_ms": round(max(0, inference_end_ns - inference_start_ns) / 1_000_000, 3),
        "analysis_ms": round(max(0, analysis_end_ns - analysis_start_ns) / 1_000_000, 3),
        "capture_to_analysis_ms": round(
            max(0, analysis_end_ns - capture_ns) / 1_000_000,
            3,
        ),
    }
    return record


def stamp_swing_event(
    event: MutableMapping[str, Any],
    *,
    session: Optional[Mapping[str, Any]] = None,
    emitted_at_unix_ns: Optional[int] = None,
    contact_frame_record: Optional[Mapping[str, Any]] = None,
) -> MutableMapping[str, Any]:
    """Attach the v1 Swing contract while preserving every legacy event field."""
    event["schema_version"] = SWING_EVENT_SCHEMA_VERSION
    if session and session.get("session_id"):
        event["session_id"] = str(session["session_id"])
    elif not event.get("session_id") and contact_frame_record:
        event["session_id"] = str(contact_frame_record.get("session_id") or "")

    if emitted_at_unix_ns is not None:
        emitted_ns = int(emitted_at_unix_ns)
        timing = dict(event.get("timing") or {})
        timing["event_emitted_at"] = utc_iso_from_ns(emitted_ns)
        timing["event_emitted_at_unix_ns"] = emitted_ns
        capture_ns = (
            ((contact_frame_record or {}).get("timing") or {}).get(
                "captured_at_unix_ns"
            )
        )
        if isinstance(capture_ns, int) and capture_ns > 0:
            timing["contact_capture_to_event_ms"] = round(
                max(0, emitted_ns - capture_ns) / 1_000_000,
                3,
            )
        event["timing"] = timing
    return event


def document_contract(
    document_type: str,
    session: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return additive metadata for a frame or Swing snapshot document."""
    schema = (
        FRAME_DOCUMENT_SCHEMA_VERSION
        if document_type == "frames"
        else SWING_DOCUMENT_SCHEMA_VERSION
    )
    return {
        "schema_version": schema,
        "document_type": str(document_type),
        "session": dict(session or {}),
    }


def inferred_session(frames: Any) -> Dict[str, Any]:
    """Recover the minimal session identity from versioned frame records."""
    if not isinstance(frames, list):
        return {}
    for frame in frames:
        if isinstance(frame, dict) and frame.get("session_id"):
            return {"session_id": str(frame["session_id"])}
    return {}
