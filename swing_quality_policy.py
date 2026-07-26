"""Shared tolerance rules for Swing evidence quality and Coach gating."""

from __future__ import annotations

from typing import Dict, Set


BALL_MIN_EVENT_RATIO = 0.20
BALL_CONTACT_WINDOW_RADIUS = 4
BALL_MIN_CONTACT_DETECTIONS = 2


def ball_tracking_requires_capture(quality: Dict) -> bool:
    """Return whether missing Active Ball evidence is severe enough to act on."""
    event_ratio = quality.get("ball_frame_ratio")
    contact_detections = quality.get("ball_contact_window_detection_frames")
    contact_ratio = quality.get("ball_contact_window_ratio")
    if event_ratio is None and contact_detections is None and contact_ratio is None:
        return True
    if event_ratio is not None and float(event_ratio) < BALL_MIN_EVENT_RATIO:
        return True
    if contact_detections is not None:
        return int(contact_detections) < BALL_MIN_CONTACT_DETECTIONS
    if contact_ratio is not None:
        minimum_ratio = BALL_MIN_CONTACT_DETECTIONS / (
            BALL_CONTACT_WINDOW_RADIUS * 2 + 1
        )
        return float(contact_ratio) < minimum_ratio
    return False


def effective_quality_warnings(quality: Dict) -> Set[str]:
    """Return actionable warnings after applying tolerant tracking policy."""
    warnings = set(quality.get("warnings") or [])
    if (
        "ball_track_gaps" in warnings
        and not ball_tracking_requires_capture(quality)
    ):
        warnings.remove("ball_track_gaps")
    return warnings


def is_ball_capture_advice(advice: Dict) -> bool:
    """Return whether advice asks to keep the incoming ball in frame."""
    focus = " ".join(
        str(advice.get(key) or "").lower()
        for key in ("code", "focus")
    )
    if any(
        token in focus
        for token in ("ball_track_gaps", "ball_capture", "ball_in_frame")
    ):
        return True
    message = str(advice.get("message") or "")
    return (
        "球" in message
        and any(token in message for token in ("入镜", "画面", "镜头"))
        and any(token in message for token in ("完整", "确保", "保持"))
    )
