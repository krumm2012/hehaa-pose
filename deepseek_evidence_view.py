"""Compact a SwingEvidencePacket into the evidence sent to DeepSeek."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Optional

from coach_evidence_policy import build_coach_decision_policy
from swing_quality_policy import (
    is_ball_capture_advice,
)


SCHEMA_VERSION = "deepseek_swing_evidence_v1"
POSE_KEYS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
MOTION_KEYS = (
    "wrist_speed",
    "wrist_accel",
    "racket_speed",
    "racket_accel",
    "ball_speed",
    "ball_racket_distance",
    "contact_score",
    "two_hand_distance",
    "active_wrist_x_offset",
    "arm_extension_deg",
    "shoulder_turn_deg",
    "hip_shoulder_sep_deg",
)
def build_deepseek_evidence_view(packet: Dict) -> Dict:
    """Return a smaller, claim-gated view while preserving every event frame."""
    event = deepcopy(packet.get("event_summary") or {})
    event_id = int(packet.get("event_id", event.get("event_id", -1)))
    raw_frames = packet.get("event_frame_records") or []
    features_by_id = _rows_by_frame(
        packet.get("motion_features"),
        "frame_id",
    )
    trace_by_id = _rows_by_frame(
        packet.get("frame_trace"),
        "frame",
        fallback_key="frame_id",
    )
    frame_sequence = [
        _compact_frame(
            frame,
            features_by_id.get(int(frame.get("frame_id", -1)), {}),
            trace_by_id.get(int(frame.get("frame_id", -1)), {}),
        )
        for frame in sorted(
            raw_frames,
            key=lambda row: int(row.get("frame_id", -1)),
        )
    ]
    quality = (
        event.get("quality_flags")
        or (packet.get("coach_metrics") or {}).get("quality_flags")
        or {}
    )
    coach_metrics = _prune_none(deepcopy(packet.get("coach_metrics") or {}))
    decision_policy = build_coach_decision_policy(
        quality=quality,
        coach_metrics=coach_metrics,
        integrity=packet.get("integrity") or {},
    )
    _sanitize_model_evidence(event, coach_metrics, decision_policy)
    return _prune_none(
        {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "video_context": deepcopy(packet.get("video_context") or {}),
            "player_context": deepcopy(packet.get("player_context") or {}),
            "decision_policy": decision_policy,
            "event": event,
            "coach_metrics": coach_metrics,
            "local_advice": deepcopy(event.get("coach_advice") or {}),
            "recent_swings": deepcopy(packet.get("recent_swings") or []),
            "integrity": deepcopy(packet.get("integrity") or {}),
            "frame_sequence": frame_sequence,
        }
    )


def _compact_frame(frame: Dict, feature: Dict, trace: Dict) -> Dict:
    pose = frame.get("pose") or {}
    diagnostics = frame.get("detection_diagnostics") or {}
    racket = _best_racket(frame.get("rackets") or [])
    racket_center = feature.get("racket_center")
    if racket_center is None and racket is not None:
        box = racket.get("box") or []
        if len(box) >= 4:
            racket_center = [
                round((float(box[0]) + float(box[2])) / 2.0, 4),
                round((float(box[1]) + float(box[3])) / 2.0, 4),
            ]
    compact_racket = None
    if racket_center is not None or racket is not None:
        compact_racket = {
            "center": racket_center,
            "confidence": racket.get("confidence") if racket else None,
        }
    return _prune_none(
        {
            "frame_id": int(frame.get("frame_id", -1)),
            "timestamp": frame.get("timestamp"),
            "label": frame.get("swing_type") or feature.get("raw_swing_type"),
            "phase": trace.get("phase"),
            "motion_energy": trace.get("motion_energy"),
            "observations": {
                "ball": frame.get("ball"),
                "racket": compact_racket,
                "pose": {
                    key: pose.get(key)
                    for key in POSE_KEYS
                    if pose.get(key) is not None
                },
            },
            "motion": {
                key: feature.get(key)
                for key in MOTION_KEYS
                if feature.get(key) is not None
            },
            "metrics": deepcopy(frame.get("metrics") or {}),
            "detection": {
                "final_decision": diagnostics.get("final_decision"),
                "rejections": deepcopy(diagnostics.get("rejections") or {}),
                "continuity_enabled": diagnostics.get("continuity_enabled"),
                "continuity_disabled_reason": diagnostics.get(
                    "continuity_disabled_reason"
                ),
            },
        }
    )


def _best_racket(rackets: Iterable[Dict]) -> Optional[Dict]:
    candidates = [row for row in rackets if isinstance(row, dict)]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row.get("confidence") or 0.0))


def _rows_by_frame(
    rows,
    key: str,
    fallback_key: Optional[str] = None,
) -> Dict[int, Dict]:
    indexed = {}
    for row in rows or []:
        value = row.get(key)
        if value is None and fallback_key:
            value = row.get(fallback_key)
        if value is not None:
            indexed[int(value)] = row
    return indexed


def _sanitize_model_evidence(
    event: Dict,
    coach_metrics: Dict,
    decision_policy: Dict,
) -> None:
    """Remove stale conclusions that contradict the effective policy."""
    event.pop("deepseek_advice", None)
    effective_warnings = list(decision_policy.get("effective_warnings") or [])
    quality_views = [
        event.get("quality_flags"),
        coach_metrics.get("quality_flags"),
        (coach_metrics.get("data_quality") or {}).get("event_quality_flags"),
    ]
    for quality in quality_views:
        if isinstance(quality, dict) and "warnings" in quality:
            quality["warnings"] = effective_warnings

    advice = event.get("coach_advice")
    if (
        decision_policy.get("coaching_allowed")
        and isinstance(advice, dict)
        and advice.get("category") in {"capture", "review"}
    ):
        event.pop("coach_advice", None)

    tolerated = set(decision_policy.get("tolerated_conditions") or [])
    if "intermittent_ball_detection" not in tolerated:
        return
    diagnosis_tags = coach_metrics.get("diagnosis_tags")
    if isinstance(diagnosis_tags, list):
        coach_metrics["diagnosis_tags"] = [
            tag for tag in diagnosis_tags if tag != "ball_track_gaps"
        ]
    advice = event.get("coach_advice")
    if isinstance(advice, dict) and is_ball_capture_advice(advice):
        event.pop("coach_advice", None)


def _prune_none(value):
    if isinstance(value, dict):
        return {
            key: _prune_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_prune_none(item) for item in value]
    if isinstance(value, tuple):
        return [_prune_none(item) for item in value]
    return value
