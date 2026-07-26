#!/usr/bin/env python3
"""Temporal swing event segmentation using pose, ball, and racket features."""

from __future__ import annotations

from statistics import median
from typing import Dict, List, Optional, Tuple

from swing_event_classifier import classify_swing_event
from swing_quality_policy import (
    BALL_CONTACT_WINDOW_RADIUS,
    ball_tracking_requires_capture,
)


def _smooth(values: List[float], window: int = 3) -> List[float]:
    if not values:
        return []
    w = max(1, int(window))
    if w <= 1:
        return list(values)
    out = []
    for idx in range(len(values)):
        lo = max(0, idx - w + 1)
        chunk = sorted(values[lo : idx + 1])
        out.append(chunk[len(chunk) // 2])
    return out


def _motion_energy(feature: Dict) -> float:
    wrist = float(feature.get("wrist_speed") or 0.0)
    racket = float(feature.get("racket_speed") or 0.0)
    accel = max(0.0, float(feature.get("wrist_accel") or 0.0))
    contact = float(feature.get("contact_score") or 0.0) * 10.0
    return max(wrist, racket * 0.75, accel * 0.6, contact)


def _phase_for(feature: Dict, energy: float, peak_energy: float) -> str:
    if energy >= peak_energy * 0.78:
        return "contact_candidate" if float(feature.get("contact_score") or 0.0) >= 0.35 else "forward_swing"
    if energy >= peak_energy * 0.45:
        return "backswing"
    if energy >= peak_energy * 0.25:
        return "follow_through"
    return "ready"


def _estimate_fps(features: List[Dict]) -> Optional[float]:
    timestamps = [float(f["timestamp"]) for f in features if f.get("timestamp") is not None]
    if len(timestamps) < 8:
        return None
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    if not deltas:
        return None
    frame_delta = median(deltas)
    if frame_delta <= 0:
        return None
    return 1.0 / frame_delta


def _robust_peak_floor(energy: List[float], min_peak_energy: float) -> float:
    center = median(energy)
    deviations = [abs(v - center) for v in energy]
    mad = median(deviations) if deviations else 0.0
    return max(float(min_peak_energy), float(center + mad * 3.0))


def _peak_quality(feature: Dict, energy_value: float) -> float:
    quality = float(energy_value)
    if feature.get("ball") is not None:
        quality += 30.0
    if feature.get("racket_center") is not None:
        quality += 20.0
    if feature.get("raw_swing_type") in {"Backhand", "Two-Handed Backhand"}:
        quality += 45.0
    return quality


def _local_peak_candidates(features: List[Dict], energy: List[float], peak_floor: float, edge_margin: int) -> List[Tuple[int, float]]:
    candidates = []
    for idx in range(1, len(energy) - 1):
        if idx < edge_margin or idx >= len(energy) - edge_margin:
            continue
        if energy[idx] >= peak_floor and energy[idx] >= energy[idx - 1] and energy[idx] >= energy[idx + 1]:
            candidates.append((idx, _peak_quality(features[idx], energy[idx])))
    return candidates


def _select_peak_indices(candidates: List[Tuple[int, float]], peak_min_distance: int) -> List[int]:
    if not candidates:
        return []

    clusters = []
    current = [candidates[0]]
    for candidate in candidates[1:]:
        if candidate[0] - current[-1][0] <= peak_min_distance:
            current.append(candidate)
        else:
            clusters.append(current)
            current = [candidate]
    clusters.append(current)

    selected = []
    for cluster in clusters:
        selected.append(max(cluster, key=lambda item: item[1])[0])
    return selected


def _peak_event_ranges(peak_indices: List[int], feature_count: int, fps: float) -> List[Tuple[int, int, int]]:
    pre_frames = max(8, int(round(fps * 0.85)))
    post_frames = max(12, int(round(fps * 1.25)))
    ranges = []
    for pos, peak_idx in enumerate(peak_indices):
        prev_boundary = 0
        next_boundary = feature_count - 1
        if pos > 0:
            prev_boundary = (peak_indices[pos - 1] + peak_idx) // 2 + 1
        if pos + 1 < len(peak_indices):
            next_boundary = (peak_idx + peak_indices[pos + 1]) // 2
        start_idx = max(prev_boundary, peak_idx - pre_frames)
        end_idx = min(next_boundary, peak_idx + post_frames)
        ranges.append((start_idx, end_idx, peak_idx))
    return ranges


def _phase_counts(features: List[Dict], energy: List[float], start_idx: int, end_idx: int, peak_energy: float) -> Tuple[Dict[str, int], Dict[int, str]]:
    phases = {}
    frame_phases = {}
    for idx in range(start_idx, end_idx + 1):
        phase = _phase_for(features[idx], energy[idx], peak_energy)
        phases[phase] = phases.get(phase, 0) + 1
        frame_phases[features[idx]["frame_id"]] = phase
    return phases, frame_phases


def _best_contact_frame(features: List[Dict], start_idx: int, end_idx: int, peak_idx: int) -> int:
    event_features = features[start_idx : end_idx + 1]
    scored = [f for f in event_features if f.get("contact_score") is not None]
    if not scored:
        return int(features[peak_idx]["frame_id"])
    peak_frame = int(features[peak_idx]["frame_id"])
    best = max(
        scored,
        key=lambda f: (
            float(f.get("contact_score") or 0.0),
            -abs(int(f.get("frame_id", peak_frame)) - peak_frame),
        ),
    )
    if float(best.get("contact_score") or 0.0) <= 0.0:
        return int(features[peak_idx]["frame_id"])
    return int(best["frame_id"])


def _event_quality_flags(features: List[Dict], start_idx: int, end_idx: int, classification: Dict, contact_frame: int) -> Dict:
    event_features = features[start_idx : end_idx + 1]
    total = max(1, len(event_features))
    pose_frames = sum(1 for f in event_features if f.get("has_pose"))
    ball_frames = sum(1 for f in event_features if f.get("ball") is not None)
    racket_frames = sum(1 for f in event_features if f.get("racket_center") is not None)
    diagnostic_counts = {}
    continuity_disabled = 0
    for feature in event_features:
        diagnostics = feature.get("detection_diagnostics") or {}
        for key, value in (diagnostics.get("rejections") or {}).items():
            diagnostic_counts[key] = diagnostic_counts.get(key, 0) + int(value or 0)
        if diagnostics.get("continuity_disabled"):
            continuity_disabled += 1

    evidence = classification.get("evidence") or {}
    warnings = []
    ball_ratio = ball_frames / total
    racket_ratio = racket_frames / total
    pose_ratio = pose_frames / total
    contact_window = [
        feature
        for feature in event_features
        if abs(int(feature["frame_id"]) - int(contact_frame))
        <= BALL_CONTACT_WINDOW_RADIUS
    ]
    contact_window_ball_frames = sum(
        1 for feature in contact_window if feature.get("ball") is not None
    )
    contact_window_ratio = contact_window_ball_frames / max(
        1,
        len(contact_window),
    )
    ball_quality = {
        "ball_frame_ratio": ball_ratio,
        "ball_contact_window_ratio": contact_window_ratio,
        "ball_contact_window_frames": len(contact_window),
        "ball_contact_window_detection_frames": contact_window_ball_frames,
    }
    if ball_tracking_requires_capture(ball_quality):
        warnings.append("ball_track_gaps")
    if racket_ratio < 0.65:
        warnings.append("racket_track_gaps")
    if pose_ratio < 0.90:
        warnings.append("pose_gaps")
    if float(evidence.get("screen_left_true_right_ratio") or 0.0) >= 0.58:
        warnings.append("mirror_handedness_rule_applied")
    if diagnostic_counts.get("static_hard_mask", 0) > 0:
        warnings.append("static_ball_mask_in_event")
    if diagnostic_counts.get("upper_mirror_unsupported", 0) > 0:
        warnings.append("mirror_ball_rejection_in_event")
    if continuity_disabled > 0:
        warnings.append("ball_continuity_disabled")
    if contact_frame == int(features[start_idx + max(0, min(end_idx - start_idx, (end_idx - start_idx) // 2))]["frame_id"]):
        warnings.append("contact_frame_needs_review")

    return {
        "pose_frame_ratio": round(pose_ratio, 4),
        "ball_frame_ratio": round(ball_ratio, 4),
        "ball_contact_window_ratio": round(contact_window_ratio, 4),
        "ball_contact_window_frames": len(contact_window),
        "ball_contact_window_detection_frames": contact_window_ball_frames,
        "racket_frame_ratio": round(racket_ratio, 4),
        "diagnostic_rejection_counts": dict(sorted(diagnostic_counts.items())),
        "continuity_disabled_frames": int(continuity_disabled),
        "warnings": sorted(set(warnings)),
        "review_recommended": bool(warnings),
    }


def _screen_left_true_right_forehand_evidence(event_features: List[Dict]) -> Dict:
    offsets = [float(f["active_wrist_x_offset"]) for f in event_features if f.get("active_wrist_x_offset") is not None]
    if not offsets:
        return {"screen_left_true_right_ratio": 0.0, "screen_left_true_right_frames": 0, "screen_right_frames": 0}
    left_frames = sum(1 for offset in offsets if offset < 0)
    right_frames = sum(1 for offset in offsets if offset > 0)
    return {
        "screen_left_true_right_ratio": round(left_frames / max(1, len(offsets)), 4),
        "screen_left_true_right_frames": int(left_frames),
        "screen_right_frames": int(right_frames),
    }


def _classify_peak_event(features: List[Dict], start_idx: int, end_idx: int, peak_idx: int, fps: float) -> Dict:
    core_radius = max(8, int(round(fps * 0.48)))
    core_start = max(start_idx, peak_idx - core_radius)
    core_end = min(end_idx, peak_idx + core_radius)
    classification = classify_swing_event(features[core_start : core_end + 1])

    # Preserve two-handed evidence in the post-impact/follow-through window,
    # while avoiding ready-position close hands from dominating the decision.
    post_classification = classify_swing_event(features[peak_idx : end_idx + 1], two_hand_min_ratio=0.28)
    if (
        classification["stroke_type"] == "Backhand"
        and post_classification["stroke_type"] == "Two-Handed Backhand"
        and post_classification["evidence"].get("two_hand_ratio", 0.0) >= 0.28
    ):
        classification = post_classification

    handedness_evidence = _screen_left_true_right_forehand_evidence(features[start_idx : end_idx + 1])
    if handedness_evidence["screen_left_true_right_ratio"] >= 0.58:
        classification = {
            "stroke_type": "Forehand",
            "confidence": max(float(classification.get("confidence", 0.0)), handedness_evidence["screen_left_true_right_ratio"]),
            "evidence": {
                **classification.get("evidence", {}),
                **handedness_evidence,
                "handedness_rule": "screen_left_is_true_right_forehand",
            },
        }
    else:
        classification["evidence"].update(handedness_evidence)
    classification["evidence"]["core_start_frame"] = int(features[core_start]["frame_id"])
    classification["evidence"]["core_end_frame"] = int(features[core_end]["frame_id"])
    return classification


def _segment_by_peaks(
    features: List[Dict],
    energy: List[float],
    min_peak_energy: float,
    min_event_frames: int,
    min_event_gap: int,
    fps: float,
) -> Optional[Dict]:
    if len(features) < 60:
        return None

    peak_floor = _robust_peak_floor(energy, min_peak_energy)
    peak_min_distance = max(min_event_gap, int(round(fps * 1.6)))
    edge_margin = max(min_event_frames, int(round(fps * 0.35)))
    candidates = _local_peak_candidates(features, energy, peak_floor, edge_margin)
    peak_indices = _select_peak_indices(candidates, peak_min_distance)
    if not peak_indices:
        return None

    events = []
    frame_to_event = {}
    frame_phases = {}
    for start_idx, end_idx, peak_idx in _peak_event_ranges(peak_indices, len(features), fps):
        if (end_idx - start_idx + 1) < min_event_frames:
            continue
        peak_energy = max(energy[start_idx : end_idx + 1])
        if peak_energy < min_peak_energy:
            continue
        classification = _classify_peak_event(features, start_idx, end_idx, peak_idx, fps)
        event_id = len(events) + 1
        contact_frame = _best_contact_frame(features, start_idx, end_idx, peak_idx)
        quality_flags = _event_quality_flags(features, start_idx, end_idx, classification, contact_frame)
        phase_counts, event_frame_phases = _phase_counts(features, energy, start_idx, end_idx, peak_energy)
        for idx in range(start_idx, end_idx + 1):
            frame_to_event[features[idx]["frame_id"]] = event_id
        frame_phases.update(event_frame_phases)
        events.append(
            {
                "event_id": event_id,
                "start_frame": int(features[start_idx]["frame_id"]),
                "end_frame": int(features[end_idx]["frame_id"]),
                "duration_frames": int(end_idx - start_idx + 1),
                "peak_frame": int(features[peak_idx]["frame_id"]),
                "contact_frame": contact_frame,
                "peak_energy": round(float(peak_energy), 4),
                "stroke_type": classification["stroke_type"],
                "confidence": classification["confidence"],
                "evidence": classification["evidence"],
                "quality_flags": quality_flags,
                "phase_counts": dict(sorted(phase_counts.items())),
            }
        )

    if not events:
        return None
    return _build_result(features, energy, events, frame_to_event, frame_phases)


def _build_result(features: List[Dict], energy: List[float], events: List[Dict], frame_to_event: Dict[int, int], frame_phases: Optional[Dict[int, str]] = None) -> Dict:
    frame_trace = []
    frame_phases = frame_phases or {}
    event_ranges = {event["event_id"]: (event["start_frame"], event["end_frame"], event["peak_energy"]) for event in events}
    for idx, feature in enumerate(features):
        event_id = frame_to_event.get(feature["frame_id"])
        phase = "ready"
        if event_id is not None:
            phase = frame_phases.get(feature["frame_id"], phase)
            if phase == "ready":
                start_frame, end_frame, peak = event_ranges[event_id]
                if start_frame <= feature["frame_id"] <= end_frame:
                    phase = _phase_for(feature, energy[idx], peak)
        frame_trace.append(
            {
                "frame": int(feature["frame_id"]),
                "event_id": event_id,
                "phase": phase,
                "motion_energy": round(float(energy[idx]), 4),
                "raw_swing_type": feature.get("raw_swing_type"),
                "wrist_speed": feature.get("wrist_speed"),
                "racket_speed": feature.get("racket_speed"),
                "ball_racket_distance": feature.get("ball_racket_distance"),
                "contact_score": feature.get("contact_score"),
                "two_hand_distance": feature.get("two_hand_distance"),
            }
        )
    return {"events": events, "frame_trace": frame_trace}


def segment_swing_events(
    features: List[Dict],
    min_peak_energy: float = 9.0,
    active_energy: float = 5.5,
    min_event_frames: int = 8,
    max_internal_gap: int = 3,
    min_event_gap: int = 18,
) -> Dict:
    """Segment complete swing events from frame features.

    The algorithm first finds broad motion islands, then merges nearby islands
    that are likely the same swing's follow-through/recovery.
    """
    if not features:
        return {"events": [], "frame_trace": []}

    energy = _smooth([_motion_energy(f) for f in features], window=3)
    fps = _estimate_fps(features)
    if fps:
        peak_result = _segment_by_peaks(features, energy, min_peak_energy, min_event_frames, min_event_gap, fps)
        if peak_result is not None:
            return peak_result

    active = [e >= active_energy for e in energy]

    islands = []
    start = None
    gap = 0
    for idx, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = idx
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_internal_gap:
                islands.append((start, idx - gap))
                start = None
                gap = 0
    if start is not None:
        islands.append((start, len(features) - 1))

    merged = []
    for island in islands:
        if not merged:
            merged.append(list(island))
            continue
        prev = merged[-1]
        if island[0] - prev[1] <= min_event_gap:
            prev[1] = island[1]
        else:
            merged.append(list(island))

    events = []
    frame_to_event = {}
    for start_idx, end_idx in merged:
        segment_energy = energy[start_idx : end_idx + 1]
        peak_energy = max(segment_energy) if segment_energy else 0.0
        if (end_idx - start_idx + 1) < min_event_frames or peak_energy < min_peak_energy:
            continue

        event_features = features[start_idx : end_idx + 1]
        classification = classify_swing_event(event_features)
        event_id = len(events) + 1
        peak_rel = segment_energy.index(peak_energy) if segment_energy else 0
        peak_idx = start_idx + peak_rel
        contact_frame = _best_contact_frame(features, start_idx, end_idx, peak_idx)
        quality_flags = _event_quality_flags(features, start_idx, end_idx, classification, contact_frame)
        phases = []
        for idx in range(start_idx, end_idx + 1):
            phase = _phase_for(features[idx], energy[idx], peak_energy)
            phases.append(phase)
            frame_to_event[features[idx]["frame_id"]] = event_id

        events.append(
            {
                "event_id": event_id,
                "start_frame": int(features[start_idx]["frame_id"]),
                "end_frame": int(features[end_idx]["frame_id"]),
                "duration_frames": int(end_idx - start_idx + 1),
                "peak_frame": int(features[peak_idx]["frame_id"]),
                "contact_frame": contact_frame,
                "peak_energy": round(float(peak_energy), 4),
                "stroke_type": classification["stroke_type"],
                "confidence": classification["confidence"],
                "evidence": classification["evidence"],
                "quality_flags": quality_flags,
                "phase_counts": {phase: phases.count(phase) for phase in sorted(set(phases))},
            }
        )

    return _build_result(features, energy, events, frame_to_event)
