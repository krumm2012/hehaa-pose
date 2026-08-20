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


def _median_absolute_deviation(values: List[float]) -> float:
    if not values:
        return 0.0
    center = median(values)
    return float(median([abs(value - center) for value in values]))


def _sustained_change(values: List[bool]) -> bool:
    """Allow one noisy sample in the four-frame onset confirmation window."""
    return bool(values) and sum(values) >= max(1, len(values) - 1)


def _timeline_time(features: List[Dict], idx: int, fps: float) -> float:
    timestamp = features[idx].get("timestamp")
    if timestamp is not None:
        return float(timestamp)
    frame_id = features[idx].get("frame_id")
    if frame_id is not None:
        return float(frame_id) / max(float(fps), 1e-6)
    return float(idx) / max(float(fps), 1e-6)


def _first_index_at_or_after(
    features: List[Dict],
    target_time: float,
    fps: float,
    lower: int,
    upper: int,
) -> int:
    for idx in range(max(0, lower), min(len(features) - 1, upper) + 1):
        if _timeline_time(features, idx, fps) >= target_time:
            return idx
    return min(len(features) - 1, upper)


def _last_index_at_or_before(
    features: List[Dict],
    target_time: float,
    fps: float,
    lower: int,
    upper: int,
) -> int:
    result = max(0, lower)
    for idx in range(max(0, lower), min(len(features) - 1, upper) + 1):
        if _timeline_time(features, idx, fps) > target_time:
            break
        result = idx
    return result


def _refine_event_start(
    features: List[Dict],
    energy: List[float],
    search_start_idx: int,
    peak_idx: int,
    fps: float,
    active_energy: float,
) -> Tuple[int, Dict, Optional[int]]:
    """Find the preparation onset after a stable quiet basin.

    A fixed pre-peak window routinely starts after unit turn has already begun.
    This search instead identifies a short, stable ready basin and then requires
    a sustained rise in motion energy or shoulder turn.  If no quiet basin is
    visible, the seam is explicitly marked as a recovery/preparation transition
    rather than inventing a static ready phase.
    """
    search_start_idx = max(0, min(int(search_start_idx), peak_idx))
    peak_time = _timeline_time(features, peak_idx, fps)
    search_end_idx = max(
        search_start_idx,
        _last_index_at_or_before(
            features,
            peak_time - 0.30,
            fps,
            search_start_idx,
            peak_idx - 1,
        ),
    )
    quiet_windows = []
    for window_start in range(search_start_idx, search_end_idx + 1):
        window_end = _first_index_at_or_after(
            features,
            _timeline_time(features, window_start, fps) + 0.16,
            fps,
            window_start,
            search_end_idx,
        )
        window_times = [
            _timeline_time(features, idx, fps)
            for idx in range(window_start, window_end + 1)
        ]
        if (
            window_end > search_end_idx
            or window_end - window_start + 1 < 3
            or window_times[-1] - window_times[0] < 0.12
            or any(
                later - earlier > max(0.12, 2.5 / fps)
                for earlier, later in zip(window_times, window_times[1:])
            )
        ):
            continue
        values = [float(value) for value in energy[window_start : window_end + 1]]
        quiet_windows.append(
            {
                "start": window_start,
                "end": window_end,
                "median": float(median(values)),
                "mad": _median_absolute_deviation(values),
            }
        )

    if not quiet_windows:
        start_idx = search_start_idx
        evidence = {
            "mode": "recovery_ready_transition",
            "confidence": "low",
            "reason": "insufficient_pre_peak_window",
            "search_start_frame": int(features[search_start_idx]["frame_id"]),
            "search_end_frame": int(features[search_end_idx]["frame_id"]),
        }
        return start_idx, evidence, start_idx

    best_median = min(window["median"] for window in quiet_windows)
    near_best_tolerance = max(1.0, best_median * 0.18)
    stable_windows = [
        window
        for window in quiet_windows
        if window["median"] <= best_median + near_best_tolerance
        and window["mad"] <= max(1.5, window["median"] * 0.30)
    ]
    quiet = stable_windows[0] if stable_windows else min(
        quiet_windows,
        key=lambda window: (window["median"], window["mad"], window["start"]),
    )
    quiet_start = int(quiet["start"])
    quiet_end = int(quiet["end"])
    quiet_is_confident = bool(stable_windows) and quiet["median"] <= max(
        float(active_energy) * 1.8,
        float(active_energy) + 3.0,
    )

    shoulder_values = [
        float(features[idx]["shoulder_turn_deg"])
        for idx in range(quiet_start, quiet_end + 1)
        if features[idx].get("shoulder_turn_deg") is not None
    ]
    shoulder_baseline = float(median(shoulder_values)) if shoulder_values else None
    shoulder_mad = _median_absolute_deviation(shoulder_values)
    shoulder_threshold = max(4.0, shoulder_mad * 3.0)
    energy_threshold = float(quiet["median"]) + max(2.5, float(quiet["mad"]) * 2.0)

    onset_idx = None
    onset_signal = "none"
    onset_search_end_idx = max(search_end_idx, peak_idx - 1)
    for idx in range(quiet_start, onset_search_end_idx + 1):
        horizon_end = _first_index_at_or_after(
            features,
            _timeline_time(features, idx, fps) + 0.16,
            fps,
            idx,
            onset_search_end_idx,
        )
        horizon = list(range(idx, min(onset_search_end_idx, horizon_end) + 1))
        if len(horizon) < 3:
            break
        horizon_times = [_timeline_time(features, pos, fps) for pos in horizon]
        if any(
            later - earlier > max(0.12, 2.5 / fps)
            for earlier, later in zip(horizon_times, horizon_times[1:])
        ):
            continue
        energy_rise = _sustained_change(
            [float(energy[pos]) >= energy_threshold for pos in horizon]
        )
        shoulder_rise = False
        if shoulder_baseline is not None:
            shoulder_rise = _sustained_change(
                [
                    features[pos].get("shoulder_turn_deg") is not None
                    and abs(float(features[pos]["shoulder_turn_deg"]) - shoulder_baseline)
                    >= shoulder_threshold
                    for pos in horizon
                ]
            )
        if energy_rise or shoulder_rise:
            onset_idx = idx
            onset_signal = (
                "energy_and_shoulder"
                if energy_rise and shoulder_rise
                else "energy"
                if energy_rise
                else "shoulder"
            )
            break

    if onset_idx is None:
        onset_idx = quiet_end + 1
        onset_signal = "quiet_basin_end"
        quiet_is_confident = False
    mode = "quiet_onset" if quiet_is_confident else "recovery_ready_transition"
    start_idx = max(search_start_idx, min(peak_idx - 1, onset_idx - 1))
    transition_end_idx = start_idx if mode == "recovery_ready_transition" else None
    confidence = (
        "high"
        if quiet_is_confident and onset_signal == "energy_and_shoulder"
        else "medium"
        if quiet_is_confident
        else "low"
    )
    evidence = {
        "mode": mode,
        "confidence": confidence,
        "onset_signal": onset_signal,
        "search_start_frame": int(features[search_start_idx]["frame_id"]),
        "search_end_frame": int(features[search_end_idx]["frame_id"]),
        "quiet_start_frame": int(features[quiet_start]["frame_id"]),
        "quiet_end_frame": int(features[quiet_end]["frame_id"]),
        "onset_frame": int(features[onset_idx]["frame_id"]),
        "energy_baseline": round(float(quiet["median"]), 4),
        "energy_mad": round(float(quiet["mad"]), 4),
        "energy_onset_threshold": round(energy_threshold, 4),
        "shoulder_turn_baseline_deg": (
            round(shoulder_baseline, 4) if shoulder_baseline is not None else None
        ),
        "shoulder_turn_onset_threshold_deg": (
            round(shoulder_threshold, 4) if shoulder_baseline is not None else None
        ),
    }
    return start_idx, evidence, transition_end_idx


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


def _local_peak_candidates(
    features: List[Dict],
    energy: List[float],
    peak_floor: float,
    edge_margin_seconds: float,
    fps: float,
) -> List[Tuple[int, float]]:
    candidates = []
    first_time = _timeline_time(features, 0, fps)
    last_time = _timeline_time(features, len(features) - 1, fps)
    for idx in range(1, len(energy) - 1):
        current_time = _timeline_time(features, idx, fps)
        if (
            current_time - first_time < edge_margin_seconds
            or last_time - current_time < edge_margin_seconds
        ):
            continue
        if energy[idx] >= peak_floor and energy[idx] >= energy[idx - 1] and energy[idx] >= energy[idx + 1]:
            candidates.append((idx, _peak_quality(features[idx], energy[idx])))
    return candidates


def _select_peak_indices(
    candidates: List[Tuple[int, float]],
    peak_min_distance: float,
    positions: Optional[Dict[int, float]] = None,
) -> List[int]:
    if not candidates:
        return []

    # Use quality-ordered non-maximum suppression. Grouping candidates by the
    # distance to the previous candidate creates transitive chains: a sequence
    # of weak intermediate peaks can join two genuinely separate swings and
    # suppress the later one for the lifetime of the realtime rolling window.
    selected = []
    for peak_idx, _quality in sorted(candidates, key=lambda item: item[1], reverse=True):
        peak_position = positions.get(peak_idx, float(peak_idx)) if positions else float(peak_idx)
        if all(
            abs(
                peak_position
                - (positions.get(kept_idx, float(kept_idx)) if positions else float(kept_idx))
            )
            > peak_min_distance
            for kept_idx in selected
        ):
            selected.append(peak_idx)
    return sorted(selected)


def _peak_event_ranges(
    features: List[Dict],
    peak_indices: List[int],
    fps: float,
) -> List[Tuple[int, int, int]]:
    ranges = []
    for pos, peak_idx in enumerate(peak_indices):
        peak_time = _timeline_time(features, peak_idx, fps)
        prev_boundary = _first_index_at_or_after(
            features,
            peak_time - 1.80,
            fps,
            0,
            peak_idx,
        )
        next_boundary = _last_index_at_or_before(
            features,
            peak_time + 1.25,
            fps,
            peak_idx,
            len(features) - 1,
        )
        if pos > 0:
            previous_peak_time = _timeline_time(features, peak_indices[pos - 1], fps)
            prev_boundary = max(
                prev_boundary,
                _first_index_at_or_after(
                    features,
                    previous_peak_time + 0.45,
                    fps,
                    peak_indices[pos - 1] + 1,
                    peak_idx,
                ),
            )
        if pos + 1 < len(peak_indices):
            next_peak_time = _timeline_time(features, peak_indices[pos + 1], fps)
            next_boundary = min(
                next_boundary,
                _last_index_at_or_before(
                    features,
                    (peak_time + next_peak_time) / 2.0,
                    fps,
                    peak_idx,
                    peak_indices[pos + 1] - 1,
                ),
            )
        ranges.append((prev_boundary, next_boundary, peak_idx))
    return ranges


def _phase_reference_energy(energy: List[float], start_idx: int, end_idx: int) -> float:
    values = sorted(float(value) for value in energy[start_idx : end_idx + 1])
    if not values:
        return 1.0
    return max(1.0, values[int(round((len(values) - 1) * 0.90))])


def _phase_counts(
    features: List[Dict],
    energy: List[float],
    start_idx: int,
    end_idx: int,
    contact_idx: int,
    transition_end_idx: Optional[int] = None,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    phases = {}
    frame_phases = {}
    reference_energy = _phase_reference_energy(energy, start_idx, end_idx)
    ready_threshold = reference_energy * 0.25
    forward_threshold = reference_energy * 0.78
    motion_started = False
    forward_started = False
    recovered = False
    for idx in range(start_idx, end_idx + 1):
        feature = features[idx]
        energy_value = float(energy[idx])
        if transition_end_idx is not None and idx <= transition_end_idx:
            phase = "recovery_ready_transition"
        elif idx == contact_idx and float(feature.get("contact_score") or 0.0) >= 0.35:
            phase = "contact_candidate"
        elif idx < contact_idx:
            if forward_started or energy_value >= forward_threshold:
                forward_started = True
                motion_started = True
                phase = "forward_swing"
            elif motion_started or energy_value >= ready_threshold:
                motion_started = True
                phase = "backswing"
            else:
                phase = "ready"
        elif idx == contact_idx:
            phase = "forward_swing"
        else:
            if not recovered:
                recovery_horizon = energy[idx : min(end_idx + 1, idx + 3)]
                recovered = len(recovery_horizon) >= 3 and all(
                    float(value) < ready_threshold for value in recovery_horizon
                )
            phase = "ready" if recovered else "follow_through"
        phases[phase] = phases.get(phase, 0) + 1
        frame_phases[feature["frame_id"]] = phase
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
    classification_context = evidence.get("classification_context") or {}
    camera_context = classification_context.get("camera") or {}
    if camera_context.get("view") in {"unknown", "side_or_uncertain"}:
        warnings.append("camera_view_uncertain")
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


def summarize_manual_event_range(
    features: List[Dict],
    start_frame: int,
    contact_frame: int,
    end_frame: int,
    classification_evidence: Optional[Dict] = None,
) -> Dict:
    """Rebuild range-dependent evidence for one human-confirmed event.

    Manual review may change the temporal boundaries without rerunning event
    segmentation.  This public seam keeps quality and phase aggregation inside
    the segmenter instead of duplicating its thresholds in the review workflow.
    """
    if not features:
        raise ValueError("Cannot summarize a manual event without frame features")

    start_frame = int(start_frame)
    contact_frame = int(contact_frame)
    end_frame = int(end_frame)
    if not start_frame <= contact_frame <= end_frame:
        raise ValueError("Manual event must satisfy start <= contact <= end")

    indices = [
        index
        for index, feature in enumerate(features)
        if start_frame <= int(feature.get("frame_id", -1)) <= end_frame
    ]
    if not indices:
        raise ValueError("Manual event range does not contain any frame features")
    start_idx, end_idx = indices[0], indices[-1]
    contact_idx = min(
        indices,
        key=lambda index: abs(int(features[index].get("frame_id", -1)) - contact_frame),
    )
    effective_contact_frame = int(features[contact_idx]["frame_id"])
    energy = _smooth([_motion_energy(feature) for feature in features], window=3)
    peak_idx = max(indices, key=lambda index: float(energy[index]))
    classification = {
        "evidence": dict(classification_evidence or {}),
    }
    quality_flags = _event_quality_flags(
        features,
        start_idx,
        end_idx,
        classification,
        effective_contact_frame,
    )
    phase_counts, frame_phases = _phase_counts(
        features,
        energy,
        start_idx,
        end_idx,
        contact_idx,
    )
    return {
        "start_frame": int(features[start_idx]["frame_id"]),
        "contact_frame": effective_contact_frame,
        "end_frame": int(features[end_idx]["frame_id"]),
        "duration_frames": int(end_idx - start_idx + 1),
        "peak_frame": int(features[peak_idx]["frame_id"]),
        "peak_energy": round(float(energy[peak_idx]), 4),
        "quality_flags": quality_flags,
        "phase_counts": dict(sorted(phase_counts.items())),
        "frame_phases": {
            str(frame_id): phase
            for frame_id, phase in sorted(frame_phases.items())
        },
    }


def _classify_impact_event(
    features: List[Dict],
    start_idx: int,
    end_idx: int,
    contact_idx: int,
    fps: float,
) -> Dict:
    """Classify from late preparation through impact, excluding recovery.

    Ready and follow-through frames often put both hands close together.  They
    are useful for segmentation but are misleading swing-side evidence, so the
    classifier receives only the impact-centred window.
    """
    contact_time = _timeline_time(features, contact_idx, fps)
    core_start = max(
        start_idx,
        _first_index_at_or_after(
            features,
            contact_time - 0.56,
            fps,
            start_idx,
            contact_idx,
        ),
    )
    core_end = min(
        end_idx,
        _last_index_at_or_before(
            features,
            contact_time + 0.12,
            fps,
            contact_idx,
            end_idx,
        ),
    )
    classification = classify_swing_event(features[core_start : core_end + 1])
    classification["evidence"]["core_start_frame"] = int(features[core_start]["frame_id"])
    classification["evidence"]["core_end_frame"] = int(features[core_end]["frame_id"])
    classification["evidence"]["classification_anchor_frame"] = int(
        features[contact_idx]["frame_id"]
    )
    return classification


def _segment_by_peaks(
    features: List[Dict],
    energy: List[float],
    min_peak_energy: float,
    active_energy: float,
    min_event_frames: int,
    min_event_gap: int,
    fps: float,
) -> Optional[Dict]:
    if len(features) < 60:
        return None

    peak_floor = _robust_peak_floor(energy, min_peak_energy)
    peak_min_distance_seconds = max(float(min_event_gap) / fps, 1.6)
    edge_margin_seconds = max(float(min_event_frames) / fps, 0.35)
    candidates = _local_peak_candidates(
        features,
        energy,
        peak_floor,
        edge_margin_seconds,
        fps,
    )
    positions = {
        idx: _timeline_time(features, idx, fps)
        for idx, _quality in candidates
    }
    peak_indices = _select_peak_indices(
        candidates,
        peak_min_distance_seconds,
        positions=positions,
    )
    if not peak_indices:
        return None

    events = []
    frame_to_event = {}
    frame_phases = {}
    refined_ranges = []
    broad_ranges = _peak_event_ranges(features, peak_indices, fps)
    for search_start_idx, end_idx, peak_idx in broad_ranges:
        start_idx, boundary_evidence, transition_end_idx = _refine_event_start(
            features,
            energy,
            search_start_idx,
            peak_idx,
            fps,
            active_energy,
        )
        refined_ranges.append(
            (start_idx, end_idx, peak_idx, boundary_evidence, transition_end_idx)
        )

    for range_pos, (
        start_idx,
        end_idx,
        peak_idx,
        boundary_evidence,
        transition_end_idx,
    ) in enumerate(refined_ranges):
        if range_pos + 1 < len(refined_ranges):
            end_idx = min(end_idx, refined_ranges[range_pos + 1][0] - 1)
        if (end_idx - start_idx + 1) < min_event_frames:
            continue
        peak_energy = max(energy[start_idx : end_idx + 1])
        if peak_energy < min_peak_energy:
            continue
        event_id = len(events) + 1
        contact_frame = _best_contact_frame(features, start_idx, end_idx, peak_idx)
        contact_idx = next(
            (
                idx
                for idx in range(start_idx, end_idx + 1)
                if int(features[idx]["frame_id"]) == int(contact_frame)
            ),
            peak_idx,
        )
        classification = _classify_impact_event(
            features,
            start_idx,
            end_idx,
            contact_idx,
            fps,
        )
        classification["evidence"]["start_boundary"] = boundary_evidence
        quality_flags = _event_quality_flags(features, start_idx, end_idx, classification, contact_frame)
        phase_counts, event_frame_phases = _phase_counts(
            features,
            energy,
            start_idx,
            end_idx,
            contact_idx,
            transition_end_idx=transition_end_idx,
        )
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
    for idx, feature in enumerate(features):
        event_id = frame_to_event.get(feature["frame_id"])
        phase = (
            frame_phases.get(feature["frame_id"], "ready")
            if event_id is not None
            else "ready"
        )
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
        peak_result = _segment_by_peaks(
            features,
            energy,
            min_peak_energy,
            active_energy,
            min_event_frames,
            min_event_gap,
            fps,
        )
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
    frame_phases = {}
    for start_idx, end_idx in merged:
        segment_energy = energy[start_idx : end_idx + 1]
        peak_energy = max(segment_energy) if segment_energy else 0.0
        if (end_idx - start_idx + 1) < min_event_frames or peak_energy < min_peak_energy:
            continue

        event_id = len(events) + 1
        peak_rel = segment_energy.index(peak_energy) if segment_energy else 0
        peak_idx = start_idx + peak_rel
        contact_frame = _best_contact_frame(features, start_idx, end_idx, peak_idx)
        contact_idx = next(
            (
                idx
                for idx in range(start_idx, end_idx + 1)
                if int(features[idx]["frame_id"]) == int(contact_frame)
            ),
            peak_idx,
        )
        classification = _classify_impact_event(
            features,
            start_idx,
            end_idx,
            contact_idx,
            fps or 25.0,
        )
        classification["evidence"]["start_boundary"] = {
            "mode": "active_island",
            "confidence": "medium",
            "onset_frame": int(features[start_idx]["frame_id"]),
        }
        quality_flags = _event_quality_flags(features, start_idx, end_idx, classification, contact_frame)
        phase_counts, event_frame_phases = _phase_counts(
            features,
            energy,
            start_idx,
            end_idx,
            contact_idx,
        )
        frame_phases.update(event_frame_phases)
        for idx in range(start_idx, end_idx + 1):
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
                "phase_counts": dict(sorted(phase_counts.items())),
            }
        )

    return _build_result(
        features,
        energy,
        events,
        frame_to_event,
        frame_phases,
    )
