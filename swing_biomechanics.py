"""Aggregate auditable single-view 2D biomechanics for one Swing event."""

from __future__ import annotations

import math
from copy import deepcopy
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Point = Tuple[float, float]


def _point(value) -> Optional[Point]:
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and value[0] is not None
        and value[1] is not None
    ):
        return float(value[0]), float(value[1])
    return None


def _distance(a: Optional[Point], b: Optional[Point]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _midpoint(a: Optional[Point], b: Optional[Point]) -> Optional[Point]:
    if a is None or b is None:
        return None
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def _body_center(pose: Dict) -> Optional[Point]:
    shoulder = _midpoint(
        _point(pose.get("left_shoulder")),
        _point(pose.get("right_shoulder")),
    )
    hip = _midpoint(
        _point(pose.get("left_hip")),
        _point(pose.get("right_hip")),
    )
    if shoulder is not None and hip is not None:
        return _midpoint(shoulder, hip)
    return shoulder or hip


def _body_width(pose: Dict) -> Optional[float]:
    widths = [
        _distance(
            _point(pose.get("left_shoulder")),
            _point(pose.get("right_shoulder")),
        ),
        _distance(
            _point(pose.get("left_hip")),
            _point(pose.get("right_hip")),
        ),
    ]
    valid = [float(value) for value in widths if value is not None and value >= 4.0]
    return median(valid) if valid else None


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def _metric(
    value: Optional[float],
    unit: str,
    confidence: float,
    source_frames: Sequence[int],
    sample_count: int,
    observability: str = "observable_2d",
    coach_eligible: bool = True,
    exclusion_reason: Optional[str] = None,
) -> Dict:
    result = {
        "value": round(float(value), 4) if value is not None else None,
        "unit": unit,
        "confidence": _bounded(confidence if value is not None else 0.0),
        "sample_count": int(sample_count),
        "source_frames": [int(frame_id) for frame_id in source_frames],
        "observability": observability,
        "coach_eligible": bool(coach_eligible and value is not None),
    }
    if exclusion_reason:
        result["exclusion_reason"] = exclusion_reason
    return result


def _window_rows(
    rows_by_frame: Dict[int, Dict],
    frame_id: int,
    radius: int = 2,
) -> List[Dict]:
    return [
        rows_by_frame[candidate]
        for candidate in range(int(frame_id) - radius, int(frame_id) + radius + 1)
        if candidate in rows_by_frame
    ]


def _median_feature_metric(
    features_by_frame: Dict[int, Dict],
    frame_id: int,
    key: str,
    pose_ratio: float,
    unit: str = "deg_2d",
    confidence_cap: float = 0.92,
    coach_eligible: bool = True,
    observability: str = "observable_2d",
    exclusion_reason: Optional[str] = None,
) -> Dict:
    window = _window_rows(features_by_frame, frame_id)
    samples = [
        (int(row["frame_id"]), float(row[key]))
        for row in window
        if row.get(key) is not None
    ]
    availability = len(samples) / max(1, len(window))
    value = median(value for _, value in samples) if samples else None
    confidence = min(confidence_cap, pose_ratio * 0.65 + availability * 0.35)
    return _metric(
        value,
        unit,
        confidence,
        [frame for frame, _ in samples],
        len(samples),
        observability=observability,
        coach_eligible=coach_eligible,
        exclusion_reason=exclusion_reason,
    )


def _angle_delta(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _shoulder_turn_change_metric(
    features_by_frame: Dict[int, Dict],
    start_frame: int,
    contact_frame: int,
    pose_ratio: float,
) -> Dict:
    samples = [
        (frame_id, float(row["shoulder_turn_deg"]))
        for frame_id, row in sorted(features_by_frame.items())
        if start_frame <= frame_id <= contact_frame
        and row.get("shoulder_turn_deg") is not None
    ]
    if not samples:
        return _metric(None, "deg_2d", 0.0, [], 0)
    baseline_count = min(5, max(2, len(samples) // 4))
    baseline = median(value for _, value in samples[:baseline_count])
    source_frame, value = max(
        samples,
        key=lambda item: _angle_delta(item[1], baseline),
    )
    change = _angle_delta(value, baseline)
    coverage = len(samples) / max(1, contact_frame - start_frame + 1)
    confidence = min(0.82, pose_ratio * 0.65 + coverage * 0.35)
    return _metric(
        change,
        "deg_2d",
        confidence,
        [frame for frame, _ in samples[:baseline_count]] + [source_frame],
        len(samples),
        observability="image_plane_change",
    )


def _joint_angle(a: Optional[Point], b: Optional[Point], c: Optional[Point]) -> Optional[float]:
    if a is None or b is None or c is None:
        return None
    left = (a[0] - b[0], a[1] - b[1])
    right = (c[0] - b[0], c[1] - b[1])
    denominator = math.hypot(*left) * math.hypot(*right)
    if denominator <= 0:
        return None
    cosine = max(-1.0, min(1.0, (left[0] * right[0] + left[1] * right[1]) / denominator))
    return math.degrees(math.acos(cosine))


def _preparation_knee_flexion_metric(
    frames_by_id: Dict[int, Dict],
    start_frame: int,
    contact_frame: int,
    pose_ratio: float,
) -> Dict:
    preparation_end = start_frame + max(1, (contact_frame - start_frame) // 2)
    samples = []
    source_frames = []
    for frame_id in range(start_frame, preparation_end + 1):
        pose = (frames_by_id.get(frame_id) or {}).get("pose") or {}
        frame_angles = []
        for side in ("left", "right"):
            angle = _joint_angle(
                _point(pose.get(f"{side}_hip")),
                _point(pose.get(f"{side}_knee")),
                _point(pose.get(f"{side}_ankle")),
            )
            if angle is not None:
                frame_angles.append(angle)
        if frame_angles:
            samples.append(max(0.0, 180.0 - sum(frame_angles) / len(frame_angles)))
            source_frames.append(frame_id)
    availability = len(source_frames) / max(1, preparation_end - start_frame + 1)
    confidence = min(0.82, pose_ratio * 0.65 + availability * 0.35)
    return _metric(
        median(samples) if samples else None,
        "deg_2d",
        confidence,
        source_frames,
        len(samples),
        observability="image_plane_joint_angle",
    )


def _median_center(
    frames_by_id: Dict[int, Dict],
    frame_id: int,
    radius: int = 2,
) -> Tuple[Optional[Point], List[int]]:
    samples = []
    for row in _window_rows(frames_by_id, frame_id, radius=radius):
        center = _body_center(row.get("pose") or {})
        if center is not None:
            samples.append((int(row["frame_id"]), center))
    if not samples:
        return None, []
    return (
        (
            median(center[0] for _, center in samples),
            median(center[1] for _, center in samples),
        ),
        [frame_id for frame_id, _ in samples],
    )


def _movement_metric(
    frames_by_id: Dict[int, Dict],
    start_frame: int,
    end_frame: int,
    body_width: Optional[float],
    pose_ratio: float,
    exclusion_reason: str,
) -> Dict:
    start, start_sources = _median_center(frames_by_id, start_frame)
    end, end_sources = _median_center(frames_by_id, end_frame)
    distance = _distance(start, end)
    value = (
        distance / body_width
        if distance is not None and body_width is not None and body_width > 0
        else None
    )
    endpoint_coverage = (bool(start_sources) + bool(end_sources)) / 2.0
    confidence = min(
        0.80,
        pose_ratio * 0.7 + endpoint_coverage * 0.3,
    )
    return _metric(
        value,
        "body_width",
        confidence,
        sorted(set(start_sources + end_sources)),
        len(start_sources) + len(end_sources),
        observability="screen_body_center_displacement",
        coach_eligible=False,
        exclusion_reason=exclusion_reason,
    )


def _contact_position_metric(
    frames_by_id: Dict[int, Dict],
    features_by_frame: Dict[int, Dict],
    contact_frame: int,
    body_width: Optional[float],
    pose_ratio: float,
    contact_evidence_confidence: float,
) -> Dict:
    candidates = []
    for frame_id in range(contact_frame - 2, contact_frame + 3):
        frame = frames_by_id.get(frame_id) or {}
        feature = features_by_frame.get(frame_id) or {}
        center = _body_center(frame.get("pose") or {})
        ball = _point(feature.get("ball"))
        if center is None or ball is None:
            continue
        candidates.append(
            (
                abs(frame_id - contact_frame),
                -float(feature.get("contact_score") or 0.0),
                frame_id,
                abs(ball[0] - center[0]),
                float(feature.get("contact_score") or 0.0),
            )
        )
    if not candidates or body_width is None or body_width <= 0:
        return _metric(None, "body_width", 0.0, [], 0)
    _, _, source_frame, lateral_distance, contact_score = min(candidates)
    proximity = 1.0 - min(1.0, abs(source_frame - contact_frame) / 3.0)
    evidence_confidence = max(
        float(contact_score),
        float(contact_evidence_confidence),
    )
    confidence = min(0.85, pose_ratio * 0.35 + evidence_confidence * 0.5 + proximity * 0.15)
    normalized_distance = lateral_distance / body_width
    plausible_geometry = 0.15 <= normalized_distance <= 2.20
    coach_eligible = evidence_confidence >= 0.35 and plausible_geometry
    if evidence_confidence < 0.35:
        exclusion_reason = "contact_not_supported_by_ball_racket_proximity"
    elif not plausible_geometry:
        exclusion_reason = "contact_geometry_outside_plausible_single_view_range"
    else:
        exclusion_reason = None
    return _metric(
        normalized_distance,
        "body_width",
        confidence,
        [source_frame],
        1,
        observability="image_plane_lateral_only",
        coach_eligible=coach_eligible,
        exclusion_reason=exclusion_reason,
    )


def _early_recovery_frame(
    features_by_frame: Dict[int, Dict],
    contact_frame: int,
    end_frame: int,
    seconds: float = 0.40,
) -> int:
    contact = features_by_frame.get(contact_frame) or {}
    contact_time = contact.get("timestamp")
    if contact_time is None:
        return min(end_frame, contact_frame + 10)
    target = float(contact_time) + seconds
    candidates = [
        (frame_id, float(row["timestamp"]))
        for frame_id, row in features_by_frame.items()
        if contact_frame <= frame_id <= end_frame and row.get("timestamp") is not None
    ]
    if not candidates:
        return min(end_frame, contact_frame + 10)
    return min(candidates, key=lambda item: abs(item[1] - target))[0]


def aggregate_event_biomechanics(
    event: Dict,
    frames: Iterable[Dict],
    features: Iterable[Dict],
) -> Dict:
    """Return compact biomechanics for a completed event.

    All spatial distances are normalized by the median visible shoulder/hip
    width inside the event. They are image-plane estimates, not 3D joint
    kinetics.
    """
    start_frame = int(event["start_frame"])
    end_frame = int(event["end_frame"])
    contact_frame = int(event.get("contact_frame", event.get("peak_frame", start_frame)))
    frames_in_event = [
        row
        for row in frames
        if start_frame <= int(row.get("frame_id", -1)) <= end_frame
    ]
    features_in_event = [
        row
        for row in features
        if start_frame <= int(row.get("frame_id", -1)) <= end_frame
    ]
    frames_by_id = {
        int(row["frame_id"]): row
        for row in frames_in_event
        if row.get("frame_id") is not None
    }
    features_by_frame = {
        int(row["frame_id"]): row
        for row in features_in_event
        if row.get("frame_id") is not None
    }
    pose_ratio = float(
        (event.get("quality_flags") or {}).get("pose_frame_ratio")
        or (
            sum(1 for row in features_in_event if row.get("has_pose"))
            / max(1, len(features_in_event))
        )
    )
    body_width_samples = [
        (int(row["frame_id"]), _body_width(row.get("pose") or {}))
        for row in frames_in_event
    ]
    valid_widths = [
        (frame_id, width)
        for frame_id, width in body_width_samples
        if width is not None
    ]
    body_width = median(width for _, width in valid_widths) if valid_widths else None
    contact_feature = features_by_frame.get(contact_frame) or {}
    contact_evidence_confidence = max(
        0.0,
        min(1.0, float(contact_feature.get("contact_score") or 0.0)),
    )
    peak_frame = int(event.get("peak_frame", contact_frame))
    arm_reference_frame = (
        contact_frame if contact_evidence_confidence >= 0.35 else peak_frame
    )
    arm_observability = (
        "contact_window_2d"
        if contact_evidence_confidence >= 0.35
        else "swing_peak_window_2d"
    )
    early_recovery_frame = _early_recovery_frame(
        features_by_frame,
        contact_frame,
        end_frame,
    )

    return {
        "schema_version": "single_view_2d_v2",
        "coordinate_space": "image_plane_normalized_by_body_width",
        "contact_frame": contact_frame,
        "reference_body_width_px": (
            round(float(body_width), 4) if body_width is not None else None
        ),
        "limitations": [
            "single_view_2d_not_3d_kinetics",
            "camera_perspective_affects_depth_and_weight_transfer",
            "hip_shoulder_separation_is_image_plane_proxy",
            "screen_translation_is_not_balance_or_weight_transfer",
        ],
        "metrics": {
            "hip_shoulder_separation": _median_feature_metric(
                features_by_frame,
                contact_frame,
                "hip_shoulder_sep_deg",
                pose_ratio,
                unit="image_plane_deg",
                confidence_cap=0.35,
                coach_eligible=False,
                observability="image_plane_proxy_only",
                exclusion_reason="true_3d_separation_not_observable_single_view",
            ),
            "shoulder_turn": _median_feature_metric(
                features_by_frame,
                contact_frame,
                "shoulder_turn_deg",
                pose_ratio,
                unit="image_plane_deg",
                confidence_cap=0.45,
                coach_eligible=False,
                observability="absolute_image_orientation_only",
                exclusion_reason="absolute_projection_is_not_turn_magnitude",
            ),
            "shoulder_turn_change": _shoulder_turn_change_metric(
                features_by_frame,
                start_frame,
                contact_frame,
                pose_ratio,
            ),
            "preparation_knee_flexion": _preparation_knee_flexion_metric(
                frames_by_id,
                start_frame,
                contact_frame,
                pose_ratio,
            ),
            "arm_extension": _median_feature_metric(
                features_by_frame,
                arm_reference_frame,
                "arm_extension_deg",
                pose_ratio,
                unit="deg_2d",
                observability=arm_observability,
            ),
            "contact_lateral_distance": _contact_position_metric(
                frames_by_id,
                features_by_frame,
                contact_frame,
                body_width,
                pose_ratio,
                contact_evidence_confidence,
            ),
            "weight_transfer": _movement_metric(
                frames_by_id,
                start_frame,
                contact_frame,
                body_width,
                pose_ratio,
                "screen_translation_is_not_true_weight_transfer",
            ),
            "balance_drift": _movement_metric(
                frames_by_id,
                contact_frame,
                early_recovery_frame,
                body_width,
                pose_ratio,
                "body_center_translation_is_not_balance_stability",
            ),
        },
        "quality": {
            "pose_frame_ratio": round(pose_ratio, 4),
            "body_scale_frame_ratio": round(
                len(valid_widths) / max(1, len(frames_in_event)),
                4,
            ),
            "contact_evidence_confidence": round(contact_evidence_confidence, 4),
            "arm_extension_reference_frame": int(arm_reference_frame),
            "early_recovery_frame": int(early_recovery_frame),
        },
    }


def enrich_events_with_biomechanics(
    events: Iterable[Dict],
    frames: Iterable[Dict],
    features: Iterable[Dict],
) -> List[Dict]:
    """Copy events and attach their compact biomechanics documents."""
    frame_rows = list(frames)
    feature_rows = list(features)
    enriched = []
    for source in events:
        event = deepcopy(source)
        event["biomechanics"] = aggregate_event_biomechanics(
            event,
            frame_rows,
            feature_rows,
        )
        enriched.append(event)
    return enriched
