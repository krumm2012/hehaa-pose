"""Aggregate auditable single-view 2D biomechanics for one Swing event."""

from __future__ import annotations

import math
from copy import deepcopy
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from metric_source_windows import MetricWindowContext, attach_window, combine_windows
from image_motion_measurements import POLICY_VERSION as IMAGE_MOTION_POLICY
from observation_policy import (finite_number, finite_point, image_joint_angle,
                                measurement_pose_with_evidence, qualified_front_point)


Point = Tuple[float, float]


def _point(value) -> Optional[Point]:
    return finite_point(value)


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
    return None


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
    valid = [float(value) for value in widths if finite_number(value) is not None and value >= 4.0]
    return median(valid) if valid else None


def _torso_length(pose: Dict) -> Optional[float]:
    """
    计算中肩至中髋的垂直躯干中轴欧氏长度。
    在网球水平转体与侧身引拍过程中，垂直轴几乎不随身体朝向发生投影透视缩短，
    保留率 > 96%，具备极高几何旋转不变性。
    """
    l_sh = _point(pose.get("left_shoulder"))
    r_sh = _point(pose.get("right_shoulder"))
    l_hip = _point(pose.get("left_hip"))
    r_hip = _point(pose.get("right_hip"))

    sh_mid = _midpoint(l_sh, r_sh) if (l_sh and r_sh) else (l_sh or r_sh)
    hip_mid = _midpoint(l_hip, r_hip) if (l_hip and r_hip) else (l_hip or r_hip)

    if sh_mid is not None and hip_mid is not None:
        dist = _distance(sh_mid, hip_mid)
        if dist is not None and dist > 10.0:
            return float(dist)
    return None


def _robust_body_scale(pose: Dict) -> Optional[float]:
    """
    自适应鲁棒人体尺度基准：
    正常体态下返回直接测量的 body_width；
    当检测到侧身引拍导致横向体宽投影骤缩（body_width < torso_length * 0.45）时，
    自动由垂直躯干中轴长等效折算 (torso_length / 1.40)，
    彻底消除侧身蓄力时的尺度几何畸变与虚高除零风险。
    """
    bw = _body_width(pose)
    tl = _torso_length(pose)
    if tl is not None and tl > 15.0:
        if bw is None or bw < tl * 0.45:
            return float(tl / 1.40)
    return bw


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, finite_number(value) or 0.0)), 4)


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
    invalid_value = value is not None and finite_number(value) is None
    value = finite_number(value)
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
    if invalid_value:
        result['details'] = {'reason': 'nonfinite_or_invalid_metric_value'}
    return result


def _metric_windows(rows_by_frame, windows=None):
    return windows if windows is not None else MetricWindowContext(
        [{**row, 'frame_id': row.get('frame_id', fid)} for fid, row in rows_by_frame.items()])


def _window_rows(
    rows_by_frame: Dict[int, Dict],
    frame_id: int,
    radius: int = 2,
    windows=None,
) -> List[Dict]:
    window = _metric_windows(rows_by_frame, windows).around(frame_id,
        before=radius / 25., after=radius / 25., legacy_before=radius, legacy_after=radius)
    return [rows_by_frame[fid] for fid in window.frame_ids if fid in rows_by_frame]


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
    windows=None,
) -> Dict:
    window = _metric_windows(features_by_frame, windows).around(frame_id)
    samples = [
        (fid, float(features_by_frame[fid][key]))
        for fid in window.frame_ids if fid in features_by_frame
        and features_by_frame[fid].get(key) is not None
        and math.isfinite(float(features_by_frame[fid][key]))
    ]
    availability = len(samples) / max(1, len(window.frame_ids))
    value = median(value for _, value in samples) if samples else None
    confidence = min(confidence_cap, pose_ratio * 0.65 + availability * 0.35)
    return attach_window(_metric(
        value,
        unit,
        confidence,
        [frame for frame, _ in samples],
        len(samples),
        observability=observability,
        coach_eligible=coach_eligible,
        exclusion_reason=exclusion_reason,
    ), window)


def _angle_delta(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _shoulder_turn_change_metric(
    features_by_frame: Dict[int, Dict],
    start_frame: int,
    contact_frame: int,
    pose_ratio: float,
    windows=None,
) -> Dict:
    windows = _metric_windows(features_by_frame, windows)
    window = windows.between(start_frame, contact_frame)
    samples = [
        (frame_id, float(row.get("shoulder_line_angle_deg", row.get("shoulder_turn_deg"))))
        for frame_id in window.frame_ids if frame_id in features_by_frame
        for row in [features_by_frame[frame_id]]
        if row.get("shoulder_line_angle_deg", row.get("shoulder_turn_deg")) is not None
        and math.isfinite(float(row.get("shoulder_line_angle_deg", row.get("shoulder_turn_deg"))))
    ]
    if not samples:
        return attach_window(_metric(None, "deg_2d", 0.0, [], 0), window)
    baseline_count = min(5, max(2, len(samples) // 4))
    baseline_samples = samples[:baseline_count]
    if windows.timed:
        baseline_window = windows.between(start_frame, contact_frame, end_fraction=.25, max_duration=.16)
        baseline_samples = [sample for sample in samples if sample[0] in baseline_window.frame_ids]
        window.evidence['baseline_window'] = baseline_window.evidence
    if not baseline_samples:
        window.evidence['reasons'].append('shoulder_baseline_not_observed')
        return attach_window(_metric(None, 'deg_2d', 0., [], 0,
            observability='image_plane_change', coach_eligible=False), window)
    anchor = samples[0][1]
    baseline = anchor + median((value - anchor + 180.0) % 360.0 - 180.0
                               for _, value in baseline_samples)
    source_frame, value = max(
        samples,
        key=lambda item: _angle_delta(item[1], baseline),
    )
    change = _angle_delta(value, baseline)
    coverage = len(samples) / max(1, len(window.frame_ids))
    confidence = min(0.82, pose_ratio * 0.65 + coverage * 0.35)
    return attach_window(_metric(
        change,
        "deg_2d",
        confidence,
        [frame for frame, _ in baseline_samples] + [source_frame],
        len(samples),
        observability="image_plane_change",
    ), window)


def _event_peak_or_median_feature(
    features_by_frame: Dict[int, Dict],
    start_frame: int,
    end_frame: int,
    key: str,
    pose_ratio: float,
    unit: str = "ratio",
    use_max: bool = True,
    confidence_cap: float = 0.90,
    coach_eligible: bool = True,
    observability: str = "dual_view_mirror_projection",
    windows=None,
) -> Dict:
    window = _metric_windows(features_by_frame, windows).between(start_frame, end_frame)
    samples = [
        (frame_id, float(features_by_frame[frame_id][key]))
        for frame_id in window.frame_ids if frame_id in features_by_frame
        and features_by_frame[frame_id].get(key) is not None
        and math.isfinite(float(features_by_frame[frame_id][key]))
    ]
    if not samples:
        return attach_window(_metric(None, unit, 0.0, [], 0, observability=observability, coach_eligible=False), window)
    coverage = len(samples) / max(1, len(window.frame_ids))
    if use_max:
        source_frame, value = max(samples, key=lambda s: s[1])
        source_frames = [source_frame]
    else:
        value = median(v for _, v in samples)
        source_frames = [f for f, _ in samples]
    confidence = min(confidence_cap, pose_ratio * 0.65 + coverage * 0.35)
    return attach_window(_metric(
        value,
        unit,
        confidence,
        source_frames,
        len(samples),
        observability=observability,
        coach_eligible=coach_eligible,
    ), window)


def _joint_angle(a: Optional[Point], b: Optional[Point], c: Optional[Point]) -> Optional[float]:
    return image_joint_angle(a, b, c)


def _preparation_knee_flexion_metric(
    frames_by_id: Dict[int, Dict],
    start_frame: int,
    contact_frame: int,
    pose_ratio: float,
    windows=None,
) -> Dict:
    window = _metric_windows(frames_by_id, windows).between(start_frame, contact_frame, end_fraction=.5)
    samples = []
    source_frames = []
    for frame_id in window.frame_ids:
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
    availability = len(source_frames) / max(1, len(window.frame_ids))
    confidence = min(0.82, pose_ratio * 0.65 + availability * 0.35)
    return attach_window(_metric(
        median(samples) if samples else None,
        "deg_2d",
        confidence,
        source_frames,
        len(samples),
        observability="image_plane_joint_angle",
    ), window)


def _median_center(
    frames_by_id: Dict[int, Dict],
    frame_id: int,
    radius: int = 2,
    windows=None,
) -> Tuple[Optional[Point], List[int]]:
    samples = []
    for row in _window_rows(frames_by_id, frame_id, radius=radius, windows=windows):
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
    windows=None,
) -> Dict:
    windows = _metric_windows(frames_by_id, windows)
    start, start_sources = _median_center(frames_by_id, start_frame, windows=windows)
    end, end_sources = _median_center(frames_by_id, end_frame, windows=windows)
    window = combine_windows([windows.around(start_frame), windows.around(end_frame)])
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
    return attach_window(_metric(
        value,
        "body_width",
        confidence,
        sorted(set(start_sources + end_sources)),
        len(start_sources) + len(end_sources),
        observability="screen_body_center_displacement",
        coach_eligible=False,
        exclusion_reason=exclusion_reason,
    ), window)


def _contact_position_metric(
    frames_by_id: Dict[int, Dict],
    features_by_frame: Dict[int, Dict],
    contact_frame: int,
    body_width: Optional[float],
    pose_ratio: float,
    contact_evidence_confidence: float,
    windows=None,
) -> Dict:
    windows = _metric_windows(frames_by_id, windows)
    window = windows.around(contact_frame)
    candidates = []
    for frame_id in window.frame_ids:
        frame = frames_by_id.get(frame_id) or {}
        feature = features_by_frame.get(frame_id) or {}
        center = _body_center(frame.get("pose") or {})
        ball = _point(feature.get("ball"))
        if center is None or ball is None:
            continue
        candidates.append(
            (
                abs(windows.times[frame_id] - windows.times[contact_frame]) if windows.timed else abs(frame_id - contact_frame),
                -float(feature.get("contact_score") or 0.0),
                frame_id,
                abs(ball[0] - center[0]),
                float(feature.get("contact_score") or 0.0),
            )
        )
    if not candidates or body_width is None or body_width <= 0:
        return attach_window(_metric(None, "body_width", 0.0, [], 0), window)
    offset, _, source_frame, lateral_distance, contact_score = min(candidates)
    proximity = 1.0 - min(1.0, offset / (.12 if windows.timed else 3.0))
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
    return attach_window(_metric(
        normalized_distance,
        "body_width",
        confidence,
        [source_frame],
        1,
        observability="image_plane_lateral_only",
        coach_eligible=coach_eligible,
        exclusion_reason=exclusion_reason,
    ), window)


def _early_recovery_frame(
    features_by_frame: Dict[int, Dict],
    contact_frame: int,
    end_frame: int,
    seconds: float = 0.40,
    windows=None,
) -> Optional[int]:
    selected, _ = _metric_windows(features_by_frame, windows).recovery(contact_frame, end_frame, seconds)
    return selected


def _calculate_extended_tier_biomechanics(
    features_in_event: List[Dict],
    start_frame: int,
    contact_frame: int,
    end_frame: int,
    body_width: Optional[float],
    fps: float = 25.0,
) -> Dict[str, Any]:
    """Raw image estimates; unsupported legacy peaks never establish a chain."""
    windows = MetricWindowContext(features_in_event)
    features_by_id = {f['frame_id']: f for f in features_in_event}
    # Uncalibrated image speed is a box-centre velocity, not racket-head km/h.
    contact_f = next((f for f in features_in_event if f.get("frame_id") == contact_frame), {})
    speed_samples = [f for f in features_in_event if f.get("racket_speed_px_s") is not None]
    image_peak = max((f["racket_speed_px_s"] for f in speed_samples), default=None)
    contact_image_speed = contact_f.get("racket_speed_px_s")

    kmh_samples = [f["racket_head_speed_kmh"] for f in features_in_event if f.get("racket_head_speed_kmh") is not None]
    max_kmh = max(kmh_samples, default=None)
    contact_kmh = contact_f.get("racket_head_speed_kmh")
    speed_calibrated = contact_kmh is not None or max_kmh is not None

    # 2. 第一梯队：由下向上刷球角与掉拍头下潜深度 (Low-to-High Brush Angle & Drop Depth)
    pre_contact_feats = [features_by_id[fid] for fid in windows.around(
        contact_frame, before=.48, after=0., legacy_before=12, legacy_after=0).frame_ids]
    racket_centers = [
        (f["frame_id"], f.get("racket_measurement_point", f.get("racket_center")))
        for f in pre_contact_feats
        if f.get("racket_measurement_point", f.get("racket_center")) is not None
        and f.get("racket_center_source", "detected") == "detected"
    ]
    low_to_high_angle = 0.0
    racket_drop_px = 0.0
    contact_racket = contact_f.get("racket_measurement_point", contact_f.get("racket_center"))
    brush_observed = len(racket_centers) >= 2 and contact_racket is not None and contact_f.get("racket_center_source", "detected") == "detected"
    if brush_observed:
        lowest_f, lowest_pt = max(racket_centers, key=lambda item: item[1][1])
        dy = lowest_pt[1] - contact_racket[1]
        dx = abs(contact_racket[0] - lowest_pt[0])
        if dy > 0:
            low_to_high_angle = round(math.degrees(math.atan2(dy, dx + 1e-5)), 1)
            racket_drop_px = round(dy, 1)

    ref_scale = body_width if (body_width and body_width > 0) else 140.0
    racket_drop_ratio = round(racket_drop_px / ref_scale, 2)

    # 3. 第二梯队：步法站位识别 (Stance Type Classification: Open vs Semi-Open vs Closed)
    foot_angles = [features_by_id[fid]['stance_angle'] for fid in windows.around(
        contact_frame, before=.12, after=.12, legacy_before=3, legacy_after=3).frame_ids
        if features_by_id[fid].get('stance_angle') is not None]
    foot_angle = median(foot_angles) if foot_angles else None
    stance_type = None

    # 4. 第二梯队：垂直蹬地发力率 (Vertical Leg Drive)
    hip_ys = [
        (f["frame_id"], f.get("hip_vertical_pos"))
        for fid in windows.between(start_frame, contact_frame).frame_ids
        for f in [features_by_id[fid]] if f.get("hip_vertical_pos") is not None
    ]
    leg_drive_px = 0.0
    contact_hip = contact_f.get("hip_vertical_pos")
    hip_observed = len(hip_ys) >= 2 and contact_hip is not None
    if hip_observed:
        lowest_hip = max(hip_ys, key=lambda item: item[1])[1]
        leg_drive_px = max(0.0, lowest_hip - contact_hip)
    leg_drive_ratio = round(leg_drive_px / ref_scale, 2)

    # Candidate signal maxima are retained for auditing, not kinetic inference.
    accel_window = windows.around(contact_frame, before=.6, after=.16,
                                  legacy_before=15, legacy_after=4)
    accel_features = [features_by_id[fid] for fid in accel_window.frame_ids]
    hip_peak_f = max(accel_features or [{}], key=lambda f: float(f.get("hip_rotation_speed") or 0.0)).get("frame_id", contact_frame)
    sh_peak_f = max(accel_features or [{}], key=lambda f: float(f.get("shoulder_rotation_speed") or 0.0)).get("frame_id", contact_frame)
    rkt_peak_f = max(accel_features or [{}], key=lambda f: float(f.get("racket_speed") or 0.0)).get("frame_id", contact_frame)

    # Legacy weighted composite is retired: missing data never receives defaults.
    # A single practice policy is attached after the evidence metrics are assembled.
    total_score = None
    grade = None
    sequence_observed = bool(accel_features) and all(
        any(float(f.get(key) or 0) > 0 for f in accel_features)
        for key in ("hip_rotation_speed", "shoulder_rotation_speed", "racket_speed")
    )

    return {
        "racket_head_speed": {
            "max_kmh": round(max_kmh, 1) if max_kmh is not None else None,
            "contact_kmh": round(contact_kmh, 1) if contact_kmh is not None else None,
            "contact_px_s": contact_image_speed,
            "max_px_s": image_peak,
            "sample_count": len(speed_samples),
            "source_frames": [f["frame_id"] for f in speed_samples],
            "measurement_policy": IMAGE_MOTION_POLICY,
            "status": "ground_homography_calibrated" if speed_calibrated else "uncalibrated",
            "contact_time_basis": contact_f.get("racket_speed_time_basis"),
            "confidence": 0.85 if speed_calibrated else 0.0,
            "coach_eligible": False,
            "observability": "ground_plane_projected_speed" if speed_calibrated else "image_box_center_speed",
        },
        "brush_angle": {
            "low_to_high_angle_deg": low_to_high_angle if brush_observed else None,
            "drop_depth_px": racket_drop_px if brush_observed else None,
            "drop_depth_ratio": racket_drop_ratio if brush_observed and body_width else None,
            "confidence": 0.0,
            "coach_eligible": False,
        },
        "stance": {
            "stance_type": stance_type,
            "image_foot_line_angle_deg": foot_angle,
            "confidence": 0.0,
            "coach_eligible": False,
        },
        "leg_drive": {
            "drive_px": round(leg_drive_px, 1) if hip_observed else None,
            "drive_ratio": leg_drive_ratio if hip_observed and body_width else None,
            "confidence": 0.0,
            "coach_eligible": False,
        },
        "kinematic_sequence": {
            "hip_peak_frame": None,
            "shoulder_peak_frame": None,
            "racket_peak_frame": None,
            "legacy_candidate_peak_frames": {'hip': int(hip_peak_f), 'shoulder': int(sh_peak_f),
                'racket': int(rkt_peak_f)} if sequence_observed else {},
            "latency_hip_to_shoulder_ms": None,
            "latency_shoulder_to_racket_ms": None,
            "is_sequential": None,
            "sequence_quality": None,
            "reason": "independent_view_records_missing",
            "window_evidence": accel_window.evidence,
            "legacy_peak_semantics": "per_observation_candidate_signal_not_segment_velocity",
            "coach_eligible": False,
            "confidence": 0.0,
        },
        "swing_quality_score": {
            "overall_score": total_score,
            "grade": grade,
            "sub_scores": {},
        },
    }


def _score_curve(value: float, curve: Sequence[Tuple[float, float]]) -> float:
    points = sorted((float(x), float(y)) for x, y in curve)
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if left_x <= value <= right_x:
            ratio = (value - left_x) / max(1e-9, right_x - left_x)
            return left_y + (right_y - left_y) * ratio
    return points[-1][1]


def extract_biomechanical_sub_scores(event: Dict) -> Dict[str, float]:
    """Extract 0-100 diagnostic sub-scores for the 5-axis biomechanical quality radar.

    RADAR_AXES:
    1. shoulder_turn (转肩)
    2. takeback (引拍)
    3. arm_extension (延展)
    4. racket_speed (挥速)
    5. leg_drive (蹬地)
    """
    bio = event.get("biomechanics") or {}
    metrics = bio.get("metrics") or event.get("metrics") or {}
    ext = bio.get("extended_biomechanics") or event.get("extended_biomechanics") or {}
    sub_scores: Dict[str, float] = {}

    def _val(obj, key=None):
        if key is not None and isinstance(obj, dict):
            val = obj.get(key)
        else:
            val = obj
        if isinstance(val, dict):
            val = val.get("value")
        if val is None or isinstance(val, bool):
            return None
        try:
            f = float(val)
            return f if math.isfinite(f) else None
        except (ValueError, TypeError):
            return None

    # 1. shoulder_turn (转肩)
    stc = _val(metrics.get("shoulder_turn_change")) or _val(ext.get("shoulder_turn_change"))
    st_val = _val(metrics.get("shoulder_turn")) or _val(ext.get("shoulder_turn"))
    if stc is not None:
        sub_scores["shoulder_turn"] = round(_score_curve(stc, [(0, 35), (12, 50), (30, 80), (45, 100)]), 1)
    elif st_val is not None:
        sub_scores["shoulder_turn"] = round(_score_curve(st_val, [(0, 35), (20, 55), (40, 80), (60, 100)]), 1)

    # 2. takeback (引拍)
    tb = _val(metrics.get("takeback_depth")) or _val(ext.get("takeback_depth"))
    scap = _val(metrics.get("scapular_retraction")) or _val(ext.get("scapular_retraction"))
    if tb is not None:
        sub_scores["takeback"] = round(_score_curve(tb, [(0.0, 35), (0.6, 50), (1.2, 75), (1.8, 90), (2.2, 100)]), 1)
    elif scap is not None:
        sub_scores["takeback"] = round(_score_curve(scap, [(0.5, 40), (0.8, 65), (1.1, 85), (1.3, 100)]), 1)

    # 3. arm_extension (延展)
    ae = _val(metrics.get("arm_extension")) or _val(ext.get("arm_extension"))
    if ae is not None:
        sub_scores["arm_extension"] = round(_score_curve(ae, [(60, 35), (120, 62), (145, 80), (165, 100)]), 1)

    # 4. racket_speed (挥速)
    rkt_ext = ext.get("racket_head_speed") or event.get("racket_speed") or {}
    max_spd = _val(rkt_ext, "max_px_s")
    c_spd = _val(rkt_ext, "contact_px_s") or _val(metrics.get("racket_head_speed"))
    if max_spd is not None:
        sub_scores["racket_speed"] = round(_score_curve(max_spd, [(0, 30), (800, 50), (1600, 70), (2400, 85), (3200, 100)]), 1)
    elif c_spd is not None:
        sub_scores["racket_speed"] = round(_score_curve(c_spd, [(0, 30), (400, 55), (700, 75), (1000, 90), (1400, 100)]), 1)

    # 5. leg_drive (蹬地)
    pkf = _val(metrics.get("preparation_knee_flexion")) or _val(ext.get("preparation_knee_flexion"))
    ld_ratio = _val(ext.get("leg_drive"), "drive_ratio") or _val(metrics.get("leg_drive"))
    if pkf is not None:
        sub_scores["leg_drive"] = round(_score_curve(pkf, [(0, 35), (12, 50), (25, 75), (45, 100)]), 1)
    elif ld_ratio is not None:
        sub_scores["leg_drive"] = round(_score_curve(ld_ratio, [(0.0, 35), (0.03, 55), (0.06, 75), (0.10, 100)]), 1)

    return sub_scores


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
    frames = list(frames)
    features = list(features)
    start_frame = int(event["start_frame"])
    end_frame = int(event["end_frame"])
    contact_frame = int(event.get("contact_frame", event.get("peak_frame", start_frame)))
    frames_in_event = []
    for row in frames:
        if start_frame <= int(row.get('frame_id', -1)) <= end_frame:
            pose, qualification = measurement_pose_with_evidence(row)
            frames_in_event.append({**row, 'pose': pose, 'pose_qualification': qualification})
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
    windows = MetricWindowContext(frames_in_event)
    supplied_ratio = (event.get('quality_flags') or {}).get('pose_frame_ratio')
    pose_ratio = finite_number(supplied_ratio) if supplied_ratio is not None else (
            sum(1 for row in features_in_event if row.get("has_pose"))
            / max(1, len(features_in_event))
    )
    pose_ratio = pose_ratio if pose_ratio is not None and 0 <= pose_ratio <= 1 else 0.
    observation_scores = []
    for row in frames_in_event:
        observations = row.get('pose_observations')
        front = observations.get('front') if isinstance(observations, dict) else None
        if not isinstance(front, dict): continue
        for point in front.values():
            if not isinstance(point, dict) or point.get('observed') is not True or point.get('recovered_from_mirror'): continue
            value, _ = qualified_front_point(point, row['frame_id'])
            observation_scores.append(value[2] if value is not None else 0.)
    if any("pose_observations" in row for row in frames_in_event):
        pose_ratio = min(pose_ratio, sum(observation_scores) / max(1, len(observation_scores)))
    body_width_samples = [
        (int(row["frame_id"]), _robust_body_scale(row.get("pose") or {}))
        for row in frames_in_event
    ]
    valid_widths = [
        (frame_id, width)
        for frame_id, width in body_width_samples
        if width is not None
    ]
    body_width = median(width for _, width in valid_widths) if valid_widths else None
    torso_samples = [
        _torso_length(row.get("pose") or {})
        for row in frames_in_event
    ]
    valid_torsos = [t for t in torso_samples if t is not None]
    torso_length = median(valid_torsos) if valid_torsos else None
    contact_feature = features_by_frame.get(contact_frame) or {}
    contact_evidence_confidence = max(
        0.0,
        min(1.0, finite_number(contact_feature.get("contact_score")) or 0.0),
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
        windows=windows,
    )

    has_robust_turn = any(
        row.get("robust_shoulder_turn_deg") is not None
        for row in features_in_event
    )
    has_dual_view = has_robust_turn or any(
        row.get("takeback_depth_ratio") is not None
        for row in features_in_event
    )

    if has_robust_turn:
        shoulder_turn_metric = _median_feature_metric(
            features_by_frame,
            contact_frame,
            "robust_shoulder_turn_deg",
            pose_ratio,
            unit="image_plane_deg",
            confidence_cap=0.0,
            coach_eligible=False,
            observability="dual_view_anti_collapse",
            windows=windows,
        )
    else:
        shoulder_turn_metric = _median_feature_metric(
            features_by_frame,
            contact_frame,
            "shoulder_turn_deg",
            pose_ratio,
            unit="image_plane_deg",
            confidence_cap=0.45,
            coach_eligible=False,
            observability="absolute_image_orientation_only",
            exclusion_reason="absolute_projection_is_not_turn_magnitude",
            windows=windows,
        )

    takeback_depth_metric = _event_peak_or_median_feature(
        features_by_frame,
        start_frame,
        contact_frame,
        "takeback_depth_ratio",
        pose_ratio,
        unit="ratio",
        use_max=True,
        coach_eligible=False,
        confidence_cap=0.0,
        observability="dual_view_mirror_projection",
        windows=windows,
    )

    scapular_retraction_metric = _event_peak_or_median_feature(
        features_by_frame,
        start_frame,
        contact_frame,
        "scapular_retraction_ratio",
        pose_ratio,
        unit="ratio",
        use_max=True,
        coach_eligible=False,
        confidence_cap=0.0,
        observability="dual_view_mirror_projection",
        windows=windows,
    )

    ext = _calculate_extended_tier_biomechanics(
        features_in_event=features_in_event,
        start_frame=start_frame,
        contact_frame=contact_frame,
        end_frame=end_frame,
        body_width=body_width,
        fps=float(event.get("fps") or 25.0),
    )
    if any(row.get("kinematic_views") for row in frames):
        from kinematic_sequence import analyze_kinematic_sequence
        fps_val = float(event.get("fps") or 25.0)
        post_contact_sec = max(0.0, (end_frame - contact_frame) / max(1.0, fps_val))
        follow_through_win = round(min(0.28, max(0.20, post_contact_sec)), 2)
        ext["kinematic_sequence"] = analyze_kinematic_sequence(
            frames,
            contact_frame,
            fps_val,
            window_seconds=(-0.60, follow_through_win),
        )
    else:
        ext["kinematic_sequence"].update(
            evidence_confidence=0.0,
            cross_validation={"status": "legacy_single_view",
                              "reason": "independent_view_records_missing"},
            validation_status="unvalidated_2d_projection",
        )

    from osd_evidence import qualify_extended_observations
    # These measurements define their own source-time windows around contact.
    # Swing segmentation must not truncate their pre-contact evidence.
    qualify_extended_observations(ext, {**event, "contact_frame": contact_frame}, frames, features, body_width)

    result = {
        "schema_version": "dual_view_2d_v1" if has_dual_view else "single_view_2d_v2",
        "coordinate_space": "image_plane_normalized_by_body_width",
        "contact_frame": contact_frame,
        "reference_body_width_px": (
            round(float(body_width), 4) if body_width is not None else None
        ),
        "reference_torso_length_px": (
            round(float(torso_length), 4) if torso_length is not None else None
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
                windows=windows,
            ),
            "shoulder_turn": shoulder_turn_metric,
            "shoulder_turn_change": _shoulder_turn_change_metric(
                features_by_frame,
                start_frame,
                contact_frame,
                pose_ratio,
                windows=windows,
            ),
            "preparation_knee_flexion": _preparation_knee_flexion_metric(
                frames_by_id,
                start_frame,
                contact_frame,
                pose_ratio,
                windows=windows,
            ),
            "arm_extension": _median_feature_metric(
                features_by_frame,
                arm_reference_frame,
                "arm_extension_deg",
                pose_ratio,
                unit="deg_2d",
                observability=arm_observability,
                windows=windows,
            ),
            "contact_lateral_distance": _contact_position_metric(
                frames_by_id,
                features_by_frame,
                contact_frame,
                body_width,
                pose_ratio,
                contact_evidence_confidence,
                windows=windows,
            ),
            "weight_transfer": _movement_metric(
                frames_by_id,
                start_frame,
                contact_frame,
                body_width,
                pose_ratio,
                "screen_translation_is_not_true_weight_transfer",
                windows=windows,
            ),
            "balance_drift": _movement_metric(
                frames_by_id,
                contact_frame,
                early_recovery_frame,
                body_width,
                pose_ratio,
                "body_center_translation_is_not_balance_stability",
                windows=windows,
            ),
            "takeback_depth": takeback_depth_metric,
            "scapular_retraction": scapular_retraction_metric,
            "racket_head_speed": {
                "coach_eligible": False,
                "exclusion_reason": "unvalidated_projection_estimate",
                "value": ext["racket_head_speed"]["contact_px_s"],
                "confidence": ext["racket_head_speed"]["confidence"] if pose_ratio > 0 else 0.0,
                "unit": "px/s",
                "observability": ext["racket_head_speed"].get("observability", "image_box_center_speed"),
                "sample_count": 1 if ext["racket_head_speed"]["contact_px_s"] is not None else 0,
                "source_frames": [contact_frame-1, contact_frame] if ext["racket_head_speed"]["contact_px_s"] is not None else [],
                "contact_kmh": ext["racket_head_speed"].get("contact_kmh") if pose_ratio > 0 else None,
                "max_kmh": ext["racket_head_speed"].get("max_kmh") if pose_ratio > 0 else None,
            },
            "brush_angle": {
                "coach_eligible": False,
                "exclusion_reason": "unvalidated_projection_estimate",
                "value": ext["brush_angle"]["low_to_high_angle_deg"] if pose_ratio > 0 else None,
                "confidence": ext["brush_angle"]["confidence"] if pose_ratio > 0 else 0.0,
                "unit": "deg",
                "drop_depth_ratio": ext["brush_angle"]["drop_depth_ratio"] if pose_ratio > 0 else None,
            },
            "stance": {
                "coach_eligible": False,
                "exclusion_reason": "unvalidated_projection_estimate",
                "value": ext["stance"]["image_foot_line_angle_deg"],
                "unit": "deg_2d",
                "confidence": ext["stance"]["confidence"] if pose_ratio > 0 else 0.0,
            },
            "leg_drive": {
                "coach_eligible": False,
                "exclusion_reason": "unvalidated_projection_estimate",
                "value": ext["leg_drive"]["drive_ratio"] if pose_ratio > 0 else None,
                "confidence": ext["leg_drive"]["confidence"] if pose_ratio > 0 else 0.0,
                "unit": "ratio",
            },
            "kinematic_sequence": {
                "coach_eligible": False,
                "exclusion_reason": "unvalidated_projection_estimate",
                "value": ext["kinematic_sequence"]["sequence_quality"],
                "confidence": 0.0,
                "evidence_confidence": ext["kinematic_sequence"].get("evidence_confidence", 0),
                "details": ext["kinematic_sequence"],
                "racket_candidate_peak_frame": ext["kinematic_sequence"].get("racket_candidate_peak_frame"),
                "racket_candidate_peak_speed": ext["kinematic_sequence"].get("racket_candidate_peak_speed"),
                "candidate_latency_shoulder_to_racket_ms": ext["kinematic_sequence"].get("candidate_latency_shoulder_to_racket_ms"),
            },
            "swing_quality_score": {
                "value": ext["swing_quality_score"]["overall_score"] if pose_ratio > 0 else None,
                "confidence": 0.90 if pose_ratio > 0 else 0.0,
                "grade": ext["swing_quality_score"]["grade"] if pose_ratio > 0 else None,
                "sub_scores": ext["swing_quality_score"]["sub_scores"] if pose_ratio > 0 else {},
            },
        },
        "extended_biomechanics": ext,
        "swing_score": ext["swing_quality_score"]["overall_score"] if pose_ratio > 0 else None,
        "swing_grade": ext["swing_quality_score"]["grade"] if pose_ratio > 0 else None,
        "quality": {
            "pose_frame_ratio": round(pose_ratio, 4),
            "body_scale_frame_ratio": round(
                len(valid_widths) / max(1, len(frames_in_event)),
                4,
            ),
            "contact_evidence_confidence": round(contact_evidence_confidence, 4),
            "arm_extension_reference_frame": int(arm_reference_frame),
            "early_recovery_frame": early_recovery_frame,
            "body_window_policy": windows.around(contact_frame).evidence,
        },
    }

    for key in ("brush_angle", "stance", "leg_drive"):
        evidence = ext[key]["measurement_evidence"]
        result["metrics"][key].update(measurement_evidence=evidence,
            sample_count=evidence["sample_count"], source_frames=evidence["source_frames"])

    # Rise and direction have separate eligibility, including a qualified zero
    # rise when the chord endpoints coincide and no direction exists.
    rise_evidence = dict(ext['brush_angle']['measurement_evidence'])
    rise_evidence.update(rise_evidence.get('fields', {}).get('drop_depth_ratio', {}))
    result['metrics']['drop_depth_ratio'] = {
        'value': ext['brush_angle']['drop_depth_ratio'], 'unit': 'ratio',
        'coach_eligible': False, 'confidence': 0.0,
        'observability': 'image_plane_proxy_only',
        'exclusion_reason': 'unvalidated_projection_estimate',
        'measurement_evidence': rise_evidence,
        'source_frames': rise_evidence['source_frames'],
        'sample_count': rise_evidence['sample_count'],
    }

    sub_scores = extract_biomechanical_sub_scores(result) if pose_ratio > 0 else {}
    ext["swing_quality_score"]["sub_scores"] = sub_scores
    result["metrics"]["swing_quality_score"]["sub_scores"] = sub_scores

    from baseline_observations import baseline_profiles, normalized_view_trends
    baseline = baseline_profiles(frames, start_frame)
    result['experimental_baseline'] = baseline
    result['experimental_view_trends'] = normalized_view_trends(frames, baseline, start_frame, end_frame)
    from practice_scoring import attach_score
    attach_score({**event, "biomechanics": result})
    from metric_contracts import attach_metric_contracts
    arm_side = next((f.get('dominant_hand') for f in features_in_event
                     if f.get('dominant_hand') in ('left', 'right')), 'right')
    torso = {'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'}
    requirements = {
        'arm_extension': {f'{arm_side}_{joint}' for joint in ('shoulder', 'elbow', 'wrist')},
        'preparation_knee_flexion': {f'{side}_{joint}' for side in ('left', 'right') for joint in ('hip','knee','ankle')},
        'shoulder_turn': {'left_shoulder','right_shoulder'},
        'shoulder_turn_change': {'left_shoulder','right_shoulder'},
        'hip_shoulder_separation': torso,
        'weight_transfer': torso, 'balance_drift': torso, 'contact_lateral_distance': torso,
    }
    for key, joints in requirements.items():
        metric = result['metrics'].get(key)
        if metric is None: continue
        selected = metric.get('window_evidence', {}).get('source_frame_ids', [])
        rejected = [{'source_frame_id': row['frame_id'], 'joint': joint, 'reason': reason}
                    for row in frames_in_event if row['frame_id'] in selected
                    for joint, reason in row['pose_qualification']['rejected_points'].items() if joint in joints]
        rejected.extend({'source_frame_id': row['frame_id'], 'joint': None,
                         'reason': row['pose_qualification']['record_reason']}
                        for row in frames_in_event if row['frame_id'] in selected
                        and row['pose_qualification'].get('record_reason'))
        if rejected:
            metric['observation_qualification'] = {'policy_version': frames_in_event[0]['pose_qualification']['policy_version'],
                'rejected_points': rejected, 'accuracy_validated': False}
    attach_metric_contracts(result)
    return result


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
        bio = aggregate_event_biomechanics(
            event,
            frame_rows,
            feature_rows,
        )
        event["biomechanics"] = bio
        if "extended_biomechanics" in bio:
            event["extended_biomechanics"] = bio["extended_biomechanics"]
        if "swing_score" in bio:
            event["swing_score"] = bio["swing_score"]
        if "swing_grade" in bio:
            event["swing_grade"] = bio["swing_grade"]
        from practice_scoring import attach_score
        attach_score(event)
        from metric_contracts import attach_metric_contracts
        attach_metric_contracts(bio)
        enriched.append(event)
    return enriched
