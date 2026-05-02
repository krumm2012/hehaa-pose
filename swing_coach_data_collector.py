#!/usr/bin/env python3
"""Build AI tennis-coach training data from swing event analysis outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from swing_event_analyzer import analyze_frame_records


Point = Tuple[float, float]


def _point(value) -> Optional[Point]:
    if isinstance(value, (list, tuple)) and len(value) >= 2 and value[0] is not None and value[1] is not None:
        return float(value[0]), float(value[1])
    return None


def _distance(a: Optional[Point], b: Optional[Point]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(p1: Optional[Point], p2: Optional[Point], p3: Optional[Point]) -> Optional[float]:
    if p1 is None or p2 is None or p3 is None:
        return None
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return None
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return round(math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2))))), 4)


def _vector_angle(a: Optional[Point], b: Optional[Point]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])), 4)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 4)


def _max(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    return round(max(valid), 4) if valid else None


def _feature_by_frame(features: List[Dict]) -> Dict[int, Dict]:
    return {int(feature["frame_id"]): feature for feature in features}


def _trace_by_frame(frame_trace: List[Dict]) -> Dict[int, Dict]:
    return {int(trace["frame"]): trace for trace in frame_trace}


def _frame_by_id(frames: List[Dict]) -> Dict[int, Dict]:
    return {int(frame.get("frame_id", idx)): frame for idx, frame in enumerate(frames)}


def _phase_frame(phase: str, traces: List[Dict]) -> Optional[int]:
    for trace in traces:
        if trace.get("phase") == phase:
            return int(trace["frame"])
    return None


def _phase_durations(traces: List[Dict]) -> Dict[str, int]:
    counts = Counter(trace.get("phase", "unknown") for trace in traces)
    return dict(sorted((str(k), int(v)) for k, v in counts.items()))


def _best_contact_frame(event: Dict, event_features: List[Dict]) -> Tuple[int, str, float]:
    scored = [f for f in event_features if f.get("contact_score") is not None]
    if scored:
        best = max(scored, key=lambda f: float(f.get("contact_score") or 0.0))
        if float(best.get("contact_score") or 0.0) > 0.0:
            return int(best["frame_id"]), "ball_racket_min_distance", float(best.get("contact_score") or 0.0)
    return int(event.get("peak_frame", event.get("start_frame", 0))), "peak_frame_fallback", 0.0


def _nearest_feature(features_by_frame: Dict[int, Dict], frame_id: int) -> Dict:
    if frame_id in features_by_frame:
        return features_by_frame[frame_id]
    if not features_by_frame:
        return {}
    nearest = min(features_by_frame.keys(), key=lambda fid: abs(fid - frame_id))
    return features_by_frame[nearest]


def _pose_at(frames_by_id: Dict[int, Dict], frame_id: int) -> Dict:
    return frames_by_id.get(frame_id, {}).get("pose") or {}


def _body_center(pose: Dict) -> Optional[Point]:
    left_shoulder = _point(pose.get("left_shoulder"))
    right_shoulder = _point(pose.get("right_shoulder"))
    left_hip = _point(pose.get("left_hip"))
    right_hip = _point(pose.get("right_hip"))
    candidates = []
    for a, b in [(left_shoulder, right_shoulder), (left_hip, right_hip)]:
        if a is not None and b is not None:
            candidates.append(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))
    if not candidates:
        return None
    return _mean([p[0] for p in candidates]), _mean([p[1] for p in candidates])


def _relative_point(point: Optional[Point], origin: Optional[Point]) -> Optional[Dict]:
    if point is None or origin is None:
        return None
    return {"x": round(point[0] - origin[0], 4), "y": round(point[1] - origin[1], 4)}


def _pose_angles(pose: Dict) -> Dict:
    return {
        "right_elbow_deg": _angle(_point(pose.get("right_shoulder")), _point(pose.get("right_elbow")), _point(pose.get("right_wrist"))),
        "left_elbow_deg": _angle(_point(pose.get("left_shoulder")), _point(pose.get("left_elbow")), _point(pose.get("left_wrist"))),
        "right_knee_deg": _angle(_point(pose.get("right_hip")), _point(pose.get("right_knee")), _point(pose.get("right_ankle"))),
        "left_knee_deg": _angle(_point(pose.get("left_hip")), _point(pose.get("left_knee")), _point(pose.get("left_ankle"))),
    }


def _stance_metrics(pose: Dict) -> Dict:
    left_ankle = _point(pose.get("left_ankle"))
    right_ankle = _point(pose.get("right_ankle"))
    left_hip = _point(pose.get("left_hip"))
    right_hip = _point(pose.get("right_hip"))
    stance_width = _distance(left_ankle, right_ankle)
    hip_width = _distance(left_hip, right_hip)
    return {
        "stance_width_px": round(stance_width, 4) if stance_width is not None else None,
        "stance_width_to_hip_ratio": round(stance_width / hip_width, 4) if stance_width is not None and hip_width else None,
    }


def _trajectory_angle(points: List[Optional[Point]]) -> Optional[float]:
    valid = [p for p in points if p is not None]
    if len(valid) < 2:
        return None
    return _vector_angle(valid[0], valid[-1])


def _shot_direction(points: List[Optional[Point]], min_dx: float = 12.0) -> Tuple[Optional[str], float]:
    valid = [p for p in points if p is not None]
    if len(valid) < 2:
        return None, 0.0
    dx = valid[-1][0] - valid[0][0]
    dy = valid[-1][1] - valid[0][1]
    travel = math.hypot(dx, dy)
    if travel < min_dx:
        return "screen_straight", 0.25
    if abs(dx) < min_dx:
        direction = "screen_straight"
    else:
        direction = "screen_right" if dx > 0 else "screen_left"
    confidence = min(1.0, len(valid) / 5.0) * min(1.0, travel / 120.0)
    return direction, round(confidence, 4)


def _bounce_candidate(features: List[Dict], contact_frame: int) -> Tuple[Optional[int], float, str]:
    post = [f for f in features if int(f["frame_id"]) > contact_frame and f.get("ball") is not None]
    if len(post) < 4:
        return None, 0.0, "insufficient_post_contact_ball_points"
    points = [(int(f["frame_id"]), _point(f.get("ball"))) for f in post]
    for (f0, p0), (f1, p1), (f2, p2) in zip(points, points[1:], points[2:]):
        if p0 is None or p1 is None or p2 is None:
            continue
        dy1 = p1[1] - p0[1]
        dy2 = p2[1] - p1[1]
        if dy1 > 2.0 and dy2 < -2.0:
            return f1, 0.45, "vertical_direction_reversal"
    return None, 0.0, "no_reliable_bounce_candidate"


def _low_to_high_ratio(points: List[Optional[Point]]) -> Optional[float]:
    valid = [p for p in points if p is not None]
    if len(valid) < 2:
        return None
    upward = 0
    comparable = 0
    for prev, curr in zip(valid, valid[1:]):
        dy = curr[1] - prev[1]
        if abs(dy) < 1.0:
            continue
        comparable += 1
        # Pixel y decreases as the racket moves upward.
        if dy < 0:
            upward += 1
    if comparable == 0:
        return 0.0
    return round(upward / comparable, 4)


def _swing_path_type(path_angle: Optional[float], low_to_high_ratio: Optional[float]) -> str:
    if low_to_high_ratio is None:
        return "unknown"
    if low_to_high_ratio >= 0.62:
        return "low_to_high"
    if low_to_high_ratio <= 0.38:
        return "high_to_low"
    if path_angle is not None and abs(path_angle) <= 18:
        return "flat"
    return "mixed"


def _window_features(features_by_frame: Dict[int, Dict], start: int, end: int) -> List[Dict]:
    return [features_by_frame[fid] for fid in range(start, end + 1) if fid in features_by_frame]


def _interpolate_point(features_by_frame: Dict[int, Dict], frame_id: int, key: str) -> Tuple[Optional[Point], str, float]:
    current = _point(features_by_frame.get(frame_id, {}).get(key))
    if current is not None:
        return current, "detected", 1.0

    before = []
    after = []
    for fid, feature in features_by_frame.items():
        point = _point(feature.get(key))
        if point is None:
            continue
        if fid < frame_id:
            before.append((fid, point))
        elif fid > frame_id:
            after.append((fid, point))
    if not before or not after:
        nearest = before[-1] if before else after[0] if after else None
        if nearest is None:
            return None, "missing", 0.0
        distance = abs(nearest[0] - frame_id)
        return nearest[1], "nearest", round(max(0.1, 1.0 - distance / 10.0), 4)

    left = max(before, key=lambda item: item[0])
    right = min(after, key=lambda item: item[0])
    span = right[0] - left[0]
    if span <= 0:
        return left[1], "nearest", 0.5
    ratio = (frame_id - left[0]) / span
    point = (left[1][0] + (right[1][0] - left[1][0]) * ratio, left[1][1] + (right[1][1] - left[1][1]) * ratio)
    confidence = max(0.1, 1.0 - min(abs(frame_id - left[0]), abs(right[0] - frame_id)) / 8.0)
    return (round(point[0], 4), round(point[1], 4)), "interpolated", round(confidence, 4)


def _score_from_ratio(value: Optional[float], low: float, high: float) -> Optional[float]:
    if value is None:
        return None
    if high == low:
        return None
    return round(max(0.0, min(1.0, (float(value) - low) / (high - low))), 4)


def _quality_scores(event: Dict, body: Dict, racket: Dict, ball: Dict, timing: Dict) -> Dict:
    contact_score = ball.get("contact_confidence")
    racket_speed = racket.get("racket_speed_at_contact") or racket.get("max_racket_speed")
    follow_frames = timing.get("phase_durations_frames", {}).get("follow_through", 0)
    preparation_frames = timing.get("phase_durations_frames", {}).get("backswing", 0)
    sep = body.get("hip_shoulder_separation_at_contact")

    scores = {
        "contact_score": round(float(contact_score), 4) if contact_score is not None else None,
        "racket_speed_score": _score_from_ratio(racket_speed, 5.0, 80.0),
        "preparation_score": _score_from_ratio(preparation_frames, 2.0, 12.0),
        "follow_through_score": _score_from_ratio(follow_frames, 4.0, 18.0),
        "power_transfer_score": _score_from_ratio(sep, 5.0, 35.0),
    }
    valid = [v for v in scores.values() if v is not None]
    scores["overall_score"] = round(sum(valid) / len(valid), 4) if valid else None
    scores["confidence"] = event.get("confidence")
    return scores


def _diagnosis_tags(scores: Dict, body: Dict, racket: Dict, ball: Dict, timing: Dict) -> List[str]:
    tags = []
    if (ball.get("contact_confidence") or 0.0) < 0.2:
        tags.append("low_contact_confidence")
    if (racket.get("max_racket_speed") or 0.0) < 10.0:
        tags.append("low_racket_speed")
    if (timing.get("phase_durations_frames", {}).get("follow_through", 0)) < 4:
        tags.append("short_follow_through")
    if body.get("contact_point_relative_to_body") is None:
        tags.append("missing_contact_body_reference")
    if body.get("hip_shoulder_separation_at_contact") is None:
        tags.append("missing_hip_shoulder_separation")
    return tags


def _unit_turn_quality(shoulder_turn: Optional[float]) -> str:
    if shoulder_turn is None:
        return "unknown"
    if shoulder_turn >= 95:
        return "strong"
    if shoulder_turn >= 75:
        return "adequate"
    return "limited"


def _stance_type(stance_ratio: Optional[float]) -> str:
    if stance_ratio is None:
        return "unknown"
    if stance_ratio >= 1.55:
        return "wide"
    if stance_ratio >= 1.05:
        return "neutral"
    return "narrow"


def _balance_state(contact_to_end_delta: Optional[Dict]) -> str:
    if not contact_to_end_delta:
        return "unknown"
    movement = math.hypot(float(contact_to_end_delta.get("x") or 0.0), float(contact_to_end_delta.get("y") or 0.0))
    if movement <= 35:
        return "stable"
    if movement <= 85:
        return "moving"
    return "unstable"


def _weight_transfer(start_to_contact_delta: Optional[Dict]) -> str:
    if not start_to_contact_delta:
        return "unknown"
    dx = float(start_to_contact_delta.get("x") or 0.0)
    dy = float(start_to_contact_delta.get("y") or 0.0)
    if abs(dx) < 10 and abs(dy) < 10:
        return "minimal"
    if abs(dx) >= abs(dy):
        return "lateral_left" if dx < 0 else "lateral_right"
    return "forward_screen_down" if dy > 0 else "backward_screen_up"


def _contact_too_close_to_body(relative_contact: Optional[Dict], two_hand_distance: Optional[float]) -> bool:
    if relative_contact is None:
        return False
    x_abs = abs(float(relative_contact.get("x") or 0.0))
    reference = float(two_hand_distance or 120.0)
    return x_abs < max(45.0, reference * 0.45)


def _late_contact(active_wrist_x_offset: Optional[float], stroke_type: Optional[str]) -> bool:
    if active_wrist_x_offset is None:
        return False
    # For the current camera convention, true right-hand forehand appears on
    # screen-left. A positive contact offset is usually late/crossed body.
    if stroke_type == "Forehand":
        return float(active_wrist_x_offset) > 20.0
    return False


def _event_frames(event: Dict, event_features: List[Dict], event_traces: List[Dict]) -> Dict:
    contact_frame, contact_source, contact_confidence = _best_contact_frame(event, event_features)
    phase_map = {trace["phase"]: int(trace["frame"]) for trace in event_traces if trace.get("phase")}
    return {
        "start": int(event["start_frame"]),
        "start_frame": int(event["start_frame"]),
        "preparation": _phase_frame("backswing", event_traces),
        "backswing_peak": phase_map.get("backswing"),
        "contact": contact_frame,
        "contact_frame": contact_frame,
        "contact_source": contact_source,
        "contact_confidence": round(contact_confidence, 4),
        "peak": int(event.get("peak_frame", contact_frame)),
        "peak_frame": int(event.get("peak_frame", contact_frame)),
        "follow_through_peak": _phase_frame("follow_through", event_traces),
        "end": int(event["end_frame"]),
        "end_frame": int(event["end_frame"]),
    }


def _ball_metrics(event_features: List[Dict], contact_frame: int, features_by_frame: Dict[int, Dict]) -> Dict:
    contact = _nearest_feature(features_by_frame, contact_frame)
    pre = _window_features(features_by_frame, contact_frame - 5, contact_frame - 1)
    post = _window_features(features_by_frame, contact_frame + 1, contact_frame + 5)
    event_balls = [f.get("ball") for f in event_features]
    shot_direction, shot_direction_confidence = _shot_direction([f.get("ball") for f in post])
    bounce_frame, bounce_confidence, bounce_source = _bounce_candidate(event_features, contact_frame)
    return {
        "contact_point": contact.get("ball"),
        "contact_confidence": contact.get("contact_score"),
        "ball_racket_distance_at_contact": contact.get("ball_racket_distance"),
        "ball_speed_at_contact": contact.get("ball_speed"),
        "incoming_ball_speed_avg": _mean(f.get("ball_speed") for f in pre),
        "outgoing_ball_speed_avg": _mean(f.get("ball_speed") for f in post),
        "incoming_angle_deg": _trajectory_angle([f.get("ball") for f in pre]),
        "outgoing_angle_deg": _trajectory_angle([f.get("ball") for f in post]),
        "ball_detection_frames": sum(1 for p in event_balls if p is not None),
        "trajectory_quality": round(sum(1 for p in event_balls if p is not None) / max(1, len(event_balls)), 4),
        "estimated_spin": None,
        "spin_confidence": None,
        "landing_point": None,
        "landing_zone": None,
        "net_clearance_px": None,
        "bounce_frame": bounce_frame,
        "bounce_confidence": bounce_confidence,
        "bounce_source": bounce_source,
        "shot_depth": None,
        "shot_direction": shot_direction,
        "shot_direction_confidence": shot_direction_confidence,
        "shot_direction_space": "screen" if shot_direction is not None else None,
    }


def _racket_metrics(event_features: List[Dict], contact_frame: int, peak_frame: int, features_by_frame: Dict[int, Dict]) -> Dict:
    contact = _nearest_feature(features_by_frame, contact_frame)
    peak = _nearest_feature(features_by_frame, peak_frame)
    detections = [f.get("racket_center") for f in event_features]
    path_angle = _trajectory_angle(detections)
    low_to_high = _low_to_high_ratio(detections)
    peak_center, peak_source, peak_confidence = _interpolate_point(features_by_frame, peak_frame, "racket_center")
    contact_center = _point(contact.get("racket_center"))
    active_wrist = _point(contact.get("wrist"))
    racket_lag = _distance(active_wrist, contact_center)
    lag_confidence = 0.0
    if racket_lag is not None:
        lag_confidence = min(1.0, 0.45 + round(sum(1 for p in detections if p is not None) / max(1, len(detections)), 4) * 0.55)
    return {
        "racket_center_at_contact": contact.get("racket_center"),
        "racket_center_at_peak": peak_center,
        "racket_center_at_peak_source": peak_source,
        "racket_center_at_peak_confidence": peak_confidence,
        "racket_speed_at_contact": contact.get("racket_speed"),
        "racket_speed_at_peak": peak.get("racket_speed"),
        "max_racket_speed": _max(f.get("racket_speed") for f in event_features),
        "max_racket_accel": _max(f.get("racket_accel") for f in event_features),
        "racket_path_angle_deg": path_angle,
        "racket_face_angle_deg": None,
        "racket_face_confidence": None,
        "swing_path_type": _swing_path_type(path_angle, low_to_high),
        "low_to_high_ratio": low_to_high,
        "racket_lag_at_contact": round(racket_lag, 4) if racket_lag is not None else None,
        "racket_lag_confidence": round(lag_confidence, 4),
        "racket_detection_frames": sum(1 for p in detections if p is not None),
        "racket_continuity_ratio": round(sum(1 for p in detections if p is not None) / max(1, len(detections)), 4),
    }


def _center_delta(frames_by_id: Dict[int, Dict], start_frame: int, end_frame: int) -> Optional[Dict]:
    start_center = _body_center(_pose_at(frames_by_id, start_frame))
    end_center = _body_center(_pose_at(frames_by_id, end_frame))
    return _relative_point(end_center, start_center)


def _body_metrics(frames_by_id: Dict[int, Dict], features_by_frame: Dict[int, Dict], start_frame: int, contact_frame: int, peak_frame: int, end_frame: int, stroke_type: Optional[str]) -> Dict:
    contact_feature = _nearest_feature(features_by_frame, contact_frame)
    peak_feature = _nearest_feature(features_by_frame, peak_frame)
    contact_pose = _pose_at(frames_by_id, contact_frame)
    peak_pose = _pose_at(frames_by_id, peak_frame)
    body_center = _body_center(contact_pose)
    contact_ball = _point(contact_feature.get("ball"))
    right_wrist = _point(contact_pose.get("right_wrist"))
    left_wrist = _point(contact_pose.get("left_wrist"))
    center_start_to_contact = _center_delta(frames_by_id, start_frame, contact_frame)
    center_contact_to_end = _center_delta(frames_by_id, contact_frame, end_frame)
    stance = _stance_metrics(contact_pose)
    relative_contact = _relative_point(contact_ball, body_center)
    return {
        "body_center_at_contact": body_center,
        "contact_point_relative_to_body": relative_contact,
        "right_wrist_relative_to_body": _relative_point(right_wrist, body_center),
        "left_wrist_relative_to_body": _relative_point(left_wrist, body_center),
        "active_wrist_x_offset_at_contact": contact_feature.get("active_wrist_x_offset"),
        "two_hand_distance_at_contact": contact_feature.get("two_hand_distance"),
        "arm_extension_at_contact": contact_feature.get("arm_extension_deg"),
        "arm_extension_at_peak": peak_feature.get("arm_extension_deg"),
        "shoulder_turn_at_contact": contact_feature.get("shoulder_turn_deg"),
        "hip_shoulder_separation_at_contact": contact_feature.get("hip_shoulder_sep_deg"),
        "pose_angles_at_contact": _pose_angles(contact_pose),
        "pose_angles_at_peak": _pose_angles(peak_pose),
        "center_movement_start_to_contact": center_start_to_contact,
        "center_movement_contact_to_end": center_contact_to_end,
        "weight_transfer": _weight_transfer(center_start_to_contact),
        "balance_state": _balance_state(center_contact_to_end),
        "stance_type": _stance_type(stance.get("stance_width_to_hip_ratio")),
        "unit_turn_quality": _unit_turn_quality(contact_feature.get("shoulder_turn_deg")),
        "contact_too_close_to_body": _contact_too_close_to_body(relative_contact, contact_feature.get("two_hand_distance")),
        "late_contact": _late_contact(contact_feature.get("active_wrist_x_offset"), stroke_type),
        **stance,
    }


def _timing_metrics(event: Dict, event_frames: Dict, event_traces: List[Dict], fps: float) -> Dict:
    def seconds(frames: Optional[int]) -> Optional[float]:
        if frames is None or fps <= 0:
            return None
        return round(frames / fps, 4)

    start = event_frames["start"]
    contact = event_frames["contact"]
    end = event_frames["end"]
    peak = event_frames["peak"]
    phase_counts = _phase_durations(event_traces)
    recovery_frames = phase_counts.get("ready", 0)
    duration = int(event["duration_frames"])
    start_to_contact = contact - start
    contact_to_end = end - contact
    tempo_balance = min(start_to_contact, contact_to_end) / max(1, max(start_to_contact, contact_to_end))
    return {
        "duration_frames": duration,
        "duration_seconds": seconds(duration),
        "start_to_contact_frames": contact - start,
        "contact_to_end_frames": end - contact,
        "peak_to_contact_offset_frames": contact - peak,
        "phase_durations_frames": phase_counts,
        "start_to_contact_seconds": seconds(contact - start),
        "contact_to_end_seconds": seconds(end - contact),
        "recovery_time_frames": recovery_frames,
        "recovery_time_seconds": seconds(recovery_frames),
        "preparation_timing_quality": "early" if start_to_contact > duration * 0.55 else "quick" if start_to_contact < duration * 0.25 else "balanced",
        "tempo_consistency": round(tempo_balance, 4),
    }


def _missing_paths(value, prefix: str = "") -> List[str]:
    missing = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            missing.extend(_missing_paths(child, child_prefix))
    elif value is None:
        missing.append(prefix)
    return missing


def _data_quality(event_features: List[Dict], ball: Dict, racket: Dict, body: Dict, timing: Dict) -> Dict:
    fields = {
        "ball": ball,
        "racket": racket,
        "body": body,
        "timing": timing,
    }
    missing = []
    for category, payload in fields.items():
        missing.extend(_missing_paths(payload, category))
    available_count = sum(1 for f in event_features if f.get("has_pose"))
    return {
        "pose_frame_ratio": round(available_count / max(1, len(event_features)), 4),
        "ball_trajectory_quality": ball.get("trajectory_quality"),
        "racket_continuity_ratio": racket.get("racket_continuity_ratio"),
        "missing_fields": sorted(missing),
        "notes": [
            "Null fields are reserved for future sensors, camera calibration, or multi-view estimation.",
            "Current dataset is generated from single-view pose, ball, and racket detections.",
        ],
    }


def _event_trace_rows(event: Dict, traces_by_frame: Dict[int, Dict], features_by_frame: Dict[int, Dict]) -> List[Dict]:
    rows = []
    for frame_id in range(int(event["start_frame"]), int(event["end_frame"]) + 1):
        trace = traces_by_frame.get(frame_id, {})
        feature = features_by_frame.get(frame_id, {})
        rows.append(
            {
                "frame_id": frame_id,
                "phase": trace.get("phase"),
                "motion_energy": trace.get("motion_energy"),
                "raw_swing_type": trace.get("raw_swing_type", feature.get("raw_swing_type")),
                "wrist_speed": feature.get("wrist_speed"),
                "racket_speed": feature.get("racket_speed"),
                "ball_speed": feature.get("ball_speed"),
                "ball_racket_distance": feature.get("ball_racket_distance"),
                "contact_score": feature.get("contact_score"),
                "two_hand_distance": feature.get("two_hand_distance"),
                "active_wrist_x_offset": feature.get("active_wrist_x_offset"),
            }
        )
    return rows


def build_coach_dataset(frame_data: Dict, event_analysis: Dict) -> Dict:
    """Return a coach-oriented event dataset from frame and event analysis JSON."""
    frames = frame_data.get("frames", [])
    video_info = frame_data.get("video_info") or {}
    fps = float(video_info.get("fps") or 25.0)
    features = event_analysis.get("features")
    if not features:
        features = analyze_frame_records(frames)["features"]
    features_by_frame = _feature_by_frame(features)
    traces_by_frame = _trace_by_frame(event_analysis.get("frame_trace", []))
    frames_by_id = _frame_by_id(frames)

    coach_events = []
    for event in event_analysis.get("events", []):
        event_features = _window_features(features_by_frame, int(event["start_frame"]), int(event["end_frame"]))
        event_traces = [traces_by_frame[fid] for fid in range(int(event["start_frame"]), int(event["end_frame"]) + 1) if fid in traces_by_frame]
        frame_markers = _event_frames(event, event_features, event_traces)
        contact_frame = int(frame_markers["contact"])
        peak_frame = int(frame_markers["peak"])
        ball = _ball_metrics(event_features, contact_frame, features_by_frame)
        racket = _racket_metrics(event_features, contact_frame, peak_frame, features_by_frame)
        body = _body_metrics(
            frames_by_id,
            features_by_frame,
            int(event["start_frame"]),
            contact_frame,
            peak_frame,
            int(event["end_frame"]),
            event.get("stroke_type"),
        )
        timing = _timing_metrics(event, frame_markers, event_traces, fps)
        scores = _quality_scores(event, body, racket, ball, timing)
        data_quality = _data_quality(event_features, ball, racket, body, timing)
        coach_events.append(
            {
                "event_id": int(event["event_id"]),
                "stroke_type": event.get("stroke_type"),
                "confidence": event.get("confidence"),
                "is_valid_hit": True,
                "is_shadow_swing": ball.get("trajectory_quality", 0.0) == 0.0,
                "frames": frame_markers,
                "ball": ball,
                "racket": racket,
                "body": body,
                "timing": timing,
                "scores": scores,
                "diagnosis_tags": _diagnosis_tags(scores, body, racket, ball, timing),
                "data_quality": data_quality,
                "classification_evidence": event.get("evidence", {}),
                "frame_trace": _event_trace_rows(event, traces_by_frame, features_by_frame),
            }
        )

    return {
        "metadata": {
            "schema_version": "coach_dataset_v1.1",
            "video_path": video_info.get("path"),
            "fps": fps,
            "resolution": video_info.get("resolution"),
            "total_frames": len(frames),
            "source": {
                "frame_json": None,
                "event_analysis": "swing_event_analysis",
            },
        },
        "summary": {
            "event_count": len(coach_events),
            "stroke_type_counts": dict(sorted(Counter(event["stroke_type"] for event in coach_events).items())),
        },
        "events": coach_events,
    }


def default_event_analysis_path(frame_json_path: str) -> str:
    path = Path(frame_json_path)
    return str(path.with_name(f"{path.stem}_swing_events.json"))


def default_coach_output_path(frame_json_path: str) -> str:
    path = Path(frame_json_path)
    return str(path.with_name(f"{path.stem}_coach_dataset.json"))


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_coach_dataset(dataset: Dict, output_path: str, frame_json_path: Optional[str] = None) -> None:
    if frame_json_path:
        dataset.setdefault("metadata", {}).setdefault("source", {})["frame_json"] = frame_json_path
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AI tennis coach dataset JSON from swing event analysis.")
    parser.add_argument("frame_json", help="Per-frame pipeline JSON, e.g. data/output_video.json")
    parser.add_argument("--event-json", help="Swing event analysis JSON. Defaults to <frame_json_stem>_swing_events.json")
    parser.add_argument("--output-json", help="Coach dataset JSON path. Defaults to <frame_json_stem>_coach_dataset.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    event_json = args.event_json or default_event_analysis_path(args.frame_json)
    output_json = args.output_json or default_coach_output_path(args.frame_json)
    frame_data = load_json(args.frame_json)
    event_analysis = load_json(event_json)
    dataset = build_coach_dataset(frame_data, event_analysis)
    write_coach_dataset(dataset, output_json, frame_json_path=args.frame_json)
    print(f"events={dataset['summary']['event_count']} {dataset['summary']['stroke_type_counts']}")
    print(f"json={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
