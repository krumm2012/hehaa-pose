#!/usr/bin/env python3
"""
Frame-level motion feature extraction for tennis swing analysis.

This module consumes the JSON frame records already produced by the
existing video pipeline. It does not run models; it derives richer
temporal features from pose, ball, and racket detections.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Tuple


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
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 == 0 or n2 == 0:
        return None
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))


def _vector_angle(a: Optional[Point], b: Optional[Point]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _parse_number(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def _racket_center(rackets: Iterable[Dict]) -> Optional[Point]:
    best = None
    for racket in rackets or []:
        if not isinstance(racket, dict) or not racket.get("box"):
            continue
        if best is None or racket.get("confidence", 0.0) > best.get("confidence", 0.0):
            best = racket
    if not best:
        return None
    x1, y1, x2, y2 = [float(x) for x in best["box"]]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _metric(metrics: Dict, category: str, key: str) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    return _parse_number((metrics.get(category) or {}).get(key))


def extract_motion_features(
    frames: List[Dict],
    dominant_hand: str = "right",
) -> List[Dict]:
    """Return one feature dictionary per frame.

    The output deliberately keeps both raw detections and derived metrics so
    later stages can be audited frame by frame.
    """
    features: List[Dict] = []
    prev = {}

    wrist_key = "right_wrist" if dominant_hand == "right" else "left_wrist"
    off_wrist_key = "left_wrist" if dominant_hand == "right" else "right_wrist"
    shoulder_key = "right_shoulder" if dominant_hand == "right" else "left_shoulder"
    elbow_key = "right_elbow" if dominant_hand == "right" else "left_elbow"

    for frame in frames:
        pose = frame.get("healed_pose") or frame.get("pose") or {}
        metrics = frame.get("metrics") or {}
        frame_id = int(frame.get("frame_id", len(features)))
        timestamp = float(frame.get("timestamp", frame_id))

        wrist = _point(pose.get(wrist_key))
        off_wrist = _point(pose.get(off_wrist_key))
        shoulder = _point(pose.get(shoulder_key))
        elbow = _point(pose.get(elbow_key))
        left_shoulder = _point(pose.get("left_shoulder"))
        right_shoulder = _point(pose.get("right_shoulder"))
        left_hip = _point(pose.get("left_hip"))
        right_hip = _point(pose.get("right_hip"))
        left_knee = _point(pose.get("left_knee"))
        right_knee = _point(pose.get("right_knee"))
        left_ankle = _point(pose.get("left_ankle"))
        right_ankle = _point(pose.get("right_ankle"))
        ball = _point(frame.get("ball"))
        racket = _point(frame.get("racket"))
        if racket is None:
            racket = _racket_center(frame.get("rackets") or [])

        wrist_speed = _distance(wrist, prev.get("wrist")) or 0.0
        racket_speed = _distance(racket, prev.get("racket")) or 0.0
        # 异常跳变抑制：防止单帧误检瞬移造成虚假超高速
        if racket_speed > 250.0:
            racket_speed = 0.0
        ball_speed = _distance(ball, prev.get("ball")) or 0.0
        wrist_accel = wrist_speed - float(prev.get("wrist_speed", 0.0))
        racket_accel = racket_speed - float(prev.get("racket_speed", 0.0))

        body_center_x = None
        shoulder_width = _distance(left_shoulder, right_shoulder)
        if left_shoulder is not None and right_shoulder is not None:
            body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0

        shoulder_line_angle = _vector_angle(left_shoulder, right_shoulder)
        hip_line_angle = _vector_angle(left_hip, right_hip)
        hip_shoulder_sep = None
        if shoulder_line_angle is not None and hip_line_angle is not None:
            hip_shoulder_sep = abs((shoulder_line_angle - hip_line_angle + 180.0) % 360.0 - 180.0)

        # 步法站位分类 (Stance Type Classification: Open vs Semi-Open vs Closed)
        stance_angle = None
        stance_type = "Unknown"
        ref_p1 = left_ankle or left_knee or left_hip
        ref_p2 = right_ankle or right_knee or right_hip
        if ref_p1 is not None and ref_p2 is not None:
            dx = abs(ref_p2[0] - ref_p1[0])
            dy = abs(ref_p2[1] - ref_p1[1])
            stance_angle = math.degrees(math.atan2(dy, dx + 1e-5))
            if stance_angle < 25.0:
                stance_type = "Open Stance"
            elif stance_angle < 55.0:
                stance_type = "Semi-Open Stance"
            else:
                stance_type = "Closed Stance"

        # 身体质心垂直位置 (Vertical Hip/COM Position)
        hip_vertical_pos = None
        if left_hip is not None and right_hip is not None:
            hip_vertical_pos = (left_hip[1] + right_hip[1]) / 2.0

        # 动力链角速度 (Hip & Shoulder Angular Rotation Speed)
        prev_hip_angle = prev.get("hip_line_angle")
        hip_rotation_speed = (
            abs(hip_line_angle - prev_hip_angle)
            if hip_line_angle is not None and prev_hip_angle is not None
            else 0.0
        )
        prev_shoulder_angle = prev.get("shoulder_line_angle")
        shoulder_rotation_speed = (
            abs(shoulder_line_angle - prev_shoulder_angle)
            if shoulder_line_angle is not None and prev_shoulder_angle is not None
            else 0.0
        )

        # 真实拍头速度换算 (Racket Head Speed in km/h & m/s)
        dt = timestamp - float(prev.get("timestamp", timestamp - 0.04))
        if dt <= 0.001:
            dt = 0.04  # 默认 25 FPS
        ppm = (shoulder_width / 0.42) if shoulder_width and shoulder_width > 15.0 else 320.0
        racket_speed_mps = (racket_speed / ppm) / dt
        racket_head_speed_kmh = min(180.0, max(0.0, racket_speed_mps * 3.6))

        ball_racket_distance = _distance(ball, racket)
        ball_wrist_distance = _distance(ball, wrist)
        contact_score = 0.0
        if ball_racket_distance is not None:
            contact_score = max(0.0, 1.0 - min(ball_racket_distance, 180.0) / 180.0)
        elif ball_wrist_distance is not None:
            contact_score = max(0.0, 1.0 - min(ball_wrist_distance, 220.0) / 220.0)

        dv_biomech = frame.get("dual_view_biomechanics") or frame.get("dual_view") or {}
        robust_turn = None
        takeback_ratio = None
        scapular_ratio = None
        dv_stroke_type = None
        dv_is_two_handed = None
        dv_contact_valid = None

        if dv_biomech:
            st = dv_biomech.get("shoulder_turn") or {}
            robust_turn = st.get("shoulder_turn_deg")
            tb = dv_biomech.get("takeback_depth") or {}
            takeback_ratio = tb.get("takeback_depth_ratio")
            sc = dv_biomech.get("scapular_retraction") or {}
            scapular_ratio = sc.get("scapular_retraction_ratio")
            sc_cls = dv_biomech.get("shot_classification") or {}
            dv_stroke_type = sc_cls.get("stroke_type")
            dv_is_two_handed = sc_cls.get("is_two_handed")
            cd_gate = dv_biomech.get("contact_distance_gate") or {}
            dv_contact_valid = cd_gate.get("is_valid_contact")

        arm_extension = _angle(shoulder, elbow, wrist)
        feature = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "dominant_hand": dominant_hand,
            "raw_swing_type": frame.get("swing_type", "No Pose"),
            "has_pose": bool(pose),
            "has_ball": ball is not None,
            "has_racket": racket is not None,
            "wrist": wrist,
            "off_wrist": off_wrist,
            "racket_center": racket,
            "ball": ball,
            "detection_diagnostics": frame.get("detection_diagnostics") or {},
            "wrist_speed": round(wrist_speed, 4),
            "wrist_accel": round(wrist_accel, 4),
            "racket_speed": round(racket_speed, 4),
            "racket_accel": round(racket_accel, 4),
            "ball_speed": round(ball_speed, 4),
            "ball_racket_distance": round(ball_racket_distance, 4) if ball_racket_distance is not None else None,
            "contact_score": round(contact_score, 4),
            "two_hand_distance": round(_distance(wrist, off_wrist), 4) if wrist and off_wrist else None,
            "two_hand_distance_body_width": (
                round(float(_distance(wrist, off_wrist)) / shoulder_width, 4)
                if wrist and off_wrist and shoulder_width
                else None
            ),
            "active_wrist_x_offset": round(wrist[0] - body_center_x, 4) if wrist and body_center_x is not None else None,
            "active_wrist_x_offset_body_width": (
                round(float(wrist[0] - body_center_x) / shoulder_width, 4)
                if wrist and body_center_x is not None and shoulder_width
                else None
            ),
            # COCO keypoints are anatomical.  When the player's right
            # shoulder projects left of the left shoulder, the player faces
            # the camera; the opposite ordering means the camera is behind.
            "camera_facing_score": (
                round(float(right_shoulder[0] - left_shoulder[0]) / shoulder_width, 4)
                if left_shoulder and right_shoulder and shoulder_width
                else None
            ),
            "shoulder_width_px": round(shoulder_width, 4) if shoulder_width else None,
            "arm_extension_deg": (
                round(arm_extension, 4)
                if arm_extension is not None
                else _metric(metrics, "swing_motion", "arm_ext")
            ),
            "shoulder_turn_deg": (
                round(float(robust_turn), 4)
                if robust_turn is not None
                else _metric(metrics, "preparation", "shoulder_turn")
            ),
            "robust_shoulder_turn_deg": (
                round(float(robust_turn), 4) if robust_turn is not None else None
            ),
            "takeback_depth_ratio": (
                round(float(takeback_ratio), 4) if takeback_ratio is not None else None
            ),
            "scapular_retraction_ratio": (
                round(float(scapular_ratio), 4) if scapular_ratio is not None else None
            ),
            "dual_view_stroke_type": dv_stroke_type,
            "dual_view_is_two_handed": dv_is_two_handed,
            "dual_view_contact_valid": dv_contact_valid,
            "hip_shoulder_sep_deg": round(hip_shoulder_sep, 4) if hip_shoulder_sep is not None else _metric(metrics, "power_indicators", "hip_shoulder_sep"),
            "racket_head_speed_kmh": round(racket_head_speed_kmh, 1),
            "racket_speed_mps": round(racket_speed_mps, 2),
            "stance_angle": round(stance_angle, 1) if stance_angle is not None else None,
            "stance_type": stance_type,
            "hip_vertical_pos": round(hip_vertical_pos, 2) if hip_vertical_pos is not None else None,
            "hip_rotation_speed": round(hip_rotation_speed, 2),
            "shoulder_rotation_speed": round(shoulder_rotation_speed, 2),
        }
        features.append(feature)
        prev = {
            "wrist": wrist,
            "racket": racket,
            "ball": ball,
            "timestamp": timestamp,
            "wrist_speed": wrist_speed,
            "racket_speed": racket_speed,
            "hip_line_angle": hip_line_angle,
            "shoulder_line_angle": shoulder_line_angle,
        }

    return features
