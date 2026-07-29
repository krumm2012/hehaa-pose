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
        pose = frame.get("pose") or {}
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
        ball = _point(frame.get("ball"))
        racket = _racket_center(frame.get("rackets") or [])

        wrist_speed = _distance(wrist, prev.get("wrist")) or 0.0
        racket_speed = _distance(racket, prev.get("racket")) or 0.0
        ball_speed = _distance(ball, prev.get("ball")) or 0.0
        wrist_accel = wrist_speed - float(prev.get("wrist_speed", 0.0))
        racket_accel = racket_speed - float(prev.get("racket_speed", 0.0))

        body_center_x = None
        if left_shoulder is not None and right_shoulder is not None:
            body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0

        shoulder_line_angle = _vector_angle(left_shoulder, right_shoulder)
        hip_line_angle = _vector_angle(left_hip, right_hip)
        hip_shoulder_sep = None
        if shoulder_line_angle is not None and hip_line_angle is not None:
            hip_shoulder_sep = abs((shoulder_line_angle - hip_line_angle + 180.0) % 360.0 - 180.0)

        ball_racket_distance = _distance(ball, racket)
        contact_score = 0.0
        if ball_racket_distance is not None:
            contact_score = max(0.0, 1.0 - min(ball_racket_distance, 180.0) / 180.0)

        arm_extension = _angle(shoulder, elbow, wrist)
        feature = {
            "frame_id": frame_id,
            "timestamp": timestamp,
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
            "active_wrist_x_offset": round(wrist[0] - body_center_x, 4) if wrist and body_center_x is not None else None,
            "arm_extension_deg": (
                round(arm_extension, 4)
                if arm_extension is not None
                else _metric(metrics, "swing_motion", "arm_ext")
            ),
            "shoulder_turn_deg": _metric(metrics, "preparation", "shoulder_turn"),
            "hip_shoulder_sep_deg": round(hip_shoulder_sep, 4) if hip_shoulder_sep is not None else _metric(metrics, "power_indicators", "hip_shoulder_sep"),
        }
        features.append(feature)
        prev = {
            "wrist": wrist,
            "racket": racket,
            "ball": ball,
            "wrist_speed": wrist_speed,
            "racket_speed": racket_speed,
        }

    return features
