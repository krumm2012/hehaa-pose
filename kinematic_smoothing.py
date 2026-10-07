"""Biomechanical kinematic smoothing and acceleration limits for tennis swing analysis."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]

# Human biomechanical ceilings for tennis swing:
# 1. Human shoulder/arm maximum linear/tangential acceleration: ~300 m/s^2 (approx 30g peak during whip snap)
MAX_TANGENTIAL_ACCEL_MPS2 = 300.0
# 2. Maximum centripetal acceleration ac = v^2 / R: with swing radius R ~ 1.2m and peak racket speed ~35 m/s (~126 km/h)
#    ac = 35^2 / 1.2 ≈ 1020 m/s^2
MAX_CENTRIPETAL_ACCEL_MPS2 = 1050.0
DEFAULT_SWING_RADIUS_M = 1.20
# 3. Maximum plausible tennis ball speed: ~270 km/h (world record is ~263 km/h)
MAX_BALL_SPEED_KMH = 270.0
# 4. Maximum plausible racket head speed: ~165 km/h
MAX_RACKET_SPEED_KMH = 165.0
# 5. Anatomical wrist-to-racket center distance bounds in pixels (for normal resolutions)
MIN_WRIST_RACKET_DIST_PX = 15.0
MAX_WRIST_RACKET_DIST_PX = 180.0


def clamp_velocity_step(
    v_prev: Optional[float],
    v_curr: float,
    dt: float,
    max_accel: float = MAX_TANGENTIAL_ACCEL_MPS2,
) -> Tuple[float, bool]:
    """Clamp non-physical velocity jump caused by single-frame detection jitter.

    Returns (clamped_v, was_clamped).
    """
    if v_prev is None or dt <= 0.0 or not math.isfinite(v_curr):
        return v_curr, False
    max_delta = max_accel * dt
    delta = v_curr - v_prev
    if delta > max_delta:
        return round(v_prev + max_delta, 3), True
    if delta < -max_delta:
        return round(max(0.0, v_prev - max_delta), 3), True
    return v_curr, False


def clamp_centripetal_speed(
    speed_mps: float,
    radius_m: float = DEFAULT_SWING_RADIUS_M,
    max_centripetal_accel: float = MAX_CENTRIPETAL_ACCEL_MPS2,
) -> Tuple[float, bool]:
    """Clamp linear speed if it implies a physiologically impossible centripetal acceleration."""
    if not math.isfinite(speed_mps) or speed_mps <= 0.0:
        return speed_mps, False
    r = max(0.4, float(radius_m or DEFAULT_SWING_RADIUS_M))
    max_v = math.sqrt(max_centripetal_accel * r)
    if speed_mps > max_v:
        return round(max_v, 3), True
    return speed_mps, False


def wrist_racket_geometric_consistency(
    racket_point: Optional[Point],
    wrist_point: Optional[Point],
    prev_racket_point: Optional[Point] = None,
    prev_wrist_point: Optional[Point] = None,
    min_dist: float = MIN_WRIST_RACKET_DIST_PX,
    max_dist: float = MAX_WRIST_RACKET_DIST_PX,
) -> Tuple[Optional[Point], bool]:
    """Validate and filter racket center position using anatomical wrist anchor.

    If the racket position leaps outside physically plausible human arm reach,
    smooth it using the wrist translation vector.
    Returns (corrected_racket_point, was_adjusted).
    """
    if racket_point is None:
        return None, False
    if wrist_point is None:
        return racket_point, False

    dist = math.hypot(racket_point[0] - wrist_point[0], racket_point[1] - wrist_point[1])
    if dist < min_dist or dist > max_dist:
        # Detected racket center drifted outside physical wrist connection cone
        if prev_racket_point is not None and prev_wrist_point is not None:
            # Predict racket point following wrist translation
            dx_w = wrist_point[0] - prev_wrist_point[0]
            dy_w = wrist_point[1] - prev_wrist_point[1]
            predicted = (round(prev_racket_point[0] + dx_w, 2), round(prev_racket_point[1] + dy_w, 2))
            return predicted, True
        # Clamp distance along the vector
        if dist > 0.001:
            scale = max_dist / dist
            clamped = (
                round(wrist_point[0] + (racket_point[0] - wrist_point[0]) * scale, 2),
                round(wrist_point[1] + (racket_point[1] - wrist_point[1]) * scale, 2),
            )
            return clamped, True
    return racket_point, False
