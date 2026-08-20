"""Fast tennis-ball candidate ranking helpers."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence


def _position(candidate: Dict) -> Optional[List[float]]:
    pos = candidate.get("position") or candidate.get("ball") or candidate.get("coords")
    if not pos or len(pos) < 2:
        return None
    return [float(pos[0]), float(pos[1])]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _primary_racket_center(racket_detections: Optional[Iterable[Dict]]) -> Optional[List[float]]:
    rackets = [r for r in racket_detections or [] if isinstance(r, dict) and r.get("box")]
    if not rackets:
        return None
    racket = max(rackets, key=lambda r: float(r.get("confidence", 0.0)))
    box = racket.get("box") or []
    if len(box) < 4:
        return None
    return [(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0]


def select_ball_candidate(
    ball_detections: List[Dict],
    previous_position: Optional[Sequence[float]] = None,
    previous_velocity: Optional[Sequence[float]] = None,
    racket_detections: Optional[Iterable[Dict]] = None,
    config: Optional[Dict] = None,
) -> Optional[Dict]:
    """Select the active ball without extra model inference.

    The detector can see both the real ball and mirror reflections. Confidence alone is
    brittle, so ranking also rewards temporal continuity and proximity to the primary
    racket while lightly penalizing the top mirror band.
    """
    if not ball_detections:
        return None

    config = config or {}
    continuity_weight = float(config.get("ball_continuity_weight", 0.45))
    continuity_distance = max(1.0, float(config.get("ball_continuity_distance_px", 180.0)))
    velocity_weight = float(config.get("ball_velocity_prediction_weight", 0.55))
    velocity_distance = max(1.0, float(config.get("ball_velocity_prediction_distance_px", 120.0)))
    racket_weight = float(config.get("ball_racket_proximity_weight", 0.20))
    racket_distance = max(1.0, float(config.get("ball_racket_proximity_distance_px", 360.0)))
    top_penalty = float(config.get("ball_top_mirror_penalty", 0.25))
    mirror_min_y_ratio = float(config.get("ball_mirror_min_y_ratio", 0.03))
    frame_height = config.get("frame_height")
    primary_racket = _primary_racket_center(racket_detections)

    def score(candidate: Dict) -> float:
        pos = _position(candidate)
        if pos is None:
            return float("-inf")

        value = float(candidate.get("confidence", 0.0))
        if previous_position is not None:
            dist = _distance(pos, previous_position)
            value += continuity_weight * max(0.0, 1.0 - dist / continuity_distance)
            if previous_velocity is not None and len(previous_velocity) >= 2:
                predicted = [
                    float(previous_position[0]) + float(previous_velocity[0]),
                    float(previous_position[1]) + float(previous_velocity[1]),
                ]
                dist_pred = _distance(pos, predicted)
                value += velocity_weight * max(0.0, 1.0 - dist_pred / velocity_distance)

        if primary_racket is not None:
            dist = _distance(pos, primary_racket)
            value += racket_weight * max(0.0, 1.0 - dist / racket_distance)

        if frame_height:
            min_play_y = float(frame_height) * mirror_min_y_ratio
            if pos[1] < min_play_y:
                value -= top_penalty

        return value

    return max(ball_detections, key=score)
