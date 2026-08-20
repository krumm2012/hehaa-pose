"""Fast tennis-racket candidate ranking helpers."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


def racket_center(racket: Dict) -> Optional[List[float]]:
    box = racket.get("box") if isinstance(racket, dict) else None
    if not box or len(box) < 4:
        return None
    return [(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def select_racket_candidate(
    racket_detections: List[Dict],
    ball_position: Optional[Sequence[float]] = None,
    previous_center: Optional[Sequence[float]] = None,
    config: Optional[Dict] = None,
) -> Optional[Dict]:
    """Select the active racket from detector candidates without extra inference."""
    if not racket_detections:
        return None

    config = config or {}
    ball_weight = float(config.get("racket_ball_proximity_weight", 0.35))
    ball_distance = max(1.0, float(config.get("racket_ball_proximity_distance_px", 460.0)))
    continuity_weight = float(config.get("racket_continuity_weight", 0.25))
    continuity_distance = max(1.0, float(config.get("racket_continuity_distance_px", 260.0)))
    top_penalty = float(config.get("racket_top_mirror_penalty", 0.30))
    mirror_min_y_ratio = float(config.get("racket_mirror_min_y_ratio", 0.18))
    frame_height = config.get("frame_height")

    def score(candidate: Dict) -> float:
        center = racket_center(candidate)
        if center is None:
            return float("-inf")

        value = float(candidate.get("confidence", 0.0))
        if ball_position is not None:
            dist = _distance(center, ball_position)
            value += ball_weight * max(0.0, 1.0 - dist / ball_distance)

        if previous_center is not None:
            dist = _distance(center, previous_center)
            value += continuity_weight * max(0.0, 1.0 - dist / continuity_distance)

        if frame_height:
            min_play_y = float(frame_height) * mirror_min_y_ratio
            if center[1] < min_play_y:
                value -= top_penalty

        return value

    return max(racket_detections, key=score)
