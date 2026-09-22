"""Camera-calibrated mirror evidence with conservative trajectory exemptions."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence


class MirrorBallFilter:
    """Assess candidates in full-frame coordinates; shadow mode never drops them."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.mode = config.get("mode", "off")
        if self.mode not in {"off", "shadow", "enforce"}:
            raise ValueError("mirror_ball_filter.mode must be off, shadow or enforce")
        self.polygon = config.get("polygon", [])
        self.max_size_ratio = float(config.get("max_size_height_ratio", 0.0125))
        self.margin_ratio = float(config.get("boundary_margin_height_ratio", 0.01))
        if self.mode != "off":
            if len(self.polygon) < 3 or any(
                len(p) != 2 or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in p)
                for p in self.polygon
            ):
                raise ValueError("mirror polygon requires >=3 normalized [x, y] vertices")
            if not 0 < self.max_size_ratio < 1 or not 0 <= self.margin_ratio < 1:
                raise ValueError("mirror size/margin ratios must be finite and in range")

    @staticmethod
    def _distance(a: Sequence[float], b: Sequence[float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _inside(self, point, width, height):
        """Return inclusion and distance from the nearest polygon edge, in pixels."""
        vertices = [(float(x) * width, float(y) * height) for x, y in self.polygon]
        x, y = point
        inside, distance = False, float("inf")
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            length2 = dx * dx + dy * dy
            t = max(0., min(1., ((x-a[0])*dx + (y-a[1])*dy) / length2)) if length2 else 0.
            distance = min(distance, math.hypot(x-a[0]-t*dx, y-a[1]-t*dy))
            if (a[1] > y) != (b[1] > y):
                if x < (b[0]-a[0]) * (y-a[1]) / (b[1]-a[1]) + a[0]:
                    inside = not inside
        return inside, distance

    def assess(self, candidate, *, frame_width, frame_height,
               previous_position=None, previous_velocity=None, missed_frames=0,
               continuity_distance=144., prediction_distance=120.,
               racket_centers=(), racket_distance=80.):
        result = {"mode": self.mode, "suspect": False, "would_reject": False,
                  "rejected": False, "reason": "disabled"}
        if self.mode == "off":
            return result
        if not frame_width or not frame_height or frame_width <= 0 or frame_height <= 0:
            return {**result, "reason": "missing_frame_size"}
        position, box = candidate.get("position"), candidate.get("box")
        try:
            if position is None or len(position) != 2 or box is None or len(box) != 4:
                return {**result, "reason": "missing_geometry"}
            if not all(math.isfinite(float(v)) for v in [*position, *box]):
                return {**result, "reason": "missing_geometry"}
            width, height = float(box[2])-float(box[0]), float(box[3])-float(box[1])
            if min(width, height) <= 0:
                return {**result, "reason": "missing_geometry"}
        except (TypeError, ValueError):
            return {**result, "reason": "missing_geometry"}
        inside, edge = self._inside(position, frame_width, frame_height)
        # The whole box must be comfortably inside the mirror. Blur elongation
        # increases max(width, height), so it cannot accidentally look small.
        margin = self.margin_ratio * frame_height + math.hypot(width, height) / 2
        result.update(size_px=max(width, height), edge_distance_px=edge)
        if not inside or edge <= margin:
            return {**result, "reason": "outside_or_boundary"}
        if max(width, height) > self.max_size_ratio * frame_height:
            return {**result, "reason": "large_enough"}
        result["suspect"] = True
        if previous_position is not None:
            if self._distance(position, previous_position) <= continuity_distance:
                return {**result, "reason": "trajectory_protected"}
            if previous_velocity is not None:
                predicted = [previous_position[i] + previous_velocity[i] * (missed_frames + 1) for i in (0, 1)]
                if self._distance(position, predicted) <= prediction_distance * (1 + .25 * missed_frames):
                    return {**result, "reason": "prediction_protected"}
        for center in racket_centers:
            racket_inside, racket_edge = self._inside(center, frame_width, frame_height)
            # A reflected racket is not independent evidence for the real ball.
            if not racket_inside and racket_edge > self.margin_ratio * frame_height:
                if self._distance(position, center) <= racket_distance:
                    return {**result, "reason": "racket_protected"}
        return {**result, "would_reject": True, "rejected": self.mode == "enforce",
                "reason": "small_mirror_unsupported"}
