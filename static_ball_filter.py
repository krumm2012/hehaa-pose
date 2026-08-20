"""Temporal static-ball suppression for multi-ball training scenes."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence


class StaticBallFilter:
    """Maintain lightweight spatial anchors for persistent static balls."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        config = config or {}
        self.enabled = bool(config.get("static_ball_suppression_enabled", True))
        self.grid_size_px = max(8.0, float(config.get("static_ball_grid_size_px", 40.0)))
        self.match_distance_px = max(6.0, float(config.get("static_ball_match_distance_px", 26.0)))
        self.min_seen_frames = max(2, int(config.get("static_ball_min_seen_frames", 6)))
        self.decay_frames = max(self.min_seen_frames + 1, int(config.get("static_ball_decay_frames", 50)))
        self.max_penalty = max(0.0, float(config.get("static_ball_penalty_max", 0.50)))
        self.hard_mask_enabled = bool(config.get("static_ball_hard_mask_enabled", True))
        self.hard_mask_min_seen_frames = max(
            self.min_seen_frames,
            int(config.get("static_ball_hard_mask_min_seen_frames", self.min_seen_frames + 2)),
        )
        self.hard_mask_radius_px = max(
            self.match_distance_px,
            float(config.get("static_ball_hard_mask_radius_px", self.match_distance_px * 1.15)),
        )
        self.hard_mask_allow_near_prev = bool(config.get("static_ball_hard_mask_allow_near_prev", True))
        self.hard_mask_allow_near_racket = bool(config.get("static_ball_hard_mask_allow_near_racket", False))
        self._frame_idx = 0
        self._anchors: Dict[str, Dict] = {}

    @staticmethod
    def _distance(a: Sequence[float], b: Sequence[float]) -> float:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def _new_key(self, pos: Sequence[float]) -> str:
        return f"{int(pos[0] / self.grid_size_px)}_{int(pos[1] / self.grid_size_px)}_{self._frame_idx}"

    def _closest_anchor(self, pos: Sequence[float]) -> Optional[Dict]:
        return self._closest_anchor_within(pos, self.match_distance_px)

    def _closest_anchor_within(self, pos: Sequence[float], distance_limit: float) -> Optional[Dict]:
        best = None
        best_dist = float("inf")
        for anchor in self._anchors.values():
            dist = self._distance(anchor["center"], pos)
            if dist <= distance_limit and dist < best_dist:
                best_dist = dist
                best = anchor
        return best

    def update(self, positions: Iterable[Sequence[float]]) -> None:
        if not self.enabled:
            return
        self._frame_idx += 1
        for pos in positions:
            if not pos or len(pos) < 2:
                continue
            anchor = self._closest_anchor(pos)
            if anchor is None:
                self._anchors[self._new_key(pos)] = {
                    "center": [float(pos[0]), float(pos[1])],
                    "hits": 1,
                    "last_seen": self._frame_idx,
                }
                continue

            # Smooth center to absorb small detection jitter.
            anchor["center"][0] = anchor["center"][0] * 0.85 + float(pos[0]) * 0.15
            anchor["center"][1] = anchor["center"][1] * 0.85 + float(pos[1]) * 0.15
            anchor["hits"] += 1
            anchor["last_seen"] = self._frame_idx

        self._anchors = {
            k: v
            for k, v in self._anchors.items()
            if (self._frame_idx - int(v.get("last_seen", self._frame_idx))) <= self.decay_frames
        }

    def penalty(
        self,
        position: Sequence[float],
        near_previous_track: bool = False,
        near_racket: bool = False,
    ) -> float:
        if not self.enabled:
            return 0.0
        anchor = self._closest_anchor(position)
        if anchor is None:
            return 0.0
        hits = int(anchor.get("hits", 0))
        if hits < self.min_seen_frames:
            return 0.0

        strength = min(1.0, float(hits - self.min_seen_frames + 1) / float(self.min_seen_frames))
        penalty = strength * self.max_penalty
        if near_previous_track:
            penalty *= 0.35
        if near_racket:
            penalty *= 0.50
        return penalty

    def should_mask(
        self,
        position: Sequence[float],
        near_previous_track: bool = False,
        near_racket: bool = False,
    ) -> bool:
        if not self.enabled or not self.hard_mask_enabled:
            return False
        anchor = self._closest_anchor_within(position, self.hard_mask_radius_px)
        if anchor is None:
            return False
        hits = int(anchor.get("hits", 0))
        if hits < self.hard_mask_min_seen_frames:
            return False
        if near_previous_track and self.hard_mask_allow_near_prev:
            return False
        if near_racket and self.hard_mask_allow_near_racket:
            return False
        return True
