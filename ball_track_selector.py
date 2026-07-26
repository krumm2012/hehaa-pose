"""Stateful selection of the Active Ball from per-frame detections."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ball_candidate_selector import select_ball_candidate
from racket_candidate_selector import racket_center
from static_ball_filter import StaticBallFilter

Detection = Dict[str, Any]
TrackContext = Tuple[Optional[Sequence[float]], Optional[Sequence[float]], Optional[float]]


@dataclass
class BallTrackSelection:
    """The selected Active Ball and compatibility diagnostics for one frame."""

    active_ball: Optional[Dict]
    diagnostics: Dict


class BallTrackSelector:
    """Keep Active Ball state behind one selection interface."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize state for selecting one Active Ball across frames."""
        self.config = dict(config or {})
        self.ball_history = deque(maxlen=10)
        self.static_ball_filter = StaticBallFilter(self.config)
        self.static_threshold = float(self.config.get("static_ball_movement_threshold_px", 6.0))
        self.reacquisition_frames = max(
            0,
            int(
                self.config.get(
                    "active_ball_reacquisition_frames",
                    self.config.get("max_lost_frames_for_track", 8),
                )
            ),
        )
        self.missed_frames = 0

    def select(
        self,
        ball_detections: Iterable[Detection],
        racket_detections: Optional[Iterable[Detection]] = None,
        frame_height: Optional[float] = None,
    ) -> BallTrackSelection:
        """Select the Active Ball and retain compatibility diagnostics."""
        candidates = [det for det in ball_detections or [] if isinstance(det, dict)]
        diagnostics = self._diagnostics(len(candidates))
        if not candidates:
            self._record_miss()
            self.static_ball_filter.update([])
            return BallTrackSelection(None, diagnostics)

        previous_position, previous_velocity, _previous_motion = self._track_context()
        continuity_enabled = previous_position is not None
        diagnostics["continuity_enabled"] = continuity_enabled
        if self.missed_frames > 0 and not continuity_enabled:
            diagnostics["continuity_disabled_reason"] = "reacquisition_window_expired"

        continuity_distance = float(self.config.get("ball_continuity_distance_px", 180.0))
        continuity_keep_ratio = float(self.config.get("static_ball_keep_if_near_prev_ratio", 0.8))
        racket_distance = float(self.config.get("ball_racket_proximity_distance_px", 360.0))
        racket_keep_ratio = float(self.config.get("static_ball_keep_if_near_racket_ratio", 0.65))
        near_prev_threshold = continuity_distance * continuity_keep_ratio
        near_racket_threshold = racket_distance * racket_keep_ratio
        racket_centers = [
            center
            for racket in racket_detections or []
            for center in [racket_center(racket)]
            if center is not None
        ]

        adjusted_candidates: List[Dict] = []
        for candidate in candidates:
            pos = candidate.get("position")
            if not pos:
                continue
            near_prev = self._near_previous(
                pos,
                previous_position,
                near_prev_threshold,
            )
            near_racket = self._near_any(pos, racket_centers, near_racket_threshold)
            if not near_prev and self.static_ball_filter.should_mask(
                pos,
                near_previous_track=False,
                near_racket=near_racket,
            ):
                diagnostics["rejections"]["static_hard_mask"] += 1
                diagnostics["top_candidates"].append(
                    self._candidate_diagnostic(
                        candidate,
                        0.0,
                        True,
                        near_prev,
                        near_racket,
                        "static_hard_mask",
                    )
                )
                continue

            penalty = self.static_ball_filter.penalty(
                pos,
                near_previous_track=near_prev,
                near_racket=near_racket,
            )
            adjusted = dict(candidate)
            adjusted["confidence"] = max(0.0, float(candidate.get("confidence", 0.0)) - penalty)
            adjusted["static_penalty"] = float(penalty)
            adjusted_candidates.append(adjusted)
            diagnostics["top_candidates"].append(
                self._candidate_diagnostic(
                    candidate,
                    adjusted["confidence"],
                    False,
                    near_prev,
                    near_racket,
                    "kept",
                )
            )

        diagnostics["kept_candidates"] = len(adjusted_candidates)
        supported_candidates = [
            candidate
            for candidate in adjusted_candidates
            if previous_position is not None
            and self._near_previous(
                candidate["position"],
                previous_position,
                near_prev_threshold,
            )
        ]
        selection_candidates = supported_candidates or adjusted_candidates
        best_ball = select_ball_candidate(
            selection_candidates,
            previous_position=previous_position,
            previous_velocity=previous_velocity,
            racket_detections=racket_detections,
            config={**self.config, "frame_height": frame_height},
        )
        if best_ball is None:
            diagnostics["final_decision"] = "no_candidate_after_filter"
            self._update_static_anchors(candidates, None)
            self._record_miss()
            self._finalize_diagnostics(diagnostics)
            return BallTrackSelection(None, diagnostics)

        active_ball = self._accept_or_reject(
            best_ball,
            previous_position,
            racket_centers,
            near_prev_threshold,
            near_racket_threshold,
            frame_height,
            diagnostics,
        )
        self._update_static_anchors(candidates, active_ball)
        if active_ball is None:
            self._record_miss()
        else:
            self.ball_history.append(active_ball["position"])
            self.missed_frames = 0
        self._finalize_diagnostics(diagnostics)
        return BallTrackSelection(active_ball, diagnostics)

    def _track_context(self) -> TrackContext:
        """Return the prior Active Ball state while the window is valid."""
        if not self.ball_history or self.missed_frames > self.reacquisition_frames:
            return None, None, None
        previous_position = self.ball_history[-1]
        previous_velocity = None
        previous_motion = None
        if len(self.ball_history) >= 2:
            previous_velocity = [
                float(self.ball_history[-1][0] - self.ball_history[-2][0]),
                float(self.ball_history[-1][1] - self.ball_history[-2][1]),
            ]
            previous_motion = float(np.linalg.norm(np.array(previous_velocity, dtype=float)))
        return previous_position, previous_velocity, previous_motion

    def _accept_or_reject(
        self,
        best_ball: Detection,
        previous_position: Optional[Sequence[float]],
        racket_centers: List[Sequence[float]],
        near_prev_threshold: float,
        near_racket_threshold: float,
        frame_height: Optional[float],
        diagnostics: Dict[str, Any],
    ) -> Optional[Detection]:
        """Apply acceptance gates after ranking a candidate."""
        best_pos = best_ball.get("position")
        best_conf = float(best_ball.get("confidence", 0.0))
        near_prev = bool(
            previous_position is not None
            and self._distance(best_pos, previous_position) <= near_prev_threshold
        )
        near_racket = self._near_any(best_pos, racket_centers, near_racket_threshold)
        supported_track = near_prev or near_racket
        selected = {
            "position": [float(best_pos[0]), float(best_pos[1])] if best_pos is not None else None,
            "confidence": best_conf,
            "supported_track": supported_track,
        }
        min_conf = float(self.config.get("ball_min_selected_confidence", 0.05))
        if best_conf < min_conf and not supported_track:
            diagnostics["rejections"]["low_conf_unsupported"] += 1
            diagnostics["selected"] = selected
            diagnostics["final_decision"] = "reject_low_conf_unsupported"
            return None

        hard_min_y_ratio = self.config.get("ball_play_area_min_y_ratio_hard")
        if hard_min_y_ratio is not None and best_pos is not None and frame_height:
            hard_min_y = float(hard_min_y_ratio) * float(frame_height)
            if float(best_pos[1]) < hard_min_y and not supported_track:
                diagnostics["rejections"]["upper_mirror_unsupported"] += 1
                diagnostics["selected"] = selected
                diagnostics["final_decision"] = "reject_upper_mirror_unsupported"
                return None

        diagnostics["selected"] = selected
        diagnostics["final_decision"] = "selected"
        return best_ball

    def _update_static_anchors(
        self,
        candidates: Iterable[Detection],
        active_ball: Optional[Detection],
    ) -> None:
        """Learn static anchors only from candidates not selected as Active Ball."""
        active_position = active_ball.get("position") if active_ball else None
        background_positions = [
            candidate["position"]
            for candidate in candidates
            if candidate.get("position") and candidate.get("position") != active_position
        ]
        self.static_ball_filter.update(background_positions)

    def _record_miss(self) -> None:
        """Advance the reacquisition window and clear an expired track."""
        self.missed_frames += 1
        if self.missed_frames > self.reacquisition_frames:
            self.ball_history.clear()

    @staticmethod
    def _distance(a: Sequence[float], b: Sequence[float]) -> float:
        """Return Euclidean distance between two image-plane positions."""
        return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))

    def _near_previous(
        self,
        position: Sequence[float],
        previous: Optional[Sequence[float]],
        threshold: float,
    ) -> bool:
        """Return whether a candidate has Trajectory Support."""
        if previous is None:
            return False
        distance = self._distance(position, previous)
        return distance <= threshold

    def _near_any(
        self,
        position: Optional[Sequence[float]],
        centers: Iterable[Sequence[float]],
        threshold: float,
    ) -> bool:
        """Return whether a position is within a threshold of any racket center."""
        return bool(
            position is not None
            and any(self._distance(position, center) <= threshold for center in centers)
        )

    @staticmethod
    def _diagnostics(raw_candidates: int) -> Dict[str, Any]:
        """Create the existing diagnostics shape for one selection."""
        return {
            "raw_candidates": raw_candidates,
            "kept_candidates": 0,
            "continuity_enabled": True,
            "continuity_disabled_reason": None,
            "rejections": {
                "static_hard_mask": 0,
                "low_conf_unsupported": 0,
                "upper_mirror_unsupported": 0,
                "track_became_static": 0,
            },
            "selected": None,
            "final_decision": "no_candidates",
            "top_candidates": [],
        }

    @staticmethod
    def _candidate_diagnostic(
        candidate: Detection,
        adjusted_confidence: float,
        masked: bool,
        near_previous: bool,
        near_racket: bool,
        reason: str,
    ) -> Dict[str, Any]:
        """Describe one candidate without exposing selector state."""
        pos = candidate["position"]
        return {
            "position": [float(pos[0]), float(pos[1])],
            "raw_confidence": float(candidate.get("confidence", 0.0)),
            "adjusted_confidence": float(adjusted_confidence),
            "masked": bool(masked),
            "near_prev": bool(near_previous),
            "near_racket": bool(near_racket),
            "reason": reason,
        }

    @staticmethod
    def _finalize_diagnostics(diagnostics: Dict[str, Any]) -> None:
        """Sort and cap diagnostic candidates without changing field names."""
        diagnostics["top_candidates"] = sorted(
            diagnostics["top_candidates"],
            key=lambda candidate: candidate.get("adjusted_confidence", 0.0),
            reverse=True,
        )[:8]
