"""Deterministic, local coaching guidance for completed Swing events."""

from __future__ import annotations

from typing import Dict, List, Optional

from swing_quality_policy import effective_quality_warnings


class LocalRealtimeCoach:
    """Return one to three concise, evidence-ranked Chinese corrections."""

    HARD_MAX_CHARS = 15
    HARD_MAX_SUGGESTIONS = 3
    DEFAULT_THRESHOLDS = {
        "min_hip_shoulder_separation_deg": 15.0,
        "min_shoulder_turn_deg": 75.0,
        "min_arm_extension_deg": 145.0,
        "min_contact_lateral_body_widths": 0.55,
        "min_weight_transfer_body_widths": 0.08,
        "max_balance_drift_body_widths": 0.65,
    }

    def __init__(
        self,
        max_chars: int = HARD_MAX_CHARS,
        max_suggestions: int = HARD_MAX_SUGGESTIONS,
        min_confidence: float = 0.45,
        thresholds: Optional[Dict] = None,
    ):
        self.max_chars = min(self.HARD_MAX_CHARS, max(1, int(max_chars)))
        self.max_suggestions = min(
            self.HARD_MAX_SUGGESTIONS,
            max(1, int(max_suggestions)),
        )
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.thresholds = {
            **self.DEFAULT_THRESHOLDS,
            **(thresholds or {}),
        }

    def advise(self, event: Dict) -> Dict:
        """Return the primary recommendation for backward compatibility."""
        return self.advise_all(event)[0]

    def advise_all(self, event: Dict) -> List[Dict]:
        """Return one to three recommendations, each with its own confidence."""
        quality = event.get("quality_flags") or {}
        warnings = effective_quality_warnings(quality)
        blocker = self._blocking_advice(event, quality, warnings)
        if blocker is not None:
            return [blocker]

        candidates = self._biomechanical_candidates(event)
        phases = event.get("phase_counts") or {}
        event_confidence = float(event.get("confidence") or 0.0)
        backswing_frames = int(phases.get("backswing") or 0)
        if backswing_frames < 2:
            candidates.append(
                self._ranked_advice(
                    priority=84,
                    code="short_backswing",
                    message="提前准备充分引拍",
                    category="technique",
                    confidence=event_confidence,
                    focus="preparation",
                    group="preparation",
                    evidence={"backswing_frames": backswing_frames},
                    source="local_rules_v1",
                )
            )
        follow_through_frames = int(phases.get("follow_through") or 0)
        if follow_through_frames < 4:
            candidates.append(
                self._ranked_advice(
                    priority=98,
                    code="short_follow_through",
                    message="击球后完成随挥",
                    category="technique",
                    confidence=event_confidence,
                    focus="follow_through",
                    group="follow_through",
                    evidence={"follow_through_frames": follow_through_frames},
                    source="local_rules_v1",
                )
            )
        if candidates:
            candidates.sort(
                key=lambda item: (
                    -int(item.pop("_priority")),
                    -float(item["confidence"]),
                    item["code"],
                )
            )
            selected = []
            groups = set()
            for candidate in candidates:
                group = candidate.pop("_group")
                if group in groups:
                    continue
                groups.add(group)
                selected.append(candidate)
                if len(selected) >= self.max_suggestions:
                    break
            return selected

        if "racket_track_gaps" in warnings:
            return [
                self._advice(
                    code="racket_track_gaps",
                    message="减少球拍遮挡",
                    category="capture",
                    confidence=float(quality.get("racket_frame_ratio") or 0.0),
                    evidence={"racket_frame_ratio": quality.get("racket_frame_ratio")},
                )
            ]
        if "contact_frame_needs_review" in warnings:
            return [
                self._advice(
                    code="contact_frame_needs_review",
                    message="触球位置需复核",
                    category="review",
                    confidence=event_confidence,
                    evidence={"warning": "contact_frame_needs_review"},
                )
            ]
        review_messages = {
            "ball_continuity_disabled": "来球轨迹需复核",
            "static_ball_mask_in_event": "网球识别需复核",
            "mirror_ball_rejection_in_event": "网球识别需复核",
        }
        for warning, message in review_messages.items():
            if warning in warnings:
                return [
                    self._advice(
                        code=warning,
                        message=message,
                        category="review",
                        confidence=event_confidence,
                        evidence={"warning": warning},
                    )
                ]
        return [
            self._advice(
                code="maintain_form",
                message="动作稳定继续保持",
                category="positive",
                confidence=event_confidence,
                evidence={},
            )
        ]

    def _blocking_advice(
        self,
        event: Dict,
        quality: Dict,
        warnings,
    ) -> Optional[Dict]:
        pose_ratio = float(quality.get("pose_frame_ratio") or 0.0)
        if "pose_gaps" in warnings and pose_ratio < 0.65:
            return self._advice(
                code="pose_gaps",
                message="保持全身清晰入镜",
                category="capture",
                confidence=pose_ratio,
                evidence={"pose_frame_ratio": quality.get("pose_frame_ratio")},
            )
        if "ball_track_gaps" in warnings:
            return self._advice(
                code="ball_track_gaps",
                message="确保来球完整入镜",
                category="capture",
                confidence=float(quality.get("ball_frame_ratio") or 0.0),
                evidence={"ball_frame_ratio": quality.get("ball_frame_ratio")},
            )
        confidence = float(event.get("confidence") or 0.0)
        if confidence < 0.55:
            return self._advice(
                code="low_confidence",
                message="本次动作建议复核",
                category="review",
                confidence=confidence,
                evidence={"event_confidence": confidence},
            )
        return None

    def _biomechanical_candidates(self, event: Dict) -> List[Dict]:
        metrics = (event.get("biomechanics") or {}).get("metrics") or {}
        candidates = []

        def add_low(
            metric_name: str,
            threshold_name: str,
            priority: int,
            code: str,
            message: str,
            focus: str,
            group: str,
        ) -> None:
            metric = metrics.get(metric_name) or {}
            value = self._float(metric.get("value"))
            metric_confidence = self._float(metric.get("confidence"))
            threshold = float(self.thresholds[threshold_name])
            if (
                value is None
                or metric_confidence is None
                or metric_confidence < self.min_confidence
                or value >= threshold
            ):
                return
            candidates.append(
                self._biomechanical_advice(
                    event,
                    metric,
                    priority,
                    code,
                    message,
                    focus,
                    group,
                    threshold,
                )
            )

        def add_high(
            metric_name: str,
            threshold_name: str,
            priority: int,
            code: str,
            message: str,
            focus: str,
            group: str,
        ) -> None:
            metric = metrics.get(metric_name) or {}
            value = self._float(metric.get("value"))
            metric_confidence = self._float(metric.get("confidence"))
            threshold = float(self.thresholds[threshold_name])
            if (
                value is None
                or metric_confidence is None
                or metric_confidence < self.min_confidence
                or value <= threshold
            ):
                return
            candidates.append(
                self._biomechanical_advice(
                    event,
                    metric,
                    priority,
                    code,
                    message,
                    focus,
                    group,
                    threshold,
                )
            )

        add_low(
            "contact_lateral_distance",
            "min_contact_lateral_body_widths",
            96,
            "contact_too_close",
            "击球点离身体远些",
            "contact_position",
            "contact_position",
        )
        add_low(
            "arm_extension",
            "min_arm_extension_deg",
            94,
            "limited_arm_extension",
            "击球时手臂再伸展",
            "arm_extension",
            "arm_extension",
        )
        add_low(
            "hip_shoulder_separation",
            "min_hip_shoulder_separation_deg",
            92,
            "limited_separation",
            "加大肩髋分离",
            "hip_shoulder_separation",
            "rotation",
        )
        add_high(
            "balance_drift",
            "max_balance_drift_body_widths",
            90,
            "unstable_balance",
            "击球后稳住重心",
            "balance",
            "gravity",
        )
        add_low(
            "weight_transfer",
            "min_weight_transfer_body_widths",
            86,
            "limited_weight_transfer",
            "击球时带动重心",
            "balance",
            "gravity",
        )
        add_low(
            "shoulder_turn",
            "min_shoulder_turn_deg",
            82,
            "limited_shoulder_turn",
            "提前转肩充分引拍",
            "shoulder_turn",
            "rotation",
        )
        return candidates

    def _biomechanical_advice(
        self,
        event: Dict,
        metric: Dict,
        priority: int,
        code: str,
        message: str,
        focus: str,
        group: str,
        threshold: float,
    ) -> Dict:
        event_confidence = float(event.get("confidence") or 0.0)
        metric_confidence = float(metric.get("confidence") or 0.0)
        confidence = event_confidence * 0.45 + metric_confidence * 0.55
        return self._ranked_advice(
            priority=priority,
            code=code,
            message=message,
            category="technique",
            confidence=confidence,
            focus=focus,
            group=group,
            evidence={
                "value": metric.get("value"),
                "unit": metric.get("unit"),
                "threshold": threshold,
                "metric_confidence": metric_confidence,
                "source_frames": metric.get("source_frames") or [],
            },
            source="local_biomechanics_v2",
        )

    def _ranked_advice(
        self,
        priority: int,
        code: str,
        message: str,
        category: str,
        confidence: float,
        focus: str,
        group: str,
        evidence: Dict,
        source: str,
    ) -> Dict:
        advice = self._advice(
            code=code,
            message=message,
            category=category,
            confidence=confidence,
            evidence=evidence,
            source=source,
        )
        advice["focus"] = focus
        advice["_priority"] = int(priority)
        advice["_group"] = group
        return advice

    def _advice(
        self,
        code: str,
        message: str,
        category: str,
        confidence: float,
        evidence: Dict,
        source: str = "local_rules_v1",
    ) -> Dict:
        return {
            "code": code,
            "message": message[: self.max_chars],
            "category": category,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "source": source,
            "evidence": evidence,
        }

    @staticmethod
    def _float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
