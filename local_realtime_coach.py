"""Deterministic, local coaching guidance for completed Swing events."""

from __future__ import annotations

from typing import Dict

from swing_quality_policy import effective_quality_warnings


class LocalRealtimeCoach:
    """Return one concise Chinese recommendation from a completed event."""

    HARD_MAX_CHARS = 15

    def __init__(self, max_chars: int = HARD_MAX_CHARS):
        self.max_chars = min(self.HARD_MAX_CHARS, max(1, int(max_chars)))

    def advise(self, event: Dict) -> Dict:
        quality = event.get("quality_flags") or {}
        warnings = effective_quality_warnings(quality)
        if "pose_gaps" in warnings:
            return self._advice(
                code="pose_gaps",
                message="保持全身清晰入镜",
                category="capture",
                confidence=float(quality.get("pose_frame_ratio") or 0.0),
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
        if confidence < 0.65:
            return self._advice(
                code="low_confidence",
                message="本次动作建议复核",
                category="review",
                confidence=confidence,
                evidence={"event_confidence": confidence},
            )

        phases = event.get("phase_counts") or {}
        backswing_frames = int(phases.get("backswing") or 0)
        if backswing_frames < 2:
            return self._advice(
                code="short_backswing",
                message="提前准备充分引拍",
                category="technique",
                confidence=confidence,
                evidence={"backswing_frames": backswing_frames},
            )
        follow_through_frames = int(phases.get("follow_through") or 0)
        if follow_through_frames < 4:
            return self._advice(
                code="short_follow_through",
                message="击球后完成随挥",
                category="technique",
                confidence=float(event.get("confidence") or 0.0),
                evidence={"follow_through_frames": follow_through_frames},
            )
        if "racket_track_gaps" in warnings:
            return self._advice(
                code="racket_track_gaps",
                message="减少球拍遮挡",
                category="capture",
                confidence=float(quality.get("racket_frame_ratio") or 0.0),
                evidence={"racket_frame_ratio": quality.get("racket_frame_ratio")},
            )
        if "contact_frame_needs_review" in warnings:
            return self._advice(
                code="contact_frame_needs_review",
                message="触球位置需复核",
                category="review",
                confidence=float(event.get("confidence") or 0.0),
                evidence={"warning": "contact_frame_needs_review"},
            )
        review_messages = {
            "ball_continuity_disabled": "来球轨迹需复核",
            "static_ball_mask_in_event": "网球识别需复核",
            "mirror_ball_rejection_in_event": "网球识别需复核",
        }
        for warning, message in review_messages.items():
            if warning in warnings:
                return self._advice(
                    code=warning,
                    message=message,
                    category="review",
                    confidence=float(event.get("confidence") or 0.0),
                    evidence={"warning": warning},
                )
        return self._advice(
            code="maintain_form",
            message="动作稳定继续保持",
            category="positive",
            confidence=float(event.get("confidence") or 0.0),
            evidence={},
        )

    def _advice(
        self,
        code: str,
        message: str,
        category: str,
        confidence: float,
        evidence: Dict,
    ) -> Dict:
        return {
            "code": code,
            "message": message[: self.max_chars],
            "category": category,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "source": "local_rules_v1",
            "evidence": evidence,
        }
