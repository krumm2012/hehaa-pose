"""Build claim-level Coach permissions from merged Swing evidence."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from swing_quality_policy import effective_quality_warnings


BODY_TOPICS = {
    "arm_extension",
    "balance",
    "contact_spacing",
    "follow_through",
    "knee_flexion",
    "positive_form",
    "power_transfer",
    "preparation",
    "stance",
    "tempo",
}
SINGLE_VIEW_BLOCKED_TOPICS = {"balance", "power_transfer"}
BALL_WARNINGS = {
    "ball_track_gaps",
    "ball_continuity_disabled",
    "static_ball_mask_in_event",
    "mirror_ball_rejection_in_event",
}
RACKET_WARNINGS = {"racket_track_gaps"}
REVIEW_WARNINGS = (
    BALL_WARNINGS
    | RACKET_WARNINGS
    | {"pose_gaps", "contact_frame_needs_review"}
)


def build_coach_decision_policy(
    quality: Dict,
    coach_metrics: Dict,
    integrity: Dict,
) -> Dict:
    """Return the complete DeepSeek coaching policy for one Swing event."""
    raw_warnings = set(quality.get("warnings") or [])
    warnings = effective_quality_warnings(quality)
    tolerated_conditions = []
    if "ball_track_gaps" in raw_warnings and "ball_track_gaps" not in warnings:
        tolerated_conditions.append("intermittent_ball_detection")

    data_quality = coach_metrics.get("data_quality") or {}
    pose_ratio = _first_float(
        quality.get("pose_frame_ratio"),
        data_quality.get("pose_frame_ratio"),
    )
    event_confidence = _first_float(coach_metrics.get("confidence"))
    aligned = bool(integrity.get("aligned_to_event_range", True))
    pose_reliable = (
        "pose_gaps" not in warnings
        and (pose_ratio is None or pose_ratio >= 0.65)
    )
    confidence_usable = event_confidence is None or event_confidence >= 0.55
    coaching_allowed = aligned and pose_reliable and confidence_usable

    blocked_topics = _blocked_topics(
        warnings=warnings,
        missing_fields=data_quality.get("missing_fields") or [],
        coaching_allowed=coaching_allowed,
    )
    blocked_topics.update(SINGLE_VIEW_BLOCKED_TOPICS)
    allowed_topics = set(BODY_TOPICS) if coaching_allowed else set()
    if "racket_path" not in blocked_topics and coaching_allowed:
        allowed_topics.add("racket_path")
    if "contact_timing" not in blocked_topics and coaching_allowed:
        allowed_topics.add("contact_timing")
    allowed_topics -= blocked_topics

    candidates = _advice_candidates(
        coach_metrics=coach_metrics,
        allowed_topics=allowed_topics,
    )
    review_required = bool(
        warnings.intersection(REVIEW_WARNINGS)
        or not aligned
        or (quality.get("review_recommended") and not raw_warnings)
    )
    if coaching_allowed:
        allowed_categories = ["technique", "positive", "review"]
        if review_required:
            allowed_categories.insert(2, "capture")
    else:
        allowed_categories = ["capture", "review"]

    prohibited_claims = _prohibited_claims(
        warnings=warnings,
        missing_fields=data_quality.get("missing_fields") or [],
    )
    return {
        "coaching_allowed": coaching_allowed,
        "review_required": review_required,
        "effective_warnings": sorted(warnings),
        "tolerated_conditions": tolerated_conditions,
        "allowed_advice_categories": allowed_categories,
        "preferred_advice_categories": (
            ["technique", "positive"] if coaching_allowed else ["capture", "review"]
        ),
        "allowed_advice_topics": sorted(allowed_topics),
        "blocked_advice_topics": sorted(blocked_topics),
        "advice_candidates": candidates,
        "prohibited_claims": sorted(prohibited_claims),
        "evidence_priority": [
            "decision_policy.advice_candidates",
            "coach_metrics",
            "event",
            "frame_sequence",
            "recent_swings",
        ],
        "rules": [
            "null_or_missing_is_unknown",
            "pixel_values_are_relative_within_this_video",
            "partial_tracking_warnings_only_block_related_topics",
            "use_only_supported_claims",
            "one_actionable_advice",
        ],
    }


def _blocked_topics(
    warnings: Set[str],
    missing_fields: Iterable[str],
    coaching_allowed: bool,
) -> Set[str]:
    blocked = set()
    if not coaching_allowed:
        blocked.update(BODY_TOPICS)
    if warnings.intersection(BALL_WARNINGS):
        blocked.update(
            {
                "ball_trajectory",
                "contact_timing",
                "landing",
                "spin",
            }
        )
    if warnings.intersection(RACKET_WARNINGS):
        blocked.update({"racket_face", "racket_path"})
    if "contact_frame_needs_review" in warnings:
        blocked.add("contact_timing")
    for field in missing_fields:
        text = str(field)
        if "spin" in text:
            blocked.add("spin")
        if any(token in text for token in ("landing", "shot_depth", "bounce")):
            blocked.add("landing")
        if "racket_face" in text:
            blocked.add("racket_face")
        if "net_clearance" in text:
            blocked.add("ball_trajectory")
    return blocked


def _prohibited_claims(
    warnings: Set[str],
    missing_fields: Iterable[str],
) -> Set[str]:
    prohibited = {
        "professional_speed_comparison",
        "true_3d_hip_shoulder_separation",
        "weight_transfer_from_screen_translation",
        "balance_from_screen_translation",
        "injury_risk_from_single_view",
    }
    for field in missing_fields:
        text = str(field)
        if "spin" in text:
            prohibited.add("spin")
        if any(token in text for token in ("landing", "shot_depth", "bounce")):
            prohibited.add("landing")
        if "racket_face" in text:
            prohibited.add("racket_face")
        if "net_clearance" in text:
            prohibited.add("net_clearance")
    if warnings.intersection(BALL_WARNINGS):
        prohibited.update(
            {
                "definitive_ball_trajectory",
                "landing",
                "spin",
            }
        )
    if warnings.intersection(RACKET_WARNINGS):
        prohibited.update({"definitive_racket_path", "racket_face"})
    if "contact_frame_needs_review" in warnings:
        prohibited.add("definitive_contact_timing")
    return prohibited


def _advice_candidates(
    coach_metrics: Dict,
    allowed_topics: Set[str],
) -> List[Dict]:
    body = coach_metrics.get("body") or {}
    timing = coach_metrics.get("timing") or {}
    scores = coach_metrics.get("scores") or {}
    diagnosis_tags = set(coach_metrics.get("diagnosis_tags") or [])
    calibration = coach_metrics.get("coach_calibration") or {}
    assessments = calibration.get("assessments") or {}
    contact_frame = (coach_metrics.get("frames") or {}).get("contact")
    candidates = []

    def add(
        focus: str,
        message_hint: str,
        priority: int,
        evidence_paths: List[str],
    ) -> None:
        if focus not in allowed_topics:
            return
        candidate = {
            "focus": focus,
            "category": "technique",
            "message_hint": message_hint,
            "priority": priority,
            "evidence_paths": evidence_paths,
        }
        if contact_frame is not None:
            candidate["evidence_frames"] = [int(contact_frame)]
        candidates.append(candidate)

    if "short_follow_through" in diagnosis_tags:
        add(
            "follow_through",
            "击球后完成随挥",
            100,
            [
                "coach_metrics.diagnosis_tags.short_follow_through",
                "coach_metrics.timing.phase_durations_frames.follow_through",
            ],
        )
    arm = assessments.get("arm_extension") or {}
    if (
        arm.get("status") == "usable"
        and (_first_float(arm.get("value")) or 0.0) < 145.0
    ):
        add(
            "arm_extension",
            "挥拍时手臂再舒展",
            97,
            ["coach_metrics.coach_calibration.assessments.arm_extension"],
        )
    knee = assessments.get("preparation_knee_flexion") or {}
    if (
        knee.get("status") == "usable"
        and (_first_float(knee.get("value")) or 0.0) < 12.0
    ):
        add(
            "knee_flexion",
            "准备时适当降低重心",
            96,
            [
                "coach_metrics.coach_calibration.assessments."
                "preparation_knee_flexion"
            ],
        )
    turn = assessments.get("shoulder_turn_change") or {}
    if (
        turn.get("status") == "usable"
        and (_first_float(turn.get("value")) or 0.0) < 12.0
    ):
        add(
            "preparation",
            "提前转肩充分引拍",
            95,
            [
                "coach_metrics.coach_calibration.assessments."
                "shoulder_turn_change"
            ],
        )
    if body.get("unit_turn_quality") == "limited":
        add(
            "preparation",
            "提前转肩充分引拍",
            95,
            ["coach_metrics.body.unit_turn_quality"],
        )
    if body.get("late_contact") is True:
        add(
            "contact_timing",
            "提前迎球击球",
            92,
            ["coach_metrics.body.late_contact"],
        )
    if body.get("contact_too_close_to_body") is True:
        add(
            "contact_spacing",
            "击球点离身体远些",
            90,
            ["coach_metrics.body.contact_too_close_to_body"],
        )
    if body.get("balance_state") == "unstable":
        add(
            "balance",
            "击球后稳住重心",
            88,
            ["coach_metrics.body.balance_state"],
        )
    power_score = _first_float(scores.get("power_transfer_score"))
    separation = _first_float(body.get("hip_shoulder_separation_at_contact"))
    if (
        power_score is not None
        and power_score < 0.35
        and separation is not None
    ):
        add(
            "power_transfer",
            "加强转髋带动挥拍",
            85,
            [
                "coach_metrics.scores.power_transfer_score",
                "coach_metrics.body.hip_shoulder_separation_at_contact",
            ],
        )
    if timing.get("preparation_timing_quality") == "quick":
        add(
            "preparation",
            "提前准备充分引拍",
            80,
            ["coach_metrics.timing.preparation_timing_quality"],
        )
    tempo = _first_float(timing.get("tempo_consistency"))
    if tempo is not None and tempo < 0.35:
        add(
            "tempo",
            "保持引拍击球节奏",
            75,
            ["coach_metrics.timing.tempo_consistency"],
        )

    # Multiple evidence paths may support the same coaching focus.  Send the
    # strongest one once so the model does not interpret repetition as extra
    # confidence.
    by_focus = {}
    for candidate in candidates:
        focus = candidate["focus"]
        current = by_focus.get(focus)
        if current is None or int(candidate["priority"]) > int(current["priority"]):
            by_focus[focus] = candidate
    candidates = list(by_focus.values())
    candidates.sort(key=lambda item: (-int(item["priority"]), item["focus"]))
    if candidates or not allowed_topics:
        return candidates
    if any(value is not None for value in body.values()):
        positive = {
            "focus": "positive_form",
            "category": "positive",
            "message_hint": "保持动作连贯",
            "priority": 10,
            "evidence_paths": ["coach_metrics.body", "coach_metrics.timing"],
        }
        if contact_frame is not None:
            positive["evidence_frames"] = [int(contact_frame)]
        return [positive]
    return []


def _first_float(*values) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None
