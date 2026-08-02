#!/usr/bin/env python3
"""Calibrate single-view biomechanics into conservative Coach evidence."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


CALIBRATION_POLICY_VERSION = "single_view_visible_coach_v1"

# These are visible-technique rubrics, not normative 3D biomechanics.  A low
# endpoint deliberately remains above zero because one camera cannot justify a
# zero-quality athletic score from a single projected angle.
VISIBLE_SCORE_CURVES: Dict[str, Sequence[Tuple[float, float]]] = {
    "arm_extension": ((60.0, 0.35), (120.0, 0.62), (145.0, 0.80), (165.0, 1.0)),
    "shoulder_turn_change": ((0.0, 0.35), (12.0, 0.50), (30.0, 0.80), (45.0, 1.0)),
    "preparation_knee_flexion": ((0.0, 0.35), (12.0, 0.50), (25.0, 0.75), (45.0, 1.0)),
}


def _float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _curve_score(value: float, curve: Sequence[Tuple[float, float]]) -> float:
    points = sorted((float(x), float(y)) for x, y in curve)
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if left_x <= value <= right_x:
            ratio = (value - left_x) / max(1e-9, right_x - left_x)
            return left_y + (right_y - left_y) * ratio
    return points[-1][1]


def calibrate_coaching_event(event: Dict, min_confidence: float = 0.45) -> Dict:
    """Return one auditable visible-technique score for local and AI Coach.

    Only metrics explicitly marked Coach-eligible participate.  The result is
    intentionally separate from stroke classification confidence and from
    capture-quality scores.
    """
    biomechanics = event.get("biomechanics") or {}
    metrics = biomechanics.get("metrics") or {}
    assessments = {}
    excluded = []
    weighted_scores: List[Tuple[float, float]] = []

    for name, curve in VISIBLE_SCORE_CURVES.items():
        metric = metrics.get(name) or {}
        value = _float(metric.get("value"))
        confidence = _float(metric.get("confidence")) or 0.0
        eligible = metric.get("coach_eligible") is not False
        reason = metric.get("exclusion_reason")
        if value is None:
            status = "missing"
            reason = reason or "metric_missing"
        elif not eligible:
            status = "excluded"
            reason = reason or "not_coach_eligible"
        elif confidence < float(min_confidence):
            status = "low_confidence"
            reason = reason or "metric_confidence_below_threshold"
        else:
            status = "usable"
            score = max(0.0, min(1.0, _curve_score(value, curve)))
            weighted_scores.append((score, confidence))
            assessments[name] = {
                "status": status,
                "value": round(value, 4),
                "unit": metric.get("unit"),
                "confidence": round(confidence, 4),
                "score_0_1": round(score, 4),
                "source_frames": metric.get("source_frames") or [],
            }
            continue
        assessments[name] = {
            "status": status,
            "value": round(value, 4) if value is not None else None,
            "unit": metric.get("unit"),
            "confidence": round(confidence, 4),
            "reason": reason,
            "source_frames": metric.get("source_frames") or [],
        }
        excluded.append({"metric": name, "reason": reason})

    expected_metrics = len(VISIBLE_SCORE_CURVES)
    if len(weighted_scores) >= 2:
        total_weight = sum(confidence for _, confidence in weighted_scores)
        score = sum(value * confidence for value, confidence in weighted_scores) / max(
            1e-9,
            total_weight,
        )
        coverage = len(weighted_scores) / expected_metrics
        confidence = (total_weight / len(weighted_scores)) * coverage
        status = "calibrated"
        score_0_1 = round(score, 4)
        score_9 = round(score * 9.0, 2)
        uncertainty_9 = round(max(0.6, (1.0 - confidence) * 3.0), 2)
    else:
        status = "insufficient_evidence"
        score_0_1 = None
        score_9 = None
        confidence = 0.0
        uncertainty_9 = None

    return {
        "policy_version": CALIBRATION_POLICY_VERSION,
        "status": status,
        "visible_technique_score": score_0_1,
        "visible_technique_score_9": score_9,
        "uncertainty_9": uncertainty_9,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "metrics_used": [
            name for name, assessment in assessments.items() if assessment["status"] == "usable"
        ],
        "assessments": assessments,
        "excluded_metrics": excluded,
        "limitations": [
            "single_view_visible_technique_only",
            "not_a_3d_kinetic_or_injury_score",
            "compare_sessions_only_after_camera_calibration",
        ],
    }
