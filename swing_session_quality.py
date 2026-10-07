"""Aggregate auditable session quality and drift from Swing events."""

from __future__ import annotations

import json
from collections import Counter
from statistics import mean, median, stdev
from typing import Dict, Iterable, List, Optional, Sequence

from practice_scoring import number
from practice_score_adapter import resolve_practice_score


SCHEMA_VERSION = "swing_session_quality_v2"
MIN_DRIFT_EVENTS = 6
MAX_WINDOW_EVENTS = 5


def _number(value) -> Optional[float]:
    return number(value)


def _rounded(value: Optional[float], digits: int = 4) -> Optional[float]:
    return round(float(value), digits) if value is not None else None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    usable = [float(value) for value in values if value is not None]
    return mean(usable) if usable else None


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    usable = [float(value) for value in values if value is not None]
    return median(usable) if usable else None


def _stdev(values: Iterable[Optional[float]]) -> Optional[float]:
    usable = [float(v) for v in values if v is not None]
    return round(float(stdev(usable)), 2) if len(usable) >= 2 else None


def _metric_value(event: Dict, name: str) -> Optional[float]:
    metric = (((event.get("biomechanics") or {}).get("metrics") or {}).get(name) or {})
    if metric.get("coach_eligible") is not True:
        return None
    return _number(metric.get("value"))


def _advice_codes(event: Dict) -> List[str]:
    advices = event.get("coach_advices") or []
    if not advices and event.get("coach_advice"):
        advices = [event["coach_advice"]]
    return [
        str(advice.get("code") or advice.get("focus"))
        for advice in advices
        if isinstance(advice, dict) and (advice.get("code") or advice.get("focus"))
    ]


def _event_point(event: Dict, index: int) -> Dict:
    quality = event.get("quality_flags") or {}
    biomechanics = event.get("biomechanics") or {}
    bio_quality = biomechanics.get("quality") or {}
    practice = resolve_practice_score(event)
    calibration = practice["calibration"]
    deepseek = event.get("deepseek_advice") or {}
    timing = event.get("timing") or {}
    pose = _number(quality.get("pose_frame_ratio"))
    ball = _number(quality.get("ball_frame_ratio"))
    racket = _number(quality.get("racket_frame_ratio"))
    contact = _number(bio_quality.get("contact_evidence_confidence"))
    weighted_quality = [
        (pose, 0.35),
        (ball, 0.20),
        (racket, 0.15),
        (contact, 0.30),
    ]
    available_weight = sum(weight for value, weight in weighted_quality if value is not None)
    evidence_quality = (
        sum(max(0.0, min(1.0, value)) * weight for value, weight in weighted_quality if value is not None)
        / available_weight
        if available_weight > 0
        else None
    )
    context = (event.get("evidence") or {}).get("classification_context") or {}
    camera = context.get("camera") or {}
    warnings = sorted(set(str(value) for value in quality.get("warnings") or []))
    return {
        "event_id": int(event.get("event_id", index + 1)),
        "start_frame": (
            int(event["start_frame"])
            if event.get("start_frame") is not None
            else None
        ),
        "contact_frame": (
            int(event["contact_frame"])
            if event.get("contact_frame") is not None
            else None
        ),
        "end_frame": (
            int(event["end_frame"])
            if event.get("end_frame") is not None
            else None
        ),
        "stroke_type": event.get("stroke_type"),
        "comparison_key": json.dumps({"policy": practice["policy_version"], "method": practice["method"], "stroke": event.get("stroke_type"), "context": event.get("practice_context") or {}}, sort_keys=True),
        "practice_score_100": practice["score"],
        "scoring_method": practice["method"],
        "score_uncertainty_100": (
            _number(calibration.get("uncertainty_9")) / 9 * 100
            if practice["method"] == "automatic_2d_projection"
            and _number(calibration.get("uncertainty_9")) is not None
            else None
        ),
        "policy_version": practice["policy_version"],
        "visible_score_9": _number(calibration.get("visible_technique_score_9")),
        "score_uncertainty_9": _number(calibration.get("uncertainty_9")),
        "calibration_confidence": _number(calibration.get("confidence")),
        "calibration_status": calibration.get("status"),
        "evidence_quality_100": (
            round(evidence_quality * 100.0, 2)
            if evidence_quality is not None
            else None
        ),
        "pose_frame_ratio": pose,
        "ball_frame_ratio": ball,
        "racket_frame_ratio": racket,
        "contact_evidence_confidence": contact,
        "reference_body_width_px": _number(biomechanics.get("reference_body_width_px")),
        "camera_view": camera.get("view"),
        "review_recommended": bool(quality.get("review_recommended")),
        "warnings": warnings,
        "advice_codes": _advice_codes(event),
        "arm_extension_deg": _metric_value(event, "arm_extension"),
        "shoulder_turn_change_deg": _metric_value(event, "shoulder_turn_change"),
        "preparation_knee_flexion_deg": _metric_value(
            event,
            "preparation_knee_flexion",
        ),
        "contact_capture_to_coach_ms": _number(
            timing.get("contact_capture_to_coach_ms")
        ),
        "deepseek_status": deepseek.get("status"),
        "deepseek_latency_ms": _number(deepseek.get("latency_ms")),
        "racket_speed_kmh": _number(
            ((event.get("extended_biomechanics") or (event.get("biomechanics") or {}).get("extended_biomechanics") or {}).get("racket_head_speed") or event.get("racket_speed") or {}).get("contact_kmh")
            if isinstance((event.get("extended_biomechanics") or (event.get("biomechanics") or {}).get("extended_biomechanics") or {}).get("racket_head_speed") or event.get("racket_speed"), dict)
            else ((event.get("extended_biomechanics") or (event.get("biomechanics") or {}).get("extended_biomechanics") or {}).get("racket_head_speed") or event.get("racket_speed"))
        ),
        "racket_speed_px_s": _number(
            ((event.get("extended_biomechanics") or (event.get("biomechanics") or {}).get("extended_biomechanics") or {}).get("racket_head_speed") or {}).get("contact_px_s")
        ),
        "latency_hip_to_shoulder_ms": _number(
            ((event.get("extended_biomechanics") or (event.get("biomechanics") or {}).get("extended_biomechanics") or {}).get("kinematic_sequence") or {}).get("latency_hip_to_shoulder_ms")
        ),
        "shoulder_turn_deg": (
            _metric_value(event, "shoulder_turn")
            or _metric_value(event, "shoulder_turn_change")
            or _number((event.get("biomechanics") or {}).get("robust_shoulder_turn_deg"))
        ),
    }


def _indicator(
    name: str,
    label: str,
    baseline: Sequence[Dict],
    recent: Sequence[Dict],
    key: str,
    threshold: float,
    ready: bool,
    higher_is_better: bool = True,
) -> Dict:
    baseline_value = _mean(_number(point.get(key)) for point in baseline)
    recent_value = _mean(_number(point.get(key)) for point in recent)
    delta = (
        recent_value - baseline_value
        if baseline_value is not None and recent_value is not None
        else None
    )
    if baseline_value is None or recent_value is None:
        status = "unavailable"
    elif not ready:
        status = "insufficient_events"
    elif abs(delta) < threshold:
        status = "stable"
    elif (delta > 0) == higher_is_better:
        status = "improving"
    else:
        status = "declining"
    return {
        "name": name,
        "label": label,
        "baseline": _rounded(baseline_value),
        "recent": _rounded(recent_value),
        "delta": _rounded(delta),
        "threshold": threshold,
        "status": status,
    }


def _camera_scale_indicator(
    baseline: Sequence[Dict],
    recent: Sequence[Dict],
    ready: bool,
) -> Dict:
    baseline_value = _median(
        _number(point.get("reference_body_width_px")) for point in baseline
    )
    recent_value = _median(
        _number(point.get("reference_body_width_px")) for point in recent
    )
    relative_delta = (
        (recent_value - baseline_value) / baseline_value
        if baseline_value is not None
        and recent_value is not None
        and abs(baseline_value) > 1e-9
        else None
    )
    if relative_delta is None:
        status = "unavailable"
    elif not ready:
        status = "insufficient_events"
    elif abs(relative_delta) >= 0.15:
        status = "shifted"
    else:
        status = "stable"
    return {
        "name": "camera_scale",
        "label": "人物画面尺度",
        "baseline": _rounded(baseline_value),
        "recent": _rounded(recent_value),
        "relative_delta": _rounded(relative_delta),
        "threshold": 0.15,
        "status": status,
    }


def _quality_grade(score: Optional[float]) -> str:
    if score is None:
        return "unavailable"
    if score >= 75.0:
        return "good"
    if score >= 50.0:
        return "watch"
    return "poor"


def build_session_quality_dashboard(events: Iterable[Dict]) -> Dict:
    """Return one complete session-quality document from ordered events.

    The interface accepts event dictionaries and returns pure JSON data. Drift
    conclusions require six events; shorter sessions expose observations only.
    """
    ordered_events = sorted(
        (event for event in events if isinstance(event, dict)),
        key=lambda event: int(event.get("event_id", 0)),
    )
    series = [_event_point(event, index) for index, event in enumerate(ordered_events)]
    event_count = len(series)
    overlapping_pairs = []
    previous = None
    for point in series:
        if (
            previous is not None
            and previous.get("end_frame") is not None
            and point.get("start_frame") is not None
            and int(point["start_frame"]) <= int(previous["end_frame"])
        ):
            overlapping_pairs.append(
                {
                    "previous_event_id": previous["event_id"],
                    "event_id": point["event_id"],
                    "overlap_frames": (
                        int(previous["end_frame"])
                        - int(point["start_frame"])
                        + 1
                    ),
                }
            )
        previous = point
    window_size = min(MAX_WINDOW_EVENTS, max(1, event_count // 3))
    baseline = series[:window_size]
    recent = series[-window_size:]
    enough_events = event_count >= MIN_DRIFT_EVENTS
    compatible_series = len({p["comparison_key"] for p in series}) <= 1
    drift_ready = enough_events and not overlapping_pairs and compatible_series

    evidence_score = _mean(
        _number(point.get("evidence_quality_100")) for point in series
    )
    calibrated_count = sum(
        1 for point in series if point.get("visible_score_9") is not None
    )
    contact_supported_count = sum(
        1
        for point in series
        if (_number(point.get("contact_evidence_confidence")) or 0.0) >= 0.35
    )
    warning_counts = Counter(
        warning for point in series for warning in point.get("warnings") or []
    )
    advice_counts = Counter(
        code for point in series for code in point.get("advice_codes") or []
    )
    deepseek_status_counts = Counter(
        str(point["deepseek_status"])
        for point in series
        if point.get("deepseek_status")
    )

    indicators = [
        _indicator(
            "visible_technique_score",
            "练习评分",
            baseline,
            recent,
            "practice_score_100",
            0.75 / 9 * 100,
            drift_ready,
        ),
        _indicator(
            "evidence_quality",
            "证据质量",
            baseline,
            recent,
            "evidence_quality_100",
            10.0,
            drift_ready,
        ),
        _indicator(
            "pose_coverage",
            "姿态覆盖",
            baseline,
            recent,
            "pose_frame_ratio",
            0.12,
            drift_ready,
        ),
        _indicator(
            "ball_coverage",
            "网球覆盖",
            baseline,
            recent,
            "ball_frame_ratio",
            0.12,
            drift_ready,
        ),
        _indicator(
            "racket_coverage",
            "球拍覆盖",
            baseline,
            recent,
            "racket_frame_ratio",
            0.12,
            drift_ready,
        ),
        _indicator(
            "contact_evidence",
            "触球证据",
            baseline,
            recent,
            "contact_evidence_confidence",
            0.15,
            drift_ready,
        ),
        _camera_scale_indicator(baseline, recent, drift_ready),
    ]
    indicator_by_name = {indicator["name"]: indicator for indicator in indicators}
    if overlapping_pairs:
        for indicator in indicators:
            if indicator.get("status") != "unavailable":
                indicator["status"] = "event_integrity_confounded"
    if not compatible_series:
        for indicator in indicators:
            indicator["baseline"] = indicator["recent"] = indicator["delta"] = None
            indicator["status"] = "incomparable_series"
    camera_shifted = indicator_by_name["camera_scale"]["status"] == "shifted"
    if camera_shifted:
        technique = indicator_by_name["visible_technique_score"]
        if technique["status"] in {"improving", "declining"}:
            technique["status"] = "camera_shift_confounded"

    alerts = []
    grade = _quality_grade(evidence_score)
    if grade == "poor" and event_count:
        alerts.append(
            {
                "code": "low_session_evidence_quality",
                "domain": "capture",
                "severity": "high",
                "message": "会话证据质量偏低，先检查球拍、网球与触球覆盖",
            }
        )
    elif grade == "watch":
        alerts.append(
            {
                "code": "session_evidence_quality_watch",
                "domain": "capture",
                "severity": "medium",
                "message": "部分动作证据不完整，技术趋势需谨慎解释",
            }
        )
    if drift_ready:
        for indicator in indicators:
            if indicator.get("status") != "declining":
                continue
            alerts.append(
                {
                    "code": f"{indicator['name']}_decline",
                    "domain": (
                        "technique"
                        if indicator["name"] == "visible_technique_score"
                        else "capture"
                    ),
                    "severity": "medium",
                    "message": f"{indicator['label']}较会话前段下降",
                }
            )
        if camera_shifted:
            alerts.append(
                {
                    "code": "camera_scale_shift",
                    "domain": "capture",
                    "severity": "high",
                    "message": "人物画面尺度变化，技术漂移暂不作结论",
                }
            )
    recurring_threshold = max(2, (event_count + 1) // 2)
    recurring_warnings = [
        {"code": code, "count": count, "ratio": round(count / event_count, 4)}
        for code, count in warning_counts.most_common()
        if event_count and count >= recurring_threshold
    ]
    if recurring_warnings:
        alerts.append(
            {
                "code": "recurring_quality_warnings",
                "domain": "capture",
                "severity": "medium",
                "message": "检测警告在多次挥拍中重复出现",
            }
        )
    if overlapping_pairs:
        alerts.append(
            {
                "code": "overlapping_event_ranges",
                "domain": "segmentation",
                "severity": "high",
                "message": "相邻挥拍事件范围重叠，漂移结论已暂停",
            }
        )

    if not compatible_series:
        alerts.append({"code": "incomparable_practice_series", "domain": "context", "severity": "medium",
                       "message": "动作、评分来源或训练条件不同，已暂停合并趋势"})
    drift_status = "warming_up"
    if overlapping_pairs:
        drift_status = "integrity_blocked"
    elif not compatible_series:
        drift_status = "incomparable_series"
    elif drift_ready:
        if alerts:
            drift_status = "attention"
        elif indicator_by_name["visible_technique_score"]["status"] == "improving":
            drift_status = "improving"
        else:
            drift_status = "stable"

    # 疲劳衰减与动作一致性分析 (Fatigue and Consistency Analytics)
    speeds = [
        _number(p.get("racket_speed_kmh")) or _number(p.get("racket_speed_px_s"))
        for p in series
        if (_number(p.get("racket_speed_kmh")) or _number(p.get("racket_speed_px_s"))) is not None
    ]
    latencies = [
        _number(p.get("latency_hip_to_shoulder_ms"))
        for p in series
        if _number(p.get("latency_hip_to_shoulder_ms")) is not None
    ]
    shoulder_turns = [
        _number(p.get("shoulder_turn_deg"))
        for p in series
        if _number(p.get("shoulder_turn_deg")) is not None
    ]

    speed_decay_percent = None
    if len(speeds) >= 4 and baseline and recent:
        base_speeds = [
            _number(p.get("racket_speed_kmh")) or _number(p.get("racket_speed_px_s"))
            for p in baseline
            if (_number(p.get("racket_speed_kmh")) or _number(p.get("racket_speed_px_s"))) is not None
        ]
        rec_speeds = [
            _number(p.get("racket_speed_kmh")) or _number(p.get("racket_speed_px_s"))
            for p in recent
            if (_number(p.get("racket_speed_kmh")) or _number(p.get("racket_speed_px_s"))) is not None
        ]
        if base_speeds and rec_speeds and _mean(base_speeds) and _mean(base_speeds) > 0:
            speed_decay_percent = round(((_mean(rec_speeds) - _mean(base_speeds)) / _mean(base_speeds)) * 100.0, 2)

    speed_std = _stdev(speeds)
    latency_jitter_std = _stdev(latencies)
    shoulder_turn_std = _stdev(shoulder_turns)

    if event_count < 4:
        fatigue_status = "WARMING_UP"
    elif speed_decay_percent is not None and speed_decay_percent <= -8.0:
        fatigue_status = "FATIGUE_OBSERVED"
    elif speed_decay_percent is not None and speed_decay_percent >= 5.0:
        fatigue_status = "WARMED_UP"
    else:
        fatigue_status = "CONSISTENT"

    fatigue_and_consistency = {
        "speed_decay_percent": speed_decay_percent,
        "speed_std": speed_std,
        "latency_jitter_std_ms": latency_jitter_std,
        "shoulder_turn_std_deg": shoulder_turn_std,
        "fatigue_status": fatigue_status,
        "samples_evaluated": len(speeds),
    }

    ready_deepseek = int(deepseek_status_counts.get("ready", 0))
    attempted_deepseek = sum(deepseek_status_counts.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "event_count": event_count,
        "monitoring_state": (
            "ready"
            if drift_ready
            else "blocked"
            if overlapping_pairs or not compatible_series
            else "warming_up"
        ),
        "quality": {
            "evidence_quality_score_100": _rounded(evidence_score, 2),
            "grade": grade,
            "practice_score_mean_100": _rounded(_mean(point["practice_score_100"] for point in series)) if len({p["comparison_key"] for p in series}) <= 1 else None,
            "visible_technique_mean_9": _rounded(
                _mean(_number(point.get("visible_score_9")) for point in series),
                2,
            ) if compatible_series else None,
            "visible_technique_median_9": _rounded(
                _median(_number(point.get("visible_score_9")) for point in series),
                2,
            ) if compatible_series else None,
            "median_uncertainty_100": _rounded(
                _median(point.get("score_uncertainty_100") for point in series), 2,
            ) if compatible_series else None,
            "median_uncertainty_9": _rounded(
                _median(_number(point.get("score_uncertainty_9")) for point in series),
                2,
            ),
            "calibrated_event_ratio": round(calibrated_count / max(1, event_count), 4),
            "contact_supported_ratio": round(
                contact_supported_count / max(1, event_count),
                4,
            ),
            "review_recommended_ratio": round(
                sum(1 for point in series if point.get("review_recommended"))
                / max(1, event_count),
                4,
            ),
            "mean_pose_frame_ratio": _rounded(
                _mean(_number(point.get("pose_frame_ratio")) for point in series)
            ),
            "mean_ball_frame_ratio": _rounded(
                _mean(_number(point.get("ball_frame_ratio")) for point in series)
            ),
            "mean_racket_frame_ratio": _rounded(
                _mean(_number(point.get("racket_frame_ratio")) for point in series)
            ),
        },
        "operations": {
            "local_coach_latency_p50_ms": _rounded(
                _median(
                    _number(point.get("contact_capture_to_coach_ms"))
                    for point in series
                ),
                2,
            ),
            "deepseek_ready_ratio": (
                round(ready_deepseek / attempted_deepseek, 4)
                if attempted_deepseek
                else None
            ),
            "deepseek_latency_p50_ms": _rounded(
                _median(_number(point.get("deepseek_latency_ms")) for point in series),
                2,
            ),
            "deepseek_status_counts": dict(sorted(deepseek_status_counts.items())),
        },
        "integrity": {
            "overlapping_event_pair_count": len(overlapping_pairs),
            "overlapping_event_pairs": overlapping_pairs,
            "event_ranges_non_overlapping": not overlapping_pairs,
        },
        "drift": {
            "status": drift_status,
            "ready": drift_ready,
            "minimum_event_count": MIN_DRIFT_EVENTS,
            "window_size": window_size,
            "baseline_event_ids": [point["event_id"] for point in baseline],
            "recent_event_ids": [point["event_id"] for point in recent],
            "indicators": indicators,
            "camera_confounded": camera_shifted,
        },
        "fatigue_and_consistency": fatigue_and_consistency,
        "recurring": {
            "warnings": recurring_warnings,
            "warning_counts": dict(sorted(warning_counts.items())),
            "advice_counts": dict(sorted(advice_counts.items())),
        },
        "alerts": alerts,
        "series": series,
        "limitations": [
            "drift_requires_at_least_six_events",
            "technique_trend_is_single_view_visible_motion_only",
            "camera_scale_shift_blocks_technique_drift_conclusions",
            "compare_sessions_only_after_camera_calibration",
        ],
    }
