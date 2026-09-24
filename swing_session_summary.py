"""Multi-ball session-level coaching summary and macro diagnostic engine.

Aggregates individual swing events across a training session into:
1. Stroke distribution (Forehand, Backhand, Shadow swings)
2. Quality score trend, average, variance, and consistency ratings
3. Common technical deficiencies (occurrence count, percentage, severity)
4. Macro coach narrative and actionable prescription for next drills
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _is_shadow(ev: Dict[str, Any]) -> bool:
    if ev.get("is_shadow_swing") is not None:
        return bool(ev["is_shadow_swing"])
    evidence = ev.get("evidence") or {}
    ca = evidence.get("contact_analysis") or (evidence.get("classification_context") or {}).get("contact_analysis") or {}
    if ca.get("is_shadow_swing") is not None:
        return bool(ca["is_shadow_swing"])
    if ca.get("has_ball") is False:
        return True
    return False


def _get_score(ev: Dict[str, Any]) -> Optional[float]:
    score = ev.get("swing_score")
    if score is None:
        score = (ev.get("biomechanics") or {}).get("swing_score")
    try:
        return float(score) if score is not None else None
    except (TypeError, ValueError):
        return None


def _get_grade(ev: Dict[str, Any]) -> str:
    grade = ev.get("swing_grade")
    if not grade:
        grade = (ev.get("biomechanics") or {}).get("swing_grade")
    return str(grade or "DEVELOPING")


def _get_speed(ev: Dict[str, Any]) -> Optional[float]:
    ext = ev.get("extended_biomechanics") or (ev.get("biomechanics") or {}).get("extended_biomechanics") or {}
    speed = (ext.get("racket_head_speed") or {}).get("contact_kmh")
    if speed is None:
        speed = (ext.get("racket_head_speed") or {}).get("max_kmh")
    try:
        return float(speed) if speed is not None else None
    except (TypeError, ValueError):
        return None


def calculate_radar_dimensions(ev: Dict[str, Any]) -> Dict[str, float]:
    """Calculate 5-dimension normalized scores (0-100) for a single swing."""
    ext = ev.get("extended_biomechanics") or (ev.get("biomechanics") or {}).get("extended_biomechanics") or {}
    
    # 1. Speed (normalized to 0-100, where 75 km/h is 85 pts)
    speed_kmh = _get_speed(ev) or 0.0
    dim_speed = min(100.0, max(20.0, (speed_kmh / 85.0) * 100.0))

    # 2. Brush & Drop (normalized from low_to_high_angle_deg and drop ratio)
    brush = ext.get("brush_angle") or {}
    angle = float(brush.get("low_to_high_angle_deg") or 0.0)
    drop = float(brush.get("drop_depth_ratio") or 0.0)
    dim_brush = min(100.0, max(20.0, (min(angle, 60.0) / 60.0 * 60.0) + (min(drop, 0.4) / 0.4 * 40.0)))

    # 3. Kinematic Sequence
    seq = ext.get("kinematic_sequence") or {}
    quality = seq.get("sequence_quality")
    if quality == "OPTIMAL":
        dim_kinematics = 92.0
    elif quality == "ACCEPTABLE":
        dim_kinematics = 75.0
    else:
        dim_kinematics = 45.0

    # 4. Leg Drive
    leg = ext.get("leg_drive") or {}
    drive_ratio = float(leg.get("drive_ratio") or 1.0)
    dim_leg = min(100.0, max(25.0, (drive_ratio / 1.6) * 85.0))

    # 5. Preparation & Quality
    raw_score = _get_score(ev) or 60.0
    dim_prep = min(100.0, max(20.0, raw_score * 0.95))

    return {
        "speed": round(dim_speed, 1),
        "brush": round(dim_brush, 1),
        "kinematics": round(dim_kinematics, 1),
        "leg_drive": round(dim_leg, 1),
        "preparation": round(dim_prep, 1),
    }


def build_session_coaching_summary(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate session-level swing statistics, score trends, and macro coach diagnosis."""
    ordered = sorted(
        [e for e in events if isinstance(e, dict)],
        key=lambda x: int(x.get("event_id") or 0)
    )
    total_swings = len(ordered)
    if total_swings == 0:
        return {
            "total_swings": 0,
            "valid_shots_count": 0,
            "distribution": {
                "forehand_count": 0,
                "backhand_count": 0,
                "shadow_count": 0,
                "other_count": 0,
                "forehand_ratio": 0.0,
                "backhand_ratio": 0.0,
                "shadow_ratio": 0.0,
            },
            "quality_metrics": {
                "average_score": None,
                "score_std": None,
                "min_score": None,
                "max_score": None,
                "stability_rating": "NO_DATA",
                "stability_label": "暂无数据",
            },
            "score_trends": [],
            "common_deficiencies": [],
            "macro_diagnosis": "当前会话暂无击球事件记录。",
            "radar_averages": {
                "speed": 0.0,
                "brush": 0.0,
                "kinematics": 0.0,
                "leg_drive": 0.0,
                "preparation": 0.0,
            },
        }

    forehand_count = 0
    backhand_count = 0
    shadow_count = 0
    other_count = 0

    valid_scores: List[float] = []
    score_trends = []
    deficiency_counter: Counter[str] = Counter()
    advice_info: Dict[str, Dict[str, Any]] = {}
    radar_accumulator: Dict[str, float] = defaultdict(float)
    radar_count = 0

    for ev in ordered:
        eid = int(ev.get("event_id") or 0)
        stroke = str(ev.get("stroke_type") or "Swing")
        shadow = _is_shadow(ev)
        score = _get_score(ev)
        grade = _get_grade(ev)
        speed = _get_speed(ev)

        if shadow:
            shadow_count += 1
        elif "Forehand" in stroke:
            forehand_count += 1
        elif "Backhand" in stroke:
            backhand_count += 1
        else:
            other_count += 1

        if not shadow and score is not None:
            valid_scores.append(score)

        # Accumulate radar
        if not shadow:
            dims = calculate_radar_dimensions(ev)
            for k, v in dims.items():
                radar_accumulator[k] += v
            radar_count += 1

        score_trends.append({
            "event_id": eid,
            "stroke_type": stroke,
            "score": score,
            "grade": grade,
            "speed_kmh": speed,
            "is_shadow": shadow,
        })

        # Process coach advices for valid swings
        if not shadow:
            advices = ev.get("coach_advices") or ([] if not ev.get("coach_advice") else [ev["coach_advice"]])
            for adv in advices:
                if not isinstance(adv, dict):
                    continue
                code = str(adv.get("code") or adv.get("focus") or "").strip()
                msg = str(adv.get("message") or code).strip()
                if not code and not msg:
                    continue
                key = code or msg
                deficiency_counter[key] += 1
                if key not in advice_info:
                    advice_info[key] = {
                        "code": code,
                        "message": msg,
                        "priority": adv.get("priority", 1),
                    }

    valid_shots_count = len(valid_scores)
    forehand_ratio = round((forehand_count / total_swings) * 100, 1)
    backhand_ratio = round((backhand_count / total_swings) * 100, 1)
    shadow_ratio = round((shadow_count / total_swings) * 100, 1)

    # Calculate statistics
    if valid_shots_count > 0:
        avg_score = round(sum(valid_scores) / valid_shots_count, 1)
        min_score = round(min(valid_scores), 1)
        max_score = round(max(valid_scores), 1)
        if valid_shots_count > 1:
            variance = sum((s - avg_score) ** 2 for s in valid_scores) / (valid_shots_count - 1)
            score_std = round(math.sqrt(variance), 2)
        else:
            score_std = 0.0

        if score_std < 5.0:
            stability_rating = "HIGH_CONSISTENCY"
            stability_label = "极高稳定性 (动作品质稳定)"
        elif score_std < 10.0:
            stability_rating = "MODERATE_VARIANCE"
            stability_label = "良好稳定性 (轻微波动)"
        else:
            stability_rating = "HIGH_VARIANCE"
            stability_label = "波动较大 (技术动作待定型)"
    else:
        avg_score = None
        score_std = None
        min_score = None
        max_score = None
        stability_rating = "ONLY_SHADOW"
        stability_label = "全为空挥试拍"

    # Radar averages
    radar_averages = {
        k: round(radar_accumulator[k] / radar_count, 1) if radar_count > 0 else 0.0
        for k in ["speed", "brush", "kinematics", "leg_drive", "preparation"]
    }

    # Common deficiencies
    common_deficiencies = []
    for key, count in deficiency_counter.most_common():
        rate = round((count / valid_shots_count) * 100, 1) if valid_shots_count > 0 else 0.0
        info = advice_info.get(key, {})
        severity = "HIGH" if rate >= 60.0 else ("MEDIUM" if rate >= 30.0 else "LOW")
        common_deficiencies.append({
            "key": key,
            "code": info.get("code") or key,
            "message": info.get("message") or key,
            "count": count,
            "occurrence_rate_percent": rate,
            "severity": severity,
        })

    # Generate macro coaching diagnostic narrative
    parts = []
    parts.append(f"本节训练共完成 {total_swings} 次挥拍")
    dist_desc = []
    if forehand_count > 0:
        dist_desc.append(f"正手 {forehand_count} 球 ({forehand_ratio}%)")
    if backhand_count > 0:
        dist_desc.append(f"反手 {backhand_count} 球 ({backhand_ratio}%)")
    if shadow_count > 0:
        dist_desc.append(f"空挥试拍 {shadow_count} 次 ({shadow_ratio}%)")
    if dist_desc:
        parts.append(f"（包含 { '，'.join(dist_desc)}）。")

    if valid_shots_count > 0:
        parts.append(f"击球平均技术质量得分为 {avg_score} 分（{stability_label}）。")
        if common_deficiencies:
            top_def = common_deficiencies[:2]
            top_desc = "、".join([f"{d['message']}（出现率 {d['occurrence_rate_percent']}%）" for d in top_def])
            parts.append(f"学员技术短板主要集中在：{top_desc}。")
            # Suggestion prescription
            prescriptions = []
            for d in top_def:
                code_lower = d["code"].lower()
                msg = d["message"]
                if "knee" in code_lower or "重心" in msg:
                    prescriptions.append("在引拍蓄力期主动屈膝降低重心，建立坚实的下肢支撑")
                elif "kinematic" in code_lower or "核心" in msg:
                    prescriptions.append("避免手臂过早主动发力，依靠躯干转体带动拍头甩出")
                elif "brush" in code_lower or "下潜" in msg:
                    prescriptions.append("在击球前让拍头沉于来球下方，向上刷球制造充足过网上旋")
                elif "prep" in code_lower or "引拍" in msg:
                    prescriptions.append("尽早侧身完成引拍架拍，提升击球点击球时效")
                else:
                    prescriptions.append(f"专项强化针对 {msg} 的技术微调")
            if prescriptions:
                parts.append(f"下阶段训练处方建议：{'；'.join(prescriptions)}。")
        else:
            parts.append("击球动作整体规范，未检测到显著技术短板，建议进入下一阶段加力与控球练习！")
    else:
        parts.append("本节全为空挥试拍或未检测到有效来球，建议进入实战击球环节。")

    macro_diagnosis = "".join(parts)

    return {
        "total_swings": total_swings,
        "valid_shots_count": valid_shots_count,
        "distribution": {
            "forehand_count": forehand_count,
            "backhand_count": backhand_count,
            "shadow_count": shadow_count,
            "other_count": other_count,
            "forehand_ratio": forehand_ratio,
            "backhand_ratio": backhand_ratio,
            "shadow_ratio": shadow_ratio,
        },
        "quality_metrics": {
            "average_score": avg_score,
            "score_std": score_std,
            "min_score": min_score,
            "max_score": max_score,
            "stability_rating": stability_rating,
            "stability_label": stability_label,
        },
        "score_trends": score_trends,
        "common_deficiencies": common_deficiencies,
        "macro_diagnosis": macro_diagnosis,
        "radar_averages": radar_averages,
    }
