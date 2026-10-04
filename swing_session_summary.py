"""Multi-ball session-level coaching summary and macro diagnostic engine.

Aggregates individual swing events across a training session into:
1. Stroke distribution (Forehand, Backhand, Shadow swings)
2. Quality score trend, average, variance, and consistency ratings
3. Common technical deficiencies (occurrence count, percentage, severity)
4. Macro coach narrative and actionable prescription for next drills
"""

from __future__ import annotations

import math
import json
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from practice_scoring import score_review, DIMENSIONS, POLICY_VERSION
from practice_score_adapter import resolve_practice_score
from analysis_metric_delivery import session_analysis_metrics


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


def _get_practice_score(ev: Dict[str, Any]) -> Dict[str, Any]:
    return resolve_practice_score(ev)


def _get_score(ev: Dict[str, Any]) -> Optional[float]:
    return _get_practice_score(ev).get("score")


def _get_grade(ev: Dict[str, Any]) -> Optional[str]:
    return _get_practice_score(ev).get("grade")


def _get_speed(ev: Dict[str, Any]) -> Optional[float]:
    ext = ev.get("extended_biomechanics") or (ev.get("biomechanics") or {}).get("extended_biomechanics") or {}
    speed = (ext.get("racket_head_speed") or {}).get("contact_kmh")
    if speed is None:
        speed = (ext.get("racket_head_speed") or {}).get("max_kmh")
    try:
        return float(speed) if speed is not None else None
    except (TypeError, ValueError):
        return None


def calculate_radar_dimensions(ev: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Only explicit coach ratings populate the five practice dimensions."""
    result = _get_practice_score(ev)
    if result["method"] != "coach_manual":
        return {key: None for key in DIMENSIONS}
    result = score_review({"ratings": result.get("ratings"),
                           "confirmed": result.get("status") == "coach_confirmed"})
    return {key: value * 20.0 if value is not None and result["status"] == "coach_confirmed" else None
            for key, value in result["ratings"].items()}


def _series_key(ev):
    result = _get_practice_score(ev)
    context = ev.get("practice_context") or {}
    return json.dumps({"policy": result["policy_version"], "method": result["method"],
                       "stroke": ev.get("stroke_type"), "context": context}, sort_keys=True)


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
            "radar_averages": {key: None for key in DIMENSIONS},
            "scoring_policy_version": POLICY_VERSION,
            "score_series": [],
            "analysis_metrics": [],
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
    radar_counts = defaultdict(int)
    score_groups = defaultdict(list)

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
            score_groups[_series_key(ev)].append(score)

        # Accumulate radar
        if not shadow:
            dims = calculate_radar_dimensions(ev)
            for k, v in dims.items():
                if v is not None:
                    radar_accumulator[k] += v
                    radar_counts[k] += 1

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
                category = str(adv.get("category") or "").strip().lower()
                code = str(adv.get("code") or adv.get("focus") or "").strip()
                msg = str(adv.get("message") or code).strip()
                if not code and not msg:
                    continue
                # Common deficiencies must only represent real technique shortcomings!
                # Exclude praise/maintain form, and exclude vision capture/review operational messages
                if category in ("review", "capture", "positive") or code == "maintain_form":
                    continue
                if "复核" in msg or "保持" in msg or "入镜" in msg or "遮挡" in msg:
                    continue

                # Group by normalized message so identical messages with different codes don't duplicate
                key = msg
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

        if len(score_groups) > 1:
            avg_score = score_std = min_score = max_score = None
            stability_rating = "INCOMPARABLE_SERIES"
            stability_label = "分动作与训练条件查看，不能合并比较"
        elif valid_shots_count < 3:
            score_std = None
            stability_rating = "INSUFFICIENT_SAMPLES"
            stability_label = "样本不足，暂不评价稳定性"
        elif score_std < 5.0:
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
        stability_rating = "INSUFFICIENT_EVIDENCE"
        stability_label = "暂无可评分证据"

    # Radar averages
    radar_averages = {
        k: round(radar_accumulator[k] / radar_counts[k], 1) if radar_counts[k] and len(score_groups) <= 1 else None
        for k in DIMENSIONS
    }

    # Common deficiencies
    common_deficiencies = []
    for key, count in deficiency_counter.most_common():
        rate = round((count / max(1, total_swings - shadow_count)) * 100, 1) if total_swings > shadow_count else 0.0
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
        parts.append(f"动作参考平均分 {avg_score}（{stability_label}）。" if avg_score is not None else f"{stability_label}。")
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
                elif "arm" in code_lower or "舒展" in msg:
                    prescriptions.append("在击球点击球时主动舒展手臂，避免过度屈肘，增大击球力矩")
                elif "kinematic" in code_lower or "核心" in msg:
                    prescriptions.append("避免手臂过早主动发力，依靠躯干转体带动拍头甩出")
                elif "brush" in code_lower or "下潜" in msg:
                    prescriptions.append("在击球前让拍头沉于来球下方，向上刷球制造充足过网上旋")
                elif "leg" in code_lower or "蹬地" in msg:
                    prescriptions.append("向前挥拍击球瞬间加强双腿垂直蹬地，利用地面反作用力加速发力")
                elif "separation" in code_lower or "肩髋" in msg:
                    prescriptions.append("加大引拍期的肩髋扭转分离角，蓄积更充分的核心弹性势能")
                elif "balance" in code_lower or "稳住" in msg:
                    prescriptions.append("击球后保持身体核心平衡，避免击球瞬间重心剧烈偏移")
                elif "takeback" in code_lower or "后背" in msg:
                    prescriptions.append("在准备期充分向后展开后背引拍，延长挥拍加速做功距离")
                elif "prep" in code_lower or "引拍" in msg:
                    prescriptions.append("尽早侧身完成引拍架拍，提升击球点击球时效")
                elif "follow" in code_lower or "随挥" in msg:
                    prescriptions.append("击球后保持随挥动作完整顺畅，保证出球深度与弧线控制")
                else:
                    prescriptions.append(f"针对 {msg} 进行专项技术微调强化")
            if prescriptions:
                parts.append(f"下阶段训练处方建议：{'；'.join(prescriptions)}。")
        else:
            parts.append("当前证据未产生技术纠错建议，仍需教练复核。")
    else:
        parts.append("全为空挥试拍。" if shadow_count == total_swings else "证据不足，暂不评分；请复核采集与动作标记。")

    if not valid_shots_count and common_deficiencies:
        parts.append("已有动作建议：" + "、".join(d["message"] for d in common_deficiencies) + "。")
    macro_diagnosis = "".join(parts)

    return {
        "total_swings": total_swings,
        "analysis_metrics": session_analysis_metrics([e for e in ordered if not _is_shadow(e)]),
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
        "radar_available_counts": dict(radar_counts),
        "scoring_policy_version": POLICY_VERSION,
        "score_series": [{"key": json.loads(key), "count": len(values),
                          "average_score": round(sum(values) / len(values), 1)}
                         for key, values in score_groups.items()],
    }
