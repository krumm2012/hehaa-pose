"""SQLite persistence for tennis sessions, players, swing events, and cross-session comparison."""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from analysis_metric_delivery import normalize_swing_event_for_delivery


DEFAULT_PLAYER_ID = "default"
DEFAULT_PLAYER_NAME = "默认球员"
MAX_PLAUSIBLE_RACKET_SPEED_PX_S = 5000.0
_PLAYER_ID_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
_SESSION_ID_SAFE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def _session_dedup_key(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (dedup_key, match_by, human_label) for local video sessions; None for live streams."""
    sha = str(row.get("source_sha256") or "").strip().lower()
    src_name = str(row.get("source_name") or "").strip()
    if sha:
        label = Path(src_name).name if src_name else f"sha256:{sha[:12]}"
        return f"sha256:{sha}", "source_sha256", label
    lower_src = src_name.lower()
    if src_name and "://" not in lower_src and Path(src_name).suffix.lower() in _VIDEO_SUFFIXES:
        base = Path(src_name).name
        return f"name:{base.lower()}", "source_name", base
    return None, None, None


def _canonical_session_rank(row: Dict[str, Any]) -> Tuple[int, int, int, float]:
    """Rank sessions so dual_view > higher matched/total swings > newer timestamp."""
    is_dual = 1 if (row.get("is_dual_view") or str(row.get("view_mode") or "") == "dual_view") else 0
    matched = int(row.get("forehand_count") or 0) + int(row.get("backhand_count") or 0)
    ev_cnt = int(row.get("event_count") or 0)
    ts = float(row.get("timestamp") or 0.0)
    return (is_dual, max(matched, ev_cnt), ev_cnt, ts)


def _group_duplicate_sessions(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Group sessions that share the same source video and choose one canonical session per group."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    meta_by_key: Dict[str, Tuple[str, str]] = {}
    order_keys: List[str] = []

    for idx, row in enumerate(rows):
        dkey, match_by, label = _session_dedup_key(row)
        if not dkey:
            dkey = f"unique:{row.get('session_id') or idx}"
            match_by = "unique"
            label = str(row.get("source_name") or row.get("session_id") or "")
        if dkey not in buckets:
            buckets[dkey] = []
            meta_by_key[dkey] = (match_by or "unique", label or "")
            order_keys.append(dkey)
        buckets[dkey].append(row)

    kept_ids = set()
    kept_duplicates_map: Dict[str, List[str]] = {}
    duplicate_groups: List[Dict[str, Any]] = []
    hidden_ids: List[str] = []

    for dkey in order_keys:
        grp = buckets[dkey]
        if len(grp) <= 1:
            kept_ids.add(str(grp[0]["session_id"]))
            continue
        sorted_grp = sorted(grp, key=_canonical_session_rank, reverse=True)
        canonical = sorted_grp[0]
        dups = sorted_grp[1:]
        c_sid = str(canonical["session_id"])
        kept_ids.add(c_sid)
        dup_sids = [str(d["session_id"]) for d in dups]
        kept_duplicates_map[c_sid] = dup_sids
        hidden_ids.extend(dup_sids)
        match_by, label = meta_by_key[dkey]
        source_names = sorted({str(x.get("source_name") or "").strip() for x in grp if str(x.get("source_name") or "").strip()})
        duplicate_groups.append(
            {
                "dedup_key": dkey,
                "match_by": match_by,
                "label": label,
                "source_sha256": canonical.get("source_sha256"),
                "source_names": source_names,
                "total_count": len(grp),
                "canonical": canonical,
                "duplicates": dups,
            }
        )

    kept_rows = [r for r in rows if str(r["session_id"]) in kept_ids]
    summary = {
        "raw_session_count": len(rows),
        "unique_session_count": len(kept_rows),
        "duplicate_session_count": len(hidden_ids),
        "duplicate_group_count": len(duplicate_groups),
        "hidden_session_ids": hidden_ids,
        "kept_duplicates_map": kept_duplicates_map,
        "groups": duplicate_groups,
    }
    return kept_rows, summary


def _sanitize_racket_speeds(
    contact_val: Any,
    max_val: Any,
) -> Tuple[Optional[float], Optional[float]]:
    c = _safe_float(contact_val)
    m = _safe_float(max_val)
    if c is not None and (c <= 0 or c > MAX_PLAUSIBLE_RACKET_SPEED_PX_S):
        c = None
    if m is not None and (m <= 0 or m > MAX_PLAUSIBLE_RACKET_SPEED_PX_S):
        m = c
    return c, (m if m is not None else c)


def _stroke_sql_clause(stroke_filter: str, column: str = "stroke_type") -> Tuple[str, List[Any]]:
    token = str(stroke_filter or "").strip().lower()
    if not token or token == "all":
        return "", []
    if token == "forehand":
        return f" AND LOWER({column}) LIKE ?", ["%forehand%"]
    if token == "backhand":
        return f" AND LOWER({column}) LIKE ?", ["%backhand%"]
    return f" AND LOWER({column}) = LOWER(?)", [stroke_filter]


def _classify_stroke_side(stroke_type: Any) -> str:
    lowered = str(stroke_type or "").strip().lower()
    if "forehand" in lowered:
        return "forehand"
    if "backhand" in lowered:
        return "backhand"
    return "other"


def _extract_payload_media_urls(
    payload: Dict[str, Any],
    session_dir: Optional[Path],
    workspace_root: Path,
) -> Tuple[str, str]:
    clip_url = str(payload.get("clip_url") or "")
    if not clip_url and payload.get("clip_path") and session_dir is not None:
        raw_cp = Path(payload["clip_path"])
        cp = raw_cp if raw_cp.is_absolute() else (session_dir / raw_cp)
        if cp.exists():
            try:
                clip_url = f"/artifacts/{quote(str(cp.resolve().relative_to(workspace_root)))}"
            except ValueError:
                pass

    freeze_url = str(payload.get("impact_freeze_url") or "")
    if not freeze_url and payload.get("impact_freeze_path") and session_dir is not None:
        raw_fp = Path(payload["impact_freeze_path"])
        fp = raw_fp if raw_fp.is_absolute() else (session_dir / raw_fp)
        if fp.exists():
            try:
                freeze_url = f"/artifacts/{quote(str(fp.resolve().relative_to(workspace_root)))}"
            except ValueError:
                pass
    return clip_url, freeze_url


_GLOBAL_METRIC_SPECS: List[Tuple[str, str, str, bool, str]] = [
    ("shoulder_turn_change", "转肩角度变化", "°", True, "all"),
    ("hip_shoulder_separation", "肩髋分离角", "°", True, "all"),
    ("arm_extension", "持拍侧肘伸展", "°", True, "all"),
    ("preparation_knee_flexion", "准备期屈膝角", "°", False, "all"),
    ("contact_lateral_distance", "触球离体距离", "bw", True, "all"),
    ("weight_transfer", "躯干重心横移", "bw", True, "all"),
    ("balance_drift", "回位期重心漂移", "bw", False, "all"),
    ("stance", "足部站位倾角", "°", True, "all"),
    ("leg_drive", "髋部上移蹬伸", "ratio", True, "all"),
    ("scapular_retraction", "正背肩宽比", "ratio", True, "dual_view_only"),
    ("takeback_depth", "镜面引拍深度比", "ratio", True, "dual_view_only"),
    ("racket_max_speed", "拍框峰值速度", "px/s", True, "all"),
]


def _extract_swing_warnings(
    payload: Dict[str, Any],
    ev_metrics: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Return (explicit_advice_messages, effective_warning_labels) for one swing."""
    advice_msgs: List[str] = []
    for adv in payload.get("coach_advices") or []:
        if isinstance(adv, dict):
            msg = str(adv.get("message") or adv.get("code") or "").strip()
            if msg:
                advice_msgs.append(msg)
    if advice_msgs:
        return advice_msgs, list(advice_msgs)

    derived: List[str] = []
    turn_v = _safe_float(ev_metrics.get("shoulder_turn_change"))
    ext_v = _safe_float(ev_metrics.get("arm_extension"))
    knee_v = _safe_float(ev_metrics.get("preparation_knee_flexion"))
    drift_v = _safe_float(ev_metrics.get("balance_drift"))
    if turn_v is not None and turn_v < 35.0:
        derived.append("转肩幅度不足 (<35°)")
    if ext_v is not None and ext_v < 110.0:
        derived.append("击球手臂未充分舒展 (<110°)")
    if knee_v is not None and knee_v > 164.0:
        derived.append("准备期直膝蓄力不足 (>164°)")
    if drift_v is not None and drift_v > 0.35:
        derived.append("回位期重心漂移偏大 (>0.35bw)")
    return advice_msgs, derived


def _analyze_side_patterns(
    swings: List[Dict[str, Any]],
    trend_series: List[Dict[str, Any]],
    metrics_out: Dict[str, Any],
    coach_warnings: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Detect cross-session commonalities and contradictions within a single stroke side."""
    total_sessions = len(trend_series)
    commonalities: List[Dict[str, Any]] = []
    contradictions: List[Dict[str, Any]] = []

    if not swings:
        return {"commonalities": commonalities, "contradictions": contradictions}

    # 1. Persistent warnings across multiple sessions
    warn_sessions: Dict[str, set] = {}
    for sw in swings:
        sid = sw["session_id"]
        for w in set(sw.get("warnings") or []):
            warn_sessions.setdefault(w, set()).add(sid)

    for w in coach_warnings:
        msg = w["message"]
        s_cnt = len(warn_sessions.get(msg, set()))
        if s_cnt >= 2 and (s_cnt / max(1, total_sessions) >= 0.5 or w["rate_pct"] >= 40.0):
            commonalities.append(
                {
                    "type": "persistent_flaw",
                    "label": msg,
                    "session_count": s_cnt,
                    "total_sessions": total_sessions,
                    "rate_pct": w["rate_pct"],
                    "summary": f"在 {s_cnt}/{total_sessions} 场训练中反复出现（累计占比 {w['rate_pct']}%）",
                }
            )

    # 2. Metric stability (commonalities) vs cross-session drift & reversals (contradictions)
    for m_key, lbl, unit, _higher_better, scope in _GLOBAL_METRIC_SPECS:
        m = metrics_out.get(m_key)
        if not m:
            continue
        med = float(m["median"])
        iqr = float(m["iqr"])
        c_score = m.get("consistency_score")
        sess_pts = [
            (pt["session_id"], pt["display_time"], float(pt[m_key]), int(pt.get("count") or 0))
            for pt in trend_series
            if pt.get(m_key) is not None
        ]
        if len(sess_pts) < 2:
            if m["count"] >= 3 and c_score is not None and c_score >= 65.0:
                commonalities.append(
                    {
                        "type": "stable_habit",
                        "key": m_key,
                        "label": lbl,
                        "unit": unit,
                        "view_scope": scope,
                        "median": med,
                        "iqr": iqr,
                        "consistency_score": c_score,
                        "session_count": len(sess_pts) or 1,
                        "total_sessions": total_sessions,
                        "summary": f"多拍稳定在 {med} {unit}（±{iqr}，定型度 {c_score}%）",
                    }
                )
            continue

        sess_vals = [p[2] for p in sess_pts]
        sess_iqr = _compute_iqr(sess_vals)
        denom = max(abs(med), 1.0)
        rel_sess_iqr = sess_iqr / denom

        min_pt = min(sess_pts, key=lambda p: p[2])
        max_pt = max(sess_pts, key=lambda p: p[2])
        delta = round(max_pt[2] - min_pt[2], 2)
        rel_spread = abs(delta) / denom
        is_drift = rel_spread >= 0.45 or (c_score is not None and c_score < 55.0 and abs(delta) > 0)

        if (not is_drift) and (
            (c_score is not None and c_score >= 60.0)
            or (c_score is None and len(sess_pts) >= 3 and rel_sess_iqr <= 0.22)
        ):
            commonalities.append(
                {
                    "type": "stable_habit",
                    "key": m_key,
                    "label": lbl,
                    "unit": unit,
                    "view_scope": scope,
                    "median": med,
                    "iqr": iqr,
                    "consistency_score": c_score,
                    "session_count": len(sess_pts),
                    "total_sessions": total_sessions,
                    "summary": f"跨越 {len(sess_pts)} 场稳定保持在 {med} {unit}（±{iqr}{f'，定型度 {c_score}%' if c_score is not None else ''}）",
                }
            )

        if is_drift:
            contradictions.append(
                {
                    "type": "session_drift",
                    "severity": "high" if (rel_spread >= 0.70 or (c_score is not None and c_score < 40.0)) else "medium",
                    "key": m_key,
                    "label": lbl,
                    "unit": unit,
                    "view_scope": scope,
                    "min_session": {"session_id": min_pt[0], "display_time": min_pt[1], "value": round(min_pt[2], 2)},
                    "max_session": {"session_id": max_pt[0], "display_time": max_pt[1], "value": round(max_pt[2], 2)},
                    "delta": delta,
                    "rel_spread_pct": round(rel_spread * 100.0, 1),
                    "consistency_score": c_score,
                    "summary": f"跨场基准明显不一致：{min_pt[1][:16]} 仅 {round(min_pt[2], 2)} {unit}，而 {max_pt[1][:16]} 达到 {round(max_pt[2], 2)} {unit}（极差 +{delta} {unit}）",
                }
            )

        # Early -> Mid -> Recent reversal check (V-shape or ^-shape)
        if len(sess_pts) >= 4:
            k3 = max(1, len(sess_pts) // 3)
            e_m = float(statistics.median([p[2] for p in sess_pts[:k3]]))
            m_m = float(statistics.median([p[2] for p in sess_pts[k3 : 2 * k3]]))
            l_m = float(statistics.median([p[2] for p in sess_pts[2 * k3 :]]))
            leg1 = round(m_m - e_m, 2)
            leg2 = round(l_m - m_m, 2)
            thresh = max(abs(med) * 0.15, abs(delta) * 0.25, 1e-3)
            if leg1 * leg2 < 0 and abs(leg1) >= thresh and abs(leg2) >= thresh:
                direction = "先升高后回落" if leg1 > 0 else "先下降后回升"
                contradictions.append(
                    {
                        "type": "trend_reversal",
                        "severity": "medium",
                        "key": m_key,
                        "label": lbl,
                        "unit": unit,
                        "view_scope": scope,
                        "early_median": round(e_m, 2),
                        "mid_median": round(m_m, 2),
                        "recent_median": round(l_m, 2),
                        "summary": f"前期到近期走势矛盾（{direction}）：早期 {round(e_m, 2)} → 中期 {round(m_m, 2)} → 近期 {round(l_m, 2)} {unit}",
                    }
                )

    dual_cnt = sum(1 for sw in swings if sw.get("view_mode") == "dual_view")
    single_cnt = len(swings) - dual_cnt
    if dual_cnt > 0 and single_cnt > 0:
        contradictions.append(
            {
                "type": "view_heterogeneity",
                "severity": "medium",
                "key": "view_mode",
                "label": "拍摄视角口径混杂",
                "summary": f"包含 {dual_cnt} 拍双视角与 {single_cnt} 拍单视角录像，部分单视角场次缺少镜面纵深指标且投影角略有差异。",
            }
        )

    commonalities.sort(
        key=lambda c: (0 if c["type"] == "persistent_flaw" else 1, -(c.get("session_count") or 0), -(c.get("consistency_score") or 0))
    )
    contradictions.sort(
        key=lambda c: (0 if c.get("severity") == "high" else 1, 0 if c["type"] == "session_drift" else 1, -(c.get("rel_spread_pct") or 0))
    )
    return {"commonalities": commonalities, "contradictions": contradictions}


def _select_video_presets(
    swings: List[Dict[str, Any]],
    contradictions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Select default 2x2 showcase candidates (best/worst, chronological 4, and cross-session conflict 4)."""
    with_media = [sw for sw in swings if sw.get("clip_url") or sw.get("impact_freeze_url")]
    pool = with_media if with_media else swings
    if not pool:
        return {"best": None, "worst": None, "timeline_4": [], "conflict_4": [], "conflict_metric_label": None}

    def _quality_score(sw: Dict[str, Any]) -> Tuple[int, float]:
        has_clip = 1 if sw.get("clip_url") else 0
        spd = _safe_float(sw.get("racket_max_speed_px_s")) or _safe_float(sw.get("racket_speed_px_s"))
        if spd is not None:
            return (has_clip, spd)
        ext = _safe_float(sw.get("arm_extension")) or 0.0
        turn = _safe_float(sw.get("shoulder_turn_change")) or 0.0
        return (has_clip, ext + turn)

    # Prefer items with clip_url and valid speed/posture metrics
    ranked = sorted(pool, key=_quality_score)
    best = ranked[-1]
    worst = ranked[0] if len(ranked) >= 2 else None
    if worst is not None and worst is best:
        worst = None

    n = len(pool)
    if n <= 4:
        indices = list(range(n))
    else:
        indices = [0, round((n - 1) / 3), round(2 * (n - 1) / 3), n - 1]
    timeline_4 = [
        {"session_id": pool[i]["session_id"], "event_id": pool[i]["event_id"]}
        for i in indices
    ]

    # Build cross-session conflict_4 (4 distinct sessions sorted by top conflicting metric)
    top_drift = next((c for c in (contradictions or []) if c.get("type") == "session_drift" and c.get("key")), None)
    sort_key = top_drift["key"] if top_drift else "shoulder_turn_change"
    sort_label = top_drift.get("label") if top_drift else "转肩角度变化"

    by_sess: Dict[str, List[Dict[str, Any]]] = {}
    for sw in pool:
        by_sess.setdefault(sw["session_id"], []).append(sw)

    sess_reps = []
    for sid, group in by_sess.items():
        vals = [float(sw[sort_key]) for sw in group if sw.get(sort_key) is not None]
        if vals:
            med_v = float(statistics.median(vals))
            rep = min(group, key=lambda sw: abs((_safe_float(sw.get(sort_key)) if sw.get(sort_key) is not None else med_v) - med_v))
            sess_reps.append((med_v, rep))
    if len(sess_reps) < 2:
        # Fallback to racket_max_speed_px_s or quality score
        sort_key = "racket_max_speed_px_s"
        sort_label = "拍框峰值速度"
        sess_reps = []
        for sid, group in by_sess.items():
            vals = [_quality_score(sw)[1] for sw in group]
            med_v = float(statistics.median(vals))
            rep = min(group, key=lambda sw: abs(_quality_score(sw)[1] - med_v))
            sess_reps.append((med_v, rep))

    sess_reps.sort(key=lambda item: item[0])
    if len(sess_reps) >= 4:
        m = len(sess_reps)
        c_indices = [0, round((m - 1) / 3), round(2 * (m - 1) / 3), m - 1]
        conflict_4 = [
            {"session_id": sess_reps[i][1]["session_id"], "event_id": sess_reps[i][1]["event_id"]}
            for i in c_indices
        ]
    else:
        picked = [item[1] for item in sess_reps]
        seen_pairs = {(sw["session_id"], sw["event_id"]) for sw in picked}
        for sw in ranked:
            if len(picked) >= 4:
                break
            pair = (sw["session_id"], sw["event_id"])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                picked.append(sw)
        conflict_4 = [{"session_id": sw["session_id"], "event_id": sw["event_id"]} for sw in picked[:4]]

    return {
        "best": {"session_id": best["session_id"], "event_id": best["event_id"]} if best else None,
        "worst": {"session_id": worst["session_id"], "event_id": worst["event_id"]} if worst else None,
        "timeline_4": timeline_4,
        "conflict_4": conflict_4,
        "conflict_metric_label": sort_label,
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if num == num else None


def _safe_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compute_iqr(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    lower = ordered[:mid]
    upper = ordered[mid:] if n % 2 == 0 else ordered[mid + 1 :]
    q1 = statistics.median(lower) if lower else ordered[0]
    q3 = statistics.median(upper) if upper else ordered[-1]
    return round(float(q3 - q1), 2)


def _compute_quantiles(values: List[float]) -> Tuple[float, float, float, float, Optional[float]]:
    ordered = sorted(values)
    n = len(ordered)
    med = float(statistics.median(ordered))
    if n < 2:
        return round(med, 2), round(med, 2), round(med, 2), 0.0, None
    mid = n // 2
    lower = ordered[:mid]
    upper = ordered[mid:] if n % 2 == 0 else ordered[mid + 1 :]
    q1 = float(statistics.median(lower)) if lower else ordered[0]
    q3 = float(statistics.median(upper)) if upper else ordered[-1]
    iqr = max(0.0, q3 - q1)
    rel_disp = iqr / max(abs(med), 1.0)
    consistency = round(max(0.0, min(100.0, 100.0 * (1.0 - min(1.0, rel_disp / 0.65)))), 1)
    return round(q1, 2), round(med, 2), round(q3, 2), round(iqr, 2), consistency


def _compute_fatigue_split(swings: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(swings) < 4:
        return {"available": False, "reason": "单次有效挥拍不足4拍"}
    mid = len(swings) // 2
    first_half = swings[:mid]
    second_half = swings[mid:]

    def _med(items: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [float(x[key]) for x in items if x.get(key) is not None]
        return round(float(statistics.median(vals)), 2) if vals else None

    spd1, spd2 = _med(first_half, "racket_max_speed_px_s"), _med(second_half, "racket_max_speed_px_s")
    turn1, turn2 = _med(first_half, "shoulder_turn_change"), _med(second_half, "shoulder_turn_change")
    knee1, knee2 = _med(first_half, "preparation_knee_flexion"), _med(second_half, "preparation_knee_flexion")
    ext1, ext2 = _med(first_half, "arm_extension"), _med(second_half, "arm_extension")

    spd_delta_pct = round(((spd2 - spd1) / max(abs(spd1), 1.0)) * 100.0, 1) if (spd1 and spd2) else None
    turn_delta = round(turn2 - turn1, 1) if (turn1 is not None and turn2 is not None) else None
    knee_delta = round(knee2 - knee1, 1) if (knee1 is not None and knee2 is not None) else None
    ext_delta = round(ext2 - ext1, 1) if (ext1 is not None and ext2 is not None) else None

    tags = []
    if knee_delta is not None and knee_delta > 6.0:
        tags.append(f"后半程屈膝变直 (+{knee_delta}°，下肢疲劳)")
    if turn_delta is not None and turn_delta < -6.0:
        tags.append(f"后半程转肩缩水 ({turn_delta}°)")
    if spd_delta_pct is not None and spd_delta_pct < -8.0:
        tags.append(f"后半程拍速下滑 ({spd_delta_pct}%)")
    if not tags:
        if spd_delta_pct is not None and spd_delta_pct > 5.0:
            tags.append(f"后半程渐入佳境 (拍速 +{spd_delta_pct}%)")
        else:
            tags.append("全程动作保持度良好")

    return {
        "available": True,
        "first_half_count": len(first_half),
        "second_half_count": len(second_half),
        "speed_delta_pct": spd_delta_pct,
        "turn_delta_deg": turn_delta,
        "knee_delta_deg": knee_delta,
        "extension_delta_deg": ext_delta,
        "summary": "；".join(tags),
    }


def _compute_personal_attribution(all_swings: List[Dict[str, Any]]) -> Dict[str, Any]:
    with_speed = [
        sw for sw in all_swings
        if (sw.get("racket_max_speed_px_s") or sw.get("racket_speed_px_s")) is not None
    ]
    if len(with_speed) < 4:
        return {"available": False, "sample_count": len(with_speed)}

    ranked = sorted(
        with_speed,
        key=lambda sw: float(sw.get("racket_max_speed_px_s") or sw.get("racket_speed_px_s") or 0.0),
    )
    k = max(1, len(ranked) // 4)
    bottom_group = ranked[:k]
    top_group = ranked[-k:]

    top_spd = round(
        float(statistics.median([float(s.get("racket_max_speed_px_s") or s.get("racket_speed_px_s")) for s in top_group])),
        1,
    )
    bot_spd = round(
        float(statistics.median([float(s.get("racket_max_speed_px_s") or s.get("racket_speed_px_s")) for s in bottom_group])),
        1,
    )

    factor_specs = [
        ("shoulder_turn_change", "转肩角度变化", "°", True),
        ("hip_shoulder_separation", "肩髋分离角", "°", True),
        ("arm_extension", "持拍侧肘伸展", "°", True),
        ("preparation_knee_flexion", "准备期屈膝角", "°", False),
        ("contact_lateral_distance", "触球离体距离", "bw", True),
        ("weight_transfer", "躯干重心横移", "bw", True),
    ]
    factors = []
    for key, label, unit, higher_better in factor_specs:
        all_vals = [float(s[key]) for s in ranked if s.get(key) is not None]
        top_vals = [float(s[key]) for s in top_group if s.get(key) is not None]
        bot_vals = [float(s[key]) for s in bottom_group if s.get(key) is not None]
        if not top_vals or not bot_vals or len(all_vals) < 4:
            continue
        t_med = float(statistics.median(top_vals))
        b_med = float(statistics.median(bot_vals))
        diff = round(t_med - b_med, 2)
        overall_iqr = max(_compute_iqr(all_vals), abs(float(statistics.median(all_vals))) * 0.08, 1e-3)
        effect = round(abs(diff) / overall_iqr, 2)
        factors.append(
            {
                "key": key,
                "label": label,
                "unit": unit,
                "top_median": round(t_med, 2),
                "bottom_median": round(b_med, 2),
                "diff": diff,
                "effect_ratio": effect,
                "higher_better": higher_better,
            }
        )

    factors.sort(key=lambda f: f["effect_ratio"], reverse=True)
    top_drivers = factors[:2]
    if top_drivers:
        parts = [
            f"【{d['label']}】（好球 {d['top_median']}{d['unit']} vs 弱球 {d['bottom_median']}{d['unit']}，差异 {('+' if d['diff'] > 0 else '')}{d['diff']}{d['unit']}）"
            for d in top_drivers
        ]
        insight = f"基于该球员 {len(ranked)} 拍个人样本归因：打出个人前25%高质球（中位 {top_spd} px/s）相比后25%弱球（{bot_spd} px/s），差异最显著的发力开关是 " + " 与 ".join(parts) + "。"
    else:
        insight = "样本指标维度尚不足以提取显著归因因子。"

    return {
        "available": True,
        "sample_count": len(ranked),
        "quartile_size": k,
        "top_speed_median": top_spd,
        "bottom_speed_median": bot_spd,
        "factors": factors,
        "insight": insight,
    }


class SessionDatabase:
    """Thread-safe SQLite store for session traceability and player comparisons."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS players (
                    player_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    dominant_hand TEXT NOT NULL DEFAULT 'right',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    player_id TEXT NOT NULL DEFAULT 'default',
                    display_time TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    source_name TEXT,
                    source_path TEXT,
                    source_sha256 TEXT,
                    stream_id TEXT,
                    processing_mode TEXT,
                    event_count INTEGER NOT NULL DEFAULT 0,
                    forehand_count INTEGER NOT NULL DEFAULT 0,
                    backhand_count INTEGER NOT NULL DEFAULT 0,
                    shadow_count INTEGER NOT NULL DEFAULT 0,
                    json_mtime_ns INTEGER NOT NULL DEFAULT 0,
                    json_size INTEGER NOT NULL DEFAULT 0,
                    session_dir TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (player_id) REFERENCES players(player_id)
                );

                CREATE TABLE IF NOT EXISTS swing_events (
                    session_id TEXT NOT NULL,
                    event_id INTEGER NOT NULL,
                    player_id TEXT NOT NULL DEFAULT 'default',
                    stroke_type TEXT,
                    is_shadow_swing INTEGER NOT NULL DEFAULT 0,
                    swing_score REAL,
                    swing_grade TEXT,
                    start_frame INTEGER,
                    end_frame INTEGER,
                    contact_frame INTEGER,
                    racket_speed_px_s REAL,
                    racket_max_speed_px_s REAL,
                    brush_angle_deg REAL,
                    drop_depth_ratio REAL,
                    leg_drive_px REAL,
                    sequence_quality TEXT,
                    hip_to_shoulder_ms REAL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, event_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS swing_metrics (
                    session_id TEXT NOT NULL,
                    event_id INTEGER NOT NULL,
                    player_id TEXT NOT NULL DEFAULT 'default',
                    stroke_type TEXT,
                    metric_key TEXT NOT NULL,
                    metric_label TEXT,
                    value_num REAL,
                    value_text TEXT,
                    unit TEXT,
                    confidence REAL,
                    coach_eligible INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (session_id, event_id, metric_key),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS coach_reviews (
                    review_id TEXT PRIMARY KEY,
                    player_id TEXT NOT NULL,
                    stroke_side TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    review_text TEXT NOT NULL,
                    session_ids_json TEXT,
                    video_paths_json TEXT,
                    evidence_json TEXT,
                    days_scope TEXT,
                    dedup INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_player_ts
                    ON sessions(player_id, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_events_player_stroke
                    ON swing_events(player_id, stroke_type, is_shadow_swing);
                CREATE INDEX IF NOT EXISTS idx_metrics_player_key
                    ON swing_metrics(player_id, metric_key, stroke_type);
                CREATE INDEX IF NOT EXISTS idx_reviews_player_side_ts
                    ON coach_reviews(player_id, stroke_side, timestamp DESC);
                """
            )
            existing_cols = {
                str(r["name"])
                for r in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "source_path" not in existing_cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN source_path TEXT")
            if "source_sha256" not in existing_cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN source_sha256 TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_sha256 ON sessions(source_sha256)"
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO players (player_id, display_name, dominant_hand, notes, created_at)
                VALUES (?, ?, 'right', '', ?)
                """,
                (DEFAULT_PLAYER_ID, DEFAULT_PLAYER_NAME, _utc_now_iso()),
            )
            # Clean up any historical bbox-jump speed spikes (> 5000 px/s)
            conn.execute(
                """
                UPDATE swing_events
                SET racket_max_speed_px_s = CASE
                    WHEN racket_speed_px_s IS NOT NULL AND racket_speed_px_s <= ? THEN racket_speed_px_s
                    ELSE NULL
                END
                WHERE racket_max_speed_px_s > ?
                """,
                (MAX_PLAUSIBLE_RACKET_SPEED_PX_S, MAX_PLAUSIBLE_RACKET_SPEED_PX_S),
            )
            conn.execute(
                """
                UPDATE swing_events
                SET racket_speed_px_s = NULL
                WHERE racket_speed_px_s > ?
                """,
                (MAX_PLAUSIBLE_RACKET_SPEED_PX_S,),
            )
            conn.execute(
                """
                DELETE FROM swing_metrics
                WHERE metric_key IN ('racket_max_speed', 'racket_head_speed')
                  AND value_num > ?
                """,
                (MAX_PLAUSIBLE_RACKET_SPEED_PX_S,),
            )

    def list_players(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.player_id, p.display_name, p.dominant_hand, p.notes, p.created_at,
                       COUNT(s.session_id) AS session_count
                FROM players p
                LEFT JOIN sessions s ON s.player_id = p.player_id
                GROUP BY p.player_id
                ORDER BY CASE WHEN p.player_id = 'default' THEN 0 ELSE 1 END, p.display_name ASC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def create_or_update_player(
        self,
        display_name: str,
        player_id: Optional[str] = None,
        dominant_hand: str = "right",
        notes: str = "",
    ) -> Dict[str, Any]:
        name = str(display_name or "").strip()
        if not name:
            raise ValueError("球员姓名不能为空")
        hand = str(dominant_hand or "right").strip().lower()
        if hand not in ("right", "left"):
            hand = "right"
        if player_id:
            pid = _PLAYER_ID_PATTERN.sub("_", str(player_id).strip()).strip("_")[:48]
        else:
            slug = _PLAYER_ID_PATTERN.sub("_", name).strip("_")[:24] or "player"
            pid = f"{slug}_{int(datetime.now(timezone.utc).timestamp())}"
        if not pid:
            raise ValueError("无效的球员 ID")

        now = _utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO players (player_id, display_name, dominant_hand, notes, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(player_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    dominant_hand = excluded.dominant_hand,
                    notes = excluded.notes
                """,
                (pid, name, hand, str(notes or "").strip(), now),
            )
            row = conn.execute(
                "SELECT player_id, display_name, dominant_hand, notes, created_at FROM players WHERE player_id = ?",
                (pid,),
            ).fetchone()
            return dict(row)

    def assign_session_player(self, session_id: str, player_id: str) -> Dict[str, Any]:
        sid = str(session_id or "").strip()
        pid = str(player_id or DEFAULT_PLAYER_ID).strip() or DEFAULT_PLAYER_ID
        if not sid:
            raise ValueError("session_id 不能为空")

        with self._lock, self._connect() as conn:
            player = conn.execute(
                "SELECT player_id, display_name FROM players WHERE player_id = ?",
                (pid,),
            ).fetchone()
            if player is None:
                raise ValueError(f"未找到球员: {pid}")

            sess = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (sid,),
            ).fetchone()
            if sess is None:
                # Create lightweight placeholder row so pre-binding works before first swing arrives.
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, player_id, display_time, timestamp, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (sid, pid, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.now().timestamp(), _utc_now_iso()),
                )
            else:
                conn.execute("UPDATE sessions SET player_id = ?, updated_at = ? WHERE session_id = ?", (pid, _utc_now_iso(), sid))
                conn.execute("UPDATE swing_events SET player_id = ? WHERE session_id = ?", (pid, sid))
                conn.execute("UPDATE swing_metrics SET player_id = ? WHERE session_id = ?", (pid, sid))

            return {
                "session_id": sid,
                "player_id": pid,
                "player_name": player["display_name"],
            }

    def assign_sessions_player_batch(
        self,
        session_ids: List[str],
        player_id: str,
    ) -> Dict[str, Any]:
        pid = str(player_id or DEFAULT_PLAYER_ID).strip() or DEFAULT_PLAYER_ID
        clean_ids: List[str] = []
        seen = set()
        for raw_sid in session_ids or []:
            sid = str(raw_sid or "").strip()
            if sid and sid not in seen:
                seen.add(sid)
                clean_ids.append(sid)
        if not clean_ids:
            raise ValueError("请至少选择一个会话")

        now = _utc_now_iso()
        now_dt = datetime.now()
        with self._lock, self._connect() as conn:
            player = conn.execute(
                "SELECT player_id, display_name FROM players WHERE player_id = ?",
                (pid,),
            ).fetchone()
            if player is None:
                raise ValueError(f"未找到球员: {pid}")

            for sid in clean_ids:
                sess = conn.execute(
                    "SELECT session_id FROM sessions WHERE session_id = ?",
                    (sid,),
                ).fetchone()
                if sess is None:
                    conn.execute(
                        """
                        INSERT INTO sessions (
                            session_id, player_id, display_time, timestamp, updated_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (sid, pid, now_dt.strftime("%Y-%m-%d %H:%M:%S"), now_dt.timestamp(), now),
                    )
                else:
                    conn.execute(
                        "UPDATE sessions SET player_id = ?, updated_at = ? WHERE session_id = ?",
                        (pid, now, sid),
                    )
                    conn.execute("UPDATE swing_events SET player_id = ? WHERE session_id = ?", (pid, sid))
                    conn.execute("UPDATE swing_metrics SET player_id = ? WHERE session_id = ?", (pid, sid))

            return {
                "updated_count": len(clean_ids),
                "session_ids": clean_ids,
                "player_id": pid,
                "player_name": player["display_name"],
            }

    def get_session_signatures(self) -> Dict[str, Tuple[int, int]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT session_id, json_mtime_ns, json_size FROM sessions").fetchall()
            return {r["session_id"]: (int(r["json_mtime_ns"]), int(r["json_size"])) for r in rows}

    def get_session_source_status(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Return stored source metadata per session so unchanged sessions can be backfilled."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, source_name, source_path, source_sha256 FROM sessions"
            ).fetchall()
            return {
                str(r["session_id"]): {
                    "source_name": r["source_name"],
                    "source_path": r["source_path"],
                    "source_sha256": r["source_sha256"],
                }
                for r in rows
            }

    def update_session_source_meta(
        self,
        session_id: str,
        *,
        source_name: Optional[str] = None,
        source_path: Optional[str] = None,
        source_sha256: Optional[str] = None,
    ) -> None:
        """Backfill source metadata without rewriting swing events."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET source_name = COALESCE(?, source_name),
                    source_path = COALESCE(?, source_path),
                    source_sha256 = COALESCE(?, source_sha256),
                    updated_at = ?
                WHERE session_id = ?
                """,
                (source_name, source_path, source_sha256, _utc_now_iso(), session_id),
            )

    def list_duplicate_sessions(self, player_id: Optional[str] = None) -> Dict[str, Any]:
        """Group sessions representing the same source video and identify canonical vs duplicate runs."""
        pid = str(player_id).strip() if player_id else None
        with self._lock, self._connect() as conn:
            sql = """
                SELECT s.session_id, s.player_id, COALESCE(p.display_name, s.player_id) AS player_name,
                       s.display_time, s.timestamp, s.source_name, s.source_path, s.source_sha256,
                       s.stream_id, s.processing_mode, s.event_count, s.forehand_count,
                       s.backhand_count, s.shadow_count, s.session_dir,
                       EXISTS(
                           SELECT 1 FROM swing_metrics m
                           WHERE m.session_id = s.session_id
                             AND m.metric_key IN ('takeback_depth', 'scapular_retraction')
                             AND m.value_num IS NOT NULL
                       ) AS is_dual_view
                FROM sessions s
                LEFT JOIN players p ON p.player_id = s.player_id
            """
            params: Tuple[Any, ...] = ()
            if pid:
                sql += " WHERE s.player_id = ?"
                params = (pid,)
            sql += " ORDER BY s.timestamp DESC"
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            for r in rows:
                r["is_dual_view"] = bool(r.get("is_dual_view"))
                r["view_mode"] = "dual_view" if r["is_dual_view"] else "single_view"

        kept_rows, summary = _group_duplicate_sessions(rows)
        return {
            "player_id": pid,
            "raw_session_count": summary["raw_session_count"],
            "unique_session_count": summary["unique_session_count"],
            "duplicate_session_count": summary["duplicate_session_count"],
            "duplicate_group_count": summary["duplicate_group_count"],
            "groups": summary["groups"],
            "kept_session_ids": [str(r["session_id"]) for r in kept_rows],
            "recommended_cleanup_ids": summary["hidden_session_ids"],
        }

    def delete_sessions_from_db(self, session_ids: List[str]) -> int:
        """Remove deleted/archived sessions and their dependent rows from SQLite."""
        cleaned = [str(s).strip() for s in session_ids if str(s or "").strip()]
        if not cleaned:
            return 0
        placeholders = ",".join("?" for _ in cleaned)
        with self._lock, self._connect() as conn:
            conn.execute(f"DELETE FROM swing_metrics WHERE session_id IN ({placeholders})", cleaned)
            conn.execute(f"DELETE FROM swing_events WHERE session_id IN ({placeholders})", cleaned)
            cur = conn.execute(f"DELETE FROM sessions WHERE session_id IN ({placeholders})", cleaned)
            return int(cur.rowcount or 0)

    def get_swing_event_record(self, session_id: str, event_id: int) -> Optional[Dict[str, Any]]:
        """Fetch one swing event together with session directory and player dominant hand."""
        sid = str(session_id or "").strip()
        if not sid:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT e.session_id, e.event_id, e.stroke_type, e.start_frame, e.end_frame,
                       e.contact_frame, e.racket_speed_px_s, e.racket_max_speed_px_s,
                       e.payload_json, s.display_time, s.session_dir,
                       COALESCE(p.dominant_hand, 'right') AS dominant_hand
                FROM swing_events e
                JOIN sessions s ON s.session_id = e.session_id
                LEFT JOIN players p ON p.player_id = s.player_id
                WHERE e.session_id = ? AND e.event_id = ?
                """,
                (sid, int(event_id)),
            ).fetchone()
            return dict(row) if row is not None else None

    def get_session_player_map(self) -> Dict[str, Dict[str, str]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id, s.player_id, p.display_name AS player_name,
                       EXISTS(
                           SELECT 1 FROM swing_metrics m
                           WHERE m.session_id = s.session_id
                             AND m.metric_key IN ('takeback_depth', 'scapular_retraction')
                       ) AS is_dual_view
                FROM sessions s
                LEFT JOIN players p ON p.player_id = s.player_id
                """
            ).fetchall()
            return {
                r["session_id"]: {
                    "player_id": r["player_id"] or DEFAULT_PLAYER_ID,
                    "player_name": r["player_name"] or DEFAULT_PLAYER_NAME,
                    "view_mode": "dual_view" if r["is_dual_view"] else "single_view",
                }
                for r in rows
            }

    def upsert_session(
        self,
        *,
        session_id: str,
        display_time: str,
        timestamp: float,
        events: List[Dict[str, Any]],
        source_name: Optional[str] = None,
        source_path: Optional[str] = None,
        source_sha256: Optional[str] = None,
        stream_id: Optional[str] = None,
        processing_mode: Optional[str] = None,
        json_mtime_ns: int = 0,
        json_size: int = 0,
        session_dir: Optional[str] = None,
        player_id: Optional[str] = None,
    ) -> None:
        if not session_id:
            return
        normalized_events = [normalize_swing_event_for_delivery(ev) for ev in (events or [])]
        forehand_count = sum(
            1 for ev in normalized_events
            if not ev.get("is_shadow_swing") and str(ev.get("stroke_type") or "").lower() == "forehand"
        )
        backhand_count = sum(
            1 for ev in normalized_events
            if not ev.get("is_shadow_swing") and "backhand" in str(ev.get("stroke_type") or "").lower()
        )
        shadow_count = sum(1 for ev in normalized_events if ev.get("is_shadow_swing"))
        now = _utc_now_iso()

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT player_id, json_mtime_ns, json_size FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if (
                existing is not None
                and json_mtime_ns > 0
                and int(existing["json_mtime_ns"]) == int(json_mtime_ns)
                and int(existing["json_size"]) == int(json_size)
                and player_id is None
            ):
                return

            resolved_pid = player_id or (existing["player_id"] if existing is not None else DEFAULT_PLAYER_ID)
            # Ensure referenced player exists
            conn.execute(
                "INSERT OR IGNORE INTO players (player_id, display_name, dominant_hand, notes, created_at) VALUES (?, ?, 'right', '', ?)",
                (resolved_pid, resolved_pid if resolved_pid != DEFAULT_PLAYER_ID else DEFAULT_PLAYER_NAME, now),
            )

            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, player_id, display_time, timestamp, source_name,
                    source_path, source_sha256,
                    stream_id, processing_mode, event_count, forehand_count,
                    backhand_count, shadow_count, json_mtime_ns, json_size,
                    session_dir, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    player_id = excluded.player_id,
                    display_time = excluded.display_time,
                    timestamp = excluded.timestamp,
                    source_name = COALESCE(excluded.source_name, sessions.source_name),
                    source_path = COALESCE(excluded.source_path, sessions.source_path),
                    source_sha256 = COALESCE(excluded.source_sha256, sessions.source_sha256),
                    stream_id = COALESCE(excluded.stream_id, sessions.stream_id),
                    processing_mode = COALESCE(excluded.processing_mode, sessions.processing_mode),
                    event_count = excluded.event_count,
                    forehand_count = excluded.forehand_count,
                    backhand_count = excluded.backhand_count,
                    shadow_count = excluded.shadow_count,
                    json_mtime_ns = excluded.json_mtime_ns,
                    json_size = excluded.json_size,
                    session_dir = COALESCE(excluded.session_dir, sessions.session_dir),
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    resolved_pid,
                    display_time,
                    float(timestamp),
                    source_name,
                    source_path,
                    source_sha256,
                    stream_id,
                    processing_mode,
                    len(normalized_events),
                    forehand_count,
                    backhand_count,
                    shadow_count,
                    int(json_mtime_ns),
                    int(json_size),
                    session_dir,
                    now,
                ),
            )

            conn.execute("DELETE FROM swing_events WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM swing_metrics WHERE session_id = ?", (session_id,))

            event_rows = []
            metric_rows = []
            for idx, ev in enumerate(normalized_events):
                eid = _safe_int(ev.get("event_id"))
                if eid is None:
                    eid = idx + 1
                stroke = str(ev.get("stroke_type") or "Swing")
                ext = ev.get("extended_biomechanics") or {}
                speed = ext.get("racket_head_speed") or {}
                brush = ext.get("brush_angle") or {}
                leg = ext.get("leg_drive") or {}
                seq = ev.get("kinematic_sequence") or {}
                clean_contact_spd, clean_max_spd = _sanitize_racket_speeds(
                    speed.get("contact_px_s"),
                    speed.get("max_px_s"),
                )

                event_rows.append(
                    (
                        session_id,
                        eid,
                        resolved_pid,
                        stroke,
                        1 if ev.get("is_shadow_swing") else 0,
                        _safe_float(ev.get("swing_score")),
                        str(ev.get("swing_grade") or "REVIEW_REQUIRED"),
                        _safe_int(ev.get("start_frame")),
                        _safe_int(ev.get("end_frame")),
                        _safe_int(ev.get("contact_frame")),
                        clean_contact_spd,
                        clean_max_spd,
                        _safe_float(brush.get("low_to_high_angle_deg")),
                        _safe_float(brush.get("drop_depth_ratio")),
                        _safe_float(leg.get("drive_px")),
                        str(seq.get("sequence_quality") or ""),
                        _safe_float(seq.get("latency_hip_to_shoulder_ms")),
                        json.dumps(ev, ensure_ascii=False, separators=(",", ":")),
                    )
                )

                for m in ev.get("analysis_metrics") or []:
                    m_key = str(m.get("key") or "").strip()
                    if not m_key:
                        continue
                    val = m.get("value")
                    val_num = _safe_float(val)
                    if m_key in ("racket_max_speed", "racket_head_speed") and val_num is not None:
                        if val_num <= 0 or val_num > MAX_PLAUSIBLE_RACKET_SPEED_PX_S:
                            val_num = clean_contact_spd if m_key == "racket_max_speed" else None
                            if val_num is None:
                                continue
                    val_text = str(val) if val_num is None and val is not None else None
                    metric_rows.append(
                        (
                            session_id,
                            eid,
                            resolved_pid,
                            stroke,
                            m_key,
                            str(m.get("label") or m_key),
                            val_num,
                            val_text,
                            str(m.get("unit") or ""),
                            _safe_float(m.get("confidence")),
                            1 if m.get("coach_eligible") else 0,
                        )
                    )

            if event_rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO swing_events (
                        session_id, event_id, player_id, stroke_type, is_shadow_swing,
                        swing_score, swing_grade, start_frame, end_frame, contact_frame,
                        racket_speed_px_s, racket_max_speed_px_s, brush_angle_deg,
                        drop_depth_ratio, leg_drive_px, sequence_quality,
                        hip_to_shoulder_ms, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    event_rows,
                )
            if metric_rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO swing_metrics (
                        session_id, event_id, player_id, stroke_type, metric_key,
                        metric_label, value_num, value_text, unit, confidence, coach_eligible
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    metric_rows,
                )

    def compare_player_sessions(
        self,
        player_id: str = DEFAULT_PLAYER_ID,
        stroke_type: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        pid = str(player_id or DEFAULT_PLAYER_ID).strip() or DEFAULT_PLAYER_ID
        stroke_filter = str(stroke_type or "").strip()
        if stroke_filter.lower() in ("", "all"):
            stroke_filter = ""

        with self._lock, self._connect() as conn:
            player_row = conn.execute(
                "SELECT player_id, display_name, dominant_hand FROM players WHERE player_id = ?",
                (pid,),
            ).fetchone()
            if player_row is None:
                raise ValueError(f"未找到球员: {pid}")

            session_rows = conn.execute(
                """
                SELECT session_id, display_time, timestamp, source_name,
                       event_count, forehand_count, backhand_count, shadow_count, session_dir
                FROM sessions
                WHERE player_id = ? AND event_count > 0
                ORDER BY timestamp ASC
                """,
                (pid,),
            ).fetchall()

            if len(session_rows) > limit:
                session_rows = session_rows[-limit:]

            workspace_root = self.db_path.resolve().parent.parent
            series = []
            for s in session_rows:
                sid = s["session_id"]
                s_dir = Path(s["session_dir"]) if s["session_dir"] else (
                    workspace_root / "data" / "analysis_results" / "control_panel" / sid
                )
                stroke_clause, stroke_params = _stroke_sql_clause(stroke_filter)
                params: List[Any] = [sid, *stroke_params]

                ev_rows = conn.execute(
                    f"""
                    SELECT event_id, stroke_type, is_shadow_swing, swing_score,
                           start_frame, end_frame, contact_frame,
                           racket_speed_px_s, racket_max_speed_px_s,
                           brush_angle_deg, leg_drive_px, sequence_quality, payload_json
                    FROM swing_events
                    WHERE session_id = ? AND is_shadow_swing = 0{stroke_clause}
                    ORDER BY event_id ASC
                    """,
                    params,
                ).fetchall()

                speeds = [r["racket_speed_px_s"] for r in ev_rows if r["racket_speed_px_s"] is not None]
                max_speeds = [r["racket_max_speed_px_s"] for r in ev_rows if r["racket_max_speed_px_s"] is not None]
                brushes = [r["brush_angle_deg"] for r in ev_rows if r["brush_angle_deg"] is not None]
                legs = [r["leg_drive_px"] for r in ev_rows if r["leg_drive_px"] is not None]
                scores = [r["swing_score"] for r in ev_rows if r["swing_score"] is not None]
                seq_total = sum(1 for r in ev_rows if r["sequence_quality"])
                seq_good = sum(
                    1 for r in ev_rows if r["sequence_quality"] in ("OPTIMAL", "ACCEPTABLE")
                )

                # Extract per-swing media URLs, key metrics, and coach warnings
                swings_summary = []
                warning_counter: Dict[str, int] = {}
                for r in ev_rows:
                    try:
                        payload = json.loads(r["payload_json"]) if r["payload_json"] else {}
                    except Exception:
                        payload = {}
                    clip_url, freeze_url = _extract_payload_media_urls(payload, s_dir, workspace_root)

                    ev_metrics = {
                        m["key"]: m.get("value")
                        for m in (payload.get("analysis_metrics") or [])
                        if isinstance(m, dict) and m.get("key")
                    }
                    advice_msgs, effective_warnings = _extract_swing_warnings(payload, ev_metrics)
                    for msg in effective_warnings:
                        warning_counter[msg] = warning_counter.get(msg, 0) + 1

                    clean_c, clean_m = _sanitize_racket_speeds(
                        r["racket_speed_px_s"],
                        r["racket_max_speed_px_s"] or ev_metrics.get("racket_max_speed"),
                    )
                    swings_summary.append(
                        {
                            "event_id": r["event_id"],
                            "stroke_type": r["stroke_type"],
                            "start_frame": r["start_frame"],
                            "end_frame": r["end_frame"],
                            "contact_frame": r["contact_frame"],
                            "clip_url": clip_url,
                            "impact_freeze_url": freeze_url,
                            "racket_speed_px_s": clean_c,
                            "racket_max_speed_px_s": clean_m,
                            "shoulder_turn_change": ev_metrics.get("shoulder_turn_change"),
                            "hip_shoulder_separation": ev_metrics.get("hip_shoulder_separation"),
                            "arm_extension": ev_metrics.get("arm_extension"),
                            "preparation_knee_flexion": ev_metrics.get("preparation_knee_flexion"),
                            "weight_transfer": ev_metrics.get("weight_transfer"),
                            "contact_lateral_distance": ev_metrics.get("contact_lateral_distance"),
                            "coach_messages": advice_msgs,
                        }
                    )

                # Choose representative swing (prefer media + closest to median peak speed)
                rep_swing = None
                if swings_summary:
                    with_media = [sw for sw in swings_summary if sw["clip_url"] or sw["impact_freeze_url"]]
                    candidates = with_media if with_media else swings_summary
                    target_spd = statistics.median(max_speeds) if max_speeds else 0.0
                    rep_swing = min(
                        candidates,
                        key=lambda sw: abs((sw.get("racket_max_speed_px_s") or target_spd) - target_spd),
                    )

                total_valid = max(1, len(ev_rows))
                coach_warnings = [
                    {
                        "message": msg,
                        "count": cnt,
                        "rate_pct": round((cnt / total_valid) * 100.0, 1),
                    }
                    for msg, cnt in sorted(warning_counter.items(), key=lambda kv: kv[1], reverse=True)[:5]
                ]

                fatigue_split = _compute_fatigue_split(swings_summary)

                # Additional metric quantiles + IQR + consistency score from swing_metrics
                m_rows = conn.execute(
                    f"""
                    SELECT metric_key, metric_label, unit, value_num
                    FROM swing_metrics
                    WHERE session_id = ? AND value_num IS NOT NULL{stroke_clause}
                    """,
                    params,
                ).fetchall()
                metric_buckets: Dict[Tuple[str, str, str], List[float]] = {}
                for mr in m_rows:
                    bucket_key = (mr["metric_key"], mr["metric_label"] or mr["metric_key"], mr["unit"] or "")
                    metric_buckets.setdefault(bucket_key, []).append(float(mr["value_num"]))

                metric_medians = {}
                consistency_scores = []
                for (k, lbl, unit), vals in metric_buckets.items():
                    p25, med, p75, iqr, c_score = _compute_quantiles(vals)
                    if c_score is not None:
                        consistency_scores.append(c_score)
                    metric_medians[k] = {
                        "label": lbl,
                        "unit": unit,
                        "p25": p25,
                        "median": med,
                        "p75": p75,
                        "iqr": iqr,
                        "consistency_score": c_score,
                        "min": round(min(vals), 2),
                        "max": round(max(vals), 2),
                        "count": len(vals),
                    }

                consistency_index = (
                    round(float(statistics.mean(consistency_scores)), 1)
                    if consistency_scores
                    else None
                )

                is_dual_view = bool(
                    "takeback_depth" in metric_medians or "scapular_retraction" in metric_medians
                )
                series.append(
                    {
                        "session_id": sid,
                        "display_time": s["display_time"],
                        "timestamp": s["timestamp"],
                        "source_name": s["source_name"],
                        "view_mode": "dual_view" if is_dual_view else "single_view",
                        "matched_swings": len(ev_rows),
                        "forehand_count": s["forehand_count"],
                        "backhand_count": s["backhand_count"],
                        "shadow_count": s["shadow_count"],
                        "median_score": round(statistics.median(scores), 1) if scores else None,
                        "median_racket_speed_px_s": round(statistics.median(speeds), 1) if speeds else (
                            metric_medians.get("racket_max_speed", {}).get("median")
                        ),
                        "median_brush_angle_deg": round(statistics.median(brushes), 1) if brushes else None,
                        "median_leg_drive_px": round(statistics.median(legs), 1) if legs else None,
                        "kinematic_ok_rate": round((seq_good / seq_total) * 100.0, 1) if seq_total > 0 else None,
                        "consistency_index": consistency_index,
                        "metrics": metric_medians,
                        "representative_event": rep_swing,
                        "swings": swings_summary,
                        "coach_warnings": coach_warnings,
                        "fatigue_split": fatigue_split,
                    }
                )

            all_player_swings = [sw for sess in series for sw in (sess.get("swings") or [])]
            personal_attribution = _compute_personal_attribution(all_player_swings)

            # Compute deltas between latest and previous session
            deltas: Dict[str, Optional[float]] = {}
            if len(series) >= 2:
                prev, curr = series[-2], series[-1]
                for field in (
                    "median_racket_speed_px_s",
                    "median_brush_angle_deg",
                    "median_leg_drive_px",
                    "kinematic_ok_rate",
                ):
                    if curr.get(field) is not None and prev.get(field) is not None:
                        deltas[field] = round(curr[field] - prev[field], 1)
                    else:
                        deltas[field] = None

            return {
                "player": dict(player_row),
                "stroke_type": stroke_filter or "all",
                "session_count": len(series),
                "sessions": series,
                "latest_delta": deltas,
                "personal_attribution": personal_attribution,
            }

    def get_player_global_profile(
        self,
        player_id: str = DEFAULT_PLAYER_ID,
        days: Optional[int] = None,
        dedup: bool = True,
    ) -> Dict[str, Any]:
        """Aggregate all non-shadow Forehand & Backhand training swings across all sessions for one player."""
        pid = str(player_id or DEFAULT_PLAYER_ID).strip() or DEFAULT_PLAYER_ID
        dedup_enabled = bool(dedup)
        days_val = _safe_int(days)
        if days_val is not None and days_val <= 0:
            days_val = None
        cutoff_ts = (datetime.now(timezone.utc).timestamp() - days_val * 86400.0) if days_val else None

        with self._lock, self._connect() as conn:
            player_row = conn.execute(
                "SELECT player_id, display_name, dominant_hand FROM players WHERE player_id = ?",
                (pid,),
            ).fetchone()
            if player_row is None:
                raise ValueError(f"未找到球员: {pid}")

            session_rows = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT s.session_id, s.display_time, s.timestamp, s.source_name,
                           s.source_path, s.source_sha256,
                           s.event_count, s.forehand_count, s.backhand_count, s.shadow_count, s.session_dir,
                           EXISTS(
                               SELECT 1 FROM swing_metrics m
                               WHERE m.session_id = s.session_id
                                 AND m.metric_key IN ('takeback_depth', 'scapular_retraction')
                                 AND m.value_num IS NOT NULL
                           ) AS is_dual_view
                    FROM sessions s
                    WHERE s.player_id = ? AND s.event_count > 0
                    ORDER BY s.timestamp ASC
                    """,
                    (pid,),
                ).fetchall()
            ]
            for s_row in session_rows:
                s_row["is_dual_view"] = bool(s_row.get("is_dual_view"))
                s_row["view_mode"] = "dual_view" if s_row["is_dual_view"] else "single_view"

            if cutoff_ts is not None:
                session_rows = [s for s in session_rows if float(s["timestamp"] or 0.0) >= cutoff_ts]

            kept_rows, dedup_summary = _group_duplicate_sessions(session_rows)
            active_session_rows = kept_rows if dedup_enabled else session_rows

            workspace_root = self.db_path.resolve().parent.parent
            side_swings: Dict[str, List[Dict[str, Any]]] = {"forehand": [], "backhand": [], "other": []}
            sessions_catalog: List[Dict[str, Any]] = []

            for s in active_session_rows:
                sid = s["session_id"]
                s_dir = Path(s["session_dir"]) if s["session_dir"] else (
                    workspace_root / "data" / "analysis_results" / "control_panel" / sid
                )
                ev_rows = conn.execute(
                    """
                    SELECT event_id, stroke_type, start_frame, end_frame, contact_frame,
                           racket_speed_px_s, racket_max_speed_px_s, sequence_quality, payload_json
                    FROM swing_events
                    WHERE session_id = ? AND is_shadow_swing = 0
                    ORDER BY event_id ASC
                    """,
                    (sid,),
                ).fetchall()

                m_rows = conn.execute(
                    """
                    SELECT event_id, metric_key, value_num
                    FROM swing_metrics
                    WHERE session_id = ? AND value_num IS NOT NULL
                    """,
                    (sid,),
                ).fetchall()
                ev_metric_map: Dict[int, Dict[str, float]] = {}
                session_has_dual = False
                for mr in m_rows:
                    eid = int(mr["event_id"])
                    mk = str(mr["metric_key"])
                    if mk in ("takeback_depth", "scapular_retraction"):
                        session_has_dual = True
                    ev_metric_map.setdefault(eid, {})[mk] = float(mr["value_num"])
                view_mode = "dual_view" if session_has_dual else "single_view"

                fh_in_sess = 0
                bh_in_sess = 0
                other_in_sess = 0

                for r in ev_rows:
                    try:
                        payload = json.loads(r["payload_json"]) if r["payload_json"] else {}
                    except Exception:
                        payload = {}
                    clip_url, freeze_url = _extract_payload_media_urls(payload, s_dir, workspace_root)
                    ev_metrics = {
                        m["key"]: m.get("value")
                        for m in (payload.get("analysis_metrics") or [])
                        if isinstance(m, dict) and m.get("key")
                    }
                    # Merge DB numeric metrics (which already sanitize >5000px/s outliers)
                    for mk, mv in ev_metric_map.get(int(r["event_id"]), {}).items():
                        ev_metrics.setdefault(mk, mv)

                    clean_c, clean_m = _sanitize_racket_speeds(
                        r["racket_speed_px_s"],
                        r["racket_max_speed_px_s"] or ev_metrics.get("racket_max_speed"),
                    )
                    if clean_m is not None:
                        ev_metrics["racket_max_speed"] = clean_m
                    else:
                        ev_metrics.pop("racket_max_speed", None)

                    advice_msgs, effective_warnings = _extract_swing_warnings(payload, ev_metrics)
                    side = _classify_stroke_side(r["stroke_type"])
                    if side == "forehand":
                        fh_in_sess += 1
                    elif side == "backhand":
                        bh_in_sess += 1
                    else:
                        other_in_sess += 1

                    record: Dict[str, Any] = {
                        "session_id": sid,
                        "display_time": s["display_time"],
                        "timestamp": float(s["timestamp"] or 0.0),
                        "source_name": s["source_name"],
                        "view_mode": view_mode,
                        "event_id": int(r["event_id"]),
                        "stroke_type": r["stroke_type"],
                        "stroke_side": side,
                        "start_frame": r["start_frame"],
                        "end_frame": r["end_frame"],
                        "contact_frame": r["contact_frame"],
                        "clip_url": clip_url,
                        "impact_freeze_url": freeze_url,
                        "racket_speed_px_s": clean_c,
                        "racket_max_speed_px_s": clean_m,
                        "coach_messages": advice_msgs,
                        "warnings": effective_warnings,
                    }
                    for key, _, _, _, _ in _GLOBAL_METRIC_SPECS:
                        val = _safe_float(ev_metrics.get(key))
                        if val is not None:
                            record[key] = round(val, 2)
                    side_swings[side].append(record)

                if fh_in_sess or bh_in_sess or other_in_sess:
                    hidden_siblings = dedup_summary["kept_duplicates_map"].get(sid, [])
                    sessions_catalog.append(
                        {
                            "session_id": sid,
                            "display_time": s["display_time"],
                            "timestamp": float(s["timestamp"] or 0.0),
                            "source_name": s["source_name"],
                            "source_sha256": s.get("source_sha256"),
                            "view_mode": view_mode,
                            "forehand_count": fh_in_sess,
                            "backhand_count": bh_in_sess,
                            "matched_swings": fh_in_sess + bh_in_sess + other_in_sess,
                            "duplicate_count": len(hidden_siblings),
                            "source_group_size": 1 + len(hidden_siblings),
                        }
                    )

            def _build_side_summary(side_key: str, swings: List[Dict[str, Any]]) -> Dict[str, Any]:
                n = len(swings)
                dual_cnt = sum(1 for sw in swings if sw["view_mode"] == "dual_view")
                sess_ids = {sw["session_id"] for sw in swings}
                mid = n // 2
                early_swings = swings[:mid] if n >= 4 else swings
                recent_swings = swings[mid:] if n >= 4 else swings

                metrics_out: Dict[str, Any] = {}
                consistency_scores: List[float] = []
                for m_key, lbl, unit, higher_better, scope in _GLOBAL_METRIC_SPECS:
                    vals = [float(sw[m_key]) for sw in swings if sw.get(m_key) is not None]
                    if not vals:
                        continue
                    p25, med, p75, iqr, c_score = _compute_quantiles(vals)
                    if c_score is not None and m_key in (
                        "shoulder_turn_change", "hip_shoulder_separation", "arm_extension",
                        "preparation_knee_flexion", "contact_lateral_distance", "weight_transfer",
                        "balance_drift", "racket_max_speed",
                    ):
                        consistency_scores.append(c_score)
                    early_vals = [float(sw[m_key]) for sw in early_swings if sw.get(m_key) is not None]
                    recent_vals = [float(sw[m_key]) for sw in recent_swings if sw.get(m_key) is not None]
                    early_med = round(float(statistics.median(early_vals)), 2) if (n >= 4 and early_vals) else None
                    recent_med = round(float(statistics.median(recent_vals)), 2) if (n >= 4 and recent_vals) else None
                    trend_delta = (
                        round(recent_med - early_med, 2)
                        if (early_med is not None and recent_med is not None)
                        else None
                    )
                    metrics_out[m_key] = {
                        "label": lbl,
                        "unit": unit,
                        "higher_better": higher_better,
                        "view_scope": scope,
                        "p25": p25,
                        "median": med,
                        "p75": p75,
                        "iqr": iqr,
                        "consistency_score": c_score,
                        "min": round(min(vals), 2),
                        "max": round(max(vals), 2),
                        "count": len(vals),
                        "early_median": early_med,
                        "recent_median": recent_med,
                        "trend_delta": trend_delta,
                    }

                consistency_index = (
                    round(float(statistics.mean(consistency_scores)), 1)
                    if consistency_scores
                    else None
                )

                # Warnings (overall + early vs recent rate comparison)
                total_counts: Dict[str, int] = {}
                early_counts: Dict[str, int] = {}
                recent_counts: Dict[str, int] = {}
                for sw in swings:
                    for w in set(sw.get("warnings") or []):
                        total_counts[w] = total_counts.get(w, 0) + 1
                for sw in early_swings:
                    for w in set(sw.get("warnings") or []):
                        early_counts[w] = early_counts.get(w, 0) + 1
                for sw in recent_swings:
                    for w in set(sw.get("warnings") or []):
                        recent_counts[w] = recent_counts.get(w, 0) + 1

                warnings_out = []
                denom_all = max(1, n)
                denom_e = max(1, len(early_swings))
                denom_r = max(1, len(recent_swings))
                for msg, cnt in sorted(total_counts.items(), key=lambda kv: kv[1], reverse=True)[:6]:
                    e_rate = round((early_counts.get(msg, 0) / denom_e) * 100.0, 1) if n >= 4 else None
                    r_rate = round((recent_counts.get(msg, 0) / denom_r) * 100.0, 1) if n >= 4 else None
                    warnings_out.append(
                        {
                            "message": msg,
                            "count": cnt,
                            "rate_pct": round((cnt / denom_all) * 100.0, 1),
                            "early_rate_pct": e_rate,
                            "recent_rate_pct": r_rate,
                            "delta_pct": round(r_rate - e_rate, 1) if (e_rate is not None and r_rate is not None) else None,
                        }
                    )

                # Per-session chronological trend series on this side
                by_sess: Dict[str, List[Dict[str, Any]]] = {}
                sess_order: List[str] = []
                for sw in swings:
                    sid_k = sw["session_id"]
                    if sid_k not in by_sess:
                        by_sess[sid_k] = []
                        sess_order.append(sid_k)
                    by_sess[sid_k].append(sw)

                trend_series = []
                for sid_k in sess_order:
                    group = by_sess[sid_k]
                    pt: Dict[str, Any] = {
                        "session_id": sid_k,
                        "display_time": group[0]["display_time"],
                        "timestamp": group[0]["timestamp"],
                        "view_mode": group[0]["view_mode"],
                        "count": len(group),
                    }
                    sess_c = []
                    for m_key, _, _, _, _ in _GLOBAL_METRIC_SPECS:
                        m_vals = [float(sw[m_key]) for sw in group if sw.get(m_key) is not None]
                        if m_vals:
                            _, med, _, _, c_s = _compute_quantiles(m_vals)
                            pt[m_key] = med
                            if c_s is not None:
                                sess_c.append(c_s)
                    pt["consistency_index"] = round(float(statistics.mean(sess_c)), 1) if sess_c else None
                    trend_series.append(pt)

                patterns = _analyze_side_patterns(swings, trend_series, metrics_out, warnings_out)
                return {
                    "side": side_key,
                    "total_swings": n,
                    "session_count": len(sess_ids),
                    "dual_view_swings": dual_cnt,
                    "single_view_swings": n - dual_cnt,
                    "consistency_index": consistency_index,
                    "metrics": metrics_out,
                    "commonalities": patterns["commonalities"],
                    "contradictions": patterns["contradictions"],
                    "personal_attribution": _compute_personal_attribution(swings),
                    "coach_warnings": warnings_out,
                    "trend_series": trend_series,
                    "video_presets": _select_video_presets(swings, patterns["contradictions"]),
                    "swings": swings,
                }

            fh_summary = _build_side_summary("forehand", side_swings["forehand"])
            bh_summary = _build_side_summary("backhand", side_swings["backhand"])
            fh_summary["latest_review"] = self.get_latest_coach_review(pid, "forehand")
            bh_summary["latest_review"] = self.get_latest_coach_review(pid, "backhand")

            symmetry_comparison = []
            for m_key, lbl, unit, higher_better, scope in _GLOBAL_METRIC_SPECS:
                fm = fh_summary["metrics"].get(m_key)
                bm = bh_summary["metrics"].get(m_key)
                if not fm and not bm:
                    continue
                f_med = fm["median"] if fm else None
                b_med = bm["median"] if bm else None
                diff = round(f_med - b_med, 2) if (f_med is not None and b_med is not None) else None
                symmetry_comparison.append(
                    {
                        "key": m_key,
                        "label": lbl,
                        "unit": unit,
                        "higher_better": higher_better,
                        "view_scope": scope,
                        "fh": fm,
                        "bh": bm,
                        "fh_minus_bh": diff,
                    }
                )

            return {
                "player": dict(player_row),
                "days": days_val,
                "dedup": {
                    "enabled": dedup_enabled,
                    "raw_session_count": dedup_summary["raw_session_count"],
                    "unique_session_count": dedup_summary["unique_session_count"],
                    "duplicate_session_count": dedup_summary["duplicate_session_count"],
                    "duplicate_group_count": dedup_summary["duplicate_group_count"],
                    "hidden_session_ids": dedup_summary["hidden_session_ids"] if dedup_enabled else [],
                    "groups": dedup_summary["groups"],
                },
                "total_sessions": len(sessions_catalog),
                "total_swings": (
                    fh_summary["total_swings"]
                    + bh_summary["total_swings"]
                    + len(side_swings["other"])
                ),
                "other_swings_count": len(side_swings["other"]),
                "forehand": fh_summary,
                "backhand": bh_summary,
                "symmetry_comparison": symmetry_comparison,
                "sessions_catalog": sessions_catalog,
            }

    def save_coach_review(
        self,
        player_id: str,
        stroke_side: str,
        provider: str,
        model: str,
        review_text: str,
        session_ids: Optional[List[str]] = None,
        video_paths: Optional[List[str]] = None,
        evidence_packet: Optional[Dict[str, Any]] = None,
        days_scope: str = "all",
        dedup: int = 1,
    ) -> Dict[str, Any]:
        """Save a generated AI coaching review record into SQLite."""
        pid = str(player_id or DEFAULT_PLAYER_ID).strip() or DEFAULT_PLAYER_ID
        side = "backhand" if str(stroke_side).lower() == "backhand" else "forehand"
        now_dt = datetime.now()
        now_ts = now_dt.timestamp()
        now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        review_id = f"rev_{int(now_ts)}_{uuid.uuid4().hex[:6]}"
        sids_json = json.dumps(session_ids or [], ensure_ascii=False)
        vpaths_json = json.dumps(video_paths or [], ensure_ascii=False)
        evidence_json = json.dumps(evidence_packet or {}, ensure_ascii=False)

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO coach_reviews (
                    review_id, player_id, stroke_side, provider, model,
                    review_text, session_ids_json, video_paths_json, evidence_json,
                    days_scope, dedup, created_at, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id, pid, side, str(provider), str(model),
                    str(review_text), sids_json, vpaths_json, evidence_json,
                    str(days_scope or "all"), int(dedup), now_str, now_ts
                ),
            )

        return {
            "review_id": review_id,
            "player_id": pid,
            "stroke_side": side,
            "provider": provider,
            "model": model,
            "review_text": review_text,
            "session_ids": session_ids or [],
            "video_paths": video_paths or [],
            "days_scope": days_scope,
            "dedup": dedup,
            "created_at": now_str,
            "timestamp": now_ts,
        }

    def get_latest_coach_review(
        self,
        player_id: str,
        stroke_side: str = "forehand",
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the latest coach review for a player and stroke side."""
        pid = str(player_id or DEFAULT_PLAYER_ID).strip() or DEFAULT_PLAYER_ID
        side = "backhand" if str(stroke_side).lower() == "backhand" else "forehand"
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT review_id, player_id, stroke_side, provider, model,
                       review_text, session_ids_json, video_paths_json, evidence_json,
                       days_scope, dedup, created_at, timestamp
                FROM coach_reviews
                WHERE player_id = ? AND stroke_side = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (pid, side),
            ).fetchone()
            if not row:
                return None
            return self._format_review_row(row)

    def list_coach_reviews(
        self,
        player_id: str,
        stroke_side: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List historical coach reviews for a player."""
        pid = str(player_id or DEFAULT_PLAYER_ID).strip() or DEFAULT_PLAYER_ID
        clause = "WHERE player_id = ?"
        params: List[Any] = [pid]
        if stroke_side:
            side = "backhand" if str(stroke_side).lower() == "backhand" else "forehand"
            clause += " AND stroke_side = ?"
            params.append(side)
        params.append(max(1, min(100, int(limit))))

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT review_id, player_id, stroke_side, provider, model,
                       review_text, session_ids_json, video_paths_json, evidence_json,
                       days_scope, dedup, created_at, timestamp
                FROM coach_reviews
                {clause}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [self._format_review_row(r) for r in rows]

    def _format_review_row(self, row: Any) -> Dict[str, Any]:
        try:
            sids = json.loads(row["session_ids_json"] or "[]")
        except Exception:
            sids = []
        try:
            vpaths = json.loads(row["video_paths_json"] or "[]")
        except Exception:
            vpaths = []
        try:
            ev = json.loads(row["evidence_json"] or "{}")
        except Exception:
            ev = {}
        return {
            "review_id": str(row["review_id"]),
            "player_id": str(row["player_id"]),
            "stroke_side": str(row["stroke_side"]),
            "provider": str(row["provider"]),
            "model": str(row["model"]),
            "review_text": str(row["review_text"]),
            "session_ids": sids,
            "video_paths": vpaths,
            "evidence_packet": ev,
            "days_scope": str(row["days_scope"]),
            "dedup": int(row["dedup"]),
            "created_at": str(row["created_at"]),
            "timestamp": float(row["timestamp"]),
        }
