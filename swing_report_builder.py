#!/usr/bin/env python3
"""Build a standalone HTML report for swing video and JSON review."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

from swing_session_quality import build_session_quality_dashboard

RADAR_AXES = [
    ("shoulder_turn", "转肩"),
    ("takeback", "引拍"),
    ("arm_extension", "延展"),
    ("racket_speed", "挥速"),
    ("leg_drive", "蹬地"),
]


def _build_radar_svg(sub_scores: Dict[str, float], width: int = 240, height: int = 220, dark_theme: bool = False) -> str:
    """Build a standalone inline SVG 5-axis biomechanical quality radar chart."""
    if not sub_scores:
        return ""
    cx, cy = width / 2.0, height / 2.0
    r_max = 66.0
    rings = [0.25, 0.5, 0.75, 1.0]
    n = len(RADAR_AXES)

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;display:block;margin:auto;" class="radar-svg">'
    ]
    # Web rings
    for lvl in rings:
        pts = []
        for i in range(n):
            ang = -math.pi / 2.0 + i * (2.0 * math.pi / n)
            x = cx + r_max * lvl * math.cos(ang)
            y = cy + r_max * lvl * math.sin(ang)
            pts.append(f"{x:.1f},{y:.1f}")
        dash = ' stroke-dasharray="2,2"' if lvl < 1.0 else ""
        if dark_theme:
            stroke_color = "#233348" if lvl < 1.0 else "#364c6a"
        else:
            stroke_color = "#d8d0c0" if lvl < 1.0 else "#b0a898"
        svg_parts.append(
            f'<polygon points="{" ".join(pts)}" fill="none" stroke="{stroke_color}" stroke-width="1"{dash}/>'
        )

    # Spokes and data points
    data_pts = []
    vertex_circles = []
    labels_svg = []
    for i, (key, label) in enumerate(RADAR_AXES):
        ang = -math.pi / 2.0 + i * (2.0 * math.pi / n)
        cos_a = math.cos(ang)
        sin_a = math.sin(ang)
        ox = cx + r_max * cos_a
        oy = cy + r_max * sin_a
        spoke_stroke = "#233348" if dark_theme else "#d8d0c0"
        svg_parts.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{ox:.1f}" y2="{oy:.1f}" stroke="{spoke_stroke}" stroke-width="1"/>'
        )

        score = float(sub_scores.get(key, 0.0) or 0.0)
        clamped = max(0.0, min(100.0, score))
        r_val = max(6.0, r_max * (clamped / 100.0))
        dx = cx + r_val * cos_a
        dy = cy + r_val * sin_a
        data_pts.append(f"{dx:.1f},{dy:.1f}")
        dot_fill = "#00f0ff" if dark_theme else "#0f7b6c"
        dot_stroke = "#0a0f1a" if dark_theme else "#fff"
        vertex_circles.append(
            f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="3.5" fill="{dot_fill}" stroke="{dot_stroke}" stroke-width="1.5"/>'
        )

        # Label placement
        lx = cx + (r_max + 18.0) * cos_a
        ly = cy + (r_max + 18.0) * sin_a
        anchor = "middle" if abs(cos_a) < 0.15 else ("start" if cos_a > 0 else "end")
        text_color = "#93a4bb" if dark_theme else "#334155"
        labels_svg.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" dominant-baseline="central" font-size="10" font-weight="600" fill="{text_color}">{label} {score:.0f}</text>'
        )

    poly_fill = "rgba(0, 240, 255, 0.22)" if dark_theme else "rgba(15,123,108,0.22)"
    poly_stroke = "#00f0ff" if dark_theme else "#0f7b6c"
    svg_parts.append(
        f'<polygon points="{" ".join(data_pts)}" fill="{poly_fill}" stroke="{poly_stroke}" stroke-width="2.2"/>'
    )
    svg_parts.extend(vertex_circles)
    svg_parts.extend(labels_svg)
    svg_parts.append("</svg>")
    return "".join(svg_parts)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_event_json(frame_json_path: str) -> str:
    path = Path(frame_json_path)
    return str(path.with_name(f"{path.stem}_swing_events.json"))


def default_coach_json(frame_json_path: str) -> str:
    path = Path(frame_json_path)
    return str(path.with_name(f"{path.stem}_coach_dataset.json"))


def default_video_path(frame_json_path: str) -> str:
    path = Path(frame_json_path)
    return str(path.with_name(f"{path.stem}_swing_annotated.mp4"))


def default_evaluation_json(frame_json_path: str) -> str:
    path = Path(frame_json_path)
    return str(path.with_name(f"{path.stem}_swing_evaluation.json"))


def default_report_path(frame_json_path: str) -> str:
    path = Path(frame_json_path)
    return str(path.with_name(f"{path.stem}_swing_report.html"))


def _rel(path: Optional[str], base_dir: Path) -> Optional[str]:
    if not path:
        return None
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return path


def _event_by_id(events: List[Dict]) -> Dict[int, Dict]:
    out = {}
    for event in events or []:
        if isinstance(event, dict) and event.get("event_id") is not None:
            out[int(event["event_id"])] = event
    return out


def _classification_text(context: Dict) -> str:
    player = context.get("player") or {}
    camera = context.get("camera") or {}
    swing = context.get("swing") or {}
    hand = {"right": "右手", "left": "左手"}.get(
        player.get("dominant_hand"),
        "惯用手未知",
    )
    view = {
        "facing_player": "球员面向相机",
        "behind_player": "相机位于球员后方",
        "side_or_uncertain": "侧向/机位不确定",
        "unknown": "机位未知",
    }.get(camera.get("view"), "机位未知")
    side = {
        "forehand": "正手侧",
        "backhand": "反手侧",
        "uncertain": "挥拍侧不确定",
        "unknown": "挥拍侧未知",
    }.get(swing.get("side"), "挥拍侧未知")
    return f"球员 {hand} · 机位 {view} · 分类证据 {side}"


def build_report_payload(
    frame_json_path: str,
    event_json_path: Optional[str] = None,
    coach_json_path: Optional[str] = None,
    video_path: Optional[str] = None,
    evaluation_json_path: Optional[str] = None,
) -> Dict:
    event_json_path = event_json_path or default_event_json(frame_json_path)
    coach_json_path = coach_json_path or default_coach_json(frame_json_path)
    video_path = video_path or default_video_path(frame_json_path)
    evaluation_json_path = evaluation_json_path or default_evaluation_json(frame_json_path)

    frame_data = load_json(frame_json_path)
    event_data = load_json(event_json_path)
    coach_data = load_json(coach_json_path) if coach_json_path and os.path.exists(coach_json_path) else {"events": []}
    evaluation_data = (
        load_json(evaluation_json_path)
        if evaluation_json_path and os.path.exists(evaluation_json_path)
        else None
    )
    coach_lookup = _event_by_id(coach_data.get("events", []))

    merged_events = []
    for event in event_data.get("events", []):
        coach_event = coach_lookup.get(int(event.get("event_id", 0)), {})
        scores = coach_event.get("scores") or {}
        quality_flags = coach_event.get("quality_flags") or event.get("quality_flags") or {}
        start_boundary = (event.get("evidence") or {}).get("start_boundary") or {}
        classification_context = (event.get("evidence") or {}).get("classification_context") or {}

        bio = event.get("biomechanics") or coach_event.get("biomechanics") or {}
        ext = event.get("extended_biomechanics") or bio.get("extended_biomechanics") or {}
        metrics = bio.get("metrics") or {}
        sqs = (
            metrics.get("swing_quality_score")
            or ext.get("swing_quality_score")
            or event.get("swing_quality_score")
            or {}
        )
        if isinstance(sqs, dict) and "value" in sqs and "overall_score" not in sqs:
            sqs = dict(sqs)
            sqs["overall_score"] = sqs.get("value")

        seq = (
            metrics.get("kinematic_sequence")
            or ext.get("kinematic_sequence")
            or event.get("kinematic_sequence")
            or {}
        )
        rkt = (
            metrics.get("racket_speed")
            or ext.get("racket_head_speed")
            or event.get("racket_speed")
            or {}
        )
        brush = (
            metrics.get("brush_angle")
            or ext.get("brush_angle")
            or event.get("brush_angle")
            or {}
        )
        stc = (
            metrics.get("stance")
            or ext.get("stance")
            or event.get("stance")
            or {}
        )
        leg = (
            metrics.get("leg_drive")
            or ext.get("leg_drive")
            or event.get("leg_drive")
            or {}
        )
        advices = event.get("coach_advice") or coach_event.get("coach_advice") or []

        has_bio = bool(bio or ext or sqs or event.get("swing_score") or event.get("swing_grade"))
        raw_score = (
            event.get("swing_score")
            or bio.get("swing_score")
            or (sqs.get("overall_score") if isinstance(sqs, dict) else None)
        )
        if raw_score is None and has_bio:
            raw_score = scores.get("overall_score") or event.get("overall_score")

        if raw_score is not None:
            try:
                raw_score = float(raw_score)
                if 0.0 < raw_score <= 1.0:
                    raw_score = raw_score * 100.0
            except (ValueError, TypeError):
                pass

        raw_grade = (
            event.get("swing_grade")
            or bio.get("swing_grade")
            or (sqs.get("grade") if isinstance(sqs, dict) else None)
        )
        if has_bio and not raw_grade and raw_score is not None and isinstance(raw_score, (int, float)):
            if raw_score >= 85.0:
                raw_grade = "PRO"
            elif raw_score >= 70.0:
                raw_grade = "ADVANCED"
            elif raw_score >= 55.0:
                raw_grade = "INTERMEDIATE"
            else:
                raw_grade = "DEVELOPING"

        merged_events.append(
            {
                "event_id": event.get("event_id"),
                "stroke_type": event.get("stroke_type"),
                "confidence": event.get("confidence"),
                "start_frame": event.get("start_frame"),
                "contact_frame": event.get("contact_frame") or (coach_event.get("frames") or {}).get("contact_frame"),
                "peak_frame": event.get("peak_frame"),
                "end_frame": event.get("end_frame"),
                "phase_counts": event.get("phase_counts") or {},
                "start_boundary": start_boundary,
                "classification_context": classification_context,
                "quality_flags": quality_flags,
                "diagnosis_tags": coach_event.get("diagnosis_tags") or list(quality_flags.get("warnings") or []),
                "overall_score": scores.get("overall_score") if scores.get("overall_score") is not None else event.get("overall_score"),
                "overall_score_9": scores.get("overall_score_9") or (event.get("coach_calibration") or {}).get("visible_technique_score_9"),
                "score_uncertainty_9": scores.get("uncertainty_9") or (event.get("coach_calibration") or {}).get("uncertainty_9"),
                "score_confidence": scores.get("confidence") or (event.get("coach_calibration") or {}).get("confidence"),
                "coach_calibration": coach_event.get("coach_calibration") or event.get("coach_calibration") or {},
                "contact_score": scores.get("contact_score"),
                "preparation_score": scores.get("preparation_score"),
                "follow_through_score": scores.get("follow_through_score"),
                "data_quality": coach_event.get("data_quality") or {},
                "coach": coach_event,
                "biomechanics": bio,
                "extended_biomechanics": ext,
                "swing_quality_score": sqs,
                "swing_score": raw_score,
                "swing_grade": raw_grade,
                "kinematic_sequence": seq,
                "racket_speed": rkt,
                "brush_angle": brush,
                "stance": stc,
                "leg_drive": leg,
                "advice_list": advices,
                "impact_freeze_path": event.get("impact_freeze_path") or (event.get("snapshots") or {}).get("impact_freeze"),
            }
        )

    return {
        "paths": {
            "frame_json": frame_json_path,
            "event_json": event_json_path,
            "coach_json": coach_json_path,
            "video": video_path,
            "evaluation_json": evaluation_json_path if evaluation_data else None,
        },
        "video_info": frame_data.get("video_info") or {},
        "summary": {
            "frames": len(frame_data.get("frames", [])),
            **(event_data.get("summary") or {}),
            "coach": coach_data.get("summary") or {},
        },
        "session_quality": (
            (event_data.get("summary") or {}).get("session_quality")
            or build_session_quality_dashboard(event_data.get("events") or [])
        ),
        "events": merged_events,
        "timeline": {
            "fps": (frame_data.get("video_info") or {}).get("fps") or 25.0,
            "total_frames": len(frame_data.get("frames", [])),
            "frame_trace": [
                {
                    "frame": trace.get("frame"),
                    "event_id": trace.get("event_id"),
                    "phase": trace.get("phase"),
                    "motion_energy": trace.get("motion_energy"),
                }
                for trace in event_data.get("frame_trace", [])
            ],
        },
        "evaluation": evaluation_data,
    }


def _score_text(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.0f}"
    except (TypeError, ValueError):
        return str(value)


def _stroke_options(selected: Optional[str]) -> str:
    labels = ["Forehand", "Backhand", "Two-Handed Backhand", "Serve", "No Swing", "Unclear"]
    return "".join(
        f'<option value="{html.escape(label)}"{" selected" if label == selected else ""}>{html.escape(label)}</option>'
        for label in labels
    )


def _json_script(payload: Dict) -> str:
    return html.escape(json.dumps(payload, ensure_ascii=False), quote=False)


def _percent_text(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return str(value)


def _frame_input_value(value) -> str:
    if value is None:
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ""


def _dashboard_number(value, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def _session_dashboard_html(dashboard: Dict) -> str:
    quality = dashboard.get("quality") or {}
    drift = dashboard.get("drift") or {}
    status_labels = {
        "warming_up": "预热中",
        "stable": "稳定",
        "improving": "改善",
        "attention": "需关注",
        "integrity_blocked": "事件重叠",
    }
    grade_labels = {
        "good": "良好",
        "watch": "留意",
        "poor": "偏低",
        "unavailable": "无数据",
        "event_integrity_confounded": "受事件重叠影响",
    }
    kpis = [
        (
            "证据质量",
            _dashboard_number(quality.get("evidence_quality_score_100"), 0, "/100"),
            grade_labels.get(quality.get("grade"), str(quality.get("grade") or "-")),
        ),
        (
            "可见动作均值",
            _dashboard_number(quality.get("visible_technique_mean_9"), 1, "/9"),
            (
                "±" + _dashboard_number(quality.get("median_uncertainty_9"), 1)
                if quality.get("median_uncertainty_9") is not None
                else "证据不足"
            ),
        ),
        (
            "校准覆盖",
            _dashboard_number(
                (quality.get("calibrated_event_ratio") or 0.0) * 100,
                0,
                "%",
            ),
            f"{int(dashboard.get('event_count') or 0)} 次挥拍",
        ),
        (
            "触球证据",
            _dashboard_number(
                (quality.get("contact_supported_ratio") or 0.0) * 100,
                0,
                "%",
            ),
            "达到 Coach 门槛",
        ),
        (
            "人工复核",
            _dashboard_number(
                (quality.get("review_recommended_ratio") or 0.0) * 100,
                0,
                "%",
            ),
            "建议复核比例",
        ),
    ]
    kpi_html = "".join(
        '<div class="session-kpi">'
        f'<span>{html.escape(label)}</span><strong>{html.escape(value)}</strong>'
        f'<small>{html.escape(note)}</small></div>'
        for label, value, note in kpis
    )
    series_rows = []
    for point in dashboard.get("series") or []:
        score = point.get("visible_score_9")
        evidence = point.get("evidence_quality_100")
        score_width = max(0.0, min(100.0, float(score or 0.0) / 9.0 * 100.0))
        evidence_width = max(0.0, min(100.0, float(evidence or 0.0)))
        warnings = len(point.get("warnings") or [])
        series_rows.append(
            '<div class="session-series-row">'
            f'<strong>#{int(point.get("event_id") or 0)}</strong>'
            '<div class="session-bars">'
            f'<i class="score-bar" style="width:{score_width:.1f}%"></i>'
            f'<i class="quality-bar" style="width:{evidence_width:.1f}%"></i>'
            '</div>'
            f'<span>{_dashboard_number(score, 1, "/9")}</span>'
            f'<small>{_dashboard_number(evidence, 0, "/100")} · {warnings}警告</small>'
            '</div>'
        )
    indicator_rows = []
    indicator_status = {
        "stable": "稳定",
        "improving": "改善",
        "declining": "下降",
        "shifted": "变化",
        "camera_shift_confounded": "受机位影响",
        "insufficient_events": "样本不足",
        "unavailable": "无数据",
    }
    for indicator in drift.get("indicators") or []:
        delta = indicator.get("delta")
        if delta is None:
            delta = indicator.get("relative_delta")
        indicator_rows.append(
            '<div class="drift-row">'
            f'<span>{html.escape(str(indicator.get("label") or indicator.get("name")))}</span>'
            f'<strong>{_dashboard_number(indicator.get("baseline"), 2)} → '
            f'{_dashboard_number(indicator.get("recent"), 2)}</strong>'
            f'<small>{html.escape(indicator_status.get(indicator.get("status"), str(indicator.get("status") or "-")))}'
            f' · Δ {_dashboard_number(delta, 2)}</small></div>'
        )
    alerts = dashboard.get("alerts") or []
    alert_html = (
        '<ul class="session-alerts">'
        + "".join(
            f'<li data-severity="{html.escape(str(alert.get("severity") or "medium"))}">'
            f'{html.escape(str(alert.get("message") or alert.get("code")))}</li>'
            for alert in alerts
        )
        + "</ul>"
        if alerts
        else '<p class="session-no-alert">当前没有会话级报警。</p>'
    )
    state = str(drift.get("status") or dashboard.get("monitoring_state") or "warming_up")
    if state == "integrity_blocked":
        readiness = "相邻事件范围重叠；修正事件切分前暂停漂移结论。"
    elif not drift.get("ready"):
        readiness = f"至少需要 {int(drift.get('minimum_event_count') or 6)} 次挥拍；当前只展示观察值。"
    else:
        readiness = f"前 {int(drift.get('window_size') or 1)} 次与最近 {int(drift.get('window_size') or 1)} 次对比。"
    return f"""
    <section class="panel session-dashboard" aria-label="会话质量与漂移看板">
      <div class="session-dashboard-head">
        <div><h2>会话质量与漂移</h2><p>{html.escape(readiness)}</p></div>
        <strong data-state="{html.escape(state)}">{html.escape(status_labels.get(state, state))}</strong>
      </div>
      <div class="session-kpis">{kpi_html}</div>
      <div class="session-dashboard-grid">
        <div><h3>逐拍趋势</h3><div class="session-series">{''.join(series_rows) or '<p>等待挥拍事件。</p>'}</div></div>
        <div><h3>前段 → 最近</h3><div class="drift-rows">{''.join(indicator_rows)}</div></div>
      </div>
      {alert_html}
      <div class="session-legend"><span><i class="score-dot"></i>可见动作分</span><span><i class="quality-dot"></i>证据质量</span></div>
    </section>
    """


def render_report_html(payload: Dict, output_path: str) -> str:
    output_dir = Path(output_path).parent
    video_src = _rel(payload["paths"].get("video"), output_dir)
    event_cards = []
    for event in payload.get("events", []):
        warnings = event.get("quality_flags", {}).get("warnings") or []
        tags = event.get("diagnosis_tags") or []
        start_boundary = event.get("start_boundary") or {}
        classification_context = event.get("classification_context") or {}
        boundary_text = " · ".join(
            str(value)
            for value in (
                start_boundary.get("mode"),
                start_boundary.get("confidence"),
            )
            if value
        ) or "legacy"
        score_9 = event.get("overall_score_9")
        uncertainty_9 = event.get("score_uncertainty_9")
        calibrated_score_text = (
            f"可见动作 {float(score_9):.1f}/9"
            + (f" ±{float(uncertainty_9):.1f}" if uncertainty_9 is not None else "")
            if score_9 is not None
            else "可见动作评分：证据不足"
        )

        # 1. 综合技术评级与100分制仪表
        swing_grade = event.get("swing_grade")
        swing_score = event.get("swing_score")
        if swing_grade:
            grade_upper = str(swing_grade).upper()
            tier_class = f"tier-{grade_upper.lower()}"
            score_display = f"{float(swing_score):.1f}分" if swing_score is not None else ""
            grade_labels = {
                "PRO": "PRO · 职业级",
                "ADVANCED": "ADVANCED · 进阶级",
                "INTERMEDIATE": "INTERMEDIATE · 中级",
                "DEVELOPING": "DEVELOPING · 基础级",
            }
            grade_label = grade_labels.get(grade_upper, grade_upper)
            head_badge_html = f'<span class="tier-pill {tier_class}">{html.escape(grade_label)} <strong style="margin-left:4px;">{score_display}</strong></span>'
        else:
            head_badge_html = f"<span>score {_score_text(event.get('overall_score'))}</span>"

        meter_pct = f"{float(swing_score):.0f}" if swing_score is not None else _score_text(event.get('overall_score'))

        # 2. 5维生物力学技术雷达图
        sqs = event.get("swing_quality_score") or {}
        sub_scores = sqs.get("sub_scores") if isinstance(sqs, dict) else {}
        radar_svg = _build_radar_svg(sub_scores) if sub_scores else ""
        radar_html = f"""
        <div class="bio-radar-wrapper">
          <div class="bio-radar-title">5维生物力学质量雷达</div>
          {radar_svg}
        </div>
        """ if radar_svg else ""

        # 3. 动力学链时序时延条
        seq = event.get("kinematic_sequence") or {}
        details = seq.get("details") if isinstance(seq, dict) and isinstance(seq.get("details"), dict) else seq
        seq_quality = (seq.get("value") or details.get("sequence_quality") or "OPTIMAL") if isinstance(seq, dict) else "OPTIMAL"
        dt_hip_sh = details.get("latency_hip_to_shoulder_ms") if isinstance(details, dict) else None
        dt_sh_rkt = details.get("latency_shoulder_to_racket_ms") if isinstance(details, dict) else None

        kinematic_html = ""
        if dt_hip_sh is not None or dt_sh_rkt is not None:
            dt_hip_sh_val = float(dt_hip_sh or 0.0)
            dt_sh_rkt_val = float(dt_sh_rkt or 0.0)
            dt_hip_sh_pct = min(100.0, max(5.0, (dt_hip_sh_val / 80.0) * 100.0))
            dt_sh_rkt_pct = min(100.0, max(5.0, (dt_sh_rkt_val / 90.0) * 100.0))
            kinematic_html = f"""
            <div class="kinematic-box">
              <div class="kinematic-header">
                <span>动力学链传递: <strong>下肢 ➔ 髋/骨盆 ➔ 肩/躯干 ➔ 球拍</strong></span>
                <span class="seq-badge seq-{html.escape(str(seq_quality).lower())}">{html.escape(str(seq_quality))}</span>
              </div>
              <div class="kinematic-bars">
                <div class="kinematic-bar-row">
                  <span class="k-label">髋-肩时序延时 (Δt_hip_sh):</span>
                  <span class="k-val">{dt_hip_sh_val:.1f} ms</span>
                  <div class="k-track"><div class="k-fill" style="width:{dt_hip_sh_pct:.0f}%;"></div></div>
                </div>
                <div class="kinematic-bar-row">
                  <span class="k-label">肩-拍时序延时 (Δt_sh_rkt):</span>
                  <span class="k-val">{dt_sh_rkt_val:.1f} ms</span>
                  <div class="k-track"><div class="k-fill k-fill-rkt" style="width:{dt_sh_rkt_pct:.0f}%;"></div></div>
                </div>
              </div>
            </div>
            """

        # 4. 击球遥测指标网格
        rkt = event.get("racket_speed") or {}
        brush = event.get("brush_angle") or {}
        stc = event.get("stance") or {}
        leg = event.get("leg_drive") or {}

        contact_kmh = rkt.get("contact_kmh") or rkt.get("contact_speed_kmh")
        max_kmh = rkt.get("max_kmh") or rkt.get("max_speed_kmh")
        brush_angle = brush.get("low_to_high_angle_deg") or brush.get("angle_deg")
        drop_ratio = brush.get("drop_depth_ratio")
        stance_type = stc.get("stance_type") or stc.get("value")
        leg_ratio = leg.get("drive_ratio") or leg.get("value")

        has_telemetry = any(v is not None for v in [contact_kmh, max_kmh, brush_angle, drop_ratio, stance_type, leg_ratio])
        telemetry_html = ""
        if has_telemetry:
            kmh_text = f"{float(contact_kmh):.1f} / {float(max_kmh):.1f} km/h" if contact_kmh is not None and max_kmh is not None else "-"
            brush_text = f"{float(brush_angle):+.1f}°" if brush_angle is not None else "-"
            if drop_ratio is not None:
                try:
                    brush_text += f" (下潜 {float(drop_ratio):.2f}x)"
                except (ValueError, TypeError):
                    brush_text += f" (下潜 {drop_ratio})"
            stance_text = str(stance_type or "-")
            if leg_ratio is not None:
                try:
                    stance_text += f" · 蹬地 {float(leg_ratio):.2f}x"
                except (ValueError, TypeError):
                    stance_text += f" · 蹬地 {leg_ratio}"
            telemetry_html = f"""
            <div class="telemetry-grid">
              <div class="telem-item"><span class="telem-label">拍头挥速 (击球/峰值)</span><strong class="telem-val">{html.escape(kmh_text)}</strong></div>
              <div class="telem-item"><span class="telem-label">刷球仰角与下潜</span><strong class="telem-val">{html.escape(brush_text)}</strong></div>
              <div class="telem-item"><span class="telem-label">击球站位与蹬地比</span><strong class="telem-val">{html.escape(stance_text)}</strong></div>
            </div>
            """

        # 5. 击球定格快照特写查找
        snap_rel = None
        c_frame = event.get("contact_frame")
        ev_id = event.get("event_id")
        candidates = []
        if event.get("impact_freeze_path"):
            candidates.append(Path(event["impact_freeze_path"]))

        search_dirs = [output_dir, output_dir / "snapshots"]
        if payload.get("paths", {}).get("video"):
            search_dirs.append(Path(payload["paths"]["video"]).parent)
        if payload.get("paths", {}).get("frame_json"):
            search_dirs.append(Path(payload["paths"]["frame_json"]).parent)
        scratch_dir = Path("/Users/krum5539/.gemini/antigravity/brain/853db2fd-bbb9-45de-8209-c65d2189b516/scratch")
        if scratch_dir.exists():
            search_dirs.append(scratch_dir)

        for s_dir in search_dirs:
            if s_dir.exists() and c_frame is not None:
                candidates.extend(list(s_dir.glob(f"*{c_frame}*impact_freeze.jpg")))
                candidates.extend(list(s_dir.glob(f"*{c_frame}*.jpg")))
            if s_dir.exists() and ev_id is not None:
                candidates.extend(list(s_dir.glob(f"*event_{ev_id}*.jpg")))

        for cand in candidates:
            if cand.exists() and cand.is_file():
                snap_rel = _rel(str(cand), output_dir)
                break

        snapshot_html = ""
        if snap_rel:
            snapshot_html = f"""
            <div class="impact-freeze-container">
              <a href="{html.escape(snap_rel)}" target="_blank" class="impact-freeze-link" title="点击查看击球瞬间定格特写">
                <img src="{html.escape(snap_rel)}" alt="击球瞬间定格特写" loading="lazy" class="impact-freeze-img" />
                <span class="impact-freeze-badge">⚡ 击球瞬间定格特写 (第 {c_frame} 帧)</span>
              </a>
            </div>
            """

        # 6. 教练纠错建议
        advices = event.get("advice_list") or []
        advices_html = ""
        if advices:
            items = []
            for adv in advices:
                msg = adv.get("message") if isinstance(adv, dict) else str(adv)
                code = adv.get("code") if isinstance(adv, dict) else ""
                conf = f" ({adv['confidence']*100:.0f}%)" if isinstance(adv, dict) and "confidence" in adv else ""
                items.append(
                    f'<li class="coach-advice-item"><span class="advice-bullet">💡</span><strong>{html.escape(str(code))}:</strong> {html.escape(str(msg))}{html.escape(conf)}</li>'
                )
            advices_html = f'<div class="coach-advices-box"><ul class="coach-advice-list">{"".join(items)}</ul></div>'

        event_cards.append(
            f"""
            <article class="event-card" data-annotation-card data-annotation-id="model-{html.escape(str(event.get('event_id')))}" data-source-event-id="{html.escape(str(event.get('event_id')))}" data-event-id="{html.escape(str(event.get('event_id')))}">
              <div class="event-head">
                <strong>Event {html.escape(str(event.get('event_id')))} · {html.escape(str(event.get('stroke_type')))}</strong>
                {head_badge_html}
              </div>
              <div class="frames">start {event.get('start_frame')} · contact {event.get('contact_frame')} · peak {event.get('peak_frame')} · end {event.get('end_frame')}</div>
              <div class="frames">start boundary {html.escape(boundary_text)}</div>
              <div class="frames">{html.escape(_classification_text(classification_context))}</div>
              <div class="frames">{html.escape(calibrated_score_text)}</div>
              <button class="event-jump" type="button" data-event-id="{html.escape(str(event.get('event_id')))}">定位到事件</button>
              <div class="meter"><i style="width:{meter_pct}%"></i></div>
              <p>confidence {_score_text(event.get('confidence'))} · contact {_score_text(event.get('contact_score'))} · prep {_score_text(event.get('preparation_score'))} · follow {_score_text(event.get('follow_through_score'))}</p>
              <p class="tags">{html.escape(', '.join(tags + warnings) or 'no quality warnings')}</p>
              {radar_html}
              {kinematic_html}
              {telemetry_html}
              {snapshot_html}
              {advices_html}
              <div class="annotation-box">
                <div class="annotation-frame-grid">
                  <label>人工开始帧
                    <input type="number" min="0" step="1" data-field="start_frame" value="{_frame_input_value(event.get('start_frame'))}">
                  </label>
                  <label>人工触球帧
                    <input type="number" min="0" step="1" data-field="contact_frame" value="{_frame_input_value(event.get('contact_frame'))}">
                  </label>
                  <label>人工结束帧
                    <input type="number" min="0" step="1" data-field="end_frame" value="{_frame_input_value(event.get('end_frame'))}">
                  </label>
                </div>
                <label>人工类型
                  <select class="annotation-stroke" data-field="actual_stroke_type">
                    {_stroke_options(event.get('stroke_type'))}
                  </select>
                </label>
                <label><input class="annotation-count-correct" type="checkbox" data-field="count_correct" checked> 计数正确</label>
                <label><input class="annotation-valid-hit" type="checkbox" data-field="valid_hit" checked> 有效击球</label>
                <label><input class="annotation-needs-review" type="checkbox" data-field="needs_review" checked> 待人工确认（确认后取消）</label>
                <div class="annotation-tags">
                  <label><input type="checkbox" data-tag="wrong_stroke_type"> 类型错</label>
                  <label><input type="checkbox" data-tag="missed_event"> 漏识别</label>
                  <label><input type="checkbox" data-tag="extra_event"> 多计</label>
                  <label><input type="checkbox" data-tag="bad_contact_frame"> 击球帧错</label>
                  <label><input type="checkbox" data-tag="bad_ball_track"> 球轨错</label>
                  <label><input type="checkbox" data-tag="bad_pose"> 姿态错</label>
                </div>
                <textarea class="annotation-note" data-field="note" rows="2" placeholder="可选备注"></textarea>
              </div>
            </article>
            """
        )

    summary = payload.get("summary") or {}
    evaluation = payload.get("evaluation") or {}
    evaluation_summary = evaluation.get("summary") or {}
    evaluation_block = ""
    if evaluation_summary:
        provisional_reasons = evaluation_summary.get("provisional_reasons") or []
        status_text = (
            "provisional: " + ", ".join(str(reason) for reason in provisional_reasons)
            if evaluation_summary.get("provisional")
            else "finalized"
        )
        evaluation_block = f"""
      <div class="evaluation-summary">
        <h2>Evaluation Summary</h2>
        <p><strong>status {html.escape(status_text)}</strong></p>
        <p>precision {_percent_text(evaluation_summary.get('precision'))} · recall {_percent_text(evaluation_summary.get('recall'))} · F1 {_percent_text(evaluation_summary.get('f1'))} · stroke accuracy {_percent_text(evaluation_summary.get('stroke_type_accuracy'))} · contact accuracy {_percent_text(evaluation_summary.get('contact_accuracy'))}</p>
      </div>
        """
    session_dashboard_block = _session_dashboard_html(
        payload.get("session_quality") or {}
    )
    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Swing Analysis Report</title>
  <style>
    :root {{
      --bg: #f5f2ea;
      --ink: #17211f;
      --muted: #63706b;
      --line: #d8d0c0;
      --accent: #0f7b6c;
      --warn: #b94f2f;
      --panel: #fffaf0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(120deg, #f5f2ea 0%, #e9f2ed 100%);
      color: var(--ink);
      font-family: "Avenir Next", "PingFang SC", sans-serif;
    }}
    header, main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; align-items: end; }}
    h1 {{ margin: 0; font-size: 30px; letter-spacing: 0; }}
    .summary {{ color: var(--muted); }}
    .layout {{ display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(340px, .55fr); gap: 18px; }}
    video {{ width: 100%; background: #111; border: 1px solid var(--line); }}
    .panel, .event-card {{ background: rgba(255,250,240,.92); border: 1px solid var(--line); border-radius: 8px; }}
    .panel {{ padding: 16px; }}
    .session-dashboard {{ grid-column: 1 / -1; }}
    .session-dashboard-head {{ display:flex; justify-content:space-between; align-items:start; gap:16px; }}
    .session-dashboard-head h2,.session-dashboard h3 {{ margin:0; }}
    .session-dashboard-head p {{ margin:4px 0 0; }}
    .session-dashboard-head > strong {{ border-radius:999px; padding:6px 11px; background:#e8dfd0; color:var(--muted); }}
    .session-dashboard-head > strong[data-state="stable"],.session-dashboard-head > strong[data-state="improving"] {{ background:#dceee6; color:var(--accent); }}
    .session-dashboard-head > strong[data-state="attention"],.session-dashboard-head > strong[data-state="integrity_blocked"] {{ background:#f7dfd6; color:var(--warn); }}
    .session-kpis {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:9px; margin:16px 0; }}
    .session-kpi {{ display:grid; gap:2px; padding:11px; border:1px solid var(--line); border-radius:8px; background:#fffdf7; }}
    .session-kpi span,.session-kpi small {{ color:var(--muted); font-size:12px; }}
    .session-kpi strong {{ font-size:21px; }}
    .session-dashboard-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    .session-series,.drift-rows {{ display:grid; gap:7px; margin-top:10px; }}
    .session-series-row {{ display:grid; grid-template-columns:30px minmax(120px,1fr) 55px 116px; align-items:center; gap:8px; font-size:12px; }}
    .session-series-row small {{ color:var(--muted); text-align:right; }}
    .session-bars {{ position:relative; height:18px; border-radius:5px; background:#e8dfd0; overflow:hidden; }}
    .session-bars i {{ position:absolute; left:0; height:8px; }}
    .session-bars .score-bar {{ top:0; background:#5346a3; }}
    .session-bars .quality-bar {{ bottom:0; background:var(--accent); }}
    .drift-row {{ display:grid; grid-template-columns:1fr auto; gap:2px 10px; padding:7px 9px; border:1px solid var(--line); border-radius:7px; }}
    .drift-row small {{ grid-column:1 / -1; color:var(--muted); }}
    .session-alerts {{ margin:14px 0 0; padding-left:20px; color:var(--warn); }}
    .session-no-alert {{ color:var(--accent); margin:14px 0 0; }}
    .session-legend {{ display:flex; gap:14px; margin-top:12px; color:var(--muted); font-size:12px; }}
    .session-legend i {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px; }}
    .score-dot {{ background:#5346a3; }} .quality-dot {{ background:var(--accent); }}
    .event-explorer {{ grid-column: 1 / -1; }}
    .event-explorer-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }}
    .event-explorer h2 {{ margin: 0; font-size: 20px; }}
    .event-timeline {{ display: grid; gap: 9px; margin: 18px 0 12px; }}
    .timeline-ruler, .timeline-lane {{ display: grid; grid-template-columns: 86px minmax(0, 1fr); gap: 10px; align-items: center; }}
    .timeline-label {{ color: var(--muted); font-size: 12px; text-align: right; }}
    .timeline-track {{ position: relative; min-height: 33px; border-radius: 8px; background: repeating-linear-gradient(90deg, #e8dfd0 0 1px, transparent 1px 10%); border: 1px solid var(--line); overflow: visible; }}
    .timeline-track--ruler {{ min-height: 22px; color: var(--muted); font-size: 11px; }}
    .timeline-track--ruler span {{ position: absolute; top: 3px; transform: translateX(-50%); }}
    .timeline-event {{ position: absolute; top: 5px; height: 21px; min-width: 8px; padding: 0 7px; border: 0; border-radius: 5px; background: var(--accent); color: white; overflow: hidden; text-align: left; text-overflow: ellipsis; white-space: nowrap; }}
    .timeline-event.is-active {{ outline: 3px solid #142a25; outline-offset: 2px; }}
    .timeline-marker {{ position: absolute; top: -4px; width: 10px; height: 40px; padding: 0; border: 0; border-radius: 999px; background: #17211f; color: transparent; }}
    .timeline-marker--contact {{ background: #bd4e2c; }}
    .timeline-marker--peak {{ background: #5346a3; }}
    .timeline-playhead {{ position: absolute; z-index: 3; top: -8px; bottom: -8px; width: 2px; background: #17211f; pointer-events: none; }}
    .timeline-playhead::before {{ content: ""; position: absolute; top: -2px; left: -4px; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 7px solid #17211f; }}
    .timeline-controls {{ display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 12px; }}
    .timeline-controls input {{ width: 100%; accent-color: var(--accent); }}
    .timeline-status {{ color: var(--muted); font-size: 13px; min-width: 145px; text-align: right; }}
    .timeline-legend {{ display: flex; gap: 14px; flex-wrap: wrap; color: var(--muted); font-size: 12px; }}
    .timeline-legend i {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; background: #17211f; }}
    .timeline-legend .contact-dot {{ background: #bd4e2c; }}
    .timeline-legend .peak-dot {{ background: #5346a3; }}
    .events {{ display: grid; gap: 12px; max-height: 70vh; overflow: auto; }}
    .event-card {{ padding: 14px; }}
    .event-card.is-active {{ border-color: var(--accent); box-shadow: 0 0 0 2px rgba(15,123,108,.16); }}
    .event-head {{ display: flex; justify-content: space-between; gap: 10px; }}
    .frames, .tags, p {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
    .tags {{ color: var(--warn); }}
    .annotation-box {{ display: grid; gap: 8px; border-top: 1px solid var(--line); margin-top: 12px; padding-top: 12px; }}
    .annotation-box label {{ color: var(--ink); font-size: 13px; }}
    .annotation-box select, .annotation-box textarea, .annotation-box input[type="number"] {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 7px; background: #fffdf7; color: var(--ink); }}
    .annotation-frame-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; }}
    .annotation-tags {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }}
    .actions {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }}
    .timeline-review {{ display: block; margin: 12px 0; padding: 10px; border: 1px solid var(--line); border-radius: 6px; background: #fffdf7; }}
    .annotation-readiness {{ margin: -4px 0 12px; color: var(--warn); font-size: 13px; }}
    .annotation-readiness[data-state="ready"] {{ color: var(--accent); }}
    .manual-events {{ display: grid; gap: 10px; margin: 14px 0; }}
    .manual-event-card {{ padding: 12px; border: 1px solid #d28d45; border-radius: 8px; background: #fff7e9; }}
    .manual-event-card h3 {{ margin: 0; font-size: 16px; }}
    .manual-event-head {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 8px; }}
    .remove-manual-event {{ border-color: var(--warn); background: transparent; color: var(--warn); padding: 5px 8px; }}
    .evaluation-summary {{ margin: 14px 0; padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: #fffdf7; }}
    .evaluation-summary h2 {{ margin: 0 0 6px; font-size: 18px; }}
    .evaluation-summary p {{ margin: 0; }}
    button {{ border: 1px solid var(--accent); background: var(--accent); color: white; border-radius: 6px; padding: 9px 12px; cursor: pointer; }}
    .event-jump {{ margin-top: 10px; padding: 6px 9px; font-size: 12px; background: transparent; color: var(--accent); }}
    .file-label {{ border: 1px solid var(--line); background: #fffdf7; color: var(--ink); border-radius: 6px; padding: 9px 12px; cursor: pointer; }}
    .file-label input {{ display: none; }}
    .import-status {{ color: var(--muted); font-size: 13px; align-self: center; }}
    .meter {{ height: 8px; background: #e4dccd; border-radius: 999px; overflow: hidden; margin: 10px 0; }}
    .meter i {{ display: block; height: 100%; background: var(--accent); }}
    /* 算法2.0 / 生物力学扩展样式 */
    .tier-pill {{
      display: inline-flex;
      align-items: center;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .02em;
      border: 1px solid var(--line);
    }}
    .tier-pro {{
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.18), rgba(0, 240, 255, 0.22));
      border-color: #f59e0b;
      color: #92400e;
    }}
    .tier-advanced {{
      background: rgba(16, 185, 129, 0.15);
      border-color: #10b981;
      color: #065f46;
    }}
    .tier-intermediate {{
      background: rgba(56, 189, 248, 0.15);
      border-color: #38bdf8;
      color: #0369a1;
    }}
    .tier-developing {{
      background: rgba(249, 115, 22, 0.15);
      border-color: #f97316;
      color: #9a3412;
    }}
    .bio-radar-wrapper {{
      margin: 12px 0 8px;
      padding: 10px;
      background: rgba(245, 242, 234, 0.6);
      border: 1px solid var(--line);
      border-radius: 8px;
      text-align: center;
    }}
    .bio-radar-title {{
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .kinematic-box {{
      margin: 10px 0;
      padding: 10px 12px;
      background: #fdfaf3;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .kinematic-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .seq-badge {{
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .seq-optimal {{ background: #d1fae5; color: #065f46; }}
    .seq-acceptable {{ background: #e0f2fe; color: #0369a1; }}
    .seq-suboptimal {{ background: #fee2e2; color: #991b1b; }}
    .kinematic-bars {{ display: grid; gap: 6px; }}
    .kinematic-bar-row {{
      display: grid;
      grid-template-columns: 140px 55px 1fr;
      align-items: center;
      gap: 8px;
      font-size: 11px;
    }}
    .k-label {{ color: var(--muted); }}
    .k-val {{ font-weight: 600; text-align: right; }}
    .k-track {{
      height: 6px;
      background: #e8dfd0;
      border-radius: 999px;
      overflow: hidden;
    }}
    .k-fill {{
      height: 100%;
      background: var(--accent);
      border-radius: 999px;
    }}
    .k-fill-rkt {{ background: #5346a3; }}
    .telemetry-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      margin: 10px 0;
    }}
    .telem-item {{
      padding: 6px 8px;
      background: #f7f3ea;
      border: 1px solid var(--line);
      border-radius: 6px;
      display: grid;
      gap: 2px;
    }}
    .telem-label {{ font-size: 10px; color: var(--muted); }}
    .telem-val {{ font-size: 11px; color: var(--ink); font-weight: 600; }}
    .impact-freeze-container {{
      margin: 10px 0;
    }}
    .impact-freeze-link {{
      display: block;
      position: relative;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--line);
      box-shadow: 0 2px 6px rgba(0,0,0,0.05);
      transition: transform .15s ease, box-shadow .15s ease;
    }}
    .impact-freeze-link:hover {{
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }}
    .impact-freeze-img {{
      display: block;
      width: 100%;
      height: auto;
      object-fit: cover;
    }}
    .impact-freeze-badge {{
      position: absolute;
      bottom: 6px;
      right: 6px;
      padding: 3px 8px;
      border-radius: 4px;
      background: rgba(15, 23, 42, 0.82);
      color: #00f0ff;
      font-size: 11px;
      font-weight: 600;
      backdrop-filter: blur(4px);
    }}
    .coach-advices-box {{
      margin: 10px 0;
      padding: 8px 10px;
      background: #eff6f4;
      border-left: 3px solid var(--accent);
      border-radius: 4px;
    }}
    .coach-advice-list {{
      margin: 0;
      padding-left: 0;
      list-style: none;
      display: grid;
      gap: 4px;
    }}
    .coach-advice-item {{
      font-size: 12px;
      line-height: 1.4;
      color: #17211f;
      display: flex;
      gap: 6px;
      align-items: baseline;
    }}
    .advice-bullet {{ font-size: 11px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #17211f; color: #eaf5ef; padding: 14px; border-radius: 8px; max-height: 360px; overflow: auto; }}
    @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} header {{ display:block; }} .session-kpis {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .session-dashboard-grid {{ grid-template-columns:1fr; }} .timeline-ruler, .timeline-lane {{ grid-template-columns: 54px minmax(0, 1fr); }} .timeline-label {{ font-size: 11px; }} .annotation-frame-grid {{ grid-template-columns: 1fr; }} .telemetry-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Swing Analysis Report</h1>
      <div class="summary">events {html.escape(str(summary.get('swing_event_count', len(payload.get('events', [])))))} · frames {html.escape(str(summary.get('frames')))} · types {html.escape(json.dumps(summary.get('swing_event_type_counts', {}), ensure_ascii=False))}</div>
    </div>
    <div class="summary">{html.escape(payload['paths'].get('frame_json') or '')}</div>
  </header>
  <main class="layout">
    {session_dashboard_block}
    <section class="panel event-explorer" aria-label="挥拍事件时间轴">
      <div class="event-explorer-head">
        <h2>事件时间轴</h2>
        <span id="timeline-status" class="timeline-status">Frame - · -</span>
      </div>
      <div id="event-timeline" class="event-timeline"></div>
      <div class="timeline-controls">
        <input id="frame-scrubber" type="range" min="0" value="0" aria-label="按帧定位视频">
        <button id="play-event" type="button">播放当前事件</button>
      </div>
      <div class="timeline-legend"><span><i></i>当前播放帧</span><span><i class="contact-dot"></i>触球候选</span><span><i class="peak-dot"></i>动作峰值</span></div>
    </section>
    <section class="panel">
      <video id="swing-video" controls src="{html.escape(video_src or '')}"></video>
      <div class="actions">
        <button type="button" onclick="refreshAnnotations()">生成标注 JSON</button>
        <button type="button" onclick="downloadAnnotations()">下载标注 JSON</button>
        <label class="file-label">导入标注 JSON
          <input id="annotation-import-file" type="file" accept="application/json,.json" onchange="importAnnotations(event)">
        </label>
        <span id="import-status" class="import-status"></span>
      </div>
      <label class="timeline-review">
        <input id="timeline-review-complete" type="checkbox">
        已完整检查整段视频（勾选后才计算 Precision / Recall / F1）
      </label>
      <p id="annotation-readiness" class="annotation-readiness"></p>
      <button id="add-missed-event" type="button" onclick="addMissedEvent()">新增漏检挥拍</button>
      <div id="manual-events" class="manual-events"></div>
      {evaluation_block}
      <h2>Raw Summary</h2>
      <pre id="raw-summary"></pre>
      <h2>Manual Annotations</h2>
      <pre id="annotation-json"></pre>
    </section>
    <aside class="events">
      {''.join(event_cards)}
    </aside>
  </main>
  <script type="application/json" id="report-data">{_json_script(payload)}</script>
  <script>
    const data = JSON.parse(document.getElementById('report-data').textContent);
    const video = document.getElementById('swing-video');
    const timeline = document.getElementById('event-timeline');
    const scrubber = document.getElementById('frame-scrubber');
    const timelineStatus = document.getElementById('timeline-status');
    const fps = Number((data.timeline || {{}}).fps || (data.video_info || {{}}).fps || 25);
    const maxEventFrame = Math.max(0, ...data.events.flatMap(event => [event.start_frame, event.contact_frame, event.peak_frame, event.end_frame].map(Number).filter(Number.isFinite)));
    const totalFrames = Math.max(1, Number((data.timeline || {{}}).total_frames || 0), maxEventFrame + 1);
    let activeEventId = data.events.length ? Number(data.events[0].event_id) : null;
    scrubber.max = String(totalFrames - 1);

    function percentForFrame(frame) {{
      return Math.max(0, Math.min(100, (Number(frame) / Math.max(1, totalFrames - 1)) * 100));
    }}
    function seekFrame(frame, shouldPlay = false) {{
      const safeFrame = Math.max(0, Math.min(totalFrames - 1, Math.round(Number(frame) || 0)));
      video.currentTime = safeFrame / fps;
      scrubber.value = String(safeFrame);
      updatePlaybackState(safeFrame);
      if (shouldPlay) video.play();
    }}
    function selectEvent(eventId, seek = true) {{
      const event = data.events.find(item => Number(item.event_id) === Number(eventId));
      if (!event) return;
      activeEventId = Number(event.event_id);
      document.querySelectorAll('[data-event-id]').forEach(node => node.classList.toggle('is-active', Number(node.dataset.eventId) === activeEventId));
      if (seek) seekFrame(event.start_frame);
    }}
    function marker(label, frame, className, eventId) {{
      if (!Number.isFinite(Number(frame))) return '';
      const position = percentForFrame(frame);
      return `<button type="button" class="timeline-marker ${{className}}" data-event-id="${{eventId}}" data-frame="${{frame}}" style="left:calc(${{position}}% - 5px)" aria-label="${{label}}：第 ${{frame}} 帧">${{label}}</button>`;
    }}
    function renderTimeline() {{
      const ruler = [0, .25, .5, .75, 1].map(ratio => {{
        const frame = Math.round((totalFrames - 1) * ratio);
        return `<span style="left:${{ratio * 100}}%">${{frame}}</span>`;
      }}).join('');
      const lanes = data.events.map(event => {{
        const start = Number(event.start_frame) || 0;
        const end = Math.max(start + 1, Number(event.end_frame) || start + 1);
        const left = percentForFrame(start);
        const width = Math.max(1.5, percentForFrame(end) - left);
        return `<div class="timeline-lane"><span class="timeline-label">事件 ${{event.event_id}}</span><div class="timeline-track"><button type="button" class="timeline-event" data-event-id="${{event.event_id}}" data-frame="${{start}}" style="left:${{left}}%;width:${{width}}%" title="${{event.stroke_type}} · ${{start}}-${{end}}">${{event.stroke_type}}</button>${{marker('触球候选', event.contact_frame, 'timeline-marker--contact', event.event_id)}}${{marker('动作峰值', event.peak_frame, 'timeline-marker--peak', event.event_id)}}<i class="timeline-playhead" aria-hidden="true"></i></div></div>`;
      }}).join('');
      timeline.innerHTML = `<div class="timeline-ruler"><span class="timeline-label">帧号</span><div class="timeline-track timeline-track--ruler">${{ruler}}</div></div>${{lanes || '<p>未检测到挥拍事件。</p>'}}`;
      timeline.querySelectorAll('[data-frame]').forEach(button => button.addEventListener('click', () => {{
        selectEvent(button.dataset.eventId, false);
        seekFrame(button.dataset.frame, button.classList.contains('timeline-event'));
      }}));
    }}
    function updatePlaybackState(frame) {{
      const currentFrame = Math.max(0, Math.min(totalFrames - 1, Math.round(Number(frame) || 0)));
      scrubber.value = String(currentFrame);
      const currentEvent = data.events.find(event => currentFrame >= Number(event.start_frame) && currentFrame <= Number(event.end_frame));
      if (currentEvent) selectEvent(currentEvent.event_id, false);
      const phase = ((data.timeline || {{}}).frame_trace || []).find(trace => Number(trace.frame) === currentFrame)?.phase || 'ready';
      timelineStatus.textContent = `Frame ${{currentFrame}} · ${{(currentFrame / fps).toFixed(2)}}s · ${{phase}}`;
      timeline.querySelectorAll('.timeline-playhead').forEach(playhead => playhead.style.left = `${{percentForFrame(currentFrame)}}%`);
    }}
    document.getElementById('raw-summary').textContent = JSON.stringify({{
      paths: data.paths,
      summary: data.summary,
      events: data.events.map(e => ({{
        event_id: e.event_id,
        stroke_type: e.stroke_type,
        frames: [e.start_frame, e.contact_frame, e.peak_frame, e.end_frame],
        quality_flags: e.quality_flags,
        diagnosis_tags: e.diagnosis_tags,
        scores: {{
          overall: e.overall_score,
          contact: e.contact_score,
          preparation: e.preparation_score,
          follow_through: e.follow_through_score
        }}
      }}))
    }}, null, 2);
    const manualEvents = document.getElementById('manual-events');
    const timelineReviewComplete = document.getElementById('timeline-review-complete');
    const annotationReadiness = document.getElementById('annotation-readiness');
    let manualCounter = 0;

    function integerField(card, field) {{
      const input = card.querySelector(`[data-field="${{field}}"]`);
      if (!input || input.value.trim() === '') return null;
      const value = Number(input.value);
      return Number.isFinite(value) ? Math.round(value) : null;
    }}
    function annotationFromCard(card) {{
      const sourceText = card.dataset.sourceEventId || '';
      const sourceNumber = Number(sourceText);
      const sourceEventId = sourceText === '' ? null : (Number.isFinite(sourceNumber) ? sourceNumber : sourceText);
      const original = data.events.find(event => String(event.event_id) === String(sourceEventId)) || {{}};
      return {{
        annotation_id: card.dataset.annotationId,
        source_event_id: sourceEventId,
        predicted_stroke_type: original.stroke_type || null,
        actual_stroke_type: card.querySelector('[data-field="actual_stroke_type"]').value,
        count_correct: card.querySelector('[data-field="count_correct"]').checked,
        valid_hit: card.querySelector('[data-field="valid_hit"]').checked,
        needs_review: card.querySelector('[data-field="needs_review"]').checked,
        issue_tags: Array.from(card.querySelectorAll('[data-tag]:checked')).map(input => input.dataset.tag),
        note: card.querySelector('[data-field="note"]').value.trim(),
        frames: {{
          start: integerField(card, 'start_frame'),
          contact: integerField(card, 'contact_frame'),
          peak: original.peak_frame ?? null,
          end: integerField(card, 'end_frame')
        }},
        quality_flags: original.quality_flags || {{}}
      }};
    }}
    function collectAnnotations() {{
      return {{
        schema_version: 'swing_manual_annotations_v2',
        timeline_review_complete: timelineReviewComplete.checked,
        source: data.paths,
        summary: data.summary,
        events: Array.from(document.querySelectorAll('[data-annotation-card]')).map(annotationFromCard)
      }};
    }}
    function updateAnnotationReadiness(payload) {{
      const pending = payload.events.filter(event => event.needs_review).length;
      if (pending > 0) {{
        annotationReadiness.dataset.state = 'blocked';
        annotationReadiness.textContent = `还有 ${{pending}} 条“需要复核”，评估指标将保持 provisional。`;
      }} else if (!payload.timeline_review_complete) {{
        annotationReadiness.dataset.state = 'blocked';
        annotationReadiness.textContent = '请完整检查整段视频后勾选确认项。';
      }} else {{
        annotationReadiness.dataset.state = 'ready';
        annotationReadiness.textContent = '已满足正式评估条件，可以下载标注 JSON。';
      }}
    }}
    function refreshAnnotations() {{
      const payload = collectAnnotations();
      updateAnnotationReadiness(payload);
      document.getElementById('annotation-json').textContent = JSON.stringify(payload, null, 2);
    }}
    function setField(card, field, value) {{
      const input = card.querySelector(`[data-field="${{field}}"]`);
      if (!input || value === undefined || value === null) return;
      if (input.type === 'checkbox') {{
        input.checked = Boolean(value);
      }} else {{
        input.value = String(value);
      }}
    }}
    function bindAnnotationInputs(root) {{
      root.querySelectorAll('input, select, textarea').forEach(element => {{
        element.addEventListener('change', refreshAnnotations);
      }});
      root.querySelectorAll('textarea, input[type="number"]').forEach(element => {{
        element.addEventListener('input', refreshAnnotations);
      }});
    }}
    function applyAnnotationToCard(card, imported) {{
      setField(card, 'actual_stroke_type', imported.actual_stroke_type);
      setField(card, 'count_correct', imported.count_correct);
      setField(card, 'valid_hit', imported.valid_hit);
      setField(card, 'needs_review', imported.needs_review);
      setField(card, 'note', imported.note || '');
      const frames = imported.frames || {{}};
      setField(card, 'start_frame', frames.start ?? imported.start_frame);
      setField(card, 'contact_frame', frames.contact ?? imported.contact_frame);
      setField(card, 'end_frame', frames.end ?? imported.end_frame);
      const tags = new Set(imported.issue_tags || []);
      card.querySelectorAll('[data-tag]').forEach(input => input.checked = tags.has(input.dataset.tag));
    }}
    function addMissedEvent(imported = null) {{
      manualCounter += 1;
      const card = document.createElement('article');
      const importedId = imported && (imported.annotation_id ?? imported.event_id);
      card.className = 'manual-event-card';
      card.dataset.annotationCard = '';
      card.dataset.annotationId = importedId != null ? String(importedId) : `manual-${{manualCounter}}`;
      card.dataset.sourceEventId = imported && imported.source_event_id != null ? String(imported.source_event_id) : '';
      card.innerHTML = `
        <div class="manual-event-head">
          <h3>人工补充挥拍</h3>
          <button class="remove-manual-event" type="button">删除</button>
        </div>
        <div class="annotation-box">
          <div class="annotation-frame-grid">
            <label>人工开始帧<input type="number" min="0" step="1" data-field="start_frame"></label>
            <label>人工触球帧<input type="number" min="0" step="1" data-field="contact_frame"></label>
            <label>人工结束帧<input type="number" min="0" step="1" data-field="end_frame"></label>
          </div>
          <label>人工类型
            <select class="annotation-stroke" data-field="actual_stroke_type">
              <option value="Forehand">Forehand</option>
              <option value="Backhand">Backhand</option>
              <option value="Two-Handed Backhand">Two-Handed Backhand</option>
              <option value="Serve">Serve</option>
              <option value="Volley">Volley</option>
              <option value="Unclear" selected>Unclear</option>
            </select>
          </label>
          <label><input type="checkbox" data-field="valid_hit" checked> 有效击球</label>
          <label><input type="checkbox" data-field="count_correct"> 计数正确</label>
          <label><input type="checkbox" data-field="needs_review"> 需要复核</label>
          <div class="annotation-tags">
            <label><input type="checkbox" data-tag="wrong_type"> 类型错误</label>
            <label><input type="checkbox" data-tag="contact_timing"> 触球帧偏差</label>
            <label><input type="checkbox" data-tag="event_boundary"> 边界偏差</label>
            <label><input type="checkbox" data-tag="missed_event"> 漏检事件</label>
          </div>
          <label>备注<textarea rows="2" data-field="note" placeholder="漏检原因、动作特点等"></textarea></label>
        </div>`;
      card.querySelector('.remove-manual-event').addEventListener('click', () => {{
        card.remove();
        refreshAnnotations();
      }});
      bindAnnotationInputs(card);
      if (imported) applyAnnotationToCard(card, imported);
      manualEvents.appendChild(card);
      refreshAnnotations();
      return card;
    }}
    function applyImportedAnnotations(payload) {{
      const importedEvents = Array.isArray(payload.events) ? payload.events : [];
      timelineReviewComplete.checked = Boolean(payload.timeline_review_complete);
      manualEvents.replaceChildren();
      let applied = 0;
      for (const imported of importedEvents) {{
        const sourceId = imported.source_event_id ?? imported.event_id ?? null;
        const card = sourceId == null ? null : document.querySelector(`.event-card[data-event-id="${{String(sourceId)}}"]`);
        if (card) {{
          if (imported.annotation_id != null) card.dataset.annotationId = String(imported.annotation_id);
          applyAnnotationToCard(card, imported);
        }} else {{
          addMissedEvent(imported);
        }}
        applied += 1;
      }}
      refreshAnnotations();
      document.getElementById('import-status').textContent = `已导入 ${{applied}} 条标注`;
      return applied;
    }}
    function importAnnotations(event) {{
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {{
        try {{
          applyImportedAnnotations(JSON.parse(reader.result));
        }} catch (err) {{
          document.getElementById('import-status').textContent = `导入失败: ${{err.message}}`;
        }}
      }};
      reader.readAsText(file);
    }}
    function downloadAnnotations() {{
      refreshAnnotations();
      const blob = new Blob([document.getElementById('annotation-json').textContent], {{type: 'application/json'}});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'swing_manual_annotations_v2.json';
      link.click();
      URL.revokeObjectURL(link.href);
    }}
    document.querySelectorAll('.event-jump').forEach(button => button.addEventListener('click', () => selectEvent(button.dataset.eventId)));
    scrubber.addEventListener('input', event => seekFrame(event.target.value));
    document.getElementById('play-event').addEventListener('click', () => {{
      const event = data.events.find(item => Number(item.event_id) === activeEventId);
      if (event) seekFrame(event.start_frame, true);
      else video.play();
    }});
    video.addEventListener('timeupdate', () => updatePlaybackState(Math.round(video.currentTime * fps)));
    bindAnnotationInputs(document);
    renderTimeline();
    updatePlaybackState(0);
    refreshAnnotations();
  </script>
</body>
</html>
"""
    return html_doc


def write_report_html(payload: Dict, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report_html(payload, str(output)), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standalone swing video + JSON report page.")
    parser.add_argument("frame_json", help="Per-frame pipeline JSON.")
    parser.add_argument("--event-json", help="Swing event JSON. Defaults to <frame_json_stem>_swing_events.json")
    parser.add_argument("--coach-json", help="Coach dataset JSON. Defaults to <frame_json_stem>_coach_dataset.json")
    parser.add_argument("--video", help="Annotated swing video. Defaults to <frame_json_stem>_swing_annotated.mp4")
    parser.add_argument("--evaluation-json", help="Optional model-vs-human evaluation JSON.")
    parser.add_argument("--output-html", help="Report path. Defaults to <frame_json_stem>_swing_report.html")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_html = args.output_html or default_report_path(args.frame_json)
    payload = build_report_payload(
        args.frame_json,
        args.event_json,
        args.coach_json,
        args.video,
        args.evaluation_json,
    )
    write_report_html(payload, output_html)
    print(f"events={len(payload.get('events', []))}")
    print(f"html={output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
