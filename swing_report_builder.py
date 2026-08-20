#!/usr/bin/env python3
"""Build a standalone HTML report for swing video and JSON review."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from swing_session_quality import build_session_quality_dashboard


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
                "overall_score": scores.get("overall_score"),
                "overall_score_9": scores.get("overall_score_9") or (event.get("coach_calibration") or {}).get("visible_technique_score_9"),
                "score_uncertainty_9": scores.get("uncertainty_9") or (event.get("coach_calibration") or {}).get("uncertainty_9"),
                "score_confidence": scores.get("confidence") or (event.get("coach_calibration") or {}).get("confidence"),
                "coach_calibration": coach_event.get("coach_calibration") or event.get("coach_calibration") or {},
                "contact_score": scores.get("contact_score"),
                "preparation_score": scores.get("preparation_score"),
                "follow_through_score": scores.get("follow_through_score"),
                "data_quality": coach_event.get("data_quality") or {},
                "coach": coach_event,
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
        event_cards.append(
            f"""
            <article class="event-card" data-annotation-card data-annotation-id="model-{html.escape(str(event.get('event_id')))}" data-source-event-id="{html.escape(str(event.get('event_id')))}" data-event-id="{html.escape(str(event.get('event_id')))}">
              <div class="event-head">
                <strong>Event {html.escape(str(event.get('event_id')))} · {html.escape(str(event.get('stroke_type')))}</strong>
                <span>score {_score_text(event.get('overall_score'))}</span>
              </div>
              <div class="frames">start {event.get('start_frame')} · contact {event.get('contact_frame')} · peak {event.get('peak_frame')} · end {event.get('end_frame')}</div>
              <div class="frames">start boundary {html.escape(boundary_text)}</div>
              <div class="frames">{html.escape(_classification_text(classification_context))}</div>
              <div class="frames">{html.escape(calibrated_score_text)}</div>
              <button class="event-jump" type="button" data-event-id="{html.escape(str(event.get('event_id')))}">定位到事件</button>
              <div class="meter"><i style="width:{_score_text(event.get('overall_score'))}%"></i></div>
              <p>confidence {_score_text(event.get('confidence'))} · contact {_score_text(event.get('contact_score'))} · prep {_score_text(event.get('preparation_score'))} · follow {_score_text(event.get('follow_through_score'))}</p>
              <p class="tags">{html.escape(', '.join(tags + warnings) or 'no quality warnings')}</p>
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
    pre {{ white-space: pre-wrap; word-break: break-word; background: #17211f; color: #eaf5ef; padding: 14px; border-radius: 8px; max-height: 360px; overflow: auto; }}
    @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} header {{ display:block; }} .session-kpis {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .session-dashboard-grid {{ grid-template-columns:1fr; }} .timeline-ruler, .timeline-lane {{ grid-template-columns: 54px minmax(0, 1fr); }} .timeline-label {{ font-size: 11px; }} .annotation-frame-grid {{ grid-template-columns: 1fr; }} }}
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
