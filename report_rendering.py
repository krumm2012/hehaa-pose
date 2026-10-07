"""Unified report rendering module using Jinja2 templates and dedicated asset files.

Decouples giant HTML/CSS/JS strings from business and pipeline logic while
preserving full DOM, CSS, data contract, and JavaScript function signatures.
"""
from functools import lru_cache
import html
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jinja2

from evaluation_reference_policy import (
    comparison_metric_rows,
    reference_metric_script,
    reference_note,
)
from manual_annotation_contract import annotation_contract_script
EVIDENCE_QUALITY_NOTE = '证据参考为启发式质量，未经准确率校准，不是技术评分。'
from practice_score_adapter import resolve_practice_score
from practice_scoring import POLICY
from report_identity_contract import normalize_report_document

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates" / "reports"


@lru_cache(maxsize=1)
def get_jinja_environment() -> jinja2.Environment:
    """Return cached Jinja2 environment configured for the report templates directory."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )


@lru_cache(maxsize=1)
def get_standalone_styles() -> str:
    """Return raw CSS styles for the standalone report."""
    css_path = TEMPLATES_DIR / "standalone_styles.css"
    with open(css_path, "r", encoding="utf-8") as f:
        return f.read()


@lru_cache(maxsize=1)
def get_standalone_script() -> str:
    """Return full client-side JavaScript for the standalone report, with inlined contract helpers."""
    js_path = TEMPLATES_DIR / "standalone_script.js"
    with open(js_path, "r", encoding="utf-8") as f:
        script = f.read()
    contract_js = annotation_contract_script()
    return script.replace("/* __ANNOTATION_CONTRACT_SCRIPT__ */", contract_js)


@lru_cache(maxsize=1)
def get_live_styles() -> str:
    """Return raw CSS styles for the live realtime session report."""
    css_path = TEMPLATES_DIR / "live_styles.css"
    with open(css_path, "r", encoding="utf-8") as f:
        return f.read()


@lru_cache(maxsize=1)
def get_live_script() -> str:
    """Return full client-side JavaScript for the live report, with inlined contract and metric helpers."""
    js_path = TEMPLATES_DIR / "live_script.js"
    with open(js_path, "r", encoding="utf-8") as f:
        script = f.read()
    contract_js = annotation_contract_script()
    ref_js = reference_metric_script()
    return script.replace(
        "/* __ANNOTATION_CONTRACT_SCRIPT__ */", contract_js
    ).replace(
        "/* __REFERENCE_METRIC_SCRIPT__ */", ref_js
    )


def _safe_rel(target: Optional[str], base: Path) -> str:
    """Resolve relative path safely, replacing Windows backslashes."""
    if not target:
        return ""
    try:
        return os.path.relpath(Path(target), base).replace(os.sep, "/")
    except ValueError:
        return str(target).replace(os.sep, "/")


def render_standalone_report_html(payload: Dict, output_path: str) -> str:
    """Render standalone comprehensive swing analysis report to HTML using Jinja2."""
    from swing_report_builder import (
        _advice_evidence_label,
        _build_event_source_timing_html,
        _build_kinematic_sequence_html,
        _build_radar_svg,
        _classification_text,
        _evidence_quality_label,
        _frame_input_value,
        _impact_freeze_label,
        _is_shadow_swing,
        _json_script,
        _rel,
        _score_text,
        _session_dashboard_html,
        _stroke_options,
        extract_biomechanical_sub_scores,
        ground_reference_html,
    )

    payload = normalize_report_document(payload)
    output_dir = Path(output_path).parent
    video_src = _rel(payload["paths"].get("video"), output_dir)
    event_cards: List[str] = []

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
        practice = resolve_practice_score(event)
        score_100 = practice["score"]
        uncertainty_9 = event.get("score_uncertainty_9")
        scope_label = "教练评分" if practice["method"] == "coach_manual" else "可见动作参考"
        calibrated_score_text = (
            f"{scope_label} {float(score_100):.1f}/100"
            + (f" 启发式范围 ±{float(uncertainty_9)/9*100:.1f}" if uncertainty_9 is not None and practice["method"] != "coach_manual" else "")
            if score_100 is not None else "可见动作评分：证据不足"
        )

        # 1. 综合技术评级与100分制仪表
        is_shadow = _is_shadow_swing(event)
        swing_grade = event.get("swing_grade")
        swing_score = event.get("swing_score")
        if is_shadow:
            head_badge_html = '<span class="tier-pill" style="background:rgba(100,116,139,0.18);color:#94a3b8;border:1px solid rgba(148,163,184,0.3);">空挥试拍 · 无来球</span>'
        elif swing_grade:
            grade_upper = str(swing_grade).upper()
            tier_class = f"tier-{grade_upper.lower()}"
            score_display = f"{float(swing_score):.1f}分" if swing_score is not None else ""
            grade_labels = {code: scope_label + " · " + label for _, code, label in POLICY["grade_bands"]}
            grade_label = grade_labels.get(grade_upper, grade_upper)
            head_badge_html = f'<span class="tier-pill {tier_class}">{html.escape(grade_label)} <strong style="margin-left:4px;">{score_display}</strong></span>'
        else:
            head_badge_html = "<span>可见动作参考分：证据不足</span>"

        meter_pct = f"{float(swing_score):.0f}" if swing_score is not None else _score_text(event.get('overall_score'))

        # 2. 5维生物力学技术雷达图
        sqs = event.get("swing_quality_score") or {}
        sub_scores = sqs.get("sub_scores") if isinstance(sqs, dict) else {}
        if not sub_scores:
            sub_scores = extract_biomechanical_sub_scores(event)
        radar_svg = _build_radar_svg(sub_scores) if sub_scores else ""
        radar_html = f"""
        <div class="bio-radar-wrapper">
          <div class="bio-radar-title">5维生物力学技术雷达 <small style="font-size:10px;color:var(--muted);font-weight:normal;">(诊断参考 · 像面投影)</small></div>
          {radar_svg}
        </div>
        """ if radar_svg else ""

        # 3. 动力学链时序时延条
        seq = event.get("kinematic_sequence") or {}
        kinematic_html = _build_kinematic_sequence_html(seq)
        phase_time_html = _build_event_source_timing_html(event.get('phase_timing'), event.get('candidate_runtime_timing'))

        # 4. 击球遥测指标网格
        rkt = event.get("racket_speed") or {}
        brush = event.get("brush_angle") or {}
        stc = event.get("stance") or {}
        leg = event.get("leg_drive") or {}

        contact_px_s = rkt.get("contact_px_s")
        max_px_s = rkt.get("max_px_s")
        contact_kmh = rkt.get("contact_kmh") if rkt.get("contact_kmh") is not None else rkt.get("contact_speed_kmh")
        max_kmh = rkt.get("max_kmh") if rkt.get("max_kmh") is not None else rkt.get("max_speed_kmh")
        speed_status = rkt.get("status", "uncalibrated")
        brush_angle = brush.get("low_to_high_angle_deg") if brush.get("low_to_high_angle_deg") is not None else brush.get("angle_deg")
        drop_ratio = brush.get("drop_depth_ratio")
        stance_type = stc.get("stance_type") or stc.get("value")
        leg_ratio = leg.get("drive_ratio") if leg.get("drive_ratio") is not None else leg.get("value")

        has_telemetry = any(v is not None for v in [contact_px_s, max_px_s, contact_kmh, max_kmh, brush_angle, drop_ratio, stance_type, leg_ratio])
        telemetry_html = ""
        if has_telemetry:
            if speed_status == "homography_height_debiased":
                speed_label = "球拍物理真速 (km/h · 高度去偏)"
                val_parts = []
                if contact_kmh is not None:
                    val_parts.append(f"{float(contact_kmh):.1f} km/h (触球)")
                if max_kmh is not None:
                    val_parts.append(f"峰值 {float(max_kmh):.1f} km/h")
                kmh_text = " · ".join(val_parts) if val_parts else "已标定"
                if contact_px_s is not None:
                    kmh_text += f' <span style="font-size:10px;color:#059669;font-weight:normal;display:block;margin-top:2px;">⚡ [单应性高度去偏 · 物理真速] (像面 {float(contact_px_s):.0f} px/s)</span>'
                else:
                    kmh_text += ' <span style="font-size:10px;color:#059669;font-weight:normal;display:block;margin-top:2px;">⚡ [单应性高度去偏 · 物理真速]</span>'
            elif speed_status in ("ground_homography_calibrated", "homography_ground_calibrated"):
                speed_label = "球拍物理估速 (km/h · 地面投影)"
                val_parts = []
                if contact_kmh is not None:
                    val_parts.append(f"{float(contact_kmh):.1f} km/h (触球)")
                if max_kmh is not None:
                    val_parts.append(f"峰值 {float(max_kmh):.1f} km/h")
                kmh_text = " · ".join(val_parts) if val_parts else "已标定"
                if contact_px_s is not None:
                    kmh_text += f' <span style="font-size:10px;color:#d97706;font-weight:normal;display:block;margin-top:2px;">⚠️ [地面单应性投影 · 缺失高度去偏] (像面 {float(contact_px_s):.0f} px/s · 建议核验站位高度)</span>'
                else:
                    kmh_text += ' <span style="font-size:10px;color:#d97706;font-weight:normal;display:block;margin-top:2px;">⚠️ [地面单应性投影 · 缺失高度去偏] (建议核验站位高度)</span>'
            else:
                speed_label = "球拍框中心像素速度 (未标定)"
                kmh_text = f"{contact_px_s:.0f} px/s" if contact_px_s is not None else "未观测"
                if max_px_s is not None:
                    kmh_text += f" · 原始峰值 {max_px_s:.0f} px/s"
                kmh_text += ' <span style="font-size:10px;color:#dc2626;font-weight:normal;display:block;margin-top:2px;">⚠️ [缺少机位场地标定矩阵 H · km/h 未标定]</span>'

            if brush_angle is not None:
                brush_text = f"{float(brush_angle):+.1f}°"
            else:
                brush_text = "平击推进 (无下沉提拉)" if (drop_ratio is not None and float(drop_ratio) == 0.0) else "-"
            if drop_ratio is not None and brush_text != "平击推进 (无下沉提拉)":
                try:
                    if float(drop_ratio) == 0.0:
                        brush_text += " · 平击推进 (无下沉提拉)"
                    else:
                        brush_text += f" (上升比 {float(drop_ratio):.2f}x)"
                except (ValueError, TypeError):
                    brush_text += f" (上升比 {drop_ratio})"
            foot_angle = stc.get("image_foot_line_angle_deg", stc.get("value"))
            stance_text = f"{foot_angle}°（像面）" if foot_angle is not None else "未观测"
            if leg_ratio is not None:
                try:
                    stance_text += f" · 髋部上移 {float(leg_ratio):.2f}x"
                except (ValueError, TypeError):
                    stance_text += f" · 髋部上移 {leg_ratio}"
            else:
                stance_text += " · 平立击球 · 无显著下蹲蓄力 (上移极微)"
            telemetry_html = f"""
            <div class="telemetry-grid">
              <div class="telem-item"><span class="telem-label">{html.escape(speed_label)}</span><strong class="telem-val">{kmh_text}</strong></div>
              <div class="telem-item"><span class="telem-label">球拍像面轨迹与上升比</span><strong class="telem-val">{html.escape(brush_text)}</strong></div>
              <div class="telem-item"><span class="telem-label">足部连线倾角与髋部上移比</span><strong class="telem-val">{html.escape(stance_text)}</strong></div>
            </div>
            """

        # 5. 击球定格快照特写查找
        snap_rel = None
        c_frame = event.get("contact_frame")
        candidates = []
        if event.get("impact_freeze_path"):
            candidates.append(Path(event["impact_freeze_path"]))

        search_dirs = [output_dir, output_dir / "snapshots"]
        for s_dir in search_dirs:
            if s_dir.exists() and c_frame is not None:
                candidates.extend(list(s_dir.glob(f"*frame_{c_frame}_impact_freeze.jpg")))

        for cand in candidates:
            if cand.exists() and cand.is_file():
                snap_rel = _rel(str(cand), output_dir)
                break

        snapshot_html = ""
        if snap_rel:
            freeze_label = html.escape(_impact_freeze_label(event))
            snapshot_html = f"""
            <div class="impact-freeze-container">
              <a href="{html.escape(snap_rel)}" target="_blank" class="impact-freeze-link" title="点击查看击球瞬间定格特写">
                <img src="{html.escape(snap_rel)}" alt="击球瞬间定格特写" loading="lazy" class="impact-freeze-img" />
                <span class="impact-freeze-badge">{freeze_label}</span>
              </a>
            </div>
            """

        # 6. 教练纠错建议
        advices = event.get("advice_list") or []
        if isinstance(advices, dict):
            advices = [advices]
        advices_html = ""
        if advices:
            items = []
            for adv in advices:
                msg = adv.get("message") if isinstance(adv, dict) else str(adv)
                code = adv.get("code") if isinstance(adv, dict) else ""
                conf = ' · ' + _advice_evidence_label(adv) if isinstance(adv, dict) else ''
                items.append(
                    f'<li class="coach-advice-item"><span class="advice-bullet">💡</span><strong>{html.escape(str(code))}:</strong> {html.escape(str(msg))}{html.escape(conf)}</li>'
                )
            advices_html = f'<div class="coach-advices-box"><ul class="coach-advice-list">{"".join(items)}</ul></div>'

        event_cards.append(
            f"""
            <article class="event-card{' is-shadow-event' if is_shadow else ''}" data-is-shadow="{'true' if is_shadow else 'false'}" data-annotation-card data-annotation-id="model-{event['event_id']}" data-source-event-id="{event['event_id']}" data-event-id="{event['event_id']}" data-peak-frame="{'' if event.get('peak_frame') is None else event['peak_frame']}">
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
              <p>动作候选：{html.escape(_evidence_quality_label(event.get('confidence')))} · 触球参考 {_score_text(event.get('contact_score'))} · 准备参考 {_score_text(event.get('preparation_score'))} · 随挥参考 {_score_text(event.get('follow_through_score'))}</p>
              <p class="frames">{EVIDENCE_QUALITY_NOTE}</p>
              <p class="tags">{html.escape(', '.join(tags + warnings) or 'no quality warnings')}</p>
              {radar_html}
              {kinematic_html}
              {ground_reference_html(event)}
              {phase_time_html}
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
        <p>{' · '.join(html.escape(label + ' ' + value) for label, value in comparison_metric_rows(evaluation))}</p>
        <p>{html.escape(reference_note(evaluation))}</p>
      </div>
        """

    session_dashboard_block = _session_dashboard_html(
        payload.get("session_quality") or {}
    )

    env = get_jinja_environment()
    template = env.get_template("standalone_report.html.j2")
    return template.render(
        styles=get_standalone_styles(),
        summary_event_count=str(summary.get("swing_event_count", len(payload.get("events", [])))),
        summary_frames=str(summary.get("frames")),
        summary_types=json.dumps(summary.get("swing_event_type_counts", {}), ensure_ascii=False),
        frame_json=payload["paths"].get("frame_json") or "",
        session_dashboard_block=session_dashboard_block,
        video_src=video_src or "",
        evaluation_block=evaluation_block,
        event_cards_html="".join(event_cards),
        payload_json=_json_script(payload),
        script=get_standalone_script(),
    )


def render_live_report_html(document: Dict, manager: Any) -> str:
    """Render live realtime session report to HTML using Jinja2."""
    from realtime_swing_pipeline import (
        _advice_evidence_label,
        _evidence_quality_label,
        _impact_freeze_label,
    )
    from swing_report_builder import (
        _build_event_source_timing_html,
        _build_kinematic_sequence_html,
        _build_radar_svg,
        extract_biomechanical_sub_scores,
    )
    from ground_reference import ground_reference_html
    from osd_evidence import display_value, evidence_label
    from analysis_metric_delivery import scoring_blockers

    document = normalize_report_document(document)
    summary = document.get("summary") or {}
    build = document.get("analysis_build") or {}
    version_html = '<p class="summary">历史分析版本未记录；需生成新版本后比较，原记录保留。</p>'
    if build:
        version_html = (
            '<details><summary>分析版本 · 原记录保留</summary><p>观测资格：'
            + html.escape(str(build.get("observation_policy") or "未记录"))
            + '</p><p>触球测量：保留完整上下文；各指标资格与评分资格分别记录。</p>'
            '<p>新旧版本请在相同源帧上对比；历史报告不会自动升级。</p></details>'
        )
    version_html += '<p class="summary">' + EVIDENCE_QUALITY_NOTE + '</p>'

    session_dashboard = manager._render_live_session_dashboard(document)
    roi = summary.get("roi") or getattr(manager, "roi_metadata", {})
    cal_meta = (
        document.get("session")
        or getattr(manager, "session_metadata", None)
        or {}
    ).get("ground_calibration")
    has_cal = bool(
        cal_meta
        and isinstance(cal_meta, dict)
        and cal_meta.get("views", {}).get("front", {}).get("H")
    )
    cal_banner_html = (
        '<div style="margin-top:10px;padding:9px 12px;border-radius:7px;background:rgba(16,185,129,0.12);border:1px solid #10b981;color:#6ee7b7;font-size:12px;display:flex;align-items:center;gap:8px;">'
        '<span>✅</span><div><strong>机位场地标定有效</strong> · 已加载正面单应性矩阵 H · 支持三维高度去偏真实物理挥速 (km/h)</div></div>'
        if has_cal
        else '<div style="margin-top:10px;padding:9px 12px;border-radius:7px;background:rgba(239,68,68,0.12);border:1px solid #ef4444;color:#fca5a5;font-size:12px;display:flex;align-items:center;gap:8px;">'
        '<span>⚠️</span><div><strong>当前机位未完成场地标定</strong> · 缺少单应性矩阵 H · 球拍挥速仅能提供图像像素参考，请在控制面板「场地标定」工具完成机位四角标定以解锁真实物理挥速 (km/h)</div></div>'
    )

    stream_content = ""
    preview_path = getattr(manager, "preview_path", None)
    output_html = getattr(manager, "output_html", Path("report.html"))
    output_json = getattr(manager, "output_json", Path("events.json"))

    if preview_path is not None and roi.get("enabled"):
        preview_href = os.path.relpath(
            preview_path,
            output_html.parent,
        ).replace(os.sep, "/")
        stream_content = f"""
        <section class="stream-card">
          <div class="stream-heading">
            <div><h2>{html.escape(str(roi.get("label") or "Live camera"))}</h2>
            <p>{html.escape(str(roi.get("source") or ""))}</p></div>
            <strong>ROI ACTIVE</strong>
          </div>
          <img id="roi-preview" src="{html.escape(preview_href)}" alt="Live stream ROI preview">
          <p class="stream-note">实时截图 · 黄色区域为推理 ROI · P1–P4 为配置点</p>
          {cal_banner_html}
        </section>
        """

    cards: List[str] = []
    for event in reversed(document.get("events") or []):
        clip_status = str(event.get("clip_status") or "pending")
        if clip_status in {"ready", "partial"}:
            clip_source = output_json.parent / str(event.get("clip_path") or "")
            clip_href = os.path.relpath(clip_source, output_html.parent).replace(os.sep, "/")
            clip_content = (
                f'<video controls preload="metadata" '
                f'src="{html.escape(clip_href)}"></video>'
            )
            if clip_status == "partial":
                clip_content += (
                    '<p class="clip-state clip-partial">片段缺少 '
                    f'{int(event.get("clip_missing_frame_count") or 0)} 个分析帧</p>'
                )
        elif clip_status == "failed":
            clip_content = (
                '<p class="clip-state clip-failed">Clip encoding failed: '
                f'{html.escape(str(event.get("clip_error") or "unknown error"))}</p>'
            )
        else:
            clip_content = (
                '<p class="clip-state">Encoding this Swing clip in the background…</p>'
            )

        bio = event.get("biomechanics") or {}
        ext = event.get("extended_biomechanics") or bio.get("extended_biomechanics") or {}
        metrics = bio.get("metrics") or {}
        sqs = (
            event.get("swing_quality_score")
            or ext.get("swing_quality_score")
            or metrics.get("swing_quality_score")
            or {}
        )
        is_shadow_swing = bool(
            event.get("is_shadow_swing")
            or (event.get("evidence") or {})
            .get("classification_context", {})
            .get("contact_analysis", {})
            .get("is_shadow_swing")
        )

        practice = resolve_practice_score(event)
        swing_grade, swing_score = practice["grade"], practice["score"]
        if is_shadow_swing:
            head_badge_html = '<span class="tier-pill" style="background:#4b5563;color:#e5e7eb;font-weight:600;">空挥练习 · 无来球</span>'
        elif swing_grade:
            grade_upper = str(swing_grade).upper()
            tier_class = f"tier-{grade_upper.lower()}"
            score_display = f"{float(swing_score):.1f}分" if swing_score is not None else ""
            grade_labels = {
                code: ("教练评分" if practice["method"] == "coach_manual" else "可见动作参考") + " · " + label
                for _, code, label in POLICY["grade_bands"]
            }
            grade_label = grade_labels.get(grade_upper, grade_upper)
            head_badge_html = f'<span class="tier-pill {tier_class}">{html.escape(grade_label)} <strong style="margin-left:4px;">{score_display}</strong></span>'
        else:
            head_badge_html = '<span>可见动作参考分：证据不足</span>'

        snapshot_html = ""
        snap_path = event.get("impact_freeze_path") or (event.get("snapshots") or {}).get("impact_freeze")
        if snap_path:
            try:
                full_p = output_json.parent / snap_path
                if full_p.exists():
                    snap_path = os.path.relpath(full_p, output_html.parent).replace(os.sep, "/")
            except ValueError:
                pass
        if snap_path:
            freeze_label = html.escape(_impact_freeze_label(event))
            snapshot_html = f"""
            <div class="impact-freeze-container">
              <a href="{html.escape(snap_path)}" target="_blank" class="impact-freeze-link" title="点击查看击球瞬间定格特写">
                <img src="{html.escape(snap_path)}" alt="击球瞬间定格特写" loading="lazy" class="impact-freeze-img" />
                <span class="impact-freeze-badge">{freeze_label}</span>
              </a>
            </div>
            """

        coach_advices = event.get("coach_advices") or []
        if not coach_advices and event.get("coach_advice"):
            coach_advices = [event["coach_advice"]]
        schema_ver = str(bio.get("schema_version") or "")
        coach_origin_text = "双视角二维观测参考" if schema_ver == "dual_view_2d_v1" else "单机位2D估计"
        coach_content = ""
        if is_shadow_swing:
            coach_content = (
                '<div class="coach-advice" style="opacity: 0.85;"><div class="coach-title">'
                '<span>实时动作纠错</span><small>空挥练习</small></div>'
                '<p style="margin:6px 0 0 0; color:#9ca3af; font-size:13px;">无来球击打 · 不派发纠错建议</p></div>'
            )
        elif coach_advices:
            advice_rows = []
            for index, advice in enumerate(coach_advices[:3], start=1):
                if not advice.get("message"):
                    continue
                advice_rows.append(
                    '<li>'
                    f'<span>{index}</span>'
                    f'<strong>{html.escape(str(advice["message"]))}</strong>'
                    f'<small>{html.escape(_advice_evidence_label(advice))}</small>'
                    '</li>'
                )
            if advice_rows:
                coach_content = (
                    '<div class="coach-advice"><div class="coach-title">'
                    f'<span>实时动作纠错</span><small>{html.escape(coach_origin_text)}</small></div>'
                    f'<ol>{"".join(advice_rows)}</ol></div>'
                )

        coach_tts = event.get("coach_tts") or {}
        coach_tts_status = str(coach_tts.get("status") or "")
        coach_tts_content = ""
        if coach_tts_status == "ready" and coach_tts.get("audio_path"):
            audio_source = output_json.parent / str(coach_tts["audio_path"])
            audio_href = os.path.relpath(audio_source, output_html.parent).replace(os.sep, "/")
            playback_text = "已通过本机扬声器播报" if coach_tts.get("played") else "可在页面播放"
            timing_detail = f'{int(coach_tts.get("latency_ms") or 0)} ms'
            if coach_tts.get("first_audio_ms"):
                timing_detail = f'首包 {int(coach_tts["first_audio_ms"])} ms · 播完 {timing_detail}'
            coach_tts_content = (
                '<div class="coach-tts"><div><span>本地语音 Coach</span>'
                f'<small>Qwen3-TTS · {timing_detail} · '
                f'{html.escape(playback_text)}</small></div>'
                f'<audio controls preload="none" src="{html.escape(audio_href)}"></audio></div>'
            )
        elif coach_tts_status == "pending":
            coach_tts_content = (
                '<div class="coach-tts pending"><span>本地语音 Coach</span>'
                '<small>Qwen3-TTS 正在合成…</small></div>'
            )
        elif coach_tts_status in {"unavailable", "skipped"}:
            coach_tts_content = (
                '<div class="coach-tts unavailable"><span>本地语音 Coach</span>'
                '<small>文字建议继续生效</small></div>'
            )

        # 5维生物力学技术雷达图
        sub_scores = sqs.get("sub_scores") if isinstance(sqs, dict) and isinstance(sqs.get("sub_scores"), dict) else {}
        if not sub_scores:
            sub_scores = extract_biomechanical_sub_scores(event)
        radar_svg = _build_radar_svg(sub_scores, dark_theme=True) if sub_scores else ""
        radar_html = f"""
        <div class="bio-radar-wrapper">
          <div class="bio-radar-title">5维生物力学技术雷达 <small style="font-size:10px;color:var(--muted);font-weight:normal;">(诊断参考 · 像面投影)</small></div>
          {radar_svg}
        </div>
        """ if radar_svg else ""

        # 动力学链时序传递延时条
        seq = (
            event.get("kinematic_sequence")
            or ext.get("kinematic_sequence")
            or metrics.get("kinematic_sequence")
            or {}
        )
        kinematic_html = _build_kinematic_sequence_html(seq)
        phase_time_html = _build_event_source_timing_html(event.get('phase_timing'), event.get('candidate_runtime_timing'))

        # 击球遥测指标网格
        rkt = (
            event.get("racket_speed")
            or ext.get("racket_head_speed")
            or metrics.get("racket_head_speed")
            or {}
        )
        brush = (
            event.get("brush_angle")
            or ext.get("brush_angle")
            or metrics.get("brush_angle")
            or {}
        )
        stc = (
            event.get("stance")
            or ext.get("stance")
            or metrics.get("stance")
            or {}
        )
        leg = (
            event.get("leg_drive")
            or ext.get("leg_drive")
            or metrics.get("leg_drive")
            or {}
        )
        tb = metrics.get("takeback_depth") or {}
        scap = metrics.get("scapular_retraction") or {}
        sh_turn = metrics.get("shoulder_turn") or {}

        def qualified(section: Dict, field: str) -> Any:
            return display_value(section, field) if section.get('measurement_evidence') else None

        contact_px_s, max_px_s = rkt.get('contact_px_s'), rkt.get('max_px_s')
        contact_kmh = rkt.get('contact_kmh') if rkt.get('contact_kmh') is not None else rkt.get('contact_speed_kmh')
        max_kmh = rkt.get('max_kmh') if rkt.get('max_kmh') is not None else rkt.get('max_speed_kmh')
        speed_status = rkt.get('status', 'uncalibrated')

        if speed_status == 'homography_height_debiased':
            speed_label = '球拍物理真速 (km/h · 高度去偏)'
            val_parts = []
            if contact_kmh is not None:
                val_parts.append(f"{float(contact_kmh):.1f} km/h (触球)")
            if max_kmh is not None:
                val_parts.append(f"峰值 {float(max_kmh):.1f} km/h")
            speed_text = ' · '.join(val_parts) if val_parts else '已标定'
            if contact_px_s is not None:
                speed_text += f' <span style="font-size:10px;color:#34d399;font-weight:normal;display:block;margin-top:2px;">⚡ [单应性高度去偏 · 物理真速] (像面 {float(contact_px_s):.0f} px/s)</span>'
            else:
                speed_text += ' <span style="font-size:10px;color:#34d399;font-weight:normal;display:block;margin-top:2px;">⚡ [单应性高度去偏 · 物理真速]</span>'
        elif speed_status in ('ground_homography_calibrated', 'homography_ground_calibrated'):
            speed_label = '球拍物理估速 (km/h · 地面投影)'
            val_parts = []
            if contact_kmh is not None:
                val_parts.append(f"{float(contact_kmh):.1f} km/h (触球)")
            if max_kmh is not None:
                val_parts.append(f"峰值 {float(max_kmh):.1f} km/h")
            speed_text = ' · '.join(val_parts) if val_parts else '已标定'
            if contact_px_s is not None:
                speed_text += f' <span style="font-size:10px;color:#fbbf24;font-weight:normal;display:block;margin-top:2px;">⚠️ [地面单应性投影 · 缺失高度去偏] (像面 {float(contact_px_s):.0f} px/s · 建议核验站位高度)</span>'
            else:
                speed_text += ' <span style="font-size:10px;color:#fbbf24;font-weight:normal;display:block;margin-top:2px;">⚠️ [地面单应性投影 · 缺失高度去偏] (建议核验站位高度)</span>'
        else:
            speed_label = '球拍框中心像素速度 (未标定)'
            speed_text = f"{float(contact_px_s):.0f} px/s" if contact_px_s is not None else '未观测'
            if max_px_s is not None:
                speed_text += f" · 事件峰值 {float(max_px_s):.0f} px/s"
            speed_text += ' <span style="font-size:10px;color:#f87171;font-weight:normal;display:block;margin-top:2px;">⚠️ [缺少机位场地标定矩阵 H · km/h 未标定]</span>'

        brush_angle = qualified(brush, 'low_to_high_angle_deg')
        drop_ratio = qualified(brush, 'drop_depth_ratio')
        foot_angle = qualified(stc, 'image_foot_line_angle_deg')
        leg_ratio = qualified(leg, 'drive_ratio')
        brush_text = f"{float(brush_angle):+.1f}°" if brush_angle is not None else evidence_label(brush)
        if drop_ratio is not None:
            if float(drop_ratio) == 0.0:
                brush_text += " · 上升比 0.00x (平击推进)"
            else:
                brush_text += f" · 上升比 {float(drop_ratio):.2f}x"
        foot_text = f"{float(foot_angle):.1f}°（像面）" if foot_angle is not None else evidence_label(stc)
        if leg_ratio is not None:
            foot_text += f" · 髋部上移 {float(leg_ratio):.2f}x"
        else:
            l_lbl = evidence_label(leg)
            if l_lbl == "位移先不作解读":
                foot_text += " · 平立击球 · 无显著下蹲蓄力 (上移极微)"
            else:
                foot_text += " · " + l_lbl
        proxy_parts = []
        for metric, label, unit in ((sh_turn,'肩宽角度代理','°'),(tb,'镜面手腕偏移比','x'),(scap,'正背肩宽比','x')):
            if metric.get('value') is not None:
                proxy_parts.append(f"{label} {float(metric['value']):.2f}{unit}")
        telemetry_html = '<div class="telemetry-grid">' + ''.join(
            f'<div class="telem-item"><span class="telem-label">{html.escape(label)}</span><strong class="telem-val">{value}</strong></div>'
            for label, value in [(speed_label, speed_text),
                                ('球拍像面轨迹 / 上升比 · 触球窗口', html.escape(brush_text)),
                                ('足部连线 / 髋部像面上移 · 触球窗口', html.escape(foot_text)),
                                ('双视角投影代理 · 未验证', html.escape(' · '.join(proxy_parts) or '缺观测'))]) + '</div>'

        reason_labels = {
            'automatic_rubric_not_independently_validated': '评分标准未独立验证',
            'contact_not_confirmed': '触球尚未确认',
            'missing_observations': '缺少相关观测',
            'shadow_swing': '空挥',
        }
        blocks = scoring_blockers(event)
        if blocks:
            telemetry_html += '<details><summary>教练五维评审 · 待评定说明 (人工专项)</summary>' + ''.join(
                '<p>' + html.escape(b['label']) + '：' + '；'.join(html.escape(reason_labels.get(r,r)) for r in b['reasons']) + '</p>'
                for b in blocks) + '<small>客观单目 5 维技术雷达已在上方展示；此处为教练人工复核专项评定标准。</small></details>'

        metric_labels = {
            "shoulder_turn": "肩宽角度代理",
            "shoulder_turn_change": "肩部连线角度变化",
            "takeback_depth": "镜面手腕偏移比",
            "scapular_retraction": "正背肩宽比",
            "preparation_knee_flexion": "准备屈膝",
            "arm_extension": "挥拍舒展",
            "contact_lateral_distance": "击球点距离",
        }
        metric_rows = []
        for key, label in metric_labels.items():
            metric = (
                ((event.get("biomechanics") or {}).get("metrics") or {}).get(key)
                or {}
            )
            if (
                metric.get("value") is None
                or metric.get("coach_eligible") is False
            ):
                continue
            raw_unit = str(metric.get("unit") or "")
            if "deg" in raw_unit:
                unit = "°"
            elif raw_unit == "body_width":
                unit = "×身宽"
            elif raw_unit == "ratio":
                unit = "x"
            else:
                unit = raw_unit
            metric_rows.append(
                '<div class="bio-metric">'
                f'<span>{html.escape(label)}</span>'
                f'<strong>{float(metric["value"]):.2f}{unit}</strong>'
                f'<small title="{EVIDENCE_QUALITY_NOTE}">{html.escape(_evidence_quality_label(metric.get("confidence")))}</small>'
                '</div>'
            )
        biomechanics_content = (
            f'<div class="biomechanics">{"".join(metric_rows)}</div>'
            if metric_rows
            else ""
        )
        biomechanics_content += ground_reference_html(event)

        deepseek_advice = event.get("deepseek_advice") or {}
        deepseek_status = str(deepseek_advice.get("status") or "")
        deepseek_content = ""
        if deepseek_status == "ready" and deepseek_advice.get("message"):
            deepseek_content = (
                '<div class="deepseek-advice"><span>DeepSeek旁路</span>'
                f'<strong>{html.escape(str(deepseek_advice["message"]))}</strong>'
                f'<small>{int(deepseek_advice.get("latency_ms") or 0)} ms</small></div>'
            )
        elif deepseek_status == "pending":
            deepseek_content = (
                '<div class="deepseek-advice pending"><span>DeepSeek旁路</span>'
                '<strong>分析中…</strong></div>'
            )
        elif deepseek_status == "skipped":
            deepseek_content = (
                '<div class="deepseek-advice unavailable"><span>DeepSeek旁路</span>'
                '<small>本次已跳过，本地建议继续生效</small></div>'
            )
        elif deepseek_status in {"failed", "unavailable"}:
            deepseek_content = (
                '<div class="deepseek-advice unavailable"><span>DeepSeek旁路</span>'
                '<strong>本地建议已生效</strong></div>'
            )

        warnings = ", ".join((event.get("quality_flags") or {}).get("warnings") or []) or "none"
        start_boundary = (event.get("evidence") or {}).get("start_boundary") or {}
        classification_context = (event.get("evidence") or {}).get("classification_context") or {}
        player_context = classification_context.get("player") or {}
        camera_context = classification_context.get("camera") or {}
        swing_context = classification_context.get("swing") or {}
        hand_text = {"right": "右手", "left": "左手"}.get(
            player_context.get("dominant_hand"),
            "未知",
        )
        camera_text = {
            "facing_player": "球员面向相机",
            "behind_player": "相机位于球员后方",
            "side_or_uncertain": "侧向/不确定",
            "unknown": "未知",
        }.get(camera_context.get("view"), "未知")
        swing_side_text = {
            "forehand": "正手侧",
            "backhand": "反手侧",
            "uncertain": "不确定",
            "unknown": "未知",
        }.get(swing_context.get("side"), "未知")

        coach_calibration = event.get("coach_calibration") or {}
        visible_score = practice["score"]
        visible_uncertainty = event.get("score_uncertainty_9")
        cal_status = coach_calibration.get("status")
        if visible_score is not None:
            calibration_text = (
                f"{float(visible_score):.1f}/100"
                + (
                    f" 启发式范围 ±{float(visible_uncertainty)/9*100:.1f}"
                    if visible_uncertainty is not None and practice["method"] != "coach_manual"
                    else ""
                )
            )
        elif cal_status == "review_required":
            calibration_text = "待教练复核 (专项规则待标定)"
        else:
            calibration_text = "证据不足 (单目视觉待标定)"

        boundary_text = " · ".join(
            str(value)
            for value in (
                start_boundary.get("mode"),
                start_boundary.get("confidence"),
            )
            if value
        ) or "legacy"

        cards.append(
            f"""
            <article class="event-card" data-annotation-card data-annotation-id="model-{event['event_id']}" data-source-event-id="{event['event_id']}" data-predicted-stroke-type="{html.escape(str(event.get('stroke_type') or 'Unknown'))}" data-peak-frame="{html.escape(str('' if event.get('peak_frame') is None else event.get('peak_frame')))}">
              <div class="event-heading">
                <h2>Swing #{event['event_id']} · {html.escape(str(event.get('stroke_type') or 'Unknown'))}</h2>
                {head_badge_html}
              </div>
              {snapshot_html}
              {clip_content}
              {coach_content}
              {coach_tts_content}
              {radar_html}
              {kinematic_html}
              {phase_time_html}
              {telemetry_html}
              {biomechanics_content}
              {deepseek_content}
              <dl>
                <div><dt>Frames</dt><dd>{int(event['start_frame'])}–{int(event['end_frame'])}</dd></div>
                <div><dt>Contact</dt><dd>{html.escape(str(event.get('contact_frame', '-')))}</dd></div>
                <div><dt>Peak</dt><dd>{html.escape(str(event.get('peak_frame', '-')))}</dd></div>
                <div><dt>Start boundary</dt><dd>{html.escape(boundary_text)}</dd></div>
                <div><dt>球员 / 机位 / 挥拍侧</dt><dd>{html.escape(hand_text)} · {html.escape(camera_text)} · {html.escape(swing_side_text)}</dd></div>
                <div><dt>可见动作校准</dt><dd>{html.escape(calibration_text)}</dd></div>
                <div><dt>Warnings</dt><dd>{html.escape(warnings)}</dd></div>
              </dl>
              <div class="annotation-box">
                <div class="annotation-frames">
                  <label>人工开始帧<input type="number" min="0" step="1" data-field="start_frame" value="{int(event['start_frame'])}"></label>
                  <label>人工触球帧<input type="number" min="0" step="1" data-field="contact_frame" value="{html.escape(str('' if event.get('contact_frame') is None else event.get('contact_frame')))}"></label>
                  <label>人工结束帧<input type="number" min="0" step="1" data-field="end_frame" value="{int(event['end_frame'])}"></label>
                </div>
                <label>人工类型
                  <select data-field="actual_stroke_type">
                    <option value="Forehand"{' selected' if event.get('stroke_type') == 'Forehand' else ''}>Forehand</option>
                    <option value="Backhand"{' selected' if event.get('stroke_type') == 'Backhand' else ''}>Backhand</option>
                    <option value="Two-Handed Backhand"{' selected' if event.get('stroke_type') == 'Two-Handed Backhand' else ''}>Two-Handed Backhand</option>
                    <option value="Serve"{' selected' if event.get('stroke_type') == 'Serve' else ''}>Serve</option>
                    <option value="Volley"{' selected' if event.get('stroke_type') == 'Volley' else ''}>Volley</option>
                    <option value="Unclear"{' selected' if event.get('stroke_type') not in {'Forehand', 'Backhand', 'Two-Handed Backhand', 'Serve', 'Volley'} else ''}>Unclear</option>
                  </select>
                </label>
                <div class="annotation-checks">
                  <label><input type="checkbox" data-field="valid_hit" checked> 有效击球</label>
                  <label><input type="checkbox" data-field="count_correct" checked> 计数正确</label>
                  <label><input type="checkbox" data-field="needs_review" checked> 待人工确认（确认后取消）</label>
                  <label><input type="checkbox" data-tag="wrong_type"> 类型错误</label>
                  <label><input type="checkbox" data-tag="contact_timing"> 触球帧偏差</label>
                  <label><input type="checkbox" data-tag="event_boundary"> 边界偏差</label>
                </div>
                <label>备注<textarea rows="2" data-field="note"></textarea></label>
              </div>
            </article>
            """
        )

    content = "\n".join(cards) if cards else '<p class="waiting">Waiting for the first completed Swing event…</p>'
    event_json_href = os.path.relpath(
        output_json,
        output_html.parent,
    ).replace(os.sep, "/")

    env = get_jinja_environment()
    template = env.get_template("live_report.html.j2")
    return template.render(
        styles=get_live_styles(),
        version_html=version_html,
        summary_event_count=int(summary.get("swing_event_count") or 0),
        summary_latest_frame=int(summary.get("latest_frame") or -1),
        session_dashboard=session_dashboard,
        stream_content=stream_content,
        cards_content=content,
        event_json_href=event_json_href,
        model_event_ids=[event["event_id"] for event in document.get("events", [])],
        script=get_live_script(),
    )
