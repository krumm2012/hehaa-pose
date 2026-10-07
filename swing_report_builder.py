#!/usr/bin/env python3
"""Build a standalone HTML report for swing video and JSON review."""

from __future__ import annotations

import argparse
from manual_annotation_contract import annotation_contract_script
from evaluation_reference_policy import comparison_metric_rows, reference_note
from ground_reference import ground_reference_html
import html
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

from practice_scoring import POLICY, number
from practice_score_adapter import resolve_practice_score
from swing_session_quality import build_session_quality_dashboard
from event_source_timing import analyze_event_source_timing, source_frame_navigation, POLICY_VERSION as PHASE_TIME_POLICY
from observation_policy import finite_number
from report_identity_contract import normalize_report_document, report_identity_info, verify_source_session_binding
from swing_biomechanics import extract_biomechanical_sub_scores

EVIDENCE_QUALITY_NOTE = '证据参考为启发式质量，未经准确率校准，不是技术评分。'


def _evidence_quality_label(value):
    score = finite_number(value)
    text = f'{score:.0%}' if score is not None and 0 <= score <= 1 else '未提供'
    return f'证据参考 {text} · 未校准'


def _advice_evidence_label(advice):
    code = advice.get('code')
    if code == 'stroke_specific_rubric_unvalidated':
        return '规则待标定'
    if code == 'insufficient_technique_evidence':
        return '待教练标定'
    return '复核提示' if advice.get('category') == 'review' else _evidence_quality_label(advice.get('confidence'))

RADAR_AXES = [
    ("shoulder_turn", "转肩"),
    ("takeback", "引拍"),
    ("arm_extension", "延展"),
    ("racket_speed", "挥速"),
    ("leg_drive", "蹬地"),
]


def _build_event_source_timing_html(timing: Dict, runtime=None) -> str:
    timing = timing if isinstance(timing, dict) else {}
    seconds = timing.get('duration_seconds')
    qualified = (timing.get('policy_version') == PHASE_TIME_POLICY
                 and timing.get('status') == 'reported_media_time'
                 and timing.get('basis') == 'media_pts'
                 and type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0)
    elapsed = f'{seconds:.3f} s' if qualified else '源时间不可核验'
    notice = ('<p>源视频结束时尚未完成等待确认；请复核动作是否完整。</p>'
              if isinstance(runtime, dict) and runtime.get('completion_status') == 'source_end_unsettled_candidate' else '')
    reasons = list(timing.get('reasons') or []) + list(timing.get('phase_reasons') or [])
    explanations = {'missing_event_anchor_record': '缺少事件锚点源帧；保持原指定帧号，不借邻帧计算',
                    'source_time_contract_missing': '缺少源时间记录',
                    'source_frame_gap_cannot_bridge_phase_labels': '源帧存在缺口，不能跨缺口累计阶段标签',
                    'duplicate_frame_identity': '源帧身份重复，观测无法唯一对应'}
    details = list(dict.fromkeys(explanations[reason] for reason in reasons if reason in explanations))
    if details:
        notice += '<p>' + html.escape('；'.join(details)) + '</p>'
    labels = {'backswing': '引拍', 'forward_swing': '前挥', 'contact_candidate': '触球候选',
              'follow_through': '随挥', 'recovery': '恢复', 'ready': '准备'}
    phases = ((timing.get('phase_durations_seconds') or {})
              if timing.get('phase_status') == 'model_label_media_time_support' else {})
    support = ' · '.join(f'{html.escape(labels.get(key, str(key)))} {value:.3f} s'
        for key, value in phases.items()
        if type(value) in (int, float) and math.isfinite(value) and value >= 0) if qualified else ''
    return (f'<div class="phase-time" aria-label="候选事件媒体时长"><strong>候选事件媒体时长：{elapsed}</strong>'
            f'<p>模型阶段标签覆盖：{support or "缺少连续标签或观测"}</p>'
            f'{notice}'
            '<small>按输入媒体 PTS 统计；曝光未核验，模型阶段边界需复核，不用于技术评分。</small></div>')


def _impact_freeze_label(event):
    fid = event.get('impact_freeze_source_frame_id')
    if type(fid) is not int:
        return '历史定格图片（源帧未核验）'
    if fid != event.get('contact_frame'):
        return f'临近候选定格（源帧 {fid}；触球候选 {event.get("contact_frame")}）'
    return f'触球候选定格（源帧 {fid}）'


def _build_kinematic_sequence_html(sequence: Dict) -> str:
    """Show peak intervals with their evidence qualification, never as compute time."""
    if not isinstance(sequence, dict) or not sequence:
        return ""
    details = sequence.get("details")
    details = details if isinstance(details, dict) else sequence
    confidence = number(sequence.get("confidence", details.get("confidence"))) or 0.0
    eligible = sequence.get("coach_eligible", details.get("coach_eligible")) is True
    qualified = eligible and confidence > 0
    quality = sequence.get("value") or details.get("sequence_quality")
    status = str(quality) if qualified and quality else "未验证 · 需复核"
    badge_class = str(quality).lower() if qualified and quality else "unvalidated"
    cross = details.get("cross_validation") or {}
    cross_status = cross.get("status")
    labels = {
        "agree": "双视角一致 · 未验证",
        "disagree": "双视角冲突 · 需复核",
        "single_view": "单视角参考 · 未验证",
        "unavailable": "证据不足 · 需复核",
        "legacy_single_view": "历史单视角 · 未验证",
    }
    if not qualified and cross_status in labels:
        status = labels[cross_status]
        if cross_status == "unavailable" and cross.get("reason") == "cadence_sensitive_peak":
            status = "短间隔敏感 · 暂停判定"
            badge_class = "cadence-sensitive"
    view_rows = []
    audit = details.get('cadence_audit') or {}
    if audit:
        frames = audit.get('short_interval_source_frames', [])
        view_rows.append('<div>时间间隔审计：短间隔 ' + html.escape(str(len(frames)))
                         + ' 个；源帧 ' + html.escape(str(frames))
                         + '。保留原始 PTS；排除短间隔仅用于峰值敏感性检查，非曝光校准。</div>')
    for view, evidence in (details.get("views") or {}).items():
        label = {"front": "正面", "back": "背面"}.get(view, str(view))
        interval = number(evidence.get("latency_hip_to_shoulder_ms"))
        if evidence.get("status") == "usable" and interval is not None:
            segments = list((evidence.get("segments") or {}).values())
            coverage = min(number(s.get("coverage")) or 0 for s in segments) if segments else 0
            kp_quality = min(number(s.get("keypoint_quality")) or 0 for s in segments) if segments else 0
            text = (f"{label}髋—肩：{interval:.1f} ms · 有效覆盖 {coverage:.0%}"
                    f" · 关键点质量 {kp_quality:.2f}/1")
        else:
            reasons = {'boundary_peak':'峰值位于窗口边界', 'ambiguous_peak':'峰值过宽或多峰',
                       'discontinuous_evidence':'有效片段不连续', 'low_coverage':'有效覆盖不足',
                       'insufficient_samples':'有效样本不足', 'insufficient_motion':'未形成明确运动峰值',
                       'usable':'峰值可用', 'cadence_sensitive_peak':'峰值对短时间间隔敏感，暂停判定'}
            parts = [f"{name}：{reasons.get((evidence.get('segments') or {}).get(key,{}).get('status'),'证据不足')}"
                     for key,name in (("hip","髋"),("shoulder","肩"))]
            text = f"{label}：" + "；".join(parts)
        view_rows.append(f"<div>{html.escape(text)}</div>")
        rejection_labels = {'joint_missing':'关节点缺失', 'joint_source_frame_mismatch':'关节点来源帧不匹配',
                            'joint_not_raw_observation':'非原始观测', 'joint_score_unavailable':'关节点分数不可用',
                            'invalid_joint_numeric':'关节点数值无效', 'low_joint_score':'关节点分数不足',
                            'projected_line_too_short':'肩髋像面线过短'}
        for key, name in (('hip', '髋'), ('shoulder', '肩')):
            segment = (evidence.get('segments') or {}).get(key, {})
            if 'observation_total_frames' in segment:
                items = [f"{rejection_labels.get(reason, reason)} {count} 帧"
                         for reason, count in segment.get('observation_rejections', {}).items()]
                detail = (f"{label}{name}原始观测：{segment.get('observation_valid_frames', 0)} / "
                          f"{segment['observation_total_frames']} 帧；" + ('；'.join(items) or '无观测拒绝'))
                view_rows.append('<div>' + html.escape(detail) + '</div>')
    evidence_quality = number(details.get("evidence_confidence"))
    if evidence_quality is not None:
        view_rows.append(f"<div>交叉验证证据质量：{evidence_quality:.2f}/1（启发式，非准确率）</div>")
    resolution = number(details.get("sampling_interval_ms"))
    time_labels = {
        "media_pts": "输入视频媒体时间（尚未核验传感器曝光）",
        "nominal_fps": "帧率推算时间（估计）",
        "legacy_frame_timestamps": "历史帧时间（来源未核验）",
        "unavailable": "源媒体时间不可用",
    }
    if details.get("time_basis") in time_labels:
        view_rows.append(f"<div>时间来源：{time_labels[details['time_basis']]}</div>")
    time_reasons = {
        "cadence_sensitive_peak": "排除异常短间隔后峰值不稳定，暂停综合时序判定；原始候选保留供复核。",
        "source_media_time_unavailable": "无法获取源媒体时间，暂停峰值间隔估计。",
        "duplicate_or_discontinuous_source_time": "源时间重复或倒退，暂停峰值间隔估计。",
        "mixed_source_time_bases": "时间来源混用，暂停峰值间隔估计。",
        "incomplete_source_time_contract": "源时间记录不完整，暂停峰值间隔估计。",
        "invalid_source_timestamps": "源时间无效，暂停峰值间隔估计。",
    }
    if cross.get("reason") in time_reasons:
        view_rows.append(f"<div>{time_reasons[cross['reason']]}</div>")
    if resolution is not None:
        view_rows.append(f"<div>采样间隔：{resolution:.1f} ms；小于峰值定位范围的先后差异不作结论。</div>")
    if cross_status == "disagree":
        view_rows.append("<div>两视角峰值时间不一致，已暂停合并估计。</div>")
    for pair_key, pair_label in (("hip_to_shoulder", "髋—肩"), ("shoulder_to_racket", "肩—拍")):
        pair = (details.get("pair_timing") or {}).get(pair_key) or {}
        bounds = pair.get("latency_range_ms")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            low, high = (number(value) for value in bounds)
            if low is not None and high is not None and low <= high:
                conclusion = "先后难以分辨" if low <= 0 <= high else "仅表示二维投影先后"
                view_rows.append(
                    f"<div>{pair_label}时差范围：{low:+.1f} ～ {high:+.1f} ms · {conclusion}"
                    "（采样与峰宽范围，非统计置信区间）</div>"
                )
    hip_dt = number(details.get("latency_hip_to_shoulder_ms"))
    uncertainty = number(details.get("peak_time_uncertainty_ms"))
    if quality == "UNRESOLVED_AT_FRAME_RATE" or (hip_dt is not None and uncertainty is not None and abs(hip_dt) <= uncertainty):
        view_rows.append("<div>峰值间隔接近时间分辨率，难以分辨先后。</div>")
    cand_frame = details.get("racket_candidate_peak_frame")
    cand_speed = details.get("racket_candidate_peak_speed")
    cand_lat = details.get("candidate_latency_shoulder_to_racket_ms")
    cand_hip_lat = details.get("candidate_latency_hip_to_shoulder_ms")
    cand_hip_frame = details.get("candidate_hip_peak_frame")
    if details.get("racket_peak_frame") is None and (hip_dt is not None or cand_lat is not None):
        rkt_status = (details.get('racket_evidence') or {}).get('status')
        racket_reason = {'low_coverage':'有效覆盖不足', 'boundary_peak':'峰值位于窗口边界',
                         'cadence_sensitive_peak':'峰值对短时间间隔敏感',
                         'ambiguous_peak':'峰值过宽或多峰', 'discontinuous_evidence':'有效片段不连续',
                         'insufficient_samples':'有效样本不足', 'insufficient_motion':'未形成明确运动峰值',
                         'usable':'已形成明确运动峰值'}.get(rkt_status, '证据不足')
        if cand_frame is not None:
            cand_spd_txt = f"{cand_speed:.1f} px/s" if cand_speed is not None else "—"
            cand_lat_txt = f"{cand_lat:+.1f} ms" if cand_lat is not None else "—"
            if rkt_status == 'usable':
                view_rows.append(
                    f"<div>球拍判定已形成明确运动峰值（拍头峰值有效），末端闭合时序受躯干时序审计影响暂缓闭合，透出<strong>候选拍峰（诊断参考）："
                    f"第 {cand_frame} 帧 · 速度 {cand_spd_txt} · 候选肩—拍时差 {cand_lat_txt}</strong>。</div>"
                )
            else:
                view_rows.append(
                    f"<div>球拍判定虽未完全闭合（{racket_reason}），但已检出<strong>候选拍峰（诊断参考）："
                    f"第 {cand_frame} 帧 · 速度 {cand_spd_txt} · 候选肩—拍时差 {cand_lat_txt}</strong>。</div>"
                )
        else:
            view_rows.append(f"<div>球拍峰值缺失（{racket_reason}），仅有髋肩投影时序；不能判断完整动力链。</div>")
    rows = []
    for label, key, scale, extra_class, cand_val, cand_lbl in [
        ("髋—肩峰值间隔", "latency_hip_to_shoulder_ms", 80.0, "", cand_hip_lat, f"候选F{cand_hip_frame or '—'}"),
        ("肩—拍峰值间隔", "latency_shoulder_to_racket_ms", 90.0, "k-fill-rkt", cand_lat, f"候选F{cand_frame or '—'}" + (f" @ {cand_speed:.0f}px/s" if cand_speed is not None else "")),
    ]:
        value = number(details.get(key))
        if value is not None:
            text = f"{value:.1f} ms"
            width = min(100.0, max(5.0, value / scale * 100.0))
            rows.append(
                '<div class="kinematic-bar-row">'
                f'<span class="k-label">{label}:</span><span class="k-val">{text}</span>'
                f'<div class="k-track"><div class="k-fill {extra_class}" '
                f'style="width:{width:.0f}%;"></div></div></div>'
            )
        elif cand_val is not None:
            text = f"{cand_val:+.1f} ms ({cand_lbl})"
            width = min(100.0, max(5.0, abs(cand_val) / scale * 100.0))
            rows.append(
                '<div class="kinematic-bar-row">'
                f'<span class="k-label">{label}:</span><span class="k-val" style="color:var(--blue, #38bdf8);">{text}</span>'
                f'<div class="k-track"><div class="k-fill {extra_class}" '
                f'style="width:{width:.0f}%; opacity: 0.7; border: 1px dashed var(--blue, #38bdf8);"></div></div></div>'
            )
        else:
            text = "未观测"
            rows.append(
                '<div class="kinematic-bar-row">'
                f'<span class="k-label">{label}:</span><span class="k-val">{text}</span>'
                f'<div class="k-track"><div class="k-fill {extra_class}" '
                f'style="width:0%;"></div></div></div>'
            )
    return (
        '<div class="kinematic-box"><div class="kinematic-header">'
        '<span>二维峰值间隔：<strong>髋/骨盆 ➔ 肩/躯干 ➔ 球拍检测点</strong></span>'
        f'<span class="seq-badge seq-{html.escape(badge_class)}">{html.escape(status)}</span>'
        '</div><p>表示动作峰值之间的时间差，不代表处理耗时；'
        '负值表示后项峰值更早。未验证结果不用于技术纠错。</p>'
        f'<div class="kinematic-evidence">{"".join(view_rows)}</div>'
        f'<div class="kinematic-bars">{"".join(rows)}</div></div>'
    )


def _is_shadow_swing(ev: Dict) -> bool:
    if ev.get("is_shadow_swing") is not None:
        return bool(ev["is_shadow_swing"])
    evidence = ev.get("evidence") or {}
    ca = evidence.get("contact_analysis") or (evidence.get("classification_context") or {}).get("contact_analysis") or {}
    if ca.get("is_shadow_swing") is not None:
        return bool(ca["is_shadow_swing"])
    if ca.get("has_ball") is False:
        return True
    return False


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
    document = normalize_report_document({'events': events}, 'Coach关联')
    return {event['event_id']: event for event in document['events']}


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
    evaluation_json_path = evaluation_json_path or default_evaluation_json(frame_json_path)

    frame_data = load_json(frame_json_path)
    declared_source = ((frame_data.get('video_info') or {}).get('path')
                       or (frame_data.get('session') or {}).get('source'))
    video_path = video_path or (declared_source if declared_source and Path(declared_source).is_file()
                               else default_video_path(frame_json_path))
    navigation = source_frame_navigation(frame_data.get('frames') or [])
    if not declared_source or Path(video_path).resolve() != Path(declared_source).resolve():
        navigation.update(status='unavailable', frames=[], reasons=['selected_video_not_bound_to_source'])
    navigation['video_binding'] = 'declared_source_path_only_not_independent_content_verification'
    event_data = normalize_report_document(load_json(event_json_path))
    coach_data = normalize_report_document(
        load_json(coach_json_path) if coach_json_path and os.path.exists(coach_json_path)
        else {"events": []}, 'Coach报告输入')
    source_binding_verified = False
    if coach_data.get('events'):
        source_binding_verified = verify_source_session_binding(event_data, coach_data)
    evaluation_data = (
        load_json(evaluation_json_path)
        if evaluation_json_path and os.path.exists(evaluation_json_path)
        else None
    )
    coach_lookup = {event['event_id']: event for event in coach_data['events']}

    merged_events = []
    for event in event_data.get("events", []):
        coach_event = coach_lookup.get(event['event_id'], {})
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
            ext.get("brush_angle")
            or metrics.get("brush_angle")
            or event.get("brush_angle")
            or {}
        )
        stc = (
            ext.get("stance")
            or metrics.get("stance")
            or event.get("stance")
            or {}
        )
        leg = (
            ext.get("leg_drive")
            or metrics.get("leg_drive")
            or event.get("leg_drive")
            or {}
        )
        advices = event.get('coach_advices') or event.get("coach_advice") or coach_event.get("coach_advice") or []
        if isinstance(advices, dict):
            advices = [advices]
        freeze_path = event.get("impact_freeze_path") or (event.get("snapshots") or {}).get("impact_freeze")
        if freeze_path and not Path(freeze_path).is_absolute():
            freeze_path = str(Path(event_json_path).parent / freeze_path)

        practice = resolve_practice_score({**event, "biomechanics": bio})
        calibration = practice["calibration"]
        raw_score, raw_grade = practice["score"], practice["grade"]
        sub_scores = (
            (ext.get("swing_quality_score") or {}).get("sub_scores")
            or (bio.get("metrics", {}).get("swing_quality_score", {}) or {}).get("sub_scores")
            or extract_biomechanical_sub_scores(event)
        )
        sqs = {"overall_score": raw_score, "grade": raw_grade, "sub_scores": sub_scores}

        merged_events.append(
            {
                "event_id": event.get("event_id"),
                "stroke_type": event.get("stroke_type"),
                "confidence": event.get("confidence"),
                "start_frame": event.get("start_frame"),
                "contact_frame": event.get("contact_frame"),
                "peak_frame": event.get("peak_frame"),
                "end_frame": event.get("end_frame"),
                "phase_counts": event.get("phase_counts") or {},
                "candidate_runtime_timing": event.get('candidate_runtime_timing'),
                "phase_timing": analyze_event_source_timing(event, frame_data.get('frames') or [],
                                                            event_data.get('frame_trace') or []),
                "start_boundary": start_boundary,
                "classification_context": classification_context,
                "quality_flags": quality_flags,
                "diagnosis_tags": coach_event.get("diagnosis_tags") or list(quality_flags.get("warnings") or []),
                "overall_score": calibration["visible_technique_score"],
                "practice_score": practice,
                "practice_context": event.get("practice_context") or {},
                "practice_review": event.get("practice_review"),
                "overall_score_9": calibration["visible_technique_score_9"],
                "score_uncertainty_9": calibration["uncertainty_9"],
                "score_confidence": calibration["confidence"],
                "coach_calibration": calibration,
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
                "ground_reference": event.get('ground_reference'),
                "racket_speed": rkt,
                "brush_angle": brush,
                "stance": stc,
                "leg_drive": leg,
                "advice_list": advices,
                "is_shadow_swing": _is_shadow_swing(event),
                "impact_freeze_path": freeze_path,
                "impact_freeze_source_frame_id": event.get('impact_freeze_source_frame_id'),
            }
        )

    return {
        'report_identity': report_identity_info(source_binding_verified=source_binding_verified),
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
            "source_time_navigation": navigation,
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
        "incomparable_series": "训练条件不同",
    }
    grade_labels = {
        "good": "良好",
        "watch": "留意",
        "poor": "偏低",
        "unavailable": "无数据",
        "event_integrity_confounded": "受事件重叠影响",
        "incomparable_series": "训练条件不同",
    }
    kpis = [
        (
            "证据质量",
            _dashboard_number(quality.get("evidence_quality_score_100"), 0, "/100"),
            grade_labels.get(quality.get("grade"), str(quality.get("grade") or "-")),
        ),
        (
            "练习评分均值",
            _dashboard_number(quality.get("practice_score_mean_100"), 1, "/100"),
            (
                "启发式范围 ±" + _dashboard_number(quality["median_uncertainty_100"], 1)
                if quality.get("median_uncertainty_100") is not None
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
        score = point.get("practice_score_100")
        evidence = point.get("evidence_quality_100")
        score_width = max(0.0, min(100.0, float(score or 0.0)))
        evidence_width = max(0.0, min(100.0, float(evidence or 0.0)))
        warnings = len(point.get("warnings") or [])
        series_rows.append(
            '<div class="session-series-row">'
            f'<strong>#{int(point.get("event_id") or 0)}</strong>'
            '<div class="session-bars">'
            f'<i class="score-bar" style="width:{score_width:.1f}%"></i>'
            f'<i class="quality-bar" style="width:{evidence_width:.1f}%"></i>'
            '</div>'
            f'<span>{_dashboard_number(score, 1, "/100")}</span>'
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

    fc = dashboard.get("fatigue_and_consistency") or {}
    fc_html = ""
    if fc.get("fatigue_status") and fc.get("fatigue_status") != "WARMING_UP":
        status_map = {
            "CONSISTENT": ("动作节奏稳定", "#10b981", "rgba(16,185,129,0.12)"),
            "FATIGUE_OBSERVED": ("体能疲劳显现", "#ef4444", "rgba(239,68,68,0.12)"),
            "WARMED_UP": ("状态逐步提升", "#3b82f6", "rgba(59,130,246,0.12)"),
        }
        title, color, bg = status_map.get(fc.get("fatigue_status"), ("监测中", "#94a3b8", "rgba(148,163,184,0.12)"))
        decay = fc.get("speed_decay_percent")
        decay_text = f"{decay:+.1f}%" if decay is not None else "-"
        speed_std = fc.get("speed_std")
        speed_std_text = f"±{speed_std:.1f}" if speed_std is not None else "-"
        jitter = fc.get("latency_jitter_std_ms")
        jitter_text = f"±{jitter:.1f}ms" if jitter is not None else "-"

        fc_html = f"""
      <div class="fatigue-consistency-card" style="margin-top:12px;padding:10px 14px;background:{bg};border:1px solid {color};border-radius:8px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:18px;">⚡</span>
          <div>
            <strong style="color:{color};font-size:13px;display:block;">体能疲劳与一致性：{html.escape(title)}</strong>
            <small style="color:#94a3b8;font-size:11px;">挥速衰减趋势与动力链时序抖动</small>
          </div>
        </div>
        <div style="display:flex;gap:14px;font-size:12px;">
          <div><span style="color:#94a3b8;">挥速衰减:</span> <strong style="color:{color};">{html.escape(decay_text)}</strong></div>
          <div><span style="color:#94a3b8;">挥速标准差:</span> <strong style="color:#e2e8f0;">{html.escape(speed_std_text)}</strong></div>
          <div><span style="color:#94a3b8;">动力链抖动:</span> <strong style="color:#e2e8f0;">{html.escape(jitter_text)}</strong></div>
        </div>
      </div>
      """

    return f"""
    <section class="panel session-dashboard" aria-label="会话质量与漂移看板">
      <div class="session-dashboard-head">
        <div><h2>会话质量与漂移</h2><p>{html.escape(readiness)}</p></div>
        <strong data-state="{html.escape(state)}">{html.escape(status_labels.get(state, state))}</strong>
      </div>
      <div class="session-kpis">{kpi_html}</div>
      {fc_html}
      <div class="session-dashboard-grid">
        <div><h3>逐拍趋势</h3><div class="session-series">{''.join(series_rows) or '<p>等待挥拍事件。</p>'}</div></div>
        <div><h3>前段 → 最近</h3><div class="drift-rows">{''.join(indicator_rows)}</div></div>
      </div>
      {alert_html}
      <div class="session-legend"><span><i class="score-dot"></i>可见动作分</span><span><i class="quality-dot"></i>证据质量</span></div>
    </section>
    """


def render_report_html(payload: Dict, output_path: str) -> str:
    """Render standalone comprehensive swing analysis report to HTML.

    Delegates to report_rendering.render_standalone_report_html for template-driven rendering.
    """
    from report_rendering import render_standalone_report_html
    return render_standalone_report_html(payload, output_path)


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
