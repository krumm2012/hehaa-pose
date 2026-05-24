#!/usr/bin/env python3
"""Build a standalone HTML report for swing video and JSON review."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


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
                "quality_flags": quality_flags,
                "diagnosis_tags": coach_event.get("diagnosis_tags") or list(quality_flags.get("warnings") or []),
                "overall_score": scores.get("overall_score"),
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
        "events": merged_events,
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


def render_report_html(payload: Dict, output_path: str) -> str:
    output_dir = Path(output_path).parent
    video_src = _rel(payload["paths"].get("video"), output_dir)
    event_cards = []
    for event in payload.get("events", []):
        warnings = event.get("quality_flags", {}).get("warnings") or []
        tags = event.get("diagnosis_tags") or []
        event_cards.append(
            f"""
            <article class="event-card" data-event-id="{html.escape(str(event.get('event_id')))}">
              <div class="event-head">
                <strong>Event {html.escape(str(event.get('event_id')))} · {html.escape(str(event.get('stroke_type')))}</strong>
                <span>score {_score_text(event.get('overall_score'))}</span>
              </div>
              <div class="frames">start {event.get('start_frame')} · contact {event.get('contact_frame')} · peak {event.get('peak_frame')} · end {event.get('end_frame')}</div>
              <div class="meter"><i style="width:{_score_text(event.get('overall_score'))}%"></i></div>
              <p>confidence {_score_text(event.get('confidence'))} · contact {_score_text(event.get('contact_score'))} · prep {_score_text(event.get('preparation_score'))} · follow {_score_text(event.get('follow_through_score'))}</p>
              <p class="tags">{html.escape(', '.join(tags + warnings) or 'no quality warnings')}</p>
              <div class="annotation-box">
                <label>人工类型
                  <select class="annotation-stroke" data-field="actual_stroke_type">
                    {_stroke_options(event.get('stroke_type'))}
                  </select>
                </label>
                <label><input class="annotation-count-correct" type="checkbox" data-field="count_correct" checked> 计数正确</label>
                <label><input class="annotation-valid-hit" type="checkbox" data-field="valid_hit" checked> 有效击球</label>
                <label><input class="annotation-needs-review" type="checkbox" data-field="needs_review" {"checked" if (event.get('quality_flags') or {}).get('review_recommended') else ""}> 需要复核</label>
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
        evaluation_block = f"""
      <div class="evaluation-summary">
        <h2>Evaluation Summary</h2>
        <p>stroke accuracy {_percent_text(evaluation_summary.get('stroke_type_accuracy'))} · contact accuracy {_percent_text(evaluation_summary.get('contact_accuracy'))} · manual review {html.escape(str(evaluation_summary.get('manual_review_count', 0)))}</p>
      </div>
        """
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
    .events {{ display: grid; gap: 12px; max-height: 70vh; overflow: auto; }}
    .event-card {{ padding: 14px; }}
    .event-head {{ display: flex; justify-content: space-between; gap: 10px; }}
    .frames, .tags, p {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
    .tags {{ color: var(--warn); }}
    .annotation-box {{ display: grid; gap: 8px; border-top: 1px solid var(--line); margin-top: 12px; padding-top: 12px; }}
    .annotation-box label {{ color: var(--ink); font-size: 13px; }}
    .annotation-box select, .annotation-box textarea {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 7px; background: #fffdf7; color: var(--ink); }}
    .annotation-tags {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }}
    .actions {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }}
    .evaluation-summary {{ margin: 14px 0; padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: #fffdf7; }}
    .evaluation-summary h2 {{ margin: 0 0 6px; font-size: 18px; }}
    .evaluation-summary p {{ margin: 0; }}
    button {{ border: 1px solid var(--accent); background: var(--accent); color: white; border-radius: 6px; padding: 9px 12px; cursor: pointer; }}
    .file-label {{ border: 1px solid var(--line); background: #fffdf7; color: var(--ink); border-radius: 6px; padding: 9px 12px; cursor: pointer; }}
    .file-label input {{ display: none; }}
    .import-status {{ color: var(--muted); font-size: 13px; align-self: center; }}
    .meter {{ height: 8px; background: #e4dccd; border-radius: 999px; overflow: hidden; margin: 10px 0; }}
    .meter i {{ display: block; height: 100%; background: var(--accent); }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #17211f; color: #eaf5ef; padding: 14px; border-radius: 8px; max-height: 360px; overflow: auto; }}
    @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} header {{ display:block; }} }}
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
    <section class="panel">
      <video controls src="{html.escape(video_src or '')}"></video>
      <div class="actions">
        <button type="button" onclick="refreshAnnotations()">生成标注 JSON</button>
        <button type="button" onclick="downloadAnnotations()">下载标注 JSON</button>
        <label class="file-label">导入标注 JSON
          <input id="annotation-import-file" type="file" accept="application/json,.json" onchange="importAnnotations(event)">
        </label>
        <span id="import-status" class="import-status"></span>
      </div>
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
    function collectAnnotations() {{
      return {{
        schema_version: 'swing_manual_annotations_v1',
        source: data.paths,
        summary: data.summary,
        events: Array.from(document.querySelectorAll('.event-card')).map(card => {{
          const eventId = Number(card.dataset.eventId);
          const original = data.events.find(e => Number(e.event_id) === eventId) || {{}};
          return {{
            event_id: eventId,
            predicted_stroke_type: original.stroke_type,
            actual_stroke_type: card.querySelector('[data-field="actual_stroke_type"]').value,
            count_correct: card.querySelector('[data-field="count_correct"]').checked,
            valid_hit: card.querySelector('[data-field="valid_hit"]').checked,
            needs_review: card.querySelector('[data-field="needs_review"]').checked,
            issue_tags: Array.from(card.querySelectorAll('[data-tag]:checked')).map(input => input.dataset.tag),
            note: card.querySelector('[data-field="note"]').value.trim(),
            frames: {{
              start: original.start_frame,
              contact: original.contact_frame,
              peak: original.peak_frame,
              end: original.end_frame
            }},
            quality_flags: original.quality_flags || {{}}
          }};
        }})
      }};
    }}
    function refreshAnnotations() {{
      document.getElementById('annotation-json').textContent = JSON.stringify(collectAnnotations(), null, 2);
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
    function applyImportedAnnotations(payload) {{
      const importedEvents = Array.isArray(payload.events) ? payload.events : [];
      let applied = 0;
      for (const imported of importedEvents) {{
        const card = document.querySelector(`.event-card[data-event-id="${{Number(imported.event_id)}}"]`);
        if (!card) continue;
        setField(card, 'actual_stroke_type', imported.actual_stroke_type);
        setField(card, 'count_correct', imported.count_correct);
        setField(card, 'valid_hit', imported.valid_hit);
        setField(card, 'needs_review', imported.needs_review);
        setField(card, 'note', imported.note || '');
        const tags = new Set(imported.issue_tags || []);
        card.querySelectorAll('[data-tag]').forEach(input => input.checked = tags.has(input.dataset.tag));
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
      link.download = 'swing_manual_annotations.json';
      link.click();
      URL.revokeObjectURL(link.href);
    }}
    document.querySelectorAll('input, select, textarea').forEach(el => el.addEventListener('change', refreshAnnotations));
    document.querySelectorAll('textarea').forEach(el => el.addEventListener('input', refreshAnnotations));
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
