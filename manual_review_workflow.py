#!/usr/bin/env python3
"""Serve the human-review page and derive audited post-review Coach results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import webbrowser
from collections import Counter
from copy import deepcopy
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from local_realtime_coach import LocalRealtimeCoach
from swing_biomechanics import aggregate_event_biomechanics
from swing_coach_calibration import calibrate_coaching_event
from swing_evaluation import evaluate_swing_events
from swing_event_segmenter import summarize_manual_event_range
from swing_motion_features import extract_motion_features
from swing_session_quality import build_session_quality_dashboard


MAX_ANNOTATION_BYTES = 5 * 1024 * 1024
VALID_STROKE_TYPES = {
    "Forehand",
    "Backhand",
    "Two-Handed Backhand",
    "Serve",
    "Volley",
    "Unclear",
    "Unknown",
}


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Frame JSONL line {line_number} is not an object")
            rows.append(payload)
    return rows


def _atomic_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_json(payload: Dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def discover_session_paths(session_dir: Path) -> Dict[str, Path]:
    session_dir = session_dir.expanduser().resolve()
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session directory does not exist: {session_dir}")

    def choose(exact: str, pattern: str) -> Path:
        exact_path = session_dir / exact
        if exact_path.exists():
            return exact_path
        candidates = sorted(session_dir.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"Missing {exact} in {session_dir}")
        return candidates[0]

    events = choose("final_events.json", "*_events.json")
    frames = choose("final_frames.jsonl", "*_frames.jsonl")
    report = choose("final_report.html", "*_report.html")
    stem = events.stem[:-len("_events")] if events.stem.endswith("_events") else events.stem
    return {
        "session_dir": session_dir,
        "events": events,
        "frames": frames,
        "report": report,
        "annotations": session_dir / f"{stem}_manual_annotations.json",
        "evaluation": session_dir / f"{stem}_evaluation.json",
        "manual_events": session_dir / f"{stem}_manual_events.json",
        "state": session_dir / f"{stem}_manual_review_state.json",
    }


def _annotation_frames(annotation: Dict) -> Tuple[int, int, int]:
    frames = annotation.get("frames") or {}
    start = frames.get("start", annotation.get("start_frame"))
    contact = frames.get("contact", annotation.get("contact_frame"))
    end = frames.get("end", annotation.get("end_frame"))
    if start is None or contact is None or end is None:
        raise ValueError(
            f"标注 {annotation.get('annotation_id') or '?'} 缺少开始/触球/结束帧"
        )
    return int(start), int(contact), int(end)


def validate_manual_annotations(
    event_document: Dict,
    annotation_document: Dict,
    frame_records: Iterable[Dict],
    event_path: Optional[Path] = None,
) -> Dict:
    if annotation_document.get("schema_version") != "swing_manual_annotations_v2":
        raise ValueError("只支持 swing_manual_annotations_v2")
    annotations = annotation_document.get("events")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("人工标注中没有事件")

    source = annotation_document.get("source") or {}
    source_event_json = source.get("event_json")
    if event_path is not None and source_event_json:
        if Path(str(source_event_json)).name != event_path.name:
            raise ValueError(
                f"标注来源 {Path(str(source_event_json)).name} 与当前会话 {event_path.name} 不匹配"
            )
    source_session_id = source.get("session_id")
    current_session_id = (event_document.get("session") or {}).get("session_id")
    if source_session_id and current_session_id and source_session_id != current_session_id:
        raise ValueError("标注 session_id 与当前会话不匹配")

    model_ids = {
        int(event["event_id"])
        for event in event_document.get("events") or []
        if event.get("event_id") is not None
    }
    frame_ids = [
        int(record["frame_id"])
        for record in frame_records
        if record.get("frame_id") is not None
    ]
    if not frame_ids:
        raise ValueError("当前会话没有可用于 Coach 重算的逐帧证据")
    min_frame, max_frame = min(frame_ids), max(frame_ids)
    seen_annotation_ids = set()
    seen_source_ids = set()
    pending_ids = []

    for index, annotation in enumerate(annotations, start=1):
        if not isinstance(annotation, dict):
            raise ValueError(f"第 {index} 条人工标注不是对象")
        annotation_id = str(annotation.get("annotation_id") or f"manual-{index}")
        if annotation_id in seen_annotation_ids:
            raise ValueError(f"人工标注 ID 重复: {annotation_id}")
        seen_annotation_ids.add(annotation_id)

        source_event_id = annotation.get("source_event_id")
        if source_event_id is not None:
            source_event_id = int(source_event_id)
            if source_event_id not in model_ids:
                raise ValueError(f"标注 {annotation_id} 引用了不存在的事件 #{source_event_id}")
            if source_event_id in seen_source_ids:
                raise ValueError(f"模型事件 #{source_event_id} 被重复标注")
            seen_source_ids.add(source_event_id)

        start, contact, end = _annotation_frames(annotation)
        if not start <= contact <= end:
            raise ValueError(f"标注 {annotation_id} 必须满足 start <= contact <= end")
        if start < min_frame or end > max_frame:
            raise ValueError(
                f"标注 {annotation_id} 超出逐帧证据范围 {min_frame}-{max_frame}"
            )
        stroke_type = str(annotation.get("actual_stroke_type") or "Unclear")
        if stroke_type not in VALID_STROKE_TYPES:
            raise ValueError(f"标注 {annotation_id} 的挥拍类型不支持: {stroke_type}")
        if annotation.get("needs_review"):
            pending_ids.append(annotation_id)

    timeline_complete = bool(annotation_document.get("timeline_review_complete"))
    return {
        "timeline_review_complete": timeline_complete,
        "pending_ids": pending_ids,
        "pending_count": len(pending_ids),
        "metrics_finalizable": timeline_complete and not pending_ids,
        "annotation_count": len(annotations),
        "frame_range": [min_frame, max_frame],
    }


def _coach_messages(event: Optional[Dict]) -> List[Dict]:
    if not event:
        return []
    advices = event.get("coach_advices") or []
    if not advices and event.get("coach_advice"):
        advices = [event["coach_advice"]]
    return [
        {
            "code": advice.get("code"),
            "message": advice.get("message"),
            "confidence": advice.get("confidence"),
            "source": advice.get("source"),
        }
        for advice in advices
        if isinstance(advice, dict)
    ]


def _changed_fields(original: Optional[Dict], manual: Dict) -> List[str]:
    if not original:
        return ["new_manual_event"]
    fields = {
        "start_frame": (original.get("start_frame"), manual.get("start_frame")),
        "contact_frame": (original.get("contact_frame"), manual.get("contact_frame")),
        "end_frame": (original.get("end_frame"), manual.get("end_frame")),
        "stroke_type": (original.get("stroke_type"), manual.get("stroke_type")),
        "coach_advice": (_coach_messages(original), _coach_messages(manual)),
    }
    return [name for name, (before, after) in fields.items() if before != after]


def derive_manual_coach_events(
    event_document: Dict,
    annotation_document: Dict,
    frame_records: List[Dict],
) -> Dict:
    features = extract_motion_features(frame_records)
    model_by_id = {
        int(event["event_id"]): event
        for event in event_document.get("events") or []
        if event.get("event_id") is not None
    }
    coach = LocalRealtimeCoach()
    annotation_hash = _sha256_json(annotation_document)
    manual_events = []

    sortable = []
    for annotation in annotation_document.get("events") or []:
        if not annotation.get("valid_hit", True):
            continue
        start, contact, end = _annotation_frames(annotation)
        sortable.append((start, contact, end, annotation))

    for manual_id, (start, contact, end, annotation) in enumerate(
        sorted(sortable, key=lambda item: (item[0], item[1], item[2])),
        start=1,
    ):
        source_event_id = annotation.get("source_event_id")
        original = model_by_id.get(int(source_event_id)) if source_event_id is not None else None
        event = deepcopy(original or {})
        event["event_id"] = manual_id
        event["source_event_id"] = int(source_event_id) if source_event_id is not None else None
        event["stroke_type"] = str(annotation.get("actual_stroke_type") or "Unclear")
        summary = summarize_manual_event_range(
            features,
            start,
            contact,
            end,
            classification_evidence=(original or {}).get("evidence") or {},
        )
        event.update({key: value for key, value in summary.items() if key != "frame_phases"})
        event["biomechanics"] = aggregate_event_biomechanics(event, frame_records, features)
        event["coach_calibration"] = calibrate_coaching_event(event)
        advices = coach.advise_all(event)
        event["coach_advices"] = advices
        event["coach_advice"] = advices[0]
        event["review_provenance"] = {
            "mode": "manual_recomputed",
            "annotation_id": annotation.get("annotation_id"),
            "source_event_id": event.get("source_event_id"),
            "annotation_sha256": annotation_hash,
            "changed_fields": _changed_fields(original, event),
            "evidence_range": [
                event["start_frame"],
                event["contact_frame"],
                event["end_frame"],
            ],
        }
        manual_events.append(event)

    type_counts = Counter(event.get("stroke_type") for event in manual_events)
    return {
        "schema_version": "tennis.manual-coach-events.v1",
        "document_type": "manual_coach_events",
        "session": deepcopy(event_document.get("session") or {}),
        "summary": {
            "event_count": len(manual_events),
            "stroke_type_counts": dict(sorted(type_counts.items())),
            "session_quality": build_session_quality_dashboard(manual_events),
            "source_annotation_sha256": annotation_hash,
        },
        "events": manual_events,
    }


def _comparison_rows(event_document: Dict, manual_document: Optional[Dict]) -> List[Dict]:
    original_by_id = {
        int(event["event_id"]): event
        for event in event_document.get("events") or []
        if event.get("event_id") is not None
    }
    rows = []
    for manual in (manual_document or {}).get("events") or []:
        source_id = manual.get("source_event_id")
        original = original_by_id.get(int(source_id)) if source_id is not None else None
        rows.append(
            {
                "event_id": manual.get("event_id"),
                "source_event_id": source_id,
                "stroke_type": manual.get("stroke_type"),
                "frames": {
                    "start": manual.get("start_frame"),
                    "contact": manual.get("contact_frame"),
                    "end": manual.get("end_frame"),
                },
                "original_coach": _coach_messages(original),
                "manual_coach": _coach_messages(manual),
                "changed_fields": (manual.get("review_provenance") or {}).get(
                    "changed_fields"
                )
                or [],
            }
        )
    return rows


def process_manual_review(paths: Dict[str, Path], annotation_document: Dict) -> Dict:
    event_document = _load_json(paths["events"])
    frame_records = _load_jsonl(paths["frames"])
    validation = validate_manual_annotations(
        event_document,
        annotation_document,
        frame_records,
        event_path=paths["events"],
    )
    _atomic_json(paths["annotations"], annotation_document)

    evaluation = evaluate_swing_events(event_document, annotation_document)
    evaluation.setdefault("source", {})["event_json"] = str(paths["events"])
    evaluation["source"]["annotation_json"] = str(paths["annotations"])
    _atomic_json(paths["evaluation"], evaluation)

    manual_document = None
    if validation["metrics_finalizable"]:
        manual_document = derive_manual_coach_events(
            event_document,
            annotation_document,
            frame_records,
        )
        _atomic_json(paths["manual_events"], manual_document)

    state = {
        "schema_version": "tennis.manual-review-state.v1",
        "status": "finalized" if manual_document else "needs_review",
        "validation": validation,
        "evaluation": evaluation,
        "comparisons": _comparison_rows(event_document, manual_document),
        "paths": {
            key: str(path)
            for key, path in paths.items()
            if key not in {"session_dir"}
        },
    }
    _atomic_json(paths["state"], state)
    return state


def load_review_state(paths: Dict[str, Path]) -> Dict:
    if paths["state"].exists():
        return _load_json(paths["state"])
    event_document = _load_json(paths["events"])
    return {
        "schema_version": "tennis.manual-review-state.v1",
        "status": "waiting_for_annotations",
        "validation": None,
        "evaluation": None,
        "comparisons": [],
        "realtime_events": [
            {
                "event_id": event.get("event_id"),
                "stroke_type": event.get("stroke_type"),
                "coach": _coach_messages(event),
            }
            for event in event_document.get("events") or []
        ],
    }


def refresh_realtime_report(paths: Dict[str, Path]) -> None:
    """Regenerate the existing report with the current renderer, without workers."""
    from realtime_swing_pipeline import RealtimeSwingOutputManager

    document = _load_json(paths["events"])
    renderer = object.__new__(RealtimeSwingOutputManager)
    renderer.output_json = paths["events"]
    renderer.output_html = paths["report"]
    renderer.roi_metadata = deepcopy((document.get("summary") or {}).get("roi") or {})
    preview = paths["report"].with_name(f"{paths['report'].stem}_roi_preview.jpg")
    renderer.preview_path = preview if preview.exists() else None
    html = renderer._render_live_html(document)
    temporary = paths["report"].with_name(f".{paths['report'].name}.tmp")
    temporary.write_text(html, encoding="utf-8")
    os.replace(temporary, paths["report"])


def make_handler(paths: Dict[str, Path]):
    session_dir = paths["session_dir"]

    class ManualReviewHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(session_dir), **kwargs)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send_response(302)
                self.send_header("Location", f"/{paths['report'].name}")
                self.end_headers()
                return
            if path == "/api/manual-review/state":
                self._send_json(200, load_review_state(paths))
                return
            super().do_GET()

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/api/manual-review/evaluate":
                self._send_json(404, {"error": "unknown_endpoint"})
                return
            try:
                content_length = int(self.headers.get("Content-Length") or 0)
                if content_length <= 0 or content_length > MAX_ANNOTATION_BYTES:
                    raise ValueError("标注文件为空或超过 5 MB")
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("标注 JSON 根节点必须是对象")
                state = process_manual_review(paths, payload)
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(422, {"error": "invalid_annotations", "message": str(exc)})
                return
            except Exception as exc:
                self._send_json(500, {"error": "workflow_failed", "message": str(exc)})
                return
            self._send_json(200, state)

        def _send_json(self, status: int, payload: Dict) -> None:
            content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format_string, *args):
            print(f"[Manual-Review] {self.address_string()} {format_string % args}")

    return ManualReviewHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a local human-review workflow for one realtime Swing session."
    )
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--no-refresh-report", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    paths = discover_session_paths(Path(args.session_dir))
    if not args.no_refresh_report:
        refresh_realtime_report(paths)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(paths))
    url = f"http://{args.host}:{server.server_port}/{paths['report'].name}"
    print(f"Manual review page: {url}")
    if args.open:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
