#!/usr/bin/env python3
"""Serve the human-review page and derive audited post-review Coach results."""

from __future__ import annotations

from manual_annotation_contract import require_manual_frame_id
from manual_review_policy import VERSION as REVIEW_POLICY_VERSION, review_summary

import argparse
import hashlib
import json
import os
import threading
import webbrowser
import base64
import fcntl
import tempfile
import uuid
from collections import Counter
from copy import deepcopy
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from local_realtime_coach import LocalRealtimeCoach
from manual_review_llm import MAX_REVIEW_CASES, ReviewCase, ReviewChoice, review_uncertain_events
from swing_biomechanics import aggregate_event_biomechanics
from swing_coach_calibration import calibrate_coaching_event
from swing_evaluation import evaluate_swing_events
from evaluation_identity_contract import normalize_evaluation_inputs, normalize_model_event_document
from swing_event_segmenter import motion_energy_by_frame, summarize_manual_event_range
from swing_motion_features import extract_motion_features
from swing_session_quality import build_session_quality_dashboard
from event_source_timing import analyze_event_source_timing


MAX_ANNOTATION_BYTES = 5 * 1024 * 1024
AUTO_REVIEW_MIN_CONFIDENCE = 0.70
DEFAULT_REVIEW_CONFIG = Path(__file__).resolve().parent / "configs" / "yolo26_tennis_config.yaml"
# Serialize review publication across HTTP handler threads in this process.
_REVIEW_LOCK = threading.RLock()
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
    _atomic_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode('utf-8'))


def _atomic_bytes(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _recover_review_publication(paths):
    journal = paths['session_dir'] / '.manual_review_pending.json'
    if not journal.exists(): return
    pending = _load_json(journal)
    current = _load_json(paths['state']) if paths['state'].exists() else {}
    if current.get('publication_id') != pending['publication_id']:
        for key, previous in pending['previous'].items():
            if key not in ('annotations','evaluation','manual_events','state'):
                raise ValueError('Invalid review publication recovery key')
            if previous is None: paths[key].unlink(missing_ok=True)
            else: _atomic_bytes(paths[key], base64.b64decode(previous))
    journal.unlink()


def _publish_review(paths, documents):
    publication_id = uuid.uuid4().hex
    documents['state']['publication_id'] = publication_id
    revision = paths['session_dir'] / 'manual_review_revisions' / f'{publication_id}.json'
    documents['state']['revision_snapshot'] = str(revision)
    # Complete immutable revision first; the state file is the commit marker.
    _atomic_json(revision, {'publication_id':publication_id, 'documents':documents})
    journal = paths['session_dir'] / '.manual_review_pending.json'
    previous = {key:base64.b64encode(paths[key].read_bytes()).decode('ascii')
                if paths[key].exists() else None for key in ('annotations','evaluation','manual_events','state')}
    _atomic_json(journal, {'publication_id':publication_id, 'previous':previous})
    try:
        for key, document in documents.items():
            _atomic_json(paths[key], document)
    except BaseException:
        _recover_review_publication(paths)
        raise
    journal.unlink()


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

    def choose(exact: str, pattern: str, plain: str) -> Path:
        exact_path = session_dir / exact
        if exact_path.exists():
            return exact_path
        if (session_dir / plain).exists():
            return session_dir / plain
        candidates = sorted(session_dir.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"Missing {exact} in {session_dir}")
        return candidates[0]

    events = choose("final_events.json", "*_events.json", "events.json")
    frames = choose("final_frames.jsonl", "*_frames.jsonl", "frames.jsonl")
    report = choose("final_report.html", "*_report.html", "report.html")
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
    frames = annotation.get("frames")
    if frames is None:
        frames = {}
    if not isinstance(frames, dict):
        raise ValueError('标注frames必须是对象')
    start = frames.get("start", annotation.get("start_frame"))
    contact = frames.get("contact", annotation.get("contact_frame"))
    end = frames.get("end", annotation.get("end_frame"))
    if start is None or contact is None or end is None:
        raise ValueError(
            f"标注 {annotation.get('annotation_id') or '?'} 缺少开始/触球/结束帧"
        )
    return tuple(require_manual_frame_id(fid) for fid in (start, contact, end))


def validate_manual_annotations(
    event_document: Dict,
    annotation_document: Dict,
    frame_records: Iterable[Dict],
    event_path: Optional[Path] = None,
) -> Dict:
    validated_events, _ = normalize_evaluation_inputs(event_document, annotation_document)
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
        event["event_id"] for event in validated_events["events"]
    }
    frame_ids = []
    for index, record in enumerate(frame_records):
        if not isinstance(record, dict):
            raise ValueError(f'逐帧证据[{index + 1}]必须是对象')
        try:
            frame_ids.append(require_manual_frame_id(record.get('frame_id')))
        except ValueError as error:
            raise ValueError(f'逐帧证据[{index + 1}].frame_id: {error}') from error
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
        if annotation.get("valid_hit", True) and not any(
            start <= frame_id <= end for frame_id in frame_ids
        ):
            raise ValueError(f"标注 {annotation_id} 区间内没有逐帧证据，无法重算 Coach")
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
    validate_manual_annotations(event_document, annotation_document, frame_records)
    summary = event_document.get("summary") or {}
    dominant_hand = (summary.get("thresholds") or {}).get("dominant_hand", "right")
    coach_configuration = summary.get("coach_configuration")
    features = extract_motion_features(frame_records, dominant_hand=dominant_hand)
    model_by_id = {
        event["event_id"]: event
        for event in event_document.get("events") or []
        if event.get("event_id") is not None
    }
    coach = LocalRealtimeCoach.from_configuration(coach_configuration)
    annotation_hash = _sha256_json(annotation_document)
    # Phase maps have JSON string keys; resolve them against validated observations.
    phase_source_ids = {str(record['frame_id']): record['frame_id'] for record in frame_records}
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
        original = model_by_id.get(source_event_id) if source_event_id is not None else None
        event = deepcopy(original or {})
        event["event_id"] = manual_id
        event["source_event_id"] = source_event_id
        event["stroke_type"] = str(annotation.get("actual_stroke_type") or "Unclear")
        # Human decisions never come from copied model-event fields.
        event.pop('practice_review', None)
        event['human_coach_review'] = deepcopy(annotation.get('coach_review') or {})
        event['human_contact_review'] = annotation.get('contact_review', 'unreviewed')
        if annotation.get('practice_review') is not None:
            event['practice_review'] = deepcopy(annotation['practice_review'])
        summary = summarize_manual_event_range(
            features,
            start,
            contact,
            end,
            classification_evidence=(original or {}).get("evidence") or {},
        )
        event.update({key: value for key, value in summary.items() if key != "frame_phases"})
        if summary['manual_anchor_evidence']['anchors']['contact_frame']['status'] != 'observed':
            event['contact_status'] = 'unknown'
        event['timing'] = {'latency_scope': 'manual_recomputation'}
        event.pop('candidate_runtime_timing', None)
        event['phase_timing'] = analyze_event_source_timing(event, frame_records,
            [{'frame': phase_source_ids[str(fid)], 'event_id': manual_id, 'phase': phase}
             for fid, phase in summary['frame_phases'].items()])
        event["biomechanics"] = aggregate_event_biomechanics(event, frame_records, features)
        from practice_scoring import attach_score
        attach_score(event)
        from practice_scoring import score_review
        event['human_practice_score'] = score_review(annotation.get('practice_review') or {})
        advices = coach.advise_all(event)
        event["coach_advices"] = advices
        event["coach_advice"] = advices[0] if advices else None
        event["review_provenance"] = {
            "dominant_hand": dominant_hand,
            "coach_configuration": coach.configuration(),
            "configuration_source": "session" if coach_configuration is not None else "legacy_defaults",
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
        event["event_id"]: event
        for event in event_document.get("events") or []
        if event.get("event_id") is not None
    }
    rows = []
    for manual in (manual_document or {}).get("events") or []:
        source_id = manual.get("source_event_id")
        original = original_by_id.get(source_id) if source_id is not None else None
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
                "human_coach_review": deepcopy(manual.get('human_coach_review') or {}),
                "human_contact_review": manual.get('human_contact_review', 'unreviewed'),
                "human_practice_score": deepcopy(manual.get('human_practice_score') or {}),
                "changed_fields": (manual.get("review_provenance") or {}).get(
                    "changed_fields"
                )
                or [],
            }
        )
    return rows


def process_manual_review(paths: Dict[str, Path], annotation_document: Dict) -> Dict:
    with _REVIEW_LOCK, (paths['session_dir'] / '.manual_review.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            _recover_review_publication(paths)
            return _process_manual_review(paths, annotation_document)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def auto_calibrate_manual_review(
    paths: Dict[str, Path],
    request_payload: Optional[Dict] = None,
    *,
    config_path: Path = DEFAULT_REVIEW_CONFIG,
) -> Dict:
    """Build a rule-based draft and optionally refine it using bounded LLM choices."""
    payload = request_payload or {}
    use_llm = payload.get("use_llm") is True
    lock_path = paths['session_dir'] / '.manual_review.lock'
    with _REVIEW_LOCK, lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            _recover_review_publication(paths)
            revision = _review_source_revision(paths) if use_llm else None
            result, cases = _auto_calibrate_manual_review(paths, payload, prepare_llm=use_llm)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

    if not use_llm:
        return result
    cases.sort(key=lambda case: (case.facts["model_confidence"], case.event_id))
    cases = cases[:MAX_REVIEW_CASES]
    outcome = review_uncertain_events(cases, config_path)
    with _REVIEW_LOCK, lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if _review_source_revision(paths) != revision:
                return {"schema_version": result["schema_version"], "llm_review": {"status": "stale"}}
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    _apply_llm_choices(result, cases, outcome)
    return result


def llm_review_request_is_local(headers):
    host = headers.get("Host", "")
    try:
        address = urlparse(f"http://{host}")
    except ValueError:
        return False
    return (
        address.hostname in {"127.0.0.1", "localhost", "::1"}
        and address.username is None and not (address.path or address.params or address.query or address.fragment)
        and headers.get_content_type() == "application/json"
        and headers.get("Origin") in (None, f"http://{host}")
    )


def _review_source_revision(paths):
    def fingerprint(path):
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    return tuple(fingerprint(paths[key]) for key in ("events", "frames", "annotations", "state"))


def _build_llm_case(event_id, row, votes, contact_feats, original_contact, model_conf, review_reasons, peak_energy):
    current_type = row["actual_stroke_type"]
    contact = row["frames"]["contact"]
    choices = {}
    for stroke, count in votes.most_common(2):
        if stroke != current_type and count >= 2:
            choices[f"stroke:{len(choices) + 1}"] = ReviewChoice(
                "actual_stroke_type", stroke, {"votes": count},
            )
    observed = sorted(
        (item for item in contact_feats if item["frame_id"] != contact
         and abs(item["frame_id"] - original_contact) <= 8
         and item.get("ball_racket_distance") is not None and item["contact_score"] >= 0.35),
        key=lambda item: (-item["contact_score"], abs(item["frame_id"] - contact)),
    )
    for index, item in enumerate(observed[:3], 1):
        choices[f"contact:{index}"] = ReviewChoice("contact", item["frame_id"], {
            "offset_from_draft_contact": item["frame_id"] - contact,
            "local_contact_score": round(item["contact_score"], 3),
        })
    if not choices:
        return None
    return ReviewCase(event_id, {
        "current_stroke_type": current_type,
        "stroke_votes": dict(votes),
        "draft_contact_score": next((item["contact_score"] for item in contact_feats
                                     if item["frame_id"] == contact), None),
        "model_confidence": model_conf,
        "motion_energy_peak": round(peak_energy, 2),
        "review_reasons": review_reasons,
    }, choices)


def _apply_llm_choices(result, cases, outcome):
    result["llm_review"] = {
        "status": outcome.status, "model": outcome.model, "candidate_count": len(cases), "applied_count": 0,
    }
    by_id = {row["source_event_id"]: row for row in result["annotations"]["events"] if row.get("needs_review")}
    for case in cases:
        selected = outcome.selections.get(case.event_id)
        if not selected:
            continue
        row = by_id[case.event_id]
        diff = result["item_diffs"][str(case.event_id)]
        changes = []
        for key in selected:
            choice = case.choices[key]
            if choice.field == "contact":
                old = row["frames"]["contact"]
                row["frames"]["contact"] = choice.value
                diff["changes"].append(f"LLM 建议触球候选 {old}→{choice.value}")
                tag = "contact_timing"
            else:
                old = row["actual_stroke_type"]
                row["actual_stroke_type"] = choice.value
                diff["changes"].append(f"LLM 建议类型 {old}→{choice.value}")
                tag = "wrong_type"
            row["issue_tags"] = sorted({*row["issue_tags"], tag})
            changes.append({"field": choice.field, "from": old, "to": choice.value, "evidence": choice.evidence})
        row["llm_suggestion"] = {"model": outcome.model, "changes": changes}
        diff["status"] = "llm_suggested"
        result["llm_review"]["applied_count"] += 1


def _default_human_review_fields(existing: Optional[Dict]) -> Dict:
    existing = existing if isinstance(existing, dict) else {}
    return {
        "contact_review": existing.get("contact_review") or "unreviewed",
        "coach_review": deepcopy(
            existing.get("coach_review")
            or {
                "reviewer": "",
                "verdict": "unreviewed",
                "observed_fact": "",
                "suggestion": "",
                "evidence": "",
            }
        ),
        "practice_review": deepcopy(
            existing.get("practice_review")
            or {
                "confirmed": False,
                "ratings": {
                    "preparation": None,
                    "positioning": None,
                    "contact": None,
                    "coordination": None,
                    "recovery": None,
                },
            }
        ),
    }


def _auto_calibrate_manual_review(paths: Dict[str, Path], request_payload: Dict, *, prepare_llm=False):
    event_document = normalize_model_event_document(_load_json(paths["events"]))
    frame_records = _load_jsonl(paths["frames"]) if paths["frames"].exists() else []
    model_events = list(event_document.get("events") or [])
    thresholds = (event_document.get("summary") or {}).get("thresholds") or {}
    dominant_hand = str(thresholds.get("dominant_hand") or "right")

    features = extract_motion_features(frame_records, dominant_hand=dominant_hand) if frame_records else []
    feat_by_frame = {row["frame_id"]: row for row in features}
    energy_by_frame = motion_energy_by_frame(features) if features else {}
    min_fid = min(feat_by_frame, default=0)
    max_fid = max(feat_by_frame, default=0)

    current_doc = request_payload.get("current_annotations")
    saved_doc = _load_json(paths["annotations"]) if paths["annotations"].exists() else {}
    saved_rows = saved_doc.get("events") or []
    current_rows = current_doc.get("events") if isinstance(current_doc, dict) else None
    explicit_edits = request_payload.get("edited_event_ids")
    protected_ids = {i for i in explicit_edits if type(i) is int} if isinstance(explicit_edits, list) else set()
    protected_rows = {
        row["source_event_id"]: row
        for row in [*saved_rows, *(current_rows or [])]
        if isinstance(row, dict) and type(row.get("source_event_id")) is int
        and (row.get("annotation_origin") not in ("model_prefill", "auto_calibration")
             or row["source_event_id"] in protected_ids)
    }
    extra_manual_events = [deepcopy(row) for row in (saved_rows if current_rows is None else current_rows)
                           if isinstance(row, dict) and row.get("source_event_id") is None]

    calibrated_events: List[Dict] = []
    llm_cases = []
    item_diffs: Dict[str, Dict] = {}
    stats = {
        "total_events": len(model_events),
        "auto_calibrated_count": 0,
        "boundary_adjusted_count": 0,
        "contact_adjusted_count": 0,
        "type_corrected_count": 0,
        "filtered_invalid_count": 0,
        "pending_review_count": 0,
        "preserved_manual_count": 0,
    }

    prev_committed_end = -1
    sorted_events = sorted(model_events, key=lambda e: (e["start_frame"] is None, e["start_frame"] or 0, e["event_id"]))
    for idx, event in enumerate(sorted_events):
        event_id = event["event_id"]
        orig_start, orig_contact, orig_end = event["start_frame"], event["contact_frame"], event["end_frame"]
        orig_type = str(event.get("stroke_type") or "Unknown")

        protected = protected_rows.get(event_id)
        if protected is not None:
            preserved = {**deepcopy(protected), "annotation_origin": "human"}
            calibrated_events.append(preserved)
            try:
                _, _, p_end = _annotation_frames(preserved)
                prev_committed_end = max(prev_committed_end, p_end)
            except ValueError:
                if orig_end is not None:
                    prev_committed_end = max(prev_committed_end, orig_end)
            stats["preserved_manual_count"] += 1
            if preserved.get("needs_review"):
                stats["pending_review_count"] += 1
            item_diffs[str(event_id)] = {
                "event_id": event_id,
                "status": "preserved_manual",
                "changes": ["保留人工修改"],
                "needs_review": bool(preserved.get("needs_review")),
                "valid_hit": bool(preserved.get("valid_hit", True)),
            }
            continue

        orig_peak = event.get("peak_frame", event["frames"].get("peak"))
        if None in (orig_start, orig_contact, orig_end):
            calibrated_events.append({
                **_default_human_review_fields(None),
                "annotation_id": f"model-{event_id}", "source_event_id": event_id,
                "annotation_origin": "auto_calibration", "predicted_stroke_type": orig_type,
                "actual_stroke_type": orig_type if orig_type in VALID_STROKE_TYPES else "Unclear",
                "valid_hit": True, "count_correct": False, "needs_review": True,
                "issue_tags": [], "note": "[自动校订] 缺少源帧锚点，待人工复核",
                "frames": {"start": orig_start, "contact": orig_contact, "peak": orig_peak, "end": orig_end},
            })
            item_diffs[str(event_id)] = {
                "event_id": event_id, "status": "needs_review", "changes": ["缺少源帧锚点"],
                "needs_review": True, "valid_hit": True, "review_reasons": ["missing_source_anchor"],
            }
            stats["pending_review_count"] += 1
            continue
        if orig_peak is None:
            orig_peak = (orig_start + orig_end) // 2

        next_start = sorted_events[idx + 1]["start_frame"] if idx + 1 < len(sorted_events) else None
        next_start_limit = next_start - 1 if next_start is not None else (max_fid if feat_by_frame else orig_end)
        lower_bound = max(min_fid, prev_committed_end + 1)
        upper_bound = min(max_fid if feat_by_frame else orig_end, next_start_limit)
        range_conflict = lower_bound > min(orig_end, upper_bound)
        if range_conflict:
            lower_bound, upper_bound = orig_start, orig_end

        win_lo = max(lower_bound, orig_start - 6)
        win_hi = min(upper_bound, orig_end + 6)
        window_feats = [feat_by_frame[f] for f in range(win_lo, win_hi + 1) if f in feat_by_frame]
        core_feats = [feat_by_frame[f] for f in range(orig_start, orig_end + 1) if f in feat_by_frame]

        peak_energy = max((energy_by_frame[r["frame_id"]] for r in core_feats),
                          default=float(event.get("peak_energy") or 0.0))
        max_contact_score = max((float(r.get("contact_score") or 0.0) for r in core_feats), default=0.0)
        ball_dists = [float(r["ball_racket_distance"]) for r in core_feats if r.get("ball_racket_distance") is not None]
        min_ball_dist = min(ball_dists) if ball_dists else None

        # 1. Boundary refinement: trim stagnant tails while preserving prep/follow-through margin
        new_start, new_end = orig_start, orig_end
        if not range_conflict and len(window_feats) >= 6 and peak_energy >= 6.0:
            active_thr = max(4.5, peak_energy * 0.22)
            active_frames = [r["frame_id"] for r in core_feats if energy_by_frame[r["frame_id"]] >= active_thr]
            if active_frames:
                cand_start = max(lower_bound, active_frames[0] - 3)
                cand_end = min(upper_bound, active_frames[-1] + 3)
                if cand_start - orig_start >= 4:
                    new_start = cand_start
                if orig_end - cand_end >= 5:
                    new_end = cand_end
        new_start = max(lower_bound, min(new_start, upper_bound))
        new_end = max(new_start, min(new_end, upper_bound))
        if new_end - new_start < 6 and orig_end - orig_start >= 6:
            new_start = max(lower_bound, min(orig_start, upper_bound))
            new_end = max(new_start, min(orig_end, upper_bound))

        # 2. Contact frame refinement: prefer physical ball-racket contact score, else racket/wrist speed peak
        new_contact = max(new_start, min(orig_contact, new_end))
        contact_feats = [feat_by_frame[f] for f in range(new_start, new_end + 1) if f in feat_by_frame]
        if contact_feats and not range_conflict:
            scored = [r for r in contact_feats if float(r.get("contact_score") or 0.0) >= 0.35]
            if scored:
                best_c = max(scored, key=lambda r: (float(r.get("contact_score") or 0.0), float(r.get("racket_speed") or 0.0)))
                cand_contact = best_c["frame_id"]
            else:
                near_peak = [r for r in contact_feats if abs(r["frame_id"] - orig_peak) <= 5] or contact_feats
                best_c = max(
                    near_peak,
                    key=lambda r: 0.65 * float(r.get("racket_speed") or 0.0) + 0.35 * float(r.get("wrist_speed") or 0.0),
                )
                cand_contact = best_c["frame_id"]
            if abs(cand_contact - orig_contact) <= 8:
                new_contact = max(new_start, min(cand_contact, new_end))

        # 3. Stroke type calibration from frame-level votes & two-hand wrist proximity
        new_type = orig_type if orig_type in VALID_STROKE_TYPES else "Unclear"
        vote_feats = [feat_by_frame[f] for f in range(max(new_start, new_contact - 6), min(new_end, new_contact + 4) + 1) if f in feat_by_frame]
        votes = Counter(
            str(r.get("raw_swing_type"))
            for r in vote_feats
            if str(r.get("raw_swing_type")) in {"Forehand", "Backhand", "Two-Handed Backhand", "Serve", "Volley"}
        )
        two_hand_dists = [float(r["two_hand_distance"]) for r in vote_feats if r.get("two_hand_distance") is not None]
        med_two_hand = sorted(two_hand_dists)[len(two_hand_dists) // 2] if two_hand_dists else None
        if votes:
            top_type, top_cnt = votes.most_common(1)[0]
            total_votes = sum(votes.values())
            if new_type in {"Unknown", "Unclear"} and top_cnt >= 3 and (top_cnt / max(1, total_votes)) >= 0.6:
                new_type = top_type
        if new_type == "Backhand" and med_two_hand is not None and med_two_hand <= 55.0:
            new_type = "Two-Handed Backhand"
        if new_type not in VALID_STROKE_TYPES:
            new_type = "Unclear"

        # 4. Valid hit & shadow-swing / false-positive filtering
        q_flags = event.get("quality_flags") or {}
        warnings = list(q_flags.get("warnings") or [])
        contact_gate = ((event.get("dual_view_biomechanics") or {}).get("contact_distance_gate") or {})
        has_ball_evidence = (
            max_contact_score >= 0.25
            or (min_ball_dist is not None and min_ball_dist <= 180.0)
            or bool(contact_gate.get("is_valid_contact") is True)
        )
        duration = new_end - new_start + 1
        is_shadow_or_noise = (
            not has_ball_evidence and sum(bool(row.get("has_pose")) for row in core_feats) >= 3
            and (peak_energy < 8.5 or duration < 6
                 or (contact_gate.get("is_valid_contact") is False and peak_energy < 14.0))
        )
        valid_hit = not is_shadow_or_noise
        count_correct = valid_hit

        # 5. Issue tags & diff notes
        issue_tags: List[str] = []
        changes: List[str] = []
        if new_start != orig_start or new_end != orig_end:
            issue_tags.append("event_boundary")
            changes.append(f"边界 {orig_start}-{orig_end}→{new_start}-{new_end}")
            stats["boundary_adjusted_count"] += 1
        if abs(new_contact - orig_contact) >= 2:
            issue_tags.append("contact_timing")
            changes.append(f"触球 {orig_contact}→{new_contact}")
            stats["contact_adjusted_count"] += 1
        elif new_contact != orig_contact:
            changes.append(f"触球 {orig_contact}→{new_contact}")
            stats["contact_adjusted_count"] += 1
        if new_type != orig_type:
            issue_tags.append("wrong_type")
            changes.append(f"类型 {orig_type}→{new_type}")
            stats["type_corrected_count"] += 1
        if not valid_hit:
            changes.append("标为空挥/误检")
            stats["filtered_invalid_count"] += 1

        # 6. Confidence gating for needs_review
        severe_warnings = any(
            w in {"source_end_without_settle_confirmation", "overlapping_event_range", "missing_pose_window"}
            for w in warnings
        )
        reported_confidence = event.get("confidence")
        model_conf = (float(reported_confidence) if type(reported_confidence) in (int, float)
                      and 0 <= reported_confidence <= 1 else 0.0)
        review_reasons = [reason for condition, reason in (
            (not core_feats, "missing_frame_evidence"),
            (range_conflict, "overlapping_event_range"),
            (severe_warnings, "event_quality_warning"),
            (not model_conf >= AUTO_REVIEW_MIN_CONFIDENCE, "low_model_confidence"),
            (valid_hit and new_type in {"Unknown", "Unclear"}, "uncertain_stroke_type"),
            (not has_ball_evidence, "missing_ball_evidence"),
        ) if condition]
        needs_review = bool(review_reasons)

        if needs_review:
            stats["pending_review_count"] += 1
        else:
            stats["auto_calibrated_count"] += 1

        rationale_parts = list(changes)
        if not rationale_parts:
            rationale_parts.append("与物理证据一致")
        if needs_review:
            rationale_parts.append("置信偏低待人工复核")
        note_text = "[自动校订] " + "；".join(rationale_parts)

        annotation_row = {
            **_default_human_review_fields(None),
            "annotation_id": f"model-{event_id}",
            "annotation_origin": "auto_calibration",
            "source_event_id": event_id,
            "predicted_stroke_type": orig_type,
            "actual_stroke_type": new_type,
            "count_correct": count_correct,
            "valid_hit": valid_hit,
            "needs_review": needs_review,
            "issue_tags": sorted(set(issue_tags)),
            "note": note_text,
            "frames": {
                "start": new_start,
                "contact": new_contact,
                "peak": max(new_start, min(orig_peak, new_end)),
                "end": new_end,
            },
        }
        calibrated_events.append(annotation_row)
        prev_committed_end = max(prev_committed_end, new_end)

        status_tag = "filtered_invalid" if not valid_hit else ("needs_review" if needs_review else "auto_calibrated")
        item_diffs[str(event_id)] = {
            "event_id": event_id,
            "status": status_tag,
            "changes": changes,
            "needs_review": needs_review,
            "valid_hit": valid_hit,
            "has_ball_evidence": has_ball_evidence,
            "review_reasons": review_reasons,
            "peak_energy": round(peak_energy, 2),
        }
        if prepare_llm and needs_review and valid_hit and not range_conflict and core_feats:
            case = _build_llm_case(event_id, annotation_row, votes, contact_feats, orig_contact,
                                   model_conf, review_reasons, peak_energy)
            if case is not None:
                llm_cases.append(case)

    for extra in extra_manual_events:
        calibrated_events.append(extra)
        if extra.get("needs_review"):
            stats["pending_review_count"] += 1

    annotation_document = {
        "schema_version": "swing_manual_annotations_v2",
        "review_policy_version": REVIEW_POLICY_VERSION,
        "timeline_review_complete": False,
        "source": {
            "event_json": paths["events"].name,
            "reference_method": "auto_calibrated_review",
            "model_predictions_visible": True,
        },
        "events": calibrated_events,
    }
    return {
        "schema_version": "tennis.auto-calibration-result.v1",
        "summary": stats,
        "item_diffs": item_diffs,
        "annotations": annotation_document,
    }, llm_cases


def _process_manual_review(paths: Dict[str, Path], annotation_document: Dict) -> Dict:
    event_document = _load_json(paths["events"])
    frame_records = _load_jsonl(paths["frames"])
    validation = validate_manual_annotations(
        event_document,
        annotation_document,
        frame_records,
        event_path=paths["events"],
    )

    evaluation = evaluate_swing_events(event_document, annotation_document)
    evaluation.setdefault("source", {})["event_json"] = str(paths["events"])
    evaluation["source"]["annotation_json"] = str(paths["annotations"])

    manual_document = None
    if validation["metrics_finalizable"]:
        manual_document = derive_manual_coach_events(
            event_document,
            annotation_document,
            frame_records,
        )

    comparisons = _comparison_rows(event_document, manual_document)
    if manual_document is not None:
        originals = {row['event_id']: row for row in event_document.get('events', [])}
        from practice_scoring import score_review
        for annotation in annotation_document['events']:
            if annotation.get('valid_hit', True):
                continue
            source_id = annotation.get('source_event_id')
            start, contact, end = _annotation_frames(annotation)
            comparisons.append({'event_id':None, 'source_event_id':source_id,
                'annotation_id':annotation.get('annotation_id'), 'excluded_event':True,
                'stroke_type':annotation.get('actual_stroke_type'), 'frames':{'start':start, 'contact':contact, 'end':end},
                'original_coach':_coach_messages(originals.get(source_id)), 'manual_coach':[],
                'human_coach_review':deepcopy(annotation.get('coach_review') or {}),
                'human_contact_review':annotation.get('contact_review', 'unreviewed'),
                'human_practice_score':score_review(annotation.get('practice_review') or {}),
                'changed_fields':['valid_hit']})
    state = {
        "schema_version": "tennis.manual-review-state.v1",
        "status": "finalized" if manual_document else "needs_review",
        "review_policy_version": REVIEW_POLICY_VERSION,
        "human_review_summary": review_summary(annotation_document['events']),
        "annotation_sha256": _sha256_json(annotation_document),
        "annotations": deepcopy(annotation_document),
        "validation": validation,
        "evaluation": evaluation,
        "comparisons": comparisons,
        "paths": {
            key: str(path)
            for key, path in paths.items()
            if key not in {"session_dir"}
        },
    }
    # Complete validation and all derived calculations before replacing artifacts.
    documents = {'annotations':annotation_document, 'evaluation':evaluation}
    if manual_document is not None:
        documents['manual_events'] = manual_document
    documents['state'] = state
    _publish_review(paths, documents)
    return state


def load_review_state(paths: Dict[str, Path]) -> Dict:
    with _REVIEW_LOCK, (paths['session_dir'] / '.manual_review.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            _recover_review_publication(paths)
            return _load_review_state(paths)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _load_review_state(paths: Dict[str, Path]) -> Dict:
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


def make_handler(paths: Dict[str, Path], config_path: Path = DEFAULT_REVIEW_CONFIG):
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
            if path not in ("/api/manual-review/evaluate", "/api/manual-review/auto-calibrate"):
                self._send_json(404, {"error": "unknown_endpoint"})
                return
            try:
                content_length = int(self.headers.get("Content-Length") or 0)
                if content_length < 0 or content_length > MAX_ANNOTATION_BYTES:
                    raise ValueError("请求体为空或超过 5 MB")
                raw = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
                payload = json.loads(raw) if raw.strip() else {}
                if not isinstance(payload, dict):
                    raise ValueError("JSON 根节点必须是对象")
                if path == "/api/manual-review/auto-calibrate":
                    if payload.get("use_llm") is True and not llm_review_request_is_local(self.headers):
                        self._send_json(403, {"error": "llm_requires_local_json_request"})
                        return
                    state = auto_calibrate_manual_review(paths, payload, config_path=config_path)
                else:
                    if content_length <= 0:
                        raise ValueError("标注文件为空或超过 5 MB")
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
    parser.add_argument("--config", type=Path, default=DEFAULT_REVIEW_CONFIG, help="LLM review configuration")
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
    server = ThreadingHTTPServer((args.host, args.port), make_handler(paths, args.config))
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
