#!/usr/bin/env python3
"""Evaluate swing events against manual annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


UNCLEAR_LABELS = {"", "Unclear", "Unknown", "No Swing", None}


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_evaluation_output_path(event_json_path: str) -> str:
    path = Path(event_json_path)
    stem = path.stem
    if stem.endswith("_swing_events"):
        stem = stem[: -len("_swing_events")]
    return str(path.with_name(f"{stem}_swing_evaluation.json"))


def _events_by_id(events: Iterable[Dict]) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for event in events or []:
        if not isinstance(event, dict):
            continue
        event_id = event.get("event_id")
        if event_id is None:
            continue
        out[int(event_id)] = event
    return out


def _manual_contact(annotation: Dict) -> Optional[int]:
    frames = annotation.get("frames") or {}
    value = frames.get("contact", annotation.get("contact_frame"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _model_contact(event: Dict) -> Optional[int]:
    value = event.get("contact_frame")
    if value is None:
        frames = event.get("frames") or {}
        value = frames.get("contact_frame") or frames.get("contact")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _norm_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_valid_manual_event(annotation: Dict) -> bool:
    actual = _norm_label(annotation.get("actual_stroke_type"))
    if actual in UNCLEAR_LABELS:
        return False
    return bool(annotation.get("valid_hit", True))


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def evaluate_swing_events(
    event_data: Dict,
    annotation_data: Dict,
    contact_tolerance_frames: int = 3,
) -> Dict:
    """Compare model swing events with human annotations.

    Matching is event-id based because the report page exports annotations from
    existing event cards. Unmatched model or annotation ids are surfaced instead
    of being silently ignored.
    """
    model_events = event_data.get("events") or []
    annotations = annotation_data.get("events") or []
    model_by_id = _events_by_id(model_events)
    annotation_by_id = _events_by_id(annotations)

    comparable_type_count = 0
    stroke_type_correct = 0
    comparable_contact_count = 0
    contact_within_tolerance = 0
    contact_abs_deltas: List[int] = []
    false_positive_ids: List[int] = []
    manual_review_ids: List[int] = []
    model_review_ids: List[int] = []
    event_reports: List[Dict] = []

    for event_id in sorted(set(model_by_id) | set(annotation_by_id)):
        event = model_by_id.get(event_id, {})
        annotation = annotation_by_id.get(event_id, {})
        predicted = _norm_label(event.get("stroke_type") or annotation.get("predicted_stroke_type"))
        actual = _norm_label(annotation.get("actual_stroke_type"))
        valid_hit = _is_valid_manual_event(annotation) if annotation else False
        needs_review = bool(annotation.get("needs_review", False))
        if needs_review:
            manual_review_ids.append(event_id)

        quality_flags = event.get("quality_flags") or annotation.get("quality_flags") or {}
        model_review = bool(quality_flags.get("review_recommended", False))
        if model_review:
            model_review_ids.append(event_id)

        stroke_match: Optional[bool] = None
        if predicted not in UNCLEAR_LABELS and actual not in UNCLEAR_LABELS and valid_hit:
            comparable_type_count += 1
            stroke_match = predicted == actual
            if stroke_match:
                stroke_type_correct += 1

        model_contact = _model_contact(event)
        manual_contact = _manual_contact(annotation)
        contact_delta: Optional[int] = None
        contact_ok: Optional[bool] = None
        if model_contact is not None and manual_contact is not None and valid_hit:
            comparable_contact_count += 1
            contact_delta = model_contact - manual_contact
            contact_abs_deltas.append(abs(contact_delta))
            contact_ok = abs(contact_delta) <= contact_tolerance_frames
            if contact_ok:
                contact_within_tolerance += 1

        if annotation and not valid_hit:
            false_positive_ids.append(event_id)

        event_reports.append(
            {
                "event_id": event_id,
                "matched_model_event": event_id in model_by_id,
                "matched_manual_annotation": event_id in annotation_by_id,
                "predicted_stroke_type": predicted,
                "actual_stroke_type": actual,
                "valid_hit": valid_hit,
                "count_correct": bool(annotation.get("count_correct", True)) if annotation else None,
                "stroke_type_correct": stroke_match,
                "model_contact_frame": model_contact,
                "manual_contact_frame": manual_contact,
                "contact_delta_frames": contact_delta,
                "contact_within_tolerance": contact_ok,
                "manual_needs_review": needs_review,
                "model_review_recommended": model_review,
                "issue_tags": list(annotation.get("issue_tags") or []),
                "quality_warnings": list(quality_flags.get("warnings") or []),
            }
        )

    manual_valid_count = sum(1 for annotation in annotations if _is_valid_manual_event(annotation))
    unmatched_model_ids = sorted(set(model_by_id) - set(annotation_by_id))
    unmatched_annotation_ids = sorted(set(annotation_by_id) - set(model_by_id))
    mean_contact_error = (
        round(sum(contact_abs_deltas) / len(contact_abs_deltas), 3) if contact_abs_deltas else None
    )

    return {
        "schema_version": "swing_evaluation_v1",
        "source": {
            "event_json": annotation_data.get("source", {}).get("event_json"),
            "annotation_schema": annotation_data.get("schema_version"),
        },
        "settings": {
            "contact_tolerance_frames": int(contact_tolerance_frames),
        },
        "summary": {
            "predicted_event_count": len(model_events),
            "manual_annotation_count": len(annotations),
            "manual_valid_event_count": manual_valid_count,
            "event_count_delta": len(model_events) - manual_valid_count,
            "stroke_type_comparable": comparable_type_count,
            "stroke_type_correct": stroke_type_correct,
            "stroke_type_accuracy": _safe_ratio(stroke_type_correct, comparable_type_count),
            "contact_comparable": comparable_contact_count,
            "contact_within_tolerance": contact_within_tolerance,
            "contact_accuracy": _safe_ratio(contact_within_tolerance, comparable_contact_count),
            "contact_mean_abs_error_frames": mean_contact_error,
            "manual_review_count": len(manual_review_ids),
            "model_review_recommended_count": len(model_review_ids),
            "false_positive_count": len(false_positive_ids),
            "unmatched_model_event_count": len(unmatched_model_ids),
            "unmatched_annotation_count": len(unmatched_annotation_ids),
        },
        "review": {
            "manual_review_event_ids": manual_review_ids,
            "model_review_event_ids": model_review_ids,
            "false_positive_event_ids": false_positive_ids,
            "unmatched_model_event_ids": unmatched_model_ids,
            "unmatched_annotation_event_ids": unmatched_annotation_ids,
        },
        "events": event_reports,
    }


def write_evaluation_report(
    event_json_path: str,
    annotation_json_path: str,
    output_path: Optional[str] = None,
    contact_tolerance_frames: int = 3,
) -> str:
    event_data = load_json(event_json_path)
    annotation_data = load_json(annotation_json_path)
    report = evaluate_swing_events(
        event_data,
        annotation_data,
        contact_tolerance_frames=contact_tolerance_frames,
    )
    report["source"]["event_json"] = event_json_path
    report["source"]["annotation_json"] = annotation_json_path

    output_path = output_path or default_evaluation_output_path(event_json_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate swing events against manual annotations.")
    parser.add_argument("--events", required=True, help="Path to *_swing_events.json")
    parser.add_argument("--annotations", required=True, help="Path to swing_manual_annotations.json")
    parser.add_argument("--output", default=None, help="Output evaluation JSON path")
    parser.add_argument("--contact-tolerance-frames", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_path = write_evaluation_report(
        args.events,
        args.annotations,
        output_path=args.output,
        contact_tolerance_frames=args.contact_tolerance_frames,
    )
    print(f"Wrote swing evaluation: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
