#!/usr/bin/env python3
"""Evaluate swing events against manual annotations."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
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


def _row_frame(row: Dict, name: str) -> Optional[int]:
    frames = row.get("frames") or {}
    value = frames.get(name)
    if value is None:
        value = row.get(f"{name}_frame")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _event_interval(row: Dict) -> Optional[Tuple[int, int]]:
    start = _row_frame(row, "start")
    end = _row_frame(row, "end")
    if start is None or end is None:
        return None
    return min(start, end), max(start, end)


def _interval_iou(left: Dict, right: Dict) -> Optional[float]:
    left_range = _event_interval(left)
    right_range = _event_interval(right)
    if left_range is None or right_range is None:
        return None
    intersection = max(
        0,
        min(left_range[1], right_range[1])
        - max(left_range[0], right_range[0])
        + 1,
    )
    union = (
        left_range[1]
        - left_range[0]
        + 1
        + right_range[1]
        - right_range[0]
        + 1
        - intersection
    )
    return round(intersection / union, 6) if union > 0 else None


def _annotation_id(annotation: Dict, index: int) -> str:
    value = annotation.get("annotation_id")
    if value is None:
        value = annotation.get("event_id")
    return str(value if value is not None else f"annotation-{index + 1}")


def _temporal_sort_key(row: Dict) -> Tuple[int, int, int]:
    contact = _row_frame(row, "contact")
    interval = _event_interval(row)
    sentinel = 2**31 - 1
    try:
        event_id = int(row.get("event_id") or 0)
    except (TypeError, ValueError):
        event_id = 0
    return (
        contact if contact is not None else (interval[0] if interval else sentinel),
        interval[0] if interval else sentinel,
        event_id,
    )


def _temporal_matches(
    model_events: List[Dict],
    annotations: List[Dict],
    match_contact_tolerance_frames: int,
    min_event_iou: float,
) -> List[Tuple[int, int, Optional[float]]]:
    """Return ordered one-to-one matches maximizing count, then evidence quality."""
    ordered_models = sorted(enumerate(model_events), key=lambda item: _temporal_sort_key(item[1]))
    ordered_annotations = sorted(
        enumerate(annotations),
        key=lambda item: _temporal_sort_key(item[1]),
    )
    tolerance = max(0, int(match_contact_tolerance_frames))
    minimum_iou = max(0.0, min(1.0, float(min_event_iou)))

    def evidence(model: Dict, annotation: Dict):
        model_contact = _model_contact(model)
        manual_contact = _manual_contact(annotation)
        contact_delta = (
            abs(model_contact - manual_contact)
            if model_contact is not None and manual_contact is not None
            else None
        )
        iou = _interval_iou(model, annotation)
        eligible = (
            (contact_delta is not None and contact_delta <= tolerance)
            or (iou is not None and iou >= minimum_iou)
        )
        if not eligible:
            return None
        contact_cost = (
            contact_delta / max(1, tolerance)
            if contact_delta is not None
            else 1.0
        )
        interval_cost = 1.0 - iou if iou is not None else 1.0
        return round(contact_cost + interval_cost, 6), iou

    @lru_cache(maxsize=None)
    def solve(model_pos: int, annotation_pos: int):
        if model_pos >= len(ordered_models) or annotation_pos >= len(ordered_annotations):
            return 0, 0.0, ()
        options = [
            solve(model_pos + 1, annotation_pos),
            solve(model_pos, annotation_pos + 1),
        ]
        model_index, model = ordered_models[model_pos]
        annotation_index, annotation = ordered_annotations[annotation_pos]
        match_evidence = evidence(model, annotation)
        if match_evidence is not None:
            cost, iou = match_evidence
            matches, future_cost, pairs = solve(model_pos + 1, annotation_pos + 1)
            options.append(
                (
                    matches + 1,
                    future_cost + cost,
                    ((model_index, annotation_index, iou),) + pairs,
                )
            )
        return min(options, key=lambda option: (-option[0], option[1]))

    return list(solve(0, 0)[2])


def _mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def _evaluate_v1(
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


def _evaluate_v2(
    event_data: Dict,
    annotation_data: Dict,
    contact_tolerance_frames: int,
    match_contact_tolerance_frames: int,
    min_event_iou: float,
) -> Dict:
    model_events = [event for event in event_data.get("events") or [] if isinstance(event, dict)]
    annotations = [
        annotation
        for annotation in annotation_data.get("events") or []
        if isinstance(annotation, dict)
    ]
    valid_annotations = [
        annotation for annotation in annotations if _is_valid_manual_event(annotation)
    ]
    matches = _temporal_matches(
        model_events,
        valid_annotations,
        match_contact_tolerance_frames=match_contact_tolerance_frames,
        min_event_iou=min_event_iou,
    )
    matched_model_indices = {model_index for model_index, _, _ in matches}
    matched_annotation_indices = {
        annotation_index for _, annotation_index, _ in matches
    }

    type_correct = 0
    contact_within_tolerance = 0
    contact_errors: List[float] = []
    start_errors: List[float] = []
    end_errors: List[float] = []
    event_ious: List[float] = []
    event_reports = []
    for model_index, annotation_index, event_iou in matches:
        event = model_events[model_index]
        annotation = valid_annotations[annotation_index]
        predicted = _norm_label(event.get("stroke_type"))
        actual = _norm_label(annotation.get("actual_stroke_type"))
        stroke_match = (
            predicted not in UNCLEAR_LABELS
            and actual not in UNCLEAR_LABELS
            and predicted == actual
        )
        if stroke_match:
            type_correct += 1

        model_contact = _model_contact(event)
        manual_contact = _manual_contact(annotation)
        contact_delta = (
            model_contact - manual_contact
            if model_contact is not None and manual_contact is not None
            else None
        )
        contact_ok = (
            abs(contact_delta) <= contact_tolerance_frames
            if contact_delta is not None
            else None
        )
        if contact_delta is not None:
            contact_errors.append(abs(contact_delta))
            if contact_ok:
                contact_within_tolerance += 1

        model_start = _row_frame(event, "start")
        manual_start = _row_frame(annotation, "start")
        model_end = _row_frame(event, "end")
        manual_end = _row_frame(annotation, "end")
        start_delta = (
            model_start - manual_start
            if model_start is not None and manual_start is not None
            else None
        )
        end_delta = (
            model_end - manual_end
            if model_end is not None and manual_end is not None
            else None
        )
        if start_delta is not None:
            start_errors.append(abs(start_delta))
        if end_delta is not None:
            end_errors.append(abs(end_delta))
        if event_iou is not None:
            event_ious.append(event_iou)

        quality_flags = event.get("quality_flags") or {}
        event_reports.append(
            {
                "model_event_id": int(event.get("event_id", model_index + 1)),
                "annotation_id": _annotation_id(annotation, annotation_index),
                "source_event_id": annotation.get("source_event_id"),
                "matched": True,
                "predicted_stroke_type": predicted,
                "actual_stroke_type": actual,
                "stroke_type_correct": stroke_match,
                "model_contact_frame": model_contact,
                "manual_contact_frame": manual_contact,
                "contact_delta_frames": contact_delta,
                "contact_within_tolerance": contact_ok,
                "model_start_frame": model_start,
                "manual_start_frame": manual_start,
                "start_delta_frames": start_delta,
                "model_end_frame": model_end,
                "manual_end_frame": manual_end,
                "end_delta_frames": end_delta,
                "event_iou": round(float(event_iou), 4) if event_iou is not None else None,
                "manual_needs_review": bool(annotation.get("needs_review", False)),
                "model_review_recommended": bool(
                    quality_flags.get("review_recommended", False)
                ),
                "issue_tags": list(annotation.get("issue_tags") or []),
                "quality_warnings": list(quality_flags.get("warnings") or []),
            }
        )

    unmatched_model_ids = sorted(
        int(event.get("event_id", index + 1))
        for index, event in enumerate(model_events)
        if index not in matched_model_indices
    )
    unmatched_annotation_ids = sorted(
        _annotation_id(annotation, index)
        for index, annotation in enumerate(valid_annotations)
        if index not in matched_annotation_indices
    )
    true_positive_count = len(matches)
    false_positive_count = len(unmatched_model_ids)
    false_negative_count = len(unmatched_annotation_ids)
    timeline_complete = bool(annotation_data.get("timeline_review_complete", False))
    manual_review_ids = [
        _annotation_id(annotation, index)
        for index, annotation in enumerate(annotations)
        if annotation.get("needs_review")
    ]
    provisional_reasons = []
    if not timeline_complete:
        provisional_reasons.append("timeline_review_incomplete")
    if manual_review_ids:
        provisional_reasons.append("manual_review_pending")
    metrics_finalized = not provisional_reasons
    precision = (
        _safe_ratio(true_positive_count, true_positive_count + false_positive_count)
        if metrics_finalized
        else None
    )
    recall = (
        _safe_ratio(true_positive_count, true_positive_count + false_negative_count)
        if metrics_finalized
        else None
    )
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    model_review_ids = [
        int(event.get("event_id", index + 1))
        for index, event in enumerate(model_events)
        if (event.get("quality_flags") or {}).get("review_recommended")
    ]

    return {
        "schema_version": "swing_evaluation_v2",
        "source": {
            "event_json": (annotation_data.get("source") or {}).get("event_json"),
            "annotation_schema": annotation_data.get("schema_version"),
        },
        "settings": {
            "contact_tolerance_frames": int(contact_tolerance_frames),
            "match_contact_tolerance_frames": int(match_contact_tolerance_frames),
            "min_event_iou": float(min_event_iou),
            "matching": "ordered_temporal_alignment",
        },
        "summary": {
            "provisional": not metrics_finalized,
            "provisional_reasons": provisional_reasons,
            "metrics_finalized": metrics_finalized,
            "timeline_review_complete": timeline_complete,
            "predicted_event_count": len(model_events),
            "manual_annotation_count": len(annotations),
            "manual_valid_event_count": len(valid_annotations),
            "event_count_delta": len(model_events) - len(valid_annotations),
            "true_positive_count": true_positive_count,
            "false_positive_count": false_positive_count,
            "false_negative_count": false_negative_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "stroke_type_comparable": len(matches),
            "stroke_type_correct": type_correct,
            "stroke_type_accuracy": _safe_ratio(type_correct, len(matches)),
            "contact_comparable": len(contact_errors),
            "contact_within_tolerance": contact_within_tolerance,
            "contact_accuracy": _safe_ratio(
                contact_within_tolerance,
                len(contact_errors),
            ),
            "contact_mean_abs_error_frames": _mean(contact_errors),
            "start_mean_abs_error_frames": _mean(start_errors),
            "end_mean_abs_error_frames": _mean(end_errors),
            "event_mean_iou": _mean(event_ious),
            "manual_review_count": len(manual_review_ids),
            "model_review_recommended_count": len(model_review_ids),
            "unmatched_model_event_count": false_positive_count,
            "unmatched_annotation_count": false_negative_count,
        },
        "review": {
            "manual_review_annotation_ids": manual_review_ids,
            "model_review_event_ids": model_review_ids,
            "false_positive_event_ids": unmatched_model_ids,
            "false_negative_annotation_ids": unmatched_annotation_ids,
            "unmatched_model_event_ids": unmatched_model_ids,
            "unmatched_annotation_ids": unmatched_annotation_ids,
            "unmatched_annotation_event_ids": unmatched_annotation_ids,
        },
        "events": event_reports,
    }


def evaluate_swing_events(
    event_data: Dict,
    annotation_data: Dict,
    contact_tolerance_frames: int = 3,
    match_contact_tolerance_frames: int = 12,
    min_event_iou: float = 0.10,
) -> Dict:
    """Evaluate V1 card annotations or V2 independent timeline truth."""
    if annotation_data.get("schema_version") == "swing_manual_annotations_v2":
        return _evaluate_v2(
            event_data,
            annotation_data,
            contact_tolerance_frames=contact_tolerance_frames,
            match_contact_tolerance_frames=match_contact_tolerance_frames,
            min_event_iou=min_event_iou,
        )
    return _evaluate_v1(
        event_data,
        annotation_data,
        contact_tolerance_frames=contact_tolerance_frames,
    )


def write_evaluation_report(
    event_json_path: str,
    annotation_json_path: str,
    output_path: Optional[str] = None,
    contact_tolerance_frames: int = 3,
    match_contact_tolerance_frames: int = 12,
    min_event_iou: float = 0.10,
) -> str:
    event_data = load_json(event_json_path)
    annotation_data = load_json(annotation_json_path)
    report = evaluate_swing_events(
        event_data,
        annotation_data,
        contact_tolerance_frames=contact_tolerance_frames,
        match_contact_tolerance_frames=match_contact_tolerance_frames,
        min_event_iou=min_event_iou,
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
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to swing_manual_annotations_v1/v2 JSON",
    )
    parser.add_argument("--output", default=None, help="Output evaluation JSON path")
    parser.add_argument("--contact-tolerance-frames", type=int, default=3)
    parser.add_argument("--match-contact-tolerance-frames", type=int, default=12)
    parser.add_argument("--min-event-iou", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_path = write_evaluation_report(
        args.events,
        args.annotations,
        output_path=args.output,
        contact_tolerance_frames=args.contact_tolerance_frames,
        match_contact_tolerance_frames=args.match_contact_tolerance_frames,
        min_event_iou=args.min_event_iou,
    )
    print(f"Wrote swing evaluation: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
