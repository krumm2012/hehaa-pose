#!/usr/bin/env python3
"""Build one model-ready evidence packet from existing Swing JSON documents."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SCHEMA_VERSION = "swing_evidence_packet_v1"
RECENT_SWING_LIMIT = 3


def build_swing_evidence_packet(
    frame_document: Dict,
    event_document: Dict,
    coach_document: Dict,
    event_id: int,
    player_context: Optional[Dict] = None,
) -> Dict:
    """Return one aligned evidence packet without reading or writing files."""
    target_event_id = int(event_id)
    event = _event_by_id(event_document.get("events"), target_event_id, "event")
    coach_event = _event_by_id(
        coach_document.get("events"),
        target_event_id,
        "coach event",
    )
    start_frame = int(event["start_frame"])
    end_frame = int(event["end_frame"])
    if start_frame > end_frame:
        raise ValueError(
            f"event_id {target_event_id} has an invalid frame range "
            f"{start_frame}-{end_frame}"
        )
    _validate_coach_range(coach_event, start_frame, end_frame, target_event_id)

    event_frame_records = _rows_in_range(
        frame_document.get("frames"),
        start_frame,
        end_frame,
        key="frame_id",
    )
    motion_features = _rows_in_range(
        event_document.get("features"),
        start_frame,
        end_frame,
        key="frame_id",
    )
    frame_trace = _rows_in_range(
        event_document.get("frame_trace"),
        start_frame,
        end_frame,
        key="frame",
        fallback_key="frame_id",
    )
    coach_metrics = deepcopy(coach_event)
    coach_metrics.pop("frame_trace", None)

    expected_ids = list(range(start_frame, end_frame + 1))
    frame_ids = _row_ids(event_frame_records, "frame_id")
    feature_ids = _row_ids(motion_features, "frame_id")
    trace_ids = _row_ids(frame_trace, "frame", fallback_key="frame_id")
    missing_frame_ids = _missing_ids(expected_ids, frame_ids)
    missing_feature_ids = _missing_ids(expected_ids, feature_ids)
    missing_trace_ids = _missing_ids(expected_ids, trace_ids)
    duplicate_frame_ids = _duplicate_ids(frame_ids)
    duplicate_feature_ids = _duplicate_ids(feature_ids)
    duplicate_trace_ids = _duplicate_ids(trace_ids)
    aligned = not any(
        (
            missing_frame_ids,
            missing_feature_ids,
            missing_trace_ids,
            duplicate_frame_ids,
            duplicate_feature_ids,
            duplicate_trace_ids,
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": target_event_id,
        "video_context": _video_context(
            frame_document,
            event_document,
            coach_document,
        ),
        "player_context": deepcopy(player_context or {}),
        "event_summary": deepcopy(event),
        "coach_metrics": coach_metrics,
        "event_frame_records": event_frame_records,
        "motion_features": motion_features,
        "frame_trace": frame_trace,
        "recent_swings": _recent_swings(
            coach_document.get("events"),
            target_event_id,
        ),
        "integrity": {
            "event_range": {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "expected_source_frame_count": len(expected_ids),
            },
            "frame_record_count": len(event_frame_records),
            "motion_feature_count": len(motion_features),
            "frame_trace_count": len(frame_trace),
            "missing_frame_record_ids": missing_frame_ids,
            "missing_motion_feature_ids": missing_feature_ids,
            "missing_frame_trace_ids": missing_trace_ids,
            "duplicate_frame_record_ids": duplicate_frame_ids,
            "duplicate_motion_feature_ids": duplicate_feature_ids,
            "duplicate_frame_trace_ids": duplicate_trace_ids,
            "aligned_to_event_range": aligned,
        },
    }


def _event_by_id(events, event_id: int, label: str) -> Dict:
    matches = [
        event
        for event in events or []
        if int(event.get("event_id", -1)) == event_id
    ]
    if not matches:
        raise ValueError(f"{label} event_id {event_id} not found")
    if len(matches) > 1:
        raise ValueError(f"{label} event_id {event_id} is duplicated")
    return matches[0]


def _validate_coach_range(
    coach_event: Dict,
    start_frame: int,
    end_frame: int,
    event_id: int,
) -> None:
    frames = coach_event.get("frames") or {}
    coach_start = frames.get("start_frame", frames.get("start"))
    coach_end = frames.get("end_frame", frames.get("end"))
    if coach_start is None or coach_end is None:
        raise ValueError(f"coach event_id {event_id} has no frame range")
    if int(coach_start) != start_frame or int(coach_end) != end_frame:
        raise ValueError(
            f"coach event_id {event_id} frame range "
            f"{coach_start}-{coach_end} does not match event range "
            f"{start_frame}-{end_frame}"
        )


def _rows_in_range(
    rows,
    start_frame: int,
    end_frame: int,
    key: str,
    fallback_key: Optional[str] = None,
) -> List[Dict]:
    selected = []
    for row in rows or []:
        frame_id = _frame_id(row, key, fallback_key)
        if frame_id is not None and start_frame <= frame_id <= end_frame:
            selected.append((frame_id, deepcopy(row)))
    selected.sort(key=lambda item: item[0])
    return [row for _, row in selected]


def _frame_id(
    row: Dict,
    key: str,
    fallback_key: Optional[str] = None,
) -> Optional[int]:
    value = row.get(key)
    if value is None and fallback_key:
        value = row.get(fallback_key)
    if value is None:
        return None
    return int(value)


def _row_ids(
    rows: Iterable[Dict],
    key: str,
    fallback_key: Optional[str] = None,
) -> List[int]:
    return [
        frame_id
        for frame_id in (
            _frame_id(row, key, fallback_key)
            for row in rows
        )
        if frame_id is not None
    ]


def _missing_ids(expected_ids: Iterable[int], actual_ids: Iterable[int]) -> List[int]:
    actual = set(actual_ids)
    return [frame_id for frame_id in expected_ids if frame_id not in actual]


def _duplicate_ids(frame_ids: Iterable[int]) -> List[int]:
    counts = Counter(frame_ids)
    return sorted(frame_id for frame_id, count in counts.items() if count > 1)


def _video_context(
    frame_document: Dict,
    event_document: Dict,
    coach_document: Dict,
) -> Dict:
    video_info = frame_document.get("video_info") or {}
    metadata = coach_document.get("metadata") or {}
    thresholds = (event_document.get("summary") or {}).get("thresholds") or {}
    frames = frame_document.get("frames") or []
    total_frames = (
        (frame_document.get("summary") or {}).get("total_frames")
        or metadata.get("total_frames")
        or len(frames)
    )
    return {
        "video_path": video_info.get("path") or metadata.get("video_path"),
        "fps": video_info.get("fps") or metadata.get("fps"),
        "resolution": (
            video_info.get("resolution")
            or metadata.get("resolution")
        ),
        "total_frames": int(total_frames),
        "dominant_hand": thresholds.get("dominant_hand"),
        "coordinate_space": "screen_pixels",
        "timestamp_unit": "seconds",
        "distance_unit": "pixels",
        "motion_feature_speed_unit": "pixels_per_processed_record",
    }


def _recent_swings(events, target_event_id: int) -> List[Dict]:
    previous = sorted(
        (
            event
            for event in events or []
            if int(event.get("event_id", -1)) < target_event_id
        ),
        key=lambda event: int(event.get("event_id", -1)),
    )[-RECENT_SWING_LIMIT:]
    return [
        {
            "event_id": int(event["event_id"]),
            "stroke_type": event.get("stroke_type"),
            "confidence": event.get("confidence"),
            "scores": deepcopy(event.get("scores") or {}),
            "timing": deepcopy(event.get("timing") or {}),
            "diagnosis_tags": deepcopy(event.get("diagnosis_tags") or []),
            "quality_warnings": deepcopy(
                (event.get("quality_flags") or {}).get("warnings") or []
            ),
        }
        for event in previous
    ]


def default_event_json(frame_json: str) -> str:
    path = Path(frame_json)
    return str(path.with_name(f"{path.stem}_swing_events.json"))


def default_coach_json(frame_json: str) -> str:
    path = Path(frame_json)
    return str(path.with_name(f"{path.stem}_coach_dataset.json"))


def default_output_json(frame_json: str, event_id: int) -> str:
    path = Path(frame_json)
    return str(path.with_name(f"{path.stem}_swing_{int(event_id)}_evidence.json"))


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return document


def _write_json(document: Dict, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, output)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate existing frame, Swing event, and Coach JSON for one event."
    )
    parser.add_argument("frame_json", help="Per-frame JSON produced by main_pipe.py")
    parser.add_argument("--event-json", help="Defaults to <frame_stem>_swing_events.json")
    parser.add_argument("--coach-json", help="Defaults to <frame_stem>_coach_dataset.json")
    parser.add_argument("--event-id", type=int, required=True, help="Swing event ID")
    parser.add_argument("--player-context-json", help="Optional player profile JSON object")
    parser.add_argument("--output-json", help="Aggregated SwingEvidencePacket output")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    event_json = args.event_json or default_event_json(args.frame_json)
    coach_json = args.coach_json or default_coach_json(args.frame_json)
    output_json = args.output_json or default_output_json(
        args.frame_json,
        args.event_id,
    )
    player_context = (
        _load_json(args.player_context_json)
        if args.player_context_json
        else None
    )
    packet = build_swing_evidence_packet(
        _load_json(args.frame_json),
        _load_json(event_json),
        _load_json(coach_json),
        event_id=args.event_id,
        player_context=player_context,
    )
    _write_json(packet, output_json)
    integrity = packet["integrity"]
    print(
        f"event={packet['event_id']}"
        f" frames={integrity['frame_record_count']}"
        f" features={integrity['motion_feature_count']}"
        f" trace={integrity['frame_trace_count']}"
        f" aligned={str(integrity['aligned_to_event_range']).lower()}"
    )
    print(f"json={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
