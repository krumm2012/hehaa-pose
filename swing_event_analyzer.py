#!/usr/bin/env python3
"""Offline event-level swing analysis from existing frame JSON output."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from analysis_data_contracts import (
    document_contract,
    inferred_session,
    stamp_swing_event,
)
from swing_biomechanics import enrich_events_with_biomechanics
from swing_coach_calibration import calibrate_coaching_event
from swing_event_segmenter import segment_swing_events
from swing_motion_features import extract_motion_features
from swing_session_quality import build_session_quality_dashboard


FRAME_FIELDS = [
    "frame_id",
    "timestamp",
    "raw_swing_type",
    "event_id",
    "phase",
    "motion_energy",
    "wrist_speed",
    "wrist_accel",
    "racket_speed",
    "racket_accel",
    "ball_speed",
    "ball_racket_distance",
    "contact_score",
    "two_hand_distance",
    "two_hand_distance_body_width",
    "active_wrist_x_offset",
    "active_wrist_x_offset_body_width",
    "camera_facing_score",
    "shoulder_width_px",
    "arm_extension_deg",
    "shoulder_turn_deg",
    "hip_shoulder_sep_deg",
]


EVENT_FIELDS = [
    "event_id",
    "start_frame",
    "end_frame",
    "duration_frames",
    "peak_frame",
    "contact_frame",
    "peak_energy",
    "stroke_type",
    "confidence",
    "quality_flags",
    "label_counts",
    "two_hand_ratio",
    "label_two_hand_ratio",
    "backhand_side_frames",
    "forehand_side_frames",
    "player_dominant_hand",
    "camera_view",
    "camera_confidence",
    "swing_side",
    "swing_side_confidence",
    "classification_rule",
    "classification_context",
    "start_boundary_mode",
    "start_boundary_confidence",
    "start_boundary_evidence",
    "phase_counts",
]


def load_frame_records(input_path: str) -> List[Dict]:
    """Load frame records from a pipeline JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("frames"), list):
        return data["frames"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported frame JSON format: {input_path}")


def _frame_trace_by_id(frame_trace: Iterable[Dict]) -> Dict[int, Dict]:
    return {int(row["frame"]): row for row in frame_trace}


def _json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _event_csv_rows(events: List[Dict]) -> List[Dict]:
    rows = []
    for event in events:
        evidence = event.get("evidence") or {}
        start_boundary = evidence.get("start_boundary") or {}
        classification_context = evidence.get("classification_context") or {}
        player_context = classification_context.get("player") or {}
        camera_context = classification_context.get("camera") or {}
        swing_context = classification_context.get("swing") or {}
        rows.append(
            {
                "event_id": event.get("event_id"),
                "start_frame": event.get("start_frame"),
                "end_frame": event.get("end_frame"),
                "duration_frames": event.get("duration_frames"),
                "peak_frame": event.get("peak_frame"),
                "contact_frame": event.get("contact_frame"),
                "peak_energy": event.get("peak_energy"),
                "stroke_type": event.get("stroke_type"),
                "confidence": event.get("confidence"),
                "quality_flags": _json_text(event.get("quality_flags") or {}),
                "label_counts": _json_text(evidence.get("label_counts") or {}),
                "two_hand_ratio": evidence.get("two_hand_ratio"),
                "label_two_hand_ratio": evidence.get("label_two_hand_ratio"),
                "backhand_side_frames": evidence.get("backhand_side_frames"),
                "forehand_side_frames": evidence.get("forehand_side_frames"),
                "player_dominant_hand": player_context.get("dominant_hand"),
                "camera_view": camera_context.get("view"),
                "camera_confidence": camera_context.get("confidence"),
                "swing_side": swing_context.get("side"),
                "swing_side_confidence": swing_context.get("confidence"),
                "classification_rule": classification_context.get("decision_rule"),
                "classification_context": _json_text(classification_context),
                "start_boundary_mode": start_boundary.get("mode"),
                "start_boundary_confidence": start_boundary.get("confidence"),
                "start_boundary_evidence": _json_text(start_boundary),
                "phase_counts": _json_text(event.get("phase_counts") or {}),
            }
        )
    return rows


def _frame_csv_rows(features: List[Dict], frame_trace: List[Dict]) -> List[Dict]:
    trace_by_id = _frame_trace_by_id(frame_trace)
    rows = []
    for feature in features:
        frame_id = int(feature["frame_id"])
        trace = trace_by_id.get(frame_id, {})
        rows.append(
            {
                "frame_id": frame_id,
                "timestamp": feature.get("timestamp"),
                "raw_swing_type": feature.get("raw_swing_type"),
                "event_id": trace.get("event_id"),
                "phase": trace.get("phase"),
                "motion_energy": trace.get("motion_energy"),
                "wrist_speed": feature.get("wrist_speed"),
                "wrist_accel": feature.get("wrist_accel"),
                "racket_speed": feature.get("racket_speed"),
                "racket_accel": feature.get("racket_accel"),
                "ball_speed": feature.get("ball_speed"),
                "ball_racket_distance": feature.get("ball_racket_distance"),
                "contact_score": feature.get("contact_score"),
                "two_hand_distance": feature.get("two_hand_distance"),
                "two_hand_distance_body_width": feature.get("two_hand_distance_body_width"),
                "active_wrist_x_offset": feature.get("active_wrist_x_offset"),
                "active_wrist_x_offset_body_width": feature.get("active_wrist_x_offset_body_width"),
                "camera_facing_score": feature.get("camera_facing_score"),
                "shoulder_width_px": feature.get("shoulder_width_px"),
                "arm_extension_deg": feature.get("arm_extension_deg"),
                "shoulder_turn_deg": feature.get("shoulder_turn_deg"),
                "hip_shoulder_sep_deg": feature.get("hip_shoulder_sep_deg"),
            }
        )
    return rows


def analyze_frame_records(
    frames: List[Dict],
    dominant_hand: str = "right",
    min_peak_energy: float = 9.0,
    active_energy: float = 5.5,
    min_event_frames: int = 8,
    max_internal_gap: int = 3,
    min_event_gap: int = 18,
    session_metadata: Optional[Dict] = None,
) -> Dict:
    """Build event-level analysis and auditable frame features."""
    features = extract_motion_features(frames, dominant_hand=dominant_hand)
    segmentation = segment_swing_events(
        features,
        min_peak_energy=min_peak_energy,
        active_energy=active_energy,
        min_event_frames=min_event_frames,
        max_internal_gap=max_internal_gap,
        min_event_gap=min_event_gap,
    )
    events = enrich_events_with_biomechanics(
        segmentation["events"],
        frames,
        features,
    )
    for event in events:
        event["coach_calibration"] = calibrate_coaching_event(event)
    effective_session = dict(session_metadata or inferred_session(frames))
    frames_by_id = {
        int(frame["frame_id"]): frame
        for frame in frames
        if isinstance(frame, dict) and frame.get("frame_id") is not None
    }
    emitted_at_unix_ns = time.time_ns()
    for event in events:
        stamp_swing_event(
            event,
            session=effective_session,
            emitted_at_unix_ns=emitted_at_unix_ns,
            contact_frame_record=frames_by_id.get(int(event["contact_frame"])),
        )
    type_counts = Counter(event["stroke_type"] for event in events)

    thresholds = {
        "dominant_hand": dominant_hand,
        "min_peak_energy": min_peak_energy,
        "active_energy": active_energy,
        "min_event_frames": min_event_frames,
        "max_internal_gap": max_internal_gap,
        "min_event_gap": min_event_gap,
    }
    return {
        **document_contract("swing_events", effective_session),
        "summary": {
            "total_frames": len(frames),
            "swing_event_count": len(events),
            "swing_event_type_counts": dict(sorted(type_counts.items())),
            "thresholds": thresholds,
            "session_quality": build_session_quality_dashboard(events),
        },
        "events": events,
        "frame_trace": segmentation["frame_trace"],
        "features": features,
    }


def write_analysis_outputs(analysis: Dict, output_json: str, events_csv: Optional[str] = None, frames_csv: Optional[str] = None) -> None:
    """Write JSON plus optional CSV audit files."""
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    if events_csv:
        _write_csv(Path(events_csv), _event_csv_rows(analysis["events"]), EVENT_FIELDS)
    if frames_csv:
        _write_csv(Path(frames_csv), _frame_csv_rows(analysis["features"], analysis["frame_trace"]), FRAME_FIELDS)


def default_output_paths(input_path: str, output_dir: Optional[str] = None) -> Dict[str, str]:
    stem = Path(input_path).stem
    directory = Path(output_dir) if output_dir else Path(input_path).parent
    return {
        "json": str(directory / f"{stem}_swing_events.json"),
        "events_csv": str(directory / f"{stem}_swing_events.csv"),
        "frames_csv": str(directory / f"{stem}_swing_frames.csv"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze precise swing events from existing per-frame JSON.")
    parser.add_argument("input_json", help="Path to JSON produced by the video/frame pipeline.")
    parser.add_argument("--output-json", help="Output JSON path. Defaults next to input.")
    parser.add_argument("--events-csv", help="Output event summary CSV path. Defaults next to input.")
    parser.add_argument("--frames-csv", help="Output frame audit CSV path. Defaults next to input.")
    parser.add_argument("--output-dir", help="Directory for default outputs.")
    parser.add_argument("--dominant-hand", choices=["right", "left"], default="right")
    parser.add_argument("--min-peak-energy", type=float, default=9.0)
    parser.add_argument("--active-energy", type=float, default=5.5)
    parser.add_argument("--min-event-frames", type=int, default=8)
    parser.add_argument("--max-internal-gap", type=int, default=3)
    parser.add_argument("--min-event-gap", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = default_output_paths(args.input_json, args.output_dir)
    output_json = args.output_json or paths["json"]
    events_csv = args.events_csv or paths["events_csv"]
    frames_csv = args.frames_csv or paths["frames_csv"]

    frames = load_frame_records(args.input_json)
    analysis = analyze_frame_records(
        frames,
        dominant_hand=args.dominant_hand,
        min_peak_energy=args.min_peak_energy,
        active_energy=args.active_energy,
        min_event_frames=args.min_event_frames,
        max_internal_gap=args.max_internal_gap,
        min_event_gap=args.min_event_gap,
    )
    write_analysis_outputs(analysis, output_json, events_csv, frames_csv)

    summary = analysis["summary"]
    print(f"frames={summary['total_frames']}")
    print(f"events={summary['swing_event_count']} {summary['swing_event_type_counts']}")
    print(f"json={output_json}")
    print(f"events_csv={events_csv}")
    print(f"frames_csv={frames_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
