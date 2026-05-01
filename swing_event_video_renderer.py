#!/usr/bin/env python3
"""Render unified swing-event annotations onto an existing source video."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


STROKE_COLORS = {
    "Forehand": (0, 190, 255),
    "Backhand": (255, 128, 0),
    "Two-Handed Backhand": (255, 0, 255),
    "Unknown": (180, 180, 180),
}

PHASE_COLORS = {
    "ready": (130, 130, 130),
    "backswing": (255, 170, 0),
    "forward_swing": (0, 220, 255),
    "contact_candidate": (0, 255, 0),
    "follow_through": (255, 0, 180),
}

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

OVERLAY_RECORD_FIELDS = [
    "frame_id",
    "event_id",
    "event_type",
    "phase",
    "is_peak_frame",
    "event_start_frame",
    "event_end_frame",
    "peak_frame",
    "motion_energy",
    "raw_swing_type",
    "wrist_speed",
    "racket_speed",
    "ball_racket_distance",
    "contact_score",
    "two_hand_distance",
]


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_event_json(frame_json_path: str) -> str:
    path = Path(frame_json_path)
    return str(path.with_name(f"{path.stem}_swing_events.json"))


def default_output_video(frame_json_path: str) -> str:
    path = Path(frame_json_path)
    return str(path.with_name(f"{path.stem}_swing_annotated.mp4"))


def default_record_csv(frame_json_path: str) -> str:
    path = Path(frame_json_path)
    return str(path.with_name(f"{path.stem}_swing_overlay_records.csv"))


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_event_lookup(analysis: Dict) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    events = {int(event["event_id"]): event for event in analysis.get("events", [])}
    traces = {}
    for trace in analysis.get("frame_trace", []):
        frame_id = _safe_int(trace.get("frame"))
        event_id = trace.get("event_id")
        event = events.get(_safe_int(event_id)) if event_id is not None else None
        traces[frame_id] = {
            "trace": trace,
            "event": event,
        }
    return events, traces


def _draw_text(frame, text: str, pos: Tuple[int, int], scale: float, color: Tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_panel(frame, x: int, y: int, w: int, h: int, alpha: float = 0.72) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)


def _draw_ball_and_racket(frame, frame_record: Dict) -> None:
    ball = frame_record.get("ball")
    if isinstance(ball, list) and len(ball) >= 2:
        center = (int(ball[0]), int(ball[1]))
        cv2.circle(frame, center, 10, (0, 255, 255), -1)
        cv2.circle(frame, center, 13, (255, 255, 255), 2)

    for racket in frame_record.get("rackets") or []:
        box = racket.get("box") if isinstance(racket, dict) else None
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
        _draw_text(frame, "Racket", (x1, max(20, y1 - 8)), 0.45, (255, 180, 80), 1)


def _draw_pose_if_available(frame, poses) -> None:
    if not isinstance(poses, dict):
        return
    for start_key, end_key in SKELETON_EDGES:
        start = poses.get(start_key)
        end = poses.get(end_key)
        if not start or not end:
            continue
        cv2.line(frame, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (80, 220, 120), 3)
    for key, point in poses.items():
        if not point or len(point) < 2:
            continue
        color = (0, 255, 255) if "wrist" in key else (0, 220, 80)
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, color, -1)


def _event_progress(event: Optional[Dict], frame_id: int) -> float:
    if not event:
        return 0.0
    start = int(event["start_frame"])
    end = int(event["end_frame"])
    return max(0.0, min(1.0, (frame_id - start) / max(1, end - start)))


def draw_unified_overlay(frame, frame_record: Dict, trace_bundle: Optional[Dict], event_count: int) -> Dict:
    frame_id = int(frame_record.get("frame_id", 0))
    trace = (trace_bundle or {}).get("trace") or {}
    event = (trace_bundle or {}).get("event")
    event_id = event.get("event_id") if event else None
    event_type = event.get("stroke_type", "No Event") if event else "No Event"
    phase = trace.get("phase", "ready")
    peak_frame = event.get("peak_frame") if event else None
    is_peak = peak_frame is not None and frame_id == int(peak_frame)
    color = STROKE_COLORS.get(event_type, STROKE_COLORS["Unknown"])
    phase_color = PHASE_COLORS.get(phase, PHASE_COLORS["ready"])

    _draw_pose_if_available(frame, frame_record.get("pose"))
    _draw_ball_and_racket(frame, frame_record)

    panel_w = min(520, max(430, frame.shape[1] // 4))
    _draw_panel(frame, 24, 24, panel_w, 250)

    title = f"Event {event_id}/{event_count}: {event_type}" if event else "Event -/{}: No Event".format(event_count)
    _draw_text(frame, title, (44, 62), 0.72, color, 2)
    _draw_text(frame, f"Frame: {frame_id:04d} | Phase: {phase}", (44, 96), 0.52, phase_color, 1)

    if event:
        _draw_text(
            frame,
            f"Range: {event['start_frame']}-{event['end_frame']} | Peak: {event.get('peak_frame', '-')}",
            (44, 126),
            0.5,
            (230, 230, 230),
            1,
        )
    else:
        _draw_text(frame, "Range: outside swing event", (44, 126), 0.5, (180, 180, 180), 1)

    metric_lines = [
        f"raw_label: {trace.get('raw_swing_type', frame_record.get('swing_type', '-'))}",
        f"energy: {trace.get('motion_energy', '-')}  wrist: {trace.get('wrist_speed', '-')}",
        f"racket: {trace.get('racket_speed', '-')}  2Hdist: {trace.get('two_hand_distance', '-')}",
        f"contact: {trace.get('contact_score', '-')}  ball-racket: {trace.get('ball_racket_distance', '-')}",
    ]
    y = 158
    for line in metric_lines:
        _draw_text(frame, line, (44, y), 0.46, (210, 210, 210), 1)
        y += 26

    if is_peak:
        cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10), color, 8)
        _draw_text(frame, "PEAK FRAME", (frame.shape[1] - 280, 70), 0.9, color, 2)

    # Unified event progress bar.
    bar_x, bar_y = 44, frame.shape[0] - 46
    bar_w, bar_h = min(760, frame.shape[1] - 88), 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    progress = _event_progress(event, frame_id)
    if event:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), color, -1)
        peak_progress = _event_progress(event, int(peak_frame)) if peak_frame is not None else 0.0
        peak_x = bar_x + int(bar_w * peak_progress)
        cv2.line(frame, (peak_x, bar_y - 7), (peak_x, bar_y + bar_h + 7), (255, 255, 255), 2)
    _draw_text(frame, "Unified Swing Event Timeline", (bar_x, bar_y - 10), 0.5, (230, 230, 230), 1)

    return {
        "frame_id": frame_id,
        "event_id": event_id,
        "event_type": event_type,
        "phase": phase,
        "is_peak_frame": bool(is_peak),
        "event_start_frame": event.get("start_frame") if event else None,
        "event_end_frame": event.get("end_frame") if event else None,
        "peak_frame": peak_frame,
        "motion_energy": trace.get("motion_energy"),
        "raw_swing_type": trace.get("raw_swing_type", frame_record.get("swing_type")),
        "wrist_speed": trace.get("wrist_speed"),
        "racket_speed": trace.get("racket_speed"),
        "ball_racket_distance": trace.get("ball_racket_distance"),
        "contact_score": trace.get("contact_score"),
        "two_hand_distance": trace.get("two_hand_distance"),
    }


def render_event_video(
    frame_json_path: str,
    event_json_path: str,
    input_video_path: Optional[str],
    output_video_path: str,
    record_csv_path: str,
) -> Dict:
    frame_data = load_json(frame_json_path)
    analysis = load_json(event_json_path)
    frames = frame_data.get("frames", [])
    video_info = frame_data.get("video_info") or {}
    source_video = input_video_path or video_info.get("path")
    if not source_video:
        raise ValueError("Input video path is required when frame JSON has no video_info.path")
    if not os.path.exists(source_video):
        raise FileNotFoundError(source_video)

    events, frame_lookup = build_event_lookup(analysis)
    frame_records = {int(row.get("frame_id", idx)): row for idx, row in enumerate(frames)}

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {source_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or float(video_info.get("fps") or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open writer: {output_video_path}")

    Path(record_csv_path).parent.mkdir(parents=True, exist_ok=True)
    records = []
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_record = frame_records.get(frame_id, {"frame_id": frame_id})
        record = draw_unified_overlay(frame, frame_record, frame_lookup.get(frame_id), len(events))
        records.append(record)
        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()

    with open(record_csv_path, "w", encoding="utf-8", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=OVERLAY_RECORD_FIELDS)
        writer_csv.writeheader()
        for row in records:
            writer_csv.writerow({field: row.get(field) for field in OVERLAY_RECORD_FIELDS})

    return {
        "source_video": source_video,
        "output_video": output_video_path,
        "record_csv": record_csv_path,
        "frames_written": frame_id,
        "event_count": len(events),
        "event_type_counts": analysis.get("summary", {}).get("swing_event_type_counts", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render unified swing-event annotations onto video.")
    parser.add_argument("frame_json", help="Per-frame pipeline JSON, e.g. data/output_video.json")
    parser.add_argument("--event-json", help="Event analysis JSON. Defaults to <frame_json_stem>_swing_events.json")
    parser.add_argument("--input-video", help="Source video path. Defaults to frame_json video_info.path")
    parser.add_argument("--output-video", help="Annotated output video path.")
    parser.add_argument("--record-csv", help="Unified per-frame overlay record CSV path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    event_json = args.event_json or default_event_json(args.frame_json)
    output_video = args.output_video or default_output_video(args.frame_json)
    record_csv = args.record_csv or default_record_csv(args.frame_json)
    result = render_event_video(args.frame_json, event_json, args.input_video, output_video, record_csv)
    print(f"frames={result['frames_written']}")
    print(f"events={result['event_count']} {result['event_type_counts']}")
    print(f"source={result['source_video']}")
    print(f"video={result['output_video']}")
    print(f"record_csv={result['record_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
