#!/usr/bin/env python3
"""Offline compatibility adapter for rendering a completed Swing session.

Live streams use ``main_pipe.py --realtime-swing-events`` as the single
primary entrypoint. This adapter intentionally contains no Swing analysis.
"""

from __future__ import annotations

import argparse
import os
import webbrowser
from pathlib import Path
from typing import Optional

from swing_event_video_renderer import (
    default_event_json,
    default_output_video,
    default_record_csv,
    render_event_video,
)
from swing_report_builder import (
    default_coach_json,
    default_evaluation_json,
    default_report_path,
    build_report_payload,
    write_report_html,
)


def _require_file(path: str, label: str) -> str:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return str(resolved)


def render_and_build_report(
    frame_json: str,
    event_json: Optional[str] = None,
    input_video: Optional[str] = None,
    output_video: Optional[str] = None,
    record_csv: Optional[str] = None,
    output_html: Optional[str] = None,
    evaluation_json: Optional[str] = None,
) -> dict:
    """Create the annotated MP4/CSV first, then build an HTML report referencing it."""
    frame_json = _require_file(frame_json, "Frame JSON")
    event_json = _require_file(event_json or default_event_json(frame_json), "Swing event JSON")
    if input_video:
        input_video = _require_file(input_video, "Input video")

    output_video = output_video or default_output_video(frame_json)
    record_csv = record_csv or default_record_csv(frame_json)
    output_html = output_html or default_report_path(frame_json)
    coach_json = default_coach_json(frame_json)
    if not os.path.isfile(coach_json):
        coach_json = None
    if evaluation_json:
        evaluation_json = _require_file(evaluation_json, "Evaluation JSON")
    elif not os.path.isfile(default_evaluation_json(frame_json)):
        evaluation_json = None

    render_result = render_event_video(
        frame_json,
        event_json,
        input_video,
        output_video,
        record_csv,
    )
    payload = build_report_payload(
        frame_json,
        event_json,
        coach_json,
        output_video,
        evaluation_json,
    )
    write_report_html(payload, output_html)
    return {
        **render_result,
        "report_html": output_html,
        "event_json": event_json,
        "coach_json": coach_json,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Swing OSD video from frame/event JSON, then build a local HTML report."
    )
    parser.add_argument("frame_json", help="Per-frame JSON produced by main_pipe.py")
    parser.add_argument("--event-json", help="Swing event JSON; defaults to <frame_json_stem>_swing_events.json")
    parser.add_argument("--input-video", help="Source video override; defaults to frame JSON video_info.path")
    parser.add_argument("--output-video", help="Annotated MP4 output path")
    parser.add_argument("--record-csv", help="Per-frame OSD CSV output path")
    parser.add_argument("--output-html", help="HTML report output path")
    parser.add_argument("--evaluation-json", help="Optional manual evaluation JSON for the report")
    parser.add_argument("--open", action="store_true", help="Open the generated report in the default browser")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = render_and_build_report(
        args.frame_json,
        args.event_json,
        args.input_video,
        args.output_video,
        args.record_csv,
        args.output_html,
        args.evaluation_json,
    )
    print(f"video={result['output_video']}")
    print(f"record_csv={result['record_csv']}")
    print(f"html={result['report_html']}")
    if args.open:
        report_uri = Path(result["report_html"]).resolve().as_uri()
        if not webbrowser.open(report_uri):
            print(f"warning=Unable to open browser automatically; open {report_uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
