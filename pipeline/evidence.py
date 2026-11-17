"""Evidence manifest generation for replayable analysis sessions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from analysis_data_contracts import SOURCE_TIME_SCHEMA_VERSION
from motion_time_contract import POLICY_VERSION as MOTION_TIME_POLICY, REFERENCE_HZ
from session_evidence_bundle import write_session_evidence_manifest


def write_evidence_manifest(pipeline, status: str) -> Optional[str]:
    if not pipeline.evidence_manifest:
        return None
    output_base = str(pipeline.config.get('video_output_path') or 'output_video.mp4')
    output_stem = os.path.splitext(output_base)[0]
    frame_jsonl = pipeline.realtime_frame_jsonl or f'{output_stem}_frames.jsonl'
    frame_snapshot = (
        pipeline.realtime_frame_snapshot_json
        or f'{output_stem}_frames_latest.json'
    )
    event_json = pipeline.realtime_swing_json or f'{output_stem}_swing_events.json'
    event_log = pipeline.realtime_swing_event_log or str(
        Path(event_json).with_suffix('.jsonl')
    )
    event_html = pipeline.realtime_swing_html or f'{output_stem}_swing_report.html'
    clips_dir = Path(
        pipeline.realtime_swing_clips_dir or f'{output_stem}_swing_clips'
    )
    artifacts = [
        {
            'role': 'frame_journal',
            'path': frame_jsonl,
            'required_for_replay': True,
        },
        {
            'role': 'event_snapshot',
            'path': event_json,
            'required_for_replay': True,
        },
        {'role': 'frame_snapshot', 'path': frame_snapshot},
        {'role': 'event_log', 'path': event_log},
        {'role': 'report', 'path': event_html},
    ]
    if pipeline.racket_pose_journal:
        for role, path in (
            ('racket_pose_journal', Path(pipeline.racket_pose_journal)),
            ('racket_pose_summary', Path(pipeline.racket_pose_journal).with_suffix('.summary.json')),
        ):
            if path.is_file(): artifacts.append({'role': role, 'path': str(path)})
    display_journal = Path(str(Path(pipeline.config['video_output_path']).with_suffix(''))+'_display.jsonl')
    if display_journal.is_file():artifacts.append({'role':'display_completion_journal','path':str(display_journal)})
    if bool(pipeline.config.get('save_video', True)):
        artifacts.append({'role': 'annotated_video', 'path': output_base})
    if not pipeline.is_stream_source and Path(pipeline.video_path).expanduser().is_file():
        artifacts.append({'role': 'source_video', 'path': pipeline.video_path})
    if clips_dir.is_dir():
        artifacts.extend(
            {'role': 'swing_clip', 'path': str(path)}
            for path in sorted(clips_dir.glob('*.mp4'))
        )
        artifacts.extend(
            {'role': 'impact_freeze', 'path': str(path)}
            for path in sorted(clips_dir.glob('*_impact_freeze.jpg'))
        )
    roi_preview = Path(f'{output_stem}_swing_report_roi_preview.jpg')
    if roi_preview.is_file():
        artifacts.append({'role': 'roi_preview', 'path': str(roi_preview)})
    coach_tts_dir = Path(f'{output_stem}_coach_audio')
    if coach_tts_dir.is_dir():
        artifacts.extend(
            {'role': 'coach_tts_audio', 'path': str(path)}
            for path in sorted(coach_tts_dir.glob('*.wav'))
        )
    effective_settle = (
        pipeline.realtime_settle_frames
        if pipeline.realtime_settle_frames is not None
        else max(0, int(round(REFERENCE_HZ * 0.6)))
    )
    effective_window = (
        pipeline.realtime_window_frames
        if pipeline.realtime_window_frames is not None
        else max(32, int(round(REFERENCE_HZ * 8.0)))
    )
    replay = {
        'engine': 'RealtimeSwingEventEngine',
        'fps': pipeline.fps,
        'analysis_interval_frames': pipeline.realtime_analysis_interval,
        'settle_frames': effective_settle,
        'window_frames': effective_window,
        'window_seconds': 8.,
        'motion_time_policy': MOTION_TIME_POLICY,
        'swing_options': dict(pipeline.swing_analysis_options),
        'coach': {
            'enabled': pipeline.realtime_coach,
            'max_chars': pipeline.realtime_coach_max_chars,
            'max_suggestions': pipeline.realtime_coach_max_suggestions,
            'min_confidence': pipeline.realtime_coach_min_confidence,
            'thresholds': dict(pipeline.realtime_coach_biomechanics),
        },
    }
    capture = {
        'processing_mode': pipeline.processing_mode,
        'inference_schedule': pipeline.inference_schedule,
        'racket_ipc': pipeline.racket_pose_config['ipc_transport'],
        'racket_pose_mode': pipeline.racket_pose_config['mode'],
        'racket_selection_policy': pipeline.racket_pose_config.get('selection_policy','mixed'),
        'racket_selection_policy_version': pipeline.racket_pose_config.get('selection_policy_version'),
        'source_kind': 'stream' if pipeline.is_stream_source else 'video_file',
        'source_time_schema_version': SOURCE_TIME_SCHEMA_VERSION,
        'source_time_policy': 'input_media_pts_or_explicit_fps_estimate; stream_device_time_unavailable',
        'source': pipeline.session_metadata.get('source'),
        'fps': pipeline.fps,
        'output_fps': pipeline.output_fps,
        'resolution': [pipeline.width, pipeline.height],
        'drop_stale_frames': pipeline.drop_stale_frames,
        'startup_readiness_barrier': pipeline.lossless_realtime,
        'continuous_frame_scope': 'all decoded frames; upstream camera/network losses cannot be verified',
        'dropped_stale_frames': int(pipeline.dropped_stale_frames.value),
        'max_frames': pipeline.max_frames,
        'roi': pipeline.roi_profile.as_metadata(),
    }
    path = write_session_evidence_manifest(
        pipeline.evidence_manifest,
        session=pipeline.session_metadata,
        status=status,
        capture=capture,
        replay=replay,
        artifacts=artifacts,
    )
    print(f'📦 [Evidence] 可重放证据清单: {path}')
    return path

