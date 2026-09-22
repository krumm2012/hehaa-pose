#!/usr/bin/env python3
"""Replay identical detections with mirror filtering off/shadow/enforce.

Counts are region proxies, not labeled precision/recall. Source videos are read-only.
Run from the repo with its Python environment; --cache skips model inference.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from ball_track_selector import BallTrackSelector


def collect(video, config):
    import cv2
    from yolo26n_unified_detector import YOLO26nUnifiedDetector
    detector = YOLO26nUnifiedDetector(str(ROOT / config['model_path']), config)
    original = detector._filter_static_balls
    raw = {}

    def capture(balls, rackets=None):
        raw.update(balls=copy.deepcopy(balls), rackets=copy.deepcopy(rackets))
        return original(balls, rackets)

    detector._filter_static_balls = capture
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {video}')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            balls, _, _ = detector.detect_unified(frame)
            rows.append(dict(frame=len(rows), time=cap.get(cv2.CAP_PROP_POS_MSEC)/1000,
                             **copy.deepcopy(raw), baseline=balls[0] if balls else None))
            if len(rows) % 50 == 0:
                print(f'Collected {len(rows)} frames', flush=True)
    finally:
        cap.release()
    if not rows:
        raise ValueError('No decoded frames')
    return dict(video=str(video.resolve()), width=width, height=height, config=config, frames=rows)


def replay(data, calibration):
    selectors = {}
    for mode in ('off', 'shadow', 'enforce'):
        cfg = copy.deepcopy(data['config'])
        cfg['mirror_ball_filter'] = {**calibration, 'mode': mode}
        selectors[mode] = BallTrackSelector(cfg)
    rows = []
    metrics = dict(frames=len(data['frames']), region_counts_are_not_labeled_accuracy=True,
                   shadow_selection_differences=0, cached_baseline_differences=0,
                   rejected_candidates=0, rejected_below_mirror_candidates=0,
                   below_mirror_candidate_frames=0)
    bottom = max(p[1] for p in calibration['polygon']) * data['height']
    for mode in selectors:
        metrics[mode] = dict(selected_frames=0, below_mirror_selected_frames=0)
    for row in data['frames']:
        result = dict(frame=row['frame'], time=row['time'])
        below = [b for b in row['balls'] if b['position'][1] > bottom]
        metrics['below_mirror_candidate_frames'] += bool(below)
        for mode, selector in selectors.items():
            selected = selector.select(row['balls'], row.get('rackets'), frame_height=data['height'], frame_width=data['width'])
            result[mode] = selected.active_ball
            result[mode + '_diagnostics'] = selected.diagnostics
            metrics[mode]['selected_frames'] += selected.active_ball is not None
            metrics[mode]['below_mirror_selected_frames'] += bool(selected.active_ball and selected.active_ball['position'][1] > bottom)
        metrics['shadow_selection_differences'] += result['off'] != result['shadow']
        if 'baseline' in row:
            metrics['cached_baseline_differences'] += result['off'] != row['baseline']
        for evidence in result['enforce_diagnostics']['mirror_filter']['candidates']:
            metrics['rejected_candidates'] += evidence['rejected']
            metrics['rejected_below_mirror_candidates'] += evidence['rejected'] and evidence['position'][1] > bottom
        rows.append(result)
    return rows, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/yolo26_tennis_config.yaml')
    parser.add_argument('--calibration', type=Path, default=ROOT/'configs/camera04_mirror_filter.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.cache and not args.video:
        parser.error('--video or --cache is required')
    args.output.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load(args.config.read_text())
    calibration = yaml.safe_load(args.calibration.read_text())['unified_detection']['mirror_ball_filter']
    if args.cache:
        data = json.loads(args.cache.read_text())
    else:
        cfg = copy.deepcopy(base['unified_detection'])
        cfg.pop('mirror_ball_filter', None)
        data = collect(args.video, cfg)
    rows, metrics = replay(data, calibration)
    for name, value in [('detections.json', data), ('comparison.json', rows), ('metrics.json', metrics)]:
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2))
    for mode in ('shadow', 'enforce'):
        cfg = copy.deepcopy(base)
        cfg['unified_detection'] = {**data['config'], 'mirror_ball_filter': {**calibration, 'mode': mode}}
        if args.video:
            cfg['video_input_path'] = str(args.video.resolve())
        cfg['video_output_path'] = str((args.output/f'{mode}-pipeline.mp4').resolve())
        (args.output/f'{mode}-config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
