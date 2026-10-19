"""Compare source crops sequentially, without changing production defaults.

All decoded frames run full, expanded union and two expanded source views.
Existing raw body observations provide identical wrist gates for every policy;
they are engineering references, never human ground truth. No body model runs
here, so measured inference throughput must not be called pipeline FPS.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml

from racket_pose_detector import RacketPoseDetector, select_pose_candidate
from racket_pose_regions import POLICY_VERSION, configured_regions, restore_source_observation
from roi_stream_config import resolve_roi_stream_profile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stats(values):
    return {k: float(np.percentile(values, q)) for k, q in (('p50', 50), ('p95', 95))} if values else {}


def references(clip):
    acceptance = ROOT / 'data/analysis_results/racket_pose_acceptance_20261018'
    candidates = list(acceptance.glob(f'roi-bound-20261019/baseline/*-{clip.stem}-r0/frames.jsonl'))
    candidates += list(acceptance.glob(f'interframe-20261018/baseline/*-{clip.stem}-r0/frames.jsonl'))
    special = {'49.43': 'live_session_20261019T020222Z_aeab89',
               '49.33': 'live_session_20261019T022749Z_698b03'}
    if clip.stem in special:
        candidates.insert(0, ROOT / 'data/analysis_results/control_panel' / special[clip.stem] / 'live_session_frames.jsonl')
    candidates += list(acceptance.glob(f'dual-comparison/realtime-active-s3-{clip.stem}-r0/frames.jsonl'))
    for path in candidates:
        if not path.is_file():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if not rows or not rows[0].get('pose_observations'):
            continue
        manifest_path = path.parent / ('live_session_evidence_manifest.json' if path.name.startswith('live_session') else 'evidence_manifest.json')
        manifest = json.loads(manifest_path.read_text())
        source = Path(manifest['capture']['source'])
        if not source.is_file() or sha(source) != sha(clip):
            raise ValueError('Body reference source does not match ' + str(clip))
        return {r['frame_id']: r for r in rows}, {'path': str(path), 'sha256': sha(path), 'frames': len(rows)}
    raise ValueError('No frozen raw body reference for ' + str(clip))


def merge_observations(parts, threshold, iou):
    observation = copy.deepcopy(parts[0])
    candidates = [c for p in parts for c in p['candidates']]
    if len(parts) > 1 and candidates:
        boxes = [[c['box'][0], c['box'][1], c['box'][2] - c['box'][0], c['box'][3] - c['box'][1]] for c in candidates]
        ids = cv2.dnn.NMSBoxes(boxes, [c['confidence'] for c in candidates], threshold, iou)
        candidates = [candidates[int(i)] for i in np.asarray(ids).reshape(-1)]
    observation['candidates'] = candidates
    observation['diagnostics'] = {key: sum(p['diagnostics'][key] for p in parts)
                                  for key in ('candidates', 'nonfinite_candidates', 'nonfinite_values')}
    observation['status'] = 'numerical_failure' if observation['diagnostics']['nonfinite_candidates'] else 'observed'
    observation['input_regions'] = [p['input_region'] for p in parts]
    observation.pop('input_region', None)
    return observation


def selected(observation, body, frame_id, size, config):
    result = {}
    for view, points in (body or {}).get('pose_observations', {}).items():
        wrists = [p for name, p in points.items() if 'wrist' in name]
        result[view] = select_pose_candidate(observation, frame_id, wrists, size, config)
    return result


def point_distance(a, b, name='grip_center'):
    p, q = a['keypoints'][name], b['keypoints'][name]
    return float(np.hypot(p['x'] - q['x'], p['y'] - q['y']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--video-dir', default='/Users/krum5539/Desktop/Camera/2026-09-30')
    parser.add_argument('--clips', nargs='+', default=['48.43.mp4', '49.43.mp4', '50.03.mp4', '49.33.mp4',
                                                     '48.53.mp4', '49.03.mp4', '49.13.mp4', '49.23.mp4', '49.53.mp4'])
    parser.add_argument('--stream-id', required=True)
    parser.add_argument('--margin', type=int, default=256)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / 'configs/yolo26_tennis_config.yaml').read_text())
    detector = RacketPoseDetector({**config['racket_pose'], 'mode': 'exclusive', 'processing_mode': 'offline'})
    warmup = detector.detect(np.zeros((1440, 2560, 3), np.uint8), -1, None)
    if warmup['diagnostics']['nonfinite_candidates']:
        raise ValueError('Nonfinite warmup')
    summary = {'schema': 'tennis.racket-region-comparison.v1', 'policy_version': POLICY_VERSION,
               'scope': 'Isolated sequential racket inference; no body/ball/render/IPC load. Model agreement and geometric coverage are not ground truth or realtime pipeline acceptance.',
               'margin_source_pixels': args.margin, 'model_sha256': detector.digest,
               'model_config': detector.config, 'implementation_sha256': {n: sha(ROOT / n) for n in
                    ('racket_pose_regions.py', 'racket_pose_detector.py', 'scripts/compare_racket_regions.py')},
               'reference_matching': 'same frozen qualified wrists; retained if grip <=50px and sweet_spot <=75px from full reference; both points must be observed',
               'clips': []}
    (out / 'run_config.json').write_text(json.dumps(summary, indent=2))
    policies = ['full', 'union', 'split']
    for filename in args.clips:
        clip = Path(args.video_dir) / filename
        body, body_info = references(clip)
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            raise ValueError('Cannot open ' + str(clip))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        profile = resolve_roi_stream_profile(config, str(clip), (width, height), target_stream_id=args.stream_id)
        if not profile.enabled or profile.stream_id != args.stream_id:
            raise ValueError('Explicit camera ROI did not resolve')
        regions = {p: configured_regions((width, height), profile.front_view, profile.mirror_view, p, args.margin) for p in policies}
        report = {'clip': filename, 'source_sha256': sha(clip), 'roi': profile.as_metadata(),
                  'body_reference': body_info, 'policies': {}, 'frames': 0}
        records = []
        started = time.perf_counter()
        try:
            with (out / (clip.stem + '_frames.jsonl')).open('x') as journal:
                fid = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    source_time = {'source_frame_id': fid, 'timestamp_seconds': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000., 'basis': 'media_pts'}
                    views = {}
                    order = policies[fid % 3:] + policies[:fid % 3]
                    for policy in order:
                        parts = []
                        start = time.perf_counter()
                        for region in regions[policy]:
                            raw = detector.detect(region.extract(frame), fid, source_time)
                            restored = restore_source_observation(raw, region, (width, height))
                            parts.append(restored)
                        observation = merge_observations(parts, detector.config['confidence_threshold'], detector.config['iou_threshold'])
                        wall = (time.perf_counter() - start) * 1000
                        views[policy] = {'observation': observation, 'selected': selected(observation, body.get(fid), fid, (width, height), detector.config),
                                         'predict_ms': sum(p['timing']['predict_ms'] for p in parts),
                                         'preprocess_ms': sum(p['timing']['preprocess_ms'] for p in parts), 'wall_ms': wall}
                    record = {'frame_id': fid, 'source_time': source_time, 'has_body_reference': fid in body, 'policies': views}
                    journal.write(json.dumps(record, allow_nan=False) + '\n')
                    records.append(record)
                    fid += 1
                    if fid % 50 == 0:
                        print(filename, fid, 'frames', round(time.perf_counter() - started, 1), 'seconds', flush=True)
        finally:
            cap.release()
        expected = int(round(cv2.VideoCapture(str(clip)).get(cv2.CAP_PROP_FRAME_COUNT)))
        if len(records) != expected:
            raise ValueError('Incomplete decoded video')
        report['frames'] = len(records)
        report['frames_without_body_reference'] = [r['frame_id'] for r in records if not r['has_body_reference']]
        for policy in policies:
            values = [r['policies'][policy] for r in records]
            metrics = {'regions': [{'name': r.name, 'box': list(r.box), 'size': list(r.size)} for r in regions[policy]],
                       'model_calls_per_frame': len(regions[policy]), 'predict_ms': stats([v['predict_ms'] for v in values]),
                       'preprocess_ms': stats([v['preprocess_ms'] for v in values]), 'wall_ms': stats([v['wall_ms'] for v in values]),
                       'nonfinite_candidates': sum(v['observation']['diagnostics']['nonfinite_candidates'] for v in values),
                       'candidate_frames': sum(bool(v['observation']['candidates']) for v in values), 'views': {}}
            for view in ('front', 'back'):
                eligible = [(r, r['policies']['full']['selected'].get(view), r['policies'][policy]['selected'].get(view)) for r in records]
                def qualified(c):
                    return c is not None and all(c['keypoints'][n]['observed'] for n in ('grip_center', 'sweet_spot'))
                refs = [(r, f, c) for r, f, c in eligible if qualified(f)]
                matched = [(r, f, c) for r, f, c in refs if qualified(c) and point_distance(f, c) <= 50 and point_distance(f, c, 'sweet_spot') <= 75]
                lost = [r['frame_id'] for r, f, c in refs if not (qualified(c) and point_distance(f, c) <= 50 and point_distance(f, c, 'sweet_spot') <= 75)]
                pts = [p for _, f, _ in refs for p in f['keypoints'].values() if p['observed']]
                outside = sum(not any(reg.box[0] <= p['x'] < reg.box[2] and reg.box[1] <= p['y'] < reg.box[3] for reg in regions[policy]) for p in pts)
                metrics['views'][view] = {'full_qualified_reference_frames': len(refs), 'retained_reference_frames': len(matched),
                                         'not_retained_frame_ids': lost, 'qualified_selection_frames': sum(qualified(c) for _, _, c in eligible),
                                         'additional_selection_frame_ids': [r['frame_id'] for r, f, c in eligible if not qualified(f) and qualified(c)],
                                         'full_reference_points': len(pts), 'full_reference_points_outside_crop': outside,
                                         'grip_displacement_px': stats([point_distance(f, c) for _, f, c in matched])}
            report['policies'][policy] = metrics
        summary['clips'].append(report)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print('COMPLETE', filename, json.dumps({p: {'p50_ms': report['policies'][p]['wall_ms']['p50'],
              'front_retained': report['policies'][p]['views']['front']['retained_reference_frames'],
              'back_retained': report['policies'][p]['views']['back']['retained_reference_frames']} for p in policies}), flush=True)


if __name__ == '__main__':
    main()
