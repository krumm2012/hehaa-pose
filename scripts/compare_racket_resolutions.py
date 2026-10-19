"""Same-pixel 960/640 engineering comparison; no quality receipt is issued."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import yaml
from racket_pose_detector import RacketPoseDetector
from scripts.compare_racket_regions import references, selected, point_distance, sha, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--video-dir', default='/Users/krum5539/Desktop/Camera/2026-09-30', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / 'configs/yolo26_tennis_config.yaml').read_text())['racket_pose']
    profiles = {'960-ANE': (config['model_path'], 960, 'ANE'),
                '640-ALL': (args.candidate, 640, 'ALL'), '640-ANE': (args.candidate, 640, 'ANE')}
    models = {}
    startup = {}
    for name, (path, size, units) in profiles.items():
        start = time.perf_counter()
        model = RacketPoseDetector({**config, 'mode': 'shadow', 'model_path': str(path), 'imgsz': size, 'compute_units': units})
        loaded = time.perf_counter()
        warmup = model.detect(np.zeros((1440, 2560, 3), np.uint8), -1, None)
        if warmup['diagnostics']['nonfinite_candidates']:
            raise ValueError('Nonfinite repaired candidate: ' + name)
        startup[name] = {'load_ms': (loaded - start) * 1000, 'warmup_ms': (time.perf_counter() - loaded) * 1000,
                         'model_sha256': model.digest, 'imgsz': size, 'requested_compute_units': units}
        models[name] = model
    summary = {'schema': 'tennis.racket-resolution-comparison.v1', 'startup': startup, 'clips': [],
               'scope': 'One round, isolated racket calls, order rotated per frame. Same source pixels and frozen wrist references. No independent accuracy or whole-pipeline acceptance.'}
    names = list(models)
    for clip in sorted(args.video_dir.glob('*.mp4')):
        body, body_info = references(clip)
        cap = cv2.VideoCapture(str(clip))
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rows = []
        try:
            with (args.output / (clip.stem + '_frames.jsonl')).open('x') as journal:
                fid = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    source = {'source_frame_id': fid, 'timestamp_seconds': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000., 'basis': 'media_pts'}
                    values = {}
                    for name in names[fid % 3:] + names[:fid % 3]:
                        start = time.perf_counter()
                        obs = models[name].detect(frame, fid, source)
                        wall = (time.perf_counter() - start) * 1000
                        values[name] = {'observation': obs, 'wall_ms': wall,
                                        'selected': selected(obs, body.get(fid), fid, (frame.shape[1], frame.shape[0]), models[name].config)}
                    row = {'frame_id': fid, 'source_time': source, 'profiles': values}
                    rows.append(row)
                    journal.write(json.dumps(row, allow_nan=False) + '\n')
                    fid += 1
                    if fid % 100 == 0:
                        print(clip.name, fid, flush=True)
        finally:
            cap.release()
        if len(rows) != expected:
            raise ValueError('Incomplete source frames')
        result = {'clip': clip.name, 'source_sha256': sha(clip), 'frames': len(rows), 'body_reference': body_info, 'profiles': {}}
        for name in names:
            values = [r['profiles'][name] for r in rows]
            metric = {'predict_ms': stats([v['observation']['timing']['predict_ms'] for v in values]),
                      'wall_ms': stats([v['wall_ms'] for v in values]),
                      'nonfinite_candidates': sum(v['observation']['diagnostics']['nonfinite_candidates'] for v in values),
                      'views': {}}
            for view in ('front', 'back'):
                def good(c):
                    return c and all(c['keypoints'][n]['observed'] for n in ('grip_center', 'sweet_spot'))
                baseline = [(r, r['profiles']['960-ANE']['selected'].get(view), r['profiles'][name]['selected'].get(view)) for r in rows]
                refs = [(r, a, b) for r, a, b in baseline if good(a)]
                retained = [(r, a, b) for r, a, b in refs if good(b) and point_distance(a, b) <= 50 and point_distance(a, b, 'sweet_spot') <= 75]
                metric['views'][view] = {'reference_frames': len(refs), 'retained_frames': len(retained),
                                          'qualified_selections': sum(bool(good(b)) for _, _, b in baseline),
                                          'additional_frame_ids': [r['frame_id'] for r, a, b in baseline if not good(a) and good(b)],
                                          'not_retained_frame_ids': [r['frame_id'] for r, a, b in refs if not (good(b) and point_distance(a, b) <= 50 and point_distance(a, b, 'sweet_spot') <= 75)]}
            result['profiles'][name] = metric
        summary['clips'].append(result)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print('COMPLETE', clip.name, json.dumps({n: result['profiles'][n]['wall_ms'] for n in names}), flush=True)


if __name__ == '__main__':
    main()
