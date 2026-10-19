"""Replay fixed rendering inputs without model calls; keep wall and thread CPU time."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from dual_view_manager import DualViewManager
from dual_view_renderer import DualViewRenderer
from dual_pose_estimator import DualPoseEstimator
from realtime_swing_runtime import build_impact_telemetry_card


def dist(values):
    return {key: float(np.percentile(values, percentile)) for key, percentile in [('p50', 50), ('p95', 95), ('p99', 99)]} | {'max': max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--count', type=int, default=120)
    parser.add_argument('--events', type=Path)
    args = parser.parse_args()
    if args.count < 20:
        parser.error('at least twenty samples required')
    cap = cv2.VideoCapture('/Users/krum5539/Desktop/Camera/2026-09-30/49.43.mp4')
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError('Source video unavailable')
    manager = DualViewManager()
    dual = manager.split_frame(frame, frame_id=0)
    estimator = DualPoseEstimator(backend='mock')
    pose = estimator.estimate_dual_pose(dual)
    renderer = DualViewRenderer()
    renderer.font_mgr.warmup_fonts()
    card = {'kinematic_sequence_text': '证据不足 · 含候选拍峰', 'hip_evidence_text': '未观测',
            'brush_evidence_text': '观测证据不足', 'racket_speed_px_s': 1399.71}
    if args.events:
        card = build_impact_telemetry_card(json.loads(args.events.read_text())['events'][0])
    samples = []
    active = {}
    for obj, name, label in [(renderer, 'draw_hud', 'hud'),
                             (renderer, 'draw_impact_telemetry_card', 'telemetry'),
                             (renderer, 'draw_skeleton', 'skeleton'),
                             (renderer, 'apply_eye_privacy_mask', 'privacy'),
                             (renderer.font_mgr, 'put_text_with_font', 'font')]:
        original = getattr(obj, name)
        def wrap(*a, _fn=original, _label=label, **kw):
            start, cpu = time.perf_counter_ns(), time.thread_time_ns()
            result = _fn(*a, **kw)
            active.setdefault(_label + '_wall_ms', []).append((time.perf_counter_ns()-start)/1e6)
            active.setdefault(_label + '_cpu_ms', []).append((time.thread_time_ns()-cpu)/1e6)
            return result
        setattr(obj, name, wrap)
    report = {'fixture': '49.43 first decoded frame; mock body pose, fixed card content; no inference',
              'source_pixels_sha256': hashlib.sha256(frame.tobytes()).hexdigest(), 'variants': {}}
    for variant in ('plain', 'coach', 'card'):
        samples = []
        for i in range(args.count + 5):
            active = {}
            start, cpu = time.perf_counter_ns(), time.thread_time_ns()
            result = renderer.render_dual_frame(dual, pose, event_label='Forehand',
                coaching_text='动作证据不足需复核' if variant != 'plain' else '',
                telemetry_card=card if variant == 'card' else None)
            active['total_wall_ms'] = [(time.perf_counter_ns()-start)/1e6]
            active['total_cpu_ms'] = [(time.thread_time_ns()-cpu)/1e6]
            if i >= 5:
                samples.append({k: sum(v) for k, v in active.items()})
        report['variants'][variant] = {'samples': samples,
            'timing': {k: dist([r[k] for r in samples]) for k in samples[0]},
            'output_sha256': hashlib.sha256(result.tobytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: {'wall': v['timing']['total_wall_ms'], 'font': v['timing'].get('font_wall_ms'),
                          'sha256': v['output_sha256']} for k,v in report['variants'].items()}, indent=2))


if __name__ == '__main__':
    main()
