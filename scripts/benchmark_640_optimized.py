"""Repeat continuous 640 shadow runs with full encoding and current optimizations."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racket_model_profiles import EXPERIMENT_640_PATH
from racket_pose_detector import model_digest
from scripts.validate_racket_pose_pipeline import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', default=3, type=int)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    expected = model_digest(ROOT / EXPERIMENT_640_PATH)
    for repeat in range(args.repeats):
        clips = ['48.43.mp4', '49.43.mp4', '50.03.mp4']
        clips = clips[repeat % 3:] + clips[:repeat % 3]
        for clip in clips:
            options = SimpleNamespace(output=str(args.output), save_video=True, fault_sidecars=False,
                production_check=False, pose_compute_units=None, display_mode='async', opencv_threads=None,
                detector_compute_units=None, model=str(ROOT / EXPERIMENT_640_PATH), acceptance=None,
                backend='coreml', compute_units='ALL', imgsz=640, dual=True, inference_schedule='barrier',
                racket_ipc='shared_memory', roi_policy='project', roi_stream_id='court02-main')
            report = run_case(options, Path('/Users/krum5539/Desktop/Camera/2026-09-30') / clip,
                              'shadow', 'buffered', 1, repeat)
            if report.get('sidecar', {}).get('model_sha256') != expected:
                raise ValueError('Cached model identity differs')
            report['tuning_profile'] = 'font_cache_and_resize_before_rgb'
            reports.append(report)
            (args.output / 'matrix.json').write_text(json.dumps(reports, indent=2))
            if report['returncode'] != 0 or not report.get('continuous_gate'):
                raise RuntimeError('Incomplete continuous run')


if __name__ == '__main__':
    main()
