"""Continuous shadow A/B performance; quality-unaccepted models never go active."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_racket_pose_pipeline import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    profiles = [('960-ANE', None, 960, 'ANE'), ('640-ALL', args.candidate, 640, 'ALL'),
                ('640-ANE', args.candidate, 640, 'ANE')]
    reports = []
    for index, clip in enumerate(('48.43.mp4', '49.43.mp4', '50.03.mp4')):
        for name, model, size, units in profiles[index:] + profiles[:index]:
            options = SimpleNamespace(output=str(args.output / name), save_video=True, fault_sidecars=False,
                                      production_check=False, pose_compute_units=None, display_mode='async', opencv_threads=None,
                                      detector_compute_units=None, model=model, acceptance=None, backend='coreml', compute_units=units,
                                      imgsz=size, dual=True, inference_schedule='barrier', racket_ipc='shared_memory',
                                      roi_policy='project', roi_stream_id='court02-main')
            (args.output / name).mkdir(exist_ok=True)
            # Each profile directory is shared by subsequent source videos.
            video = Path('/Users/krum5539/Desktop/Camera/2026-09-30') / clip
            report = run_case(options, video, 'shadow', 'buffered', 1, 0)
            report['resolution_profile'] = name
            report['performance_only'] = True
            report['quality_accepted'] = False
            report['observed_output_ge_25fps'] = report.get('output_fps', 0) >= 25
            reports.append(report)
            (args.output / 'matrix.json').write_text(json.dumps(reports, indent=2))
            if report['returncode'] != 0:
                raise RuntimeError('Pipeline failed: ' + report['session'])
            if not report.get('continuous_model_complete') or not report.get('continuous_gate'):
                raise RuntimeError('Incomplete continuous model/pipeline run')


if __name__ == '__main__':
    main()
