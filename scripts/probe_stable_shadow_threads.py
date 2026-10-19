"""Sequential same-pixel OpenCV thread-budget comparison, with the full workflow."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racket_model_profiles import EXPERIMENT_640_PATH
from scripts.validate_racket_pose_pipeline import run_case
from scripts.validate_stable_shadow import strict_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--clips', nargs='+', default=['49.53', '50.03', '48.43'])
    parser.add_argument('--budgets', nargs='+', type=int, default=[10, 1])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(p for p in ROOT.glob('*.py')) + sorted((ROOT/'configs').glob('*.yaml'))
    frozen = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.output/'frozen-runtime.json').write_text(json.dumps(frozen, indent=2))
    reports = []
    for i, clip in enumerate(args.clips):
        budgets = args.budgets[i % len(args.budgets):] + args.budgets[:i % len(args.budgets)]
        for budget in budgets:
            if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != h for p,h in frozen.items()):
                raise RuntimeError('Runtime changed during comparison')
            options = SimpleNamespace(output=str(args.output/f'threads-{budget}'), save_video=True,
                fault_sidecars=False, production_check=True, pose_compute_units=None,
                display_mode='async', opencv_threads=budget, detector_compute_units=None,
                model=str(ROOT/EXPERIMENT_640_PATH), acceptance=None, backend='coreml',
                compute_units='ALL', imgsz=640, dual=True, inference_schedule='barrier',
                racket_ipc='shared_memory', roi_policy='project', roi_stream_id='court02-main',
                collect_clip_evidence=True, pose_resize_policy='legacy')
            report = run_case(options, Path('/Users/krum5539/Desktop/Camera/2026-09-30')/(clip+'.mp4'),
                              'shadow', 'buffered', 1, 0)
            report['opencv_thread_budget'] = budget
            report['strict_stable_shadow_gate'] = strict_gate(report)
            reports.append(report)
            (args.output/'assessment.json').write_text(json.dumps({'runs':reports,
                'scope':'Sequential thread-budget engineering comparison; no release acceptance'},
                ensure_ascii=False, indent=2))
            print('THREADS', budget, clip, report.get('output_fps'),
                  report.get('nominal_input_to_output_ms'), strict_gate(report), flush=True)
            if not report.get('continuous_gate'):
                raise RuntimeError('Continuity failed')


if __name__ == '__main__':
    main()
