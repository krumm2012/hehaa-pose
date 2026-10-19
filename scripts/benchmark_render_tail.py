"""Keep whole-frame render stages, with/without-card groups and slow frame identities."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racket_model_profiles import EXPERIMENT_640_PATH
from scripts.validate_racket_pose_pipeline import distribution, run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clips', nargs='+', default=['48.43.mp4', '49.43.mp4', '50.03.mp4'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob('buffered-*')):
        parser.error('Use a new output directory to avoid reusing measurements from another build')
    reports = []
    for clip in args.clips:
        options = SimpleNamespace(output=str(args.output), save_video=True, fault_sidecars=False,
            production_check=False, pose_compute_units=None, display_mode='async', opencv_threads=None,
            detector_compute_units=None, model=str(ROOT / EXPERIMENT_640_PATH), acceptance=None,
            backend='coreml', compute_units='ALL', imgsz=640, dual=True, inference_schedule='barrier',
            racket_ipc='shared_memory', roi_policy='project', roi_stream_id='court02-main')
        report = run_case(options, Path('/Users/krum5539/Desktop/Camera/2026-09-30') / clip,
                          'shadow', 'buffered', 1, 0)
        if not report.get('continuous_gate'):
            raise RuntimeError('Continuous frame coverage failed')
        session = Path(report['session'])
        rows = [json.loads(line) for line in (session/'frames.jsonl').read_text().splitlines()]
        if any('render_stage_ms' not in r for r in rows):
            raise ValueError('Run predates render instrumentation; use a new directory')
        groups = {}
        for group, selected in [('all', rows), ('card', [r for r in rows if r['render_features'].get('telemetry_card')]),
                                ('no_card', [r for r in rows if not r['render_features'].get('telemetry_card')])]:
            names = {name for r in selected for name in r['render_stage_ms']}
            groups[group] = {'frames': len(selected), 'render_ms': distribution([r['timing']['render_ms'] for r in selected]),
                'stages': {name: distribution([r['render_stage_ms'][name] for r in selected if name in r['render_stage_ms']]) for name in names}}
        report['render_groups'] = groups
        report['slow_render_frames'] = [{key: r.get(key) for key in ('frame_id', 'timing', 'render_stage_ms', 'render_features')}
                                        for r in sorted(rows,key=lambda r:r['timing']['render_ms'],reverse=True)[:12]]
        report['render_build_sha256'] = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                                         for f in ('dual_view_renderer.py', 'font_manager.py', 'main_pipe.py',
                                                   'realtime_swing_pipeline.py')}
        reports.append(report)
        (args.output/'matrix.json').write_text(json.dumps(reports, indent=2))
        print(json.dumps({'clip': clip, 'fps': report['output_fps'], 'render': groups}, indent=2), flush=True)


if __name__ == '__main__':
    main()
