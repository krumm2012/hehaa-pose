"""Strict, fail-closed release assessment for continuous 640 shadow (not active)."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racket_model_profiles import EXPERIMENT_640_PATH
from racket_pose_detector import model_digest
from scripts.validate_racket_pose_pipeline import run_case


def environment():
    import cv2
    return {'python':sys.version,'numpy':version('numpy'),'coremltools':version('coremltools'),
            'opencv':cv2.__version__,'opencv_default_threads':cv2.getNumThreads()}


def strict_gate(report):
    metrics = report.get('sidecar', {}).get('metrics', {})
    clips = report.get('clip_evidence', {})
    return bool(report.get('returncode') == 0 and report.get('continuous_gate') is True
        and report.get('frames') == 250 and report.get('unique_source_frames') == 250
        and report.get('new_same_frame') == 250 and report.get('encoded_frames') == 250
        and report.get('display_frames_complete') is True
        and report.get('source_time_valid') is True and report.get('replay_semantic_match') is True
        and all(metrics.get(k) == 0 for k in ('skipped', 'replaced_pending', 'late', 'fallback'))
        and metrics.get('submitted') == 250 and metrics.get('matched') == 250
        and report.get('output_fps', 0) >= 25
        and report.get('nominal_input_to_output_ms', {}).get('p95', float('inf')) <= 250
        and report.get('output_lag_growth_ms', float('inf')) <= 40
        and clips.get('passed') is True)


def clip_evidence(session):
    document = json.loads((session / 'events.json').read_text())
    events = document.get('events', [])
    rows = [{k: event.get(k) for k in ('event_id', 'clip_status', 'clip_frame_count', 'clip_missing_frame_count')}
            for event in events]
    dropped = document.get('summary', {}).get('clip_buffer_dropped_frames')
    return {'events': rows, 'buffer_dropped_frames': dropped,
            'passed': bool(rows and dropped == 0 and all(r['clip_status'] == 'ready'
                and isinstance(r['clip_frame_count'], int) and r['clip_frame_count'] > 0
                and r['clip_missing_frame_count'] == 0 for r in rows))}


def fingerprint(videos):
    paths = sorted(ROOT.glob('*.py')) + sorted((ROOT/'scripts').glob('*.py'))
    paths += sorted((ROOT/'configs').rglob('*.yaml')) + list(videos)
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def freeze(videos):
    """Bind every validation phase to ordinary source files and all three models."""
    import yaml
    config=yaml.safe_load((ROOT/'configs/yolo26_tennis_config.yaml').read_text())
    paths=[ROOT/config['yolo_pose_model_path'],
           ROOT/config['unified_detection']['model_path'], ROOT/EXPERIMENT_640_PATH]
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=ROOT,check=True)
    result={'files':fingerprint(videos),'environment':environment(),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'models':{str(p):model_digest(p) for p in paths},
            'model_sha256':model_digest(ROOT/EXPERIMENT_640_PATH)}
    result['build_id']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result


def verify_frozen(path):
    frozen=json.loads(Path(path).read_text())
    content={k:v for k,v in frozen.items() if k!='build_id'}
    if hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()!=frozen.get('build_id'):
        raise ValueError('Frozen identity digest differs')
    if 'environment' in frozen and frozen['environment']!=environment():
        raise ValueError('Frozen dependency environment changed')
    files=frozen['files']
    if any(not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h
           for p,h in files.items()):
        raise ValueError('Frozen source/config/input changed')
    if any(model_digest(Path(p))!=h for p,h in frozen['models'].items()):
        raise ValueError('Frozen model changed')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if commit!=frozen['source_commit']:
        raise ValueError('Frozen commit changed')
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=ROOT,check=True)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--videos', type=Path, default=Path('/Users/krum5539/Desktop/Camera/2026-09-30'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    videos = sorted(args.videos.glob('*.mp4'))
    if len(videos) != 9:
        raise ValueError('Requires the nine original regression clips')
    frozen = freeze(videos)
    model_hash = model_digest(ROOT / EXPERIMENT_640_PATH)
    if model_hash != '0dec1259f50b2cfb286f3ed5408a2f5f589749a0ddaf7a9b8a4139f29c1a025e':
        raise ValueError('Unexpected candidate model')
    frozen_path=args.output/'frozen-inputs.json'
    frozen_path.write_text(json.dumps(frozen, indent=2))
    reports = []
    for repeat in range(3):
        for video in videos[repeat:] + videos[:repeat]:
            verify_frozen(frozen_path)
            options = SimpleNamespace(output=str(args.output), save_video=True, fault_sidecars=False,
                production_check=True, pose_compute_units=None, display_mode='async', opencv_threads=None,
                detector_compute_units=None, model=str(ROOT / EXPERIMENT_640_PATH), acceptance=None,
                backend='coreml', compute_units='ALL', imgsz=640, dual=True, inference_schedule='barrier',
                racket_ipc='shared_memory', roi_policy='project', roi_stream_id='court02-main',
                collect_clip_evidence=True)
            report = run_case(options, video, 'shadow', 'buffered', 1, repeat)
            verify_frozen(frozen_path)
            report.update(build_id=frozen['build_id'],source_commit=frozen['source_commit'])
            if report.get('sidecar', {}).get('model_sha256') != model_hash:
                raise ValueError('Run model identity differs')
            report['strict_stable_shadow_gate'] = strict_gate(report)
            reports.append(report)
            summary = {'schema': 'tennis.stable-shadow-regression.v1', 'assessment_only': True,
                'model_sha256': model_hash, 'source_files_unchanged': True,
                'build_id':frozen['build_id'],'source_commit':frozen['source_commit'],
                'expected_runs': 27, 'completed_runs': len(reports),
                'passed_runs': sum(r['strict_stable_shadow_gate'] for r in reports),
                'passed': len(reports) == 27 and all(r['strict_stable_shadow_gate'] for r in reports),
                'requirements': {'fps_min': 25, 'nominal_p95_ms_max': 250, 'stride': 1, 'source_frames': 250},
                'runs': reports}
            (args.output / 'assessment.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            print('STRICT', video.name, repeat, report['strict_stable_shadow_gate'], flush=True)
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
