"""Same-source legacy compatibility and enabled network/TTS fault isolation."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from racket_model_profiles import EXPERIMENT_640_PATH
from scripts.validate_racket_pose_pipeline import run_case
from scripts.validate_stable_shadow import strict_gate, verify_frozen

FRAME_FIELDS=('frame_id','timestamp','swing_type','ball','rackets','racket','pose','metrics',
    'ball_observation','healed_pose','dual_view_biomechanics','pose_observations')
EVENT_FIELDS=('event_id','start_frame','contact_frame','peak_frame','end_frame','stroke_type',
    'confidence','contact_status','evidence','quality_flags','phase_counts','biomechanics',
    'extended_biomechanics','swing_score','practice_score','scoring_blockers','coach_advices','coach_advice')


def compare(old,new):
    left=[json.loads(s) for s in (old/'frames.jsonl').read_text().splitlines()]
    right=[json.loads(s) for s in (new/'frames.jsonl').read_text().splitlines()]
    frame_checks={k:len(left)==len(right)==250 and all((k in a)==(k in b) and a.get(k)==b.get(k)
        for a,b in zip(left,right)) for k in FRAME_FIELDS}
    a=json.loads((old/'events.json').read_text())['events']
    b=json.loads((new/'events.json').read_text())['events']
    event_checks={k:len(a)==len(b) and all(x.get(k)==y.get(k) for x,y in zip(a,b)) for k in EVENT_FIELDS}
    return {'passed':all(frame_checks.values()) and all(event_checks.values()),
        'frame_fields':frame_checks,'event_fields':event_checks,'events':len(a)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shadow-runs',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--frozen-inputs',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    frozen=verify_frozen(args.frozen_inputs)
    checks=[];faults=[]
    videos=sorted(Path('/Users/krum5539/Desktop/Camera/2026-09-30').glob('*.mp4'))
    for fault in (False,True):
        selected=[v for v in videos if v.name in ('48.43.mp4','49.43.mp4','50.03.mp4')] if fault else videos
        for video in selected:
            verify_frozen(args.frozen_inputs)
            directory=args.output/('faults' if fault else 'legacy')
            options=SimpleNamespace(output=str(directory),save_video=True,fault_sidecars=fault,
                production_check=True,pose_compute_units=None,display_mode='async',opencv_threads=None,
                detector_compute_units=None,model=str(ROOT/EXPERIMENT_640_PATH),acceptance=None,
                backend='coreml',compute_units='ALL',imgsz=640,dual=True,inference_schedule='barrier',
                racket_ipc='shared_memory',roi_policy='project',roi_stream_id='court02-main',collect_clip_evidence=True)
            report=run_case(options,video,'shadow' if fault else 'legacy','buffered',1,0)
            verify_frozen(args.frozen_inputs)
            original=args.shadow_runs/f'buffered-shadow-s1-{video.stem}-r0'
            compared=compare(Path(report['session']),original)
            if fault:
                statuses=report.get('sidecar_fault_statuses',[])
                observed=bool(any((s.get('deepseek_advice') or {}).get('status')=='failed' for s in statuses)
                    and any((s.get('coach_tts') or {}).get('status')=='unavailable' for s in statuses))
                faults.append({'clip':video.name,'passed':bool(strict_gate(report) and compared['passed'] and observed),
                    'faults_observed':observed,'comparison':compared,'measurement':report})
            else:
                no_worker=not list(Path(report['session']).glob('output_racket_pose*'))
                checks.append({'clip':video.name,'source_sha256':report['source_sha256'],
                    'passed':bool(compared['passed'] and report.get('continuous_gate') and report.get('clip_evidence',{}).get('passed') and no_worker),
                    'no_seven_point_worker_files':no_worker,'comparison':compared,'measurement':report})
            identity={'build_id':frozen['build_id'],'source_commit':frozen['source_commit']}
            result={'compatibility':{**identity,'passed':len(checks)==9 and all(c['passed'] for c in checks),'runs':checks},
                    'fault_workflow':{**identity,'passed':len(faults)==3 and all(c['passed'] for c in faults),'runs':faults}}
            (args.output/'assessment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
            print('WORKFLOW',video.name,'fault' if fault else 'legacy',(faults if fault else checks)[-1]['passed'],flush=True)
    if not result['compatibility']['passed'] or not result['fault_workflow']['passed']:raise SystemExit(1)


if __name__=='__main__':main()
