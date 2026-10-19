"""Compare legacy two-stage pose pixels against direct source ROI resizing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from racket_model_profiles import EXPERIMENT_640_PATH
from scripts.validate_racket_pose_pipeline import run_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--policies',nargs='+',choices=('legacy','direct_front','direct_source'),
                        default=['legacy','direct_front','direct_source'])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    frozen={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
        'main_pipe.py','dual_view_manager.py','dual_pose_estimator.py','pose_estimator_yolo26.py',
        'configs/yolo26_tennis_config.yaml','configs/roi_config.yaml','configs/dual_view_config.yaml')}
    reports=[]
    for i,clip in enumerate(('48.43.mp4','49.43.mp4','50.03.mp4')):
        policies=args.policies[i%len(args.policies):]+args.policies[:i%len(args.policies)]
        for policy in policies:
            options=SimpleNamespace(output=str(args.output/policy),save_video=True,fault_sidecars=False,
                production_check=True,pose_compute_units=None,display_mode='async',opencv_threads=None,
                detector_compute_units=None,model=str(ROOT/EXPERIMENT_640_PATH),acceptance=None,
                backend='coreml',compute_units='ALL',imgsz=640,dual=True,inference_schedule='barrier',
                racket_ipc='shared_memory',roi_policy='project',roi_stream_id='court02-main',
                collect_clip_evidence=True,pose_resize_policy=policy)
            r=run_case(options,Path('/Users/krum5539/Desktop/Camera/2026-09-30')/clip,'shadow','buffered',1,0)
            if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in frozen.items()):
                raise ValueError('Runtime changed during comparison')
            r['resize_build_sha256']=frozen;reports.append(r)
            (args.output/'matrix.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
            print('RESIZE',policy,clip,r.get('output_fps'),r.get('continuous_gate'),flush=True)
            if not r.get('continuous_gate') or not r.get('clip_evidence',{}).get('passed'):
                raise RuntimeError('Frame or clip completeness failed')


if __name__=='__main__':main()
