"""Re-run pose inference at multiple video resolutions; stability, not accuracy."""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
from statistics import mean
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from dual_pose_estimator import DualPoseEstimator
from dual_view_manager import DualViewManager
from dual_view_renderer import DualViewRenderer

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--frames',default='')
    parser.add_argument('--continuous',action='store_true')
    args=parser.parse_args();manifest=json.loads(Path(args.manifest).read_text())
    code_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('dual_pose_estimator.py','dual_view_manager.py','dual_view_renderer.py')}
    source=manifest['session']['source'];expected=next(a['sha256'] for a in manifest['artifacts'] if a['role']=='source_video')
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==expected,'source changed'
    config={k:manifest['capture']['roi'][k] for k in ('front_view','mirror_view')}
    roi=manifest['capture']['roi']['points'];bbox=[min(p[0] for p in roi),min(p[1] for p in roi),max(p[0] for p in roi),max(p[1] for p in roi)]
    ids=sorted(set(range(0,250,10))|{27,64,65,66,110,191});scales=[1,.75,.5]
    if args.continuous:ids=list(range(250));scales=[1,.5]
    if args.frames:ids=[int(x) for x in args.frames.split(',')]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    estimator=DualPoseEstimator(model_path=manifest['session']['models']['pose'],backend='coreml',concurrent=False)
    if estimator.backend!='coreml':raise RuntimeError('required Core ML unavailable')
    estimators={1:estimator}
    managers={scale:DualViewManager(config_dict=config) for scale in scales}
    if args.continuous:
        for scale in scales[1:]:
            estimators[scale]=DualPoseEstimator(model_path=manifest['session']['models']['pose'],backend='coreml',concurrent=False)
    renderer=DualViewRenderer(show_hud=False);samples={};timings=[]
    cap=cv2.VideoCapture(source)
    try:
        for fid in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,original=cap.read()
            if not ok:raise RuntimeError('source frame unavailable '+str(fid))
            samples[str(fid)]={}
            for scale in scales:
                frame=cv2.resize(original,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale!=1 else original
                manager=managers[scale] if args.continuous else DualViewManager(config_dict=config)
                dual=manager.split_frame(frame,frame_id=fid,front_crop_bbox=None if args.continuous else tuple(round(v*scale) for v in bbox))
                active=estimators[scale] if args.continuous else estimator
                if not args.continuous:active.reset()
                t=time.perf_counter();result=active.estimate_dual_pose(dual);timings.append(time.perf_counter()-t)
                if args.continuous:manager.update_player_from_keypoints(result.front_pose_orig)
                samples[str(fid)][str(scale)]={view:{k:{'x':p.x/scale,'y':p.y/scale,'confidence':p.conf,'observed':p.observed,'source_frame_id':p.source_frame_id}
                    for k,p in getattr(result,view+'_pose_orig').items()} for view in ('front','back')}
                if fid in (65,130,180):cv2.imwrite(str(out/('frame'+str(fid)+'_scale_'+str(scale)+'.jpg')),renderer.render_dual_frame(dual,result))
            print('frame',fid,'done',flush=True)
    finally:
        cap.release()
        for active in estimators.values():active.close()
    comparisons=[]
    for scale in scales[1:]:
        for view in ('front','back'):
            distances=[];both=reference=candidate=0;lost=gained=0
            for item in samples.values():
                a=item['1'][view];b=item[str(scale)][view]
                for joint in set(a)|set(b):
                    pa=a.get(joint,{});pb=b.get(joint,{})
                    va=pa.get('observed') and pa.get('confidence',0)>=.5
                    vb=pb.get('observed') and pb.get('confidence',0)>=.5
                    reference+=bool(va);candidate+=bool(vb);lost+=bool(va and not vb);gained+=bool(vb and not va)
                    if va and vb:
                        both+=1;distances.append(math.hypot(pa['x']-pb['x'],pa['y']-pb['y']))
            comparisons.append({'scale':scale,'view':view,'reference_qualified_points':reference,'candidate_qualified_points':candidate,
                'paired_points':both,'lost_points':lost,'gained_points':gained,'reference_retention_rate':both/reference if reference else None,
                'mean_difference_original_px':mean(distances) if distances else None,'p95_difference_original_px':float(np.percentile(distances,95)) if distances else None})
    model=Path(manifest['session']['models']['pose'])
    model_hashes={str(p.relative_to(model)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(model.rglob('*')) if p.is_file()}
    report={'code_sha256_at_start':code_hashes,'schema':'tennis.pose-resolution-audit.v1','source_sha256':expected,'model_files_sha256':model_hashes,
        'roi':config,'front_bbox_original':bbox,'frame_ids':ids,'scales':scales,'timing_seconds_mean':mean(timings),
        'mode':'continuous_dynamic_roi' if args.continuous else 'independent_fixed_roi',
        'limitations':(['sparse_independent_frames_not_temporal_validation','fixed_ROI_not_dynamic_tracking'] if not args.continuous else ['single_video_temporal_audit'])+['native_resolution_is_not_ground_truth',
                      'difference_on_paired_points_must_be_read_with_retention','same_model_input_shape_at_all_resolutions'],
        'comparisons':comparisons,'samples':samples,'measurement_accuracy':None}
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(comparisons,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
