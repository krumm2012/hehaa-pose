"""Numerical/parity/quality receipt bound to exact racket model and backend."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from racket_pose_detector import RacketPoseDetector, letterbox, model_digest, decode_candidates
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--videos',default='/Users/krum5539/Desktop/Camera/2026-09-30')
    parser.add_argument('--quality',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--compute-units',choices=('ANE','ALL','CPU'),default='ANE')
    args=parser.parse_args()
    detector=RacketPoseDetector({'mode':'shadow','model_path':args.model,'compute_units':args.compute_units})
    reference=YOLO(args.reference).model.float().cpu().eval();torch.set_num_threads(4)
    for k,v in reference.state_dict().items():
        if not torch.isfinite(v).all(): print('BAD REFERENCE BUFFER',k,flush=True)
    rows=[];parity=[]
    for video in sorted(Path(args.videos).glob('*.mp4')):
        cap=cv2.VideoCapture(str(video));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fid in sorted({0,n//5,2*n//5,3*n//5,4*n//5,n-1}):
            cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,frame=cap.read()
            if not ok:raise RuntimeError('Cannot decode '+str(video))
            observation=detector.detect(frame,fid,{'timestamp_seconds':fid/cap.get(cv2.CAP_PROP_FPS)})
            rows.append({'video':video.name,'frame_id':fid,'diagnostics':observation['diagnostics'],'timing':observation['timing']})
            if fid==0:
                from PIL import Image
                rgb,transform=letterbox(frame)
                tensor=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None]/255
                with torch.no_grad():out=reference(tensor)
                pt=(out[0] if isinstance(out,tuple) else out).numpy()
                ml=detector.model.predict({'image':Image.fromarray(rgb)})[detector.output_name]
                finite=bool(np.isfinite(pt).all() and np.isfinite(ml).all())
                selected=np.isfinite(pt[0]).all(0)&(pt[0,4]>=.25)
                pt_points=pt[0,5:,selected].reshape(-1,7,3)
                ml_points=ml[0,5:,selected].reshape(-1,7,3)
                good=pt_points[:,:,2]>=.5
                errors=np.linalg.norm(pt_points[:,:,:2]-ml_points[:,:,:2],axis=-1)[good]
                pt_decoded,_=decode_candidates(pt,transform,detector.config)
                ml_decoded,_=decode_candidates(ml,transform,detector.config)
                remaining=list(ml_decoded);matched=0
                for detection in pt_decoded:
                    a=np.asarray(detection['box'])
                    def iou(other):
                        b=np.asarray(other['box']);wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
                        intersection=float(np.prod(wh));union=float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)
                        return intersection/union if union>0 else 0
                    if remaining:
                        best=max(range(len(remaining)),key=lambda i:iou(remaining[i]))
                        if iou(remaining[best])>=.5:matched+=1;remaining.pop(best)
                parity.append({'video':video.name,'raw_finite':finite,'qualified_points':int(good.sum()),
                               'pytorch_nms_count':len(pt_decoded),'coreml_nms_count':len(ml_decoded),
                               'matched_nms_count':matched,'missing_reference_detections':len(pt_decoded)-matched,
                               'extra_coreml_detections':len(remaining),
                               'threshold_crossing_candidates':int(np.count_nonzero((pt[0,4]>=.25)!=(ml[0,4]>=.25))),
                               'point_error_p95_input_px':float(np.percentile(errors,95)) if len(errors) and np.isfinite(errors).all() else None,
                               'confidence_error_max':float(np.max(np.abs(pt[0,4]-ml[0,4]))) if finite else None})
        cap.release();print('AUDIT',video.name,'bad candidates',sum(r['diagnostics']['nonfinite_candidates'] for r in rows if r['video']==video.name),flush=True)
    quality=json.loads(Path(args.quality).read_text())
    old=quality.get('original',{}).get('pose_map');new=quality.get('candidate',{}).get('pose_map')
    numerical=bool(rows) and all(r['diagnostics']['nonfinite_candidates']==0 for r in rows)
    parity_pass=all(r['raw_finite'] and r['point_error_p95_input_px'] is not None and r['point_error_p95_input_px']<=3 and r['confidence_error_max']<=.03 and r['missing_reference_detections']==0 for r in parity)
    quality_pass=old is not None and new is not None and new>=old-.02
    quality_pass = quality_pass and quality.get('candidate',{}).get('box_map',-1)>=quality.get('original',{}).get('box_map',0)-.03
    report={'schema':'tennis.racket-model-acceptance.v1','model_sha256':detector.digest,'backend':'coreml',
            'compute_units':args.compute_units,'model_path':str(Path(args.model).resolve()),'reference':args.reference,
            'numerical_valid':numerical and parity_pass,'quality_gate_passed':quality_pass,
            'quality':quality,'parity':parity,'samples':rows,'sample_count':len(rows),
            'accuracy_validated':False,'scope':'Same-corpus engineering acceptance; 54 raw video frames and 9 identical-input reference comparisons; not an independent accuracy test.'}
    report['reference_sha256']=model_digest(args.reference)
    report['quality_report_sha256']=model_digest(args.quality)
    report['actual_head']=type(reference.model[-1]).__name__
    report['architecture_yaml']=str(reference.yaml.get('yaml_file','unavailable'))
    report['sources']=[{'name':p.name,'sha256':model_digest(p)} for p in sorted(Path(args.videos).glob('*.mp4'))]
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ('numerical_valid','quality_gate_passed','sample_count')},ensure_ascii=False),flush=True)
    if not numerical or not parity_pass or not quality_pass:raise SystemExit(1)

if __name__=='__main__':main()
