"""Audit actual racket model heads/contracts and all raw outputs on one local clip.

No export, model mutation, skipped source frames, or production config changes.
Each Core ML worker loads exactly one package and calls predict once per frame.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def contract(path):
    import coremltools as ct
    from racket_pose_detector import model_digest
    spec=ct.models.utils.load_spec(str(path))
    def feature(f):
        kind=f.type.WhichOneof('Type')
        return {'name':f.name,'type':kind,**({'width':f.type.imageType.width,'height':f.type.imageType.height} if kind=='imageType' else {'shape':list(f.type.multiArrayType.shape)} if kind=='multiArrayType' else {})}
    meta=dict(spec.description.metadata.userDefined)
    return {'path':str(Path(path).resolve()),'sha256':model_digest(path),
        'inputs':[feature(f) for f in spec.description.input],
        'outputs':[feature(f) for f in spec.description.output],
        'metadata':{k:meta[k] for k in ('names','kpt_shape','task','imgsz','batch','args') if k in meta},
        'description':spec.description.metadata.shortDescription}


def worker(args):
    import cv2
    import coremltools as ct
    import numpy as np
    from PIL import Image
    from racket_pose_detector import letterbox, decode_candidates, pose_config, draw_racket_pose, model_digest
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    report={'contract':contract(args.model),'compute_units':'CPU_AND_NE','samples':[]}
    model=ct.models.MLModel(args.model,compute_units=ct.ComputeUnit.CPU_AND_NE)
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened():raise RuntimeError('Cannot open input')
    expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fid=0
    with (out/'predictions.jsonl').open('x') as journal:
        while True:
            ok,frame=cap.read()
            if not ok:break
            pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            rgb,transform=letterbox(frame)
            started=time.perf_counter()
            outputs=model.predict({'image':Image.fromarray(rgb)})
            elapsed=(time.perf_counter()-started)*1000
            actual={name:{'shape':list(np.asarray(value).shape),'nonfinite_values':int((~np.isfinite(value)).sum())} for name,value in outputs.items()}
            sample={'frame_id':fid,'media_pts_seconds':pts,'outputs':actual,'predict_ms':elapsed}
            if len(outputs)==1:
                raw=next(iter(outputs.values()))
                if raw.shape==(1,26,18900):
                    candidates,diag=decode_candidates(raw,transform,pose_config({'mode':'shadow'}))
                    sample.update(diagnostics=diag,racket_candidates=len(candidates),points_per_candidate=7)
                    if fid==125:
                        canvas=frame.copy()
                        if diag['nonfinite_values']==0:
                            draw_racket_pose(canvas,{'status':'observed','candidates':candidates})
                            for candidate in candidates:
                                x1,y1,x2,y2=map(round,candidate['box'])
                                cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,220,255),2)
                        text='7 racket points; no human-pose output' if diag['nonfinite_values']==0 else 'Original raw output contains NaN/Inf; numeric result is invalid'
                        cv2.putText(canvas,text,(30,48),cv2.FONT_HERSHEY_SIMPLEX,1,(0,220,255),2)
                        if not cv2.imwrite(str(out/'frame_0125.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,85]):raise RuntimeError('Image write failed')
            journal.write(json.dumps(sample,allow_nan=False)+'\n');report['samples'].append(sample)
            fid+=1
            if fid%50==0:print(Path(args.model).name,fid,'frames',flush=True)
    cap.release()
    if fid!=expected or fid!=250:raise RuntimeError(f'Incomplete source: {fid}/{expected}')
    assert [r['frame_id'] for r in report['samples']]==list(range(250))
    report['frames']=fid;report['expected_frames']=expected;report['predict_calls']=fid
    report['source_sha256']=model_digest(args.video)
    report['raw_output_contract_constant']=all(r['outputs'].keys()==report['samples'][0]['outputs'].keys() and all(v['shape']==[1,26,18900] for v in r['outputs'].values()) for r in report['samples'])
    report['nonfinite_frames']=sum(any(v['nonfinite_values']>0 for v in r['outputs'].values()) for r in report['samples'])
    report['nonfinite_candidate_range']=[min(r['diagnostics']['nonfinite_candidates'] for r in report['samples']),max(r['diagnostics']['nonfinite_candidates'] for r in report['samples'])]
    report['candidate_count_range']=[min(r['racket_candidates'] for r in report['samples']),max(r['racket_candidates'] for r in report['samples'])]
    (out/'result.json').write_text(json.dumps({k:v for k,v in report.items() if k!='samples'},ensure_ascii=False,indent=2))


def main(args):
    import yaml
    from ultralytics import YOLO
    from racket_pose_detector import model_digest
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((ROOT/'configs/yolo26_tennis_config.yaml').read_text())
    repair=json.loads((ROOT/'models/racket_pose/candidates/bn-recalibrated-v1/repair.json').read_text())
    training=Path('/Users/krum5539/Documents/yolo26pose-racket/configs/training/exp-007-racket-pose-yolo26m-v2.0-gold.yaml')
    dataset=Path('/Users/krum5539/Documents/yolo26pose-racket/configs/dataset_racket_v4.yaml')
    data=yaml.safe_load(dataset.read_text())
    report={'schema':'tennis.racket-joint-output-audit.v1','date':'2026-10-18',
        'video':str(Path(args.video).resolve()),'source_sha256':model_digest(args.video),
        'training_config':{'path':str(training),'sha256':model_digest(training)},
        'dataset':{'path':str(dataset),'sha256':model_digest(dataset),'names':data['names'],'kpt_shape':data['kpt_shape']},
        'human_model_contract':contract(ROOT/cfg['yolo_pose_model_path']), 'heads':[], 'models':{}}
    for path in (repair['original'],str(ROOT/'models/racket_pose/candidates/bn-recalibrated-v1/bn_repaired_fp32.pt')):
        if not Path(path).is_file():raise FileNotFoundError(path)
        yolo=YOLO(path)
        heads=[{'module':name,'class':type(m).__name__,'nc':int(m.nc),'kpt_shape':list(m.kpt_shape),'nk':int(m.nk)} for name,m in yolo.model.named_modules() if hasattr(m,'kpt_shape') and hasattr(m,'nc') and hasattr(m,'nk')]
        report['heads'].append({'path':path,'sha256':model_digest(path),'names':yolo.names,'pose_heads':heads})
        assert len(heads)==1 and heads[0]['nc']==1 and heads[0]['kpt_shape']==[7,3]
        del yolo
    for label,path in [('original',ROOT/'models/racket_pose/racket_pose_yolo26m_fp16.mlpackage'),('validated',ROOT/cfg['racket_pose']['model_path'])]:
        subprocess.run([sys.executable,__file__,'--worker','--model',str(path),'--video',args.video,'--output',str(out/label)],cwd=ROOT,check=True,timeout=120)
        report['models'][label]=json.loads((out/label/'result.json').read_text())
        assert report['models'][label]['raw_output_contract_constant']
    assert report['models']['validated']['nonfinite_frames']==0
    report.update(can_output_human_and_racket=False,
        reason='One class tennis_racket, one 7x3 pose head, one [1,26,18900] output = 4 box + 1 class score + 7x3 racket points. No person class, 17-point human head, or other human output.',
        inference_scope='One package per isolated worker; one predict per source frame; no human model inference invoked. The human package is inspected only for its contract.',
        limitations='This proves the deployed model output capability, not detection accuracy or the absence of people in the source video. Original package numeric failures are reported, never interpreted as valid detections.')
    (out/'joint-output-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'can_output_human_and_racket':False,'frames_per_model':250,'original_nonfinite_frames':report['models']['original']['nonfinite_frames'],'validated_nonfinite_frames':report['models']['validated']['nonfinite_frames']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--video',default='/Users/krum5539/Desktop/Camera/2026-09-30/48.43.mp4');parser.add_argument('--worker',action='store_true');parser.add_argument('--model');args=parser.parse_args()
    worker(args) if args.worker else main(args)
