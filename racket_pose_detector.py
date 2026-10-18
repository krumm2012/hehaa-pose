"""Optional 7-point racket observations. Importing this module loads no ML runtime."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

KEYPOINT_NAMES = ('butt_cap', 'grip_center', 'throat', 'sweet_spot', 'head_top', 'frame_left', 'frame_right')
SKELETON_EDGES = ((0,1),(1,2),(2,3),(3,4),(2,5),(2,6),(5,4),(6,4))
DEFAULT_MODEL = 'models/racket_pose/racket_pose_yolo26m_fp16.mlpackage'


def model_digest(path):
    path = Path(path)
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob('*') if p.is_file() and p.name != '.DS_Store') if path.is_dir() else [path]
    for p in files:
        if path.is_dir():
            h.update(p.relative_to(path).as_posix().encode()+b'\0')
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
    return h.hexdigest()


def pose_config(config=None):
    c = {'mode':'legacy', 'model_path':DEFAULT_MODEL, 'backend':'coreml', 'compute_units':'ANE',
         'imgsz':960, 'confidence_threshold':.25, 'keypoint_threshold':.5, 'iou_threshold':.45,
         'inference_stride':1, 'join_ms':8.0, 'timeout_seconds':10.0, 'startup_timeout_seconds':60.0,
         'max_wrist_distance_ratio':.05, 'render_skeleton':True, 'ipc_transport':'queue'}
    c.update(config or {})
    if c['mode']=='exclusive':
        c['mode']='active'
        c['selection_policy']='exclusive'
    c.setdefault('selection_policy','mixed')
    if c['mode']=='active' and c['selection_policy']=='exclusive':
        from racket_source_policy import SOURCE_POLICY_VERSION
        c['selection_policy_version']=SOURCE_POLICY_VERSION
    else:
        c.pop('selection_policy_version',None)
    if c['mode'] not in ('legacy','shadow','active'):
        raise ValueError('racket_pose.mode must be legacy, exclusive, shadow or active')
    if c['mode']=='legacy':return c  # Bypass remains usable even with a broken optional model configuration.
    if c['ipc_transport'] not in ('queue','shared_memory'):
        raise ValueError('Unsupported racket IPC transport')
    if c['selection_policy'] not in ('mixed','exclusive'):
        raise ValueError('Unsupported racket selection policy')
    if c['backend'] not in ('coreml','pytorch') or c['compute_units'] not in ('ANE','ALL','CPU'):
        raise ValueError('Unsupported racket pose backend/compute units; GPU is quarantined')
    for key in ('confidence_threshold','keypoint_threshold','iou_threshold'):
        c[key] = float(c[key])
        if not math.isfinite(c[key]) or not 0 <= c[key] <= 1: raise ValueError('Invalid '+key)
    c['inference_stride'] = int(c['inference_stride'])
    if c['inference_stride'] not in (1,2,3): raise ValueError('racket pose stride must be 1, 2 or 3')
    c['imgsz'] = int(c['imgsz'])
    if c['imgsz'] != 960: raise ValueError('Only the validated 960 racket pose contract is supported')
    for key in ('join_ms','timeout_seconds','startup_timeout_seconds','max_wrist_distance_ratio'):
        c[key] = float(c[key])
        if not math.isfinite(c[key]) or c[key] < 0: raise ValueError('Invalid '+key)
    c['join_ms'] = min(8.0,c['join_ms'])
    return c


def verify_acceptance(config, digest):
    path = config.get('acceptance_path')
    if not path: raise ValueError('Active racket pose requires an acceptance report')
    report = json.loads(Path(path).read_text())
    if (report.get('model_sha256') != digest or report.get('numerical_valid') is not True
            or report.get('quality_gate_passed') is not True
            or report.get('backend') != config['backend']
            or report.get('compute_units') != config['compute_units']):
        raise ValueError('Racket model numerical/quality acceptance missing or model/backend identity differs')
    return report


def verify_pipeline_acceptance(config, digest):
    """Production realtime enhancement is scoped to a measured pipeline profile."""
    path=config.get('runtime_acceptance_path')
    if not path:raise ValueError('Realtime enhancement requires pipeline acceptance')
    report=json.loads(Path(path).read_text())
    if (report.get('model_sha256')!=digest or report.get('backend')!=config['backend']
            or report.get('compute_units')!=config['compute_units']):
        raise ValueError('Pipeline acceptance model/backend identity differs')
    profile=(report.get('profiles') or {}).get(config.get('pipeline_profile','single')) or {}
    if profile.get('passed') is not True or profile.get('runs',0)<27:
        raise ValueError('Realtime pipeline profile has not passed three-round acceptance')
    for key in ('inference_stride','imgsz','confidence_threshold','keypoint_threshold','iou_threshold'):
        if profile.get(key)!=config[key]:raise ValueError('Unvalidated realtime setting: '+key)
    if list(config.get('source_dimensions') or [])!=profile.get('source_dimensions'):
        raise ValueError('Realtime input dimensions outside accepted profile; use shadow or offline')
    fps=config.get('source_fps')
    if not isinstance(fps,(float,int)) or not math.isfinite(fps) or fps<=0 or fps>profile.get('max_source_fps',0):
        raise ValueError('Realtime source rate outside accepted profile; use shadow or offline')
    return profile


def letterbox(frame, size=960):
    h,w = frame.shape[:2]
    scale = min(size/w,size/h)
    nw,nh = round(w*scale),round(h*scale)
    left,top = round((size-nw)/2-.1),round((size-nh)/2-.1)
    canvas = np.full((size,size,3),114,np.uint8)
    canvas[top:top+nh,left:left+nw] = cv2.resize(frame,(nw,nh))
    return cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB), {'gain':scale,'left':left,'top':top,'width':w,'height':h}


def decode_candidates(raw, transform, config):
    a = np.asarray(raw)
    if a.shape != (1,26,18900): raise ValueError('Unexpected racket output shape: '+str(a.shape))
    rows = a[0].T.astype(np.float32)
    finite = np.isfinite(rows).all(axis=1)
    diagnostics = {'candidates':len(rows),'nonfinite_candidates':int((~finite).sum()),
                   'nonfinite_values':int((~np.isfinite(rows)).sum())}
    # Filtering is for diagnostics only. ANY nonfinite raw candidate blocks measurement.
    selected = rows[finite & (rows[:,4]>=config['confidence_threshold'])]
    if not len(selected): return [],diagnostics
    boxes = selected[:,:4].copy()
    boxes[:,:2] -= boxes[:,2:]/2
    indices = cv2.dnn.NMSBoxes(boxes.tolist(),selected[:,4].tolist(),config['confidence_threshold'],config['iou_threshold'])
    result=[]
    for index in np.asarray(indices).reshape(-1):
        row=selected[int(index)]; box=boxes[int(index)].copy();box[2:]+=box[:2]
        box[[0,2]]=(box[[0,2]]-transform['left'])/transform['gain']
        box[[1,3]]=(box[[1,3]]-transform['top'])/transform['gain']
        box[[0,2]]=np.clip(box[[0,2]],0,transform['width'])
        box[[1,3]]=np.clip(box[[1,3]],0,transform['height'])
        if box[2]<=box[0] or box[3]<=box[1]:continue
        points=row[5:].reshape(7,3).copy()
        points[:,0]=(points[:,0]-transform['left'])/transform['gain']
        points[:,1]=(points[:,1]-transform['top'])/transform['gain']
        kpts={}
        for name,p in zip(KEYPOINT_NAMES,points):
            valid=bool(0<=p[0]<transform['width'] and 0<=p[1]<transform['height'] and config['keypoint_threshold']<=p[2]<=1)
            kpts[name]={'x':float(p[0]),'y':float(p[1]),'confidence':float(p[2]),'observed':valid}
        result.append({'box':box.tolist(),'confidence':float(row[4]),'keypoints':kpts,
                       'observed':bool(finite.all()),
                       'measurement_eligible':bool(finite.all() and kpts['grip_center']['observed'] and kpts['sweet_spot']['observed']),
                       'person_assignment_verified':False,'contact_verified':False,
                       'source':'racket_pose_model','coordinate_space':'original_source_pixels'})
    return result,diagnostics


class RacketPoseDetector:
    def __init__(self, config):
        self.config=pose_config(config);self.digest=model_digest(self.config['model_path'])
        if self.config['mode']=='active':
            verify_acceptance(self.config,self.digest)
            if self.config.get('processing_mode')=='realtime' and not self.config.get('acceptance_run',False):
                verify_pipeline_acceptance(self.config,self.digest)
        if self.config['backend']=='coreml':
            import coremltools as ct
            units={'ANE':ct.ComputeUnit.CPU_AND_NE,'ALL':ct.ComputeUnit.ALL,'CPU':ct.ComputeUnit.CPU_ONLY}
            self.model=ct.models.MLModel(self.config['model_path'],compute_units=units[self.config['compute_units']])
            spec=self.model.get_spec();self.output_name=spec.description.output[0].name
            image=spec.description.input[0]
            if (image.name!='image' or image.type.imageType.width!=960 or image.type.imageType.height!=960
                    or image.type.imageType.colorSpace!=ct.proto.FeatureTypes_pb2.ImageFeatureType.RGB):
                raise ValueError('Racket model requires RGB image 960x960')
        else:
            from ultralytics import YOLO
            self.model=YOLO(self.config['model_path'])
            if self.model.model.yaml.get('kpt_shape') != [7,3]:raise ValueError('Expected 7x3 racket topology')
            self.model.model.float().eval()

    def detect(self, frame, frame_id, source_time, prepared=None):
        started=time.perf_counter()
        if prepared is None:
            image,t=letterbox(frame,self.config['imgsz']);pre_ms=(time.perf_counter()-started)*1000
        else:
            image,t=frame,prepared['transform'];pre_ms=prepared['preprocess_ms']
        inference=time.perf_counter()
        if self.config['backend']=='coreml':
            from PIL import Image
            raw=self.model.predict({'image':Image.fromarray(image)})[self.output_name]
        else:
            import torch
            tensor=torch.from_numpy(image.transpose(2,0,1).copy()).float()[None]/255
            with torch.no_grad():
                output=self.model.model(tensor)
                raw=(output[0] if isinstance(output,tuple) else output).cpu().numpy()
        predict_ms=(time.perf_counter()-inference)*1000;post=time.perf_counter()
        candidates,diag=decode_candidates(raw,t,self.config)
        for c in candidates:
            c['source_frame_id']=frame_id
            c['view']='source_frame';c['view_assignment']='unassigned'
            for point in c['keypoints'].values():point['source_frame_id']=frame_id
        return {'schema_version':'tennis.racket-pose.v1','frame_id':frame_id,'source_time':source_time,
                'coordinate_space':'original_source_pixels','view':'source_frame','view_assignment':'unassigned','model_sha256':self.digest,
                'backend':self.config['backend'],'compute_units':self.config['compute_units'],
                'status':'numerical_failure' if diag['nonfinite_candidates'] else 'observed',
                'candidates':candidates,'diagnostics':diag,
                'timing':{'preprocess_ms':pre_ms,'predict_ms':predict_ms,'postprocess_ms':(time.perf_counter()-post)*1000,
                          'total_ms':(time.perf_counter()-started)*1000+(pre_ms if prepared is not None else 0)}}


def select_pose_candidate(observation, frame_id, wrists, frame_size, config):
    """Only source-matched, finite, qualified grip observations may change selection."""
    if not observation or observation.get('status')!='observed' or observation.get('frame_id')!=frame_id:return None
    valid_wrists=[]
    for p in wrists:
        if isinstance(p,dict):
            if p.get('observed') is not True or p.get('source_frame_id')!=frame_id:continue
            score=p.get('confidence')
            if not isinstance(score,(int,float)) or not math.isfinite(score) or score<config['keypoint_threshold']:continue
            xy=(p.get('x'),p.get('y'))
        else:xy=p
        if xy is not None and len(xy)>=2 and all(isinstance(v,(float,int)) and math.isfinite(v) for v in xy[:2]):valid_wrists.append(xy[:2])
    if not valid_wrists:return None
    limit=math.hypot(*frame_size)*config['max_wrist_distance_ratio'];rank=[]
    for candidate in observation.get('candidates',[]):
        if candidate.get('source_frame_id')!=frame_id:continue
        grip=(candidate.get('keypoints') or {}).get('grip_center') or {}
        if grip.get('source_frame_id')!=frame_id or not grip.get('observed') or grip.get('confidence',0)<config['keypoint_threshold']:continue
        distance=min(math.hypot(grip['x']-w[0],grip['y']-w[1]) for w in valid_wrists)
        if distance<=limit:rank.append((distance,candidate))
    if not rank:return None
    result=dict(min(rank,key=lambda item:item[0])[1]);result['observed']=True;result['source_frame_id']=frame_id
    result['association_view']=config.get('association_view','source_frame')
    result['association_method']='same_frame_wrist_proximity'
    return result


def draw_racket_pose(canvas, observation):
    if not observation or observation.get('status')!='observed':return canvas
    for candidate in observation.get('candidates',[]):
        pts=candidate.get('keypoints') or {}
        for a,b in SKELETON_EDGES:
            p,q=pts.get(KEYPOINT_NAMES[a],{}),pts.get(KEYPOINT_NAMES[b],{})
            if p.get('observed') and q.get('observed'):
                cv2.line(canvas,(round(p['x']),round(p['y'])),(round(q['x']),round(q['y'])),(0,220,255),2,cv2.LINE_AA)
        for name,p in pts.items():
            if p.get('observed'):cv2.circle(canvas,(round(p['x']),round(p['y'])),5,(0,0,255) if name=='sweet_spot' else (0,220,255),-1)
    return canvas
