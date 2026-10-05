"""Offline anomaly queue: raw pose proportions, continuity and local shoe-edge evidence."""
import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from observation_policy import qualified_front_point

POLICY='pose_contact_consistency_v1_review_queue_only'
JOINTS=[f'{side}_{part}' for part in ('shoulder','elbow','wrist','hip','knee','ankle') for side in ('left','right')]


def qualified_pose(row,view):
    fid=row['frame_id'];raw=(row.get('pose_observations') or {}).get(view) or {};points={};missing={}
    for name in JOINTS:
        p=raw.get(name);value,reason=qualified_front_point(p,fid,minimum_score=.5)
        if row.get('pose_observation_coordinate_space')!='original_source_pixels':reason='coordinate_space_unverified'
        if not isinstance(p,dict) or type(p.get('source_frame_id')) is not int:reason='source_frame_identity_missing'
        if reason:missing[name]=reason
        else:points[name]=value[:2]
    return points,missing


def dimensions(points):
    lengths={};dependencies={}
    for name,joints in {'shoulder_width':['left_shoulder','right_shoulder'],'hip_width':['left_hip','right_hip'],
                        'left_shin':['left_knee','left_ankle'],'right_shin':['right_knee','right_ankle'],
                        'left_torso':['left_shoulder','left_hip'],'right_torso':['right_shoulder','right_hip']}.items():
        if all(j in points for j in joints):lengths[name]=math.dist(*(points[j] for j in joints));dependencies[name]=joints
    shins=[lengths[n] for n in ('left_shin','right_shin') if lengths.get(n,0)>1]
    scale=statistics.median(shins) if shins else None
    return lengths,dependencies,scale


def robust_ratio_outlier(value,neighbors):
    valid=[math.log(x) for x in neighbors if x>0 and math.isfinite(x)]
    if value<=0 or len(valid)<5:return None
    center=statistics.median(valid);mad=statistics.median(abs(x-center) for x in valid)
    limit=max(math.log(1.35),4.5*1.4826*mad)
    return {'local_median_ratio':math.exp(center),'log_deviation':abs(math.log(value)-center),'threshold_log':limit} if abs(math.log(value)-center)>limit else None


def source_dt(previous,current):
    if current['frame_id']!=previous['frame_id']+1:return None
    times=[]
    for row in (previous,current):
        t=row.get('source_time') or {};value=t.get('timestamp_seconds')
        if (t.get('basis')!='media_pts' or t.get('quality')!='reported' or t.get('source_kind')!='video_file'
            or t.get('source_frame_id')!=row['frame_id'] or type(value) not in (int,float) or not math.isfinite(value)):return None
        times.append(value)
    return times[1]-times[0] if times[1]>times[0] else None


def shoe_edge_evidence(image,point,ankle,knee):
    """Local Canny is boundary evidence, NOT semantic shoe segmentation/contact truth."""
    leg=math.dist(ankle,knee)
    if leg<8:return {'status':'unavailable','reason':'shin_scale_too_short'}
    radius=max(12,int(.9*leg));ax,ay=ankle
    x0=max(0,int(ax-radius));y0=max(0,int(ay-.4*leg));x1=min(image.shape[1],int(ax+radius+1));y1=min(image.shape[0],int(ay+.8*leg+1))
    if x1<=x0 or y1<=y0:return {'status':'unavailable','reason':'shoe_region_outside_image'}
    crop=image[y0:y1,x0:x1];gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);gray=cv2.GaussianBlur(gray,(3,3),0)
    px,py=point;inside=x0<=px<x1 and y0<=py<y1
    distances=[];primary=None
    for low,high in ((30,90),(60,150)):
        edges=cv2.Canny(gray,low,high);yy,xx=np.nonzero(edges)
        distances.append(float(np.min(np.hypot(xx+x0-px,yy+y0-py)))/leg if len(xx) else None)
        if primary is None:primary=edges
    return {'status':'candidate_boundary_evidence','roi':[x0,y0,x1,y1],
            'contact_inside_ankle_based_roi':inside,'contact_to_ankle_shin_units':math.dist(point,ankle)/leg,
            'nearest_edge_shin_units_by_threshold':distances,
            'weak_edge_support':all(d is None or d>.08 for d in distances),
            'semantic_shoe_mask_available':False,'ground_contact_verified':False,'edges':primary}


def audit(rows,labels,images):
    by_id={r['frame_id']:r for r in rows}
    if len(by_id)!=len(rows):raise ValueError('Duplicate source frames')
    ids=sorted(f['frame_id'] for f in labels['frames']);cache={};ratios={};flags=[];foot=[];eligibility=[];clock_rejections=[]
    def flag(fid,view,joints,reason,evidence):flags.append({'frame_id':fid,'view':view,'joints':joints,'reason':reason,'evidence':evidence,'automatic_correction':False})
    for fid in ids:
        if fid not in by_id or fid not in images:raise ValueError('Missing selected source frame')
        for view in ('front','back'):
            p,missing=qualified_pose(by_id[fid],view);ls,deps,scale=dimensions(p);cache[fid,view]=(p,ls,deps,scale)
            eligibility.append({'frame_id':fid,'view':view,'qualified_joints':len(p),'rejected_joints':missing})
            for pair in ('shoulder','hip','ankle'):
                js=[f'left_{pair}',f'right_{pair}']
                if scale and all(j in p for j in js) and math.dist(*(p[j] for j in js))<.08*scale:
                    flag(fid,view,js,'bilateral_points_nearly_coincident',{'distance_shin_units':math.dist(*(p[j] for j in js))/scale})
        for dim in cache[fid,'front'][1].keys() & cache[fid,'back'][1].keys():
            a,b=cache[fid,'front'][1][dim],cache[fid,'back'][1][dim]
            if min(a,b)>1:ratios[fid,dim]=a/b
    for (fid,dim),ratio in ratios.items():
        neighbors=[]
        for offset in range(-5,6):
            other=fid+offset
            # Neighborhood must remain in one selected contiguous segment.
            if offset and all(f in ids for f in range(min(fid,other),max(fid,other)+1)) and (other,dim) in ratios:neighbors.append(ratios[other,dim])
        result=robust_ratio_outlier(ratio,neighbors)
        if result:flag(fid,'both',cache[fid,'front'][2][dim],'cross_view_ratio_temporal_outlier',{'dimension':dim,'front_back_ratio':ratio,**result})
    for fid in ids:
        if fid-1 not in ids:continue
        prev,current=by_id[fid-1],by_id[fid];dt=source_dt(prev,current)
        neighboring=[source_dt(by_id[f-1],by_id[f]) for f in range(fid-5,fid+6) if f in by_id and f-1 in by_id]
        good=[x for x in neighboring if x is not None];median_dt=statistics.median(good) if good else None
        if dt is None or median_dt is None or dt<median_dt/4:
            clock_rejections.append({'frame_id':fid,'dt_seconds':dt,'reason':'invalid_or_short_source_interval'});continue
        for view in ('front','back'):
            p,_,_,scale=cache[fid,view];q,_,_,oldscale=cache[fid-1,view]
            if not scale or not oldscale:continue
            scale=(scale+oldscale)/2
            hips=['left_hip','right_hip']
            shift=np.mean([np.asarray(p[j])-q[j] for j in hips],axis=0) if all(j in p and j in q for j in hips) else np.zeros(2)
            for j in p.keys() & q.keys():
                displacement=float(np.linalg.norm(np.asarray(p[j])-q[j]-shift))/scale
                if displacement>.45 and displacement/dt>6:
                    flag(fid,view,[j],'joint_relative_jump',{'previous_frame_id':fid-1,'displacement_shin_units':displacement,'source_interval_ms':dt*1000})
            for part in ('shoulder','hip','knee','ankle'):
                a,b=f'left_{part}',f'right_{part}'
                if all(j in p and j in q for j in (a,b)):
                    direct=sum(math.dist(p[j],q[j]) for j in (a,b));swapped=math.dist(p[a],q[b])+math.dist(p[b],q[a])
                    if direct-swapped>.3*scale and swapped<.55*direct:
                        flag(fid,view,[a,b],'possible_left_right_swap',{'previous_frame_id':fid-1,'direct_cost_px':direct,'swapped_cost_px':swapped})
    for key,contact in labels['labels'].items():
        fid,view,joint=key.split(':');fid=int(fid)
        if not contact.get('visible'):continue
        p=cache[fid,view][0];side=joint.split('_')[0];ankle=side+'_ankle';knee=side+'_knee'
        if ankle not in p or knee not in p:
            foot.append({'key':key,'status':'unavailable','reason':'ankle_or_knee_not_qualified'});continue
        ev=shoe_edge_evidence(images[fid],[contact['x'],contact['y']],p[ankle],p[knee]);ev.pop('edges',None);foot.append({'key':key,**ev})
        if ev['status']=='unavailable':continue
        reasons=[]
        if not ev['contact_inside_ankle_based_roi'] or ev['contact_to_ankle_shin_units']>.6:reasons.append('contact_far_from_same_side_ankle')
        if ev['weak_edge_support']:reasons.append('weak_local_boundary_support')
        other=('right' if side=='left' else 'left')+'_ankle'
        if other in p and math.dist([contact['x'],contact['y']],p[other])+.15*math.dist(p[ankle],p[knee])<math.dist([contact['x'],contact['y']],p[ankle]):reasons.append('contact_closer_to_opposite_ankle')
        for reason in reasons:flag(fid,view,[joint,ankle,knee],reason,{k:v for k,v in ev.items() if k!='status'})
    return {'schema':'tennis.pose-contact-consistency.v1','policy_version':POLICY,'flags':flags,'observation_eligibility':eligibility,
            'body_ratios':[{'frame_id':fid,'dimension':dim,'front_back_ratio':value} for (fid,dim),value in sorted(ratios.items())],
            'shoe_boundary_checks':foot,'clock_rejections':clock_rejections,'accuracy_validated':False,'calibration_modified':False,
            'limitations':['Ratios are projected lengths; turning and occlusion can cause legitimate outliers.',
                          'Ankle ROIs depend on model joints and may be wrong; edge support is not a semantic shoe mask.',
                          'Edges include seams, clothing and foreground people; strong edge support never confirms contact.',
                          'Thresholds are uncalibrated review heuristics, not accuracy probabilities.']}


def main():
    parser=argparse.ArgumentParser()
    for k in ('journal','labels','source','frames','output'):parser.add_argument('--'+k,required=True)
    a=parser.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Use new revision directory')
    paths={k:Path(getattr(a,k)) for k in ('journal','labels','source')};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    labels=json.loads(paths['labels'].read_text())
    if labels['source_sha256']!=hashes['source'] or labels['journal_sha256']!=hashes['journal']:raise ValueError('Source or journal hash mismatch')
    rows=[json.loads(x) for x in paths['journal'].read_text().splitlines() if x.strip()];ids={f['frame_id'] for f in labels['frames']};images={};frame_hashes={}
    cap=cv2.VideoCapture(str(paths['source']));fid=0
    while True:
        ok,image=cap.read()
        if not ok:break
        if fid in ids:
            path=Path(a.frames)/f'frame_{fid}.png';saved=cv2.imread(str(path))
            if saved is None or not np.array_equal(saved,image):raise ValueError(f'Sequential decode disagrees with review frame {fid}')
            images[fid]=image;frame_hashes[str(fid)]=hashlib.sha256(path.read_bytes()).hexdigest()
        fid+=1
    cap.release()
    if set(images)!=ids:raise ValueError('Source decode incomplete')
    result=audit(rows,labels,images);result.update(inputs_sha256=hashes,frame_png_sha256=frame_hashes,sequential_source_decode_verified=True,generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.mkdir(parents=True);(out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    byfid={}
    for f in result['flags']:byfid.setdefault(f['frame_id'],[]).append(f)
    titles={'cross_view_ratio_temporal_outlier':'双视角比例局部异常','joint_relative_jump':'关节相对位置跳变','possible_left_right_swap':'左右点疑似交换','bilateral_points_nearly_coincident':'左右点近乎重合','contact_far_from_same_side_ankle':'接地点远离同侧脚踝','weak_local_boundary_support':'接地点附近边缘支持弱','contact_closer_to_opposite_ankle':'接地点更接近另一侧脚踝'}
    cards=[]
    rowmap={r['frame_id']:r for r in rows}
    for fid,flags in sorted(byfid.items(),key=lambda x:(-len(x[1]),x[0])):
        image=images[fid].copy()
        for view in ('front','back'):
            p,_=qualified_pose(rowmap[fid],view)
            for name,xy in p.items():cv2.circle(image,tuple(int(x) for x in xy),4,(0,200,255),-1)
        for key,c in labels['labels'].items():
            if key.startswith(str(fid)+':') and c.get('visible'):
                xy=(round(c['x']),round(c['y']));cv2.circle(image,xy,7,(255,255,0),2);cv2.putText(image,key.split(':')[2][0].upper(),(xy[0]+10,xy[1]),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,0),2)
        cv2.imwrite(str(out/f'frame_{fid}_evidence.jpg'),image)
        items=''.join('<li>'+html.escape(f['view']+' · '+','.join(f['joints'])+' · '+titles[f['reason']])+'<details><summary>触发数值</summary><pre>'+html.escape(json.dumps(f['evidence'],ensure_ascii=False,indent=2))+'</pre></details></li>' for f in flags)
        crops=[]
        contact_targets={(f['view'],j) for f in flags for j in f['joints'] if j.endswith('_contact') and f['view'] in ('front','back')}
        for view,joint in sorted(contact_targets):
            contact=labels['labels'][f'{fid}:{view}:{joint}'];p,_=qualified_pose(rowmap[fid],view);side=joint.split('_')[0]
            if not contact.get('visible') or side+'_ankle' not in p or side+'_knee' not in p:continue
            ev=shoe_edge_evidence(images[fid],[contact['x'],contact['y']],p[side+'_ankle'],p[side+'_knee'])
            if ev['status']=='unavailable':continue
            x0,y0,x1,y1=ev['roi'];crop=images[fid][y0:y1,x0:x1].copy();crop[ev['edges']>0]=(60,200,60)
            cv2.circle(crop,(round(contact['x']-x0),round(contact['y']-y0)),4,(255,255,0),1)
            filename=f'foot_{fid}_{view}_{side}.png';cv2.imwrite(str(out/filename),crop)
            crops.append(f'<figure><figcaption>{view} {side} · 脚踝邻域；绿线是图像边缘，青圈是接地点（区域外的点可能不在小图内）</figcaption><img style="width:300px;image-rendering:auto" loading="lazy" src="{filename}"></figure>')
        cards.append(f'<section data-frame="{fid}"><h2>源帧 {fid} · {len(flags)} 个提示</h2><img loading="lazy" src="frame_{fid}_evidence.jpg"><ul>{items}</ul>'+''.join(crops)+'</section>')
    counts=dict(Counter(f['reason'] for f in result['flags']))
    (out/'report.html').write_text('''<!doctype html><meta charset="utf-8"><title>人体比例、连续性与足部边界复核队列</title><style>body{font:16px/1.6 system-ui;background:#111827;color:#eee;max-width:1300px;margin:30px auto;padding:20px}a{color:#67e8f9}section{background:#1e293b;padding:20px;margin:20px 0}img{width:100%}pre{white-space:pre-wrap}button,input{padding:10px}</style><h1>人体比例＋连续性＋足部边界复核队列</h1><p>审计66个已选源帧；逐帧图与原片顺序解码一致。仅使用原始独立双视角观测，低分、旧帧、镜面补点不参加几何检查。黄色为合格原始关节点，青色L/R为当前鞋底候选。</p><p>同帧肩宽/髋宽/躯干侧长/小腿长度的前后视图比例，只与邻近连续帧比较；没有固定人体比例尺。短PTS间隔及无效时钟不参加跳变判断。全部阈值为未校准复核启发式。</p><p>足部检查使用脚踝附近候选区域和两组Canny边缘阈值；这不是鞋子语义分割，地砖缝、衣物和前景人体也会有边缘。靠近边缘不等于接地正确，缺少可靠脚踝/膝时显示不可检查。所有提示均需原图复核，不自动改点或四角。</p><p>筛选源帧：<input id="fid" type="number"><button onclick="for(const s of document.querySelectorAll('section'))s.hidden=document.getElementById('fid').value&&s.dataset.frame!==document.getElementById('fid').value">筛选</button></p><pre>'''+html.escape(json.dumps({'flagged_frames':len(byfid),'flags_by_reason':counts},ensure_ascii=False,indent=2))+'''</pre><p><a href="audit.json">全部数值与观测拒绝原因</a> · <a href="../mirror_right_contact_fix_20261005/index.html">鞋底候选修正页</a></p>'''+''.join(cards))
    print(json.dumps({'frames':len(ids),'flagged_frames':len(byfid),'flags':counts,'output':str(out)},ensure_ascii=False))

if __name__=='__main__':main()
