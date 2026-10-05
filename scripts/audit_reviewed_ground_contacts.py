"""Offline audit of accepted foot candidates; never promote them to live truth."""
from __future__ import annotations
import argparse
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ground_reference import map_point, normalize_calibration

POLICY = 'reviewed_ground_candidates_v1_sensitivity_no_scoring'


def inside(point, polygon):
    turns = [(b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])
             for a,b in zip(polygon, polygon[1:]+polygon[:1])]
    return all(v >= -1e-8 for v in turns) or all(v <= 1e-8 for v in turns)


def angle_to_lane(a, b):
    # Unoriented foot line relative to AD (length axis); invariant to left/right swap.
    dx,dy=b[0]-a[0],b[1]-a[1]
    return math.degrees(math.atan2(abs(dx), abs(dy))) if math.hypot(dx,dy)>1e-9 else None


def perturbation_samples(matrix, pixel, polygon, radius):
    pixels = [[pixel[0]+dx, pixel[1]+dy] for dx,dy in itertools.product((-radius,0,radius),repeat=2)]
    if not all(inside(p,polygon) for p in pixels):
        return None  # Do not extrapolate sensitivity samples beyond the fitted ground patch.
    return [map_point(matrix,p) for p in pixels]


def pair_metrics(a,b,samples_a,samples_b):
    result={'distance_m':math.dist(a,b),'line_angle_to_lane_deg':angle_to_lane(a,b),'sensitivity':{}}
    for radius in (3,6):
        aa,bb=samples_a.get(str(radius)),samples_b.get(str(radius))
        if not aa or not bb:
            result['sensitivity'][str(radius)]={'reason':'perturbation_outside_ground_patch'}
            continue
        pairs=list(itertools.product(aa,bb))
        distances=[math.dist(x,y) for x,y in pairs]
        angles=[angle_to_lane(x,y) for x,y in pairs]
        result['sensitivity'][str(radius)]={'distance_sample_range_m':[min(distances),max(distances)],
            'angle_sample_range_deg':None if any(v is None for v in angles) else [min(angles),max(angles)]}
    return result


def analyze(labels, calibration, journal):
    cal=normalize_calibration(calibration)
    if labels.get('schema')!='tennis.ground-contact-review.v1' or labels.get('coordinate_space')!='original_source_pixels':
        raise ValueError('Unsupported review coordinate contract')
    if labels.get('confirmed') is not True or labels.get('independent_reference') is not False:
        raise ValueError('Requires accepted assisted review; do not reinterpret independent labels')
    if not cal['dimensions_measured'] or not cal['camera_geometry_confirmed']:
        raise ValueError('Measured dimensions and confirmed camera geometry required')
    if cal['binding'].get('kind')!='video_sha256' or cal['binding']['source_id']!=labels['source_sha256']:
        raise ValueError('Calibration actual source binding mismatch')
    frame_map={f['frame_id']:f for f in labels['frames']}
    if len(frame_map)!=len(labels['frames']):raise ValueError('Duplicate review frame')
    if any([f['width'],f['height']]!=cal['image_size'] for f in frame_map.values()):
        raise ValueError('Image size mismatch')
    rows={r['frame_id']:r for r in journal}
    if len(rows)!=len(journal) or not set(frame_map)<=set(rows):raise ValueError('Missing or duplicate source frames')
    expected={f'{fid}:{view}:{side}_contact' for fid in frame_map for view in ('front','back') for side in ('left','right')}
    if set(labels['labels'])!=expected:raise ValueError('Review is incomplete or contains extra identities')
    points=[];by_key={}
    for key,p in labels['labels'].items():
        if p.get('reviewed') is not True:raise ValueError('Unreviewed candidate')
        fid,view,side=key.split(':');fid=int(fid)
        state=p.get('contact_state')
        if state not in ('ground_contact_visible','unknown','airborne') or p.get('visible')!=(state=='ground_contact_visible'):
            raise ValueError('Contact state mismatch')
        item={'key':key,'frame_id':fid,'view':view,'side':side,'contact_state':state,
              'projected_xy_m':None,'samples':{},'reason':None,'accuracy_validated':False,
              'ground_contact_verified':False,'coaching_eligible':False}
        if not p['visible']:
            if p.get('x') is not None or p.get('y') is not None:raise ValueError('Absent contact must have null coordinates')
            item['reason']=state
        else:
            pixel=[p.get('x'),p.get('y')]
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in pixel):raise ValueError('Invalid contact pixel')
            if not (0<=pixel[0]<cal['image_size'][0] and 0<=pixel[1]<cal['image_size'][1]):raise ValueError('Pixel outside source image')
            geometry=cal['views'].get(view)
            if not geometry:item['reason']='calibration_view_missing'
            elif not inside(pixel,geometry['points']):item['reason']='outside_calibrated_ground_patch'
            else:
                item['image_xy']=pixel;item['projected_xy_m']=map_point(geometry['H'],pixel)
                item['samples']={str(radius):perturbation_samples(geometry['H'],pixel,geometry['points'],radius) for radius in (3,6)}
                hint=labels.get('ankle_references',{}).get(f"{fid}:{view}:{side.replace('_contact','_ankle')}")
                if hint and inside([hint['x'],hint['y']],geometry['points']):
                    item['ankle_to_contact_projection_difference_m']=math.dist(item['projected_xy_m'],map_point(geometry['H'],[hint['x'],hint['y']]))
                if view=='back' and not cal['correspondence_confirmed']:item['reason']='mirror_correspondence_unconfirmed_diagnostic_only'
        by_key[key]=item;points.append(item)
    pairs=[];steps=[];cross=[]
    for fid in sorted(frame_map):
        for view in ('front','back'):
            a,b=[by_key[f'{fid}:{view}:{side}_contact'] for side in ('left','right')]
            if all(p['projected_xy_m'] is not None for p in (a,b)):
                pairs.append({'frame_id':fid,'view':view,'diagnostic_only':view=='back' and not cal['correspondence_confirmed'],
                    **pair_metrics(a['projected_xy_m'],b['projected_xy_m'],a['samples'],b['samples'])})
            for side in ('left','right'):
                prev=by_key.get(f'{fid-1}:{view}:{side}_contact');current=by_key[f'{fid}:{view}:{side}_contact']
                if not prev or any(p['projected_xy_m'] is None for p in (prev,current)):continue
                times=[rows[f].get('source_time') or {} for f in (fid-1,fid)]
                if any(t.get('basis')!='media_pts' or t.get('quality')!='reported' or t.get('source_kind')!='video_file' or t.get('source_frame_id')!=f for f,t in zip((fid-1,fid),times)):continue
                ts=[t.get('timestamp_seconds') for t in times]
                if any(type(t) not in (int,float) or not math.isfinite(t) for t in ts) or ts[1]<=ts[0] or any(t.get('basis')!=times[0].get('basis') for t in times):continue
                result=pair_metrics(prev['projected_xy_m'],current['projected_xy_m'],prev['samples'],current['samples'])
                range6=result['sensitivity']['6'].get('distance_sample_range_m')
                radii=[max(math.dist(p['projected_xy_m'],s) for s in (p['samples'].get('6') or [p['projected_xy_m']])) for p in (prev,current)]
                steps.append({'frame_id':fid,'previous_frame_id':fid-1,'view':view,'side':side,'source_interval_ms':(ts[1]-ts[0])*1000,
                    'candidate_displacement_m':result['distance_m'],'sensitivity':result['sensitivity'],
                    'larger_than_6px_sample_envelope':bool(range6 is not None and result['distance_m']>sum(radii)),
                    'support_foot_change_verified':False})
        for side in ('left','right'):
            a,b=[by_key[f'{fid}:{view}:{side}_contact'] for view in ('front','back')]
            if all(p['projected_xy_m'] is not None for p in (a,b)):
                cross.append({'frame_id':fid,'side':side,'diagnostic_difference_m':math.dist(a['projected_xy_m'],b['projected_xy_m']),
                              'fusion_eligible':False,'reason':'mirror_correspondence_unconfirmed' if not cal['correspondence_confirmed'] else 'same_contact_point_semantics_and_independent_accuracy_unverified'})
    return {'schema':'tennis.reviewed-ground-audit.v1','policy_version':POLICY,'calibration_id':cal['calibration_id'],
            'mirror_correspondence_confirmed':cal['correspondence_confirmed'],'independent_check_point_count':len(cal['check_points']),
            'points':points,'foot_pairs':pairs,'consecutive_candidate_displacements':steps,'cross_view_diagnostics':cross,
            'accuracy_validated':False,'coaching_eligible':False,'runtime_modified':False,
            'sensitivity_semantics':'3/6 source-pixel perturbation samples only; not confidence intervals or calibrated error bounds; calibration error excluded',
            'position_semantics':'Visible shoe contact-region image candidate, not pressure center; pivot/contact-patch change can move candidate with stationary support.'}


def main():
    parser=argparse.ArgumentParser()
    for key in ('review','history','manifest','journal','source','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();paths={k:Path(getattr(args,k)) for k in ('review','history','manifest','journal','source')}
    hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    labels=json.loads(paths['review'].read_text());history=json.loads(paths['history'].read_text())
    if history['history'][-1]!=labels:raise ValueError('Reviewed file differs from submitted latest revision')
    if labels['source_sha256']!=hashes['source'] or labels['journal_sha256']!=hashes['journal']:raise ValueError('Evidence hash mismatch')
    manifest=json.loads(paths['manifest'].read_text())
    entries=[p for p in manifest['artifacts'] if p.get('role')=='frame_journal']
    if len(entries)!=1 or entries[0]['sha256']!=hashes['journal']:raise ValueError('Manifest journal mismatch')
    binding=manifest['session']['ground_calibration_application']['input_binding']
    if binding.get('kind')!='video_sha256' or binding['source_id']!=hashes['source']:raise ValueError('Manifest source mismatch')
    journal=[json.loads(line) for line in paths['journal'].read_text().splitlines() if line.strip()]
    result=analyze(labels,manifest['session']['ground_calibration'],journal)
    result['inputs_sha256']=hashes;result['review_revision']=labels['draft_revision'];result['annotator_id_missing']=not labels.get('annotator_id')
    result['generator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out=Path(args.output)
    if out.exists():raise FileExistsError('Use a new audit directory')
    out.mkdir(parents=True);(out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    cards=[]
    for view in ('front','back'):
        pairs=[p for p in result['foot_pairs'] if p['view']==view];steps=[p for p in result['consecutive_candidate_displacements'] if p['view']==view]
        distances=[p['distance_m'] for p in pairs];angles=[p['line_angle_to_lane_deg'] for p in pairs if p['line_angle_to_lane_deg'] is not None]
        summary={'mapped_points':sum(p['view']==view and p['projected_xy_m'] is not None for p in result['points']),
                 'paired_frames':len(pairs),'foot_distance_m_range':[min(distances),max(distances)] if distances else None,
                 'line_angle_to_lane_deg_range':[min(angles),max(angles)] if angles else None,
                 'consecutive_candidate_changes':len(steps),'larger_than_6px_sample_envelope_count':sum(p['larger_than_6px_sample_envelope'] for p in steps)}
        cards.append('<h2>'+view+'</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
    differences=[p['diagnostic_difference_m'] for p in result['cross_view_diagnostics']]
    cross_summary={'same_frame_same_named_foot_pairs':len(differences),
                   'median_diagnostic_difference_m':statistics.median(differences) if differences else None,
                   'max_diagnostic_difference_m':max(differences) if differences else None,
                   'fusion_eligible':False}
    cards.append('<h2>跨视角一致性诊断</h2><pre>'+html.escape(json.dumps(cross_summary,ensure_ascii=False,indent=2))+'</pre><p>差异可能来自四角对应、左右足身份、镜面图像几何或所选接地区域不同。不得用该差异直接给标注准确率或自动调整标定。</p>')
    table=''.join('<tr><td>'+str(p['frame_id'])+'</td><td>'+p['view']+'</td><td>'+f"{p['distance_m']:.3f}"+'</td><td>'+('—' if p['line_angle_to_lane_deg'] is None else f"{p['line_angle_to_lane_deg']:.1f}")+'</td><td>'+html.escape(str(p['sensitivity']['6'].get('distance_sample_range_m')))+'</td></tr>' for p in result['foot_pairs'])
    (out/'report.html').write_text('''<!doctype html><meta charset="utf-8"><title>已复核足部候选 · 地面映射审计</title><style>body{font:16px/1.6 system-ui;background:#111827;color:#eee;max-width:1200px;margin:30px auto;padding:20px}a{color:#67e8f9}td,th{padding:8px;text-align:left}pre{white-space:pre-wrap}</style><h1>已复核足部候选 · 地面映射审计</h1><p>150 个可见鞋底候选进入映射；不可辨认、越界和不在标定地面内的点不补齐。正面以 A 为原点、AB 为宽轴、AD 为长轴。足部连线角取与 AD 的无向夹角 0–90°，不推导开放/关闭站姿。</p><p>±3/±6 原图像素仅是定位扰动场景；下表范围为有限扰动采样范围，不是误差界或置信区间，不含四角误差。目前无独立实测检查点，镜中对应尚未确认，背面仅诊断，不融合。</p><p>连续帧变化只表示候选接地区域位置变化，转脚或接地面变化也会改变候选；不代表支撑脚真实移动、压力中心、速度或技术评分。标注来自人工批量接受的辅助草稿，无独立准确率证明。</p>'''+''.join(cards)+'''<table><tr><th>源帧</th><th>视角</th><th>足间距 m</th><th>与球道长轴夹角 °</th><th>±6px 距离采样范围 m</th></tr>'''+table+'''</table><p><a href="audit.json">逐点坐标、连续变化与跨视角诊断 JSON</a></p><p><a href="../cadence_ground_iteration_20261005_v2/report.html">已完成：原始 PTS 峰值敏感性与背面拒绝原因审计</a> · <a href="../ground_contact_review_accepted_20261005/report.html">标注接收记录</a></p><p>下一步：确认镜中四角对应、增加未参与拟合的实测地面检查点；再复核球拍及触球时间区间。实时轻量流程保持不变。</p>''')
    print(json.dumps({'mapped':sum(p['projected_xy_m'] is not None for p in result['points']),'pairs':len(result['foot_pairs']),'consecutive_changes':len(result['consecutive_candidate_displacements']),'cross_diagnostics':len(result['cross_view_diagnostics'])}))


if __name__=='__main__':main()
