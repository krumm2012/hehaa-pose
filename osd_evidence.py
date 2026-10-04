"""Auditable display qualification for 2D observations, not accuracy calibration."""
import math
from statistics import median
from image_motion_measurements import source_timestamp

POLICY = 'osd_observation_qualification_v5_field_source_identity'
PARAMETERS = {'minimum_coverage': .8, 'minimum_point_score': .5,
              'minimum_ankle_span_body_width': .2, 'maximum_foot_range_deg': 10,
              'hip_motion_floor_body_width': .04}
FIELDS = {'brush_angle': ('low_to_high_angle_deg','drop_depth_px','drop_depth_ratio'),
          'stance': ('image_foot_line_angle_deg',), 'leg_drive': ('drive_px','drive_ratio')}


def _points(row, names):
    result=[]
    for name in names:
        p=((row.get('pose_observations') or {}).get('front') or {}).get(name,{})
        if p.get('observed') is not True or p.get('recovered_from_mirror') or p.get('confidence_source') == 'unavailable':
            return None
        if p.get('source_frame_id') is not None and p['source_frame_id'] != row.get('frame_id'):
            return None
        values=[p.get(k) for k in ('x','y','confidence')]
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values) or values[2]<.5:
            return None
        result.append(values)
    return result


def qualify_extended_observations(ext, event, frames, features, scale):
    """Retain raw estimates separately; excluded values cannot leak into consumers."""
    scale_valid = isinstance(scale, (int, float)) and not isinstance(scale, bool) and math.isfinite(scale) and scale > 0
    contact=int(event['contact_frame'])
    rows={r['frame_id']:r for r in frames}
    feats={r['frame_id']:r for r in features}
    anchor=rows.get(contact,{})
    anchor_time,basis=source_timestamp(anchor)
    times={i:source_timestamp(r) for i,r in rows.items()}
    timed = anchor_time is not None and basis in ('media_pts','nominal_fps')
    def window(before,after):
        if not timed: return []
        return [i for i,(t,b) in sorted(times.items()) if t is not None and b==basis and anchor_time-before-1e-9<=t<=anchor_time+after+1e-9]
    invalid_time_ids = [i for i,(t,b) in times.items() if t is None or b != basis]
    ordered_times = [t for _,(t,b) in sorted(times.items()) if t is not None and b == basis]
    time_discontinuous = any(b <= a for a,b in zip(ordered_times, ordered_times[1:]))
    def record(key,ids,valid,reasons,details):
        m=ext[key]
        if not scale_valid: reasons.append('body_scale_unavailable')
        m['raw_estimate']={f:m.get(f) for f in FIELDS[key]}
        # Invalid clock rows cannot be located in a source-time window. Include
        # them conservatively and abstain rather than invent their timestamps.
        denominator_ids = sorted(set(ids) | set(invalid_time_ids))
        missing=[i for i in denominator_ids if i not in valid]
        if invalid_time_ids: reasons.append('incomplete_source_time')
        if time_discontinuous: reasons.append('nonmonotonic_source_time')
        if any(b-a != 1 for a,b in zip(ids,ids[1:])): reasons.append("source_frame_gap")
        if not timed: reasons.append('source_time_unavailable')
        if not ids or len(valid)/max(1,len(denominator_ids))<.8: reasons.append('low_observation_coverage')
        m['measurement_evidence']={'policy':POLICY,'display_eligible':not reasons,
            'status':'image_observation_only' if not reasons else 'insufficient_evidence',
            'reasons':list(dict.fromkeys(reasons)), 'source_frames':valid,
            'window_frames':ids, 'missing_frames':missing,
            'coverage_denominator_frames':denominator_ids, 'unlocated_time_frames':invalid_time_ids,
            'sample_count':len(valid), 'coverage':round(len(valid)/len(denominator_ids),3) if denominator_ids else 0,
            'time_basis':basis,'reference_width_px':scale,'parameters':PARAMETERS,
            'accuracy_validated':False, **details}
        if reasons:
            for field in FIELDS[key]: m[field]=None
        field_evidence={}
        for field in FIELDS[key]:
            field_reasons=list(dict.fromkeys(reasons))
            if key=='brush_angle' and field in ('drop_depth_px','drop_depth_ratio'):
                field_reasons=[reason for reason in field_reasons if reason!='path_endpoints_coincide']
                if not field_reasons:
                    m[field]=m['raw_estimate'][field]
            field_evidence[field]={'display_eligible':not field_reasons,'reasons':field_reasons}
        m['measurement_evidence']['fields']=field_evidence
    contact_status=event.get('contact_status')
    anchor_reasons=[] if contact_status in ('candidate','confirmed') else ['contact_anchor_unconfirmed']
    ids=window(.48,0)
    valid=[]
    for i in ids:
        detections=rows[i].get('rackets') or []
        d=detections[0] if detections else {}
        if feats.get(i,{}).get('racket_measurement_point') is not None and d.get('observed') is True and (d.get('confidence') or 0)>=.5 and (d.get('source_frame_id') is None or d['source_frame_id'] == i):
            valid.append(i)
    reasons=list(anchor_reasons)
    if len(valid)<5: reasons.append('too_few_racket_observations')
    if contact not in valid: reasons.append('contact_racket_missing')
    if valid and any(b-a>2 for a,b in zip(valid,valid[1:])): reasons.append('racket_track_has_gaps')
    # Compute the chord from this same time window, never a separate N-frame window.
    points=[(i,feats[i]['racket_measurement_point']) for i in valid]
    endpoint=feats.get(contact,{}).get('racket_measurement_point')
    if points and contact in valid:
        low_id, low=max(points,key=lambda p:p[1][1])
        dy=max(0,low[1]-endpoint[1]);dx=abs(low[0]-endpoint[0])
        ext['brush_angle']['low_to_high_angle_deg']=round(math.degrees(math.atan2(dy,dx)),1) if dx or dy else None
        ext['brush_angle']['drop_depth_px']=round(dy,1)
        ext['brush_angle']['drop_depth_ratio']=round(dy/scale,3) if scale and scale>0 else None
        if not dx and not dy: reasons.append('path_endpoints_coincide')
    else:
        low_id=None
    record('brush_angle',ids,valid,reasons,{'definition':'box-centre chord from lowest observed point to contact candidate; not brush/spin', 'window_seconds':[-.48,0], 'path_endpoint_frames':[low_id,contact]})

    ids=window(.12,.12);valid=[];angles=[];spans=[]
    for i in ids:
        pts=_points(rows[i],('left_ankle','right_ankle'))
        if pts:
            dx,dy=pts[1][0]-pts[0][0],pts[1][1]-pts[0][1]
            span=math.hypot(dx,dy)
            if scale_valid and span >= .2 * scale:
                valid.append(i);spans.append(span);angles.append(math.degrees(math.atan2(abs(dy),abs(dx))))
    reasons=[]
    if len(valid)<3: reasons.append('too_few_ankle_observations')
    if angles and max(angles)-min(angles)>10: reasons.append('unstable_ankle_line')
    # Recompute from the qualified raw ankles in a source-time window.
    ext['stance']['image_foot_line_angle_deg']=round(median(angles),1) if angles else None
    record('stance',ids,valid,reasons,{'definition':'image ankle-line inclination; not court stance',
        'window_seconds':[-.12,.12], 'range_deg':[round(min(angles),1),round(max(angles),1)] if angles else None,
        'minimum_span_px':round(min(spans),1) if spans else None})

    ids=window(.6,0);valid=[];ys=[]
    for i in ids:
        pts=_points(rows[i],('left_hip','right_hip'))
        if pts: valid.append(i);ys.append((pts[0][1]+pts[1][1])/2)
    reasons=list(anchor_reasons)
    if len(valid)<5: reasons.append('too_few_hip_observations')
    if contact not in valid: reasons.append('contact_hips_missing')
    floor = .04 * scale if scale_valid else math.inf
    rise=max(ys)-ys[valid.index(contact)] if contact in valid else None
    if rise is None or rise<=floor: reasons.append('below_motion_resolution_guard')
    ext['leg_drive']['drive_px']=round(rise,1) if rise is not None else None
    ext['leg_drive']['drive_ratio']=round(rise/scale,3) if rise is not None and scale and scale>0 else None
    record('leg_drive',ids,valid,reasons,{'definition':'image hip-midpoint rise; not leg force or percentage effort',
        'window_seconds':[-.6,0], 'motion_guard_px':round(floor,2) if scale_valid else None,
        'guard_is_heuristic_not_statistical_uncertainty':True})
    return ext


def display_value(metric, field):
    evidence=metric.get('measurement_evidence') or {}
    evidence=(evidence.get('fields') or {}).get(field,evidence)
    return metric.get(field) if evidence.get('display_eligible') is True else None


def sequence_osd_label(seq):
    cross=(seq.get('cross_validation') or {}).get('status')
    if cross=='disagree': return '双视角冲突·不判定'
    if cross in (None,'unavailable','legacy_single_view'): return '二维参考·证据不足·未验证'
    dt=seq.get('latency_hip_to_shoulder_ms')
    uncertainty=seq.get('peak_time_uncertainty_ms')
    pair = (seq.get('pair_timing') or {}).get('hip_to_shoulder')
    unresolved=(pair is not None and not pair['resolved']) or seq.get('sequence_quality')=='UNRESOLVED_AT_FRAME_RATE' or (
        dt is not None and uncertainty is not None and abs(dt)<=uncertainty)
    if seq.get('racket_peak_frame') is None:
        return '髋肩先后难辨·缺拍峰' if unresolved else '仅髋肩·缺拍峰'
    if unresolved: return '二维峰值先后难辨'
    if seq.get('sequence_quality') == 'PROJECTED_REVERSE_ORDER': return '投影峰值反序·待复核'
    return '双视角时序·未验证' if cross=='agree' else '单视角时序·未验证'


def evidence_label(metric):
    reasons=(metric.get('measurement_evidence') or {}).get('reasons') or []
    if 'below_motion_resolution_guard' in reasons: return '位移先不作解读'
    if 'contact_anchor_unconfirmed' in reasons: return '触球待确认'
    if 'low_observation_coverage' in reasons or 'racket_track_has_gaps' in reasons: return '观测不连续'
    if 'path_endpoints_coincide' in reasons: return '轨迹方向不明确'
    return '观测证据不足'
