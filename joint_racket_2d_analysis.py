"""Source-bound 2D body/racket candidates and missing-observation diagnostics.

No inference, interpolation, 3D conversion or automatic technical scoring.
"""
import math
from collections import Counter
from statistics import median

POLICY = 'joint_racket_2d_v1'
RACKET_POINTS = ('butt_cap', 'grip_center', 'throat', 'sweet_spot', 'head_top', 'frame_left', 'frame_right')
BODY_POINTS = tuple(f'{side}_{joint}' for side in ('left', 'right')
                    for joint in ('shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle'))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def qualified_point(point, frame_id, threshold=.5):
    if not isinstance(point, dict):
        return None, 'missing_point'
    if point.get('source_frame_id') != frame_id or type(point.get('source_frame_id')) is not int:
        return None, 'source_frame_mismatch'
    if point.get('observed') is not True or point.get('recovered_from_mirror'):
        return None, 'derived_or_unobserved'
    if point.get('confidence_source') == 'unavailable':
        return None, 'score_unavailable'
    score = point.get('confidence')
    if not finite(score) or not 0 <= score <= 1:
        return None, 'invalid_score'
    if score < threshold:
        return None, 'low_score'
    if not finite(point.get('x')) or not finite(point.get('y')):
        return None, 'invalid_coordinates'
    return [point['x'], point['y']], None


def source_seconds(row):
    source = row.get('source_time') or {}
    if (source.get('schema_version') == 'tennis.source-time.v1'
            and source.get('source_frame_id') == row['frame_id']
            and source.get('source_kind') == 'video_file'
            and source.get('basis') == 'media_pts' and source.get('quality') == 'reported'
            and finite(source.get('timestamp_seconds')) and source['timestamp_seconds'] >= 0):
        return source['timestamp_seconds']
    return None


def associate_rackets(row, bodies, size, distance_ratio=.05, ambiguity_ratio=.005):
    """Independent wrist/grip proximity candidates, rejecting duplicate view use."""
    fid = row['frame_id']; obs = row.get('racket_pose') or {}
    limit = math.hypot(*size) * distance_ratio
    margin = math.hypot(*size) * ambiguity_ratio
    candidates = obs.get('candidates') or []
    result = {}
    for view in ('front', 'back'):
        wrists = [bodies[view].get(side+'_wrist') for side in ('left', 'right')]
        wrists = [w for w in wrists if w is not None]
        eligible = []
        reason = 'no_qualified_wrist' if not wrists else 'no_qualified_racket_grip'
        if (wrists and obs.get('frame_id') == fid and obs.get('status') == 'observed'
                and obs.get('coordinate_space') == 'original_source_pixels'):
            for index, candidate in enumerate(candidates):
                if (candidate.get('source') != 'racket_pose_model' or candidate.get('observed') is not True
                        or candidate.get('source_frame_id') != fid
                        or not finite(candidate.get('confidence')) or not .25 <= candidate['confidence'] <= 1):
                    continue
                grip, _ = qualified_point((candidate.get('keypoints') or {}).get('grip_center'), fid)
                if grip is not None:
                    distance = min(math.dist(w, grip) for w in wrists)
                    if distance <= limit:
                        eligible.append((distance, index))
        eligible.sort()
        if eligible and (len(eligible) == 1 or eligible[1][0]-eligible[0][0] >= margin):
            distance, index = eligible[0]
            result[view] = {'status':'proximity_candidate', 'candidate_index':index,
                            'grip_wrist_distance_px':distance, 'reason':None,
                            'person_assignment_verified':False}
        else:
            result[view] = {'status':'unavailable','candidate_index':None,
                            'reason':'ambiguous_grip_assignment' if eligible else reason}
    if (result['front']['candidate_index'] is not None
            and result['front']['candidate_index'] == result['back']['candidate_index']):
        for view in result:
            result[view] = {'status':'unavailable','candidate_index':None,
                            'reason':'same_candidate_competes_for_both_views'}
    return result


def angle(a, b, c):
    if any(p is None for p in (a, b, c)):
        return None
    u = (a[0]-b[0], a[1]-b[1]);v = (c[0]-b[0], c[1]-b[1])
    norm = math.hypot(*u)*math.hypot(*v)
    if norm <= 1e-9:
        return None
    return math.degrees(math.acos(max(-1, min(1, (u[0]*v[0]+u[1]*v[1])/norm))))


def line_angle(a, b):
    if a is None or b is None or math.dist(a,b) < 12:
        return None
    return math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))


def analyze_frames(rows, image_size):
    from ball_observation_contract import measurement_ball
    ordered = sorted(rows, key=lambda r:r['frame_id'])
    ids = [r['frame_id'] for r in ordered]
    if any(type(fid) is not int or fid < 0 for fid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Source frame identities must be unique nonnegative integers')
    times = [source_seconds(r) for r in ordered]
    dts = [b-a for a,b in zip(times,times[1:]) if a is not None and b is not None and b>a]
    cadence = median(dts) if dts else None
    frames = []
    for row, timestamp in zip(ordered,times):
        fid = row['frame_id'];bodies={};reasons={};tracks={}
        if row.get('pose_observation_coordinate_space') != 'original_source_pixels':
            observations = {}
        else:
            observations = row.get('pose_observations') or {}
        for view in ('front','back'):
            bodies[view] = {}
            for name in BODY_POINTS:
                xy, reason = qualified_point((observations.get(view)or{}).get(name),fid)
                bodies[view][name]=xy;key=f'{view}.body.{name}'
                tracks[key]=xy;reasons[key]=reason
        assoc=associate_rackets(row,bodies,image_size)
        obs=row.get('racket_pose')or{}; candidates=obs.get('candidates')or[]
        metrics={};rackets={}
        for view in ('front','back'):
            index=assoc[view]['candidate_index'];rackets[view]={}
            candidate=candidates[index] if index is not None else {}
            for name in RACKET_POINTS:
                xy, reason = qualified_point((candidate.get('keypoints')or{}).get(name),fid)
                if index is None:reason=assoc[view]['reason']
                rackets[view][name]=xy;key=f'{view}.racket.{name}'
                tracks[key]=xy;reasons[key]=reason
            body=bodies[view];rkt=rackets[view]
            width=math.dist(body['left_shoulder'],body['right_shoulder']) if body['left_shoulder'] and body['right_shoulder'] else None
            if width is not None and width<12:width=None
            hip=line_angle(body['left_hip'],body['right_hip']);shoulder=line_angle(body['left_shoulder'],body['right_shoulder'])
            metrics[view]={'body_width_px':width,'hip_line_deg':hip,'shoulder_line_deg':shoulder,
                'hip_shoulder_projection_deg':abs((shoulder-hip+90)%180-90) if hip is not None and shoulder is not None else None,
                'shaft_image_angle_deg':line_angle(rkt['butt_cap'],rkt['head_top']),
                'grip_wrist_gap_body_widths':assoc[view].get('grip_wrist_distance_px')/width if width and index is not None else None}
            for side in ('left','right'):
                metrics[view][side+'_elbow_deg']=angle(body[side+'_shoulder'],body[side+'_elbow'],body[side+'_wrist'])
                metrics[view][side+'_knee_deg']=angle(body[side+'_hip'],body[side+'_knee'],body[side+'_ankle'])
        for view in ('front','back'):
            for group,names in (('arm_chain',('right_shoulder','right_elbow','right_wrist')),('left_arm_chain',('left_shoulder','left_elbow','left_wrist')),('torso',('left_hip','right_hip','left_shoulder','right_shoulder'))):
                key=f'{view}.complete.{group}';tracks[key]=[0,0] if all(bodies[view][name] is not None for name in names) else None;reasons[key]=None if tracks[key] is not None else 'incomplete_body_chain'
            key=f'{view}.complete.seven_points';tracks[key]=[0,0] if all(rackets[view][name] is not None for name in RACKET_POINTS) else None;reasons[key]=None if tracks[key] is not None else 'incomplete_associated_racket'
        ball, origin=measurement_ball(row)
        if origin != 'fresh_model_observation':ball=None
        tracks['ball']=ball;reasons['ball']=None if ball else 'no_qualified_active_ball'
        tracks['racket_detector_any']=[0,0] if obs.get('frame_id')==fid and obs.get('status')=='observed' and candidates else None
        reasons['racket_detector_any']=None if tracks['racket_detector_any'] else 'no_raw_racket_candidate'
        for view in ('front','back'):
            metrics[view]['ball_sweetspot_distance_px']=math.dist(ball,rackets[view]['sweet_spot']) if view=='front' and ball is not None and rackets[view]['sweet_spot'] is not None else None
        frame={'frame_id':fid,'source_seconds':timestamp,'body':bodies,'racket':rackets,'associations':assoc,
               'ball':ball,'metrics':metrics,'tracks':tracks,'missing_reasons':reasons}
        previous=frames[-1] if frames else None
        for view in ('front','back'):
            m=metrics[view];m.update(sweet_spot_speed_px_s=None,wrist_speed_px_s=None)
            dt=timestamp-previous['source_seconds'] if previous and timestamp is not None and previous['source_seconds'] is not None else None
            valid_dt=(dt is not None and cadence is not None and .25*cadence<dt<=3*cadence
                      and fid==previous['frame_id']+1)
            m['velocity_status']='qualified_source_interval' if valid_dt else 'missing_or_unstable_source_interval'
            for metric,point in (('sweet_spot_speed_px_s',view+'.racket.sweet_spot'),('wrist_speed_px_s',view+'.body.right_wrist')):
                a=previous['tracks'][point] if previous else None;b=tracks[point]
                if valid_dt and a is not None and b is not None:m[metric]=math.dist(a,b)/dt
        frames.append(frame)
    return frames,cadence


def missing_runs(frames, key, minimum_length=3):
    """Observation gaps, split at absent journal IDs; never imply decoded loss."""
    runs=[];current=[]
    def close():
        if len(current)>=minimum_length:
            first,last=current[0],current[-1]
            runs.append({'track':key,'start_frame':first['frame_id'],'end_frame':last['frame_id'],
                         'count':len(current),'start_seconds':first['source_seconds'],'end_seconds':last['source_seconds'],
                         'pts_span_ms':(last['source_seconds']-first['source_seconds'])*1000
                         if first['source_seconds'] is not None and last['source_seconds'] is not None else None,
                         'reasons':dict(Counter(f['missing_reasons'].get(key) for f in current))})
    for f in frames:
        if current and f['frame_id']!=current[-1]['frame_id']+1:close();current=[]
        if f['tracks'].get(key) is None:current.append(f)
        else:close();current=[]
    close();return runs


def summarize(frames, events, expected_source_frame_count=None):
    keys=list(frames[0]['tracks']) if frames else []
    summary=[];runs=[]
    for key in keys:
        all_gaps=missing_runs(frames,key,1)
        gaps=[gap for gap in all_gaps if gap['count']>=3]
        for gap in gaps:
            gap['event_ids']=[e['event_id'] for e in events if gap['end_frame']>=e['start_frame'] and gap['start_frame']<=e['end_frame']]
            contact=[]
            for e in events:
                row=next((f for f in frames if f['frame_id']==e['contact_frame']),None)
                t=row['source_seconds'] if row else None
                if t is not None and gap['start_seconds'] is not None and gap['end_seconds'] is not None:
                    if gap['end_seconds']>=t-.12 and gap['start_seconds']<=t+.12:contact.append(e['event_id'])
            gap['near_contact_event_ids']=contact
            gap['priority']='contact_window' if contact else 'swing_window' if gap['event_ids'] else 'outside_event'
        runs.extend(gaps)
        n=sum(f['tracks'].get(key) is not None for f in frames)
        summary.append({'track':key,'qualified_frames':n,'total_frames':len(frames),
                        'coverage':n/len(frames) if frames else 0,'runs_ge_3':len(gaps),
                        'exactly_3':sum(g['count']==3 for g in gaps),'runs_1':sum(g['count']==1 for g in all_gaps),'runs_2':sum(g['count']==2 for g in all_gaps),'longest_run':max((g['count'] for g in all_gaps),default=0)})
    journal_gaps=[{'start_frame':a['frame_id']+1,'end_frame':b['frame_id']-1,'count':b['frame_id']-a['frame_id']-1}
                  for a,b in zip(frames,frames[1:]) if b['frame_id']>a['frame_id']+1]
    if expected_source_frame_count is not None:
        if type(expected_source_frame_count) is not int or expected_source_frame_count<0 or any(f['frame_id']>=expected_source_frame_count for f in frames):
            raise ValueError('Invalid decoded source frame count')
        missing=sorted(set(range(expected_source_frame_count))-{f['frame_id'] for f in frames})
        journal_gaps=[]
        for fid in missing:
            if journal_gaps and fid==journal_gaps[-1]['end_frame']+1:
                journal_gaps[-1]['end_frame']=fid;journal_gaps[-1]['count']+=1
            else:journal_gaps.append({'start_frame':fid,'end_frame':fid,'count':1})
    return {'tracks':summary,'observation_gaps_ge_3':runs,'journal_gaps':journal_gaps,'expected_source_frame_count':expected_source_frame_count,
            'semantics':'Point unavailable includes missing, low score, unverified assignment; not visibility truth or detector recall. PTS span is last-first, not exposure duration.',
            'accuracy_validated':False,'coach_eligible':False}
