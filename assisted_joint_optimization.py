"""Conservative automatic candidate review; preserve human decisions verbatim."""
import copy
import math
from collections import Counter
from observation_policy import qualified_front_point, finite_number

POLICY = 'assisted_joint_bounded_review_v1'


def optimize_remaining(review, rows, audit):
    result=copy.deepcopy(review)
    original=review['labels'];by_frame={r['frame_id']:r for r in rows}
    human={k:v for k,v in original.items() if v.get('reviewed') is True and v.get('review_actor')!='automatic'}
    hidden=[(int(k.split(':')[0]),':'.join(k.split(':')[1:])) for k,v in human.items() if not v['visible']]
    flags={s['key']:s['flags'] for s in audit['samples']}
    counts=Counter();changes=[]
    def time(fid):
        clock=by_frame.get(fid,{}).get('source_time') or {}
        if clock.get('source_frame_id')!=fid or clock.get('basis')!='media_pts' or clock.get('quality')!='reported':return None
        return finite_number(clock.get('timestamp_seconds'))
    def point(fid,view,joint):
        raw=(by_frame.get(fid,{}).get('pose_observations',{}).get(view) or {}).get(joint,{})
        value,_=qualified_front_point(raw,fid,minimum_score=.5)
        return value if raw.get('source_frame_id')==fid else None
    for frame in review['frames']:
        fid=frame['frame_id'];t=time(fid)
        for view in ('front','back'):
            for joint in review['requested_joints']:
                key=f'{fid}:{view}:{joint}'
                if key in human:counts['human_preserved']+=1;continue
                raw=point(fid,view,joint);reasons=[]
                if raw is None:reasons.append('no_fresh_source_observation')
                elif raw[2]<.95:reasons.append('visibility_score_uncertain')
                if any(reason in flags.get(key,[]) for reason in ('large_temporal_step','left_right_image_order_changed','unqualified_media_time')):
                    reasons.append('temporal_identity_uncertain')
                for source,suffix in hidden:
                    other=time(source)
                    if suffix==f'{view}:{joint}' and abs(fid-source)<=2 and t is not None and other is not None and abs(t-other)<=.08:
                        reasons.append('near_human_unidentifiable_observation');break
                # Small projected body width makes near/far-side correspondence
                # uncertain. This is a review proxy, not physical occlusion truth.
                body={name:point(fid,view,name) for name in ('left_shoulder','right_shoulder','left_hip','right_hip')}
                if all(body.values()) and raw:
                    sh=[(body['left_shoulder'][i]+body['right_shoulder'][i])/2 for i in (0,1)]
                    hip=[(body['left_hip'][i]+body['right_hip'][i])/2 for i in (0,1)]
                    length=math.dist(sh,hip)
                    segment=joint.split('_')[1];a=body['left_'+segment];b=body['right_'+segment]
                    if length>0 and math.dist(a[:2],b[:2])<length*.18:reasons.append('projected_joint_pair_overlap')
                    # A foreground forearm can obscure a torso landmark. Abstain
                    # instead of claiming visible anatomy from a high score.
                    for side in ('left','right'):
                        elbow=point(fid,view,side+'_elbow');wrist=point(fid,view,side+'_wrist')
                        if not elbow or not wrist or length<=0:continue
                        dx,dy=wrist[0]-elbow[0],wrist[1]-elbow[1];den=dx*dx+dy*dy
                        if den<=0:continue
                        u=((raw[0]-elbow[0])*dx+(raw[1]-elbow[1])*dy)/den
                        if 0<=u<=1 and math.dist(raw[:2],(elbow[0]+u*dx,elbow[1]+u*dy))<=length*.04:
                            reasons.append('foreground_limb_overlap_proxy');break
                entry={'reviewed':True,'review_actor':'automatic','review_policy':POLICY,
                       'visibility_verified':False,'measurement_eligible':False}
                if reasons:
                    entry.update(visible=False,x=None,y=None,reason='visibility_or_identity_uncertain',
                                 abstention_reasons=sorted(set(reasons)),origin='automatic_abstention')
                    counts['automatic_unidentifiable']+=1
                else:
                    xy=list(raw[:2]);before=point(fid-1,view,joint);after=point(fid+1,view,joint)
                    tb,ta=time(fid-1),time(fid+1)
                    smoothed=False
                    neighbors_safe = before and after and before[2]>=.95 and after[2]>=.95 and all(not flags.get(f'{neighbor}:{view}:{joint}') for neighbor in (fid-1,fid+1))
                    if neighbors_safe and tb is not None and t is not None and ta is not None and .015<=t-tb<=.08 and .015<=ta-t<=.08:
                        # Only a bounded current-observation adjustment. No gaps
                        # are interpolated and history does not become observed.
                        fraction=(t-tb)/(ta-tb)
                        neighbor=[before[i]+fraction*(after[i]-before[i]) for i in (0,1)]
                        proposed=[.8*raw[i]+.2*neighbor[i] for i in (0,1)]
                        delta=math.dist(raw[:2],proposed)
                        if math.dist(raw[:2],[round(v,2) for v in proposed])<=2:
                            xy=proposed;smoothed=delta>.005
                    entry.update(visible=True,x=round(xy[0],2),y=round(xy[1],2),
                        model_confidence=raw[2],source_frame_id=fid,origin='automatic_model_review',
                        temporal_adjustment_applied=smoothed)
                    counts['automatic_visible_candidate']+=1
                    if smoothed:counts['bounded_adjustment']+=1
                result['labels'][key]=entry;changes.append({'key':key,'decision':entry})
    if any(result['labels'][k]!=v for k,v in human.items()):raise AssertionError('Human decisions changed')
    result.update(confirmed=False,automatic_review_complete=True,human_review_complete=False,
                  review_policy=POLICY,review_summary=dict(counts),independent_reference=False)
    result['review_queue']=[dict(item,priority=0 if result['labels'][item['key']].get('visible') is False else item['priority']) for item in result['review_queue']]
    result['review_queue'].sort(key=lambda item:(item['priority'],int(item['key'].split(':')[0]),item['key']))
    return result,{'policy':POLICY,'counts':dict(counts),'changes':changes,
        'accuracy_validated':False,'occlusion_truth_verified':False,
        'limitations':['Automatic visibility proxies can abstain on visible joints or miss occlusions.',
                       'Bounded 2D candidates do not validate true 3D peaks or teaching accuracy.']}
