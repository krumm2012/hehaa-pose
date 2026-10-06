"""Audit every assisted suggestion; never mark it human reviewed or accurate."""
import math
from collections import Counter
from observation_policy import qualified_front_point, finite_number


def audit_all_prelabels(doc, rows):
    by_frame = {r['frame_id']:r for r in rows}
    if len(by_frame) != len(rows): raise ValueError('Duplicate journal source frames')
    samples = []
    diagonal = math.hypot(doc['frames'][0]['width'],doc['frames'][0]['height'])
    for frame in doc['frames']:
        fid = frame['frame_id']
        row = by_frame.get(fid)
        if row is None: raise ValueError('Missing requested journal frame')
        for view in ('front','back'):
            for joint in doc['requested_joints']:
                key = f'{fid}:{view}:{joint}'
                point = (row.get('pose_observations',{}).get(view) or {}).get(joint,{})
                value, why = qualified_front_point(point,fid,minimum_score=.5)
                flags = []
                if why or point.get('source_frame_id') != fid: flags.append(why or 'source_frame_mismatch')
                suggestion = doc['model_suggestions'].get(key)
                if not suggestion: flags.append('missing_suggestion')
                elif not value or math.dist((suggestion['x'],suggestion['y']),value[:2]) > 1e-6:
                    raise ValueError('Suggestion differs from bound journal: '+key)
                previous = by_frame.get(fid-1)
                displacement = None
                if previous and value:
                    other = (previous.get('pose_observations',{}).get(view) or {}).get(joint,{})
                    prior, _ = qualified_front_point(other,fid-1,minimum_score=.5)
                    if prior:
                        displacement = math.dist(value[:2],prior[:2])/diagonal
                        if displacement > .02: flags.append('large_temporal_step')
                    # A sudden left/right image-order change can indicate a
                    # mirror label swap or a real turn; prioritize human review.
                    segment = joint.split('_',1)[1]
                    current_pair = []; prior_pair = []
                    for side in ('left','right'):
                        name = side+'_'+segment
                        cp=(row.get('pose_observations',{}).get(view) or {}).get(name,{})
                        pp=(previous.get('pose_observations',{}).get(view) or {}).get(name,{})
                        c,_=qualified_front_point(cp,fid,minimum_score=.5)
                        old,_=qualified_front_point(pp,fid-1,minimum_score=.5)
                        if c and old:
                            current_pair.append(c[0]);prior_pair.append(old[0])
                    if len(current_pair)==2 and (current_pair[0]-current_pair[1])*(prior_pair[0]-prior_pair[1]) < 0:
                        flags.append('left_right_image_order_changed')
                    t = row.get('source_time') or {};old_t=previous.get('source_time') or {}
                    current = finite_number(t.get('timestamp_seconds')); old=finite_number(old_t.get('timestamp_seconds'))
                    if (t.get('basis') != 'media_pts' or t.get('quality') != 'reported'
                            or t.get('source_frame_id') != fid or old_t.get('source_frame_id') != fid-1
                            or old_t.get('basis') != 'media_pts' or old_t.get('quality') != 'reported'
                            or current is None or old is None or current <= old): flags.append('unqualified_media_time')
                    elif current-old < .015: flags.append('short_source_interval')
                if value and value[2]<.8: flags.append('low_score')
                samples.append({'key':key,'source_frame_id':fid,'view':view,'joint':joint,
                    'flags':flags,'step_image_diagonal_units':displacement,'human_reviewed':False,
                    'position_accuracy_verified':False})
    counts = Counter(flag for sample in samples for flag in sample['flags'])
    indexed={sample['key']:sample for sample in samples}
    for item in doc['review_queue']:
        flags=indexed[item['key']]['flags'];item['audit_flags']=flags
        if flags:
            item['priority']=min(item['priority'],1)
            item['reason']=', '.join(flags)
    doc['review_queue'].sort(key=lambda x:(x['priority'],int(x['key'].split(':')[0]),x['key']))
    return {'schema':'tennis.assisted-joint-prelabel-audit.v1','status':'awaiting_complete_human_review',
        'source_sha256':doc['source_sha256'],'planned_count':len(samples),'automatically_audited_count':len(samples),
        'model_suggestion_count':len(doc['model_suggestions']),'flagged_count':sum(bool(s['flags']) for s in samples),
        'flag_counts':dict(counts),'human_confirmed_count':0,'independent_reference':False,
        'accuracy_validated':False,'heuristic_parameters':{'step_image_diagonal_ratio':.02,'short_interval_s':.015},
        'heuristic_semantics':'Review priorities, not validated anatomical error thresholds','samples':samples}
