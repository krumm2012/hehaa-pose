"""Evaluate independent visible-joint labels without treating missing truth as bad motion."""
import math
import re
from collections import defaultdict
from statistics import mean, median

JOINTS = tuple(side+'_'+joint for side in ('left','right')
               for joint in ('shoulder','hip','elbow','wrist','knee','ankle'))

def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def evaluate_joint_labels(labels, predictions, tolerance_px):
    if not finite(tolerance_px) or tolerance_px <= 0:
        raise ValueError('tolerance_px must be explicit, finite and positive')
    digest=labels.get('source_sha256','')
    if not re.fullmatch('[0-9a-f]{64}',digest) or predictions.get('source_sha256')!=digest:
        raise ValueError('source video hash mismatch')
    if labels.get('schema')!='tennis.independent-joint-labels.v1' or labels.get('coordinate_space')!='original_source_pixels' or labels.get('frame_index_base')!=0:
        raise ValueError('unsupported label coordinate contract')
    if (labels.get('annotation_mode') == 'model_assisted' or labels.get('independent_reference') is False
        or labels.get('model_suggestions') or any('model' in str(p.get('origin','')).lower()
            for p in labels.get('labels',{}).values())):
        raise ValueError('model-assisted labels are not independent references')
    if predictions.get('schema')!='tennis.pose-resolution-audit.v1':
        raise ValueError('unsupported prediction contract')
    frames={}
    joints=labels.get('requested_joints',list(JOINTS))
    if (not isinstance(joints,list) or not joints or len(set(joints))!=len(joints)
        or any(j not in JOINTS for j in joints)):
        raise ValueError('invalid requested joint set')
    for frame in labels.get('frames',[]):
        fid=frame.get('frame_id')
        if type(fid)!=int or fid<0 or fid in frames:
            raise ValueError('invalid or duplicate source frame')
        if not all(type(frame.get(k))==int and frame[k]>0 for k in ('width','height')):
            raise ValueError('invalid frame dimensions')
        frames[fid]=frame
    truth={}
    for key,point in labels.get('labels',{}).items():
        parts=key.split(':')
        if len(parts)!=3 or not parts[0].isdigit():raise ValueError('invalid label key')
        fid,view,joint=int(parts[0]),parts[1],parts[2]
        if fid not in frames or view not in ('front','back') or joint not in joints:
            raise ValueError('label outside declared frame/view/joint set')
        if type(point.get('visible'))!=bool:raise ValueError('visibility must be explicit')
        if point['visible']:
            if not all(finite(point.get(k)) for k in ('x','y')):raise ValueError('invalid truth point')
            if not 0<=point['x']<frames[fid]['width'] or not 0<=point['y']<frames[fid]['height']:
                raise ValueError('truth point outside source frame')
        elif point.get('x') is not None or point.get('y') is not None:
            raise ValueError('unidentifiable point must not supply guessed coordinates')
        truth[fid,view,joint]=point
    base={'schema':'tennis.joint-error-evaluation.v1','source_sha256':digest,
          'tolerance_px':tolerance_px,'tolerance_status':'evaluation_parameter_not_clinical_or_coaching_standard',
          'planned_labels':len(frames)*2*len(joints),'provided_labels':len(truth),
          'unlabelled_count':len(frames)*2*len(joints)-len(truth),
          'limitations':['human_labels_require_independent_review','errors_only_on_identifiable_labelled_points',
                        'no_3d_or_technique_accuracy_claim','model_confidence_is_not_accuracy_probability']}
    if labels.get('confirmed') is not True or not str(labels.get('annotator_id') or '').strip():
        return {**base,'status':'pending_independent_confirmation','groups':[]}
    if not truth:return {**base,'status':'pending_labels','groups':[]}
    scales=predictions.get('scales',[])
    if not scales or any(not finite(s) or s<=0 for s in scales) or len(set(scales))!=len(scales):
        raise ValueError('invalid prediction scales')
    groups=defaultdict(list)
    for scale in scales:
        # The audit maps predictions at every scale back to original pixels.
        scale_key=str(scale)
        for (fid,view,joint),target in truth.items():
            point=predictions.get('samples',{}).get(str(fid),{}).get(scale_key,{}).get(view,{}).get(joint,{})
            usable=(point.get('observed') is True and finite(point.get('confidence')) and point['confidence']>=.5
                    and all(finite(point.get(k)) for k in ('x','y'))
                    and type(point.get('source_frame_id'))==int and point['source_frame_id']==fid)
            row={'frame_id':fid,'identifiable':target['visible'],'model_output_qualified':usable,'error_px':None}
            if target['visible'] and usable:row['error_px']=math.hypot(point['x']-target['x'],point['y']-target['y'])
            groups[scale_key,view,'all'].append(row)
            groups[scale_key,view,joint].append(row)
    summaries=[]
    for (scale,view,joint),rows in sorted(groups.items()):
        visible=[r for r in rows if r['identifiable']];errors=[r['error_px'] for r in visible if r['error_px'] is not None]
        bad=sum(e>tolerance_px for e in errors);good=len(errors)-bad
        summaries.append({'scale':scale,'view':view,'joint':joint,'visible_truth_count':len(visible),
            'paired_count':len(errors),'no_qualified_output_count':len(visible)-len(errors),
            'mean_error_px':mean(errors) if errors else None,'median_error_px':median(errors) if errors else None,
            'max_error_px':max(errors) if errors else None,
            'qualified_output_rate':len(errors)/len(visible) if visible else None,
            'incorrect_among_qualified_rate':bad/len(errors) if errors else None,
            'within_tolerance_among_visible_truth_rate':good/len(visible) if visible else None,
            'unidentifiable_truth_count':len(rows)-len(visible),
            'model_outputs_on_unidentifiable_truth':sum(not r['identifiable'] and r['model_output_qualified'] for r in rows),
            'samples':rows})
    return {**base,'status':'evaluated_independent_labels','annotator_id':labels['annotator_id'],'groups':summaries}
