"""Startup-fixed racket source policy; no ML runtime or legacy fallback."""
import copy
import math

SOURCE_POLICY_VERSION='exclusive_grip_views_v1'


def is_exclusive(config):
    return config.get('mode') == 'active' and config.get('selection_policy') == 'exclusive'


def current_pose_candidates(observation, frame_id):
    """Retain source and mirror candidates; anatomical association stays downstream."""
    if (not observation or observation.get('status') != 'observed'
            or observation.get('frame_id') != frame_id
            or (observation.get('diagnostics') or {}).get('nonfinite_candidates', 0)):
        return []
    result=[]
    for item in observation.get('candidates', []):
        box=item.get('box')
        if (item.get('source') != 'racket_pose_model' or item.get('source_frame_id') != frame_id
                or item.get('observed') is not True or not isinstance(box, (list, tuple)) or len(box) != 4):
            continue
        values=[*box, item.get('confidence')]
        if not all(isinstance(v, (int,float)) and math.isfinite(v) for v in values):continue
        if not 0 <= values[-1] <= 1 or box[2] <= box[0] or box[3] <= box[1]:continue
        result.append(copy.deepcopy(item))
    return result


def validate_exclusive_output(rackets, frame_id):
    """Fail the session if a later resolver ever leaks a legacy/current mismatch."""
    for item in rackets or []:
        if item.get('source') != 'racket_pose_model':
            raise RuntimeError('Exclusive racket profile received a non-pose racket')
        if item.get('observed') is True and item.get('source_frame_id') != frame_id:
            raise RuntimeError('Exclusive racket profile received a stale observed racket')


def associated_pose_candidates(observation, frame_id, wrists_by_view, frame_size, config):
    """Apply the existing qualified-grip gate separately to source-coordinate views."""
    from racket_pose_detector import select_pose_candidate
    fresh=current_pose_candidates(observation,frame_id)
    if not fresh:return []
    result=[];seen=set()
    for view,wrists in wrists_by_view.items():
        selected=select_pose_candidate({**observation,'candidates':fresh},frame_id,wrists,frame_size,
                                      {**config,'association_view':view})
        if selected is not None:
            identity=tuple(selected['box'])
            if identity not in seen:result.append(selected);seen.add(identity)
    return result
