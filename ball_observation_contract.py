"""Selected-ball provenance; ranking scores are not model probabilities."""
import math

SCHEMA = 'tennis.ball-observation.v1'


def stamp_ball_detections(detections, frame_id):
    for detection in detections or []:
        if detection.get('source') == 'model_detection':
            detection.setdefault('observed', True)
            detection.setdefault('source_frame_id', frame_id)
    return detections


def selected_ball_observation(detection, frame_id, legacy_position=None):
    detection = detection or {}
    position = detection.get('position', legacy_position)
    valid_position = (isinstance(position,(list,tuple)) and len(position)==2 and
                      all(type(v) in (int,float) and math.isfinite(v) for v in position))
    source = detection.get('source') or ('legacy_xy_unverified' if valid_position else 'unavailable')
    score = detection.get('model_confidence')
    score = score if type(score) in (int,float) and math.isfinite(score) and 0<=score<=1 else None
    fresh = detection.get('observed') is True and detection.get('source_frame_id') == frame_id
    eligible = valid_position and fresh and source == 'model_detection' and score is not None
    reasons = []
    if not valid_position: reasons.append('ball_not_observed')
    if source != 'model_detection': reasons.append('ball_source_not_model_observation')
    if not fresh: reasons.append('ball_source_frame_unverified_or_mismatched')
    if score is None: reasons.append('model_score_unavailable')
    return {'schema_version':SCHEMA,'position':list(position) if valid_position else None,
            'coordinate_space':'original_source_pixels','source':source,
            'source_frame_id':detection.get('source_frame_id'), 'observed':bool(eligible),
            'model_confidence':score,'selection_score':detection.get('confidence'),
            'confidence_meaning':'raw_model_score_not_calibrated_accuracy',
            'measurement_eligible':bool(eligible),'reasons':reasons,'accuracy_validated':False}


def measurement_ball(frame):
    observation = frame.get('ball_observation')
    if observation is None:
        return frame.get('ball'), 'legacy_source_unverified'
    if (observation.get('schema_version') == SCHEMA and observation.get('measurement_eligible') is True
        and observation.get('observed') is True and observation.get('source') == 'model_detection'
        and observation.get('source_frame_id') == frame.get('frame_id')
        and observation.get('coordinate_space') == 'original_source_pixels'
        and type(observation.get('model_confidence')) in (int,float)
        and math.isfinite(observation['model_confidence']) and 0<=observation['model_confidence']<=1
        and isinstance(observation.get('position'), (list,tuple)) and len(observation['position'])==2
        and all(type(v) in (int,float) and math.isfinite(v) for v in observation['position'])):
        return observation.get('position'), 'fresh_model_observation'
    return None, 'unqualified_ball_observation'
