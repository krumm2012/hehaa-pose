"""Lightweight eligibility gates; no inference and no accuracy probabilities."""
import math
from numbers import Real

POSE_POLICY = 'fresh_front_pose_v2_finite_observations'


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def finite_point(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = finite_number(value[0]), finite_number(value[1])
    return (x, y) if x is not None and y is not None else None


def image_joint_angle(a, b, c):
    """Normalize vectors before the dot product; invalid geometry stays missing."""
    a, b, c = finite_point(a), finite_point(b), finite_point(c)
    if a is None or b is None or c is None:
        return None
    left, right = (a[0]-b[0], a[1]-b[1]), (c[0]-b[0], c[1]-b[1])
    nl, nr = math.hypot(*left), math.hypot(*right)
    if not all(math.isfinite(v) for v in (*left, *right, nl, nr)) or nl <= 0 or nr <= 0:
        return None
    cosine = (left[0]/nl)*(right[0]/nr) + (left[1]/nl)*(right[1]/nr)
    return math.degrees(math.acos(max(-1., min(1., cosine))))


def qualified_front_point(point, frame_id, minimum_score=.35):
    if not isinstance(point, dict):
        return None, 'invalid_joint_record'
    if point.get('observed') is not True:
        return None, 'not_fresh_observation'
    if point.get('recovered_from_mirror'):
        return None, 'mirror_recovery_not_measurement'
    source = point.get('source_frame_id')
    if source is not None and (type(source) is not int or source < 0 or source != frame_id):
        return None, 'source_frame_mismatch'
    if point.get('confidence_source') == 'unavailable':
        return None, 'point_score_unavailable'
    score = finite_number(point.get('confidence'))
    if score is None or not 0 <= score <= 1:
        return None, 'invalid_point_score'
    if score < minimum_score:
        return None, 'low_point_score'
    xy = finite_point((point.get('x'), point.get('y')))
    return ((xy[0], xy[1], score), None) if xy is not None else (None, 'invalid_point_coordinates')


def contact_evidence(event):
    evidence = event.get('evidence') or {}
    return evidence.get('contact_analysis') or (evidence.get('classification_context') or {}).get('contact_analysis') or {}


def review_reason(event):
    """Explicit uncertainty blocks automatic technique ratings/corrections."""
    stroke = event.get('stroke_type')
    if stroke is not None and stroke not in ('Forehand', 'Backhand', 'Two-Handed Backhand'):
        return 'unknown_stroke_type'
    contact = contact_evidence(event)
    if event.get('contact_status', contact.get('contact_status')) == 'unknown':
        return 'unconfirmed_contact'
    if stroke in ('Backhand', 'Two-Handed Backhand'):
        return 'stroke_specific_rubric_unvalidated'
    return None


def measurement_pose_with_evidence(frame):
    """Build a read-only qualified view; preserve the original raw observations."""
    observations = frame.get('pose_observations')
    pose, rejected = {}, {}
    evidence = {'policy_version': POSE_POLICY, 'rejected_points': rejected,
                'accuracy_validated': False, 'confidence_semantics': 'raw_model_score_not_position_accuracy'}
    if observations is None:
        raw = frame.get('pose') or {}
        evidence['source_semantics'] = 'legacy_xy_unverified'
        if not isinstance(raw, dict):
            evidence['record_reason'] = 'invalid_legacy_pose_record'
            raw = {}
        for name, point in raw.items():
            xy = finite_point(point)
            if xy is not None: pose[name] = xy
            else: rejected[name] = 'invalid_point_coordinates'
    else:
        evidence['source_semantics'] = 'fresh_front_observations'
        raw = observations.get('front', {}) if isinstance(observations, dict) else None
        if not isinstance(raw, dict):
            evidence['record_reason'] = 'invalid_front_pose_observations'
            raw = {}
        for name, point in raw.items():
            value, reason = qualified_front_point(point, frame.get('frame_id'))
            if value is not None: pose[name] = value[:2]
            else: rejected[name] = reason
    evidence.update(input_point_count=len(raw), measurement_point_count=len(pose))
    return pose, evidence


def measurement_pose(frame):
    """Raw, finite, freshly observed front points; legacy XY stays unverified."""
    return measurement_pose_with_evidence(frame)[0]
