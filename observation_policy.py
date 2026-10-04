"""Lightweight eligibility gates; no inference and no accuracy probabilities."""


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


def measurement_pose(frame):
    """Raw, freshly observed front points only; old XY journals remain replayable."""
    observations = frame.get('pose_observations')
    if observations is None:
        return frame.get('pose') or {}
    return {name: (p['x'], p['y']) for name, p in observations.get('front', {}).items()
            if p.get('observed') is True and not p.get('recovered_from_mirror')
            and (p.get('source_frame_id') is None or p['source_frame_id'] == frame.get('frame_id'))
            and p.get('confidence_source') != 'unavailable'
            and (p.get('confidence') or 0) >= .35}
