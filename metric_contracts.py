"""Shared, versioned metric definitions; evidence quality is not accuracy."""
from copy import deepcopy

VERSION = 'tennis.metric-contract.v3'
# formula, coordinate system, window, qualification; operational definitions only.
DEFINITIONS = {
    'hip_shoulder_separation': ('abs(wrap180(shoulder_line_angle-hip_line_angle)); sample median', 'front image plane', 'contact source PTS +/-0.08s, clipped to event', 'observed shoulder and hip endpoints; reported source time; projected separation only'),
    'shoulder_turn': ('sample median atan2(back_shoulder_width,front_shoulder_width); fallback absolute shoulder image angle', 'ROI coordinates restored to source pixel scale for width proxy; front image for fallback', 'contact source PTS +/-0.08s, clipped to event', 'both source-scale widths >15 px for dual proxy; reported source time; no calibrated 3D interpretation'),
    'shoulder_turn_change': ('max abs(wrap180(angle-circular_sample_median(baseline)))', 'front image shoulder-line orientation', 'event start through contact; baseline first quarter source elapsed interval capped at0.16s', 'finite observed shoulder angles and observed baseline; baseline unwrapped relative to first angle'),
    'preparation_knee_flexion': ('sample median(180-mean(available left/right hip-knee-ankle angles))', 'front image joint coordinates', 'first half of start-to-contact source elapsed interval', 'reported source time; at least one complete nondegenerate leg; side availability can vary'),
    'arm_extension': ('sample median hitting-side shoulder-elbow-wrist interior angle', 'front image joint coordinates', 'source PTS +/-0.08s at contact if contact score>=.35, otherwise peak; clipped to event', 'reported source time; complete nondegenerate observed arm'),
    'contact_lateral_distance': ('abs(ball_x-body_center_x)/event_body_width; closest supported source-time sample', 'front original image pixels; body center=mean of shoulder and hip midpoints', 'contact source PTS +/-0.08s, clipped to event', 'ball and torso center plus positive body scale; reported source time; coaching gates are separate'),
    'weight_transfer': ('distance(sample_median_center(start),sample_median_center(contact))/event_body_width', 'front image Euclidean displacement, not physical weight transfer', 'each endpoint source PTS +/-0.08s, clipped to event', 'both torso centers and positive body scale; reported source time'),
    'balance_drift': ('distance(sample_median_center(contact),sample_median_center(early_recovery))/event_body_width', 'front image Euclidean displacement, not balance stability', 'endpoint source PTS +/-0.08s; recovery observed PTS nearest contact+0.40s within event', 'both torso centers and positive body scale; reported source time'),
    'takeback_depth': ('max min(2.5,abs(back_wrist_x-back_shoulder_mid_x)/back_shoulder_width)', 'back ROI projection restored to source pixel scale', 'start through contact', 'back wrist and shoulders; width >20 px; not physical depth'),
    'scapular_retraction': ('max min(3,back_shoulder_width/front_shoulder_width)', 'ROI coordinates restored to source pixel scale', 'start through contact', 'both shoulder widths >15 px; not anatomical scapular motion'),
    'racket_head_speed': ('distance(current_box_center,previous_box_center)/source_dt', 'original image pixels per source second', 'contact and immediately preceding frame', 'fresh consecutive boxes; positive same-basis source dt; no km/h calibration'),
    'racket_max_speed': ('max valid box-center pixel speed', 'original image pixels per source second', 'event window', 'same source-time and observation requirements as contact speed'),
    'brush_angle': ('atan2(max(0,lowest_y-contact_y),abs(lowest_x-contact_x))', 'front original image box centers; y increases downward', 'source time contact [-.48,0] seconds', 'candidate/confirmed contact; >=5 observations; >=.8 coverage; qualified contact box; no clock gaps; noncoincident chord'),
    'drop_depth_ratio': ('max(0,lowest_y-contact_y)/event_body_width', 'front original image; normalized by body width', 'source time contact [-.48,0] seconds', 'qualified contact and track coverage plus positive body scale; coincident chord permits zero rise but no direction'),
    'stance': ('median atan2(abs(right_ankle_y-left_ankle_y),abs(right_ankle_x-left_ankle_x))', 'front original image ankle-line inclination', 'source time contact [-.12,+.12] seconds', '>=3 fresh ankle pairs score>=.5, span>=.2*event_body_width; coverage>=.8; range<=10 degrees; valid clocks'),
    'leg_drive': ('(max(hip_mid_y)-contact_hip_mid_y)/event_body_width', 'front original image vertical displacement', 'source time contact [-.6,0] seconds', '>=5 fresh hip pairs score>=.5; coverage>=.8; contact observed; rise>.04*event_body_width; valid clocks'),
    'kinematic_sequence': ('circular angle derivatives; per-continuous-run median3; peak>=90% max interval; pair lag intervals including cadence', 'independent original front/back image lines and racket box centers; source seconds', 'contact [-.6,+.16] seconds', 'source-time contract; point score>=.5; line>=12px; >=6 samples per run; coverage>=.6; reject broad/boundary/ambiguous peaks and view conflicts'),
    'swing_quality_score': ('practice scoring policy; automatic curves disabled pending independent validation; confirmed manual rubric only', 'dimensionless policy score, not physical measurement', 'event', 'practice policy eligibility; no calibrated accuracy probability'),
}


def metric_contract(key, metric):
    formula, coordinates, window, conditions = DEFINITIONS[key]
    evidence = metric.get('measurement_evidence') or {}
    window_evidence = deepcopy(metric.get('window_evidence') or {})
    reasons = list(dict.fromkeys(list(evidence.get('reasons') or []) + list(window_evidence.get('reasons') or [])))
    if metric.get('value') is None and not reasons:
        details = metric.get('details') or {}
        reasons = [details.get('reason') or 'insufficient_inputs_or_qualification']
    return {'version': VERSION, 'metric_id': key, 'formula': formula,
            'coordinate_system': coordinates, 'unit': metric.get('unit'),
            'window': window, 'valid_conditions': conditions,
            'window_evidence': window_evidence,
            'legacy_compatibility': 'legacy windows explicitly unverified; declared invalid source clocks abstain',
            'missing_reasons': reasons if metric.get('value') is None else [],
            'evidence_reasons': reasons, 'source_frames': list(metric.get('source_frames') or []),
            'accuracy_validated': False, 'confidence_meaning': 'heuristic_evidence_quality_not_accuracy',
            'coaching_exclusion_reason': metric.get('exclusion_reason')}


def attach_metric_contracts(bio):
    bio['metric_contract_version'] = VERSION
    for key, metric in bio.get('metrics', {}).items():
        if key in DEFINITIONS:
            metric['contract'] = metric_contract(key, metric)
    ext = bio.get('extended_biomechanics') or {}
    for key, section, field, unit in [
        ('racket_head_speed','racket_head_speed','contact_px_s','px/s'),
        ('racket_max_speed','racket_head_speed','max_px_s','px/s'),
        ('brush_angle','brush_angle','low_to_high_angle_deg','deg'),
        ('drop_depth_ratio','brush_angle','drop_depth_ratio','ratio'),
        ('stance','stance','image_foot_line_angle_deg','deg_2d'),
        ('leg_drive','leg_drive','drive_ratio','ratio'),
        ('kinematic_sequence','kinematic_sequence','sequence_quality','category')]:
        if section not in ext:
            continue
        canonical = bio.get('metrics', {}).get(key)
        evidence = deepcopy(ext[section].get('measurement_evidence') or {})
        evidence.update((evidence.get('fields') or {}).get(field, {}))
        contract = deepcopy(canonical['contract']) if canonical and 'contract' in canonical else metric_contract(key, {
            'value': ext[section].get(field), 'unit': unit,
            'measurement_evidence': evidence})
        ext[section].setdefault('metric_contracts', {})[key] = contract
