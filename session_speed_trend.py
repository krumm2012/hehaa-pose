"""Unit-preserving descriptive speed trends; never a fatigue diagnosis."""
import json
from collections import defaultdict
from statistics import mean, stdev

POLICY = 'session_speed_trend_v1'


def speed_contract(event):
    extended = event.get('extended_biomechanics') or (event.get('biomechanics') or {}).get('extended_biomechanics') or {}
    speed = extended.get('racket_head_speed') or event.get('racket_speed') or {}
    if not isinstance(speed, dict):
        speed = {}
    return {key: speed.get(key) for key in (
        'measurement_policy', 'status', 'observability', 'contact_time_basis',
        'geometry_assumptions', 'ground_calibration_sha256', 'measurement_point')}


def build_speed_trend(series, baseline, recent, *, overlapping, compatible, camera_shifted):
    groups = defaultdict(list)
    reasons = []
    for p in series:
        contract = p['speed_contract']
        # A missing metric value does not silently become the same kind of sample.
        for field, unit in (('racket_speed_kmh', 'km/h'), ('racket_speed_px_s', 'px/s')):
            value = p.get(field)
            if value is None or value < 0:
                continue
            identity = dict(contract)
            if unit == 'px/s':
                # Pixel measurements are independent of metric-height correction.
                identity.pop('geometry_assumptions', None)
                identity['status'] = 'image_plane'
                identity['observability'] = 'image_box_center_speed'
            key = json.dumps({'unit': unit, 'contract': identity}, sort_keys=True)
            groups[key].append((p['event_id'], value))
    summaries = []
    selected = None
    for key, samples in sorted(groups.items()):
        identity = json.loads(key)
        summaries.append({**identity, 'sample_count': len(samples),
                          'event_ids': [i for i, _ in samples],
                          'mean': round(mean(v for _, v in samples), 3)})
    # Prefer one complete metric series; otherwise one complete pixel series.
    # Never combine partial metric and pixel histories, even when both exist.
    for unit in ('km/h', 'px/s'):
        candidates = [(k, v) for k, v in groups.items()
                      if json.loads(k)['unit'] == unit and len(v) == len(series)]
        if len(candidates) == 1:
            selected = candidates[0]
            break
    if overlapping:
        reasons.append('overlapping_event_ranges')
    if len({p['event_id'] for p in series}) != len(series):
        reasons.append('duplicate_event_identity')
    if not compatible or len({p.get('camera_view') for p in series}) > 1:
        reasons.append('incomparable_practice_series')
    if any(str(p.get('stroke_type') or '').lower() in ('', 'unknown') for p in series):
        reasons.append('unknown_stroke_type')
    if camera_shifted:
        reasons.append('camera_scale_shift')
    if selected is None:
        reasons.append('incomplete_or_mixed_measurement_series')
    else:
        identity = json.loads(selected[0]); contract = identity['contract']
        if not contract.get('measurement_policy') or contract.get('contact_time_basis') != 'media_pts':
            reasons.append('unverified_measurement_identity_or_time')
        if identity['unit'] == 'km/h' and (
                not contract.get('ground_calibration_sha256') or not contract.get('geometry_assumptions')):
            reasons.append('missing_metric_geometry_identity')
    widths = [[p.get('reference_body_width_px') for p in window] for window in (baseline, recent)]
    if any(not w or any(v is None or v <= 0 for v in w) for w in widths):
        reasons.append('missing_camera_scale_evidence')
    enough = len(series) >= 6 and min(len(baseline), len(recent)) >= 2
    ready = enough and not reasons
    values = dict(selected[1]) if selected else {}
    base = [values[p['event_id']] for p in baseline] if selected else []
    rec = [values[p['event_id']] for p in recent] if selected else []
    change = None
    if ready and mean(base) > 0:
        change = round(100 * (mean(rec) - mean(base)) / mean(base), 2)
    elif ready:
        reasons.append('zero_baseline_speed'); ready = False
    status = ('BLOCKED' if reasons else 'WARMING_UP') if not ready else (
        'SPEED_DECREASED' if change <= -8 else 'SPEED_INCREASED' if change >= 5 else 'SPEED_STABLE')
    vals = list(values.values())
    return {'policy_version': POLICY, 'fatigue_status': status,
            'speed_decay_percent': change,
            'speed_std': round(stdev(vals), 2) if ready and len(vals) >= 2 else None,
            'speed_unit': json.loads(selected[0])['unit'] if selected else None,
            'samples_evaluated': len(vals), 'comparison_ready': ready,
            'blocked_reasons': reasons, 'minimum_event_count': 6,
            'measurement_groups': summaries, 'physiological_fatigue_validated': False,
            'latency_jitter_std_ms': None, 'shoulder_turn_std_deg': None}
