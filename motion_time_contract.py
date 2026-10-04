"""Candidate timing and heuristic tuning, separate from measured phase times."""
from image_motion_measurements import source_timestamp

POLICY_VERSION = 'motion_candidates_v1_source_time'
REFERENCE_HZ = 25.0  # Explicit historical tuning reference, not the input FPS.
SMOOTH_SECONDS = .08
RACKET_GAP_SECONDS = .12


def candidate_timeline(features):
    """Use one monotonic time basis, or disclose an observation-order heuristic.

    Malformed declared source time never falls back to a compatibility clock.
    A candidate clock does not establish sensor exposure or movement accuracy.
    The raw feature/source dictionaries are not modified.
    """
    samples = [source_timestamp(row) for row in features]
    declared = any('source_time' in row for row in features)
    times = [t for t, _ in samples]
    bases = {basis for _, basis in samples}
    valid = bool(features) and all(t is not None for t in times)
    valid = valid and len(bases) == 1 and all(b > a for a, b in zip(times, times[1:]))
    if declared:
        ids = [row.get('frame_id') for row in features]
        valid = valid and all(type(fid) is int and fid >= 0 for fid in ids)
        valid = valid and all(b > a for a,b in zip(ids,ids[1:]))
    qualified = valid and declared and bases == {'media_pts'} and all('source_time' in row for row in features)
    if valid and not (declared and 'legacy_unverified' in bases):
        basis = next(iter(bases))
        basis = 'legacy_timestamp_unverified' if basis == 'legacy_unverified' else basis
        reasons = [] if qualified else ['candidate_clock_not_reported_file_media_time']
    else:
        basis = 'processing_order_unverified'
        times = [i / REFERENCE_HZ for i in range(len(features))]
        reasons = ['missing_mixed_or_nonmonotonic_candidate_clock']
    evidence = {'policy_version': POLICY_VERSION, 'basis': basis,
                'source_time_qualified': bool(qualified), 'reasons': reasons,
                'reference_sample_rate_hz': REFERENCE_HZ,
                'reference_rate_semantics': 'heuristic_tuning_only_not_input_or_exposure_FPS',
                'smoothing_window_seconds': SMOOTH_SECONDS if qualified else None,
                'accuracy_validated': False, 'sensor_exposure_verified': False,
                'coach_eligible': False}
    prepared = [dict(row, timestamp=t, candidate_timing=dict(evidence))
                for row, t in zip(features, times)]
    return prepared, evidence


def attach_candidate_timing(result, evidence):
    result['candidate_timing'] = dict(evidence)
    for event in result['events']:
        event.setdefault('evidence', {})['candidate_timing'] = dict(evidence)
    return result
