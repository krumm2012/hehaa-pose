"""Event elapsed time and model-label support on reported media PTS.

No FPS or receipt-clock fallback; model phases and source PTS do not establish
sensor exposure, true phase boundaries, or a validated teaching rubric.
"""
import math

POLICY_VERSION = 'event_phase_media_time_v1'
PHASE_RULE_EXCLUSION = 'phase_duration_rubric_not_independently_validated'


def source_frame_navigation(frames):
    """Reported input PTS for navigation, never an annotated-video FPS mapping."""
    result = {'policy_version': POLICY_VERSION, 'status': 'unavailable',
              'frames': [], 'reasons': [], 'sensor_exposure_verified': False}
    known = {}
    for row in frames:
        fid, source = row.get('frame_id'), row.get('source_time')
        if type(fid) is not int or fid < 0 or fid in known:
            result['reasons'] = ['invalid_or_duplicate_frame_identity']
            return result
        if not isinstance(source, dict):
            result['reasons'] = ['source_time_contract_missing']
            return result
        seconds = source.get('timestamp_seconds')
        if (source.get('schema_version') != 'tennis.source-time.v1'
                or type(source.get('source_frame_id')) is not int
                or source['source_frame_id'] != fid
                or source.get('source_kind') != 'video_file'
                or source.get('basis') != 'media_pts' or source.get('quality') != 'reported'
                or type(seconds) not in (int, float)
                or not math.isfinite(seconds) or seconds < 0):
            result['reasons'] = ['reported_source_frame_time_unavailable']
            return result
        known[fid] = seconds
    pairs = sorted(known.items())
    if not pairs or any(b[1] <= a[1] for a, b in zip(pairs, pairs[1:])):
        result['reasons'] = ['empty_or_nonmonotonic_source_time']
        return result
    result.update(status='reported_media_time', frames=[[fid, t] for fid, t in pairs])
    return result


def analyze_event_source_timing(event, frames, traces=()):
    result = {
        'policy_version': POLICY_VERSION, 'status': 'unavailable', 'reasons': [],
        'basis': None, 'duration_seconds': None, 'start_to_contact_seconds': None,
        'contact_to_end_seconds': None, 'peak_to_contact_offset_seconds': None,
        'phase_durations_seconds': None, 'recovery_time_seconds': None,
        'phase_status': 'unavailable', 'phase_reasons': [], 'anchors': {},
        'source_frame_ids': [], 'observation_frame_gaps': [],
        'phase_interval_source_frames': {}, 'coach_eligible': False,
        'coaching_exclusion_reason': PHASE_RULE_EXCLUSION,
        'accuracy_validated': False, 'sensor_exposure_verified': False,
        'duration_semantics': 'last_anchor_PTS_minus_first_anchor_PTS_no_extra_tail_frame',
        'phase_semantics': 'model_label_at_left_sample_supports_next_observed_interval_only',
    }

    def unavailable(reason):
        result['reasons'].append(reason)
        return result

    keys = ('start_frame', 'contact_frame', 'peak_frame', 'end_frame')
    ids = {key: event.get(key) for key in keys}
    if any(type(value) is not int or value < 0 for value in ids.values()):
        return unavailable('invalid_event_anchor_identity')
    start, end = ids['start_frame'], ids['end_frame']
    if start > end or any(not start <= value <= end for value in ids.values()):
        return unavailable('event_anchor_outside_boundaries')
    by_id = {}
    for row in frames:
        fid = row.get('frame_id')
        if type(fid) is not int:
            return unavailable('invalid_frame_identity')
        if not start <= fid <= end:
            continue
        if fid in by_id:
            return unavailable('duplicate_frame_identity')
        by_id[fid] = row
    if any(fid not in by_id for fid in ids.values()):
        return unavailable('missing_event_anchor_record')
    ordered = sorted(by_id)
    times = {}
    for fid in ordered:
        source = by_id[fid].get('source_time')
        if not isinstance(source, dict):
            return unavailable('source_time_contract_missing')
        if (source.get('schema_version') != 'tennis.source-time.v1'
                or type(source.get('source_frame_id')) is not int
                or source['source_frame_id'] != fid):
            return unavailable('source_time_identity_unverified')
        if (source.get('source_kind') != 'video_file' or source.get('basis') != 'media_pts'
                or source.get('quality') != 'reported'):
            return unavailable('reported_file_media_time_unavailable')
        seconds = source.get('timestamp_seconds')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            return unavailable('invalid_source_timestamp')
        times[fid] = seconds
    if any(times[right] <= times[left] for left, right in zip(ordered, ordered[1:])):
        return unavailable('non_monotonic_source_time')
    result.update(status='reported_media_time', basis='media_pts', source_frame_ids=ordered,
        duration_seconds=times[end]-times[start],
        start_to_contact_seconds=times[ids['contact_frame']]-times[start],
        contact_to_end_seconds=times[end]-times[ids['contact_frame']],
        peak_to_contact_offset_seconds=times[ids['contact_frame']]-times[ids['peak_frame']],
        anchors={key: {'source_frame_id': fid, 'timestamp_seconds': times[fid]}
                 for key, fid in ids.items()})
    gaps = [[left, right] for left, right in zip(ordered, ordered[1:]) if right != left+1]
    result['observation_frame_gaps'] = gaps
    if gaps:
        result['phase_reasons'] = ['source_frame_gap_cannot_bridge_phase_labels']
        return result
    phase_by_id = {}
    for row in traces:
        fid = row.get('frame')
        if type(fid) is not int or not start <= fid <= end:
            continue
        if fid in phase_by_id:
            result['phase_reasons'] = ['duplicate_phase_trace']
            return result
        if row.get('event_id') not in (None, event.get('event_id')):
            result['phase_reasons'] = ['phase_trace_event_identity_mismatch']
            return result
        phase_by_id[fid] = row.get('phase')
    if any(not isinstance(phase_by_id.get(fid), str) or not phase_by_id[fid]
           for fid in ordered[:-1]):
        result['phase_reasons'] = ['phase_trace_missing']
        return result
    durations, intervals = {}, {}
    for left, right in zip(ordered, ordered[1:]):
        phase = phase_by_id[left]
        durations[phase] = durations.get(phase, 0.) + times[right]-times[left]
        intervals.setdefault(phase, []).append([left, right])
    result.update(phase_status='model_label_media_time_support',
                  phase_durations_seconds=durations,
                  phase_interval_source_frames=intervals,
                  recovery_time_seconds=durations.get('ready'))
    return result
