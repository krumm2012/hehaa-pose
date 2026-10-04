"""Small source-time window index for body and trajectory summaries.

Reported media time describes input presentation, not verified exposure. Legacy
records keep an explicitly unverified summary path. A malformed declared source
contract never borrows a compatibility or receipt clock.
"""
from dataclasses import dataclass

from image_motion_measurements import source_timestamp

POLICY_VERSION = 'body_and_trajectory_windows_v1_source_time'


@dataclass
class WindowSelection:
    frame_ids: list
    evidence: dict


class MetricWindowContext:
    def __init__(self, rows):
        rows = list(rows)
        self.by_id = {}
        self.reasons = []
        self.declared = any('source_time' in row for row in rows)
        for row in rows:
            fid = row.get('frame_id')
            if type(fid) is not int or fid < 0 or fid in self.by_id:
                self.reasons.append('invalid_or_duplicate_source_frame_identity')
            else:
                self.by_id[fid] = row
        self.ids = sorted(self.by_id)
        samples = [source_timestamp(self.by_id[fid]) for fid in self.ids]
        times = [t for t, _ in samples]
        bases = {basis for _, basis in samples}
        monotonic = bool(times) and all(t is not None for t in times)
        monotonic = monotonic and all(b > a for a, b in zip(times, times[1:]))
        self.qualified = (self.declared and not self.reasons and monotonic
            and bases == {'media_pts'} and all('source_time' in r for r in rows))
        legacy_timed = not self.declared and not self.reasons and monotonic
        self.timed = self.qualified or legacy_timed
        self.times = dict(zip(self.ids, times)) if self.timed else {}
        if self.qualified:
            self.basis = 'media_pts'
        elif self.declared:
            self.basis = 'unavailable'
            self.reasons.append('reported_source_time_unavailable')
        else:
            self.basis = 'legacy_timestamp_unverified' if legacy_timed else 'legacy_frame_window_unverified'
            self.reasons.append('legacy_window_time_unverified')

    def _selection(self, ids, request, reasons=()):
        ids = list(ids)
        evidence = {'policy_version': POLICY_VERSION, 'basis': self.basis,
            'source_time_qualified': self.qualified,
            'requested_window': request, 'source_frame_ids': ids,
            'sample_count': len(ids),
            'reasons': list(dict.fromkeys(self.reasons + list(reasons))),
            'observation_frame_gaps': [[a, b] for a, b in zip(ids, ids[1:]) if b != a + 1],
            'coverage_semantics': 'retained_sample_availability_not_temporal_coverage',
            'accuracy_validated': False, 'sensor_exposure_verified': False}
        evidence['actual_time_range_seconds'] = [self.times[ids[0]], self.times[ids[-1]]] if ids and self.timed else None
        return WindowSelection(ids, evidence)

    def around(self, anchor, before=.08, after=.08, legacy_before=2,
               legacy_after=2, exclude_anchor=False):
        request = {'anchor_frame_id': anchor, 'before_seconds': before,
                   'after_seconds': after, 'exclude_anchor': exclude_anchor}
        if self.declared and not self.qualified:
            return self._selection([], request)
        if anchor not in self.by_id:
            return self._selection([], request, ['missing_window_anchor_record'])
        if self.timed:
            t = self.times[anchor]
            ids = [fid for fid in self.ids
                   if t - before - 1e-9 <= self.times[fid] <= t + after + 1e-9]
        else:
            ids = [fid for fid in self.ids if anchor - legacy_before <= fid <= anchor + legacy_after]
        if exclude_anchor:
            ids = [fid for fid in ids if fid != anchor]
        return self._selection(ids, request)

    def between(self, start, end, end_fraction=1., max_duration=None):
        request = {'start_frame_id': start, 'end_frame_id': end,
                   'end_fraction': end_fraction, 'maximum_seconds': max_duration}
        if self.declared and not self.qualified:
            return self._selection([], request)
        if start not in self.by_id or end not in self.by_id or end < start:
            return self._selection([], request, ['missing_or_invalid_window_anchor'])
        if self.timed:
            duration = (self.times[end] - self.times[start]) * end_fraction
            if max_duration is not None:
                duration = min(duration, max_duration)
            ids = [fid for fid in self.ids if self.times[start] <= self.times[fid]
                   <= self.times[start] + duration + 1e-9]
        else:
            stop = end if end_fraction == 1 else min(end, start + max(1, int((end - start) * end_fraction)))
            ids = [fid for fid in self.ids if start <= fid <= stop]
        return self._selection(ids, request)

    def recovery(self, contact, end, seconds=.4):
        window = self.between(contact, end)
        if not window.frame_ids:
            return None, window
        if self.timed:
            target = self.times[contact] + seconds
            selected = min(window.frame_ids, key=lambda fid: (abs(self.times[fid] - target), fid))
        else:
            selected = min(window.frame_ids, key=lambda fid: (abs(fid - min(end, contact + 10)), fid))
        window.evidence.update(target_offset_seconds=seconds, selected_source_frame_id=selected)
        return selected, window


def attach_window(metric, window):
    metric['window_evidence'] = window.evidence
    if not window.frame_ids:
        metric.update(value=None, confidence=0., coach_eligible=False)
    return metric


def combine_windows(windows):
    windows = list(windows)
    evidence = {'policy_version': POLICY_VERSION,
        'source_time_qualified': all(w.evidence['source_time_qualified'] for w in windows),
        'basis': windows[0].evidence['basis'] if windows else 'unavailable',
        'endpoints': [w.evidence for w in windows],
        'reasons': list(dict.fromkeys(reason for w in windows for reason in w.evidence['reasons'])),
        'accuracy_validated': False, 'sensor_exposure_verified': False,
        'coverage_semantics': 'retained_sample_availability_not_temporal_coverage'}
    ids = sorted({fid for w in windows for fid in w.frame_ids})
    evidence.update(source_frame_ids=ids, sample_count=len(ids))
    return WindowSelection(ids, evidence)
