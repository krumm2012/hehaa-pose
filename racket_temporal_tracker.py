"""Recover current low-score racket candidates using bounded front-view continuity.

Scores remain raw model scores. This does not synthesize missing observations or
validate contact timing/physical measurements.
"""
import math
from observation_policy import finite_number, qualified_front_point


class RacketTemporalTracker:
    def __init__(self, config):
        self.high = float(config.get('racket_confidence_threshold', .524))
        self.low = float(config.get('racket_temporal_recovery_min_confidence', .25))
        self.max_age = float(config.get('racket_temporal_recovery_max_seconds', .16))
        self.max_weak = int(config.get('racket_temporal_recovery_max_frames', 3))
        self.anchor = self.last = None
        self.weak_count = 0

    @staticmethod
    def _time(clock, fid):
        if not isinstance(clock, dict) or clock.get('source_frame_id') != fid:
            return None
        if clock.get('basis') != 'media_pts' or clock.get('quality') != 'reported':
            return None
        return finite_number(clock.get('timestamp_seconds'))

    def select(self, candidates, front_points, frame_id, source_time, frame_size):
        diag = {'policy': 'bounded_front_racket_continuity_v1', 'recovered': False,
                'candidate_count': len(candidates), 'rejected': {}, 'accuracy_validated': False}
        diagonal = math.hypot(*frame_size)
        wrists = []
        for name, point in front_points.items():
            if 'wrist' not in name or point.get('source_frame_id') != frame_id:
                continue
            value, _ = qualified_front_point(point, frame_id)
            if value: wrists.append(value[:2])
        t = self._time(source_time, frame_id)
        valid = []
        for item in candidates:
            box = item.get('box')
            score = finite_number(item.get('confidence'))
            if not isinstance(box, (list, tuple)) or len(box) != 4 or score is None or not 0 <= score <= 1:
                continue
            box = [finite_number(v) for v in box]
            if any(v is None for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                continue
            distance = min((math.hypot(max(box[0]-x, 0, x-box[2]),
                                        max(box[1]-y, 0, y-box[3])) for x, y in wrists), default=math.inf)
            reason = None
            if distance > diagonal * .035: reason = 'no_fresh_front_wrist'
            elif score < self.low: reason = 'below_recovery_floor'
            elif score < self.high:
                if self.anchor is None or self.last is None: reason = 'no_contiguous_anchor'
                elif t is None or self.anchor['time'] is None or self.last['time'] is None: reason = 'missing_media_clock'
                elif frame_id != self.last['fid'] + 1: reason = 'frame_gap'
                elif not 0 < t-self.last['time'] <= .08: reason = 'invalid_time_step'
                elif t-self.anchor['time'] > self.max_age or self.weak_count >= self.max_weak: reason = 'anchor_expired'
                else:
                    old = self.last['box']
                    step = math.hypot((box[0]+box[2]-old[0]-old[2])/2,
                                      (box[1]+box[3]-old[1]-old[3])/2)
                    ratio = ((box[2]-box[0])*(box[3]-box[1])) / ((old[2]-old[0])*(old[3]-old[1]))
                    if step > diagonal*.06 or not 1/3 <= ratio <= 3: reason = 'geometry_discontinuity'
            if reason:
                diag['rejected'][reason] = diag['rejected'].get(reason, 0) + 1
            else: valid.append((score, -distance, item))
        if not valid:
            self.last = None
            return None, diag
        _, _, selected = max(valid, key=lambda entry: entry[:2])
        result = dict(selected, observed=True, source_frame_id=frame_id)
        state = {'box': list(selected['box']), 'fid': frame_id, 'time': t}
        if selected['confidence'] >= self.high:
            self.anchor = state
            self.weak_count = 0
        else:
            self.weak_count += 1
            result['temporal_recovery'] = {'policy': diag['policy'],
                'anchor_source_frame_id': self.anchor['fid'],
                'anchor_age_seconds': t-self.anchor['time'], 'weak_frame_count': self.weak_count}
            result['measurement_eligible'] = False
            diag['recovered'] = True
        self.last = state
        diag['selected_box'] = result['box']
        return result, diag
