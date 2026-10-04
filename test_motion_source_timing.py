"""Source-clock counterexamples through actual feature/segmentation seams."""
from copy import deepcopy
import math
import unittest

from swing_motion_features import extract_motion_features
from swing_event_segmenter import _motion_energy, _event_quality_flags, segment_swing_events
from motion_time_contract import candidate_timeline
from realtime_swing_pipeline import RealtimeSwingEventEngine


def frame(fid, seconds, x=0., compatibility_time=None):
    return {'frame_id': fid, 'timestamp': seconds if compatibility_time is None else compatibility_time,
            'source_time': {'schema_version': 'tennis.source-time.v1',
                'source_kind': 'video_file', 'source_frame_id': fid,
                'timestamp_seconds': seconds, 'basis': 'media_pts', 'quality': 'reported'},
            'rackets': [{'box': [x-2, -2, x+2, 2], 'confidence': .9,
                         'observed': True, 'source_frame_id': fid}]}


class MotionSourceTimingTests(unittest.TestCase):
    def test_source_ending_during_motion_retains_unsettled_candidate_for_review(self):
        engine = RealtimeSwingEventEngine(fps=25, execution_mode='replay',
                                          min_event_frames=3, min_event_gap=3)
        for i in range(12):
            row = frame(i, i*.04, max(0, i-3)*20.)
            row['swing_type'] = 'Forehand'
            engine.push_frame(row)
        events = engine.flush()
        self.assertTrue(events)
        self.assertEqual(events[-1]['candidate_runtime_timing']['completion_status'], 'source_end_unsettled_candidate')
        self.assertIn('source_end_without_settle_confirmation', events[-1]['quality_flags']['warnings'])

    def test_source_contact_quality_window_uses_elapsed_time(self):
        times = [0., .04, .08, .12, .16, .3, .4, .5, .6]
        features, _ = candidate_timeline([frame(i, t) for i,t in enumerate(times)])
        quality = _event_quality_flags(features, 0, 8, {'evidence': {}}, 4)
        self.assertEqual(quality['ball_contact_window_frames'], 6)

    def test_legacy_angular_step_wraps_across_half_turn_boundary(self):
        rows = []
        for i, degrees in enumerate((179., -179.)):
            row = frame(i, i*.04)
            endpoint = [20*math.cos(math.radians(degrees)), 20*math.sin(math.radians(degrees))]
            row['pose'] = {'left_shoulder': [0,0], 'right_shoulder': endpoint,
                           'left_hip': [0,10], 'right_hip': [endpoint[0], endpoint[1]+10]}
            rows.append(row)
        features = extract_motion_features(rows)
        self.assertEqual(features[1]['shoulder_rotation_speed'], 2.)
        self.assertEqual(features[1]['hip_rotation_speed'], 2.)

    def test_reported_source_engine_is_independent_of_declared_fps(self):
        rows = []
        x = 0.
        for i in range(180):
            if 20 <= i <= 30 or 90 <= i <= 105:
                x += 20.
            row = frame(i, i*.04, x)
            row['swing_type'] = 'Forehand'
            row['pose'] = {'right_wrist': [x, 20], 'left_wrist': [x-100, 20],
                           'left_shoulder': [0, 0], 'right_shoulder': [20, 0],
                           'right_elbow': [x-10, 10]}
            rows.append(row)
        signatures = []
        for fps in (10, 25, 50, 240):
            engine = RealtimeSwingEventEngine(fps=fps, execution_mode='replay')
            for row in rows:
                engine.push_frame(row)
            engine.flush()
            signatures.append([(v['start_frame'], v['contact_frame'], v['peak_frame'], v['end_frame'])
                               for v in engine.snapshot()['events']])
        self.assertEqual(len(signatures[1]), 2)
        for signature in signatures:
            self.assertEqual(signature, signatures[1])

    def test_same_source_pts_ignore_compatibility_clock_and_receipt_time(self):
        rows = [frame(0, 0.), frame(1, .04, 10.)]
        variant = deepcopy(rows)
        variant[1]['timestamp'] = .4
        variant[1]['timing'] = {'captured_at_unix_ns': 123456789}
        a, b = extract_motion_features(rows), extract_motion_features(variant)
        self.assertEqual([f['timestamp'] for f in a], [f['timestamp'] for f in b])
        self.assertEqual(a[-1]['racket_speed_px_s'], b[-1]['racket_speed_px_s'])

    def test_valid_source_clock_does_not_parse_a_broken_compatibility_field(self):
        rows = [frame(0, 0.), frame(1, .04, 10., 'invalid')]
        features = extract_motion_features(rows)
        self.assertEqual(features[-1]['timestamp'], .04)

    def test_vfr_interpolation_uses_elapsed_time(self):
        rows = [frame(0, 0., 0.), frame(1, .01), frame(2, .12, 12.)]
        rows[1]['rackets'] = []
        middle = extract_motion_features(rows)[1]
        self.assertEqual(middle['racket_center_source'], 'interpolated')
        self.assertAlmostEqual(middle['racket_center'][0], 1.)
        self.assertIsNone(middle['racket_speed_px_s'])

    def test_large_elapsed_gap_cannot_be_healed_as_two_nearby_rows(self):
        rows = [frame(0, 0., 0.), frame(1, .4), frame(2, .8, 12.)]
        rows[1]['rackets'] = []
        self.assertIsNone(extract_motion_features(rows)[1]['racket_center'])

    def test_source_identity_gap_cannot_be_healed(self):
        rows = [frame(0, 0., 0.), frame(5, .01), frame(6, .02, 12.)]
        rows[1]['rackets'] = []
        self.assertIsNone(extract_motion_features(rows)[1]['racket_center'])

    def test_repeated_clock_cannot_generate_interpolation_or_measured_speed(self):
        rows = [frame(0, 0., 0.), frame(1, 0.), frame(2, .08, 12.)]
        rows[1]['rackets'] = []
        self.assertIsNone(extract_motion_features(rows)[1]['racket_center'])

    def test_equal_physical_velocity_produces_equal_candidate_energy(self):
        fast = extract_motion_features([frame(0, 0.), frame(1, .04, 4.)])
        slow = extract_motion_features([frame(0, 0.), frame(1, .08, 8.)])
        self.assertAlmostEqual(_motion_energy(fast[1]), _motion_energy(slow[1]))

    def test_wrong_source_identity_cannot_produce_image_speed(self):
        rows = [frame(0, 0.), frame(1, .04, 10.)]
        rows[1]['source_time']['source_frame_id'] = 99
        self.assertIsNone(extract_motion_features(rows)[1]['racket_speed_px_s'])

    def test_duplicate_source_identity_cannot_qualify_a_candidate_timeline(self):
        rows = [frame(0, 0.), frame(0, .04, 10.)]
        _, evidence = candidate_timeline(rows)
        self.assertFalse(evidence['source_time_qualified'])
        self.assertEqual(evidence['basis'], 'processing_order_unverified')

    def test_short_motion_islands_respect_source_elapsed_gap(self):
        features = []
        for i in range(20):
            row = frame(i, i*.04 + (1. if i >= 8 else 0.))
            row.update(raw_swing_type='Forehand', wrist_speed=20. if i in range(2, 6) or i in range(10, 14) else 0.,
                       racket_speed=0., wrist_accel=0., contact_score=0.)
            features.append(row)
        result = segment_swing_events(features, min_event_frames=3, min_peak_energy=10.,
                                      active_energy=5., max_internal_gap=2, min_event_gap=6)
        self.assertEqual(len(result['events']), 2)

    def test_invalid_source_clock_keeps_candidates_but_abstains_from_media_time(self):
        features = []
        for i in range(20):
            row = frame(i, i*.04)
            row.update(raw_swing_type='Forehand', wrist_speed=20. if 4 <= i <= 12 else 0.,
                       racket_speed=0., wrist_accel=0., contact_score=0.)
            features.append(row)
        features[10]['source_time']['quality'] = 'duplicate'
        result = segment_swing_events(features, min_event_frames=3, min_peak_energy=10.)
        self.assertTrue(result['events'])
        self.assertFalse(result['candidate_timing']['source_time_qualified'])
        self.assertEqual(result['candidate_timing']['basis'], 'processing_order_unverified')


if __name__ == '__main__':
    unittest.main()
