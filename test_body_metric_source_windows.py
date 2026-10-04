"""Exercise real aggregation/collector windows with nonuniform reported PTS."""
from copy import deepcopy
import math
import unittest

from swing_biomechanics import (aggregate_event_biomechanics,
    _preparation_knee_flexion_metric, _early_recovery_frame,
    _calculate_extended_tier_biomechanics, _shoulder_turn_change_metric)
from swing_coach_data_collector import (_ball_metrics, _interpolate_point,
    _nearest_feature, _bounce_candidate, build_coach_dataset)


def row(fid, seconds, angle=40):
    return {'frame_id': fid, 'timestamp': fid / 25,
            'source_time': {'schema_version': 'tennis.source-time.v1',
                'source_kind': 'video_file', 'source_frame_id': fid,
                'timestamp_seconds': seconds, 'basis': 'media_pts', 'quality': 'reported'},
            'pose': {'left_shoulder': [0, 0], 'right_shoulder': [100, 0],
                     'left_hip': [0, 100], 'right_hip': [100, 100]},
            'has_pose': True, 'arm_extension_deg': angle, 'contact_score': .9}


class BodySourceWindowTests(unittest.TestCase):
    def fixture(self, stride=1):
        rows = [row(i * stride, t, 10 + i * 10)
                for i, t in enumerate([0, .02, .04, .05, .10, .12, .20])]
        event = {'start_frame': 0, 'contact_frame': 3 * stride,
                 'peak_frame': 3 * stride, 'end_frame': 6 * stride,
                 'quality_flags': {'pose_frame_ratio': 1}}
        return event, rows

    def test_contact_median_uses_point08_seconds_not_two_frame_ids(self):
        event, rows = self.fixture()
        metric = aggregate_event_biomechanics(event, rows, rows)['metrics']['arm_extension']
        self.assertEqual(metric['source_frames'], [0, 1, 2, 3, 4, 5])
        self.assertEqual(metric['value'], 35)

    def test_same_observations_and_pts_are_invariant_to_source_id_spacing(self):
        values = []
        for stride in (1, 10, 100):
            event, rows = self.fixture(stride)
            values.append(aggregate_event_biomechanics(event, rows, rows)['metrics']['arm_extension']['value'])
        self.assertEqual(values, [35, 35, 35])

    def test_preparation_is_first_half_of_source_elapsed_interval(self):
        rows = [row(i, t) for i, t in enumerate([0, .01, .02, .03, .15, .20])]
        for i, r in enumerate(rows):
            for side in ('left', 'right'):
                flex = math.radians(i * 10)
                r['pose'][side + '_hip'] = [0, 0]
                r['pose'][side + '_knee'] = [0, 100]
                r['pose'][side + '_ankle'] = [100 * math.sin(flex), 100 + 100 * math.cos(flex)]
        metric = _preparation_knee_flexion_metric({r['frame_id']: r for r in rows}, 0, 5, 1)
        self.assertEqual(metric['source_frames'], [0, 1, 2, 3])
        self.assertAlmostEqual(metric['value'], 15)

    def test_recovery_uses_source_pts_and_ignores_compatibility_clock(self):
        rows = [row(i, t) for i, t in enumerate([0, .04, .05, .08, .09, .48, .7])]
        for r in rows:
            r['timestamp'] = r['frame_id'] * .5
        self.assertEqual(_early_recovery_frame({r['frame_id']: r for r in rows}, 3, 6), 5)

    def test_invalid_declared_source_clock_cannot_produce_window_measurement(self):
        event, rows = self.fixture()
        rows[2]['source_time']['quality'] = 'duplicate'
        metric = aggregate_event_biomechanics(event, rows, rows)['metrics']['arm_extension']
        self.assertIsNone(metric['value'])
        self.assertFalse(metric['coach_eligible'])
        self.assertIn('reported_source_time_unavailable', metric['window_evidence']['reasons'])

    def test_legacy_peak_order_is_not_validated_kinetic_sequence(self):
        rows = [dict(row(i, i * .04), hip_rotation_speed=i + 1,
                     shoulder_rotation_speed=i + 1, racket_speed=i + 1) for i in range(5)]
        seq = _calculate_extended_tier_biomechanics(rows, 0, 2, 4, 100, fps=240)['kinematic_sequence']
        self.assertIsNone(seq['latency_hip_to_shoulder_ms'])
        self.assertIsNone(seq['is_sequential'])
        self.assertNotIn(seq['sequence_quality'], ('OPTIMAL', 'DISCONNECTED'))

    def test_shoulder_baseline_uses_initial_source_interval(self):
        rows = [dict(row(i, t), shoulder_line_angle_deg=i * 10)
                for i, t in enumerate([0, .01, .02, .03, .15, .20])]
        metric = _shoulder_turn_change_metric({r['frame_id']: r for r in rows}, 0, 5, 1)
        self.assertEqual(metric['window_evidence']['baseline_window']['source_frame_ids'], [0, 1, 2, 3])
        self.assertEqual(metric['value'], 35)

    def test_raw_frame_clock_authorizes_windows_without_feature_clock_fallback(self):
        event, rows = self.fixture()
        original = deepcopy(rows)
        features = [{k: v for k, v in r.items() if k != 'source_time'} for r in rows]
        for feature in features:
            feature['timestamp'] = 'unusable compatibility clock'
        metric = aggregate_event_biomechanics(event, rows, features)['metrics']['arm_extension']
        self.assertEqual(metric['value'], 35)
        self.assertTrue(metric['contract']['window_evidence']['source_time_qualified'])
        self.assertEqual(rows, original)

    def test_estimated_source_clock_does_not_authorize_media_windows(self):
        event, rows = self.fixture()
        for r in rows:
            r['source_time'].update(basis='nominal_fps', quality='estimated')
        metric = aggregate_event_biomechanics(event, rows, rows)['metrics']['arm_extension']
        self.assertIsNone(metric['value'])
        self.assertIn('reported_source_time_unavailable', metric['contract']['missing_reasons'])

    def test_source_identity_gaps_are_disclosed_without_temporal_coverage_claim(self):
        event, rows = self.fixture(stride=10)
        evidence = aggregate_event_biomechanics(event, rows, rows)['metrics']['arm_extension']['window_evidence']
        self.assertTrue(evidence['observation_frame_gaps'])
        self.assertEqual(evidence['coverage_semantics'], 'retained_sample_availability_not_temporal_coverage')

    def test_nonfinite_body_feature_is_not_a_measurement(self):
        event, rows = self.fixture()
        for r in rows:
            r['arm_extension_deg'] = float('nan')
        metric = aggregate_event_biomechanics(event, rows, rows)['metrics']['arm_extension']
        self.assertIsNone(metric['value'])
        self.assertEqual(metric['confidence'], 0)


class CollectorSourceWindowTests(unittest.TestCase):
    def test_ball_paths_use_contact_point20_second_windows(self):
        rows = [dict(row(i, t), ball=p, ball_speed=1)
                for i, (t, p) in enumerate(zip([0, .6, .82, .9, 1, 1.04, 1.15, 1.4],
                    [[0, 0], [10, 20], [20, 30], [30, 30], [40, 30], [50, 30], [60, 30], [70, 60]]))]
        metric = _ball_metrics(rows, 4, {r['frame_id']: r for r in rows})
        self.assertEqual(metric['incoming_angle_deg'], 0)
        self.assertEqual(metric['outgoing_angle_deg'], 0)
        self.assertEqual(metric['incoming_window']['source_frame_ids'], [2, 3])
        self.assertEqual(metric['outgoing_window']['source_frame_ids'], [5, 6])

    def test_missing_anchor_is_not_nearest_frame_observation(self):
        self.assertEqual(_nearest_feature({0: row(0, 0), 10: row(10, .4)}, 5), {})

    def test_interpolation_uses_source_elapsed_fraction(self):
        rows = [dict(row(i, t), racket_center=p)
                for i, (t, p) in enumerate(zip([0, .01, .1], [[0, 0], None, [100, 0]]))]
        point, source, _ = _interpolate_point({r['frame_id']: r for r in rows}, 1, 'racket_center')
        self.assertEqual(point, (10, 0))
        self.assertEqual(source, 'interpolated')

    def test_long_source_gap_cannot_be_interpolated(self):
        rows = [dict(row(i, t), racket_center=p)
                for i, (t, p) in enumerate(zip([0, .5, 1], [[0, 0], None, [100, 0]]))]
        point, source, _ = _interpolate_point({r['frame_id']: r for r in rows}, 1, 'racket_center')
        self.assertIsNone(point)
        self.assertEqual(source, 'missing')

    def test_existing_candidate_point_keeps_interpolated_source(self):
        r = dict(row(0, 0), racket_center=[10, 20], racket_center_source='interpolated')
        point, source, confidence = _interpolate_point({0: r}, 0, 'racket_center')
        self.assertEqual(point, (10, 20))
        self.assertEqual(source, 'interpolated')
        self.assertEqual(confidence, 0)

    def test_bounce_candidate_cannot_bridge_missing_source_frames(self):
        rows = [dict(row(fid, fid * .04), ball=[100, y])
                for fid, y in zip([1, 3, 5, 7], [0, 10, 0, -10])]
        self.assertIsNone(_bounce_candidate(rows, 0)[0])

    def test_dataset_uses_raw_source_clock_and_retains_unmeasured_speed_units(self):
        rows = [dict(row(i, t), ball=[i * 10, 30])
                for i, t in enumerate([0, .6, .82, .9, 1, 1.04, 1.15, 1.4])]
        features = [{k: v for k, v in r.items() if k != 'source_time'} for r in rows]
        for f in features:
            f.update(timestamp=999, ball_speed=1, contact_score=.9 if f['frame_id'] == 4 else 0)
        event = {'event_id': 1, 'start_frame': 0, 'contact_frame': 4, 'peak_frame': 4,
                 'end_frame': 7, 'duration_frames': 8}
        result = build_coach_dataset({'video_info': {'fps': 240}, 'frames': rows},
                                    {'events': [event], 'features': features})['events'][0]['ball']
        self.assertEqual(result['incoming_window']['source_frame_ids'], [2, 3])
        self.assertEqual(result['outgoing_window']['source_frame_ids'], [5, 6])
        self.assertEqual(result['speed_units'], 'pixels_per_observation_not_per_second_or_kmh')

    def test_collector_body_does_not_measure_fused_or_cached_pose(self):
        rows = [dict(row(i, i * .04), ball=[40, 30]) for i in range(5)]
        for r in rows:
            r['pose_observations'] = {'front': {k: {'x': p[0], 'y': p[1],
                'confidence': .9, 'observed': True, 'confidence_source': 'model',
                'source_frame_id': r['frame_id']} for k, p in r['pose'].items()}}
            r['pose'] = {k: [p[0] + 1000, p[1] + 1000] for k, p in r['pose'].items()}
        event = {'event_id': 1, 'start_frame': 0, 'contact_frame': 2, 'peak_frame': 2,
                 'end_frame': 4, 'duration_frames': 5}
        body = build_coach_dataset({'frames': rows}, {'events': [event], 'features': rows})['events'][0]['body']
        self.assertEqual(body['body_center_at_contact'], (50, 50))


if __name__ == '__main__':
    unittest.main()
