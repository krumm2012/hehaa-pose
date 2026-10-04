"""Classifier call-site regressions for media windows and bounded candidates."""
from copy import deepcopy
import unittest

from motion_time_contract import candidate_timeline
from swing_event_classifier import classify_swing_event
from swing_event_segmenter import _classify_impact_event, _best_contact_frame
from swing_biomechanics import _calculate_extended_tier_biomechanics
from image_motion_measurements import POLICY_VERSION as IMAGE_MOTION_POLICY


def feature(fid, seconds, stroke='Forehand', weight=1):
    return {'frame_id': fid, 'timestamp': seconds,
        'source_time': {'schema_version': 'tennis.source-time.v1',
            'source_kind': 'video_file', 'source_frame_id': fid,
            'timestamp_seconds': seconds, 'basis': 'media_pts', 'quality': 'reported'},
        'raw_swing_type': stroke, 'dual_view_stroke_type': stroke,
        'dual_view_confidence': .9, 'wrist_speed': weight}


class ClassifierSourceWindowTests(unittest.TestCase):
    def test_equal_contact_candidates_choose_nearest_source_time(self):
        rows = [dict(feature(fid, t), contact_score=score)
                for fid, t, score in [(9, .1, .8), (10, 1., 0), (12, 1.01, .8)]]
        prepared, _ = candidate_timeline(rows)
        evidence = {}
        self.assertEqual(_best_contact_frame(prepared, 0, 2, 1, evidence), 12)
        self.assertEqual(evidence['tie_break_basis'], 'media_pts')
        self.assertEqual(evidence['tied_candidate_frame_ids'], [9, 12])

    def test_legacy_anchor_choice_is_explicitly_unverified(self):
        rows = [dict(feature(fid, t), contact_score=score)
                for fid, t, score in [(9, .1, .8), (10, 1., 0), (12, 1.01, .8)]]
        for r in rows:
            r.pop('source_time')
        evidence = {}
        self.assertEqual(_best_contact_frame(rows, 0, 2, 1, evidence), 9)
        self.assertFalse(evidence['source_time_qualified'])
        self.assertFalse(evidence['coach_eligible'])

    def test_bad_contact_scores_leave_only_a_motion_peak_candidate(self):
        rows = [dict(feature(i, i * .04), contact_score=score)
                for i, score in enumerate([float('nan'), float('inf'), -1])]
        evidence = {}
        self.assertEqual(_best_contact_frame(rows, 0, 2, 1, evidence), 1)
        self.assertIn('no_positive_contact_score_using_motion_peak_candidate', evidence['reasons'])

    def test_actual_impact_classifier_is_invariant_to_source_id_spacing(self):
        results = []
        for stride in (1, 10):
            rows = [feature(i * stride, t, 'Forehand' if i < 3 else 'Backhand',
                            1 if i < 3 else 30)
                    for i, t in enumerate([0, .01, .02, .03, .04, .05])]
            prepared, _ = candidate_timeline(rows)
            results.append(_classify_impact_event(prepared, 0, 5, 2, 25)['stroke_type'])
        self.assertEqual(results, ['Backhand', 'Backhand'])

    def test_late_follow_through_cannot_override_preimpact_classification(self):
        rows = [feature(i, t, 'Forehand' if i < 3 else 'Backhand',
                        1 if i < 3 else 30)
                for i, t in enumerate([0, .01, .02, .20, .30])]
        self.assertEqual(classify_swing_event(rows, contact_frame=2)['stroke_type'], 'Forehand')

    def test_sparse_core_abstains_instead_of_reusing_full_follow_through(self):
        rows = [feature(i * 10, t, 'Backhand' if i < 2 else 'Forehand')
                for i, t in enumerate([0, .01, .20, .30, .40, .50, .60, .70])]
        self.assertEqual(classify_swing_event(rows, contact_frame=10)['stroke_type'], 'Unknown')

    def test_contact_geometry_outside_time_window_is_not_contact_evidence(self):
        rows = [feature(i, t) for i, t in enumerate([0, .30, .31, .32])]
        for i, row in enumerate(rows):
            row['ball_racket_distance'] = 10 if i == 0 else 1000
        result = classify_swing_event(rows, contact_frame=3)
        self.assertEqual(result['contact_status'], 'unknown')
        self.assertEqual(result['min_ball_distance'], 1000)

    def test_contact_window_is_separate_from_shorter_stroke_window(self):
        rows = [feature(i * 10, t) for i, t in enumerate([0, .01, .02, .12])]
        for i, row in enumerate(rows):
            row['ball_racket_distance'] = 10 if i == 3 else 1000
        result = classify_swing_event(rows, contact_frame=20)
        self.assertEqual(result['contact_status'], 'candidate')
        self.assertEqual(result['evidence']['classification_context']['contact_analysis']['closest_evidence_frame'], 30)

    def test_missing_contact_anchor_does_not_borrow_a_nearby_classification(self):
        rows = [feature(i, i * .04) for i in (0, 1, 3, 4)]
        result = classify_swing_event(rows, contact_frame=2)
        self.assertEqual(result['stroke_type'], 'Unknown')
        self.assertEqual(result['contact_status'], 'unknown')

    def test_window_selection_and_raw_input_provenance_are_preserved(self):
        rows = [feature(i, t) for i, t in enumerate([0, .01, .02, .10, .20])]
        for r in rows:
            r['timestamp'] = r['frame_id'] * 10
        original = deepcopy(rows)
        result = classify_swing_event(rows, contact_frame=2)
        window = result['evidence']['classification_context']['window_evidence']
        self.assertEqual(window['source_frame_ids'], [0, 1, 2, 3])
        self.assertTrue(window['source_time_qualified'])
        self.assertFalse(window['accuracy_validated'])
        self.assertEqual(rows, original)

    def test_unavailable_stream_time_keeps_only_disclosed_bounded_candidate(self):
        rows = [feature(i, i * .04) for i in range(10)]
        for r in rows:
            r['source_time'].update(source_kind='stream', basis='unavailable',
                                    quality='unavailable', timestamp_seconds=None)
        result = classify_swing_event(rows, contact_frame=2)
        window = result['evidence']['classification_context']['window_evidence']
        self.assertFalse(window['source_time_qualified'])
        self.assertEqual(window['source_frame_ids'], [0, 1, 2, 3, 4])
        self.assertIn('candidate_window_time_unverified', window['reasons'])
        self.assertEqual(result['stroke_type'], 'Forehand')

    def test_image_speed_metadata_uses_implemented_policy(self):
        rows = [dict(feature(i, i * .04), racket_speed_px_s=10,
                     racket_speed_time_basis='media_pts') for i in range(3)]
        extended = _calculate_extended_tier_biomechanics(rows, 0, 1, 2, 100)
        self.assertEqual(extended['racket_head_speed']['measurement_policy'], IMAGE_MOTION_POLICY)

    def test_invalid_declared_clocks_do_not_borrow_compatibility_time(self):
        for kind in ('estimated', 'duplicate', 'mixed', 'identity_mismatch', 'malformed'):
            with self.subTest(kind=kind):
                rows = [feature(i, i * .04) for i in range(6)]
                if kind == 'estimated':
                    for r in rows:
                        r['source_time'].update(basis='nominal_fps', quality='estimated')
                elif kind == 'duplicate':
                    rows[1]['source_time']['timestamp_seconds'] = 0
                elif kind == 'mixed':
                    rows[1].pop('source_time')
                elif kind == 'identity_mismatch':
                    rows[1]['source_time']['source_frame_id'] = 999
                else:
                    rows[1]['source_time'] = None
                result = classify_swing_event(rows, contact_frame=2)
                window = result['evidence']['classification_context']['window_evidence']
                self.assertFalse(window['source_time_qualified'])
                self.assertEqual(window['basis'], 'source_frame_offsets_unverified')
                self.assertIsNone(window['actual_time_range_seconds'])
                self.assertEqual(result['stroke_type'], 'Forehand')

    def test_duplicate_identity_cannot_inflate_classification_votes(self):
        rows = [feature(1, .04) for _ in range(3)]
        self.assertEqual(classify_swing_event(rows, contact_frame=1)['stroke_type'], 'Unknown')

    def test_nonfinite_or_negative_contact_distance_is_missing(self):
        for distance in (float('nan'), float('inf'), -1, True):
            rows = [dict(feature(i, i * .04), ball_racket_distance=distance) for i in range(3)]
            result = classify_swing_event(rows, contact_frame=1)
            self.assertIsNone(result['min_ball_distance'])
            self.assertEqual(result['contact_status'], 'unknown')

    def test_legacy_timed_candidate_is_disclosed_and_bounded(self):
        rows = [feature(i, t) for i, t in enumerate([0, .01, .02, .30])]
        for r in rows:
            r.pop('source_time')
        result = classify_swing_event(rows, contact_frame=2)
        window = result['evidence']['classification_context']['window_evidence']
        self.assertEqual(window['source_frame_ids'], [0, 1, 2])
        self.assertEqual(window['basis'], 'legacy_timestamp_unverified')
        self.assertFalse(window['source_time_qualified'])
        self.assertFalse(window['coach_eligible'])

    def test_missing_clock_sparse_candidate_does_not_restore_whole_event(self):
        rows = [feature(i * 10, i * .04) for i in range(8)]
        for r in rows:
            r['source_time'] = None
        self.assertEqual(classify_swing_event(rows, contact_frame=10)['stroke_type'], 'Unknown')


if __name__ == '__main__':
    unittest.main()
