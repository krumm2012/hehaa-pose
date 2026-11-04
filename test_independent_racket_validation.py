import copy
import unittest
from independent_racket_validation import evaluate


class IndependentErrorsTests(unittest.TestCase):
    def setUp(self):
        self.ref = {'schema': 'tennis.independent-racket-reference.v1', 'source_sha256': 'a'*64,
                    'coordinate_space': 'original_source_pixels', 'frame_index_base': 0,
                    'image_size': [100, 100], 'reviewer': 'test reviewer',
                    'split_provenance': 'held out before fitting', 'reference_evidence': 'blind source review',
                    'model_assisted': False, 'confirmed': True,
                    'frames': [{'frame_id': i, 'source_seconds': i*.04,
                                'points': {'sweet_spot': [10, 10]}} for i in range(4)],
                    'contact_pairs': [{'reference_id': 'r1', 'start_frame': 1, 'end_frame': 2}]}
        self.pred = {'schema': 'tennis.independent-racket-prediction.v1', 'source_sha256': 'a'*64,
                     'model_sha256': 'b'*64, 'coordinate_space': 'original_source_pixels',
                     'frame_index_base': 0, 'image_size': [100, 100],
                     'frames': [{'frame_id': 0, 'points': {'sweet_spot': [13, 14]}}],
                     'contacts': {'r1': 3}}

    def test_missing_points_count_against_coverage_and_source_pts_error(self):
        result = evaluate(self.pred, self.ref, [], 5)
        sweet = next(p for p in result['points'] if p['point'] == 'sweet_spot')
        self.assertEqual(sweet['mean_error_px'], 5)
        self.assertEqual(sweet['missing_predictions'], 3)
        self.assertEqual(sweet['within_tolerance_over_visible'], .25)
        self.assertAlmostEqual(result['contact_pairs'][0]['outside_interval_ms'], 40)
        self.assertFalse(result['active_or_exclusive_allowed'])

    def test_development_source_and_assisted_labels_rejected(self):
        with self.assertRaises(ValueError): evaluate(self.pred, self.ref, ['a'*64], 5)
        self.ref['model_assisted'] = True
        with self.assertRaises(ValueError): evaluate(self.pred, self.ref, [], 5)

    def test_mismatched_source_and_duplicate_frames_rejected(self):
        p = copy.deepcopy(self.pred); p['source_sha256'] = 'c'*64
        with self.assertRaises(ValueError): evaluate(p, self.ref, [], 5)
        self.pred['frames'] *= 2
        with self.assertRaises(ValueError): evaluate(self.pred, self.ref, [], 5)
