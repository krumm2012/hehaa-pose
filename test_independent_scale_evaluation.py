import copy
import unittest
from independent_scale_evaluation import evaluate_scale_checks
from ground_reference import normalize_calibration
from test_ground_reference import calibration


class IndependentScaleTests(unittest.TestCase):
    def setUp(self):
        self.cal = normalize_calibration(calibration())
        self.review = {'schema': 'tennis.independent-scale-check-draft.v1',
            'calibration_id': self.cal['calibration_id'], 'calibration_sha256': 'b'*64,
            'source_sha256': 'a'*64, 'binding': self.cal['binding'],
            'coordinate_space': 'original_source_pixels', 'frame_index_base': 0,
            'frame_id': 110, 'image_size': self.cal['image_size'], 'confirmed': True,
            'annotator_id': 'fixture', 'instrument': 'synthetic ruler',
            'measurement_evidence': 'Synthetic test geometry, not real evidence',
            'measurement_uncertainty_m': .01,
            'check_points': [{'id': str(i), 'view': 'front', 'image': xy, 'world_m': world}
                for i, (xy, world) in enumerate([([60,120],[.5,1]),([160,120],[1.5,1]),([60,220],[.5,2])])]}

    def run_review(self, review=None):
        return evaluate_scale_checks(review or self.review, self.cal, 'b'*64, .05)

    def test_reports_measured_error_without_mutating_calibration(self):
        old = copy.deepcopy(self.cal)
        result = self.run_review()
        self.assertTrue(result['groups'][0]['evaluation_passed'])
        self.assertAlmostEqual(result['groups'][0]['max_error_m'], 0)
        self.assertEqual(self.cal, old)
        self.assertFalse(result['accuracy_validated'])
        self.assertEqual(result['untested_views'], ['back'])

    def test_excess_error_and_uncertainty_fail_evaluation(self):
        self.review['check_points'][0]['world_m'][0] += .2
        self.assertFalse(self.run_review()['groups'][0]['evaluation_passed'])
        self.review['measurement_uncertainty_m'] = .2
        self.assertFalse(self.run_review()['groups'][0]['evaluation_passed'])

    def test_rejects_fit_corner_and_duplicate_points(self):
        for image in [[10,20],[160,120]]:
            review = copy.deepcopy(self.review)
            review['check_points'][0]['image'] = image
            with self.assertRaises(ValueError): self.run_review(review)

    def test_collinear_points_cannot_pass(self):
        self.review['check_points'][2].update(image=[260,120], world_m=[2.5,1])
        self.assertFalse(self.run_review()['groups'][0]['evaluation_passed'])

    def test_empty_draft_is_pending_and_wrong_identity_rejected(self):
        self.review.update(confirmed=False, check_points=[])
        self.assertEqual(self.run_review()['status'], 'awaiting_physical_measurements')
        self.review['calibration_sha256'] = 'c'*64
        with self.assertRaises(ValueError): self.run_review()

    def test_invalid_provenance_and_nonfinite_point_rejected(self):
        for field, value in [('instrument',''), ('measurement_uncertainty_m',float('nan'))]:
            review=copy.deepcopy(self.review); review[field]=value
            with self.assertRaises(ValueError): self.run_review(review)
        self.review['check_points'][0]['world_m'][0] = True
        with self.assertRaises(ValueError): self.run_review()


if __name__ == '__main__': unittest.main()
