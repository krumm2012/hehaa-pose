import copy
import unittest

from scripts.accept_contact_racket_review import validate_review


def fixture():
    audit = {'source_sha256': 'source', 'inputs_sha256': {'events': 'events'},
             'session_id': 'session', 'coordinate_space': 'original_source_pixels',
             'frame_index_base': 0, 'events': [{'event_id': 1, 'frames': [
                 {'frame_id': 20, 'source_pts_seconds': .8, 'fresh_rackets': [{}]},
                 {'frame_id': 21, 'source_pts_seconds': .84, 'fresh_rackets': []},
                 {'frame_id': 22, 'source_pts_seconds': .88, 'fresh_rackets': [{}]}]}]}
    review = {k: audit[k] for k in ('source_sha256', 'inputs_sha256', 'session_id', 'coordinate_space', 'frame_index_base')}
    review.update(schema='tennis.contact-racket-review.v1', annotation_mode='model_assisted_review',
                  independent_reference=False, annotator_id='reviewer', event_reviews_complete=True,
                  events={'1': {'decision': 'interval', 'start_frame': 20, 'end_frame': 22,
                                'racket_roles': {'20:0': 'front'}}},
                  contact_interval_pts={'1': {'start_seconds': .8, 'end_seconds': .88}})
    return review, audit


class ContactReviewReceiptTests(unittest.TestCase):
    def test_accepts_interval_but_does_not_invent_roles_or_truth(self):
        review, audit = fixture()
        before = copy.deepcopy((review, audit))
        receipt = validate_review(review, audit)
        self.assertTrue(receipt['event_reviews_complete'])
        self.assertEqual(receipt['events'][0]['annotated_racket_roles'], 1)
        self.assertFalse(receipt['events'][0]['racket_roles_complete'])
        self.assertFalse(receipt['contact_truth_verified'])
        self.assertEqual((review, audit), before)

    def test_rejects_cross_session_and_stale_role_identity(self):
        review, audit = fixture()
        review['session_id'] = 'other'
        with self.assertRaises(ValueError): validate_review(review, audit)
        review, audit = fixture()
        review['events']['1']['racket_roles'] = {'21:0': 'front'}
        with self.assertRaises(ValueError): validate_review(review, audit)

    def test_rejects_wrong_pts_and_outside_window(self):
        review, audit = fixture()
        review['contact_interval_pts']['1']['end_seconds'] = .9
        with self.assertRaises(ValueError): validate_review(review, audit)
        review, audit = fixture()
        review['events']['1']['end_frame'] = 23
        with self.assertRaises(ValueError): validate_review(review, audit)

    def test_rejects_false_completion_and_independent_claim(self):
        review, audit = fixture()
        review['event_reviews_complete'] = False
        with self.assertRaises(ValueError): validate_review(review, audit)
        review, audit = fixture()
        review['independent_reference'] = True
        with self.assertRaises(ValueError): validate_review(review, audit)


if __name__ == '__main__':
    unittest.main()
