import copy
import unittest
from scripts.audit_contact_racket_evidence import analyze, qualified_racket, frame_evidence


def row(fid, time):
    return {'frame_id': fid, 'source_time': {'schema_version': 'tennis.source-time.v1',
            'source_frame_id': fid, 'source_kind': 'video_file', 'basis': 'media_pts',
            'quality': 'reported', 'timestamp_seconds': time}, 'rackets': []}


class ContactRacketEvidenceTests(unittest.TestCase):
    def test_stale_and_unbound_boxes_are_rejected(self):
        c = {'box': [1, 2, 10, 20], 'confidence': .8, 'observed': True, 'source_frame_id': 2}
        self.assertEqual(qualified_racket(c, 3, [100, 100])[1], 'source_frame_mismatch')
        c['source_frame_id'] = 3; c['observed'] = False
        self.assertEqual(qualified_racket(c, 3, [100, 100])[1], 'not_fresh_observation')
        c['observed'] = True; del c['source_frame_id']
        self.assertIsNone(qualified_racket(c, 3, [100, 100])[0])

    def test_invalid_coordinates_and_scores(self):
        c = {'box': [1, 2, 10, 20], 'confidence': .8, 'observed': True, 'source_frame_id': 3}
        for box in [[1, 2, 0, 20], [1, 2, float('nan'), 20], [-1, 2, 10, 20]]:
            self.assertIsNone(qualified_racket({**c, 'box': box}, 3, [100, 100])[0])
        self.assertIsNone(qualified_racket({**c, 'confidence': True}, 3, [100, 100])[0])

    def test_source_pts_window_does_not_use_frame_distance(self):
        rows = [row(10, 0), row(11, .1), row(12, .2), row(13, 1.)]
        event = {'event_id': 1, 'start_frame': 10, 'end_frame': 13, 'contact_frame': 12}
        result = analyze(rows, [event], [100, 100], .12)['events'][0]
        self.assertEqual([r['frame_id'] for r in result['frames']], [11, 12])
        self.assertFalse(result['contact_verified'])

    def test_nonmonotonic_and_missing_anchor_abstain(self):
        rows = [row(10, .1), row(11, .1)]
        event = {'event_id': 1, 'start_frame': 10, 'end_frame': 11, 'contact_frame': 10}
        self.assertEqual(analyze(rows, [event], [100, 100])['events'][0]['status'], 'source_time_unqualified')
        event['contact_frame'] = None
        self.assertEqual(analyze(rows, [event], [100, 100])['events'][0]['status'], 'missing_contact_anchor')

    def test_inside_box_never_confirms_contact(self):
        r = row(3, .1)
        r['rackets'] = [{'box': [1, 2, 10, 20], 'confidence': .8, 'observed': True, 'source_frame_id': 3}]
        r['ball_observation'] = {'schema_version': 'tennis.ball-observation.v1', 'position': [5, 5],
                'source': 'model_detection', 'observed': True, 'source_frame_id': 3,
                'coordinate_space': 'original_source_pixels', 'model_confidence': .8, 'measurement_eligible': True}
        prior = copy.deepcopy(r); result = frame_evidence(r, [100, 100])
        self.assertTrue(result['joint_ball_racket_observation_available'])
        self.assertEqual(result['fresh_rackets'][0]['ball_to_box_distance_px'], 0)
        self.assertFalse(result['contact_verified'])
        self.assertFalse(result['fresh_rackets'][0]['person_assignment_verified'])
        self.assertEqual(r, prior)

    def test_duplicate_identities_fail(self):
        r = row(3, .1)
        with self.assertRaises(ValueError): analyze([r, r], [], [100, 100])
        e = {'event_id': 1, 'start_frame': 3, 'end_frame': 3, 'contact_frame': 3}
        with self.assertRaises(ValueError): analyze([r], [e, e], [100, 100])


if __name__ == '__main__': unittest.main()
