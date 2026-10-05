import unittest
from racket_temporal_tracker import RacketTemporalTracker


def clock(fid, t):
    return {'source_frame_id': fid, 'source_kind': 'video_file', 'basis': 'media_pts',
            'quality': 'reported', 'timestamp_seconds': t}


def wrist(fid, x=1280, y=550):
    return {'right_wrist': {'x': x, 'y': y, 'confidence': .9, 'observed': True,
                            'source_frame_id': fid, 'confidence_source': 'model'}}


def candidate(box, score):
    return {'box': box, 'confidence': score}


class RacketTemporalTrackerTests(unittest.TestCase):
    def tracker(self):
        return RacketTemporalTracker({'racket_confidence_threshold': .524,
                                      'racket_temporal_recovery_min_confidence': .25})

    def test_blurred_front_candidates_beat_high_scoring_mirror(self):
        tracker = self.tracker()
        seed = candidate([1090, 473, 1288, 568], .874023)
        self.assertIsNotNone(tracker.select([seed], wrist(20), 20, clock(20, .77), [2560, 1440])[0])
        for fid, box, score in [(21, [1101, 498, 1294, 600], .477539),
                                (22, [1100, 512, 1316, 614], .270020),
                                (23, [1112, 513, 1347, 620], .304443)]:
            mirror = candidate([1182, 61, 1316, 123], .91)
            result, diagnostics = tracker.select([mirror, candidate(box, score)], wrist(fid), fid,
                                                  clock(fid, .77 + (fid - 20) * .04), [2560, 1440])
            self.assertEqual(result['box'], box)
            self.assertEqual(result['confidence'], score)
            self.assertTrue(result['observed'])
            self.assertEqual(result['source_frame_id'], fid)
            self.assertEqual(result['temporal_recovery']['anchor_source_frame_id'], 20)
            self.assertFalse(result['measurement_eligible'])
        result, _ = tracker.select([candidate([1120, 513, 1347, 620], .3)], wrist(24), 24,
                                    clock(24, .93), [2560, 1440])
        self.assertIsNone(result, 'Weak observations cannot perpetually renew the anchor')

    def test_no_seed_no_candidate_and_frame_gap_do_not_create_detection(self):
        weak = candidate([1101, 498, 1294, 600], .4)
        tracker = self.tracker()
        self.assertIsNone(tracker.select([weak], wrist(21), 21, clock(21, .84), [2560, 1440])[0])
        tracker.select([candidate(weak['box'], .8)], wrist(22), 22, clock(22, .88), [2560, 1440])
        self.assertIsNone(tracker.select([], wrist(23), 23, clock(23, .92), [2560, 1440])[0])
        self.assertIsNone(tracker.select([weak], wrist(24), 24, clock(24, .96), [2560, 1440])[0])

    def test_stale_pose_and_invalid_time_cannot_authorize_weak_detection(self):
        weak = candidate([1101, 498, 1294, 600], .4)
        for bad_time in [clock(21, .7), {**clock(21, .84), 'quality': 'estimated'}, clock(21, 1.2)]:
            tracker = self.tracker()
            tracker.select([candidate(weak['box'], .8)], wrist(20), 20, clock(20, .8), [2560, 1440])
            self.assertIsNone(tracker.select([weak], wrist(21), 21, bad_time, [2560, 1440])[0])
        tracker = self.tracker()
        tracker.select([candidate(weak['box'], .8)], wrist(20), 20, clock(20, .8), [2560, 1440])
        self.assertIsNone(tracker.select([weak], wrist(20), 21, clock(21, .84), [2560, 1440])[0])

    def test_far_or_changed_size_candidate_is_not_recovered(self):
        for box in [[1500, 500, 1700, 600], [1100, 300, 1900, 900]]:
            tracker = self.tracker()
            tracker.select([candidate([1090, 473, 1288, 568], .8)], wrist(20), 20, clock(20, .8), [2560, 1440])
            self.assertIsNone(tracker.select([candidate(box, .4)], wrist(21, x=box[2], y=550),
                                            21, clock(21, .84), [2560, 1440])[0])


if __name__ == '__main__':
    unittest.main()
