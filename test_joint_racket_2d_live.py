import unittest
from joint_racket_2d_live import LiveJointRacket2D
from test_joint_racket_2d_analysis import row


class LiveTrailsTests(unittest.TestCase):
    def test_three_frame_gaps_clear_on_recovery_without_bridging(self):
        live = LiveJointRacket2D()
        for fid in range(6):
            r = row(fid)
            if 1 <= fid <= 3: r['racket_pose']['candidates'] = []
            result = live.update(r, (1000, 600))
            if fid == 2: self.assertEqual(result['active_gaps_ge_3'], [])
            if fid == 3:
                gap = next(g for g in result['active_gaps_ge_3'] if g['track'] == 'front.racket.sweet_spot')
                self.assertEqual((gap['start_frame'], gap['end_frame'], gap['count']), (1, 3, 3))
            if fid == 4: self.assertEqual(result['active_gaps_ge_3'], [])
        segments = [s for s in live.segments() if s[0] == 'front.racket.sweet_spot']
        self.assertEqual(len(segments), 1)

    def test_source_gap_resets_bounded_history_and_gap_counters(self):
        live = LiveJointRacket2D(history_frames=4)
        for fid in range(8): live.update(row(fid), (1000, 600))
        self.assertEqual(len(live.history), 4)
        result = live.update(row(10), (1000, 600))
        self.assertTrue(result['source_discontinuity'])
        self.assertEqual(len(live.history), 1)
        self.assertEqual(live.segments(), [])
