import unittest

from reader_runtime import DeadlinePacer, SourceFrameClock, SourceMediaClock


class DeadlinePacerTests(unittest.TestCase):
    def test_compensates_for_previous_sleep_overshoot(self):
        pacer = DeadlinePacer(fps=25.0, start_time=0.0)

        self.assertAlmostEqual(pacer.next_delay(now=0.005), 0.035)
        self.assertAlmostEqual(pacer.next_delay(now=0.043), 0.037)
        self.assertAlmostEqual(pacer.next_delay(now=0.082), 0.038)

    def test_disabled_pacer_never_waits(self):
        pacer = DeadlinePacer(fps=0.0, start_time=0.0)

        self.assertEqual(pacer.next_delay(now=10.0), 0.0)

    def test_rebases_after_large_source_stall(self):
        pacer = DeadlinePacer(
            fps=25.0,
            start_time=0.0,
            max_lag_intervals=1.0,
        )

        self.assertEqual(pacer.next_delay(now=0.250), 0.0)
        self.assertAlmostEqual(pacer.next_delay(now=0.255), 0.035)


class SourceFrameClockTests(unittest.TestCase):
    def test_grabbed_frames_advance_source_timeline(self):
        clock = SourceFrameClock()

        self.assertEqual(clock.accepted(), 0)
        self.assertEqual(clock.dropped(), 1)
        self.assertEqual(clock.dropped(), 2)
        self.assertEqual(clock.accepted(), 3)
        self.assertEqual(clock.processed_count, 2)
        self.assertEqual(clock.source_count, 4)


class SourceMediaClockTests(unittest.TestCase):
    def test_vfr_pts_and_dropped_frame_ids_do_not_use_nominal_rate(self):
        clock = SourceMediaClock(25)
        rows = [clock.observe(i, ms) for i, ms in [(0, 0), (1, 33), (4, 171)]]
        self.assertEqual([r['timestamp_seconds'] for r in rows], [0, .033, .171])
        self.assertEqual(rows[-1]['source_frame_id'], 4)
        self.assertTrue(all(r['quality'] == 'reported' for r in rows))
        self.assertFalse(rows[-1]['exposure_time_verified'])

    def test_duplicate_and_backward_pts_are_preserved_and_flagged(self):
        clock = SourceMediaClock(25)
        clock.observe(0, 0)
        self.assertEqual(clock.observe(1, 0)['quality'], 'duplicate')
        clock.observe(2, 80)
        row = clock.observe(3, 40)
        self.assertEqual(row['quality'], 'discontinuous')
        self.assertEqual(row['timestamp_seconds'], .04)

    def test_unavailable_pts_is_explicit_fps_estimate(self):
        row = SourceMediaClock(25).observe(5, float('nan'))
        self.assertEqual(row['timestamp_seconds'], .2)
        self.assertEqual(row['basis'], 'nominal_fps')
        self.assertEqual(row['quality'], 'estimated')

    def test_stream_never_promotes_reader_or_backend_time_to_exposure(self):
        row = SourceMediaClock(25, 'stream').observe(5, 200)
        self.assertIsNone(row['timestamp_seconds'])
        self.assertEqual(row['basis'], 'unavailable')

    def test_missing_rate_and_pts_stay_unavailable(self):
        row = SourceMediaClock(0).observe(5, None)
        self.assertIsNone(row['timestamp_seconds'])


if __name__ == "__main__":
    unittest.main()
