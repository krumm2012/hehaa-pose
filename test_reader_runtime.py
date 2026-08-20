import unittest

from reader_runtime import DeadlinePacer, SourceFrameClock


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


if __name__ == "__main__":
    unittest.main()
