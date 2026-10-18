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


class ReaderStarvationRegressionTests(unittest.TestCase):
    def test_continuous_reader_waits_for_slots_without_grabbing_or_replacing(self):
        import queue
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from main_pipe import MultiprocessPipeline
        pipeline=MultiprocessPipeline.__new__(MultiprocessPipeline)
        pipeline.stop_event=Mock();pipeline.stop_event.is_set.return_value=False;pipeline.stop_event.wait.return_value=False
        pipeline.inf_ready=Mock();pipeline.inf_ready.wait.return_value=True
        pipeline.analyzer_ready=Mock();pipeline.analyzer_ready.wait.return_value=True
        pipeline.rd_done=Mock();pipeline.shm_names=['fixture'];pipeline.shm_num=3;pipeline.width=2;pipeline.height=2
        pipeline.config={'pipeline_perf':{'live_limit_reader_fps':False}}
        pipeline.live_mode=True;pipeline.lossless_realtime=True;pipeline.drop_stale_frames=False
        pipeline.fps=25.;pipeline.max_frames=0;pipeline.is_stream_source=False
        pipeline.q_inference=queue.Queue(maxsize=3)
        pipeline.q_free=Mock();pipeline.q_free.get.side_effect=[queue.Empty,0,0,0,0]
        capture=Mock();capture.read.side_effect=[(True,np.zeros((2,2,3),np.uint8))]*3+[(False,None)]
        capture.get.return_value=0;pipeline._open_capture=Mock(return_value=capture)
        shm=SimpleNamespace(buf=bytearray(12),close=Mock())
        with patch('main_pipe.shared_memory.SharedMemory',return_value=shm):pipeline.reader_process()
        self.assertEqual([pipeline.q_inference.get_nowait()['idx'] for _ in range(3)],[0,1,2])
        capture.grab.assert_not_called();pipeline.analyzer_ready.wait.assert_called_once()

    def test_discarded_local_frames_are_paced_too(self):
        """A 3-slot stall must not fast-forward a paced local video via grab()."""
        import contextlib
        import queue
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from main_pipe import MultiprocessPipeline
        now=[0.];waits=[];grabs=[0]
        class Stop:
            def is_set(self):return grabs[0]>=3
            def wait(self,seconds):waits.append(seconds);now[0]+=seconds;return self.is_set()
        def grab():grabs[0]+=1;now[0]+=.001;return True
        pipeline=MultiprocessPipeline.__new__(MultiprocessPipeline)
        pipeline.stop_event=Stop();pipeline.inf_ready=Mock();pipeline.inf_ready.wait.return_value=True
        pipeline.rd_done=Mock();pipeline.shm_names=[];pipeline.shm_num=3;pipeline.width=2560;pipeline.height=1440
        pipeline.config={'pipeline_perf':{'live_limit_reader_fps':True}}
        pipeline.live_mode=True;pipeline.drop_stale_frames=True;pipeline.live_slot_wait=.002
        pipeline.fps=25.;pipeline.max_frames=0;pipeline.is_stream_source=False
        pipeline.q_inference=Mock();pipeline.q_inference.get_nowait.side_effect=queue.Empty
        pipeline.q_free=Mock();pipeline.q_free.get.side_effect=queue.Empty
        pipeline.dropped_stale_frames=SimpleNamespace(value=0,get_lock=lambda:contextlib.nullcontext())
        capture=Mock();capture.grab.side_effect=grab;pipeline._open_capture=Mock(return_value=capture)
        factory=lambda fps,**kwargs:DeadlinePacer(fps,clock=lambda:now[0],**kwargs)
        with patch('main_pipe.DeadlinePacer',side_effect=factory):pipeline.reader_process()
        self.assertEqual(pipeline.dropped_stale_frames.value,3)
        self.assertEqual(len(waits),3)
        self.assertAlmostEqual(now[0],.12)
        pipeline.rd_done.set.assert_called_once()


if __name__ == "__main__":
    unittest.main()
