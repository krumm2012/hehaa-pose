import threading
import unittest

from realtime_swing_runtime import RealtimeSwingRuntime


class FakeEngine:
    def __init__(self, fail=False):
        self.frames = []
        self.fail = fail

    def push_frame(self, record):
        if self.fail:
            raise ValueError("bad frame")
        self.frames.append(record)
        return []

    def flush(self):
        return []

    def snapshot(self):
        return {"summary": {"swing_event_count": 0}, "events": [], "frame_trace": []}


class FakeOutput:
    def __init__(self):
        self.frames = []
        self.closed = False

    def record_frame(self, frame_id, frame):
        self.frames.append((frame_id, frame))

    def publish_events(self, events, snapshot):
        pass

    def update_event(self, event_id, patch):
        return True

    def close(self):
        self.closed = True


class RealtimeSwingRuntimeTests(unittest.TestCase):
    def test_processes_frame_records_behind_small_runtime_interface(self):
        engine = FakeEngine()
        output = FakeOutput()
        runtime = RealtimeSwingRuntime(engine=engine, output=output)

        runtime.submit_frame({"frame_id": 1})
        runtime.close()

        self.assertEqual(engine.frames, [{"frame_id": 1}])
        self.assertTrue(output.closed)

    def test_worker_failure_sets_pipeline_stop_and_surfaces_on_close(self):
        stop_event = threading.Event()
        runtime = RealtimeSwingRuntime(
            engine=FakeEngine(fail=True),
            output=FakeOutput(),
            stop_event=stop_event,
            logger=lambda message: None,
        )

        runtime.submit_frame({"frame_id": 1})
        with self.assertRaisesRegex(RuntimeError, "data runtime failed"):
            runtime.close()

        self.assertTrue(stop_event.is_set())


class OutputSnapshotRegressionTests(unittest.TestCase):
    def test_pending_writer_snapshot_survives_later_clip_and_coach_updates(self):
        import threading
        from concurrent.futures import Future
        from unittest.mock import Mock
        from latest_snapshot_queue import LatestSnapshotQueue
        from realtime_swing_pipeline import RealtimeSwingOutputManager
        manager=RealtimeSwingOutputManager.__new__(RealtimeSwingOutputManager)
        manager._output_lock=threading.Lock();manager._output_queue=LatestSnapshotQueue()
        manager.check_health=Mock();manager._event_journal=Mock();manager._dropped_compression_frames=0
        manager._events=[{'event_id':1,'coach':{'status':'pending'},'clip_status':'pending'}]
        manager._document={'events':list(manager._events),'frame_trace':[{'frame':7}], 'summary':{'total_frames':10}}
        manager._queue_live_outputs(manager._document)
        prior=manager._output_queue.get_nowait();manager._output_queue.task_done()
        manager.update_event(1,{'coach':{'status':'ready'}})
        done=Future();done.set_result({'status':'ready','frame_count':20})
        manager._finish_clip(1,done)
        manager._document['summary']['total_frames']=20
        manager._document['frame_trace'].append({'frame':8})
        self.assertEqual(prior['events'][0]['coach']['status'],'pending')
        self.assertEqual(prior['events'][0]['clip_status'],'pending')
        self.assertEqual(prior['summary']['total_frames'],10)
        self.assertEqual(len(prior['frame_trace']),1)
        self.assertEqual(manager._events[0]['coach']['status'],'ready')
        self.assertEqual(manager._events[0]['clip_status'],'ready')

    def test_publishing_does_not_copy_the_engine_event_history_again(self):
        import time
        from unittest.mock import Mock
        class Engine:
            def snapshot(self):
                time.sleep(.25)  # Cost grows with the retained full event history.
                return {'events': []}
            def snapshot_for_output(self):return {'events': [], 'summary': {}}
        runtime=RealtimeSwingRuntime.__new__(RealtimeSwingRuntime)
        runtime.engine=Engine();runtime.output=Mock()
        runtime.deepseek_sidecar=None;runtime.coach_tts_sidecar=None
        started=time.monotonic();runtime._publish([],final=True)
        self.assertLess(time.monotonic()-started,.1)
        runtime.output.publish_events.assert_called_once()


if __name__ == "__main__":
    unittest.main()
