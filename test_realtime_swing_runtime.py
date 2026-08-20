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


if __name__ == "__main__":
    unittest.main()
