import sys
import threading
import time
import unittest
from unittest.mock import Mock

import numpy as np

from latest_snapshot_queue import LatestSnapshotQueue
from tts_worker_client import SpeechWorkerClient
from streaming_audio_player import StreamingAudioPlayer
from deepseek_realtime_coach import DeepSeekCoachSidecar
from realtime_swing_pipeline import RealtimeSwingOutputManager


class RuntimeResilienceTests(unittest.TestCase):
    def test_noisy_worker_does_not_block_response(self):
        client = SpeechWorkerClient([sys.executable, "-u", "-c",
            "import sys; sys.stdin.readline(); sys.stderr.write('x'*200000); sys.stderr.flush(); print('{\"ok\":true}')"], startup_timeout=3)
        try:
            self.assertTrue(client.request({})["ok"])
        finally:
            client.close()
        self.assertIsNone(client.process)

    def test_timeout_reaps_worker(self):
        client = SpeechWorkerClient([sys.executable, "-c", "import time; time.sleep(30)"], startup_timeout=0.1)
        try:
            with self.assertRaises(TimeoutError):
                client.request({})
            self.assertIsNone(client.process)
        finally:
            client.close()

    def test_close_cancels_active_request(self):
        client = SpeechWorkerClient([sys.executable, "-c", "import time; time.sleep(30)"])
        errors = []
        def request():
            try:
                client.request({})
            except RuntimeError as exc:
                errors.append(exc)
        thread = threading.Thread(target=request)
        thread.start()
        started = time.monotonic()
        client.close()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertLess(time.monotonic() - started, 2)
        self.assertTrue(errors)

    def test_snapshots_coalesce_and_join_completes(self):
        snapshots = LatestSnapshotQueue()
        for index in range(1000):
            snapshots.publish(index)
        self.assertEqual(snapshots.qsize(), 1)
        self.assertEqual(snapshots.get(), 999)
        snapshots.task_done()
        self.assertEqual(snapshots.unfinished_tasks, 0)

    def test_writer_error_surfaces_before_close(self):
        output = RealtimeSwingOutputManager.__new__(RealtimeSwingOutputManager)
        output._output_queue = LatestSnapshotQueue()
        output._output_sentinel = object()
        output._output_worker_error = None
        output._write_live_outputs = Mock(side_effect=OSError("disk full"))
        thread = threading.Thread(target=output._write_outputs_loop)
        thread.start()
        output._queue_live_outputs({"events": []})
        output._output_queue.join()
        try:
            with self.assertRaisesRegex(RuntimeError, "writer failed"):
                output.check_health()
        finally:
            output._output_queue.put(output._output_sentinel)
            thread.join(2)

    def test_player_consumes_buffer_while_producer_pauses(self):
        consumed = threading.Event()
        values = []
        class Stream:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def write(self, chunk):
                values.append(float(chunk[0, 0]))
                if len(values) == 2: consumed.set()
                return False
        player = StreamingAudioPlayer(Stream, prebuffer_chunks=2)
        try:
            player.push(np.array([1.0]))
            player.push(np.array([2.0]))
            self.assertTrue(consumed.wait(2))
        finally:
            player.finish()
        self.assertEqual(values, [1.0, 2.0])
        self.assertEqual(player.error, "")

    def test_deepseek_backlog_bounded_and_completed_futures_removed(self):
        release = threading.Event()
        started = threading.Event()
        sidecar = DeepSeekCoachSidecar("test", workers=1, max_pending=0)
        def generate(*args):
            started.set()
            release.wait(2)
            return {"status": "ready"}
        sidecar._generate = generate
        try:
            first = sidecar.submit({}, lambda result: None)
            self.assertTrue(started.wait(1))
            second = sidecar.submit({}, lambda result: None)
            self.assertEqual(second.result()["reason"], "deepseek_backlog")
            release.set()
            first.result(timeout=2)
        finally:
            release.set()
            sidecar.close()
        self.assertEqual(len(sidecar._futures), 0)


if __name__ == "__main__":
    unittest.main()
