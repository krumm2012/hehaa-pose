import multiprocessing as mp
import queue
import unittest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from queued_video_display import QueuedVideoDisplay


def ready_worker(incoming,ready,close_event,stop_event,errors,options):
    ready.set();close_event.wait(10)


def broken_worker(incoming,ready,close_event,stop_event,errors,options):
    errors.put('fixture failure');ready.set();close_event.wait(10)


def draining_worker(incoming,ready,close_event,stop_event,errors,options):
    ready.set()
    with open(options['journal_path'], 'w') as journal:
        while True:
            task = incoming.get()
            if task is None:return
            time.sleep(.025)
            journal.write(str(task[0])+'\n');journal.flush()


def fail_last_worker(incoming,ready,close_event,stop_event,errors,options):
    ready.set();incoming.get();errors.put('last image failed')


class QueuedDisplayTests(unittest.TestCase):
    def test_show_owns_copy_and_only_replaces_preview(self):
        display=object.__new__(QueuedVideoDisplay)
        display.preserve_frames=False
        display.shape=(3,4,3);display.closed=False;display.stop_event=mp.get_context('spawn').Event()
        display.incoming=queue.Queue(maxsize=1);display.errors=queue.Queue();display.process=Mock()
        display.replaced=display.submitted=display.dropped=0
        frame=np.ones(display.shape,dtype=np.uint8)
        display.show(frame,7);frame[:]=9
        fid,owned,_=display.incoming.get_nowait()
        self.assertEqual(fid,7);self.assertTrue(np.all(owned==1))
        display.show(frame,8);display.show(frame,9)
        self.assertEqual(display.incoming.qsize(),1)
        self.assertEqual(display.incoming.get_nowait()[0],9)
        self.assertEqual(display.replaced,1)

    def test_start_stop_and_escape_do_not_leak_process(self):
        stop=mp.get_context('spawn').Event()
        for _ in range(2):
            display=QueuedVideoDisplay((3,4,3),stop,_worker_target=ready_worker)
            stop.set();self.assertTrue(display.show(np.zeros((3,4,3),dtype=np.uint8)))
            display.close();display.close();self.assertFalse(display.process.is_alive());stop.clear()

    def test_worker_failure_is_reported_not_silently_hidden(self):
        stop=mp.get_context('spawn').Event()
        display=QueuedVideoDisplay((3,4,3),stop,_worker_target=ready_worker)
        display.process.terminate();display.process.join(2)
        try:
            with self.assertRaises(RuntimeError):display.show(np.zeros((3,4,3),dtype=np.uint8))
        finally:display.close()

    def test_backpressure_and_eof_drain_preserve_all_images_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'display.txt'
            display=QueuedVideoDisplay((3,4,3),mp.get_context('spawn').Event(),
                journal_path=str(path),_worker_target=draining_worker)
            try:
                for fid in range(12):display.show(np.full((3,4,3),fid,np.uint8),fid)
            finally:display.close()
            self.assertEqual([int(x) for x in path.read_text().splitlines()],list(range(12)))
            self.assertEqual(display.replaced+display.dropped,0)
            self.assertEqual(display.incoming._maxsize,1)

    def test_failure_after_final_submission_cannot_report_complete(self):
        display=QueuedVideoDisplay((3,4,3),mp.get_context('spawn').Event(),_worker_target=fail_last_worker)
        display.show(np.zeros((3,4,3),np.uint8),0)
        with self.assertRaisesRegex(RuntimeError,'last image failed'):display.close()
        self.assertFalse(display.process.is_alive())
