"""Bounded HighGUI worker, preserving every submitted image by default.

Backpressure bounds memory; GUI completion is journaled apart from submission.
The optional freshness policy is only for callers explicitly allowing replacement.
"""
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time

import numpy as np


def _display_worker(incoming, ready, close_event, stop_event, errors, options):
    display = None
    journal = None
    try:
        import cv2
        from processed_video_display import ProcessedVideoDisplay
        display = ProcessedVideoDisplay(cv2, origin=options['origin'],
            fullscreen=options['fullscreen'], window_name=options['window_name'])
        if options.get('journal_path'):
            path=Path(options['journal_path']);path.parent.mkdir(parents=True, exist_ok=True)
            journal=path.open('w', encoding='utf-8')
        # Realize Cocoa before Reader starts; the first source frame is evidence.
        if display.show(np.zeros(options['frame_shape'], dtype=np.uint8)):
            stop_event.set()
        ready.set()
        while not close_event.is_set() and not stop_event.is_set():
            try:
                task = incoming.get(timeout=.04)
            except queue.Empty:
                if cv2.waitKey(1) & 0xff in (27, ord('q'), ord('Q')):
                    stop_event.set()
                continue
            if task is None:
                break
            fid, frame, submitted_ns = task
            began_ns=time.time_ns()
            if display.show(frame):stop_event.set()
            completed_ns=time.time_ns()
            if journal:
                journal.write(json.dumps({'frame_id':fid,'submitted_at_unix_ns':submitted_ns,
                    'display_started_at_unix_ns':began_ns,'display_returned_at_unix_ns':completed_ns,
                    'scope':'HighGUI return; not sensor exposure or display photons'})+'\n')
                journal.flush()
    except BaseException as exc:
        try:errors.put(str(exc),timeout=.1)
        except queue.Full:pass
    finally:
        ready.set()
        if journal:journal.close()
        if display:display.close()


class QueuedVideoDisplay:
    """One pending image with backpressure; report ESC and worker failure upstream."""
    def __init__(self, frame_shape, stop_event, *, origin=(0,0), fullscreen=False,
                 window_name='Tennis AI Analyzer', journal_path=None,
                 startup_timeout=10.0, preserve_frames=True, _worker_target=None):
        self.preserve_frames=bool(preserve_frames)
        self.shape=tuple(frame_shape)
        self.stop_event=stop_event
        self.closed=False
        self.failed=False
        self.replaced=0
        self.submitted=0
        self.dropped=0
        context=mp.get_context('spawn')
        self.incoming=context.Queue(maxsize=1)
        self.errors=context.Queue(maxsize=1)
        self.ready=context.Event()
        self.close_event=context.Event()
        options={'frame_shape':self.shape,'origin':origin,'fullscreen':fullscreen,
                 'window_name':window_name,'journal_path':journal_path}
        self.process=context.Process(target=_worker_target or _display_worker,
            args=(self.incoming,self.ready,self.close_event,self.stop_event,self.errors,options),
            name='tennis-preview', daemon=True)
        self.process.start()
        if not self.ready.wait(startup_timeout):
            self.failed=True
            self.close()
            raise RuntimeError('Preview worker startup timed out')
        try:self._check()
        except BaseException:
            self.close();raise

    def _check(self):
        try:error=self.errors.get_nowait()
        except queue.Empty:error=None
        if error:
            self.failed=True
            raise RuntimeError('Preview worker failed: '+error)
        if not self.process.is_alive() and not self.stop_event.is_set():
            self.failed=True
            raise RuntimeError('Preview worker exited: '+str(self.process.exitcode))

    def show(self, frame, frame_id=None):
        if self.closed:raise RuntimeError('Preview already closed')
        self._check()
        if self.stop_event.is_set():return True
        if frame.shape!=self.shape or frame.dtype!=np.uint8:
            raise ValueError('Preview frame shape/dtype changed')
        # Caller may reuse input/render memory immediately after this returns.
        owned=np.array(frame,copy=True,order='C')
        task=(frame_id,owned,time.time_ns())
        if self.preserve_frames:
            while True:
                self._check()
                if self.stop_event.is_set():return True
                try:self.incoming.put(task,timeout=.05);break
                except queue.Full:continue
            self.submitted+=1
            return False
        try:self.incoming.put_nowait(task)
        except queue.Full:
            try:self.incoming.get_nowait();self.replaced+=1
            except queue.Empty:pass  # Feeder still owns the mailbox; never wait here.
            try:self.incoming.put_nowait(task)
            except queue.Full:self.dropped+=1;return False
        self.submitted+=1
        return False

    def close(self):
        if self.closed:return
        self.closed=True
        drain_failed=False
        # On normal EOF the sentinel follows every submitted frame. Cancellation
        # can stop immediately. Never touch an Event owned by a terminated child.
        if self.process.is_alive():
            if self.stop_event.is_set():
                self.close_event.set()
            else:
                try:self.incoming.put(None,timeout=2)
                except queue.Full:drain_failed=True
        self.process.join(timeout=5 if not self.stop_event.is_set() else 2)
        if self.process.is_alive():drain_failed=not self.stop_event.is_set()
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=1)
        try:error=self.errors.get(timeout=.1)
        except queue.Empty:error=None
        failure=(drain_failed or error or self.process.exitcode not in (0,None))
        for channel in (self.incoming,self.errors):
            channel.cancel_join_thread();channel.close()
        if failure and not self.stop_event.is_set() and not self.failed:
            raise RuntimeError('Preview worker failed to drain all submitted frames: '+str(error or self.process.exitcode))
