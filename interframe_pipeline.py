"""Two frames in flight; one ordered preparation owner and one consumer.

The credit is retained across yield, so preparation cannot run arbitrarily far
ahead of stateful aggregation. Source slots belong to the caller until Analyzer
returns them, including all preparation and aggregation work here.
"""
import queue
import threading


class InterframePipeline:
    def __init__(self, read_next, prepare, stop_event, close_timeout=12, max_in_flight=2, preprocess=None):
        self.read_next = read_next
        self.prepare = prepare
        self.preprocess = preprocess
        self.stop_event = stop_event
        self.cancelled = threading.Event()
        self.credits = threading.BoundedSemaphore(max(2, int(max_in_flight)))
        self.ready = queue.Queue(maxsize=1)
        self.pre_queue = queue.Queue(maxsize=1) if preprocess is not None else None
        self.close_timeout = close_timeout
        self.pre_thread = None
        if self.preprocess is not None:
            self.pre_thread = threading.Thread(target=self._run_preprocess, name='frame-stage-pre', daemon=True)
            self.pre_thread.start()
        self.thread = threading.Thread(target=self._run, name='frame-stage-a', daemon=True)
        self.thread.start()

    def _stopped(self):
        return self.cancelled.is_set() or self.stop_event.is_set()

    def _put(self, target_queue, kind, value):
        while not self._stopped():
            try:
                target_queue.put((kind, value), timeout=.05)
                return
            except queue.Full:
                pass

    def _send(self, kind, value):
        self._put(self.ready, kind, value)

    def _run_preprocess(self):
        try:
            while not self._stopped():
                if not self.credits.acquire(timeout=.05):
                    continue
                task = self.read_next(self.cancelled)
                if task is None:
                    self._put(self.pre_queue, 'end', None)
                    return
                self._put(self.pre_queue, 'frame', self.preprocess(task))
        except BaseException as exc:
            self._put(self.pre_queue, 'error', exc)

    def _run(self):
        try:
            if self.pre_queue is not None:
                while not self._stopped():
                    try:
                        kind, value = self.pre_queue.get(timeout=.05)
                    except queue.Empty:
                        continue
                    if kind == 'end':
                        self._send('end', None)
                        return
                    if kind == 'error':
                        self._send('error', value)
                        return
                    self._send('frame', self.prepare(value))
            else:
                while not self._stopped():
                    if not self.credits.acquire(timeout=.05):
                        continue
                    task = self.read_next(self.cancelled)
                    if task is None:
                        self._send('end', None)
                        return
                    self._send('frame', self.prepare(task))
        except BaseException as exc:
            self._send('error', exc)

    def __iter__(self):
        while not self._stopped():
            try:
                kind, value = self.ready.get(timeout=.05)
            except queue.Empty:
                continue
            if kind == 'end':
                return
            if kind == 'error':
                raise RuntimeError('Interframe preparation failed') from value
            try:
                yield value
            finally:
                self.credits.release()

    def close(self):
        self.cancelled.set()
        if self.pre_thread is not None:
            self.pre_thread.join(timeout=self.close_timeout)
            if self.pre_thread.is_alive():
                raise RuntimeError('Interframe preprocess did not terminate')
        self.thread.join(timeout=self.close_timeout)
        if self.thread.is_alive():
            raise RuntimeError('Interframe preparation did not terminate')
