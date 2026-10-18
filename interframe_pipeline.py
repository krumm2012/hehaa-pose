"""Two frames in flight; one ordered preparation owner and one consumer.

The credit is retained across yield, so preparation cannot run arbitrarily far
ahead of stateful aggregation. Source slots belong to the caller until Analyzer
returns them, including all preparation and aggregation work here.
"""
import queue
import threading


class InterframePipeline:
    def __init__(self, read_next, prepare, stop_event, close_timeout=12):
        self.read_next = read_next
        self.prepare = prepare
        self.stop_event = stop_event
        self.cancelled = threading.Event()
        self.credits = threading.BoundedSemaphore(2)
        self.ready = queue.Queue(maxsize=1)
        self.close_timeout = close_timeout
        self.thread = threading.Thread(target=self._run, name='frame-stage-a', daemon=True)
        self.thread.start()

    def _stopped(self):
        return self.cancelled.is_set() or self.stop_event.is_set()

    def _send(self, kind, value):
        while not self._stopped():
            try:
                self.ready.put((kind, value), timeout=.05)
                return
            except queue.Full:
                pass

    def _run(self):
        try:
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
        self.thread.join(timeout=self.close_timeout)
        if self.thread.is_alive():
            raise RuntimeError('Interframe preparation did not terminate')
