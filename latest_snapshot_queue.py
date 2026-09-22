"""A single pending snapshot; consumers still finish any in-progress write."""

import queue
import threading


class LatestSnapshotQueue(queue.Queue):
    def __init__(self):
        super().__init__(maxsize=1)
        self._publish_lock = threading.Lock()

    def publish(self, snapshot):
        with self._publish_lock:
            try:
                self.put_nowait(snapshot)
            except queue.Full:
                try:
                    self.get_nowait()
                except queue.Empty:
                    pass
                else:
                    self.task_done()
                self.put_nowait(snapshot)
