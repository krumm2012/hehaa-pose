"""Consume generated PCM independently of model inference."""

import threading
import time
from collections import deque


class StreamingAudioPlayer:
    def __init__(self, stream_factory, prebuffer_chunks=2, max_chunks=64):
        self.factory = stream_factory
        self.prebuffer = max(1, prebuffer_chunks)
        self.capacity = max(max_chunks, self.prebuffer)
        self.chunks = deque()
        self.condition = threading.Condition()
        self.finished = False
        self.error = ""
        self.played = False
        self.underflows = 0
        self.first_audio_ms = None
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def push(self, chunk):
        with self.condition:
            self.condition.wait_for(lambda: len(self.chunks) < self.capacity or self.error)
            if not self.error:
                self.chunks.append(chunk)
                self.condition.notify_all()

    def finish(self):
        with self.condition:
            self.finished = True
            self.condition.notify_all()
        self.thread.join()

    def _run(self):
        try:
            with self.condition:
                self.condition.wait_for(lambda: len(self.chunks) >= self.prebuffer or self.finished)
                if not self.chunks:
                    return
            with self.factory() as stream:
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda: self.chunks or self.finished)
                        if not self.chunks:
                            break
                        chunk = self.chunks.popleft()
                        self.condition.notify_all()
                    if self.first_audio_ms is None:
                        self.first_audio_ms = round((time.monotonic() - self.started) * 1000)
                    self.underflows += int(bool(stream.write(chunk.reshape(-1, 1))))
                    self.played = True
        except Exception as exc:
            with self.condition:
                self.error = str(exc) or type(exc).__name__
                self.condition.notify_all()
