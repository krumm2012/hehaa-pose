"""Bounded JSON-line requests to an isolated speech process (Python 3.9)."""

import json
import queue
import subprocess
import threading
import time
from collections import deque


class SpeechWorkerClient:
    def __init__(self, command, startup_timeout=120.0, request_timeout=30.0):
        self.command = command
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.process = None
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.readers = []

    @staticmethod
    def _stdout(stream, responses):
        try:
            for line in stream:
                responses.put(line)
        finally:
            responses.put(None)

    @staticmethod
    def _stderr(stream, tail):
        # Fixed-size reads also drain progress output without newline characters.
        while True:
            data = stream.read(1024)
            if not data:
                break
            tail.append(data)

    def request(self, payload):
        with self.lock:
            if self.closed.is_set():
                raise RuntimeError("Speech worker closed")
            first = self.process is None or self.process.poll() is not None
            if first:
                self._reap()
                self.responses = queue.Queue()
                self.tail = deque(maxlen=8)
                self.process = subprocess.Popen(
                    self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                )
                self.readers = [
                    threading.Thread(target=self._stdout, args=(self.process.stdout, self.responses), daemon=True),
                    threading.Thread(target=self._stderr, args=(self.process.stderr, self.tail), daemon=True),
                ]
                for reader in self.readers:
                    reader.start()
            try:
                self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
                deadline = time.monotonic() + (self.startup_timeout if first else self.request_timeout)
                while not self.closed.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Speech worker response timed out")
                    try:
                        response = self.responses.get(timeout=min(0.1, remaining))
                    except queue.Empty:
                        continue
                    if response is None:
                        raise RuntimeError("Speech worker exited: " + "".join(self.tail)[-2000:])
                    return json.loads(response)
                raise RuntimeError("Speech request cancelled")
            except BaseException:
                self._reap()
                raise

    def _reap(self):
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
        process.wait(timeout=1)
        for reader in self.readers:
            reader.join(timeout=1)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()

    def close(self):
        self.closed.set()
        with self.lock:
            self._reap()
