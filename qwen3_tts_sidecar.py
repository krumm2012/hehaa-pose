"""Python 3.9-safe client for the isolated Qwen3-TTS MLX worker.

The video pipeline currently runs in Python 3.9, while current MLX-Audio
Qwen3-TTS support requires Python 3.10+. This sidecar keeps the model in a
separate Python 3.11 process and only exchanges short JSON-line requests.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional
from tts_worker_client import SpeechWorkerClient


DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
DEFAULT_VOICE = "Vivian"
DEFAULT_LANGUAGE = "Chinese"


class CoachTtsSidecar:
    """Serialize local TTS synthesis and optional macOS speaker playback."""

    def __init__(
        self,
        *,
        output_dir: str,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        language: str = DEFAULT_LANGUAGE,
        playback: bool = True,
        max_pending: int = 2,
        streaming_interval_seconds: float = 0.32,
        streaming_prebuffer_chunks: int = 2,
        audio_path_prefix: str = "",
        python_executable: Optional[str] = None,
        worker_path: Optional[str] = None,
        synthesizer: Optional[Callable[[int, str, Path], Dict]] = None,
        logger: Callable[[str], None] = print,
        startup_timeout_seconds: float = 120.0,
        request_timeout_seconds: float = 30.0,
    ):
        self.output_dir = Path(output_dir)
        self.model_name = str(model)
        self.voice = str(voice)
        self.language = str(language)
        self.playback = bool(playback)
        self.audio_path_prefix = str(audio_path_prefix).strip("/")
        self.streaming_interval_seconds = max(0.08, float(streaming_interval_seconds))
        self.streaming_prebuffer_chunks = max(1, int(streaming_prebuffer_chunks))
        self.python_executable = str(
            python_executable
            or os.environ.get("TENNIS_QWEN3_TTS_PYTHON")
            or Path(__file__).with_name("venv_qwen3_tts") / "bin" / "python"
        )
        self.worker_path = Path(
            worker_path or Path(__file__).with_name("qwen3_tts_worker.py")
        )
        self.synthesizer = synthesizer
        self.logger = logger
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(max_pending)))
        self._sentinel = object()
        self._closed = False
        self._state_lock = threading.Lock()
        self._cancel = threading.Event()
        self._client = SpeechWorkerClient(
            [self.python_executable, str(self.worker_path), "--model", self.model_name,
             "--voice", self.voice, "--language", self.language],
            startup_timeout=max(0.1, float(startup_timeout_seconds)),
            request_timeout=max(0.1, float(request_timeout_seconds)),
        )
        self._thread = threading.Thread(
            target=self._run,
            name="qwen3-tts-coach",
            daemon=True,
        )
        self._thread.start()

    def pending_payload(self, event: Dict) -> Dict:
        return {
            "status": "pending",
            "engine": "qwen3-tts-mlx",
            "model": self.model_name,
            "voice": self.voice,
            "message": self._speech_text(event),
        }

    def submit(self, event: Dict, callback: Callable[[Dict], None]) -> None:
        text = self._speech_text(event)
        if not text:
            callback({"status": "skipped", "engine": "qwen3-tts-mlx", "reason": "no_local_coach_advice"})
            return
        try:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Cannot submit after speech sidecar closes")
                self._queue.put_nowait((int(event["event_id"]), text, callback))
        except queue.Full:
            callback({"status": "skipped", "engine": "qwen3-tts-mlx", "reason": "tts_backlog"})

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        # Keep shutdown shorter than the pipeline supervisor's grace period.
        self._thread.join(timeout=0.2)
        self._cancel.set()
        self._client.close()
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                item[2]({"status": "skipped", "reason": "session_closed"})
            except Exception as exc:
                self.logger(f"⚠️ [Qwen3-TTS] cancellation callback failed: {exc}")
            finally:
                self._queue.task_done()
        self._queue.put(self._sentinel)
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            raise RuntimeError("Speech sidecar did not stop within deadline")

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._sentinel:
                    return
                event_id, text, callback = item
                self._synthesize(event_id, text, callback)
            except BaseException as exc:
                self.logger(f"⚠️ [Qwen3-TTS] worker error: {exc}")
            finally:
                self._queue.task_done()

    def _synthesize(self, event_id: int, text: str, callback: Callable[[Dict], None]) -> None:
        started = time.perf_counter()
        target = self.output_dir / f"swing_{event_id:03d}_coach.wav"
        try:
            worker_result = (
                self.synthesizer(event_id, text, target)
                if self.synthesizer is not None
                else self._request_worker(event_id, text, target)
            )
            if not worker_result.get("ok") or not target.is_file():
                raise RuntimeError(str(worker_result.get("error") or "Qwen3-TTS did not create WAV"))
            latency_ms = round((time.perf_counter() - started) * 1000)
            callback({
                "status": "ready", "engine": "qwen3-tts-mlx", "model": self.model_name,
                "voice": self.voice, "language": self.language,
                "audio_path": (f"{self.audio_path_prefix}/{target.name}" if self.audio_path_prefix else target.name),
                "sample_rate": int(worker_result.get("sample_rate") or 24000),
                "latency_ms": latency_ms, "played": False,
                "first_audio_ms": worker_result.get("first_audio_ms"),
                "audio_underflows": worker_result.get("audio_underflows", 0),
            })
            streamed = bool(worker_result.get("stream_playback"))
            callback({"status": "ready", "played": streamed and not worker_result.get("playback_error"), "playback_error": str(worker_result.get("playback_error") or "")})
            self.logger(f"🔊 [Qwen3-TTS] Swing #{event_id} | {latency_ms}ms | {target.name}")
            if self.playback and not streamed and not self._cancel.is_set():
                played, playback_error = self._play(target)
                callback({"status": "ready", "played": played, "playback_error": playback_error})
        except BaseException as exc:
            callback({"status": "unavailable", "engine": "qwen3-tts-mlx", "model": self.model_name, "reason": str(exc)})
            self.logger(f"⚠️ [Qwen3-TTS] Swing #{event_id} | unavailable | 本地文字建议继续生效: {exc}")

    def _request_worker(self, event_id: int, text: str, target: Path) -> Dict:
        return self._client.request({
            "event_id": event_id, "text": text, "output_path": str(target),
            "stream_playback": self.playback,
            "streaming_interval": self.streaming_interval_seconds,
            "streaming_prebuffer_chunks": self.streaming_prebuffer_chunks,
        })

    @staticmethod
    def _speech_text(event: Dict) -> str:
        rows = event.get("coach_advices") or ([] if not event.get("coach_advice") else [event["coach_advice"]])
        messages = [str(row.get("message") or "").strip() for row in rows[:3] if isinstance(row, dict) and str(row.get("message") or "").strip()]
        if not messages:
            return ""
        return f"第{int(event.get('event_id') or 0)}次{str(event.get('stroke_type') or '挥拍')}。" + "。".join(messages)

    def _play(self, path: Path) -> tuple:
        player = shutil.which("afplay")
        if not player:
            return False, "afplay_not_available"
        try:
            process = subprocess.Popen([player, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 30
            while process.poll() is None:
                if self._cancel.wait(0.05) or time.monotonic() >= deadline:
                    process.kill()
                    process.wait()
                    return False, "playback_cancelled"
            return process.returncode == 0, "" if process.returncode == 0 else "playback_failed"
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)
