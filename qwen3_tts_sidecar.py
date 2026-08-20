"""Python 3.9-safe client for the isolated Qwen3-TTS MLX worker.

The video pipeline currently runs in Python 3.9, while current MLX-Audio
Qwen3-TTS support requires Python 3.10+. This sidecar keeps the model in a
separate Python 3.11 process and only exchanges short JSON-line requests.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional


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
        self._process: Optional[subprocess.Popen] = None
        self._process_lock = threading.RLock()
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
            self._queue.put_nowait((int(event["event_id"]), text, callback))
        except queue.Full:
            callback({"status": "skipped", "engine": "qwen3-tts-mlx", "reason": "tts_backlog"})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.join()
        self._queue.put(self._sentinel)
        self._queue.join()
        self._thread.join()
        self._close_worker()

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
            })
            streamed = bool(worker_result.get("stream_playback"))
            callback({"status": "ready", "played": streamed, "playback_error": str(worker_result.get("playback_error") or "")})
            self.logger(f"🔊 [Qwen3-TTS] Swing #{event_id} | {latency_ms}ms | {target.name}")
            if self.playback and not streamed:
                played, playback_error = self._play(target)
                callback({"status": "ready", "played": played, "playback_error": playback_error})
        except BaseException as exc:
            callback({"status": "unavailable", "engine": "qwen3-tts-mlx", "model": self.model_name, "reason": str(exc)})
            self.logger(f"⚠️ [Qwen3-TTS] Swing #{event_id} | unavailable | 本地文字建议继续生效: {exc}")

    def _request_worker(self, event_id: int, text: str, target: Path) -> Dict:
        with self._process_lock:
            process = self._ensure_worker()
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("Qwen3-TTS worker has no standard input/output")
            process.stdin.write(json.dumps({"event_id": event_id, "text": text, "output_path": str(target), "stream_playback": self.playback, "streaming_interval": self.streaming_interval_seconds, "streaming_prebuffer_chunks": self.streaming_prebuffer_chunks}, ensure_ascii=False) + "\n")
            process.stdin.flush()
            response = process.stdout.readline()
            if not response:
                stderr = process.stderr.read() if process.stderr is not None else ""
                self._close_worker()
                raise RuntimeError(f"Qwen3-TTS worker exited: {stderr.strip()}")
            return json.loads(response)

    def _ensure_worker(self) -> subprocess.Popen:
        if self._process is not None and self._process.poll() is None:
            return self._process
        executable = Path(self.python_executable)
        if not executable.is_file():
            raise RuntimeError("未找到 Qwen3-TTS Python 环境；请运行 scripts/setup_qwen3_tts_mlx.sh")
        if not self.worker_path.is_file():
            raise RuntimeError("缺少 qwen3_tts_worker.py")
        self._process = subprocess.Popen(
            [str(executable), str(self.worker_path), "--model", self.model_name, "--voice", self.voice, "--language", self.language],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
        return self._process

    def _close_worker(self) -> None:
        with self._process_lock:
            process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=3)
        except (OSError, subprocess.SubprocessError):
            process.terminate()

    @staticmethod
    def _speech_text(event: Dict) -> str:
        rows = event.get("coach_advices") or ([] if not event.get("coach_advice") else [event["coach_advice"]])
        messages = [str(row.get("message") or "").strip() for row in rows[:3] if isinstance(row, dict) and str(row.get("message") or "").strip()]
        if not messages:
            return ""
        return f"第{int(event.get('event_id') or 0)}次{str(event.get('stroke_type') or '挥拍')}。" + "。".join(messages)

    @staticmethod
    def _play(path: Path) -> tuple:
        player = shutil.which("afplay")
        if not player:
            return False, "afplay_not_available"
        try:
            subprocess.run([player, str(path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            return True, ""
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)
