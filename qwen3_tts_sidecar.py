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

DEFAULT_CONCISE_CUE_MAP = {
    # 动作稳定
    "动作稳定继续保持": "继续保持",
    # 核心动力链与挥拍肢体
    "用身体核心带动球拍发力": "核心发力",
    "挥拍时手臂再舒展": "手臂舒展",
    "准备时适当降低重心": "降低重心",
    "加大肩髋分离": "肩髋分离",
    "击球后稳住重心": "稳住重心",
    "击球瞬间双腿蹬地发力": "双腿蹬地",
    "击球前拍头下潜刷球": "拍头下潜",
    "击球时带动重心": "带动重心",
    "提前准备充分引拍": "提前引拍",
    "充分展开后背引拍": "充分引拍",
    "转肩蓄力拉开后背": "转肩蓄力",
    "提前转肩充分引拍": "提前引拍",
    "击球后完成随挥": "随挥完整",
    # 视野与拍摄引导
    "减少球拍遮挡": "减少球拍遮挡",
    "保持全身清晰入镜": "全身入镜",
    "确保来球完整入镜": "来球入镜",
    # 复核类
    "网球识别需复核": "识别复核",
    "来球轨迹需复核": "轨迹复核",
    "触球位置需复核": "触球复核",
    "本次动作建议复核": "动作复核",
}


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
        preempt: bool = True,
        single_core_advice: bool = True,
        concise_speech: bool = False,
        warmup: bool = False,
        min_interval_seconds: float = 0.5,
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
        self.preempt = bool(preempt)
        self.single_core_advice = bool(single_core_advice)
        self.concise_speech = bool(concise_speech)
        self.warmup = bool(warmup)
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
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
        self._play_lock = threading.Lock()
        self._active_play_process: Optional[subprocess.Popen] = None
        self._last_speech_time = 0.0
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
        if self.warmup and self.synthesizer is None:
            self._start_warmup()

    def warmup_worker(self) -> Dict:
        if self.synthesizer is not None:
            return {"ok": True, "type": "warmup", "mock": True}
        return self._client.warmup()

    def _start_warmup(self) -> None:
        def _warmup_task():
            try:
                self.logger(f"⏳ [Qwen3-TTS] 正在后台预热模型 ({self.model_name})...")
                res = self.warmup_worker()
                if res.get("ok"):
                    self.logger("⚡ [Qwen3-TTS] 模型预热就绪，首帧已常驻统一内存！")
                else:
                    self.logger(f"⚠️ [Qwen3-TTS] 模型预热返回: {res.get('error')}")
            except Exception as exc:
                if not self._closed and not self._cancel.is_set():
                    self.logger(f"⚠️ [Qwen3-TTS] 后台预热异常 (首个击球将重新尝试): {exc}")

        warmup_thread = threading.Thread(
            target=_warmup_task,
            name="qwen3-tts-warmup",
            daemon=True,
        )
        warmup_thread.start()

    def pending_payload(self, event: Dict) -> Dict:
        return {
            "status": "pending",
            "engine": "qwen3-tts-mlx",
            "model": self.model_name,
            "voice": self.voice,
            "message": self._speech_text(
                event,
                single_core_advice=self.single_core_advice,
                concise_speech=self.concise_speech,
            ),
        }

    def submit(self, event: Dict, callback: Callable[[Dict], None]) -> None:
        text = self._speech_text(
            event,
            single_core_advice=self.single_core_advice,
            concise_speech=self.concise_speech,
        )
        if not text:
            callback({"status": "skipped", "engine": "qwen3-tts-mlx", "reason": "no_local_coach_advice"})
            return
        try:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Cannot submit after speech sidecar closes")
                if self.preempt:
                    # 1. Drain pending stale items from queue
                    while True:
                        try:
                            old_item = self._queue.get_nowait()
                            if old_item is self._sentinel:
                                self._queue.put(self._sentinel)
                                break
                            self._queue.task_done()
                            old_eid, _, old_cb = old_item
                            try:
                                old_cb({
                                    "status": "preempted",
                                    "engine": "qwen3-tts-mlx",
                                    "reason": "superseded_by_newer_swing",
                                    "event_id": old_eid,
                                })
                            except Exception:
                                pass
                        except queue.Empty:
                            break
                    # 2. Terminate active audio playback if running
                    with self._play_lock:
                        if self._active_play_process and self._active_play_process.poll() is None:
                            try:
                                self._active_play_process.kill()
                            except Exception:
                                pass
                            self._active_play_process = None

                self._queue.put_nowait((int(event["event_id"]), text, callback))
        except queue.Full:
            callback({"status": "skipped", "engine": "qwen3-tts-mlx", "reason": "tts_backlog"})

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        # Gracefully wait up to 2.5s for in-flight task to finish
        deadline = time.monotonic() + 2.5
        while self._queue.unfinished_tasks > 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        self._cancel.set()
        with self._play_lock:
            if self._active_play_process and self._active_play_process.poll() is None:
                try:
                    self._active_play_process.kill()
                except Exception:
                    pass
                self._active_play_process = None
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
        self._thread.join(timeout=2.0)
        self._client.close()
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
        now = time.monotonic()
        if self._last_speech_time > 0 and self.min_interval_seconds > 0:
            elapsed = now - self._last_speech_time
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)
        self._last_speech_time = time.monotonic()

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
                "message": text,
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

    @classmethod
    def concise_cue(cls, message: str) -> str:
        msg = str(message or "").strip()
        return DEFAULT_CONCISE_CUE_MAP.get(msg, msg)

    @classmethod
    def _speech_text(
        cls,
        event: Dict,
        single_core_advice: bool = True,
        concise_speech: bool = False,
    ) -> str:
        rows = event.get("coach_advices") or ([] if not event.get("coach_advice") else [event["coach_advice"]])
        valid_rows = [row for row in rows if isinstance(row, dict) and str(row.get("message") or "").strip()]
        if not valid_rows:
            return ""
        eid = int(event.get('event_id') or 0)
        stroke = str(event.get('stroke_type') or '挥拍')
        if single_core_advice:
            sorted_rows = sorted(valid_rows, key=lambda r: int(r.get("priority", 1)))
            top_msg = str(sorted_rows[0].get("message") or "").strip()
            if concise_speech:
                return cls.concise_cue(top_msg)
            return f"第{eid}次{stroke}：{top_msg}"
        messages = [
            (cls.concise_cue(str(row.get("message") or "").strip()) if concise_speech else str(row.get("message") or "").strip())
            for row in valid_rows[:3]
        ]
        if concise_speech:
            return "，".join(messages)
        return f"第{eid}次{stroke}。" + "。".join(messages)

    def _play(self, path: Path) -> tuple:
        player = shutil.which("afplay")
        if not player:
            return False, "afplay_not_available"
        try:
            process = subprocess.Popen([player, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with self._play_lock:
                self._active_play_process = process
            deadline = time.monotonic() + 30
            while process.poll() is None:
                if self._cancel.wait(0.05) or time.monotonic() >= deadline:
                    process.kill()
                    process.wait()
                    with self._play_lock:
                        if self._active_play_process is process:
                            self._active_play_process = None
                    return False, "playback_cancelled"
            with self._play_lock:
                if self._active_play_process is process:
                    self._active_play_process = None
            return process.returncode == 0, "" if process.returncode == 0 else "playback_failed"
        except (OSError, subprocess.SubprocessError) as exc:
            with self._play_lock:
                self._active_play_process = None
            return False, str(exc)
