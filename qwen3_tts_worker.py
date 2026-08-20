#!/usr/bin/env python3
"""Long-lived MLX Qwen3-TTS worker; run only from the Python >=3.10 TTS venv."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from mlx_audio.tts.utils import load_model


def write_wav(path: Path, sample_rate: int, samples) -> None:
    pcm = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes((pcm * 32767).astype("<i2").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--language", required=True)
    args = parser.parse_args()
    # The parent uses stdout as a strict JSON-line protocol. MLX/Transformers
    # may print progress and warnings, so route all backend chatter to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        model = load_model(args.model)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            sample_rate = int(getattr(model, "sample_rate", None) or 24000)
            playback_stream = None
            playback_error = ""
            if request.get("stream_playback"):
                try:
                    playback_stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                        blocksize=1024,
                        latency="high",
                    )
                except BaseException as exc:
                    playback_error = str(exc)
                    playback_stream = None
            prebuffer_chunks = max(
                1,
                int(request.get("streaming_prebuffer_chunks") or 2),
            )
            chunks = []
            playback_queue = deque()
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    results = model.generate(
                        text=request["text"],
                        voice=args.voice,
                        language=args.language,
                        stream=True,
                        streaming_interval=max(0.08, float(request.get("streaming_interval") or 0.32)),
                    )
                    for result in results:
                        chunk = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                        if not chunk.size:
                            continue
                        chunks.append(chunk)
                        if playback_stream is not None:
                            playback_queue.append(chunk)
                            if len(playback_queue) >= prebuffer_chunks:
                                if not playback_stream.active:
                                    playback_stream.start()
                                playback_stream.write(
                                    playback_queue.popleft().reshape(-1, 1)
                                )
            finally:
                if playback_stream is not None:
                    try:
                        while playback_queue:
                            if not playback_stream.active:
                                playback_stream.start()
                            playback_stream.write(
                                playback_queue.popleft().reshape(-1, 1)
                            )
                    finally:
                        playback_stream.stop()
                        playback_stream.close()
            if not chunks:
                raise RuntimeError("Qwen3-TTS returned no audio samples")
            write_wav(Path(request["output_path"]), sample_rate, np.concatenate(chunks))
            response = {
                "ok": True,
                "sample_rate": sample_rate,
                "stream_playback": playback_stream is not None,
                "playback_error": playback_error,
            }
        except BaseException as exc:
            response = {"ok": False, "error": str(exc)}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
