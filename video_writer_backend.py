"""Video writer backends with a macOS VideoToolbox fast path."""

from functools import lru_cache
import os
import platform
import shutil
import subprocess

import cv2
import numpy as np


class OpenCVVideoWriter:
    backend_name = "opencv"

    def __init__(self, output_path, width, height, fps):
        self._writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"avc1"),
            float(fps),
            (int(width), int(height)),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"OpenCV无法打开视频编码器: {output_path}")
        self._released = False

    def write(self, frame):
        self._writer.write(frame)

    def release(self):
        if self._released:
            return
        self._writer.release()
        self._released = True


class VideoToolboxVideoWriter:
    backend_name = "videotoolbox"

    def __init__(self, output_path, width, height, fps, bitrate="12M"):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("找不到ffmpeg，无法使用VideoToolbox")
        self.width = int(width)
        self.height = int(height)
        self._released = False
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(float(fps)),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            str(bitrate),
            "-realtime",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "avc1",
            "-movflags",
            "+faststart",
            output_path,
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

    def write(self, frame):
        if self._released:
            raise RuntimeError("VideoToolbox编码器已经关闭")
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"编码帧尺寸不匹配: {frame.shape[1]}x{frame.shape[0]} "
                f"!= {self.width}x{self.height}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"编码帧类型必须是uint8，当前为{frame.dtype}")
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        try:
            self._process.stdin.write(memoryview(frame).cast("B"))
        except BrokenPipeError as exc:
            error = self._read_error()
            raise RuntimeError(f"VideoToolbox编码失败: {error}") from exc

    def _read_error(self):
        if self._process.stderr is None:
            return "unknown ffmpeg error"
        return self._process.stderr.read().decode("utf-8", errors="replace").strip()

    def release(self):
        if self._released:
            return
        self._released = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait(timeout=15)
        error = self._read_error()
        if self._process.stderr is not None:
            self._process.stderr.close()
        if return_code != 0:
            raise RuntimeError(
                f"VideoToolbox编码器退出码 {return_code}: {error or 'unknown error'}"
            )


@lru_cache(maxsize=1)
def videotoolbox_available():
    ffmpeg = shutil.which("ffmpeg")
    if platform.system() != "Darwin" or not ffmpeg:
        return False
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:r=1",
        "-frames:v",
        "1",
        "-an",
        "-c:v",
        "h264_videotoolbox",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def create_video_writer(
    output_path,
    width,
    height,
    fps,
    backend="auto",
    bitrate="12M",
):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    normalized_backend = str(backend or "auto").lower()
    if normalized_backend not in {"auto", "opencv", "videotoolbox"}:
        raise ValueError(f"未知编码后端: {backend}")

    if normalized_backend == "videotoolbox":
        if not videotoolbox_available():
            raise RuntimeError("当前环境不可用VideoToolbox")
        return VideoToolboxVideoWriter(
            output_path,
            width,
            height,
            fps,
            bitrate=bitrate,
        )

    if normalized_backend == "auto" and videotoolbox_available():
        try:
            return VideoToolboxVideoWriter(
                output_path,
                width,
                height,
                fps,
                bitrate=bitrate,
            )
        except (OSError, RuntimeError):
            pass

    return OpenCVVideoWriter(output_path, width, height, fps)
