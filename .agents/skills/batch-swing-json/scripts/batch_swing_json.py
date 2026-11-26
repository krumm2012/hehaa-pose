#!/usr/bin/env python3
"""Batch-export `<stem>_swing_events.json` into each source video's directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PYTHON = REPO_ROOT / "venv_yolo26" / "bin" / "python"
DEFAULT_CONFIG = "configs/yolo26_tennis_config.yaml"
DEFAULT_POSE_MODEL = "yolo26n-pose.mlpackage"
DEFAULT_RACKET_POSE_MODE = "legacy"


@dataclass
class VideoProcessResult:
    video: Path
    target_json: Path
    status: str  # "ok" | "skipped" | "dry_run" | "failed"
    events_count: Optional[int] = None
    elapsed_s: float = 0.0
    error: Optional[str] = None


def _has_hidden_component(rel_path: Path) -> bool:
    return any(part.startswith(".") for part in rel_path.parts)


def discover_videos(target: Path, recursive: bool = True) -> List[Path]:
    """Return sorted video paths under `target` (file or directory), ignoring hidden/._* files."""
    resolved = target.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Target path does not exist: {target}")
    if resolved.is_file():
        if resolved.name.startswith(".") or resolved.suffix.lower() not in VIDEO_EXTENSIONS:
            return []
        return [Path(os.path.abspath(resolved))]

    iterator = resolved.rglob("*") if recursive else resolved.iterdir()
    videos: List[Path] = []
    for candidate in iterator:
        if not candidate.is_file():
            continue
        try:
            rel = candidate.relative_to(resolved)
        except ValueError:
            continue
        if _has_hidden_component(rel):
            continue
        if candidate.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(Path(os.path.abspath(candidate)))
    return sorted(videos)


def read_swing_events_count(path: Path) -> Optional[int]:
    """Return event count if `path` is a valid non-empty swing events JSON dict, else None."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    events = data.get("events")
    if not isinstance(events, list):
        return None
    return len(events)


def is_valid_swing_json(path: Path) -> bool:
    """Return True iff `path` exists, is non-empty, and parses as a valid swing JSON dict."""
    return read_swing_events_count(path) is not None


def resolve_python_executable(repo_root: Path = REPO_ROOT) -> Path:
    """Resolve the repository's `./venv_yolo26/bin/python` interpreter."""
    preferred = repo_root / "venv_yolo26" / "bin" / "python"
    if preferred.exists():
        return preferred
    return Path(sys.executable)


def build_main_pipe_command(
    video_path: Path,
    tmp_dir: Path,
    *,
    repo_root: Path = REPO_ROOT,
    python_executable: Optional[Path] = None,
    config_path: str = DEFAULT_CONFIG,
    pose_model: str = DEFAULT_POSE_MODEL,
    racket_pose_mode: str = DEFAULT_RACKET_POSE_MODE,
    stream_id: Optional[str] = None,
    det_workers: Optional[int] = None,
    shm_slots: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> List[str]:
    """Build exact headless offline `main_pipe.py` command writing intermediates to `tmp_dir`."""
    py_exe = python_executable or resolve_python_executable(repo_root)
    tmp_output_video = tmp_dir / f"{video_path.stem}.mp4"
    tmp_swing_json = tmp_dir / f"{video_path.stem}_swing_events.json"

    cmd = [
        str(py_exe),
        str(repo_root / "main_pipe.py"),
        "--config",
        str(config_path),
        "--input",
        str(video_path),
        "--output",
        str(tmp_output_video),
        "--algo2-dual-view",
        "--processing-mode",
        "offline",
        "--headless",
        "--racket-pose-mode",
        str(racket_pose_mode),
        "--pose-model",
        str(pose_model),
        "--analyze-swings",
        "--swing-output-json",
        str(tmp_swing_json),
    ]
    if stream_id:
        cmd.extend(["--stream-id", str(stream_id)])
    if det_workers is not None:
        cmd.extend(["--det-workers", str(int(det_workers))])
    if shm_slots is not None:
        cmd.extend(["--shm-slots", str(int(shm_slots))])
    if max_frames is not None:
        cmd.extend(["--max-frames", str(int(max_frames))])
    return cmd


def process_video(
    video: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
    repo_root: Path = REPO_ROOT,
    python_executable: Optional[Path] = None,
    config_path: str = DEFAULT_CONFIG,
    pose_model: str = DEFAULT_POSE_MODEL,
    racket_pose_mode: str = DEFAULT_RACKET_POSE_MODE,
    stream_id: Optional[str] = None,
    det_workers: Optional[int] = None,
    shm_slots: Optional[int] = None,
    max_frames: Optional[int] = None,
    quiet: bool = False,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> VideoProcessResult:
    """Analyze a single video in a temp dir and atomically copy `<stem>_swing_events.json`."""
    target_json = video.parent / f"{video.stem}_swing_events.json"
    existing_count = read_swing_events_count(target_json)
    if not force and existing_count is not None:
        return VideoProcessResult(
            video=video,
            target_json=target_json,
            status="skipped",
            events_count=existing_count,
        )
    if dry_run:
        return VideoProcessResult(
            video=video,
            target_json=target_json,
            status="dry_run",
        )

    exec_fn = runner or subprocess.run
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tennis_swing_") as tmp_str:
        tmp_dir = Path(tmp_str)
        tmp_swing_json = tmp_dir / f"{video.stem}_swing_events.json"
        cmd = build_main_pipe_command(
            video,
            tmp_dir,
            repo_root=repo_root,
            python_executable=python_executable,
            config_path=config_path,
            pose_model=pose_model,
            racket_pose_mode=racket_pose_mode,
            stream_id=stream_id,
            det_workers=det_workers,
            shm_slots=shm_slots,
            max_frames=max_frames,
        )
        run_kwargs = {
            "cwd": str(repo_root),
            "check": False,
        }
        if quiet:
            run_kwargs["stdout"] = subprocess.PIPE
            run_kwargs["stderr"] = subprocess.STDOUT
            run_kwargs["text"] = True
        proc = exec_fn(cmd, **run_kwargs)
        elapsed_s = time.perf_counter() - started
        if proc.returncode != 0:
            err_msg = f"main_pipe exited with code {proc.returncode}"
            if quiet and getattr(proc, "stdout", None):
                tail = "\n".join(str(proc.stdout).splitlines()[-10:])
                err_msg = f"{err_msg}: {tail}"
            return VideoProcessResult(
                video=video,
                target_json=target_json,
                status="failed",
                elapsed_s=elapsed_s,
                error=err_msg,
            )

        events_count = read_swing_events_count(tmp_swing_json)
        if events_count is None:
            return VideoProcessResult(
                video=video,
                target_json=target_json,
                status="failed",
                elapsed_s=elapsed_s,
                error=f"Invalid or missing swing JSON produced at {tmp_swing_json}",
            )

        staged_tmp = video.parent / f".{video.stem}_swing_events.json.tmp.{os.getpid()}"
        try:
            shutil.copyfile(tmp_swing_json, staged_tmp)
            os.replace(staged_tmp, target_json)
        except Exception as exc:
            try:
                staged_tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return VideoProcessResult(
                video=video,
                target_json=target_json,
                status="failed",
                elapsed_s=elapsed_s,
                error=f"Failed writing {target_json}: {exc}",
            )

    return VideoProcessResult(
        video=video,
        target_json=target_json,
        status="ok",
        events_count=events_count,
        elapsed_s=elapsed_s,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量离线无渲染分析目录中的网球视频，并将 <视频名>_swing_events.json 输出到各视频同目录"
    )
    parser.add_argument("target", help="目标视频文件或目录路径")
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="仅扫描目标目录第一层，不递归子目录",
    )
    parser.set_defaults(recursive=True)
    parser.add_argument("--force", action="store_true", help="覆盖重跑已存在的 *_swing_events.json")
    parser.add_argument("--dry-run", action="store_true", help="仅预览待处理/跳过清单，不执行推理")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"主配置文件路径，默认 {DEFAULT_CONFIG}")
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL, help=f"姿态模型路径，默认 {DEFAULT_POSE_MODEL}")
    parser.add_argument(
        "--racket-pose-mode",
        choices=("legacy", "exclusive", "shadow", "active"),
        default=DEFAULT_RACKET_POSE_MODE,
        help=f"球拍方案，默认 {DEFAULT_RACKET_POSE_MODE}",
    )
    parser.add_argument("--stream-id", default=None, help="显式指定 ROI/镜面配置 stream_id")
    parser.add_argument("--det-workers", type=int, default=None, help="覆盖独立 Core ML 检测子进程数")
    parser.add_argument("--shm-slots", type=int, default=None, help="覆盖共享内存缓冲槽数")
    parser.add_argument("--max-frames", type=int, default=None, help="每个视频最多处理帧数")
    parser.add_argument("--quiet", action="store_true", help="静默子进程日志，仅输出批处理进度与汇总")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    target = Path(args.target)
    try:
        videos = discover_videos(target, recursive=args.recursive)
    except FileNotFoundError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    if not videos:
        print(f"⚠️ 未在 {target} 找到视频文件 ({', '.join(sorted(VIDEO_EXTENSIONS))})")
        return 0

    print(f"📂 共发现 {len(videos)} 个视频文件 | pose={args.pose_model} | mode={args.racket_pose_mode}")
    results: List[VideoProcessResult] = []
    for idx, video in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] 🎬 {video}")
        res = process_video(
            video,
            force=args.force,
            dry_run=args.dry_run,
            config_path=args.config,
            pose_model=args.pose_model,
            racket_pose_mode=args.racket_pose_mode,
            stream_id=args.stream_id,
            det_workers=args.det_workers,
            shm_slots=args.shm_slots,
            max_frames=args.max_frames,
            quiet=args.quiet,
        )
        results.append(res)
        if res.status == "ok":
            print(f"  ✅ 完成 ({res.elapsed_s:.2f}s) | {res.events_count} events -> {res.target_json}")
        elif res.status == "skipped":
            print(f"  ⏭️ 已跳过（已存在有效结果 {res.events_count} events）: {res.target_json}")
        elif res.status == "dry_run":
            print(f"  📝 [Dry-Run] 将输出 -> {res.target_json}")
        else:
            print(f"  ❌ 失败 ({res.elapsed_s:.2f}s): {res.error}", file=sys.stderr)

    ok_n = sum(1 for r in results if r.status == "ok")
    skip_n = sum(1 for r in results if r.status == "skipped")
    dry_n = sum(1 for r in results if r.status == "dry_run")
    fail_n = sum(1 for r in results if r.status == "failed")
    print(f"\n📊 汇总: 成功={ok_n} | 跳过={skip_n} | 预览={dry_n} | 失败={fail_n}")
    return 1 if fail_n > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
