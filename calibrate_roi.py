#!/usr/bin/env python3
"""Interactive four-point ROI calibration for a video or RTSP source."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import yaml

from roi_manager import ROIManager
from roi_stream_config import sanitize_stream_source


Point = Tuple[int, int]
FrameSize = Tuple[int, int]


def fit_frame_size(
    frame_size: FrameSize,
    max_width: int,
    max_height: int,
) -> Tuple[FrameSize, float]:
    """Fit a source frame inside the calibration window without stretching it."""
    width, height = frame_size
    scale = min(
        1.0,
        max(1, int(max_width)) / max(1, width),
        max(1, int(max_height)) / max(1, height),
    )
    display_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return display_size, scale


def points_to_source(
    display_points: Sequence[Sequence[float]],
    scale: float,
    frame_size: FrameSize,
) -> List[Point]:
    """Map points from the fitted calibration window to source pixels."""
    width, height = frame_size
    if scale <= 0:
        raise ValueError("Display scale must be positive")
    points = []
    for point in display_points:
        x = max(0, min(width - 1, int(round(float(point[0]) / scale))))
        y = max(0, min(height - 1, int(round(float(point[1]) / scale))))
        points.append((x, y))
    return points


def normalize_roi_points(points: Sequence[Point]) -> List[Point]:
    """Normalize four corners to TL, TR, BR, BL regardless of click order."""
    if len(points) != 4:
        return [tuple(point) for point in points]
    by_height = sorted(
        ((int(point[0]), int(point[1])) for point in points),
        key=lambda point: (point[1], point[0]),
    )
    top = sorted(by_height[:2], key=lambda point: point[0])
    bottom = sorted(by_height[2:], key=lambda point: point[0], reverse=True)
    return [top[0], top[1], bottom[0], bottom[1]]


def validate_roi_points(
    points: Sequence[Point],
    frame_size: FrameSize,
) -> Tuple[bool, str]:
    """Reject crossed, reversed, or implausibly small quadrilaterals."""
    if len(points) != 4:
        return False, "ROI 必须包含 4 个点"
    p1, p2, p3, p4 = points
    if p1[0] >= p2[0] or p4[0] >= p3[0]:
        return False, "请按左上、右上、右下、左下顺序点击"
    if max(p1[1], p2[1]) >= min(p3[1], p4[1]):
        return False, "上方两点必须位于下方两点之上"
    polygon = np.asarray(points, dtype=np.int32)
    if not cv2.isContourConvex(polygon):
        return False, "ROI 四边形发生交叉或不是凸四边形"
    area = abs(float(cv2.contourArea(polygon)))
    frame_area = max(1, int(frame_size[0]) * int(frame_size[1]))
    if area / frame_area < 0.05:
        return False, "ROI 面积小于画面的 5%，请扩大选区"
    return True, ""


def _matching_stream_profile(
    document: Dict[str, Any],
    sanitized_source: str,
) -> Dict[str, Any] | None:
    streams = document.get("streams")
    candidates: Iterable[Dict[str, Any]]
    if isinstance(streams, list):
        candidates = (item for item in streams if isinstance(item, dict))
    elif isinstance(streams, dict):
        candidates = (item for item in streams.values() if isinstance(item, dict))
    else:
        return document
    for candidate in candidates:
        candidate_source = str(
            candidate.get("source")
            or candidate.get("stream_source")
            or candidate.get("rtsp_url")
            or ""
        )
        if sanitize_stream_source(candidate_source) == sanitized_source:
            return candidate
    return None


def update_roi_document(
    document: Dict[str, Any],
    source: str,
    frame_size: FrameSize,
    roi_points: Sequence[Point],
    stream_id: str,
    stream_label: str,
) -> Dict[str, Any]:
    """Update the matching profile while preserving unrelated ROI metadata."""
    updated = deepcopy(document or {})
    sanitized_source = sanitize_stream_source(source)
    target = _matching_stream_profile(updated, sanitized_source)
    if target is None:
        streams = updated.setdefault("streams", [])
        target = {}
        if isinstance(streams, list):
            streams.append(target)
        elif isinstance(streams, dict):
            streams[str(stream_id)] = target
        else:
            raise ValueError("ROI config streams must be a list or mapping")

    target.update(
        {
            "roi_enabled": True,
            "stream_id": str(stream_id),
            "stream_label": str(stream_label),
            "stream_source": sanitized_source,
            "frame_size": [int(frame_size[0]), int(frame_size[1])],
            "roi_points": [
                [int(point[0]), int(point[1])]
                for point in roi_points
            ],
            "roi_description": (
                "4-point ROI: "
                + str([[int(point[0]), int(point[1])] for point in roi_points])
            ),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "created_by": target.get("created_by", "Tennis Analyzer ROI Calibration"),
            "version": str(target.get("version") or updated.get("version") or "1.0"),
        }
    )
    return updated


def save_roi_document(
    path: Path,
    document: Dict[str, Any],
) -> Path | None:
    """Atomically save a calibration and retain a timestamped backup."""
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if path.exists():
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = path.with_name(f"{path.name}.backup-{timestamp}")
        shutil.copy2(path, backup_path)

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(
            document,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return backup_path


def _open_capture(source: str, config: Dict[str, Any]):
    perf = config.get("pipeline_perf") or {}
    params = []
    open_timeout = max(0, int(perf.get("reader_open_timeout_ms", 5000)))
    read_timeout = max(0, int(perf.get("reader_read_timeout_ms", 3000)))
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC") and open_timeout:
        params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout])
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC") and read_timeout:
        params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout])
    try:
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    except (TypeError, cv2.error):
        return cv2.VideoCapture(source)


def capture_calibration_frame(
    source: str,
    config: Dict[str, Any],
    warmup_frames: int,
):
    """Read through a short warmup and return the freshest decoded frame."""
    capture = _open_capture(source, config)
    if not capture.isOpened():
        raise RuntimeError(
            f"无法打开校准输入: {sanitize_stream_source(source)}"
        )
    latest = None
    try:
        for _ in range(max(1, int(warmup_frames))):
            ok, frame = capture.read()
            if ok and frame is not None:
                latest = frame
        if latest is None:
            raise RuntimeError(
                f"无法读取校准画面: {sanitize_stream_source(source)}"
            )
        return latest
    finally:
        capture.release()


def select_roi_points(
    source_frame,
    config: Dict[str, Any],
    max_width: int,
    max_height: int,
    confirm_preview: bool = False,
) -> List[Point] | None:
    """Run selection and confirmation loops, returning source-pixel points."""
    source_height, source_width = source_frame.shape[:2]
    display_size, scale = fit_frame_size(
        (source_width, source_height),
        max_width,
        max_height,
    )
    display_frame = cv2.resize(
        source_frame,
        display_size,
        interpolation=cv2.INTER_AREA,
    )

    while True:
        manager = ROIManager(config)
        print("\n鼠标点击 ROI 的四个角点，顺序不限，程序会自动识别 P1–P4")
        print("选点窗口：c 确认，r 重选，q/ESC 取消")
        display_points = manager.interactive_roi_selection(
            display_frame.copy(),
            "Court 01 ROI Calibration",
        )
        if len(display_points) != 4:
            return None

        clicked_source_points = points_to_source(
            display_points,
            scale,
            (source_width, source_height),
        )
        source_points = normalize_roi_points(clicked_source_points)
        if source_points != clicked_source_points:
            print(
                "🔄 已自动排序为 P1左上→P2右上→P3右下→P4左下: "
                f"{source_points}"
            )
        valid, validation_error = validate_roi_points(
            source_points,
            (source_width, source_height),
        )
        if not valid:
            print(f"⚠️ {validation_error}，请重新选择")
            continue
        if not confirm_preview:
            return source_points

        preview_manager = ROIManager(config)
        preview_manager.set_roi_points(source_points)
        preview = preview_manager.draw_roi(source_frame, show_fill=True)
        preview = cv2.resize(preview, display_size, interpolation=cv2.INTER_AREA)
        cv2.namedWindow("ROI Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI Preview", *display_size)
        cv2.imshow("ROI Preview", preview)
        print(f"原图坐标: {source_points}")
        print("预览窗口：s/c/Enter 保存，r 重新选点，q/ESC 取消")

        action = None
        while action is None:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("s"), ord("c"), 10, 13):
                action = "save"
            elif key == ord("r"):
                action = "retry"
            elif key in (ord("q"), 27):
                action = "cancel"
        cv2.destroyWindow("ROI Preview")
        if action == "save":
            return source_points
        if action == "cancel":
            return None


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从视频或 RTSP 画面鼠标校准四点 ROI",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="视频或 RTSP 地址；日志和配置不会保存认证信息",
    )
    parser.add_argument(
        "--config",
        default="configs/yolo26_tennis_config.yaml",
        help="主配置路径",
    )
    parser.add_argument(
        "--roi-config",
        default=None,
        help="ROI 配置路径；默认读取主配置 roi_config_path",
    )
    parser.add_argument("--stream-id", default=None, help="摄像机标识")
    parser.add_argument("--stream-label", default=None, help="摄像机显示名称")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=15,
        help="选取画面前读取的帧数，默认 15",
    )
    parser.add_argument(
        "--display-max-width",
        type=int,
        default=1280,
        help="选点窗口最大宽度，默认 1280",
    )
    parser.add_argument(
        "--display-max-height",
        type=int,
        default=800,
        help="选点窗口最大高度，默认 800",
    )
    parser.add_argument(
        "--preview-output",
        default=None,
        help="保存校准后的原分辨率预览图",
    )
    parser.add_argument(
        "--confirm-preview",
        action="store_true",
        help="保存前增加第二个预览确认窗口；默认选点窗口按 c 后立即保存",
    )
    return parser


def main(argv=None) -> int:
    args = build_argument_parser().parse_args(argv)
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    roi_settings = config.get("roi_settings") or {}
    roi_path = Path(
        args.roi_config
        or roi_settings.get("roi_config_path")
        or "configs/roi_config.yaml"
    )
    existing = (
        yaml.safe_load(roi_path.read_text(encoding="utf-8")) or {}
        if roi_path.exists()
        else {}
    )
    if not isinstance(existing, dict):
        raise ValueError("ROI config must contain a YAML mapping")

    current_profile = _matching_stream_profile(
        existing,
        sanitize_stream_source(args.input),
    )
    current_profile = current_profile or {}
    stream_id = (
        args.stream_id
        or current_profile.get("stream_id")
        or current_profile.get("id")
        or "court01-main"
    )
    stream_label = (
        args.stream_label
        or current_profile.get("stream_label")
        or current_profile.get("label")
        or "Court 01 Main Camera"
    )

    print(f"🎥 校准输入: {sanitize_stream_source(args.input)}")
    frame = capture_calibration_frame(
        args.input,
        config,
        args.warmup_frames,
    )
    height, width = frame.shape[:2]
    print(f"📐 原始分辨率: {width}x{height}")
    points = select_roi_points(
        frame,
        config,
        args.display_max_width,
        args.display_max_height,
        confirm_preview=args.confirm_preview,
    )
    cv2.destroyAllWindows()
    if points is None:
        print("ℹ️ 已取消校准，配置未修改")
        return 0

    updated = update_roi_document(
        existing,
        args.input,
        (width, height),
        points,
        str(stream_id),
        str(stream_label),
    )
    backup_path = save_roi_document(roi_path, updated)
    persisted = yaml.safe_load(roi_path.read_text(encoding="utf-8")) or {}
    persisted_profile = _matching_stream_profile(
        persisted,
        sanitize_stream_source(args.input),
    )
    persisted_points = (persisted_profile or {}).get("roi_points")
    expected_points = [list(point) for point in points]
    if persisted_points != expected_points:
        raise RuntimeError(
            f"ROI 写入后校验失败: {roi_path.resolve()}"
        )
    print(f"✅ ROI 已保存并复核: {roi_path.resolve()}")
    if backup_path is not None:
        print(f"🛟 原配置备份: {backup_path.resolve()}")

    preview_manager = ROIManager(config)
    preview_manager.set_roi_points(points)
    preview = preview_manager.draw_roi(frame, show_fill=True)
    preview_path = Path(
        args.preview_output
        or roi_path.with_name(f"{roi_path.stem}_calibrated_preview.jpg")
    )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"无法写入预览图: {preview_path}")

    print(f"🖼️ 校准预览: {preview_path.resolve()}")
    print(f"📍 roi_points: {[list(point) for point in points]}")
    print("请重启 main_pipe.py 载入新 ROI。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
