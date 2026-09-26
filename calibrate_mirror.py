#!/usr/bin/env python3
"""
calibrate_mirror.py
───────────────────
算法 2.0 镜面边界描点与标定独立页面服务：
启动轻量 HTTP 服务，提供现代 Web Canvas 界面，供用户交互式标定室内训练仓后墙镜面的边界顶点。
支持：
1. 任意指定视频帧提取与拖拽标点
2. Backview (540x720 水平翻转) 实时渲染联动预览
3. 一键安全保存至 configs/dual_view_config.yaml (带时间戳备份)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlsplit

import cv2
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MirrorCalibrator")


class MirrorCalibrationServer:
    def __init__(
        self,
        video_path: Path,
        config_path: Path,
        html_path: Path,
        host: str = "127.0.0.1",
        port: int = 8502,
        stream_id: Optional[str] = None,
        roi_config_path: Optional[Path] = None,
    ):
        self.video_path = video_path.resolve()
        self.config_path = config_path.resolve()
        self.html_path = html_path.resolve()
        self.host = host
        self.port = port
        self.stream_id = stream_id

        if roi_config_path is not None:
            self.roi_config_path = roi_config_path.resolve()
        elif self.config_path.name == "roi_config.yaml":
            self.roi_config_path = self.config_path
        else:
            default_roi_path = self.config_path.parent / "roi_config.yaml"
            self.roi_config_path = default_roi_path.resolve() if default_roi_path.exists() else None

        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        if not self.html_path.exists():
            raise FileNotFoundError(f"HTML模板文件不存在: {self.html_path}")

        # 读取视频基础属性
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.release()

        # 缓存抽取的关键帧以加速交互
        self._frame_cache: Dict[int, bytes] = {}

    def get_frame_raw(self, index: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(self.total_frames - 1, index)))
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            frame = cv2.imread(str(self.video_path)) if self.total_frames == 1 else None
            if frame is None:
                raise ValueError(f"无法读取视频第 {index} 帧")
        return frame

    def get_frame_jpeg(self, index: int) -> bytes:
        if index in self._frame_cache:
            return self._frame_cache[index]

        frame = self.get_frame_raw(index)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("JPEG 编码失败")

        jpeg_bytes = buf.tobytes()
        if len(self._frame_cache) < 50:
            self._frame_cache[index] = jpeg_bytes
        return jpeg_bytes

    def get_current_config(self, stream_id: Optional[str] = None) -> Dict[str, Any]:
        target_id = stream_id or self.stream_id
        # 1. 优先尝试从 roi_config.yaml 中加载指定 stream_id 的 mirror_view
        if self.roi_config_path and self.roi_config_path.exists():
            try:
                with open(self.roi_config_path, "r", encoding="utf-8") as f:
                    doc = yaml.safe_load(f) or {}
                streams = doc.get("streams", [])
                matched = None
                if target_id:
                    matched = next((s for s in streams if s.get("stream_id") == target_id), None)
                if matched is None:
                    matched = next((s for s in streams if s.get("default", False)), None)
                if matched is None and streams:
                    matched = streams[0]
                if matched:
                    res = dict(matched.get("mirror_view") or {})
                    res["stream_id"] = matched.get("stream_id")
                    res["stream_label"] = matched.get("stream_label")
                    res["available_streams"] = [
                        {"stream_id": s.get("stream_id"), "label": s.get("stream_label", s.get("stream_id"))}
                        for s in streams
                    ]
                    return res
            except Exception as e:
                logger.error(f"从 roi_config.yaml 读取流配置异常: {e}")

        # 2. 回退到 dual_view_config.yaml
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return data.get("mirror_view", {})
        except Exception as e:
            logger.error(f"读取配置异常: {e}")
            return {}

    def render_backview_preview(
        self,
        frame_idx: int,
        reflection_roi: List[float],
        polygon: List[List[float]],
        mask_polygon: Optional[List[List[float]]] = None,
        preview_mode: str = "back",
    ) -> bytes:
        """根据动态调整的多边形和屏蔽区，实时合成背面视角或双视角合成 JPEG 预览。"""
        import numpy as np
        frame = self.get_frame_raw(frame_idx)
        cfg_dict = {
            "mirror_view": {
                "reflection_roi": reflection_roi,
                "polygon": polygon,
                "mask_polygon": mask_polygon or [],
                "horizontal_flip": True,
                "target_size": [540, 720],
                "padding": 0.0,
            },
            "front_view": {
                "default_roi": [0.34, 0.3, 0.58, 0.85],
                "target_size": [540, 720],
            },
        }
        from dual_view_manager import DualViewManager
        mgr = DualViewManager(config_dict=cfg_dict)
        dual = mgr.split_frame(frame, frame_id=frame_idx)
        if preview_mode == "sbs":
            preview_img = np.hstack([dual.front_frame, dual.back_frame])
        else:
            preview_img = dual.back_frame
        ok, buf = cv2.imencode(".jpg", preview_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Backview preview encode failed")
        return buf.tobytes()

    def save_config(
        self,
        reflection_roi: List[float],
        polygon: List[List[float]],
        mask_polygon: Optional[List[List[float]]] = None,
        stream_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """将标定好的镜面参数持久化写入 roi_config.yaml（特定机位）与 dual_view_config.yaml，并创建备份。"""
        # 1. 校验点位合法性
        if not (len(reflection_roi) == 4 and all(0.0 <= v <= 1.0 for v in reflection_roi)):
            raise ValueError("reflection_roi 必须是 4 个介于 0.0~1.0 的浮点数 [x1, y1, x2, y2]")
        if not (len(polygon) >= 3 and all(len(p) == 2 and 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0 for p in polygon)):
            raise ValueError("polygon 必须包含至少 3 个归一化坐标点 [[x, y], ...]")
        if mask_polygon is not None and len(mask_polygon) > 0:
            if not (len(mask_polygon) >= 3 and all(len(p) == 2 and 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0 for p in mask_polygon)):
                raise ValueError("mask_polygon 若存在必须包含至少 3 个归一化坐标点 [[x, y], ...]")

        target_stream_id = stream_id or self.stream_id
        new_mirror_view = {
            "enabled": True,
            "reflection_roi": [float(f"{v:.4f}") for v in reflection_roi],
            "polygon": [[float(f"{p[0]:.4f}"), float(f"{p[1]:.4f}")] for p in polygon],
            "padding": 0.0,
            "horizontal_flip": True,
            "target_size": [540, 720],
        }
        if mask_polygon and len(mask_polygon) >= 3:
            new_mirror_view["mask_polygon"] = [[float(f"{p[0]:.4f}"), float(f"{p[1]:.4f}")] for p in mask_polygon]
        else:
            new_mirror_view["mask_polygon"] = []

        backup_names = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 2. 如果存在 roi_config.yaml，将配置写入对应的 stream 中
        if self.roi_config_path and self.roi_config_path.exists():
            with open(self.roi_config_path, "r", encoding="utf-8") as f:
                roi_doc = yaml.safe_load(f) or {}
            streams = roi_doc.get("streams", [])
            target = None
            if target_stream_id:
                target = next((s for s in streams if s.get("stream_id") == target_stream_id), None)
            if target is None:
                target = next((s for s in streams if s.get("default", False)), None)
            if target is None and streams:
                target = streams[0]

            if target is not None:
                target["mirror_view"] = new_mirror_view
                roi_bak = self.roi_config_path.with_name(f"{self.roi_config_path.name}.bak_{ts}")
                shutil.copy2(self.roi_config_path, roi_bak)
                backup_names.append(roi_bak.name)
                with open(self.roi_config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(roi_doc, f, allow_unicode=True, sort_keys=False)
                logger.info(f"✅ 镜面标定已成功写入 {self.roi_config_path.name} (stream: {target.get('stream_id')})")

        # 3. 同时更新 dual_view_config.yaml（保持向下兼容）
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                full_cfg = yaml.safe_load(f) or {}
            dual_bak = self.config_path.with_name(f"{self.config_path.name}.bak_{ts}")
            shutil.copy2(self.config_path, dual_bak)
            backup_names.append(dual_bak.name)

            if "mirror_view" not in full_cfg:
                full_cfg["mirror_view"] = {}
            full_cfg["mirror_view"].update(new_mirror_view)
            if not new_mirror_view.get("mask_polygon") and "mask_polygon" in full_cfg["mirror_view"]:
                del full_cfg["mirror_view"]["mask_polygon"]

            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(full_cfg, f, allow_unicode=True, sort_keys=False)
            logger.info(f"✅ dual_view_config.yaml 兼容性同步更新完成: {self.config_path}")

        return {
            "success": True,
            "stream_id": target_stream_id,
            "backup": backup_names[0] if backup_names else "",
            "reflection_roi": new_mirror_view["reflection_roi"],
            "polygon": new_mirror_view["polygon"],
            "mask_polygon": new_mirror_view.get("mask_polygon", []),
        }


def make_handler(server_instance: MirrorCalibrationServer):
    class MirrorHandler(BaseHTTPRequestHandler):
        server_version = "MirrorCalibrator/1.0"

        def log_message(self, format_string, *args):
            sys.stdout.write(f"[mirror-calib] {format_string % args}\n")

        def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlsplit(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                content = server_instance.html_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

            if path == "/api/info":
                s_id = query.get("stream_id", [None])[0]
                self._send_json({
                    "video_path": str(server_instance.video_path),
                    "width": server_instance.width,
                    "height": server_instance.height,
                    "total_frames": server_instance.total_frames,
                    "fps": server_instance.fps,
                    "current_config": server_instance.get_current_config(s_id),
                })
                return

            if path == "/api/frame":
                try:
                    f_idx = int(query.get("index", ["0"])[0])
                    jpeg_bytes = server_instance.get_frame_jpeg(f_idx)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg_bytes)))
                    self.send_header("Cache-Control", "public, max-age=60")
                    self.end_headers()
                    self.wfile.write(jpeg_bytes)
                except Exception as e:
                    self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def do_POST(self):
            parsed = urlsplit(self.path)
            path = parsed.path

            if path in ("/api/preview_backview", "/api/mirror/preview_backview"):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length)
                    data = json.loads(raw.decode("utf-8"))
                    f_idx = int(data.get("frame_index", 0))
                    roi = data.get("reflection_roi", [0.27, 0.10, 0.61, 0.46])
                    poly = data.get("polygon", [])
                    mask_poly = data.get("mask_polygon", [])
                    mode = data.get("mode", "back")
                    jpeg_bytes = server_instance.render_backview_preview(f_idx, roi, poly, mask_poly, preview_mode=mode)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg_bytes)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(jpeg_bytes)
                except Exception as e:
                    logger.error(f"生成预览失败: {e}")
                    self._send_json({"success": False, "error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return

            if path == "/api/save_config":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length)
                    data = json.loads(raw.decode("utf-8"))
                    roi = data.get("reflection_roi", [])
                    poly = data.get("polygon", [])
                    mask_poly = data.get("mask_polygon", [])
                    s_id = data.get("stream_id")
                    res = server_instance.save_config(roi, poly, mask_poly, stream_id=s_id)
                    self._send_json(res)
                except Exception as e:
                    logger.error(f"保存配置失败: {e}")
                    self._send_json({"success": False, "error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    return MirrorHandler


def main():
    parser = argparse.ArgumentParser(description="算法 2.0 镜面边界描点与标定控制台")
    parser.add_argument(
        "--video",
        default="/Users/krum5539/Desktop/Camera/test/40.26.mp4",
        help="待标定的原视频文件路径",
    )
    parser.add_argument(
        "--config",
        default="configs/dual_view_config.yaml",
        help="双机位配置文件路径",
    )
    parser.add_argument(
        "--roi-config",
        default="configs/roi_config.yaml",
        help="ROI与机位主配置文件路径",
    )
    parser.add_argument(
        "--stream-id",
        default=None,
        help="待标定的机位ID (如 court01-main, camera04-main)",
    )
    parser.add_argument(
        "--html",
        default="mirror_calibration.html",
        help="前端页面文件路径",
    )
    parser.add_argument("--port", type=int, default=8502, help="本地服务端口 (默认: 8502)")
    parser.add_argument("--no-browser", action="store_true", help="不自动拉起浏览器")
    args = parser.parse_args()

    server_inst = MirrorCalibrationServer(
        video_path=Path(args.video),
        config_path=Path(args.config),
        html_path=Path(args.html),
        port=args.port,
        stream_id=args.stream_id,
        roi_config_path=Path(args.roi_config),
    )

    httpd = ThreadingHTTPServer((server_inst.host, server_inst.port), make_handler(server_inst))
    url = f"http://{server_inst.host}:{server_inst.port}"
    print(f"\n" + "=" * 65)
    print(f"🪞 算法 2.0 镜面边界描点与标定控制台服务已启动！")
    print(f"📍 访问地址: {url}")
    print(f"🎬 绑定视频: {server_inst.video_path} ({server_inst.width}x{server_inst.height})")
    print(f"⚙️ 配置文件: {server_inst.config_path}")
    print(f"=" * 65 + "\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 标定服务已正常退出。")
        httpd.server_close()


if __name__ == "__main__":
    main()
