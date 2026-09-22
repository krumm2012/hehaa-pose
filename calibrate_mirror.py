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
    ):
        self.video_path = video_path.resolve()
        self.config_path = config_path.resolve()
        self.html_path = html_path.resolve()
        self.host = host
        self.port = port

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

    def get_frame_jpeg(self, index: int) -> bytes:
        if index in self._frame_cache:
            return self._frame_cache[index]

        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(self.total_frames - 1, index)))
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            # 返回空帧或纯黑
            frame = cv2.imread(str(self.video_path)) if self.total_frames == 1 else None
            if frame is None:
                raise ValueError(f"无法读取视频第 {index} 帧")

        # 压缩为 JPEG
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("JPEG 编码失败")

        jpeg_bytes = buf.tobytes()
        if len(self._frame_cache) < 50:
            self._frame_cache[index] = jpeg_bytes
        return jpeg_bytes

    def get_current_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return data.get("mirror_view", {})
        except Exception as e:
            logger.error(f"读取配置异常: {e}")
            return {}

    def save_config(self, reflection_roi: List[float], polygon: List[List[float]]) -> Dict[str, Any]:
        """将标定好的镜面参数持久化写入 dual_view_config.yaml 并创建备份。"""
        # 1. 校验点位合法性
        if not (len(reflection_roi) == 4 and all(0.0 <= v <= 1.0 for v in reflection_roi)):
            raise ValueError("reflection_roi 必须是 4 个介于 0.0~1.0 的浮点数 [x1, y1, x2, y2]")
        if not (len(polygon) >= 3 and all(len(p) == 2 and 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0 for p in polygon)):
            raise ValueError("polygon 必须包含至少 3 个归一化坐标点 [[x, y], ...]")

        # 2. 读取现有配置完整文档
        with open(self.config_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}

        # 3. 创建时间戳备份
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.config_path.with_name(f"{self.config_path.name}.bak_{ts}")
        shutil.copy2(self.config_path, backup_path)
        logger.info(f"已创建配置历史备份: {backup_path}")

        # 4. 更新 mirror_view 节点
        if "mirror_view" not in full_cfg:
            full_cfg["mirror_view"] = {}

        full_cfg["mirror_view"]["reflection_roi"] = [float(f"{v:.4f}") for v in reflection_roi]
        full_cfg["mirror_view"]["polygon"] = [[float(f"{p[0]:.4f}"), float(f"{p[1]:.4f}")] for p in polygon]

        # 5. 写回文件
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(full_cfg, f, allow_unicode=True, sort_keys=False)

        logger.info(f"✅ 镜面标定配置已成功更新: {self.config_path}")
        return {
            "success": True,
            "backup": str(backup_path.name),
            "reflection_roi": full_cfg["mirror_view"]["reflection_roi"],
            "polygon": full_cfg["mirror_view"]["polygon"],
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
                self._send_json({
                    "video_path": str(server_instance.video_path),
                    "width": server_instance.width,
                    "height": server_instance.height,
                    "total_frames": server_instance.total_frames,
                    "fps": server_instance.fps,
                    "current_config": server_instance.get_current_config(),
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

            if path == "/api/save_config":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length)
                    data = json.loads(raw.decode("utf-8"))
                    roi = data.get("reflection_roi", [])
                    poly = data.get("polygon", [])
                    res = server_instance.save_config(roi, poly)
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
        help="配置文件路径",
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
