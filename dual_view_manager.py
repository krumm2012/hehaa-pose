"""
dual_view_manager.py
─────────────────────
算法 2.0 核心机位解耦组件：
利用室内网球训练仓“正面摄像头 + 背后平镜”的光学特性，将单目 2.5K 原始画面解耦为：
1. Front View（正面真实选手视角）：带 0.45 扩展边界的动态/静态选手 ROI
2. Back View（背面虚拟机位）：镜面多边形裁剪并做水平翻转（Horizontal Flip），使动作视角与真实后置机位一致

提供高精度双向坐标映射（View 坐标 ⟷ 原始全图坐标），为后续前后双视角姿态互补融合奠定几何基础。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

logger = logging.getLogger("DualViewManager")


@dataclass
class DualViewCropInfo:
    """单个视角的裁剪与几何变换元数据。"""
    # 原始全图中的矩形包围盒 (x1, y1, x2, y2)，以像素为单位
    bbox_orig: Tuple[int, int, int, int]
    # 原始裁剪宽高 (w, h)
    crop_size: Tuple[int, int]
    # 最终输出视图宽高 (w, h)
    view_size: Tuple[int, int]
    # 是否进行了水平镜像翻转
    is_horizontally_flipped: bool = False

    @property
    def scale_x(self) -> float:
        return self.view_size[0] / max(1, self.crop_size[0])

    @property
    def scale_y(self) -> float:
        return self.view_size[1] / max(1, self.crop_size[1])

    def map_to_original(self, x_view: float, y_view: float) -> Tuple[float, float]:
        """将当前视角下的坐标 (x_view, y_view) 映射回原始全图坐标 (x_orig, y_orig)。"""
        x1, y1, x2, y2 = self.bbox_orig
        crop_w, crop_h = self.crop_size

        # 1. 消除目标分辨率缩放
        x_unscaled = x_view / max(1e-6, self.scale_x)
        y_unscaled = y_view / max(1e-6, self.scale_y)

        # 2. 消除水平镜像翻转 (如果是翻转视角)
        if self.is_horizontally_flipped:
            x_crop = (crop_w - 1.0) - x_unscaled
        else:
            x_crop = x_unscaled

        y_crop = y_unscaled

        # 3. 映射回原图偏移
        x_orig = x1 + x_crop
        y_orig = y1 + y_crop
        return float(x_orig), float(y_orig)

    def map_from_original(self, x_orig: float, y_orig: float) -> Tuple[float, float]:
        """将原始全图坐标 (x_orig, y_orig) 映射至当前视角局部坐标 (x_view, y_view)。"""
        x1, y1, x2, y2 = self.bbox_orig
        crop_w, crop_h = self.crop_size

        x_crop = x_orig - x1
        y_crop = y_orig - y1

        if self.is_horizontally_flipped:
            x_unscaled = (crop_w - 1.0) - x_crop
        else:
            x_unscaled = x_crop

        y_unscaled = y_crop

        x_view = x_unscaled * self.scale_x
        y_view = y_unscaled * self.scale_y
        return float(x_view), float(y_view)


@dataclass
class DualViewFrame:
    """单帧解耦后的双视角数据容器。"""
    frame_id: int
    original_frame: np.ndarray
    front_frame: np.ndarray
    back_frame: np.ndarray
    front_info: DualViewCropInfo
    back_info: DualViewCropInfo
    timestamp_ms: Optional[float] = None


class DualViewManager:
    """
    虚拟双机位解耦管理器
    """

    DEFAULT_CONFIG_PATH = "configs/dual_view_config.yaml"

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        if config_dict is not None:
            self.config = config_dict
        else:
            path_to_load = config_path or self.DEFAULT_CONFIG_PATH
            self.config = self._load_yaml(path_to_load)

        # 解析镜像机位配置
        m_cfg = self.config.get("mirror_view", {})
        self.mirror_reflection_roi_norm = tuple(m_cfg.get("reflection_roi", [0.36, 0.11, 0.54, 0.44]))
        self.mirror_polygon_norm = np.array(
            m_cfg.get(
                "polygon",
                [
                    [0.0, 0.20],
                    [0.34, 0.105],
                    [0.615, 0.125],
                    [0.59, 0.478],
                    [0.0, 0.478],
                ],
            ),
            dtype=np.float32,
        )
        self.mirror_flip = bool(m_cfg.get("horizontal_flip", True))
        self.mirror_target_size = tuple(m_cfg.get("target_size", [540, 720]))
        self.mirror_padding = float(m_cfg.get("padding", 0.10))

        # 解析正面机位配置
        f_cfg = self.config.get("front_view", {})
        self.front_default_roi_norm = tuple(f_cfg.get("default_roi", [0.34, 0.30, 0.58, 0.85]))
        self.front_bbox_padding = float(f_cfg.get("bbox_padding", 0.45))
        self.front_target_size = tuple(f_cfg.get("target_size", [540, 720]))

        # 可视化配置
        v_cfg = self.config.get("visualization", {})
        self.front_label = v_cfg.get("front_label", "FRONT VIEW")
        self.back_label = v_cfg.get("back_label", "BACK VIEW (MIRROR FLIPPED)")

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            logger.warning(f"Config path {path} does not exist, using fallback defaults.")
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading {path}: {e}, using defaults.")
            return {}

    def _pad_and_clamp_bbox(
        self,
        box: Tuple[float, float, float, float],
        padding_ratio: float,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[int, int, int, int]:
        """按指定比例外扩边界框并限制在画面范围内。"""
        x1, y1, x2, y2 = box
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        pad_x = bw * padding_ratio
        pad_y = bh * padding_ratio

        nx1 = max(0, int(np.floor(x1 - pad_x)))
        ny1 = max(0, int(np.floor(y1 - pad_y)))
        nx2 = min(frame_w, int(np.ceil(x2 + pad_x)))
        ny2 = min(frame_h, int(np.ceil(y2 + pad_y)))

        return nx1, ny1, nx2, ny2

    def get_mirror_bbox_pixels(
        self,
        frame_w: int,
        frame_h: int,
        player_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[int, int, int, int]:
        """获取镜中人像区域的外接矩形框（像素坐标）。
        支持结合前景选手的水平横向坐标进行物理对齐。
        """
        rx1, ry1, rx2, ry2 = self.mirror_reflection_roi_norm
        y1_px = ry1 * frame_h
        y2_px = ry2 * frame_h

        if player_bbox is not None:
            # 物理几何：平面镜成像横向 X 轴投影一致
            px1, _, px2, _ = player_bbox
            pw = max(100.0, px2 - px1)
            x1_px = max(0.0, px1 - pw * 0.3)
            x2_px = min(float(frame_w), px2 + pw * 0.3)
        else:
            x1_px = rx1 * frame_w
            x2_px = rx2 * frame_w

        return self._pad_and_clamp_bbox((x1_px, y1_px, x2_px, y2_px), self.mirror_padding, frame_w, frame_h)

    def split_frame(
        self,
        frame: np.ndarray,
        player_bbox: Optional[Tuple[float, float, float, float]] = None,
        frame_id: int = 0,
        timestamp_ms: Optional[float] = None,
    ) -> DualViewFrame:
        """
        核心分流方法：将单路原帧拆分为 Front 与 Back 两路视角。

        Args:
            frame: 原始视频帧 (BGR)
            player_bbox: 前景选手检测框 (x1, y1, x2, y2) 像素坐标，若为 None 则采用默认 ROI
            frame_id: 当前帧序号
            timestamp_ms: 毫秒级时间戳

        Returns:
            DualViewFrame 包含正面帧、背面帧及双向映射元数据
        """
        fh, fw = frame.shape[:2]

        # 1. 提取正面机位区域 (Front ROI)
        if player_bbox is not None:
            # 采用 Tennis-Vision 的 0.45 黄金 padding
            fx1, fy1, fx2, fy2 = self._pad_and_clamp_bbox(
                player_bbox, self.front_bbox_padding, fw, fh
            )
        else:
            rx1, ry1, rx2, ry2 = self.front_default_roi_norm
            fx1 = int(rx1 * fw)
            fy1 = int(ry1 * fh)
            fx2 = int(rx2 * fw)
            fy2 = int(ry2 * fh)

        # 确保裁剪区域有效
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(fw, max(fx1 + 10, fx2)), min(fh, max(fy1 + 10, fy2))
        f_crop = frame[fy1:fy2, fx1:fx2].copy()
        f_orig_size = (f_crop.shape[1], f_crop.shape[0])

        if self.front_target_size is not None:
            f_view = cv2.resize(f_crop, self.front_target_size)
            f_view_size = self.front_target_size
        else:
            f_view = f_crop
            f_view_size = f_orig_size

        front_info = DualViewCropInfo(
            bbox_orig=(fx1, fy1, fx2, fy2),
            crop_size=f_orig_size,
            view_size=f_view_size,
            is_horizontally_flipped=False,
        )

        # 2. 提取背面机位区域 (Mirror ROI)
        bx1, by1, bx2, by2 = self.get_mirror_bbox_pixels(fw, fh, player_bbox=player_bbox)
        b_crop = frame[by1:by2, bx1:bx2].copy()
        b_orig_size = (b_crop.shape[1], b_crop.shape[0])

        # 水平镜像翻转 (纠正左右反向)
        if self.mirror_flip:
            b_processed = cv2.flip(b_crop, 1)
        else:
            b_processed = b_crop

        if self.mirror_target_size is not None:
            b_view = cv2.resize(b_processed, self.mirror_target_size)
            b_view_size = self.mirror_target_size
        else:
            b_view = b_processed
            b_view_size = b_orig_size

        back_info = DualViewCropInfo(
            bbox_orig=(bx1, by1, bx2, by2),
            crop_size=b_orig_size,
            view_size=b_view_size,
            is_horizontally_flipped=self.mirror_flip,
        )

        return DualViewFrame(
            frame_id=frame_id,
            original_frame=frame,
            front_frame=f_view,
            back_frame=b_view,
            front_info=front_info,
            back_info=back_info,
            timestamp_ms=timestamp_ms,
        )

    def render_side_by_side(
        self,
        dual_frame: DualViewFrame,
        draw_labels: bool = True,
        target_height: Optional[int] = None,
    ) -> np.ndarray:
        """生成 Side-by-Side 双画面同屏展示图像。"""
        f_img = dual_frame.front_frame.copy()
        b_img = dual_frame.back_frame.copy()

        # 对齐高度
        th = target_height or max(f_img.shape[0], b_img.shape[0])
        if f_img.shape[0] != th:
            fw = int(f_img.shape[1] * th / f_img.shape[0])
            f_img = cv2.resize(f_img, (fw, th))
        if b_img.shape[0] != th:
            bw = int(b_img.shape[1] * th / b_img.shape[0])
            b_img = cv2.resize(b_img, (bw, th))

        if draw_labels:
            # 标注文字
            cv2.putText(
                f_img,
                self.front_label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                b_img,
                self.back_label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

        return np.hstack([f_img, b_img])
