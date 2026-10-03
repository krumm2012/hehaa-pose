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
    is_point_in_mirror_fn: Optional[Any] = None



class DualViewManager:
    """
    虚拟双机位解耦管理器
    """

    DEFAULT_CONFIG_PATH = "configs/dual_view_config.yaml"

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        stream_profile: Optional[Any] = None,
    ):
        base_config: Dict[str, Any] = {}
        path_to_load = config_path or self.DEFAULT_CONFIG_PATH
        loaded_yaml = self._load_yaml(path_to_load)
        if isinstance(loaded_yaml, dict):
            base_config.update(loaded_yaml)

        if config_dict is not None:
            if "mirror_view" in config_dict:
                base_config.update(config_dict)
            elif "reflection_roi" in config_dict or "polygon" in config_dict:
                base_config["mirror_view"] = config_dict
            else:
                base_config.update(config_dict)

        if stream_profile is not None:
            profile_mirror = getattr(stream_profile, "mirror_view", None)
            if isinstance(profile_mirror, dict) and profile_mirror:
                base_config["mirror_view"] = dict(profile_mirror)
            profile_front = getattr(stream_profile, "front_view", None)
            if isinstance(profile_front, dict) and profile_front:
                base_config["front_view"] = dict(profile_front)

        self.config = base_config

        # 选手位置动态追踪状态 (EMA 平滑)
        self._tracked_player_bbox: Optional[Tuple[float, float, float, float]] = None
        self._tracking_lost_counter: int = 0
        self._last_frame_shape: Optional[Tuple[int, int]] = None

        # 解析镜像机位配置
        m_cfg = self.config.get("mirror_view", {})
        self.mirror_reflection_roi_norm = tuple(m_cfg.get("reflection_roi", [0.2708, 0.1018, 0.6139, 0.4642]))
        self.mirror_polygon_norm = np.array(
            m_cfg.get(
                "polygon",
                [
                    [0.2708, 0.4593],
                    [0.2742, 0.1018],
                    [0.6139, 0.1214],
                    [0.5961, 0.4642],
                ],
            ),
            dtype=np.float32,
        )
        self.mirror_flip = bool(m_cfg.get("horizontal_flip", True))
        self.mirror_target_size = tuple(m_cfg.get("target_size", [540, 720]))
        self.mirror_padding = float(m_cfg.get("padding", 0.0))

        # 解析正面人像屏蔽区配置 (Mask Polygon)
        mask_poly = m_cfg.get("mask_polygon")
        if mask_poly and len(mask_poly) >= 3:
            self.mirror_mask_polygon_norm = np.array(mask_poly, dtype=np.float32)
        else:
            self.mirror_mask_polygon_norm = None

        # 解析正面机位配置
        f_cfg = self.config.get("front_view", {})
        self.front_default_roi_norm = tuple(f_cfg.get("default_roi", [0.34, 0.30, 0.58, 0.85]))
        self.front_bbox_padding = float(f_cfg.get("bbox_padding", 0.45))
        self.front_target_size = tuple(f_cfg.get("target_size", [540, 720]))

        # 可视化配置
        v_cfg = self.config.get("visualization", {})
        self.front_label = v_cfg.get("front_label", "FRONT VIEW")
        self.back_label = v_cfg.get("back_label", "BACK VIEW (MIRROR FLIPPED)")

    @property
    def tracked_player_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        return self._tracked_player_bbox

    def update_player_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        ema_alpha: float = 0.35,
    ) -> Tuple[float, float, float, float]:
        """更新并平滑前景选手包围盒 (x1, y1, x2, y2) 绝对像素坐标。"""
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return self._tracked_player_bbox or bbox

        self._tracking_lost_counter = 0
        if self._tracked_player_bbox is None:
            self._tracked_player_bbox = (float(x1), float(y1), float(x2), float(y2))
        else:
            tx1, ty1, tx2, ty2 = self._tracked_player_bbox
            smooth_x1 = tx1 * (1.0 - ema_alpha) + float(x1) * ema_alpha
            smooth_y1 = ty1 * (1.0 - ema_alpha) + float(y1) * ema_alpha
            smooth_x2 = tx2 * (1.0 - ema_alpha) + float(x2) * ema_alpha
            smooth_y2 = ty2 * (1.0 - ema_alpha) + float(y2) * ema_alpha
            self._tracked_player_bbox = (smooth_x1, smooth_y1, smooth_x2, smooth_y2)
        return self._tracked_player_bbox

    def update_player_from_keypoints(
        self,
        keypoints_orig: Dict[str, Any],
        min_conf: float = 0.25,
        ema_alpha: float = 0.35,
    ) -> Optional[Tuple[float, float, float, float]]:
        """从姿态关键点提取选手全身包围盒并平滑更新追踪。"""
        if not keypoints_orig:
            self._tracking_lost_counter += 1
            if self._tracking_lost_counter > 45:
                self._tracked_player_bbox = None
            return self._tracked_player_bbox

        valid_xs = []
        valid_ys = []
        for kp in keypoints_orig.values():
            if kp is None:
                continue
            conf = getattr(kp, 'conf', 1.0)
            if conf >= min_conf:
                valid_xs.append(kp.x)
                valid_ys.append(kp.y)

        if len(valid_xs) < 4:
            self._tracking_lost_counter += 1
            if self._tracking_lost_counter > 45:
                self._tracked_player_bbox = None
            return self._tracked_player_bbox

        min_x = min(valid_xs)
        max_x = max(valid_xs)
        min_y = min(valid_ys)
        max_y = max(valid_ys)

        # 防镜面虚影污染：若传入关键点完全落在后墙镜面区域内（包括脚底也在镜中），坚决拒绝污染前景选手追踪
        if getattr(self, "_last_frame_shape", None) is not None:
            fh, fw = self._last_frame_shape
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            if self.is_point_in_mirror(cx, max_y, fw, fh) or (self.is_point_in_mirror(cx, cy, fw, fh) and max_y < 0.55 * fh):
                return self._tracked_player_bbox

        # 保护：若关键点缺少下肢（例如被球网遮挡或半身帧），按人体工学比例扩展为全身高度
        has_lower_body = any(
            k in keypoints_orig and getattr(keypoints_orig[k], 'conf', 0) >= min_conf
            for k in ("left_knee", "right_knee", "left_ankle", "right_ankle")
        )
        if not has_lower_body:
            width = max(140.0, max_x - min_x)
            min_height = max(550.0, width * 3.5)
            max_y = max(max_y, min_y + min_height)

        return self.update_player_bbox((min_x, min_y, max_x, max_y), ema_alpha=ema_alpha)

    @property
    def mask_polygon_norm(self) -> Optional[np.ndarray]:
        return self.mirror_mask_polygon_norm

    @mask_polygon_norm.setter
    def mask_polygon_norm(self, value):
        if value is not None and len(value) >= 3:
            self.mirror_mask_polygon_norm = np.array(value, dtype=np.float32)
        else:
            self.mirror_mask_polygon_norm = None

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
        支持结合前景选手的水平横向坐标在镜面边界内进行物理对齐。
        """
        rx1, ry1, rx2, ry2 = self.mirror_reflection_roi_norm
        y1_px = ry1 * frame_h
        y2_px = ry2 * frame_h
        crop_h = max(10.0, y2_px - y1_px)

        target_w, target_h = self.mirror_target_size or (540, 720)
        target_aspect = float(target_w) / max(1.0, float(target_h))
        desired_w = crop_h * target_aspect

        min_mirror_x = rx1 * frame_w
        max_mirror_x = rx2 * frame_w
        mirror_span = max(10.0, max_mirror_x - min_mirror_x)

        effective_player = player_bbox or self._tracked_player_bbox
        if effective_player is not None:
            # 物理几何：平面镜成像横向 X 轴投影一致，以选手中心为基准居中
            px1, _, px2, _ = effective_player
            cx = (px1 + px2) / 2.0
        else:
            cx = (min_mirror_x + max_mirror_x) / 2.0

        if desired_w >= mirror_span:
            x1_px = min_mirror_x
            x2_px = max_mirror_x
        else:
            x1_px = cx - desired_w / 2.0
            x2_px = cx + desired_w / 2.0
            if x1_px < min_mirror_x:
                x2_px = min(max_mirror_x, x2_px + (min_mirror_x - x1_px))
                x1_px = min_mirror_x
            if x2_px > max_mirror_x:
                x1_px = max(min_mirror_x, x1_px - (x2_px - max_mirror_x))
                x2_px = max_mirror_x

        return self._pad_and_clamp_bbox((x1_px, y1_px, x2_px, y2_px), self.mirror_padding, frame_w, frame_h)

    def compute_front_crop_bbox(
        self,
        frame_w: int,
        frame_h: int,
        player_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[int, int, int, int]:
        """计算正面机位最佳裁剪视口（保持比例，充分容纳全身、球拍挥击与来球空间）。"""
        if player_bbox is not None:
            self.update_player_bbox(player_bbox)

        effective_player = player_bbox or self._tracked_player_bbox
        if effective_player is not None:
            px1, py1, px2, py2 = effective_player
            pw = max(160.0, px2 - px1)
            ph = max(550.0, py2 - py1, pw * 2.8)

            target_w, target_h = self.front_target_size or (540, 720)
            target_aspect = float(target_w) / max(1.0, float(target_h))

            # 高度：覆盖引拍过顶空间与地面触球空间，视口至少要有 0.62 * frame_h
            min_crop_h = max(ph * (1.0 + self.front_bbox_padding * 1.1), frame_h * 0.62)
            crop_h = max(min_crop_h, pw / target_aspect)
            crop_w = crop_h * target_aspect

            # 宽度：若持拍延展范围更广，优先扩大并按宽高比同步外扩
            min_width = pw * (1.0 + self.front_bbox_padding * 1.8)
            if min_width > crop_w:
                crop_w = min_width
                crop_h = crop_w / target_aspect

            # 水平居中
            cx = (px1 + px2) / 2.0
            fx1 = int(round(cx - crop_w / 2.0))
            fx2 = int(round(cx + crop_w / 2.0))
            if fx1 < 0:
                fx2 = min(frame_w, fx2 - fx1)
                fx1 = 0
            if fx2 > frame_w:
                fx1 = max(0, fx1 - (fx2 - frame_w))
                fx2 = frame_w

            # 垂直布局：确保脚底留有球场地面余量
            fy2 = int(round(min(frame_h, py2 + ph * 0.18)))
            fy1 = int(round(fy2 - crop_h))
            if fy1 < 0:
                fy1 = 0
                fy2 = int(min(frame_h, crop_h))

            fx1 = max(0, min(frame_w - 10, fx1))
            fy1 = max(0, min(frame_h - 10, fy1))
            fx2 = min(frame_w, max(fx1 + 10, fx2))
            fy2 = min(frame_h, max(fy1 + 10, fy2))
            return fx1, fy1, fx2, fy2

        # 回退至配置的默认 Front ROI
        rx1, ry1, rx2, ry2 = self.front_default_roi_norm
        fx1 = int(rx1 * frame_w)
        fy1 = int(ry1 * frame_h)
        fx2 = int(rx2 * frame_w)
        fy2 = int(ry2 * frame_h)
        fx1 = max(0, min(frame_w - 10, fx1))
        fy1 = max(0, min(frame_h - 10, fy1))
        fx2 = min(frame_w, max(fx1 + 10, fx2))
        fy2 = min(frame_h, max(fy1 + 10, fy2))
        return fx1, fy1, fx2, fy2

    def get_mirror_polygon_pixels(self, frame_w: int, frame_h: int) -> np.ndarray:
        """获取镜面多边形在原图中的绝对像素坐标 (N, 2)，用于几何碰撞检测与目标过滤。"""
        if self.mirror_polygon_norm is None or len(self.mirror_polygon_norm) == 0:
            return np.empty((0, 2), dtype=np.int32)
        pts = self.mirror_polygon_norm.copy()
        pts[:, 0] *= frame_w
        pts[:, 1] *= frame_h
        return pts.astype(np.int32)

    def is_point_in_mirror(self, x: float, y: float, frame_w: int, frame_h: int) -> bool:
        """判断原始画面中的绝对坐标点 (x, y) 是否落在镜面区域（多边形或外接框）内部。"""
        poly_px = self.get_mirror_polygon_pixels(frame_w, frame_h)
        if len(poly_px) >= 3:
            return cv2.pointPolygonTest(poly_px, (float(x), float(y)), False) >= 0
        rx1, ry1, rx2, ry2 = self.mirror_reflection_roi_norm
        return (rx1 * frame_w <= x <= rx2 * frame_w) and (ry1 * frame_h <= y <= ry2 * frame_h)

    def split_frame(
        self,
        frame: np.ndarray,
        player_bbox: Optional[Tuple[float, float, float, float]] = None,
        frame_id: int = 0,
        timestamp_ms: Optional[float] = None,
        front_crop_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> DualViewFrame:
        """
        核心分流方法：将单路原帧拆分为 Front 与 Back 两路视角。

        Args:
            frame: 原始视频帧 (BGR)
            player_bbox: 前景选手检测框 (x1, y1, x2, y2) 像素坐标，若为 None 则采用内部追踪或默认 ROI
            frame_id: 当前帧序号
            timestamp_ms: 毫秒级时间戳
            front_crop_bbox: 显式指定正面视口 (x1, y1, x2, y2) 像素坐标，避免跨进程重复计算漂移

        Returns:
            DualViewFrame 包含正面帧、背面帧及双向映射元数据
        """
        fh, fw = frame.shape[:2]
        self._last_frame_shape = (fh, fw)

        # 1. 提取正面机位区域 (Front ROI，优先使用显式指定的精确视口，其次自适应选手跟踪)
        if front_crop_bbox is not None and len(front_crop_bbox) == 4:
            fx1, fy1, fx2, fy2 = front_crop_bbox
        else:
            fx1, fy1, fx2, fy2 = self.compute_front_crop_bbox(fw, fh, player_bbox=player_bbox)
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

        # 3. 若配置了正面人像屏蔽区域 (mask_polygon)，对背面视角应用隐私暗色遮罩
        if self.mirror_mask_polygon_norm is not None:
            b_view = self.apply_mask_to_view(b_view, back_info, fw, fh)

        return DualViewFrame(
            frame_id=frame_id,
            original_frame=frame,
            front_frame=f_view,
            back_frame=b_view,
            front_info=front_info,
            back_info=back_info,
            timestamp_ms=timestamp_ms,
            is_point_in_mirror_fn=lambda x, y: self.is_point_in_mirror(x, y, fw, fh),
        )

    def apply_mask_to_view(
        self,
        b_view: np.ndarray,
        back_info: DualViewCropInfo,
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        """在背面视角上对正面人像屏蔽多边形应用遮罩。"""
        if self.mirror_mask_polygon_norm is None or len(self.mirror_mask_polygon_norm) < 3:
            return b_view

        canvas = b_view.copy()
        pts = []
        for nx, ny in self.mirror_mask_polygon_norm:
            vx, vy = back_info.map_from_original(nx * frame_w, ny * frame_h)
            pts.append([int(round(vx)), int(round(vy))])

        pts_arr = np.array([pts], dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, pts_arr, (12, 16, 24))
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.polylines(canvas, pts_arr, isClosed=True, color=(60, 80, 110), thickness=1, lineType=cv2.LINE_AA)
        return canvas

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
