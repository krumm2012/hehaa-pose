"""
dual_pose_estimator.py
───────────────────────
算法 2.0 双视角姿态估计与统合器：
在解耦出的正面机位 (Front View) 与背面机位 (Back View) 上并行运行姿态估计。
支持 Core ML (yolo26m-pose / ANE加速) 与 Ultralytics (YOLOv8-pose) 双后端。
自动完成：
1. 双视角姿态独立检测
2. 视角局部坐标 ⟷ 原始全图全局坐标映射
3. 遮挡自愈与生物力学指标聚合
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from dual_view_biomechanics import (
    COCO_KEYPOINTS,
    DualViewBiomechanicsEngine,
    DualViewBiomechanicsResult,
    Keypoint,
)
from dual_view_manager import DualViewCropInfo, DualViewFrame

logger = logging.getLogger("DualPoseEstimator")


@dataclass
class DualPoseResult:
    """双视角姿态检测综合结果。"""
    frame_id: int
    # 正面视角关键点（局部坐标）
    front_pose_local: Dict[str, Keypoint]
    # 背面视角关键点（局部坐标）
    back_pose_local: Dict[str, Keypoint]
    # 正面视角关键点（映射至原始全图坐标）
    front_pose_orig: Dict[str, Keypoint]
    # 背面视角关键点（映射至原始全图坐标，考虑水平翻转逆运算）
    back_pose_orig: Dict[str, Keypoint]
    # 融合姿态（包含正面遮挡时自愈补全的关键点，局部坐标）
    fused_pose_local: Dict[str, Keypoint]
    # 生物力学分析结果
    biomechanics: DualViewBiomechanicsResult
    # 融合姿态（映射至原始全图全局坐标）
    fused_pose_orig: Dict[str, Keypoint] = field(default_factory=dict)
    timestamp_ms: Optional[float] = None


class DualPoseEstimator:
    """
    双视角姿态估计器
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        dominant_hand: str = "right",
        max_contact_distance: float = 200.0,
        backend: str = "auto",  # 'auto' | 'coreml' | 'ultralytics'
    ):
        self.conf_threshold = conf_threshold
        self.biomech_engine = DualViewBiomechanicsEngine(
            dominant_hand=dominant_hand,
            max_contact_distance=max_contact_distance,
            min_keypoint_conf=conf_threshold,
        )
        self.backend = backend
        self.model = None
        self._init_backend(model_path)

    def _init_backend(self, model_path: Optional[str]):
        """初始化姿态推理后端（优先 Core ML，回退至 Ultralytics）。"""
        # 1. 尝试 Core ML
        if self.backend in ("auto", "coreml"):
            try:
                from pose_estimator_yolo26 import PoseEstimatorYOLO26
                candidate_paths = [
                    model_path,
                    "yolo26m-pose.mlpackage",
                    "yolo26n-pose.mlpackage",
                    "models/yolo26m-pose.mlpackage",
                ]
                for cp in candidate_paths:
                    if cp and Path(cp).exists():
                        self.model = PoseEstimatorYOLO26(
                            cp,
                            {
                                "pose_confidence_threshold": self.conf_threshold,
                                "pose_keypoint_confidence": self.conf_threshold * 0.7,
                                "pose_smoothing_enabled": False,
                            },
                        )
                        self.backend = "coreml"
                        logger.info(f"Loaded CoreML pose model from {cp}")
                        return
            except Exception as e:
                logger.debug(f"CoreML backend not initialized: {e}")

        # 2. 尝试 Ultralytics YOLO Pose
        if self.backend in ("auto", "ultralytics"):
            try:
                from ultralytics import YOLO
                candidate_paths = [
                    model_path,
                    "yolov8n-pose.pt",
                    "yolov8s-pose.pt",
                    "yolo11n-pose.pt",
                ]
                for cp in candidate_paths:
                    if cp and Path(cp).exists():
                        self.model = YOLO(cp)
                        self.backend = "ultralytics"
                        logger.info(f"Loaded Ultralytics pose model from {cp}")
                        return
                # 默认加载 yolov8n-pose.pt
                self.model = YOLO("yolov8n-pose.pt")
                self.backend = "ultralytics"
                logger.info("Loaded default yolov8n-pose.pt backend")
                return
            except Exception as e:
                logger.warning(f"Ultralytics backend error: {e}")

        logger.warning("No ML pose model loaded. DualPoseEstimator running in dummy/pass-through mode.")
        self.backend = "mock"

    def _predict_single_view(
        self,
        view_frame: np.ndarray,
        is_back_view: bool = False,
    ) -> Dict[str, Keypoint]:
        """对单路裁剪视角画面进行姿态估计。"""
        if self.model is None or self.backend == "mock":
            return {}

        h, w = view_frame.shape[:2]

        if self.backend == "ultralytics":
            results = self.model(view_frame, conf=self.conf_threshold, verbose=False)
            if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
                return {}
            kp_data = results[0].keypoints.data.cpu().numpy()  # [N, 17, 3] or [N, 17, 2]
            if len(kp_data) == 0:
                return {}

            # 背面镜面机位：过滤底部前景闯入人像（双肩位于画面高度 65% 以外）
            best_person = None
            if is_back_view:
                candidates = []
                for p in kp_data:
                    ls_y = p[5, 1] if p.shape[0] > 5 and (p.shape[1] <= 2 or p[5, 2] >= self.conf_threshold) else None
                    rs_y = p[6, 1] if p.shape[0] > 6 and (p.shape[1] <= 2 or p[6, 2] >= self.conf_threshold) else None
                    sh_ys = [y for y in (ls_y, rs_y) if y is not None]
                    if sh_ys and min(sh_ys) <= h * 0.65:
                        candidates.append((min(sh_ys), p))
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    best_person = candidates[0][1]
                else:
                    return {}

            if best_person is None:
                best_person = kp_data[0]

            parsed = {}
            for idx, name in COCO_KEYPOINTS.items():
                x = float(best_person[idx, 0])
                y = float(best_person[idx, 1])
                conf = float(best_person[idx, 2]) if best_person.shape[1] > 2 else 1.0
                if conf >= self.conf_threshold:
                    parsed[name] = Keypoint(x=x, y=y, conf=conf)
            return parsed

        # Core ML 分支
        if self.backend == "coreml":
            try:
                kpts_list = self.model.get_keypoints(view_frame)
                if not kpts_list:
                    return {}

                best = None
                if is_back_view:
                    # 背面镜面机位：镜中人像位于镜像区中上部，过滤底部前景人体
                    candidates = []
                    for p in kpts_list:
                        ls = p.get("left_shoulder")
                        rs = p.get("right_shoulder")
                        sh_ys = [pt[1] for pt in (ls, rs) if pt is not None]
                        if sh_ys and min(sh_ys) <= h * 0.65:
                            candidates.append((min(sh_ys), p))
                    if candidates:
                        candidates.sort(key=lambda x: x[0])
                        best = candidates[0][1]
                    else:
                        return {}

                if best is None:
                    best = kpts_list[0]

                parsed = {}
                for name, pt in best.items():
                    if pt is not None:
                        parsed[name] = Keypoint(x=float(pt[0]), y=float(pt[1]), conf=0.85)
                return parsed
            except Exception as e:
                logger.debug(f"CoreML prediction exception: {e}")
                return {}

        return {}

    def _map_pose_to_original(
        self,
        pose_local: Dict[str, Keypoint],
        crop_info: DualViewCropInfo,
    ) -> Dict[str, Keypoint]:
        """将视角局部坐标下的关键点映射回原始全景画面空间。"""
        mapped: Dict[str, Keypoint] = {}
        for name, kp in pose_local.items():
            orig_x, orig_y = crop_info.map_to_original(kp.x, kp.y)
            mapped[name] = Keypoint(
                x=orig_x,
                y=orig_y,
                conf=kp.conf,
                z=kp.z,
                recovered_from_mirror=kp.recovered_from_mirror,
            )
        return mapped

    def estimate_dual_pose(
        self,
        dual_frame: DualViewFrame,
        ball_pos: Optional[Tuple[float, float]] = None,
    ) -> DualPoseResult:
        """
        双视角主估计流程：
        1. 分别在正面和背面提取姿态
        2. 坐标转换映射至原始全局像素
        3. 进行生物力学融合、抗侧身塌陷及遮挡自愈
        """
        # 1. 独立估计
        front_pose_local = self._predict_single_view(dual_frame.front_frame, is_back_view=False)
        back_pose_local = self._predict_single_view(dual_frame.back_frame, is_back_view=True)

        # 2. 映射回原图坐标
        front_pose_orig = self._map_pose_to_original(front_pose_local, dual_frame.front_info)
        back_pose_orig = self._map_pose_to_original(back_pose_local, dual_frame.back_info)

        # 3. 生物力学融合计算
        # 如果提供了原图 ball_pos，将其映射至正面机位坐标用于击球手与触球间距分析
        front_ball_pos = None
        if ball_pos is not None:
            bx, by = dual_frame.front_info.map_from_original(ball_pos[0], ball_pos[1])
            front_ball_pos = (bx, by)

        biomech_res = self.biomech_engine.calculate_dual_biomechanics(
            front_pose_raw=front_pose_local,
            back_pose_raw=back_pose_local,
            ball_pos=front_ball_pos,
        )

        # 4. 生成自愈后的完整正面姿态并映射至全局全景坐标
        fused_local, _ = self.biomech_engine.heal_occluded_pose(front_pose_local, back_pose_local)
        fused_orig = self._map_pose_to_original(fused_local, dual_frame.front_info)

        return DualPoseResult(
            frame_id=dual_frame.frame_id,
            front_pose_local=front_pose_local,
            back_pose_local=back_pose_local,
            front_pose_orig=front_pose_orig,
            back_pose_orig=back_pose_orig,
            fused_pose_local=fused_local,
            fused_pose_orig=fused_orig,
            biomechanics=biomech_res,
            timestamp_ms=dual_frame.timestamp_ms,
        )
