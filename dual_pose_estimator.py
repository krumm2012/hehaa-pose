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
    # 融合姿态（包含正面遮挡时自愈补全的关键点）
    fused_pose_local: Dict[str, Keypoint]
    # 生物力学分析结果
    biomechanics: DualViewBiomechanicsResult
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
                import coremltools as ct
                candidate_paths = [
                    model_path,
                    "yolo26m-pose.mlpackage",
                    "yolo26n-pose.mlpackage",
                    "models/yolo26m-pose.mlpackage",
                ]
                for cp in candidate_paths:
                    if cp and Path(cp).exists():
                        self.model = ct.models.MLModel(cp)
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

    def _predict_single_view(self, view_frame: np.ndarray) -> Dict[str, Keypoint]:
        """对单路裁剪视角画面进行姿态估计。"""
        if self.model is None or self.backend == "mock":
            return {}

        if self.backend == "ultralytics":
            results = self.model(view_frame, conf=self.conf_threshold, verbose=False)
            if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
                return {}
            # 取置信度最高的一个人体
            kp_data = results[0].keypoints.data.cpu().numpy() # [N, 17, 3] or [N, 17, 2]
            if len(kp_data) == 0:
                return {}
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
            # 将图像预处理为模型所需尺寸进行推理
            # 根据 yolo26m-pose 规范提取关键点
            try:
                # 简易通用 CoreML 预测包装
                import PIL.Image
                pil_img = PIL.Image.fromarray(cv2.cvtColor(view_frame, cv2.COLOR_BGR2RGB))
                out = self.model.predict({"image": pil_img})
                # 解析 CoreML 输出中的 keypoints
                # 如果是 yolo26 coreml 输出格式：
                if "keypoints" in out:
                    raw_kp = out["keypoints"]
                    return self.biomech_engine.parse_pose_dict(raw_kp)
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
        front_pose_local = self._predict_single_view(dual_frame.front_frame)
        back_pose_local = self._predict_single_view(dual_frame.back_frame)

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

        # 4. 生成自愈后的完整正面姿态
        fused_local, _ = self.biomech_engine.heal_occluded_pose(front_pose_local, back_pose_local)

        return DualPoseResult(
            frame_id=dual_frame.frame_id,
            front_pose_local=front_pose_local,
            back_pose_local=back_pose_local,
            front_pose_orig=front_pose_orig,
            back_pose_orig=back_pose_orig,
            fused_pose_local=fused_local,
            biomechanics=biomech_res,
            timestamp_ms=dual_frame.timestamp_ms,
        )
