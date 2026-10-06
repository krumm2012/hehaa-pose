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
from dataclasses import replace

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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


def measurement_points(points, info):
    """Undo ROI resize, preserving view handedness and observation metadata."""
    return {k: replace(v, x=v.x / info.scale_x, y=v.y / info.scale_y)
            for k, v in points.items() if v.observed}


PAIRED_KEYPOINT_NAMES: List[Tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
]

TORSO_KEYPOINT_PAIRS: List[Tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
]

LEG_KEYPOINT_PAIRS: List[Tuple[str, str]] = [
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]

ARM_KEYPOINT_PAIRS: List[Tuple[str, str]] = [
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
]


def swap_pose_pairs(pose: Dict[str, Keypoint], pairs: List[Tuple[str, str]]) -> None:
    """对姿态关键点字典中的指定成对解剖部位进行原位标签交换。"""
    for l_name, r_name in pairs:
        has_l = l_name in pose
        has_r = r_name in pose
        if has_l and has_r:
            pose[l_name], pose[r_name] = pose[r_name], pose[l_name]
        elif has_l:
            pose[r_name] = pose.pop(l_name)
        elif has_r:
            pose[l_name] = pose.pop(r_name)


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
    # 背面机位检测到的人脸眼睛隐私遮挡框 [(x1, y1, x2, y2), ...]
    back_view_eyes: List[Tuple[int, int, int, int]] = field(default_factory=list)


class DualPoseEstimator:
    """
    双视角姿态估计器（支持双实例并发推理与单实例兼容模式）
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        dominant_hand: str = "right",
        max_contact_distance: float = 200.0,
        backend: str = "auto",  # 'auto' | 'coreml' | 'ultralytics'
        concurrent: bool = True,
    ):
        self.conf_threshold = conf_threshold
        self.biomech_engine = DualViewBiomechanicsEngine(
            dominant_hand=dominant_hand,
            max_contact_distance=max_contact_distance,
            min_keypoint_conf=conf_threshold,
        )
        self.backend = backend
        self.concurrent = bool(concurrent)
        self.model_front = None
        self.model_back = None
        self.model = None  # 兼容原有单实例属性访问 (指向 model_front)
        self._pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="dual-pose")
            if self.concurrent
            else None
        )
        self._last_back_eyes: List[Tuple[int, int, int, int]] = []
        self._last_valid_back_pose: Dict[str, Keypoint] = {}
        self._back_missing_count: int = 0
        self._has_back_orientation_anchor: bool = False
        self._last_valid_front_pose: Dict[str, Keypoint] = {}
        self._front_missing_count: int = 0
        self._init_backend(model_path)

    def _init_backend(self, model_path: Optional[str]):
        """初始化姿态推理后端（优先 Core ML 双实例并发，回退至 Ultralytics）。"""
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
                        cfg = {
                            "pose_confidence_threshold": self.conf_threshold,
                            "pose_keypoint_confidence": self.conf_threshold * 0.7,
                            "pose_smoothing_enabled": False,
                        }
                        self.model_front = PoseEstimatorYOLO26(str(cp), cfg)
                        if self.concurrent:
                            try:
                                self.model_back = PoseEstimatorYOLO26(str(cp), cfg)
                                logger.info(f"Loaded CoreML dual-instance pose models (front & back) from {cp}")
                            except Exception as e_back:
                                logger.warning(
                                    f"Dual-instance back model creation failed, falling back to shared instance: {e_back}"
                                )
                                self.model_back = self.model_front
                        else:
                            self.model_back = self.model_front
                        self.model = self.model_front
                        self.backend = "coreml"
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
                        self.model_front = YOLO(cp)
                        if self.concurrent:
                            try:
                                self.model_back = YOLO(cp)
                            except Exception:
                                self.model_back = self.model_front
                        else:
                            self.model_back = self.model_front
                        self.model = self.model_front
                        self.backend = "ultralytics"
                        logger.info(f"Loaded Ultralytics pose model from {cp}")
                        return
                # 默认加载 yolov8n-pose.pt
                self.model_front = YOLO("yolov8n-pose.pt")
                self.model_back = YOLO("yolov8n-pose.pt") if self.concurrent else self.model_front
                self.model = self.model_front
                self.backend = "ultralytics"
                logger.info("Loaded default yolov8n-pose.pt backend")
                return
            except Exception as e:
                logger.warning(f"Ultralytics backend error: {e}")

        logger.warning("No ML pose model loaded. DualPoseEstimator running in dummy/pass-through mode.")
        self.backend = "mock"
        self.model_front = None
        self.model_back = None
        self.model = None

    def _extract_eye_boxes_from_kpts_list(
        self, kpts_list: List[Dict[str, Any]], h: int, w: int
    ) -> List[Tuple[int, int, int, int]]:
        """从 Core ML 关键点列表中提取人脸眼睛隐私遮挡区域（区域判断：仅针对下半部真实人脸）。"""
        boxes: List[Tuple[int, int, int, int]] = []
        for p in kpts_list:
            le = p.get("left_eye")
            re = p.get("right_eye")
            nose = p.get("nose")

            pts = [pt for pt in (le, re, nose) if pt is not None]
            if not pts:
                continue

            # 区域判断：背面视口中，真实人脸仅出现在画面下半部前景闯入区 (y > 0.40 * h)
            avg_y = sum(pt[1] for pt in pts) / len(pts)
            if avg_y <= 0.40 * h:
                continue

            eye_pts = [pt for pt in (le, re) if pt is not None]
            if not eye_pts:
                # 只有鼻子，根据鼻子估算眼睛位置
                cx, cy = int(nose[0]), int(nose[1] - 25)
                dist = 32.0
            else:
                xs = [pt[0] for pt in eye_pts]
                ys = [pt[1] for pt in eye_pts]
                cx = int(sum(xs) / len(xs))
                cy = int(sum(ys) / len(ys))
                if le is not None and re is not None:
                    dist = float(np.hypot(le[0] - re[0], le[1] - re[1]))
                elif nose is not None:
                    dist = float(np.hypot(eye_pts[0][0] - nose[0], eye_pts[0][1] - nose[1])) * 1.1
                else:
                    dist = 32.0

            bar_w = int(max(60.0, dist * 2.2))
            bar_h = int(max(16.0, dist * 0.7))
            x1 = max(0, cx - bar_w // 2)
            x2 = min(w, cx + bar_w // 2)
            y1 = max(0, cy - bar_h // 2)
            y2 = min(h, cy + bar_h // 2)
            boxes.append((x1, y1, x2, y2))
        return boxes

    def _extract_eye_boxes_from_kp_data(
        self, kp_data: np.ndarray, h: int, w: int
    ) -> List[Tuple[int, int, int, int]]:
        """从 Ultralytics 姿态输出中提取人脸眼睛隐私遮挡区域（区域判断：仅针对下半部真实人脸）。"""
        boxes: List[Tuple[int, int, int, int]] = []
        for p in kp_data:
            conf_th = self.conf_threshold
            has_nose = p.shape[0] > 0 and (p.shape[1] <= 2 or p[0, 2] >= conf_th)
            has_le = p.shape[0] > 1 and (p.shape[1] <= 2 or p[1, 2] >= conf_th)
            has_re = p.shape[0] > 2 and (p.shape[1] <= 2 or p[2, 2] >= conf_th)

            nose = (float(p[0, 0]), float(p[0, 1])) if has_nose else None
            le = (float(p[1, 0]), float(p[1, 1])) if has_le else None
            re = (float(p[2, 0]), float(p[2, 1])) if has_re else None

            pts = [pt for pt in (le, re, nose) if pt is not None]
            if not pts:
                continue

            # 区域判断：背面视口中，真实人脸仅出现在画面下半部前景闯入区 (y > 0.40 * h)
            avg_y = sum(pt[1] for pt in pts) / len(pts)
            if avg_y <= 0.40 * h:
                continue

            eye_pts = [pt for pt in (le, re) if pt is not None]
            if not eye_pts:
                cx, cy = int(nose[0]), int(nose[1] - 25)
                dist = 32.0
            else:
                xs = [pt[0] for pt in eye_pts]
                ys = [pt[1] for pt in eye_pts]
                cx = int(sum(xs) / len(xs))
                cy = int(sum(ys) / len(ys))
                if le is not None and re is not None:
                    dist = float(np.hypot(le[0] - re[0], le[1] - re[1]))
                elif nose is not None:
                    dist = float(np.hypot(eye_pts[0][0] - nose[0], eye_pts[0][1] - nose[1])) * 1.1
                else:
                    dist = 32.0

            bar_w = int(max(60.0, dist * 2.2))
            bar_h = int(max(16.0, dist * 0.7))
            x1 = max(0, cx - bar_w // 2)
            x2 = min(w, cx + bar_w // 2)
            y1 = max(0, cy - bar_h // 2)
            y2 = min(h, cy + bar_h // 2)
            boxes.append((x1, y1, x2, y2))
        return boxes

    def _select_front_candidate_ultralytics(
        self,
        kp_data: np.ndarray,
        h: int,
        w: int,
        crop_info: Optional[DualViewCropInfo] = None,
        is_point_in_mirror_fn: Optional[Callable[[float, float], bool]] = None,
    ) -> Optional[np.ndarray]:
        """
        正面机位候选人筛选：在多候选人场景中准确识别球场真实前景选手，坚决排除后墙镜面虚影。
        综合多边形几何碰撞、画面垂直深度、人体尺度及帧间时序连续性进行动态评分。
        """
        if len(kp_data) == 0:
            return None

        scored_candidates = []
        for p in kp_data:
            valid_mask = p[:, 2] >= self.conf_threshold if p.shape[1] > 2 else np.ones(len(p), dtype=bool)
            valid_pts = p[valid_mask]
            if len(valid_pts) < 4:
                continue

            min_x = float(np.min(valid_pts[:, 0]))
            max_x = float(np.max(valid_pts[:, 0]))
            min_y = float(np.min(valid_pts[:, 1]))
            max_y = float(np.max(valid_pts[:, 1]))
            cand_h = max_y - min_y
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0

            # 1. 镜面多边形与几何碰撞检测 (最高优先级)
            is_mirror = False
            if crop_info is not None and is_point_in_mirror_fn is not None:
                orig_cx, orig_cy = crop_info.map_to_original(cx, cy)
                orig_bx, orig_by = crop_info.map_to_original(cx, max_y)
                in_mirror_center = bool(is_point_in_mirror_fn(orig_cx, orig_cy))
                in_mirror_bottom = bool(is_point_in_mirror_fn(orig_bx, orig_by))
                # 真实选手的双脚站在球场地面上（底部在镜面多边形下方），绝不是后墙镜中虚影；
                # 只有当底部双脚也完全落在镜面内部，或中心在镜面内且人像位于局部视口上半部/小尺度时，才是后墙镜中倒影。
                if in_mirror_bottom or (in_mirror_center and cy < 0.45 * h and max_y < 0.55 * h):
                    is_mirror = True
            elif cy < 0.40 * h and max_y < 0.50 * h:
                # 当无原图多边形回调时，退化为局部视口顶部禁区几何判断
                is_mirror = True

            score = 0.0
            if is_mirror:
                score -= 1000.0  # 镜面虚影重罚，严禁作为正面选手

            # 2. 尺度特征：前景选手更靠近摄像头，像素高度更高
            score += (cand_h / max(1.0, float(h))) * 120.0

            # 3. 接地特征：真实选手双脚踩在球场地面上，在画面的下半部 (y 坐标更大)
            score += (max_y / max(1.0, float(h))) * 100.0

            # 4. 置信度特征
            mean_conf = float(np.mean(valid_pts[:, 2])) if p.shape[1] > 2 else 0.85
            score += mean_conf * 25.0

            # 5. 帧间时序连续性（平滑追踪）
            if self._last_valid_front_pose:
                prev_xs = [kp.x for kp in self._last_valid_front_pose.values()]
                prev_ys = [kp.y for kp in self._last_valid_front_pose.values()]
                if prev_xs and prev_ys:
                    pcx = (min(prev_xs) + max(prev_xs)) / 2.0
                    pcy = (min(prev_ys) + max(prev_ys)) / 2.0
                    dist = float(np.hypot(cx - pcx, cy - pcy))
                    norm_dist = dist / max(1.0, float(h))
                    if norm_dist < 0.25:
                        score += 30.0 * (1.0 - norm_dist / 0.25)
                    elif norm_dist > 0.40:
                        score -= 50.0 * norm_dist

            scored_candidates.append((score, p))

        if not scored_candidates:
            return None

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_candidate = scored_candidates[0]
        # 若最高分仍低于 -500 分，说明所有候选均在镜面内部，拒绝选用镜面虚影
        if best_score < -500.0:
            return None
        return best_candidate

    def _select_front_candidate_coreml(
        self,
        kpts_list: List[Dict[str, Any]],
        h: int,
        w: int,
        crop_info: Optional[DualViewCropInfo] = None,
        is_point_in_mirror_fn: Optional[Callable[[float, float], bool]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Core ML 正面机位候选人筛选：在多候选人场景中准确识别球场真实前景选手，坚决排除后墙镜面虚影。
        """
        if not kpts_list:
            return None

        scored_candidates = []
        for p in kpts_list:
            pts = [pt for pt in p.values() if pt is not None]
            if len(pts) < 4:
                continue

            xs = [pt[0] for pt in pts]
            ys = [pt[1] for pt in pts]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            cand_h = max_y - min_y
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0

            is_mirror = False
            if crop_info is not None and is_point_in_mirror_fn is not None:
                orig_cx, orig_cy = crop_info.map_to_original(cx, cy)
                orig_bx, orig_by = crop_info.map_to_original(cx, max_y)
                in_mirror_center = bool(is_point_in_mirror_fn(orig_cx, orig_cy))
                in_mirror_bottom = bool(is_point_in_mirror_fn(orig_bx, orig_by))
                # 真实选手的双脚站在球场地面上（底部在镜面多边形下方），绝不是后墙镜中虚影；
                # 只有当底部双脚也完全落在镜面内部，或中心在镜面内且人像位于局部视口上半部/小尺度时，才是后墙镜中倒影。
                if in_mirror_bottom or (in_mirror_center and cy < 0.45 * h and max_y < 0.55 * h):
                    is_mirror = True
            elif cy < 0.40 * h and max_y < 0.50 * h:
                is_mirror = True

            score = 0.0
            if is_mirror:
                score -= 1000.0

            score += (cand_h / max(1.0, float(h))) * 120.0
            score += (max_y / max(1.0, float(h))) * 100.0
            score += 0.85 * 25.0

            if self._last_valid_front_pose:
                prev_xs = [kp.x for kp in self._last_valid_front_pose.values()]
                prev_ys = [kp.y for kp in self._last_valid_front_pose.values()]
                if prev_xs and prev_ys:
                    pcx = (min(prev_xs) + max(prev_xs)) / 2.0
                    pcy = (min(prev_ys) + max(prev_ys)) / 2.0
                    dist = float(np.hypot(cx - pcx, cy - pcy))
                    norm_dist = dist / max(1.0, float(h))
                    if norm_dist < 0.25:
                        score += 30.0 * (1.0 - norm_dist / 0.25)
                    elif norm_dist > 0.40:
                        score -= 50.0 * norm_dist

            scored_candidates.append((score, p))

        if not scored_candidates:
            return None

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_candidate = scored_candidates[0]
        if best_score < -500.0:
            return None
        return best_candidate

    def _cached_pose(self, view, crop_info):
        """Keep held points fixed in source coordinates across moving crops."""
        points = getattr(self, '_last_valid_'+view+'_pose')
        previous_crop = getattr(self, '_last_valid_'+view+'_crop', None)
        if (previous_crop is None) != (crop_info is None):
            return {}  # Unknown transform cannot be reconstructed safely.
        result = {}
        for name, point in points.items():
            if previous_crop is not None:
                x,y = previous_crop.map_to_original(point.x,point.y)
                x,y = crop_info.map_from_original(x,y)
            else:
                x,y = point.x,point.y
            result[name] = replace(point,x=x,y=y,observed=False)
        return result

    def _select_back_candidate(self, candidates, crop_info, is_point_in_mirror_fn):
        """Gate fresh torso evidence by mirror geometry and source-space continuity."""
        ranked = []
        previous = getattr(self, '_back_identity_anchor', None)
        if self._back_missing_count >= 3:
            previous = None
        for index, pose in enumerate(candidates):
            shoulders = [pose.get(name) for name in ('left_shoulder','right_shoulder')]
            if any(p is None or len(p)<3 or p[2]<.5
                   or not all(math.isfinite(float(v)) for v in p[:3]) for p in shoulders):
                continue
            hips = [pose.get(name) for name in ('left_hip','right_hip')]
            # A partial foreground face/shoulder detection may also lie inside
            # the mirror polygon. Require an independently observed torso.
            if any(p is None or len(p)<3 or p[2]<.5
                   or not all(math.isfinite(float(v)) for v in p[:3]) for p in hips):
                continue
            project = crop_info.map_to_original if crop_info else lambda x,y:(x,y)
            points = [project(p[0],p[1]) for p in shoulders]
            center = tuple(sum(p[i] for p in points)/2 for i in (0,1))
            hip_points = [project(p[0], p[1]) for p in hips]
            hip_center = tuple(sum(p[i] for p in hip_points)/2 for i in (0, 1))
            # Identity continuity uses torso length: shoulder width collapses
            # during a side-on turn and a fixed pixel gate changes with input
            # resolution. This does not grant shoulder-angle eligibility.
            torso_length = math.dist(center, hip_center)
            span = max(math.dist(*points), torso_length)
            if torso_length <= 1e-6:
                continue
            if is_point_in_mirror_fn is not None:
                if not all(is_point_in_mirror_fn(*point) for point in points):
                    continue
                hips = [pose.get(n) for n in ('left_hip','right_hip')]
                if any(p is not None and len(p)>2 and p[2]>=.5
                       and not is_point_in_mirror_fn(*project(p[0],p[1])) for p in hips):
                    continue
            else:
                # Without mirror geometry, retain conservative face rejection.
                eyes = [pose.get(n) for n in ('left_eye','right_eye')]
                if all(p is not None and len(p)>2 and p[2]>=.7 for p in eyes):
                    continue
            distance = 0.0
            if previous is not None:
                old_center, old_span = previous
                distance = math.dist(center,old_center)/max(span,old_span)
                if distance > 2 or not .5 <= span/old_span <= 2:
                    continue
            ranked.append((distance if previous else center[1], index, center, span))
        if not ranked:
            return None
        _, index, center, span = min(ranked)
        self._back_identity_anchor = (center,span)
        return index

    @staticmethod
    def _get_pose_centroid(pose: Dict[str, Keypoint]) -> Optional[Tuple[float, float]]:
        """计算姿态躯干核心质心（去平移参考锚点）。"""
        pts = []
        for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            kp = pose.get(k)
            if kp is not None and getattr(kp, "conf", 0.0) >= 0.3:
                pts.append((kp.x, kp.y))
        if not pts:
            for kp in pose.values():
                if kp is not None and getattr(kp, "conf", 0.0) >= 0.3:
                    pts.append((kp.x, kp.y))
        if not pts:
            return None
        return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)

    @staticmethod
    def _compute_bipartite_costs(
        curr_pose: Dict[str, Keypoint],
        prev_pose: Dict[str, Keypoint],
        curr_c: Tuple[float, float],
        prev_c: Tuple[float, float],
        pairs: List[Tuple[str, str]],
    ) -> Tuple[float, float, int]:
        """计算躯干质心去平移后的相对坐标二分图匹配代价 (正向代价 vs 左右对调代价)。"""
        cost_norm = 0.0
        cost_swap = 0.0
        valid_count = 0
        for l_name, r_name in pairs:
            if (
                l_name in curr_pose
                and r_name in curr_pose
                and l_name in prev_pose
                and r_name in prev_pose
            ):
                c_l = (curr_pose[l_name].x - curr_c[0], curr_pose[l_name].y - curr_c[1])
                c_r = (curr_pose[r_name].x - curr_c[0], curr_pose[r_name].y - curr_c[1])
                p_l = (prev_pose[l_name].x - prev_c[0], prev_pose[l_name].y - prev_c[1])
                p_r = (prev_pose[r_name].x - prev_c[0], prev_pose[r_name].y - prev_c[1])

                d_norm = math.hypot(c_l[0] - p_l[0], c_l[1] - p_l[1]) + math.hypot(
                    c_r[0] - p_r[0], c_r[1] - p_r[1]
                )
                d_swap = math.hypot(c_l[0] - p_r[0], c_l[1] - p_r[1]) + math.hypot(
                    c_r[0] - p_l[0], c_r[1] - p_l[1]
                )
                cost_norm += d_norm
                cost_swap += d_swap
                valid_count += 1
        return cost_norm, cost_swap, valid_count

    def _filter_backview_temporal_swap(
        self,
        curr_pose: Dict[str, Keypoint],
    ) -> Optional[str]:
        """
        时序二分图抗翻转滤波器（Temporal Anti-Swap Filter）：
        对背面机位（镜面视口）姿态进行躯干去平移的局部相对时序二分图匹配。
        杜绝 2D 网络在背影/侧身及双手反拍胸前交叉时发生的 180° 解剖左右关节颠倒问题。
        """
        if not self._last_valid_back_pose or not curr_pose:
            return None

        prev_pose = self._last_valid_back_pose
        curr_c = self._get_pose_centroid(curr_pose)
        prev_c = self._get_pose_centroid(prev_pose)
        if not curr_c or not prev_c:
            return None

        t_norm, t_swap, t_cnt = self._compute_bipartite_costs(
            curr_pose, prev_pose, curr_c, prev_c, TORSO_KEYPOINT_PAIRS
        )
        all_norm, all_swap, all_cnt = self._compute_bipartite_costs(
            curr_pose, prev_pose, curr_c, prev_c, PAIRED_KEYPOINT_NAMES
        )

        # 1. 全身 180° 朝向翻转判定：总体代价显著偏向对调，且躯干无正向阻抗
        if (
            all_cnt >= 2
            and all_norm > 25.0
            and all_swap < 0.55 * all_norm
            and (t_cnt == 0 or t_swap <= t_norm * 1.05)
        ):
            swap_pose_pairs(curr_pose, PAIRED_KEYPOINT_NAMES)
            logger.info(
                f"Back view temporal whole-body swap detected and corrected: "
                f"norm={all_norm:.1f}, swap={all_swap:.1f}, ratio={all_swap / all_norm:.2f}"
            )
            return "whole_body"

        # 2. 下肢局部翻转判定：躯干朝向正常，但随挥下肢跨步导致下肢局部翻转
        l_norm, l_swap, l_cnt = self._compute_bipartite_costs(
            curr_pose, prev_pose, curr_c, prev_c, LEG_KEYPOINT_PAIRS
        )
        if (
            l_cnt >= 1
            and l_norm > 30.0
            and l_swap < 0.50 * l_norm
            and (t_cnt == 0 or t_norm <= t_swap)
        ):
            swap_pose_pairs(curr_pose, LEG_KEYPOINT_PAIRS)
            logger.info(
                f"Back view temporal leg swap detected and corrected: "
                f"norm={l_norm:.1f}, swap={l_swap:.1f}, ratio={l_swap / l_norm:.2f}"
            )
            return "legs"

        return None

    def _verify_cross_view_cold_start(
        self,
        front_pose_orig: Dict[str, Keypoint],
        back_pose_orig: Dict[str, Keypoint],
        back_pose_local: Dict[str, Keypoint],
    ) -> Optional[str]:
        """
        跨视角解剖朝向冷启动校验 (Level 2)：
        利用具备人脸五官锚定的正面视角横轴朝向符号，防范冷启动或长丢帧后的初始镜像倒置。
        """
        if not front_pose_orig or not back_pose_orig:
            return None

        f_ls, f_rs = front_pose_orig.get("left_shoulder"), front_pose_orig.get("right_shoulder")
        b_ls, b_rs = back_pose_orig.get("left_shoulder"), back_pose_orig.get("right_shoulder")
        f_lh, f_rh = front_pose_orig.get("left_hip"), front_pose_orig.get("right_hip")
        b_lh, b_rh = back_pose_orig.get("left_hip"), back_pose_orig.get("right_hip")

        sh_disagree = False
        if f_ls and f_rs and b_ls and b_rs:
            f_dx = f_ls.x - f_rs.x
            b_dx = b_ls.x - b_rs.x
            if abs(f_dx) > 30.0 and abs(b_dx) > 25.0 and (f_dx * b_dx < 0):
                sh_disagree = True

        hip_disagree = False
        if f_lh and f_rh and b_lh and b_rh:
            f_dx = f_lh.x - f_rh.x
            b_dx = b_lh.x - b_rh.x
            if abs(f_dx) > 25.0 and abs(b_dx) > 20.0 and (f_dx * b_dx < 0):
                hip_disagree = True

        if sh_disagree or hip_disagree:
            swap_pose_pairs(back_pose_orig, PAIRED_KEYPOINT_NAMES)
            swap_pose_pairs(back_pose_local, PAIRED_KEYPOINT_NAMES)
            logger.info("Cold-start cross-view orientation mismatch: inverted back view corrected to match front view.")
            return "cold_start_whole_body"

        return None

    def _predict_single_view(
        self,
        view_frame: np.ndarray,
        is_back_view: bool = False,
        model: Optional[Any] = None,
        crop_info: Optional[DualViewCropInfo] = None,
        is_point_in_mirror_fn: Optional[Callable[[float, float], bool]] = None,
    ) -> Dict[str, Keypoint]:
        """对单路裁剪视角画面进行姿态估计。"""
        active_model = model or (self.model_back if is_back_view and self.model_back is not None else self.model_front) or self.model
        if active_model is None or self.backend == "mock":
            return {}

        h, w = view_frame.shape[:2]

        if self.backend == "ultralytics":
            results = active_model(view_frame, conf=self.conf_threshold, verbose=False)
            if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
                if is_back_view and self._last_valid_back_pose and self._back_missing_count < 3:
                    self._back_missing_count += 1
                    return self._cached_pose('back', crop_info)
                elif not is_back_view and self._last_valid_front_pose and self._front_missing_count < 3:
                    self._front_missing_count += 1
                    return self._cached_pose('front', crop_info)
                return {}
            kp_data = results[0].keypoints.data.cpu().numpy()  # [N, 17, 3] or [N, 17, 2]
            if len(kp_data) == 0:
                if is_back_view and self._last_valid_back_pose and self._back_missing_count < 3:
                    self._back_missing_count += 1
                    return self._cached_pose('back', crop_info)
                elif not is_back_view and self._last_valid_front_pose and self._front_missing_count < 3:
                    self._front_missing_count += 1
                    return self._cached_pose('front', crop_info)
                return {}

            best_person = None
            if is_back_view:
                self._last_back_eyes = self._extract_eye_boxes_from_kp_data(kp_data, h, w)
                candidate_maps = [{name: tuple(person[i]) for i,name in COCO_KEYPOINTS.items() if i < len(person)}
                                  for person in kp_data]
                index = self._select_back_candidate(candidate_maps, crop_info, is_point_in_mirror_fn)
                if index is not None:
                    best_person = kp_data[index]
            else:
                best_person = self._select_front_candidate_ultralytics(
                    kp_data, h, w, crop_info=crop_info, is_point_in_mirror_fn=is_point_in_mirror_fn
                )

            parsed = {}
            if best_person is not None:
                for idx, name in COCO_KEYPOINTS.items():
                    x = float(best_person[idx, 0])
                    y = float(best_person[idx, 1])
                    conf = float(best_person[idx, 2]) if best_person.shape[1] > 2 else 1.0
                    if conf >= self.conf_threshold:
                        parsed[name] = Keypoint(x=x, y=y, conf=conf, source_frame_id=getattr(self, "_source_frame_id", None), confidence_source="model")

            if is_back_view:
                if parsed:
                    self._filter_backview_temporal_swap(parsed)
                    self._last_valid_back_pose = parsed
                    self._last_valid_back_crop = crop_info
                    self._back_missing_count = 0
                    return parsed
                elif self._last_valid_back_pose and self._back_missing_count < 3:
                    self._back_missing_count += 1
                    return self._cached_pose('back', crop_info)
                else:
                    self._back_missing_count += 1
                    return {}
            else:
                if parsed:
                    self._last_valid_front_pose = parsed
                    self._last_valid_front_crop = crop_info
                    self._front_missing_count = 0
                    return parsed
                elif self._last_valid_front_pose and self._front_missing_count < 3:
                    self._front_missing_count += 1
                    return self._cached_pose('front', crop_info)
                else:
                    self._front_missing_count += 1
                    return {}

        # Core ML 分支
        if self.backend == "coreml":
            try:
                predict = getattr(active_model, "get_keypoints_with_confidence", active_model.get_keypoints)
                kpts_list = predict(view_frame)
                if not kpts_list:
                    if is_back_view and self._last_valid_back_pose and self._back_missing_count < 3:
                        self._back_missing_count += 1
                        return self._cached_pose('back', crop_info)
                    elif not is_back_view and self._last_valid_front_pose and self._front_missing_count < 3:
                        self._front_missing_count += 1
                        return self._cached_pose('front', crop_info)
                    return {}

                best = None
                if is_back_view:
                    # 背面机位提取人脸眼睛遮挡区域（区域判断：仅针对下部真实人脸）
                    self._last_back_eyes = self._extract_eye_boxes_from_kpts_list(kpts_list, h, w)
                    index = self._select_back_candidate(kpts_list, crop_info, is_point_in_mirror_fn)
                    if index is not None:
                        best = kpts_list[index]
                else:
                    best = self._select_front_candidate_coreml(
                        kpts_list, h, w, crop_info=crop_info, is_point_in_mirror_fn=is_point_in_mirror_fn
                    )

                parsed = {}
                if best is not None:
                    for name, pt in best.items():
                        if pt is not None:
                            parsed[name] = Keypoint(x=float(pt[0]), y=float(pt[1]), conf=float(pt[2]) if len(pt) > 2 else 0.0, source_frame_id=getattr(self, "_source_frame_id", None), confidence_source="model" if len(pt) > 2 else "unavailable")

                if is_back_view:
                    if parsed:
                        self._filter_backview_temporal_swap(parsed)
                        self._last_valid_back_pose = parsed
                        self._last_valid_back_crop = crop_info
                        self._back_missing_count = 0
                        return parsed
                    elif self._last_valid_back_pose and self._back_missing_count < 3:
                        self._back_missing_count += 1
                        return self._cached_pose('back', crop_info)
                    else:
                        self._back_missing_count += 1
                        return {}
                else:
                    if parsed:
                        self._last_valid_front_pose = parsed
                        self._last_valid_front_crop = crop_info
                        self._front_missing_count = 0
                        return parsed
                    elif self._last_valid_front_pose and self._front_missing_count < 3:
                        self._front_missing_count += 1
                        return self._cached_pose('front', crop_info)
                    else:
                        self._front_missing_count += 1
                        return {}

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
                observed=kp.observed, source_frame_id=kp.source_frame_id, confidence_source=kp.confidence_source,
            )
        return mapped

    def estimate_dual_pose(
        self,
        dual_frame: DualViewFrame,
        ball_pos: Optional[Tuple[float, float]] = None,
    ) -> DualPoseResult:
        """
        双视角主估计流程：
        1. 分别在正面和背面提取姿态（支持独立双实例并发）
        2. 坐标转换映射至原始全局像素
        3. 进行生物力学融合、抗侧身塌陷及遮挡自愈
        """
        # 1. 独立并发估计
        self._source_frame_id = dual_frame.frame_id
        self._last_back_eyes = []
        is_mirror_fn = getattr(dual_frame, "is_point_in_mirror_fn", None)
        if (
            self.concurrent
            and self._pool is not None
            and self.model_front is not None
            and self.model_back is not None
            and self.model_front is not self.model_back
        ):
            fut_front = self._pool.submit(
                self._predict_single_view,
                dual_frame.front_frame,
                False,
                self.model_front,
                dual_frame.front_info,
                is_mirror_fn,
            )
            fut_back = self._pool.submit(
                self._predict_single_view,
                dual_frame.back_frame,
                True,
                self.model_back,
                dual_frame.back_info,
                is_mirror_fn,
            )
            front_pose_local = fut_front.result()
            back_pose_local = fut_back.result()
        else:
            front_pose_local = self._predict_single_view(
                dual_frame.front_frame,
                is_back_view=False,
                model=self.model_front,
                crop_info=dual_frame.front_info,
                is_point_in_mirror_fn=is_mirror_fn,
            )
            back_pose_local = self._predict_single_view(
                dual_frame.back_frame,
                is_back_view=True,
                model=self.model_back,
                crop_info=dual_frame.back_info,
                is_point_in_mirror_fn=is_mirror_fn,
            )
        back_eyes = list(self._last_back_eyes)

        # 2. 映射回原图坐标
        front_pose_orig = self._map_pose_to_original(front_pose_local, dual_frame.front_info)
        back_pose_orig = self._map_pose_to_original(back_pose_local, dual_frame.back_info)

        # 2.5 跨视角解剖朝向冷启动与长丢帧重对齐保护 (Level 2)
        if not self._has_back_orientation_anchor or self._back_missing_count >= 3:
            if back_pose_local and front_pose_orig and back_pose_orig:
                if self._verify_cross_view_cold_start(front_pose_orig, back_pose_orig, back_pose_local):
                    self._last_valid_back_pose = back_pose_local
                self._has_back_orientation_anchor = True

        # 3. 生物力学融合计算
        # 如果提供了原图 ball_pos，将其映射至正面机位坐标用于击球手与触球间距分析
        front_ball_pos = None
        if ball_pos is not None:
            bx, by = dual_frame.front_info.map_from_original(ball_pos[0], ball_pos[1])
            front_ball_pos = (bx, by)

        # Undo independent ROI resizing while retaining each view's handedness.
        # Mirror reflection and crop translation do not affect widths/ratios.
        measurement_ball = (front_ball_pos[0] / dual_frame.front_info.scale_x,
                            front_ball_pos[1] / dual_frame.front_info.scale_y) if front_ball_pos else None
        biomech_res = self.biomech_engine.calculate_dual_biomechanics(
            front_pose_raw=measurement_points(front_pose_local, dual_frame.front_info),
            back_pose_raw=measurement_points(back_pose_local, dual_frame.back_info),
            ball_pos=measurement_ball,
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
            back_view_eyes=back_eyes,
        )

    def reset(self):
        """重置内部姿态追踪与时序平滑状态。"""
        self._last_back_eyes = []
        self._back_identity_anchor = None
        self._last_valid_back_pose = {}
        self._back_missing_count = 0
        self._has_back_orientation_anchor = False
        self._last_valid_front_pose = {}
        self._front_missing_count = 0

    def close(self):
        """关闭内部并发线程池，释放系统资源。"""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
