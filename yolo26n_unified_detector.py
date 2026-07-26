#!/usr/bin/env python3
"""
YOLO26n Core ML 统一检测器
同时检测球和球拍，简化系统架构
"""

import cv2
import numpy as np
from PIL import Image
import time
from collections import deque

try:
    import coremltools as ct
except ModuleNotFoundError:
    ct = None

from ball_candidate_selector import select_ball_candidate
from ball_track_selector import BallTrackSelector
from racket_candidate_selector import racket_center, select_racket_candidate
from static_ball_filter import StaticBallFilter


class YOLO26nUnifiedDetector:
    """YOLO26n Core ML 统一检测器 - 同时检测球和球拍"""
    
    def __init__(self, model_path, config, roi_manager=None):
        """
        初始化统一检测器
        
        Args:
            model_path: Core ML 模型路径
            config: 配置字典
            roi_manager: ROI 管理器（可选）
        """
        if ct is None:
            raise RuntimeError("coremltools is required to initialize YOLO26nUnifiedDetector")

        print(f"🚀 初始化 YOLO26n 统一检测器...")
        
        self.config = config
        self.roi_manager = roi_manager
        
        # 获取计算单元配置
        compute_units_str = config.get('compute_units', 'ALL')
        compute_units_map = {
            'CPU': ct.ComputeUnit.CPU_ONLY,
            'GPU': ct.ComputeUnit.CPU_AND_GPU,
            'ANE': ct.ComputeUnit.CPU_AND_NE,
            'ALL': ct.ComputeUnit.ALL
        }
        compute_units = compute_units_map.get(compute_units_str.upper(), ct.ComputeUnit.ALL)
        
        # 加载 Core ML 模型
        print(f"   加载模型: {model_path}")
        print(f"   计算单元: {compute_units_str}")
        self.model = ct.models.MLModel(model_path, compute_units=compute_units)
        
        # 获取模型输入尺寸
        spec = self.model.get_spec()
        self._coreml_input_names = {inp.name for inp in spec.description.input}
        self._coreml_output_names = {out.name for out in spec.description.output}
        input_desc = spec.description.input[0]
        if hasattr(input_desc.type, 'imageType'):
            self.input_height = input_desc.type.imageType.height
            self.input_width = input_desc.type.imageType.width
        else:
            self.input_height = 640
            self.input_width = 640
        
        print(f"   模型输入尺寸: {self.input_width}x{self.input_height}")
        
        # COCO 类别 ID
        self.ball_class_id = 32  # sports ball
        self.racket_class_id = 38  # tennis racket
        
        # 置信度阈值
        self.ball_conf_threshold = config.get('ball_confidence_threshold', 0.02)
        self.racket_conf_threshold = config.get('racket_confidence_threshold', 0.3)
        self.coreml_iou_threshold = float(config.get('coreml_iou_threshold', 0.45))
        self.coreml_confidence_threshold = float(
            config.get('coreml_confidence_threshold', min(self.ball_conf_threshold, self.racket_conf_threshold))
        )
        
        print(f"   球检测阈值: {self.ball_conf_threshold}")
        print(f"   球拍检测阈值: {self.racket_conf_threshold}")
        
        # 球追踪历史（用于过滤静态球）
        self.ball_history = deque(maxlen=10)
        self.racket_history = deque(maxlen=10)
        self.static_threshold = config.get('static_ball_movement_threshold_px', 6)
        self.static_ball_filter = StaticBallFilter(config)
        self.ball_track_selector = BallTrackSelector(config)
        
        # 性能统计
        self.detection_times = deque(maxlen=100)
        self.last_ball_diagnostics = {}
        
        print("✅ YOLO26n 统一检测器初始化完成")

    def get_last_ball_diagnostics(self):
        """Return lightweight diagnostics for the last ball-selection step."""
        return dict(self.last_ball_diagnostics) if isinstance(self.last_ball_diagnostics, dict) else {}
    
    def detect_unified(self, frame):
        """
        统一检测球和球拍
        
        Args:
            frame: 输入图像 (BGR)
            
        Returns:
            tuple: (ball_detections, racket_detections, inference_time)
        """
        # 保存原始尺寸
        self.original_height, self.original_width = frame.shape[:2]
        
        # 预处理
        input_image = self._preprocess(frame)
        
        # 推理
        start_time = time.time()
        predict_inputs = {'image': input_image}
        if 'iouThreshold' in self._coreml_input_names:
            predict_inputs['iouThreshold'] = self.coreml_iou_threshold
        if 'confidenceThreshold' in self._coreml_input_names:
            predict_inputs['confidenceThreshold'] = self.coreml_confidence_threshold
        predictions = self.model.predict(predict_inputs)
        inference_time = time.time() - start_time
        
        # 记录性能
        self.detection_times.append(inference_time)
        
        # 解析结果
        ball_detections, racket_detections = self._parse_predictions(predictions)
        
        # 在已有候选中选择真实运动球/主拍；不增加模型推理，只做轻量距离打分。
        ball_detections = self._filter_static_balls(ball_detections, racket_detections)
        racket_detections = self._select_primary_racket(racket_detections, ball_detections)
        
        return ball_detections, racket_detections, inference_time
    
    def _preprocess(self, frame):
        """预处理图像"""
        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 缩放到模型输入尺寸
        resized = cv2.resize(rgb_frame, (self.input_width, self.input_height))
        
        # 转换为 PIL Image
        pil_image = Image.fromarray(resized)
        
        return pil_image
    
    def _parse_predictions(self, predictions):
        """解析模型输出"""
        ball_detections = []
        racket_detections = []

        # YOLO26n Core ML 输出格式: var_1441 [1, 300, 6]
        # 格式: [x_center, y_center, width, height, confidence, class_id]
        if 'var_1441' in predictions:
            output = predictions['var_1441']
            
            # 移除 batch 维度
            if len(output.shape) == 3:
                detections = output[0]
            else:
                detections = output
            
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                cls_id = int(cls)
                
                # 计算原始尺寸下的坐标
                # 注意：Core ML 输出通常是针对 640x640 的像素坐标
                scale_x = self.original_width / self.input_width
                scale_y = self.original_height / self.input_height
                
                real_x1 = max(0, min(float(x1) * scale_x, self.original_width))
                real_y1 = max(0, min(float(y1) * scale_y, self.original_height))
                real_x2 = max(0, min(float(x2) * scale_x, self.original_width))
                real_y2 = max(0, min(float(y2) * scale_y, self.original_height))
                
                box = [real_x1, real_y1, real_x2, real_y2]
                
                # 球检测
                if cls_id == self.ball_class_id and conf >= self.ball_conf_threshold:
                    ball_detections.append({
                        'position': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],  # 中心点
                        'box': box,
                        'confidence': float(conf),
                        'radius': int((box[2] - box[0]) / 2)
                    })
                
                # 球拍检测
                elif cls_id == self.racket_class_id and conf >= self.racket_conf_threshold:
                    # 计算边界框面积
                    box_width = box[2] - box[0]
                    box_height = box[3] - box[1]
                    box_area = box_width * box_height
                    
                    # 过滤过大的边界框（可能是误检）
                    # 修正后的过滤逻辑：球拍面积通常较小
                    max_area = (self.original_width * self.original_height) / 10 # 缩小范围到 1/10
                    min_area = 500  # 缩小最小面积
                    
                    # 宽高比检查（球拍通常是长条形）
                    aspect_ratio = box_height / max(box_width, 1)
                    min_aspect_ratio = 0.3  # 扩大范围
                    max_aspect_ratio = 10.0  # 扩大范围
                    
                    if (min_area <= box_area <= max_area and 
                        min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                        racket_detections.append({
                            'box': [int(b) for b in box],
                            'confidence': float(conf),
                            'class_name': 'tennis racket',
                            'area': int(box_area),
                            'aspect_ratio': round(aspect_ratio, 2)
                        })
        # 兼容带 NMS 的 CoreML 检测输出（坐标 + 80类置信）
        elif 'coordinates' in predictions and 'confidence' in predictions:
            coordinates = np.asarray(predictions['coordinates'])
            confidence = np.asarray(predictions['confidence'])
            if coordinates.ndim == 1:
                coordinates = coordinates.reshape(1, -1)
            if confidence.ndim == 1:
                confidence = confidence.reshape(1, -1)
            num_det = min(coordinates.shape[0], confidence.shape[0])
            for i in range(num_det):
                coord = coordinates[i]
                cls_conf = confidence[i]
                if coord.shape[0] < 4 or cls_conf.shape[0] <= max(self.ball_class_id, self.racket_class_id):
                    continue

                # coordinates is (xc, yc, w, h), usually normalized [0,1].
                xc, yc, bw, bh = float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])
                if max(abs(xc), abs(yc), abs(bw), abs(bh)) <= 2.0:
                    xc *= self.original_width
                    yc *= self.original_height
                    bw *= self.original_width
                    bh *= self.original_height
                x1 = max(0.0, min(self.original_width, xc - bw / 2.0))
                y1 = max(0.0, min(self.original_height, yc - bh / 2.0))
                x2 = max(0.0, min(self.original_width, xc + bw / 2.0))
                y2 = max(0.0, min(self.original_height, yc + bh / 2.0))
                box = [x1, y1, x2, y2]

                ball_conf = float(cls_conf[self.ball_class_id])
                if ball_conf >= self.ball_conf_threshold:
                    ball_detections.append({
                        'position': [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0],
                        'box': box,
                        'confidence': ball_conf,
                        'radius': max(1, int((box[2] - box[0]) / 2.0)),
                    })

                racket_conf = float(cls_conf[self.racket_class_id])
                if racket_conf >= self.racket_conf_threshold:
                    box_width = box[2] - box[0]
                    box_height = box[3] - box[1]
                    box_area = box_width * box_height
                    max_area = (self.original_width * self.original_height) / 10
                    min_area = 500
                    aspect_ratio = box_height / max(box_width, 1)
                    min_aspect_ratio = 0.3
                    max_aspect_ratio = 10.0
                    if (min_area <= box_area <= max_area and
                        min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                        racket_detections.append({
                            'box': [int(b) for b in box],
                            'confidence': racket_conf,
                            'class_name': 'tennis racket',
                            'area': int(box_area),
                            'aspect_ratio': round(aspect_ratio, 2),
                        })

        return ball_detections, racket_detections
    
    def _scale_box(self, x_center, y_center, width, height):
        """缩放边界框到原始图像尺寸"""
        # 计算缩放比例
        scale_x = self.original_width / self.input_width
        scale_y = self.original_height / self.input_height
        
        # 转换为原始图像坐标
        x1 = (x_center - width / 2) * scale_x
        y1 = (y_center - height / 2) * scale_y
        x2 = (x_center + width / 2) * scale_x
        y2 = (y_center + height / 2) * scale_y
        
        # 限制在图像范围内
        x1 = max(0, min(x1, self.original_width))
        y1 = max(0, min(y1, self.original_height))
        x2 = max(0, min(x2, self.original_width))
        y2 = max(0, min(y2, self.original_height))
        
        return [x1, y1, x2, y2]
    
    def _filter_static_balls(self, ball_detections, racket_detections=None):
        """Select the Active Ball through the dedicated track-selection module."""
        selection = self.ball_track_selector.select(
            ball_detections,
            racket_detections=racket_detections,
            frame_height=getattr(self, "original_height", None),
        )
        self.last_ball_diagnostics = selection.diagnostics
        return [selection.active_ball] if selection.active_ball is not None else []

    def _filter_static_balls_legacy(self, ball_detections, racket_detections=None):
        """Select the active ball and filter persistent static false positives."""
        diagnostics = {
            "raw_candidates": len(ball_detections or []),
            "kept_candidates": 0,
            "continuity_enabled": True,
            "continuity_disabled_reason": None,
            "rejections": {
                "static_hard_mask": 0,
                "low_conf_unsupported": 0,
                "upper_mirror_unsupported": 0,
                "track_became_static": 0,
            },
            "selected": None,
            "final_decision": "no_candidates",
            "top_candidates": [],
        }
        if not ball_detections:
            self.static_ball_filter.update([])
            self.last_ball_diagnostics = diagnostics
            return []

        previous_position = self.ball_history[-1] if self.ball_history else None
        previous_velocity = None
        previous_motion_px = None
        if len(self.ball_history) >= 2:
            previous_velocity = [
                float(self.ball_history[-1][0] - self.ball_history[-2][0]),
                float(self.ball_history[-1][1] - self.ball_history[-2][1]),
            ]
            previous_motion_px = float(
                np.linalg.norm(
                    np.array(self.ball_history[-1], dtype=float) - np.array(self.ball_history[-2], dtype=float)
                )
            )
        previous_in_static_zone = bool(
            previous_position is not None
            and self.static_ball_filter.should_mask(
                previous_position,
                near_previous_track=False,
                near_racket=False,
            )
        )
        max_motion_for_continuity = float(
            self.config.get("ball_max_motion_for_continuity_px", 140.0)
        )
        continuity_enabled = (
            (previous_motion_px is None or previous_motion_px <= max_motion_for_continuity)
            and not previous_in_static_zone
        )
        diagnostics["continuity_enabled"] = bool(continuity_enabled)
        if not continuity_enabled:
            if previous_in_static_zone:
                diagnostics["continuity_disabled_reason"] = "previous_in_static_zone"
            else:
                diagnostics["continuity_disabled_reason"] = "previous_motion_too_large"
        scoring_previous_position = previous_position if continuity_enabled else None
        scoring_previous_velocity = previous_velocity if continuity_enabled else None
        continuity_distance = float(self.config.get("ball_continuity_distance_px", 180.0))
        continuity_keep_ratio = float(self.config.get("static_ball_keep_if_near_prev_ratio", 0.8))
        racket_distance = float(self.config.get("ball_racket_proximity_distance_px", 360.0))
        racket_keep_ratio = float(self.config.get("static_ball_keep_if_near_racket_ratio", 0.65))
        hard_mask_allow_near_prev_min_motion_px = float(
            self.config.get("static_ball_hard_mask_allow_near_prev_min_motion_px", 16.0)
        )
        hard_mask_allow_near_prev_distance_px = float(
            self.config.get("static_ball_hard_mask_allow_near_prev_distance_px", 24.0)
        )
        near_prev_threshold = continuity_distance * continuity_keep_ratio
        near_racket_threshold = racket_distance * racket_keep_ratio

        racket_centers = []
        for racket in racket_detections or []:
            center = racket_center(racket)
            if center is not None:
                racket_centers.append(center)

        self.static_ball_filter.update([det.get("position") for det in ball_detections if det.get("position")])
        adjusted_candidates = []
        for det in ball_detections:
            pos = det.get("position")
            if not pos:
                continue
            dist_to_prev = None
            if previous_position is not None:
                dist_to_prev = float(
                    np.linalg.norm(np.array(pos, dtype=float) - np.array(previous_position, dtype=float))
                )
            near_prev = (
                scoring_previous_position is not None
                and np.linalg.norm(np.array(pos, dtype=float) - np.array(scoring_previous_position, dtype=float)) <= near_prev_threshold
            )
            near_racket = any(
                np.linalg.norm(np.array(pos, dtype=float) - np.array(center, dtype=float)) <= near_racket_threshold
                for center in racket_centers
            )
            near_prev_for_mask = bool(
                near_prev
                and (
                    (previous_motion_px is not None and previous_motion_px >= hard_mask_allow_near_prev_min_motion_px)
                    or (
                        dist_to_prev is not None
                        and dist_to_prev <= hard_mask_allow_near_prev_distance_px
                        and not previous_in_static_zone
                    )
                )
            )
            if self.static_ball_filter.should_mask(
                pos,
                near_previous_track=near_prev_for_mask,
                near_racket=near_racket,
            ):
                diagnostics["rejections"]["static_hard_mask"] += 1
                diagnostics["top_candidates"].append({
                    "position": [float(pos[0]), float(pos[1])],
                    "raw_confidence": float(det.get("confidence", 0.0)),
                    "adjusted_confidence": 0.0,
                    "masked": True,
                    "near_prev": bool(near_prev),
                    "near_racket": bool(near_racket),
                    "reason": "static_hard_mask",
                })
                continue
            penalty = self.static_ball_filter.penalty(
                pos,
                near_previous_track=near_prev,
                near_racket=near_racket,
            )
            adjusted = dict(det)
            adjusted["confidence"] = max(0.0, float(det.get("confidence", 0.0)) - penalty)
            adjusted["static_penalty"] = float(penalty)
            adjusted_candidates.append(adjusted)
            diagnostics["top_candidates"].append({
                "position": [float(pos[0]), float(pos[1])],
                "raw_confidence": float(det.get("confidence", 0.0)),
                "adjusted_confidence": float(adjusted["confidence"]),
                "masked": False,
                "near_prev": bool(near_prev),
                "near_racket": bool(near_racket),
                "reason": "kept",
            })

        diagnostics["kept_candidates"] = len(adjusted_candidates)
        selector_config = {
            **self.config,
            "frame_height": getattr(self, "original_height", None),
        }
        best_ball = select_ball_candidate(
            adjusted_candidates,
            previous_position=scoring_previous_position,
            previous_velocity=scoring_previous_velocity,
            racket_detections=racket_detections,
            config=selector_config,
        )
        if best_ball is None:
            diagnostics["final_decision"] = "no_candidate_after_filter"
            diagnostics["top_candidates"] = sorted(
                diagnostics["top_candidates"],
                key=lambda c: c.get("adjusted_confidence", 0.0),
                reverse=True,
            )[:8]
            self.last_ball_diagnostics = diagnostics
            return []

        best_pos = best_ball.get("position")
        best_conf = float(best_ball.get("confidence", 0.0))
        near_prev_selected = (
            scoring_previous_position is not None
            and best_pos is not None
            and np.linalg.norm(np.array(best_pos, dtype=float) - np.array(scoring_previous_position, dtype=float)) <= near_prev_threshold
        )
        near_racket_selected = False
        if best_pos is not None and racket_centers:
            near_racket_selected = any(
                np.linalg.norm(np.array(best_pos, dtype=float) - np.array(center, dtype=float)) <= near_racket_threshold
                for center in racket_centers
            )

        supported_track = near_prev_selected or near_racket_selected
        min_selected_conf = float(self.config.get("ball_min_selected_confidence", 0.05))
        if best_conf < min_selected_conf and not supported_track:
            diagnostics["rejections"]["low_conf_unsupported"] += 1
            diagnostics["selected"] = {
                "position": [float(best_pos[0]), float(best_pos[1])] if best_pos is not None else None,
                "confidence": float(best_conf),
                "supported_track": False,
            }
            diagnostics["final_decision"] = "reject_low_conf_unsupported"
            diagnostics["top_candidates"] = sorted(
                diagnostics["top_candidates"],
                key=lambda c: c.get("adjusted_confidence", 0.0),
                reverse=True,
            )[:8]
            self.last_ball_diagnostics = diagnostics
            return []

        mirror_hard_min_y_ratio = self.config.get("ball_play_area_min_y_ratio_hard")
        if mirror_hard_min_y_ratio is not None and best_pos is not None:
            try:
                hard_min_y = float(mirror_hard_min_y_ratio) * float(self.original_height)
                if float(best_pos[1]) < hard_min_y and not supported_track:
                    diagnostics["rejections"]["upper_mirror_unsupported"] += 1
                    diagnostics["selected"] = {
                        "position": [float(best_pos[0]), float(best_pos[1])],
                        "confidence": float(best_conf),
                        "supported_track": False,
                    }
                    diagnostics["final_decision"] = "reject_upper_mirror_unsupported"
                    diagnostics["top_candidates"] = sorted(
                        diagnostics["top_candidates"],
                        key=lambda c: c.get("adjusted_confidence", 0.0),
                        reverse=True,
                    )[:8]
                    self.last_ball_diagnostics = diagnostics
                    return []
            except (TypeError, ValueError):
                pass

        ball_pos = best_ball['position']
        
        # 添加到历史
        self.ball_history.append(ball_pos)
        
        # 检查是否静止
        if len(self.ball_history) >= 10:
            positions = np.array(self.ball_history)
            movement = np.max(np.std(positions, axis=0))
            
            if movement < self.static_threshold:
                # 静止球，过滤掉
                diagnostics["rejections"]["track_became_static"] += 1
                diagnostics["selected"] = {
                    "position": [float(ball_pos[0]), float(ball_pos[1])],
                    "confidence": float(best_conf),
                    "supported_track": bool(supported_track),
                }
                diagnostics["final_decision"] = "reject_track_became_static"
                diagnostics["top_candidates"] = sorted(
                    diagnostics["top_candidates"],
                    key=lambda c: c.get("adjusted_confidence", 0.0),
                    reverse=True,
                )[:8]
                self.last_ball_diagnostics = diagnostics
                return []

        diagnostics["selected"] = {
            "position": [float(ball_pos[0]), float(ball_pos[1])],
            "confidence": float(best_conf),
            "supported_track": bool(supported_track),
        }
        diagnostics["final_decision"] = "selected"
        diagnostics["top_candidates"] = sorted(
            diagnostics["top_candidates"],
            key=lambda c: c.get("adjusted_confidence", 0.0),
            reverse=True,
        )[:8]
        self.last_ball_diagnostics = diagnostics
        return [best_ball]

    def _select_primary_racket(self, racket_detections, ball_detections=None):
        """Keep only the active racket, suppressing mirror/reflection candidates."""
        if not racket_detections:
            return []

        ball_position = None
        if ball_detections:
            ball_position = ball_detections[0].get("position")
        previous_center = self.racket_history[-1] if self.racket_history else None
        selector_config = {
            **self.config,
            "frame_height": getattr(self, "original_height", None),
        }
        best_racket = select_racket_candidate(
            racket_detections,
            ball_position=ball_position,
            previous_center=previous_center,
            config=selector_config,
        )
        if best_racket is None:
            return []

        center = racket_center(best_racket)
        if center is not None:
            self.racket_history.append(center)
        return [best_racket]
    
    def get_average_detection_time(self):
        """获取平均检测时间"""
        if self.detection_times:
            return np.mean(self.detection_times) * 1000  # ms
        return 0.0
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理 YOLO26n 统一检测器资源...")
        # Core ML 模型会自动释放
        pass


# 兼容性接口 - 模拟 BallTracker
class BallDetectionWrapper:
    """球检测包装器 - 提供与 BallTracker 兼容的接口"""
    
    def __init__(self, unified_detector):
        self.unified_detector = unified_detector
        self.last_detection_time = 0
    
    def predict_ball(self, frame):
        """检测球（兼容接口）"""
        balls, _, detection_time = self.unified_detector.detect_unified(frame)
        self.last_detection_time = detection_time
        
        # 转换为 BallTracker 格式 - 返回位置列表
        if balls:
            ball = balls[0]
            # BallTracker 返回格式: [x, y, radius, confidence]
            return [[
                ball['position'][0],  # x
                ball['position'][1],  # y
                ball['radius'],       # radius
                ball['confidence']    # confidence
            ]]
        return []
    
    def advanced_ball_processing(self, ball_positions, frame_num):
        """高级球处理（兼容接口）- 直接返回输入"""
        return ball_positions
    
    def draw_trajectory(self, frame):
        """绘制轨迹（兼容接口）- 空实现"""
        pass
    
    def get_trajectory_points(self):
        """获取轨迹点（兼容接口）"""
        return []
    
    def draw_static_balls(self, frame):
        """绘制静态球（兼容接口）- 空实现"""
        pass


# 兼容性接口 - 模拟 RacketDetector
class RacketDetectionWrapper:
    """球拍检测包装器 - 提供与 RacketDetector 兼容的接口"""
    
    def __init__(self, unified_detector):
        self.unified_detector = unified_detector
        self.last_detection_time = 0
    
    def detect_rackets(self, frame):
        """检测球拍（兼容接口）"""
        _, rackets, detection_time = self.unified_detector.detect_unified(frame)
        self.last_detection_time = detection_time
        return rackets
