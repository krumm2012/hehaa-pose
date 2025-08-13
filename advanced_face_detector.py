#!/usr/bin/env python3
# advanced_face_detector.py
"""
高级人脸检测器，集成增强的OpenCV检测和其他方法
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import os

# 导入增强的OpenCV检测器
from enhanced_opencv_detector import EnhancedOpenCVDetector

# 检查MTCNN是否可用
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
    print("MTCNN模块加载成功")
except ImportError:
    MTCNN_AVAILABLE = False
    print("警告: MTCNN模块未安装或依赖缺失")

# 检查GLIP是否可用（保持兼容性）
try:
    from maskrcnn_benchmark.config import cfg
    from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
    GLIP_AVAILABLE = True
    print("GLIP模块加载成功")
except ImportError:
    GLIP_AVAILABLE = False

class AdvancedFaceDetector:
    """
    高级人脸检测器，支持多种检测方法：
    1. Enhanced OpenCV (增强的OpenCV检测，主要方法)
    2. MTCNN (深度学习，如果可用)
    3. GLIP (文本引导检测，如果可用)
    """
    
    def __init__(self, config: Dict):
        self.config = config.get('glip_model', {})
        self.face_config = config.get('face_replacement', {})
        # 获取高级人脸检测配置
        self.advanced_config = config.get('advanced_face_detection', {})
        self.enabled = self.config.get('enabled', False)
        
        # 加载过滤参数
        self.size_constraints = self.advanced_config.get('face_size_constraints', {})
        self.quality_filters = self.advanced_config.get('quality_filters', {})
        self.position_constraints = self.advanced_config.get('position_constraints', {})
        
        # 初始化各检测器
        self.enhanced_opencv_detector = None
        self.mtcnn_detector = None
        self.glip_demo = None
        
        # 初始化统计
        self.detection_stats = {
            'enhanced_opencv': 0,
            'mtcnn': 0,
            'glip': 0,
            'fallback_opencv': 0,
            'filtered_out': 0,  # 被过滤掉的检测数量
            'size_filtered': 0,  # 尺寸过滤
            'quality_filtered': 0,  # 质量过滤
            'position_filtered': 0,  # 位置过滤
            'total_frames': 0
        }
        
        # 初始化检测器（按优先级）
        self._init_enhanced_opencv()
        self._init_mtcnn()
        self._init_glip()
        
        # 确定使用顺序
        self.detection_methods = self._get_detection_order()
        
        logging.info(f"高级人脸检测器初始化完成，可用方法: {list(self.detection_methods.keys())}")
        logging.info(f"过滤配置: 最大尺寸{self.size_constraints.get('max_face_size', 150)}px, 最小置信度{self.quality_filters.get('min_confidence', 0.85)}")
    
    def _init_enhanced_opencv(self):
        """初始化增强的OpenCV检测器（优先使用）"""
        try:
            self.enhanced_opencv_detector = EnhancedOpenCVDetector(self.config)
            logging.info("增强OpenCV检测器初始化成功")
        except Exception as e:
            logging.error(f"增强OpenCV检测器初始化失败: {e}")
            self.enhanced_opencv_detector = None
    
    def _init_mtcnn(self):
        """初始化MTCNN检测器"""
        if not MTCNN_AVAILABLE:
            return
            
        # 获取MTCNN配置
        mtcnn_config = self.advanced_config.get('mtcnn', {})
        
        if not mtcnn_config.get('enabled', True):
            logging.info("MTCNN在配置中被禁用")
            return
            
        try:
            # MTCNN 1.0.0版本只支持stages和device参数
            device = mtcnn_config.get('device', 'cpu')
            device_str = f"CPU:0" if device.lower() == 'cpu' else f"GPU:0"
            
            # 初始化MTCNN，使用正确的参数
            self.mtcnn_detector = MTCNN(
                stages='face_and_landmarks_detection',
                device=device_str
            )
            
            # 保存其他配置参数用于后处理
            self.mtcnn_min_face_size = mtcnn_config.get('min_face_size', 50)
            self.mtcnn_confidence_threshold = mtcnn_config.get('confidence_threshold', 0.8)
            self.mtcnn_steps_threshold = mtcnn_config.get('steps_threshold', [0.7, 0.8, 0.8])
            
            logging.info(f"MTCNN检测器初始化成功 - 设备: {device_str}, 置信度阈值: {self.mtcnn_confidence_threshold}")
        except Exception as e:
            logging.error(f"MTCNN初始化失败: {e}")
            self.mtcnn_detector = None
    
    def _init_glip(self):
        """初始化GLIP检测器（保持原有逻辑）"""
        if not self.enabled or not GLIP_AVAILABLE:
            return
            
        config_path = self.config.get('model_config_path', '')
        checkpoint_path = self.config.get('model_checkpoint_path', '')
        
        if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
            logging.warning("GLIP模型文件不存在")
            return
            
        try:
            cfg.merge_from_file(config_path)
            cfg.merge_from_list([
                "MODEL.WEIGHT", checkpoint_path,
                "MODEL.DEVICE", self.config.get('device', 'cpu')
            ])
            cfg.freeze()
            
            self.glip_demo = GLIPDemo(
                cfg,
                min_image_size=800,
                confidence_threshold=self.config.get('confidence_threshold', 0.7),
                show_mask_heatmaps=False
            )
            logging.info("GLIP模型加载成功")
        except Exception as e:
            logging.error(f"GLIP模型加载失败: {e}")
            self.glip_demo = None
    
    def _get_detection_order(self) -> Dict:
        """确定检测方法的优先顺序"""
        methods = {}
        
        # 优先级：Enhanced OpenCV > MTCNN > GLIP
        if self.enhanced_opencv_detector is not None:
            methods['enhanced_opencv'] = self._enhanced_opencv_detect
            
        if self.mtcnn_detector is not None:
            methods['mtcnn'] = self._mtcnn_detect
            
        if self.glip_demo is not None:
            methods['glip'] = self._glip_detect
            
        return methods
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        检测人脸主函数
        """
        try:
            detections = []
            
            # 获取帧信息
            height, width = frame.shape[:2]
            self.detection_stats['total_frames'] += 1
            
            logging.debug(f"开始处理帧: 尺寸={width}x{height}")
            
            # 使用增强的OpenCV检测
            if self.enhanced_opencv_detector:
                try:
                    enhanced_detections = self.enhanced_opencv_detector.detect_faces(frame)
                    detections.extend(enhanced_detections)
                    logging.debug(f"增强OpenCV检测器返回: {len(enhanced_detections)}个检测")
                    
                    # 打印每个检测的详细信息
                    for i, det in enumerate(enhanced_detections):
                        bbox = det.get('bbox', [])
                        if len(bbox) >= 4:
                            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                            logging.debug(f"  检测{i+1}: 位置=({bbox[0]},{bbox[1]}), 尺寸={w}x{h}, 置信度={det.get('confidence', 0):.2f}, 方法={det.get('method', 'unknown')}")
                    
                except Exception as e:
                    logging.error(f"增强OpenCV检测失败: {e}")
                    
            # 使用MTCNN检测（如果可用且启用）
            if self.mtcnn_detector and self.mtcnn_config.get('enabled', False):
                try:
                    mtcnn_detections = self._detect_with_mtcnn(frame)
                    detections.extend(mtcnn_detections)
                    logging.debug(f"MTCNN检测器返回: {len(mtcnn_detections)}个检测")
                except Exception as e:
                    logging.error(f"MTCNN检测失败: {e}")
            
            # 应用过滤器
            filtered_detections = self._filter_faces(detections, (height, width))
            logging.debug(f"过滤前: {len(detections)}个检测, 过滤后: {len(filtered_detections)}个检测")
            
            # 如果过滤后没有检测，提供详细原因
            if len(detections) > 0 and len(filtered_detections) == 0:
                logging.warning(f"警告: 所有{len(detections)}个检测都被过滤掉了！")
                # 尝试分析过滤原因
                for i, det in enumerate(detections):
                    bbox = det.get('bbox', [])
                    if len(bbox) >= 4:
                        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        confidence = det.get('confidence', 0)
                        
                        reasons = []
                        # 检查尺寸
                        if w < self.size_constraints.get('min_face_size', 30) or h < self.size_constraints.get('min_face_size', 30):
                            reasons.append(f"尺寸太小({w}x{h})")
                        if w > self.size_constraints.get('max_face_size', 120) or h > self.size_constraints.get('max_face_size', 120):
                            reasons.append(f"尺寸太大({w}x{h})")
                        
                        # 检查置信度
                        if confidence < self.quality_filters.get('min_confidence', 0.6):
                            reasons.append(f"置信度太低({confidence:.2f})")
                        
                        logging.debug(f"  被过滤的检测{i+1}: {', '.join(reasons) if reasons else '未知原因'}")
            
            # 更新统计信息
            if len(filtered_detections) > 0 and self.detection_methods:
                first_method = list(self.detection_methods.keys())[0]
                self.detection_stats[f'{first_method}_detections'] = self.detection_stats.get(f'{first_method}_detections', 0) + len(filtered_detections)
            
            logging.debug(f"最终返回: {len(filtered_detections)}个检测")
            return filtered_detections
            
        except Exception as e:
            logging.error(f"人脸检测失败: {e}")
            return []
    
    def _enhanced_opencv_detect(self, frame: np.ndarray) -> List[Dict]:
        """增强OpenCV人脸检测"""
        if self.enhanced_opencv_detector is None:
            return []
        return self.enhanced_opencv_detector.detect_faces(frame)
    
    def _mtcnn_detect(self, frame: np.ndarray) -> List[Dict]:
        """MTCNN人脸检测"""
        if self.mtcnn_detector is None:
            return []
            
        try:
            # 转换为RGB格式（MTCNN需要RGB）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测人脸
            detections = self.mtcnn_detector.detect_faces(rgb_frame)
            
            faces = []
            for detection in detections:
                confidence = detection['confidence']
                if confidence >= self.mtcnn_confidence_threshold:
                    bbox = detection['box']  # [x, y, width, height]
                    
                    # 检查最小人脸尺寸
                    face_width, face_height = bbox[2], bbox[3]
                    if (face_width >= self.mtcnn_min_face_size and 
                        face_height >= self.mtcnn_min_face_size):
                        
                        faces.append({
                            'bbox': [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                            'confidence': confidence,
                            'keypoints': detection.get('keypoints', {}),
                            'method': 'mtcnn'
                        })
            
            return faces
            
        except Exception as e:
            logging.error(f"MTCNN检测错误: {e}")
            return []
    
    def _glip_detect(self, frame: np.ndarray) -> List[Dict]:
        """GLIP人脸检测"""
        if self.glip_demo is None:
            return []
            
        try:
            from PIL import Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            text_prompt = self.config.get('text_prompt', 'face . head')
            predictions = self.glip_demo.compute_prediction(pil_image, text_prompt)
            
            faces = []
            if predictions is not None and len(predictions) > 0:
                boxes = predictions.bbox.cpu().numpy()
                scores = predictions.get_field("scores").cpu().numpy()
                
                for i, (box, score) in enumerate(zip(boxes, scores)):
                    if score >= self.config.get('confidence_threshold', 0.7):
                        faces.append({
                            'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                            'confidence': float(score),
                            'method': 'glip'
                        })
            
            return faces
            
        except Exception as e:
            logging.error(f"GLIP检测错误: {e}")
            return []
    
    def _filter_faces(self, faces: List[Dict], frame_shape: tuple) -> List[Dict]:
        """严格的人脸过滤和优化"""
        if not faces:
            return []
            
        frame_h, frame_w = frame_shape[:2]
        filtered = []
        filter_stats = {'size': 0, 'quality': 0, 'position': 0, 'duplicates': 0}
        
        # 1. 基础过滤：尺寸、置信度、边界
        basic_filtered = []
        for face in faces:
            bbox = face['bbox']
            confidence = face.get('confidence', 0)
            
            # 置信度过滤
            min_confidence = self.quality_filters.get('min_confidence', 0.85)
            if confidence < min_confidence:
                filter_stats['quality'] += 1
                continue
            
            x1, y1, x2, y2 = bbox
            face_width = x2 - x1
            face_height = y2 - y1
            
            # 尺寸约束过滤
            min_size = self.size_constraints.get('min_face_size', 60)
            max_size = self.size_constraints.get('max_face_size', 150)
            
            if face_width < min_size or face_height < min_size:
                filter_stats['size'] += 1
                continue
            if face_width > max_size or face_height > max_size:
                filter_stats['size'] += 1
                continue
                
            # 相对尺寸约束
            relative_max = self.size_constraints.get('relative_max_size', 0.25)
            max_relative_width = frame_w * relative_max
            max_relative_height = frame_h * relative_max
            if face_width > max_relative_width or face_height > max_relative_height:
                filter_stats['size'] += 1
                continue
            
            # 宽高比过滤
            aspect_ratio = face_width / face_height
            min_ratio = self.size_constraints.get('min_aspect_ratio', 0.75)
            max_ratio = self.size_constraints.get('max_aspect_ratio', 1.25)
            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                filter_stats['size'] += 1
                continue
            
            # 边界检查
            if x1 < 0 or y1 < 0 or x2 >= frame_w or y2 >= frame_h:
                filter_stats['position'] += 1
                continue
            
            # 边缘排除
            if self.position_constraints.get('exclude_edges', True):
                edge_margin = self.position_constraints.get('edge_margin_ratio', 0.15)
                margin_w = frame_w * edge_margin
                margin_h = frame_h * edge_margin
                
                if (x1 < margin_w or y1 < margin_h or 
                    x2 > frame_w - margin_w or y2 > frame_h - margin_h):
                    filter_stats['position'] += 1
                    continue
            
            # 标志区域排除
            if self.position_constraints.get('exclude_logo_areas', True):
                logo_zones = self.position_constraints.get('logo_exclusion_zones', [])
                in_logo_zone = False
                
                for zone in logo_zones:
                    if len(zone) == 4:
                        zx1, zy1, zx2, zy2 = zone
                        # 支持相对坐标（0-1）
                        if zx2 <= 1.0 and zy2 <= 1.0:
                            zx1, zy1, zx2, zy2 = int(zx1 * frame_w), int(zy1 * frame_h), int(zx2 * frame_w), int(zy2 * frame_h)
                        
                        # 检查人脸是否与标志区域重叠
                        if not (x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2):
                            in_logo_zone = True
                            break
                
                if in_logo_zone:
                    filter_stats['position'] += 1
                    continue
            
            basic_filtered.append(face)
        
        # 2. 质量分数增强（如果检测器支持）
        quality_enhanced = []
        for face in basic_filtered:
            # 计算位置权重（中心偏好）
            if self.position_constraints.get('center_preference', True):
                bbox = face['bbox']
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 距离中心的归一化距离
                dist_from_center = np.sqrt(
                    ((center_x - frame_w/2) / (frame_w/2))**2 + 
                    ((center_y - frame_h/2) / (frame_h/2))**2
                )
                
                # 中心权重
                center_weight = self.position_constraints.get('center_weight', 1.5)
                position_bonus = max(0, (1 - dist_from_center) * (center_weight - 1))
                
                # 增强置信度
                original_confidence = face.get('confidence', 0)
                position_bonus_factor = 0.1  # 位置奖励因子，可以配置
                face['confidence'] = min(1.0, original_confidence + position_bonus * position_bonus_factor)
                face['position_score'] = 1 - dist_from_center
            
            quality_enhanced.append(face)
        
        # 3. 非极大值抑制 (NMS)
        if self.quality_filters.get('enable_nms', True):
            nms_filtered = self._apply_nms(quality_enhanced)
        else:
            nms_filtered = quality_enhanced
        
        # 4. 限制每帧最大检测数量
        max_detections = self.quality_filters.get('max_detections_per_frame', 1)
        if len(nms_filtered) > max_detections:
            # 按置信度排序，取前N个
            nms_filtered.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            final_filtered = nms_filtered[:max_detections]
            filter_stats['duplicates'] += len(nms_filtered) - max_detections
        else:
            final_filtered = nms_filtered
        
        # 更新统计
        self.detection_stats['size_filtered'] += filter_stats['size']
        self.detection_stats['quality_filtered'] += filter_stats['quality']
        self.detection_stats['position_filtered'] += filter_stats['position']
        self.detection_stats['filtered_out'] += sum(filter_stats.values())
        
        if final_filtered:
            logging.debug(f"过滤结果: 输入{len(faces)}个检测，输出{len(final_filtered)}个 "
                         f"(尺寸过滤{filter_stats['size']}, 质量过滤{filter_stats['quality']}, "
                         f"位置过滤{filter_stats['position']}, 重复过滤{filter_stats['duplicates']})")
        
        return final_filtered
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """应用非极大值抑制"""
        if len(detections) <= 1:
            return detections
        
        # 按置信度排序
        sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        keep = []
        iou_threshold = self.quality_filters.get('nms_iou_threshold', 0.4)
        
        while sorted_detections:
            # 取置信度最高的
            current = sorted_detections.pop(0)
            keep.append(current)
            
            # 移除与当前检测重叠过多的其他检测
            remaining = []
            for detection in sorted_detections:
                iou = self._compute_iou(current['bbox'], detection['bbox'])
                if iou <= iou_threshold:
                    remaining.append(detection)
            
            sorted_detections = remaining
        
        return keep
    
    def _compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算相交区域
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    
    def get_detection_info(self) -> Dict:
        """获取检测器信息"""
        # 获取增强OpenCV检测器的详细信息
        opencv_info = {}
        if self.enhanced_opencv_detector:
            opencv_info = self.enhanced_opencv_detector.get_detection_info()
        
        return {
            'mtcnn_available': MTCNN_AVAILABLE,
            'mtcnn_loaded': self.mtcnn_detector is not None,
            'glip_available': GLIP_AVAILABLE,
            'glip_loaded': self.glip_demo is not None,
            'enhanced_opencv_available': self.enhanced_opencv_detector is not None,
            'opencv_available': opencv_info.get('opencv_available', False),
            'enabled': self.enabled,
            'detection_methods': list(self.detection_methods.keys()),
            'detection_stats': self.detection_stats.copy(),
            'opencv_detector_info': opencv_info
        }
    
    def reset_stats(self):
        """重置检测统计"""
        self.detection_stats = {
            'enhanced_opencv': 0,
            'mtcnn': 0,
            'glip': 0,
            'fallback_opencv': 0,
            'filtered_out': 0,
            'size_filtered': 0,
            'quality_filtered': 0,
            'position_filtered': 0,
            'total_frames': 0
        }
        if self.enhanced_opencv_detector:
            self.enhanced_opencv_detector.reset_stats() 