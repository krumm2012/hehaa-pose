#!/usr/bin/env python3
# enhanced_opencv_detector.py
"""
增强的OpenCV人脸检测器
集成多种改进技术：多尺度检测、质量评估、跟踪稳定性
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import os

class EnhancedOpenCVDetector:
    """
    增强的OpenCV人脸检测器
    特点：
    1. 多级检测（正面、侧面、上身）
    2. 质量评估和过滤
    3. 时间平滑和跟踪
    4. 多尺度检测
    """
    
    def __init__(self, config: Dict):
        self.config = config.get('glip_model', {})
        self.face_config = config.get('face_replacement', {})
        self.opencv_config = config.get('advanced_face_detection', {})
        self.enabled = self.config.get('enabled', False)
        
        # 从配置文件读取动态参数
        size_constraints = self.opencv_config.get('face_size_constraints', {})
        self.min_face_size = size_constraints.get('min_face_size', 25)
        self.max_face_size = size_constraints.get('max_face_size', 300)
        self.min_aspect_ratio = size_constraints.get('min_aspect_ratio', 0.5)
        self.max_aspect_ratio = size_constraints.get('max_aspect_ratio', 2.0)
        
        # 检测器启用控制
        detector_control = self.opencv_config.get('detector_control', {})
        self.enabled_detectors = {
            'front_face': detector_control.get('enable_front_face', True),
            'front_face_alt': detector_control.get('enable_front_face_alt', True),
            'front_face_alt2': detector_control.get('enable_front_face_alt2', True),
            'profile_face': detector_control.get('enable_profile_face', False),  # 默认禁用
            'full_body': detector_control.get('enable_full_body', False)  # 默认禁用
        }
        
        # OpenCV检测器参数配置 - 从配置文件读取
        opencv_params = self.opencv_config.get('opencv_detection_params', {})
        self.detection_params = {}
        
        for detector_name in ['front_face', 'front_face_alt', 'front_face_alt2', 'profile_face', 'full_body']:
            detector_config = opencv_params.get(detector_name, {})
            self.detection_params[detector_name] = {
                'scaleFactor': detector_config.get('scale_factor', 1.1),
                'minNeighbors': detector_config.get('min_neighbors', 3),
                'minSize': (self.min_face_size, self.min_face_size),
                'maxSize': (self.max_face_size, self.max_face_size),
                'confidence': detector_config.get('confidence', 0.75)
            }
        
        # 质量评估参数
        quality_assessment = self.opencv_config.get('quality_assessment', {})
        self.contrast_norm_factor = quality_assessment.get('contrast_normalization_factor', 50.0)
        self.target_brightness = quality_assessment.get('target_brightness', 128)
        self.brightness_tolerance = quality_assessment.get('brightness_tolerance', 128)
        self.canny_low = quality_assessment.get('canny_low_threshold', 50)
        self.canny_high = quality_assessment.get('canny_high_threshold', 150)
        self.edge_density_factor = quality_assessment.get('edge_density_factor', 10.0)
        self.size_norm_factor = quality_assessment.get('size_normalization_factor', 10000.0)
        
        # 质量过滤参数
        quality_filters = self.opencv_config.get('quality_filters', {})
        self.min_quality_score = quality_filters.get('min_quality_score', 0.2)
        self.enable_quality_filter = quality_filters.get('enable_opencv_quality_filter', False)
        
        # 检测合并参数
        merging_config = self.opencv_config.get('detection_merging', {})
        self.overlap_threshold = merging_config.get('overlap_threshold', 0.3)
        
        # 人脸跟踪参数
        tracking_config = self.opencv_config.get('face_tracking', {})
        self.enable_tracking = tracking_config.get('enable_tracking', True)
        self.max_disappeared_frames = tracking_config.get('max_disappeared_frames', 8)
        self.max_match_distance = tracking_config.get('max_match_distance', 80)
        self.smoothing_factor = tracking_config.get('smoothing_factor', 0.7)
        
        # 侧脸特殊参数
        profile_config = opencv_params.get('profile_face', {})
        self.profile_ratio_tolerance = profile_config.get('profile_ratio_tolerance', 0.8)
        
        logging.info(f"动态参数配置: 尺寸范围={self.min_face_size}-{self.max_face_size}px")
        logging.info(f"宽高比范围: {self.min_aspect_ratio}-{self.max_aspect_ratio}")
        logging.info(f"质量过滤: {'启用' if self.enable_quality_filter else '禁用'} (阈值={self.min_quality_score})")
        logging.info(f"跟踪配置: {'启用' if self.enable_tracking else '禁用'} (平滑因子={self.smoothing_factor})")
        
        # 显示启用的检测器
        enabled_list = [name for name, enabled in self.enabled_detectors.items() if enabled]
        disabled_list = [name for name, enabled in self.enabled_detectors.items() if not enabled]
        logging.info(f"启用的检测器: {enabled_list}")
        if disabled_list:
            logging.info(f"禁用的检测器: {disabled_list}")
        
        # 初始化多个检测器
        self.detectors = {}
        self.detection_stats = {}
        
        # 时间平滑跟踪
        if self.enable_tracking:
            self.face_tracker = FaceTracker(max_disappeared=self.max_disappeared_frames, max_match_distance=self.max_match_distance, smoothing_factor=self.smoothing_factor)
        else:
            self.face_tracker = None
        
        # 初始化检测器
        self._init_detectors()
        
        logging.info(f"增强OpenCV检测器初始化完成，可用检测器: {list(self.detectors.keys())}")
    
    def _init_detectors(self):
        """初始化多个Haar级联检测器"""
        detector_configs = [
            ('front_face', 'haarcascade_frontalface_default.xml'),
            ('front_face_alt', 'haarcascade_frontalface_alt.xml'),  
            ('front_face_alt2', 'haarcascade_frontalface_alt2.xml'),
            ('profile_face', 'haarcascade_profileface.xml'),
            ('full_body', 'haarcascade_fullbody.xml'),  # 可能检测到头部
        ]
        
        for name, xml_file in detector_configs:
            # 检查检测器是否被启用
            if not self.enabled_detectors.get(name, False):
                logging.info(f"跳过禁用的检测器: {name}")
                continue
                
            try:
                cascade_path = cv2.data.haarcascades + xml_file
                if os.path.exists(cascade_path):
                    self.detectors[name] = cv2.CascadeClassifier(cascade_path)
                    self.detection_stats[name] = 0
                    logging.info(f"加载检测器: {name}")
                else:
                    logging.warning(f"检测器文件不存在: {cascade_path}")
            except Exception as e:
                logging.warning(f"无法加载检测器 {name}: {e}")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        使用多种方法检测人脸
        
        Args:
            frame: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表
        """
        if not self.detectors:
            return []
        
        # 预处理图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 直方图均衡化增强对比度
        enhanced_gray = cv2.equalizeHist(gray)
        
        all_detections = []
        
        # 动态构建检测方法列表，只包含启用的检测器
        detection_methods = []
        
        # 添加正面人脸检测器
        if 'front_face' in self.detectors:
            detection_methods.append(('front_face', self._detect_front_faces, enhanced_gray))
        
        # 添加备选正面人脸检测器
        if 'front_face_alt' in self.detectors:
            detection_methods.append(('front_face_alt', self._detect_alt_faces, enhanced_gray))
            
        # 添加第二备选正面人脸检测器
        if 'front_face_alt2' in self.detectors:
            detection_methods.append(('front_face_alt2', self._detect_alt_faces, enhanced_gray))
        
        # 添加侧脸检测器（如果启用）
        if 'profile_face' in self.detectors:
            detection_methods.append(('profile_face', self._detect_profile_faces, enhanced_gray))
        
        # 添加全身检测器（如果启用）
        if 'full_body' in self.detectors:
            detection_methods.append(('full_body', self._detect_alt_faces, enhanced_gray))  # 使用通用检测方法
        
        for method_name, detect_func, input_img in detection_methods:
            try:
                detections = detect_func(input_img, method_name)
                all_detections.extend(detections)
            except Exception as e:
                logging.error(f"检测方法 {method_name} 失败: {e}")
        
        # 合并和过滤重复检测
        merged_detections = self._merge_overlapping_detections(all_detections)
        
        # 质量评估和过滤
        quality_filtered = self._quality_filter(merged_detections, frame)
        
        # 时间平滑
        if self.face_tracker:
            smoothed_detections = self.face_tracker.update_tracks(quality_filtered)
        else:
            smoothed_detections = quality_filtered
        
        # 更新统计
        for detection in smoothed_detections:
            method = detection.get('method', 'unknown')
            if method in self.detection_stats:
                self.detection_stats[method] += 1
        
        return smoothed_detections
    
    def _detect_front_faces(self, gray_img: np.ndarray, method_name: str) -> List[Dict]:
        """检测正面人脸"""
        detector = self.detectors.get(method_name)
        if detector is None:
            return []
        
        # 多尺度检测 - 优化参数以提高检测率
        faces = detector.detectMultiScale(
            gray_img,
            scaleFactor=self.detection_params[method_name]['scaleFactor'],
            minNeighbors=self.detection_params[method_name]['minNeighbors'],
            minSize=self.detection_params[method_name]['minSize'],
            maxSize=self.detection_params[method_name]['maxSize'],
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detections = []
        for (x, y, w, h) in faces:
            # 基本质量检查
            aspect_ratio = w / h
            if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                detections.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': self.detection_params[method_name]['confidence'],
                    'method': method_name,
                    'quality_score': self._compute_quality_score(gray_img, (x, y, w, h))
                })
        
        return detections
    
    def _detect_alt_faces(self, gray_img: np.ndarray, method_name: str) -> List[Dict]:
        """使用替代检测器检测人脸"""
        detector = self.detectors.get(method_name)
        if detector is None:
            return []
        
        faces = detector.detectMultiScale(
            gray_img,
            scaleFactor=self.detection_params[method_name]['scaleFactor'],
            minNeighbors=self.detection_params[method_name]['minNeighbors'],
            minSize=self.detection_params[method_name]['minSize'],
            maxSize=self.detection_params[method_name]['maxSize'],
        )
        
        detections = []
        for (x, y, w, h) in faces:
            aspect_ratio = w / h
            if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                detections.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': self.detection_params[method_name]['confidence'],
                    'method': method_name,
                    'quality_score': self._compute_quality_score(gray_img, (x, y, w, h))
                })
        
        return detections
    
    def _detect_profile_faces(self, gray_img: np.ndarray, method_name: str) -> List[Dict]:
        """检测侧面人脸 - 优化参数"""
        detector = self.detectors.get(method_name)
        if detector is None:
            return []
        
        faces = detector.detectMultiScale(
            gray_img,
            scaleFactor=self.detection_params[method_name]['scaleFactor'],
            minNeighbors=self.detection_params[method_name]['minNeighbors'],
            minSize=self.detection_params[method_name]['minSize'],
            maxSize=self.detection_params[method_name]['maxSize'],
        )
        
        detections = []
        for (x, y, w, h) in faces:
            # 侧脸宽高比更宽松
            aspect_ratio = w / h
            profile_min_ratio = max(0.4, self.min_aspect_ratio * self.profile_ratio_tolerance)  # 侧脸可以更窄
            profile_max_ratio = min(2.5, self.max_aspect_ratio * (2.0 - self.profile_ratio_tolerance))  # 侧脸可以更宽
            
            if profile_min_ratio <= aspect_ratio <= profile_max_ratio:
                detections.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': self.detection_params[method_name]['confidence'],
                    'method': method_name,
                    'quality_score': self._compute_quality_score(gray_img, (x, y, w, h))
                })
        
        return detections
    
    def _compute_quality_score(self, gray_img: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """计算检测质量分数"""
        x, y, w, h = bbox
        if x < 0 or y < 0 or x + w >= gray_img.shape[1] or y + h >= gray_img.shape[0]:
            return 0.0
        
        face_region = gray_img[y:y+h, x:x+w]
        
        # 计算多个质量指标
        quality_factors = []
        
        # 1. 对比度评估
        contrast = np.std(face_region)
        quality_factors.append(min(contrast / self.contrast_norm_factor, 1.0))
        
        # 2. 亮度评估（避免过暗或过亮）
        brightness = np.mean(face_region)
        brightness_score = 1.0 - abs(brightness - self.target_brightness) / self.brightness_tolerance
        quality_factors.append(max(brightness_score, 0.0))
        
        # 3. 边缘密度（人脸应该有较多边缘）
        edges = cv2.Canny(face_region, self.canny_low, self.canny_high)
        edge_density = np.sum(edges > 0) / (w * h)
        quality_factors.append(min(edge_density * self.edge_density_factor, 1.0))
        
        # 4. 尺寸分数（中等尺寸更好）
        size_score = min(w * h / self.size_norm_factor, 1.0)
        quality_factors.append(size_score)
        
        # 综合质量分数
        return np.mean(quality_factors)
    
    def _merge_overlapping_detections(self, detections: List[Dict]) -> List[Dict]:
        """合并重叠的检测结果"""
        if not detections:
            return []
        
        # 按质量分数排序
        sorted_detections = sorted(detections, key=lambda x: x.get('quality_score', 0), reverse=True)
        
        merged = []
        used_indices = set()
        
        for i, detection in enumerate(sorted_detections):
            if i in used_indices:
                continue
                
            bbox1 = detection['bbox']
            overlapping_group = [detection]
            
            # 寻找重叠的检测
            for j, other_detection in enumerate(sorted_detections[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                bbox2 = other_detection['bbox']
                iou = self._compute_iou(bbox1, bbox2)
                
                if iou > self.overlap_threshold:  # 重叠阈值
                    overlapping_group.append(other_detection)
                    used_indices.add(j)
            
            # 合并重叠组
            if len(overlapping_group) > 1:
                merged_detection = self._merge_detection_group(overlapping_group)
            else:
                merged_detection = detection
            
            merged.append(merged_detection)
            used_indices.add(i)
        
        return merged
    
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
    
    def _merge_detection_group(self, group: List[Dict]) -> Dict:
        """合并一组重叠的检测"""
        # 选择质量最高的作为基础
        best_detection = max(group, key=lambda x: x.get('quality_score', 0))
        
        # 计算平均边界框
        bboxes = [d['bbox'] for d in group]
        avg_bbox = [
            int(np.mean([bbox[0] for bbox in bboxes])),  # x1
            int(np.mean([bbox[1] for bbox in bboxes])),  # y1
            int(np.mean([bbox[2] for bbox in bboxes])),  # x2
            int(np.mean([bbox[3] for bbox in bboxes]))   # y2
        ]
        
        # 合并置信度
        avg_confidence = np.mean([d.get('confidence', 0) for d in group])
        avg_quality = np.mean([d.get('quality_score', 0) for d in group])
        
        return {
            'bbox': avg_bbox,
            'confidence': avg_confidence,
            'quality_score': avg_quality,
            'method': best_detection['method'],
            'merged_count': len(group)
        }
    
    def _quality_filter(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """简化的质量过滤，避免与上层检测器双重过滤"""
        if not detections:
            return []
        
        # 如果禁用质量过滤，只做基本的边界检查
        if not self.enable_quality_filter:
            frame_h, frame_w = frame.shape[:2]
            boundary_filtered = []
            
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # 只做基本边界检查
                if (x1 >= 0 and y1 >= 0 and x2 < frame_w and y2 < frame_h and
                    x2 > x1 and y2 > y1):
                    boundary_filtered.append(detection)
            
            logging.debug(f"OpenCV质量过滤: 输入{len(detections)}个，边界过滤后{len(boundary_filtered)}个")
            return boundary_filtered
        
        # 启用质量过滤时的完整逻辑
        quality_filtered = [d for d in detections if d.get('quality_score', 0) >= self.min_quality_score]
        
        frame_h, frame_w = frame.shape[:2]
        boundary_filtered = []
        
        for detection in quality_filtered:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            if (x1 >= 0 and y1 >= 0 and x2 < frame_w and y2 < frame_h and
                x2 > x1 and y2 > y1):
                
                width = x2 - x1
                height = y2 - y1
                
                # 使用动态尺寸限制
                if (self.min_face_size <= width <= self.max_face_size and 
                    self.min_face_size <= height <= self.max_face_size):
                    
                    aspect_ratio = width / height
                    if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                        boundary_filtered.append(detection)
        
        logging.debug(f"OpenCV质量过滤: 输入{len(detections)}个，质量过滤{len(quality_filtered)}个，最终{len(boundary_filtered)}个")
        return boundary_filtered
    
    def get_detection_info(self) -> Dict:
        """获取检测器信息"""
        return {
            'mtcnn_available': False,
            'mtcnn_loaded': False,
            'glip_available': False,
            'glip_loaded': False,
            'opencv_available': len(self.detectors) > 0,
            'enabled': self.enabled,
            'detection_methods': ['enhanced_opencv'],
            'detection_stats': self.detection_stats.copy(),
            'available_detectors': list(self.detectors.keys())
        }
    
    def reset_stats(self):
        """重置检测统计"""
        self.detection_stats = {name: 0 for name in self.detection_stats}


class FaceTracker:
    """简单的人脸跟踪器，用于时间平滑"""
    
    def __init__(self, max_disappeared: int = 8, max_match_distance: float = 80, smoothing_factor: float = 0.7):
        self.next_id = 0
        self.face_tracks = {}
        self.max_disappeared = max_disappeared
        self.max_match_distance = max_match_distance
        self.smoothing_factor = smoothing_factor
    
    def update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """更新人脸跟踪，提供时间一致性"""
        if not detections:
            # 增加所有跟踪的消失计数
            disappeared_ids = []
            for face_id in self.face_tracks:
                self.face_tracks[face_id]['disappeared'] += 1
                if self.face_tracks[face_id]['disappeared'] > self.max_disappeared:
                    disappeared_ids.append(face_id)
            
            for face_id in disappeared_ids:
                del self.face_tracks[face_id]
                
            return []
        
        # 关联检测与现有跟踪
        updated_detections = []
        used_detections = set()
        
        for face_id, track_info in list(self.face_tracks.items()):
            best_match_idx = None
            best_distance = float('inf')
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                    
                # 计算中心点距离
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                track_center = track_info['center']
                distance = np.sqrt((center_x - track_center[0])**2 + (center_y - track_center[1])**2)
                
                if distance < best_distance and distance < self.max_match_distance:  # 距离阈值
                    best_distance = distance
                    best_match_idx = i
            
            if best_match_idx is not None:
                # 更新跟踪
                detection = detections[best_match_idx]
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # 平滑中心点
                alpha = self.smoothing_factor  # 平滑因子
                old_center = track_info['center']
                new_center = (
                    alpha * center_x + (1 - alpha) * old_center[0],
                    alpha * center_y + (1 - alpha) * old_center[1]
                )
                
                self.face_tracks[face_id]['center'] = new_center
                self.face_tracks[face_id]['bbox'] = bbox
                self.face_tracks[face_id]['disappeared'] = 0
                
                detection['track_id'] = face_id
                detection['smoothed'] = True
                updated_detections.append(detection)
                used_detections.add(best_match_idx)
        
        # 为未匹配的检测创建新跟踪
        for i, detection in enumerate(detections):
            if i not in used_detections:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                self.face_tracks[self.next_id] = {
                    'center': (center_x, center_y),
                    'bbox': bbox,
                    'disappeared': 0
                }
                
                detection['track_id'] = self.next_id
                detection['smoothed'] = False
                updated_detections.append(detection)
                self.next_id += 1
        
        return updated_detections 