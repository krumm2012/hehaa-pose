# face_replacer.py
import cv2
import numpy as np
from PIL import Image, ImageFilter
import logging
from typing import List, Dict, Tuple, Optional
import os

class FaceReplacer:
    """头像替换模块，支持多种混合模式和平滑过渡"""
    
    def __init__(self, config: Dict):
        self.config = config.get('face_replacement', {})
        self.enabled = self.config.get('enabled', False)
        
        if not self.enabled:
            logging.info("头像替换功能未启用")
            return
            
        # 加载Judy头像
        self.judy_head = self._load_judy_head()
        
        # 配置参数
        self.scale_factor = self.config.get('judy_head_scale_factor', 1.2)
        self.blend_mode = self.config.get('blend_mode', 'seamless')
        self.opacity = self.config.get('replacement_opacity', 0.95)
        self.margin_ratio = self.config.get('face_margin_ratio', 0.1)
        self.smooth_transition = self.config.get('smooth_transition', True)
        self.track_faces = self.config.get('track_faces', True)
        
        # 人脸跟踪状态
        self.face_tracker = FaceTracker() if self.track_faces else None
        
        logging.info(f"头像替换模块初始化完成，混合模式: {self.blend_mode}")
    
    def _load_judy_head(self) -> Optional[np.ndarray]:
        """加载Judy头像图片"""
        judy_path = self.config.get('judy_head_image_path')
        
        if not judy_path:
            logging.error("未配置Judy头像路径")
            return None
            
        if not os.path.exists(judy_path):
            logging.warning(f"Judy头像文件不存在: {judy_path}")
            # 创建一个默认的替换图像
            return self._create_default_replacement()
            
        try:
            # 加载图片，支持透明通道
            judy_img = cv2.imread(judy_path, cv2.IMREAD_UNCHANGED)
            
            if judy_img is None:
                logging.error(f"无法加载Judy头像: {judy_path}")
                return self._create_default_replacement()
                
            # 如果是3通道图片，添加alpha通道
            if len(judy_img.shape) == 3 and judy_img.shape[2] == 3:
                alpha = np.ones((judy_img.shape[0], judy_img.shape[1], 1), dtype=judy_img.dtype) * 255
                judy_img = np.concatenate([judy_img, alpha], axis=2)
                
            logging.info(f"成功加载Judy头像: {judy_img.shape}")
            return judy_img
            
        except Exception as e:
            logging.error(f"加载Judy头像时出错: {e}")
            return self._create_default_replacement()
    
    def _create_default_replacement(self) -> np.ndarray:
        """创建默认替换图像（简单的笑脸）"""
        try:
            # 创建一个200x200的默认图像
            size = 200
            img = np.ones((size, size, 4), dtype=np.uint8) * 255
            
            # 绘制一个简单的笑脸
            center = (size // 2, size // 2)
            radius = size // 3
            
            # 脸部圆形（黄色）
            cv2.circle(img, center, radius, (0, 255, 255, 255), -1)
            
            # 眼睛
            eye_offset = radius // 3
            cv2.circle(img, (center[0] - eye_offset, center[1] - eye_offset), radius // 8, (0, 0, 0, 255), -1)
            cv2.circle(img, (center[0] + eye_offset, center[1] - eye_offset), radius // 8, (0, 0, 0, 255), -1)
            
            # 嘴巴
            mouth_start = (center[0] - eye_offset, center[1] + eye_offset)
            mouth_end = (center[0] + eye_offset, center[1] + eye_offset)
            cv2.ellipse(img, center, (eye_offset, eye_offset // 2), 0, 0, 180, (0, 0, 0, 255), 3)
            
            logging.info("创建了默认替换图像（笑脸）")
            return img
            
        except Exception as e:
            logging.error(f"创建默认替换图像失败: {e}")
            return None
    
    def replace_faces(self, frame: np.ndarray, face_detections: List[Dict]) -> np.ndarray:
        """
        在帧中替换检测到的人脸
        
        Args:
            frame: 输入帧
            face_detections: 人脸检测结果
            
        Returns:
            替换后的帧
        """
        if not self.enabled or self.judy_head is None or not face_detections:
            return frame
            
        result_frame = frame.copy()
        
        # 如果启用人脸跟踪，更新跟踪器
        if self.face_tracker:
            face_detections = self.face_tracker.update_tracks(face_detections)
        
        # 遍历每个检测到的人脸
        for face_info in face_detections:
            bbox = face_info['bbox']
            confidence = face_info.get('confidence', 1.0)
            
            # 更严格的置信度过滤
            if confidence < 0.6:
                continue
                
            # 检查边界框的合理性
            x1, y1, x2, y2 = bbox
            face_width = x2 - x1
            face_height = y2 - y1
            
            # 过滤异常尺寸的检测结果
            if face_width < 40 or face_height < 40:  # 太小
                continue
            if face_width > 400 or face_height > 400:  # 太大
                continue
                
            # 检查宽高比
            aspect_ratio = face_width / face_height
            if aspect_ratio < 0.6 or aspect_ratio > 1.5:  # 不合理的宽高比
                continue
                
            # 检查是否在画面边界内
            if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                continue
                
            try:
                result_frame = self._replace_single_face(result_frame, bbox)
            except Exception as e:
                logging.error(f"替换人脸时出错: {e}")
                continue
                
        return result_frame
    
    def _replace_single_face(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """替换单个人脸"""
        x1, y1, x2, y2 = bbox
        
        # 扩展边界框
        face_width = x2 - x1
        face_height = y2 - y1
        margin_x = int(face_width * self.margin_ratio)
        margin_y = int(face_height * self.margin_ratio)
        
        # 扩展后的边界框
        exp_x1 = max(0, x1 - margin_x)
        exp_y1 = max(0, y1 - margin_y)
        exp_x2 = min(frame.shape[1], x2 + margin_x)
        exp_y2 = min(frame.shape[0], y2 + margin_y)
        
        # 计算目标尺寸
        target_width = exp_x2 - exp_x1
        target_height = exp_y2 - exp_y1
        
        # 调整Judy头像尺寸
        judy_resized = self._resize_judy_head(target_width, target_height)
        
        if judy_resized is None:
            return frame
            
        # 根据混合模式进行替换
        if self.blend_mode == 'direct':
            return self._direct_replacement(frame, judy_resized, exp_x1, exp_y1, exp_x2, exp_y2)
        elif self.blend_mode == 'alpha':
            return self._alpha_blend(frame, judy_resized, exp_x1, exp_y1, exp_x2, exp_y2)
        elif self.blend_mode == 'seamless':
            return self._seamless_clone(frame, judy_resized, exp_x1, exp_y1, exp_x2, exp_y2)
        else:
            return self._direct_replacement(frame, judy_resized, exp_x1, exp_y1, exp_x2, exp_y2)
    
    def _resize_judy_head(self, target_width: int, target_height: int) -> Optional[np.ndarray]:
        """调整Judy头像尺寸"""
        try:
            # 应用缩放因子
            scaled_width = int(target_width * self.scale_factor)
            scaled_height = int(target_height * self.scale_factor)
            
            # 保持宽高比
            judy_h, judy_w = self.judy_head.shape[:2]
            aspect_ratio = judy_w / judy_h
            
            if scaled_width / scaled_height > aspect_ratio:
                # 基于高度缩放
                final_height = scaled_height
                final_width = int(final_height * aspect_ratio)
            else:
                # 基于宽度缩放
                final_width = scaled_width
                final_height = int(final_width / aspect_ratio)
            
            # 调整尺寸
            resized_judy = cv2.resize(self.judy_head, (final_width, final_height), interpolation=cv2.INTER_LANCZOS4)
            
            # 如果缩放后的图片比目标区域大，进行裁剪
            if final_width > target_width or final_height > target_height:
                start_x = max(0, (final_width - target_width) // 2)
                start_y = max(0, (final_height - target_height) // 2)
                end_x = min(final_width, start_x + target_width)
                end_y = min(final_height, start_y + target_height)
                resized_judy = resized_judy[start_y:end_y, start_x:end_x]
            
            # 如果缩放后的图片比目标区域小，进行padding
            elif final_width < target_width or final_height < target_height:
                pad_x = (target_width - final_width) // 2
                pad_y = (target_height - final_height) // 2
                
                padded_judy = np.zeros((target_height, target_width, 4), dtype=resized_judy.dtype)
                end_y = min(target_height, pad_y + final_height)
                end_x = min(target_width, pad_x + final_width)
                padded_judy[pad_y:end_y, pad_x:end_x] = resized_judy[:end_y-pad_y, :end_x-pad_x]
                resized_judy = padded_judy
            
            return resized_judy
            
        except Exception as e:
            logging.error(f"调整Judy头像尺寸时出错: {e}")
            return None
    
    def _direct_replacement(self, frame: np.ndarray, judy_img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """直接替换模式"""
        result = frame.copy()
        
        try:
            # 提取RGB和alpha通道
            judy_rgb = judy_img[:, :, :3]
            judy_alpha = judy_img[:, :, 3] / 255.0
            
            # 应用不透明度
            judy_alpha = judy_alpha * self.opacity
            
            # 获取目标区域
            h, w = judy_img.shape[:2]
            actual_h = min(h, result.shape[0] - y1)
            actual_w = min(w, result.shape[1] - x1)
            
            target_region = result[y1:y1+actual_h, x1:x1+actual_w]
            judy_rgb_crop = judy_rgb[:actual_h, :actual_w]
            judy_alpha_crop = judy_alpha[:actual_h, :actual_w]
            
            # Alpha混合
            for c in range(3):
                target_region[:, :, c] = (
                    judy_alpha_crop * judy_rgb_crop[:, :, c] + 
                    (1 - judy_alpha_crop) * target_region[:, :, c]
                )
            
            result[y1:y1+actual_h, x1:x1+actual_w] = target_region
            return result
            
        except Exception as e:
            logging.error(f"直接替换模式错误: {e}")
            return frame
    
    def _alpha_blend(self, frame: np.ndarray, judy_img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Alpha混合模式，带边缘羽化"""
        result = frame.copy()
        
        try:
            # 创建羽化蒙版
            h, w = judy_img.shape[:2]
            actual_h = min(h, result.shape[0] - y1)
            actual_w = min(w, result.shape[1] - x1)
            
            mask = judy_img[:actual_h, :actual_w, 3].astype(np.float32) / 255.0
            
            # 应用高斯模糊进行羽化
            mask = cv2.GaussianBlur(mask, (5, 5), 2.0)
            mask = mask * self.opacity
            
            # 提取RGB通道
            judy_rgb = judy_img[:actual_h, :actual_w, :3].astype(np.float32)
            target_region = result[y1:y1+actual_h, x1:x1+actual_w].astype(np.float32)
            
            # 执行混合
            mask_3d = np.stack([mask] * 3, axis=2)
            blended = mask_3d * judy_rgb + (1 - mask_3d) * target_region
            
            result[y1:y1+actual_h, x1:x1+actual_w] = blended.astype(np.uint8)
            return result
            
        except Exception as e:
            logging.error(f"Alpha混合模式错误: {e}")
            return self._direct_replacement(frame, judy_img, x1, y1, x2, y2)
    
    def _seamless_clone(self, frame: np.ndarray, judy_img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """无缝克隆模式，使用OpenCV的seamlessClone"""
        try:
            # 创建mask
            h, w = judy_img.shape[:2]
            actual_h = min(h, frame.shape[0] - y1)
            actual_w = min(w, frame.shape[1] - x1)
            
            judy_crop = judy_img[:actual_h, :actual_w]
            mask = (judy_crop[:, :, 3] > 128).astype(np.uint8) * 255
            
            # 腐蚀mask减少边缘artifact
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            
            # 提取RGB通道
            judy_rgb = judy_crop[:, :, :3]
            
            # 计算中心点
            center_x = x1 + actual_w // 2
            center_y = y1 + actual_h // 2
            
            # 确保中心点在图像范围内
            center_x = max(actual_w // 2, min(frame.shape[1] - actual_w // 2, center_x))
            center_y = max(actual_h // 2, min(frame.shape[0] - actual_h // 2, center_y))
            
            # 使用OpenCV的seamlessClone
            result = cv2.seamlessClone(
                judy_rgb, frame, mask, 
                (center_x, center_y), 
                cv2.NORMAL_CLONE
            )
            
            return result
            
        except Exception as e:
            logging.error(f"无缝克隆失败，回退到alpha混合: {e}")
            return self._alpha_blend(frame, judy_img, x1, y1, x2, y2)


class FaceTracker:
    """简单的人脸跟踪器，用于减少闪烁"""
    
    def __init__(self, max_disappeared: int = 10):
        self.next_id = 0
        self.face_tracks = {}
        self.max_disappeared = max_disappeared
    
    def update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """更新人脸跟踪"""
        # 如果没有检测结果，增加所有跟踪的消失计数
        if not detections:
            disappeared_ids = []
            for face_id in self.face_tracks:
                self.face_tracks[face_id]['disappeared'] += 1
                if self.face_tracks[face_id]['disappeared'] > self.max_disappeared:
                    disappeared_ids.append(face_id)
            
            # 删除长时间消失的跟踪
            for face_id in disappeared_ids:
                del self.face_tracks[face_id]
                
            return []
        
        # 关联检测结果与现有跟踪
        updated_detections = []
        used_detections = set()
        
        for face_id, track_info in self.face_tracks.items():
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
                
                if distance < best_distance and distance < 50:  # 距离阈值
                    best_distance = distance
                    best_match_idx = i
            
            if best_match_idx is not None:
                # 更新跟踪
                detection = detections[best_match_idx]
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                self.face_tracks[face_id]['center'] = (center_x, center_y)
                self.face_tracks[face_id]['bbox'] = bbox
                self.face_tracks[face_id]['disappeared'] = 0
                
                detection['track_id'] = face_id
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
                updated_detections.append(detection)
                self.next_id += 1
        
        return updated_detections 