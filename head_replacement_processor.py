# head_replacement_processor.py
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional
from advanced_face_detector import AdvancedFaceDetector  # 使用新的高级检测器
from face_replacer import FaceReplacer

class HeadReplacementProcessor:
    """头像替换处理器，集成人脸检测和替换功能"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('face_replacement', {}).get('enabled', False)
        
        if not self.enabled:
            logging.info("头像替换功能未启用")
            return
            
        # 初始化高级人脸检测器
        self.face_detector = AdvancedFaceDetector(config)
        
        # 初始化人脸替换器
        self.face_replacer = FaceReplacer(config)
        
        # 统计信息
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'faces_replaced': 0,
            'detection_method_usage': {},
            'error_count': 0
        }
        
        logging.info("头像替换处理器初始化完成")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧图像，进行人脸检测和替换
        
        Args:
            frame: 输入帧
            
        Returns:
            处理后的帧
        """
        if not self.enabled:
            return frame
            
        try:
            self.stats['frames_processed'] += 1
            
            # 人脸检测
            face_detections = self.face_detector.detect_faces(frame)
            
            if face_detections:
                self.stats['faces_detected'] += len(face_detections)
                
                # 更新检测方法使用统计
                for face in face_detections:
                    method = face.get('method', 'unknown')
                    self.stats['detection_method_usage'][method] = \
                        self.stats['detection_method_usage'].get(method, 0) + 1
                
                # 头像替换
                result_frame = self.face_replacer.replace_faces(frame, face_detections)
                
                # 如果替换成功（帧有变化），增加替换计数
                if not np.array_equal(frame, result_frame):
                    self.stats['faces_replaced'] += len(face_detections)
                
                return result_frame
            else:
                return frame
                
        except Exception as e:
            logging.error(f"帧处理错误: {e}")
            self.stats['error_count'] += 1
            return frame
    
    def get_status(self) -> Dict:
        """获取处理器状态信息"""
        if not self.enabled:
            return {'enabled': False, 'status': '头像替换功能未启用'}
            
        # 获取检测器信息
        detector_info = self.face_detector.get_detection_info()
        
        # 确定当前使用的检测方法
        if detector_info['mtcnn_loaded']:
            current_method = 'MTCNN (深度学习)'
        elif detector_info['glip_loaded']:
            current_method = 'GLIP (文本引导)'
        elif detector_info['opencv_available']:
            current_method = 'OpenCV (传统方法)'
        else:
            current_method = '无可用检测器'
        
        return {
            'enabled': True,
            'current_detection_method': current_method,
            'available_methods': detector_info['detection_methods'],
            'stats': self.stats.copy(),
            'detector_info': detector_info
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'faces_replaced': 0,
            'detection_method_usage': {},
            'error_count': 0
        }
        
        if hasattr(self, 'face_detector'):
            self.face_detector.reset_stats()
    
    def draw_debug_info(self, frame: np.ndarray, face_detections: List[Dict]) -> np.ndarray:
        """
        在帧上绘制调试信息（检测框、置信度等）
        
        Args:
            frame: 输入帧
            face_detections: 人脸检测结果
            
        Returns:
            绘制了调试信息的帧
        """
        if not face_detections:
            return frame
            
        result = frame.copy()
        
        for i, face in enumerate(face_detections):
            bbox = face['bbox']
            confidence = face.get('confidence', 0)
            method = face.get('method', 'unknown')
            
            x1, y1, x2, y2 = bbox
            
            # 根据检测方法选择颜色
            if method == 'mtcnn':
                color = (0, 255, 0)  # 绿色
            elif method == 'glip':
                color = (255, 0, 0)  # 蓝色
            else:
                color = (0, 0, 255)  # 红色
            
            # 绘制检测框
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # 绘制置信度和方法标签
            label = f"{method}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result 