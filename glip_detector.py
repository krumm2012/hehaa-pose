# glip_detector.py
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import List, Tuple, Dict, Optional
import logging
import os

# 检查GLIP是否可用
try:
    from maskrcnn_benchmark.config import cfg
    from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
    GLIP_AVAILABLE = True
    print("GLIP模块加载成功")
except ImportError:
    print("警告: GLIP模块未安装。将使用OpenCV人脸检测作为备选")
    GLIP_AVAILABLE = False

class GLIPDetector:
    """GLIP模型检测器，用于基于文本描述检测图像中的目标"""
    
    def __init__(self, config: Dict):
        self.config = config.get('glip_model', {})
        self.face_config = config.get('face_replacement', {})
        self.enabled = self.config.get('enabled', False)
        
        # 默认值设置
        self.demo = None
        self.face_cascade = None
        self.profile_cascade = None
        
        # 文本提示词
        self.text_prompt = self.config.get('text_prompt', 'face . head')
        
        # 初始化OpenCV人脸检测器作为备选（总是初始化）
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            logging.info("OpenCV人脸检测器初始化成功")
        except Exception as e:
            logging.error(f"OpenCV人脸检测器初始化失败: {e}")
            self.face_cascade = None
            self.profile_cascade = None
        
        if not self.enabled:
            logging.info("GLIP检测器未启用，将使用OpenCV备选方案")
            return
            
        if not GLIP_AVAILABLE:
            logging.warning("GLIP模块未安装，将使用OpenCV备选方案")
            return
            
        # 检查模型文件是否存在
        config_path = self.config.get('model_config_path', '')
        checkpoint_path = self.config.get('model_checkpoint_path', '')
        
        if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
            logging.warning(f"GLIP模型文件不存在，将使用OpenCV备选方案")
            logging.warning(f"配置文件: {config_path}")
            logging.warning(f"权重文件: {checkpoint_path}")
            return
            
        try:
            # 初始化GLIP配置
            cfg.merge_from_file(config_path)
            cfg.merge_from_list([
                "MODEL.WEIGHT", checkpoint_path,
                "MODEL.DEVICE", self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            ])
            cfg.freeze()
            
            # 创建GLIP预测器
            self.demo = GLIPDemo(
                cfg,
                min_image_size=800,
                confidence_threshold=self.config.get('confidence_threshold', 0.7),
                show_mask_heatmaps=False
            )
            logging.info("GLIP模型加载成功")
            
        except Exception as e:
            logging.error(f"GLIP模型加载失败: {e}")
            self.demo = None
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        使用GLIP检测图像中的人脸/头部
        
        Args:
            frame: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表，每个结果包含 {'bbox': [x1, y1, x2, y2], 'confidence': float}
        """
        if not self.enabled or self.demo is None:
            return self._opencv_face_detection(frame)
            
        try:
            # 转换为RGB格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # 执行检测
            predictions = self.demo.compute_prediction(pil_image, self.text_prompt)
            
            # 提取检测结果
            faces = []
            if predictions is not None and len(predictions) > 0:
                boxes = predictions.bbox.cpu().numpy()
                scores = predictions.get_field("scores").cpu().numpy()
                labels = predictions.get_field("labels").cpu().numpy()
                
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    if score >= self.config.get('confidence_threshold', 0.7):
                        faces.append({
                            'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                            'confidence': float(score),
                            'label': int(label),
                            'method': 'glip'
                        })
            
            return faces
            
        except Exception as e:
            logging.error(f"GLIP人脸检测错误: {e}")
            return self._opencv_face_detection(frame)
    
    def detect_with_fallback(self, frame: np.ndarray) -> List[Dict]:
        """
        使用GLIP检测，如果失败则使用OpenCV人脸检测作为备选
        """
        # 首先尝试GLIP检测
        faces = self.detect_faces(frame)
        
        # 如果GLIP检测失败或没有结果，使用OpenCV作为备选
        if not faces:
            faces = self._opencv_face_detection(frame)
            
        return faces
    
    def _opencv_face_detection(self, frame: np.ndarray) -> List[Dict]:
        """OpenCV人脸检测备选方案"""
        if self.face_cascade is None:
            return []
            
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 使用更严格的参数检测正面人脸
            faces_front = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,          # 增加最小邻居数，减少误检
                minSize=(50, 50),        # 增加最小尺寸
                maxSize=(80, 80),      # 设置最大尺寸限制
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            faces = []
            for (x, y, w, h) in faces_front:
                # 添加宽高比检查，人脸通常宽高比在0.7-1.3之间
                aspect_ratio = w / h
                if 0.7 <= aspect_ratio <= 1.3:
                    faces.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 0.8,  # OpenCV不提供置信度，使用固定值
                        'label': 0,
                        'method': 'opencv_front'
                    })
            
            # 如果没有检测到正面人脸，尝试侧面人脸（使用更严格的参数）
            if not faces and self.profile_cascade is not None:
                faces_profile = self.profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=10,      # 侧面人脸使用更严格的参数
                    minSize=(60, 60),
                    maxSize=(250, 250)
                )
                
                for (x, y, w, h) in faces_profile:
                    aspect_ratio = w / h
                    if 0.7 <= aspect_ratio <= 1.3:
                        faces.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': 0.6,  # 侧面人脸置信度稍低
                            'label': 0,
                            'method': 'opencv_profile'
                        })
                
            return faces
            
        except Exception as e:
            logging.error(f"OpenCV人脸检测错误: {e}")
            return []
    
    def get_detection_info(self) -> Dict:
        """获取检测器信息"""
        return {
            'glip_available': GLIP_AVAILABLE,
            'glip_loaded': self.demo is not None,
            'opencv_available': self.face_cascade is not None,
            'enabled': self.enabled,
            'current_method': 'glip' if self.demo is not None else 'opencv'
        } 