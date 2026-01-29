#!/usr/bin/env python3
"""
yolo26_detector.py
使用 YOLO26n Core ML 模型进行球和球拍检测
"""

import cv2
import numpy as np
import coremltools as ct
from typing import List, Tuple, Optional, Dict


class YOLO26Detector:
    """YOLO26 球和球拍检测器（Core ML）"""
    
    def __init__(self, model_path: str, config: dict = None):
        """
        初始化 YOLO26 检测器
        
        Args:
            model_path: Core ML 模型路径 (.mlpackage)
            config: 配置字典
        """
        self.config = config or {}
        self.model_path = model_path
        
        # 加载 Core ML 模型
        print(f"🔧 加载 YOLO26 模型: {model_path}")
        try:
            self.model = ct.models.MLModel(model_path)
            print("✅ YOLO26 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
        
        # 获取模型输入输出规格
        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        self.input_shape = self._get_input_shape(spec)
        
        print(f"   输入名称: {self.input_name}")
        print(f"   输入形状: {self.input_shape}")
        
        # 配置参数
        self.ball_confidence = self.config.get('ball_confidence_threshold', 0.5)
        self.racket_confidence = self.config.get('racket_confidence_threshold', 0.4)
        
        # 类别映射 (假设 YOLO26 训练时的类别)
        # 0: ball, 1: racket
        self.class_names = ['ball', 'racket']
        
        print(f"   球置信度阈值: {self.ball_confidence}")
        print(f"   球拍置信度阈值: {self.racket_confidence}")
    
    def _get_input_shape(self, spec):
        """获取模型输入形状"""
        input_desc = spec.description.input[0]
        if input_desc.type.HasField('imageType'):
            # 图像输入
            width = input_desc.type.imageType.width
            height = input_desc.type.imageType.height
            return (height, width)
        elif input_desc.type.HasField('multiArrayType'):
            # 多维数组输入
            shape = input_desc.type.multiArrayType.shape
            return tuple(shape)
        else:
            # 默认 YOLO 输入尺寸
            return (640, 640)
    
    def preprocess(self, frame: np.ndarray) -> Dict:
        """
        预处理图像
        
        Args:
            frame: 输入图像 (BGR)
            
        Returns:
            预处理后的输入字典
        """
        # 保存原始尺寸
        self.original_height, self.original_width = frame.shape[:2]
        
        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        input_h, input_w = self.input_shape
        resized = cv2.resize(rgb_frame, (input_w, input_h))
        
        # 转换为 PIL Image (Core ML 期望的格式)
        from PIL import Image
        pil_image = Image.fromarray(resized)
        
        return {self.input_name: pil_image}
    
    def postprocess(self, predictions: Dict) -> Tuple[List, List]:
        """
        后处理模型输出
        
        Args:
            predictions: 模型预测结果
            
        Returns:
            (ball_detections, racket_detections)
        """
        ball_detections = []
        racket_detections = []
        
        try:
            # 检查输出格式
            if 'coordinates' in predictions and 'confidence' in predictions:
                # 格式1: 带 NMS 的输出 (yolo26m.mlpackage)
                # coordinates: [N, 4] - [x1, y1, x2, y2]
                # confidence: [N] - 置信度
                coords = predictions['coordinates']
                confs = predictions['confidence']
                
                print(f"   检测到 {len(coords)} 个目标 (NMS 后)")
                
                # 注意：带 NMS 的模型可能没有类别信息
                # 假设所有检测都是球（需要根据实际情况调整）
                for i in range(len(coords)):
                    # 提取置信度（可能是数组）
                    conf_value = confs[i]
                    if hasattr(conf_value, '__iter__') and not isinstance(conf_value, str):
                        conf = float(conf_value[0]) if len(conf_value) > 0 else 0.0
                    else:
                        conf = float(conf_value)
                    
                    if conf < 0.1:
                        continue
                    
                    # 坐标已经是 [x1, y1, x2, y2] 格式
                    box = coords[i].tolist()
                    
                    # 缩放到原始图像尺寸
                    input_h, input_w = self.input_shape
                    scale_x = self.original_width / input_w
                    scale_y = self.original_height / input_h
                    
                    box = [
                        box[0] * scale_x,
                        box[1] * scale_y,
                        box[2] * scale_x,
                        box[3] * scale_y
                    ]
                    
                    # 确保在图像范围内
                    box = [
                        max(0, min(box[0], self.original_width)),
                        max(0, min(box[1], self.original_height)),
                        max(0, min(box[2], self.original_width)),
                        max(0, min(box[3], self.original_height))
                    ]
                    
                    # 根据置信度阈值分类（简化版本）
                    if conf >= self.ball_confidence:
                        ball_detections.append({
                            'box': box,
                            'confidence': conf,
                            'class': 'ball',
                            'class_id': 0
                        })
            
            else:
                # 格式2: 不带 NMS 的输出 (yolo26n.mlpackage)
                # [1, 300, 6] - [x_center, y_center, width, height, confidence, class]
                output_key = list(predictions.keys())[0]
                detections = predictions[output_key][0]  # 移除 batch 维度
                
                for detection in detections:
                    x_center, y_center, width, height, conf, cls = detection
                    
                    # 过滤低置信度检测
                    if conf < 0.1:
                        continue
                    
                    # 转换为 [x1, y1, x2, y2] 格式并缩放
                    box = self._scale_box([x_center, y_center, width, height])
                    
                    # 根据类别分类
                    cls_id = int(cls)
                    
                    if cls_id == 0 and conf >= self.ball_confidence:
                        ball_detections.append({
                            'box': box,
                            'confidence': float(conf),
                            'class': 'ball',
                            'class_id': cls_id
                        })
                    
                    elif cls_id == 1 and conf >= self.racket_confidence:
                        racket_detections.append({
                            'box': box,
                            'confidence': float(conf),
                            'class': 'racket',
                            'class_id': cls_id
                        })
        
        except Exception as e:
            print(f"⚠️ 后处理错误: {e}")
            import traceback
            traceback.print_exc()
        
        return ball_detections, racket_detections
    
    def _parse_detection(self, coord, conf):
        """解析单个检测结果"""
        # 根据实际坐标格式调整
        if len(coord) == 4:
            # [x, y, w, h] 或 [x1, y1, x2, y2]
            box = self._scale_box(coord)
        else:
            box = coord
        
        return {
            'box': box,
            'confidence': float(conf)
        }
    
    def _scale_box(self, box):
        """将检测框从模型尺寸缩放到原始图像尺寸"""
        input_h, input_w = self.input_shape
        scale_x = self.original_width / input_w
        scale_y = self.original_height / input_h
        
        # 模型输出格式: [x_center, y_center, width, height] (相对于 640x640)
        x_center, y_center, width, height = box
        
        # 缩放到原始尺寸
        x_center_scaled = x_center * scale_x
        y_center_scaled = y_center * scale_y
        width_scaled = width * scale_x
        height_scaled = height * scale_y
        
        # 转换为 [x1, y1, x2, y2]
        x1 = x_center_scaled - width_scaled / 2
        y1 = y_center_scaled - height_scaled / 2
        x2 = x_center_scaled + width_scaled / 2
        y2 = y_center_scaled + height_scaled / 2
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, self.original_width))
        y1 = max(0, min(y1, self.original_height))
        x2 = max(0, min(x2, self.original_width))
        y2 = max(0, min(y2, self.original_height))
        
        return [x1, y1, x2, y2]
    
    def detect(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        检测球和球拍
        
        Args:
            frame: 输入图像 (BGR)
            
        Returns:
            (ball_detections, racket_detections)
        """
        # 预处理
        input_dict = self.preprocess(frame)
        
        # 推理
        try:
            predictions = self.model.predict(input_dict)
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            return [], []
        
        # 后处理
        ball_detections, racket_detections = self.postprocess(predictions)
        
        return ball_detections, racket_detections
    
    def detect_balls(self, frame: np.ndarray) -> List:
        """仅检测球"""
        ball_detections, _ = self.detect(frame)
        return ball_detections
    
    def detect_rackets(self, frame: np.ndarray) -> List:
        """仅检测球拍"""
        _, racket_detections = self.detect(frame)
        return racket_detections


def test_yolo26_detector():
    """测试 YOLO26 检测器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试 YOLO26 检测器')
    parser.add_argument('--model', type=str, default='yolo26n.mlpackage',
                       help='模型路径')
    parser.add_argument('--image', type=str, required=True,
                       help='测试图像路径')
    parser.add_argument('--output', type=str, default='output/yolo26_test.jpg',
                       help='输出图像路径')
    
    args = parser.parse_args()
    
    # 创建检测器
    config = {
        'ball_confidence_threshold': 0.5,
        'racket_confidence_threshold': 0.4
    }
    detector = YOLO26Detector(args.model, config)
    
    # 读取图像
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"❌ 无法读取图像: {args.image}")
        return
    
    print(f"\n📸 处理图像: {args.image}")
    print(f"   尺寸: {frame.shape[1]}x{frame.shape[0]}")
    
    # 检测
    ball_detections, racket_detections = detector.detect(frame)
    
    print(f"\n📊 检测结果:")
    print(f"   球: {len(ball_detections)} 个")
    print(f"   球拍: {len(racket_detections)} 个")
    
    # 可视化
    result_frame = frame.copy()
    
    # 绘制球
    for det in ball_detections:
        box = det['box']
        conf = det['confidence']
        x1, y1, x2, y2 = map(int, box)
        
        # 绘制边框
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签
        label = f"Ball {conf:.2f}"
        cv2.putText(result_frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"   球: box=[{x1},{y1},{x2},{y2}], conf={conf:.3f}")
    
    # 绘制球拍
    for det in racket_detections:
        box = det['box']
        conf = det['confidence']
        x1, y1, x2, y2 = map(int, box)
        
        # 绘制边框
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        # 绘制标签
        label = f"Racket {conf:.2f}"
        cv2.putText(result_frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        print(f"   球拍: box=[{x1},{y1},{x2},{y2}], conf={conf:.3f}")
    
    # 保存结果
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, result_frame)
    print(f"\n💾 结果已保存: {args.output}")


if __name__ == "__main__":
    test_yolo26_detector()
