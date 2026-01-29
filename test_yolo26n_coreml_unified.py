#!/usr/bin/env python3
"""
测试 YOLO26n Core ML 统一检测球和球拍
使用 yolo26n.mlpackage 进行检测
"""

import cv2
import numpy as np
import coremltools as ct
from PIL import Image
import time

class YOLO26nCoreMLDetector:
    """YOLO26n Core ML 检测器"""
    
    def __init__(self, model_path="yolo26n.mlpackage"):
        print(f"📦 加载 Core ML 模型: {model_path}")
        self.model = ct.models.MLModel(model_path)
        
        # 获取模型规格
        spec = self.model.get_spec()
        
        # 获取输入尺寸
        input_desc = spec.description.input[0]
        if hasattr(input_desc.type, 'imageType'):
            self.input_height = input_desc.type.imageType.height
            self.input_width = input_desc.type.imageType.width
        else:
            self.input_height = 640
            self.input_width = 640
        
        print(f"   输入尺寸: {self.input_width}x{self.input_height}")
        
        # 类别 ID
        self.ball_class_id = 32  # sports ball
        self.racket_class_id = 38  # tennis racket
        
    def preprocess(self, frame):
        """预处理图像"""
        # 保存原始尺寸
        self.original_height, self.original_width = frame.shape[:2]
        
        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 缩放到模型输入尺寸
        resized = cv2.resize(rgb_frame, (self.input_width, self.input_height))
        
        # 转换为 PIL Image
        pil_image = Image.fromarray(resized)
        
        return pil_image
    
    def detect(self, frame, conf_threshold=0.1):
        """检测球和球拍"""
        # 预处理
        input_image = self.preprocess(frame)
        
        # 推理
        start_time = time.time()
        predictions = self.model.predict({'image': input_image})
        inference_time = time.time() - start_time
        
        # 解析结果
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
                x_center, y_center, width, height, conf, cls = detection
                
                # 过滤低置信度
                if conf < conf_threshold:
                    continue
                
                cls_id = int(cls)
                
                # 缩放边界框
                box = self._scale_box_from_center(x_center, y_center, width, height)
                
                if cls_id == self.ball_class_id:
                    ball_detections.append({
                        'box': box,
                        'confidence': float(conf),
                        'class': 'ball',
                        'class_id': cls_id
                    })
                elif cls_id == self.racket_class_id:
                    racket_detections.append({
                        'box': box,
                        'confidence': float(conf),
                        'class': 'racket',
                        'class_id': cls_id
                    })
        
        return ball_detections, racket_detections, inference_time
    
    def _scale_box_from_center(self, x_center, y_center, width, height):
        """从中心点格式缩放边界框到原始图像尺寸"""
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



def test_yolo26n_coreml():
    """测试 YOLO26n Core ML 检测"""
    print("=" * 60)
    print("🧪 测试 YOLO26n Core ML 统一检测")
    print("=" * 60)
    
    # 初始化检测器
    detector = YOLO26nCoreMLDetector("yolo26n.mlpackage")
    
    # 加载测试视频
    test_video = "data/16.10.mp4"
    cap = cv2.VideoCapture(test_video)
    
    if not cap.isOpened():
        print("❌ 无法打开视频")
        return
    
    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取帧")
        return
    
    print(f"\n✅ 图像尺寸: {frame.shape[1]}x{frame.shape[0]}")
    
    # 测试不同的置信度阈值
    thresholds = [0.05, 0.1, 0.2, 0.3]
    
    for conf_thresh in thresholds:
        print(f"\n{'='*60}")
        print(f"🔍 测试置信度阈值: {conf_thresh}")
        print(f"{'='*60}")
        
        balls, rackets, inference_time = detector.detect(frame, conf_threshold=conf_thresh)
        
        print(f"\n⏱️  推理时间: {inference_time*1000:.1f}ms")
        print(f"\n📊 检测结果:")
        print(f"   🎾 球: {len(balls)} 个")
        for i, det in enumerate(balls):
            print(f"      [{i+1}] 置信度: {det['confidence']:.3f}, 位置: {det['box']}")
        
        print(f"   🏓 球拍: {len(rackets)} 个")
        for i, det in enumerate(rackets):
            print(f"      [{i+1}] 置信度: {det['confidence']:.3f}, 位置: {det['box']}")
        
        # 可视化最佳结果
        if conf_thresh == 0.1:
            vis_frame = frame.copy()
            
            # 绘制球
            for det in balls:
                x1, y1, x2, y2 = [int(v) for v in det['box']]
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Ball {det['confidence']:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制球拍
            for det in rackets:
                x1, y1, x2, y2 = [int(v) for v in det['box']]
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis_frame, f"Racket {det['confidence']:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 保存结果
            output_path = "output/yolo26n_coreml_test.jpg"
            cv2.imwrite(output_path, vis_frame)
            print(f"\n💾 可视化结果已保存: {output_path}")
    
    cap.release()
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_yolo26n_coreml()
