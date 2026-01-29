#!/usr/bin/env python3
"""
test_yolo26_detector_simple.py
简单测试 YOLO26 检测器
"""

import cv2
import numpy as np
from yolo26_detector import YOLO26Detector


def main():
    print("\n🎾 测试 YOLO26 检测器")
    print("="*60)
    
    # 配置
    config = {
        'ball_confidence_threshold': 0.3,
        'racket_confidence_threshold': 0.3
    }
    
    # 创建检测器
    print("\n1. 加载模型...")
    detector = YOLO26Detector('yolo26n.mlpackage', config)
    
    # 创建测试图像
    print("\n2. 创建测试图像...")
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # 在图像上绘制一些形状模拟球和球拍
    # 绘制"球"（绿色圆圈）
    cv2.circle(test_frame, (400, 300), 20, (0, 255, 0), -1)
    cv2.circle(test_frame, (600, 400), 15, (0, 255, 0), -1)
    
    # 绘制"球拍"（矩形）
    cv2.rectangle(test_frame, (200, 200), (280, 400), (255, 0, 255), -1)
    cv2.rectangle(test_frame, (800, 300), (900, 500), (255, 0, 255), -1)
    
    print(f"   图像尺寸: {test_frame.shape[1]}x{test_frame.shape[0]}")
    
    # 检测
    print("\n3. 执行检测...")
    ball_detections, racket_detections = detector.detect(test_frame)
    
    print(f"\n📊 检测结果:")
    print(f"   球: {len(ball_detections)} 个")
    for i, det in enumerate(ball_detections[:5]):
        print(f"      {i+1}. 置信度={det['confidence']:.3f}, box={[int(x) for x in det['box']]}")
    
    print(f"   球拍: {len(racket_detections)} 个")
    for i, det in enumerate(racket_detections[:5]):
        print(f"      {i+1}. 置信度={det['confidence']:.3f}, box={[int(x) for x in det['box']]}")
    
    # 可视化
    print("\n4. 可视化结果...")
    result_frame = test_frame.copy()
    
    # 绘制球
    for det in ball_detections:
        box = det['box']
        conf = det['confidence']
        x1, y1, x2, y2 = map(int, box)
        
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Ball {conf:.2f}"
        cv2.putText(result_frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 绘制球拍
    for det in racket_detections:
        box = det['box']
        conf = det['confidence']
        x1, y1, x2, y2 = map(int, box)
        
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        label = f"Racket {conf:.2f}"
        cv2.putText(result_frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # 保存结果
    import os
    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/yolo26_test_simple.jpg', result_frame)
    print(f"   结果已保存: output/yolo26_test_simple.jpg")
    
    print("\n✅ 测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
