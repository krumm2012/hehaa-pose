#!/usr/bin/env python3
"""
测试 YOLO26m 统一检测球和球拍
对比 YOLO26m vs 当前混合方案（HSV球检测 + YOLOv8n球拍检测）
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

def test_yolo26m_unified():
    """测试 YOLO26m 同时检测球和球拍"""
    print("=" * 60)
    print("🧪 测试 YOLO26m 统一检测")
    print("=" * 60)
    
    # 加载模型
    print("\n📦 加载 YOLO26m 模型...")
    model = YOLO("models/yolo26m.pt")
    
    # 加载测试图像
    test_image = "data/16.10.mp4"
    cap = cv2.VideoCapture(test_image)
    
    if not cap.isOpened():
        print("❌ 无法打开视频")
        return
    
    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取帧")
        return
    
    print(f"✅ 图像尺寸: {frame.shape[1]}x{frame.shape[0]}")
    
    # 测试检测
    print("\n🔍 开始检测...")
    print("   目标类别: 32 (sports ball), 38 (tennis racket)")
    
    start_time = time.time()
    results = model(frame, verbose=False, classes=[32, 38])
    detection_time = time.time() - start_time
    
    print(f"⏱️  检测耗时: {detection_time*1000:.1f}ms")
    
    # 解析结果
    ball_detections = []
    racket_detections = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        for i in range(len(boxes)):
            cls_id = int(class_ids[i])
            conf = float(confidences[i])
            box = boxes[i]
            
            if cls_id == 32:  # sports ball
                ball_detections.append({
                    'box': box,
                    'confidence': conf,
                    'class': 'ball'
                })
            elif cls_id == 38:  # tennis racket
                racket_detections.append({
                    'box': box,
                    'confidence': conf,
                    'class': 'racket'
                })
    
    # 显示结果
    print(f"\n📊 检测结果:")
    print(f"   🎾 球: {len(ball_detections)} 个")
    for i, det in enumerate(ball_detections):
        print(f"      [{i+1}] 置信度: {det['confidence']:.3f}, 位置: {det['box']}")
    
    print(f"   🏓 球拍: {len(racket_detections)} 个")
    for i, det in enumerate(racket_detections):
        print(f"      [{i+1}] 置信度: {det['confidence']:.3f}, 位置: {det['box']}")
    
    # 可视化
    vis_frame = frame.copy()
    
    # 绘制球
    for det in ball_detections:
        x1, y1, x2, y2 = det['box'].astype(int)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Ball {det['confidence']:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 绘制球拍
    for det in racket_detections:
        x1, y1, x2, y2 = det['box'].astype(int)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis_frame, f"Racket {det['confidence']:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # 保存结果
    output_path = "output/yolo26m_unified_test.jpg"
    cv2.imwrite(output_path, vis_frame)
    print(f"\n💾 可视化结果已保存: {output_path}")
    
    cap.release()
    
    return {
        'balls': len(ball_detections),
        'rackets': len(racket_detections),
        'time': detection_time
    }


def test_current_hybrid():
    """测试当前混合方案（HSV + YOLOv8n）"""
    print("\n" + "=" * 60)
    print("🧪 测试当前混合方案")
    print("=" * 60)
    
    from ball_tracker import BallTracker
    from racket_detector import RacketDetector
    import yaml
    
    # 加载配置
    with open("configs/yolo26_tennis_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化检测器
    print("\n📦 初始化检测器...")
    ball_tracker = BallTracker(None, config, None)
    racket_detector = RacketDetector("models/yolov8n.pt", config, None)
    
    # 加载测试图像
    test_image = "data/16.10.mp4"
    cap = cv2.VideoCapture(test_image)
    ret, frame = cap.read()
    
    print(f"✅ 图像尺寸: {frame.shape[1]}x{frame.shape[0]}")
    
    # 球检测
    print("\n🔍 开始球检测（HSV）...")
    start_time = time.time()
    ball_positions = ball_tracker.predict_ball(frame)
    ball_time = time.time() - start_time
    print(f"⏱️  球检测耗时: {ball_time*1000:.1f}ms")
    
    # 球拍检测
    print("\n🔍 开始球拍检测（YOLOv8n）...")
    start_time = time.time()
    racket_detections = racket_detector.detect_rackets(frame)
    racket_time = time.time() - start_time
    print(f"⏱️  球拍检测耗时: {racket_time*1000:.1f}ms")
    
    total_time = ball_time + racket_time
    
    # 显示结果
    print(f"\n📊 检测结果:")
    print(f"   🎾 球: {len(ball_positions) if ball_positions else 0} 个")
    print(f"   🏓 球拍: {len(racket_detections)} 个")
    print(f"   ⏱️  总耗时: {total_time*1000:.1f}ms")
    
    cap.release()
    
    return {
        'balls': len(ball_positions) if ball_positions else 0,
        'rackets': len(racket_detections),
        'time': total_time
    }


def compare_results():
    """对比两种方案"""
    print("\n" + "=" * 60)
    print("📊 性能对比")
    print("=" * 60)
    
    # 测试 YOLO26m
    yolo26m_result = test_yolo26m_unified()
    
    # 测试当前方案
    hybrid_result = test_current_hybrid()
    
    # 对比
    print("\n" + "=" * 60)
    print("🏆 最终对比")
    print("=" * 60)
    
    print(f"\n{'指标':<20} {'YOLO26m':<15} {'混合方案':<15} {'差异':<15}")
    print("-" * 65)
    
    # 球检测数量
    ball_diff = yolo26m_result['balls'] - hybrid_result['balls']
    print(f"{'球检测数量':<20} {yolo26m_result['balls']:<15} {hybrid_result['balls']:<15} {ball_diff:+d}")
    
    # 球拍检测数量
    racket_diff = yolo26m_result['rackets'] - hybrid_result['rackets']
    print(f"{'球拍检测数量':<20} {yolo26m_result['rackets']:<15} {hybrid_result['rackets']:<15} {racket_diff:+d}")
    
    # 检测时间
    time_diff = (yolo26m_result['time'] - hybrid_result['time']) * 1000
    print(f"{'检测时间(ms)':<20} {yolo26m_result['time']*1000:<15.1f} {hybrid_result['time']*1000:<15.1f} {time_diff:+.1f}")
    
    # 速度比较
    speedup = hybrid_result['time'] / yolo26m_result['time']
    print(f"\n{'速度比较':<20} {'YOLO26m 是混合方案的':<30} {speedup:.2f}x")
    
    # 结论
    print("\n" + "=" * 60)
    print("💡 结论")
    print("=" * 60)
    
    if yolo26m_result['time'] < hybrid_result['time']:
        print("✅ YOLO26m 更快！")
    else:
        print("✅ 混合方案更快！")
    
    if yolo26m_result['balls'] > 0 and yolo26m_result['rackets'] > 0:
        print("✅ YOLO26m 可以同时检测球和球拍")
    else:
        print("⚠️  YOLO26m 检测结果需要优化")
    
    print("\n建议:")
    if speedup > 1.2:
        print("  • YOLO26m 统一检测速度更快，建议切换")
    elif speedup < 0.8:
        print("  • 混合方案速度更快，建议保持当前方案")
    else:
        print("  • 两种方案性能相当，根据准确性选择")


if __name__ == "__main__":
    compare_results()
