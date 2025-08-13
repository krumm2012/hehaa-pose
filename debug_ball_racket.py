#!/usr/bin/env python3
"""
球检测和球拍检测详细诊断脚本
"""

import cv2
import yaml
import numpy as np
from ball_tracker import BallTracker
from racket_detector import RacketDetector

def load_config(config_path="configs/balanced_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def debug_ball_detection(frame, ball_module, frame_num):
    """详细调试球检测流程"""
    print(f"\n=== 帧 {frame_num} 球检测详细调试 ===")
    
    # 1. HSV检测
    hsv_detections = ball_module._detect_with_hsv(frame)
    print(f"1. HSV颜色检测结果: {len(hsv_detections)} 个球")
    for i, ball in enumerate(hsv_detections):
        print(f"   球 {i+1}: 位置 ({ball[0]}, {ball[1]})")
    
    # 2. 模拟检测（如果HSV失败）
    if not hsv_detections:
        sim_detections = ball_module._simulate_ball_detection(frame)
        print(f"2. 模拟检测结果: {len(sim_detections)} 个球")
        for i, ball in enumerate(sim_detections):
            print(f"   模拟球 {i+1}: 位置 ({ball[0]}, {ball[1]})")
    
    # 3. 完整检测流程
    all_detections = ball_module.predict_ball(frame)
    print(f"3. predict_ball() 总检测结果: {len(all_detections)} 个球")
    
    # 4. 边界检查
    if ball_module.config['use_boundary']:
        print(f"4. 边界检查启用: ({ball_module.config['boundary_x1']}, {ball_module.config['boundary_y1']}) 到 ({ball_module.config['boundary_x2']}, {ball_module.config['boundary_y2']})")
        for ball in all_detections:
            x, y = ball
            in_boundary = (ball_module.config['boundary_x1'] <= x <= ball_module.config['boundary_x2'] and 
                          ball_module.config['boundary_y1'] <= y <= ball_module.config['boundary_y2'])
            print(f"   球 ({x}, {y}) 在边界内: {in_boundary}")
    
    # 5. 高级处理（需要手动调用）
    if hasattr(ball_module, 'advanced_ball_processing'):
        advanced_result = ball_module.advanced_ball_processing(all_detections, frame_num)
        print(f"5. 高级球处理结果: {len(advanced_result)} 个活动球")
        for ball in advanced_result:
            print(f"   活动球: 位置 ({ball[0]}, {ball[1]})")
    
    return all_detections

def debug_racket_detection(frame, racket_module):
    """详细调试球拍检测流程"""
    print(f"\n=== 球拍检测详细调试 ===")
    
    # 1. 原始YOLO检测
    detections = racket_module.detect_rackets(frame)
    print(f"1. YOLO检测结果: {len(detections)} 个球拍")
    
    for i, racket in enumerate(detections):
        box = racket['box']
        confidence = racket['confidence']
        print(f"   球拍 {i+1}: 置信度 {confidence:.3f}, 边界框 [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
        
        # 检查置信度阈值
        threshold = racket_module.confidence_threshold
        passed = confidence >= threshold
        print(f"            置信度阈值 {threshold}, 通过: {passed}")
    
    return detections

def main():
    """主诊断函数"""
    print("🔍 球检测和球拍检测详细诊断开始...")
    
    # 加载配置
    config = load_config()
    print(f"配置文件: configs/balanced_config.yaml")
    print(f"球检测置信度阈值: {config.get('ball_confidence_threshold', 'N/A')}")
    print(f"球拍检测置信度阈值: {config.get('racket_confidence_threshold', 'N/A')}")
    print(f"球尺寸范围: {config.get('min_ball_radius', 'N/A')} - {config.get('max_ball_radius', 'N/A')} 像素")
    
    # 初始化模块
    ball_module = BallTracker(config.get('tracknet_model_path', None), config)
    racket_module = RacketDetector(config['racket_yolo_model_path'], config)
    
    # 打开视频
    video_path = config['video_input_path']
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    # 测试前几帧
    test_frames = [10, 25, 50, 75]
    
    for frame_num in test_frames:
        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"❌ 无法读取帧 {frame_num}")
            continue
        
        print(f"\n{'='*60}")
        print(f"测试帧 {frame_num}")
        print(f"{'='*60}")
        
        # 调试球检测
        ball_detections = debug_ball_detection(frame, ball_module, frame_num)
        
        # 调试球拍检测  
        racket_detections = debug_racket_detection(frame, racket_module)
        
        # 可视化结果
        debug_frame = frame.copy()
        
        # 绘制球检测结果
        for ball in ball_detections:
            cv2.circle(debug_frame, (int(ball[0]), int(ball[1])), 10, (0, 255, 0), 2)
            cv2.putText(debug_frame, "BALL", (int(ball[0])-20, int(ball[1])-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制球拍检测结果
        for racket in racket_detections:
            box = racket['box']
            cv2.rectangle(debug_frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(debug_frame, f"RACKET {racket['confidence']:.2f}", 
                       (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 绘制边界（若启用且允许显示）
        if config.get('use_boundary', False) and config.get('draw_boundary', False):
            cv2.rectangle(debug_frame, 
                         (config['boundary_x1'], config['boundary_y1']),
                         (config['boundary_x2'], config['boundary_y2']), 
                         (0, 255, 255), 2)
        
        # 保存调试图像
        output_path = f"debug_frame_{frame_num}.jpg"
        cv2.imwrite(output_path, debug_frame)
        print(f"调试图像已保存: {output_path}")
    
    cap.release()
    
    print(f"\n📊 诊断总结:")
    print(f"✅ 诊断完成，请检查生成的调试图像")
    print(f"🔍 如果球检测失效，可能原因:")
    print(f"   - HSV颜色阈值不适合当前视频")
    print(f"   - 球的尺寸超出设定范围")
    print(f"   - 球在边界范围之外")
    print(f"   - 置信度阈值过高")
    print(f"🔍 如果球拍检测失效，可能原因:")
    print(f"   - YOLO模型置信度阈值过高")
    print(f"   - 视频中球拍不够清晰")
    print(f"   - 模型未正确加载")

if __name__ == "__main__":
    main() 