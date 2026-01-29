#!/usr/bin/env python3
"""
详细性能分析脚本
分阶段计时，找出性能瓶颈
"""

import cv2
import yaml
import time
import numpy as np
from collections import defaultdict

# 导入所有模块
from pose_estimator import PoseEstimator
from yolo26n_unified_detector import YOLO26nUnifiedDetector, BallDetectionWrapper, RacketDetectionWrapper
from roi_manager import ROIManager
from full_swing_analyzer import FullSwingAnalyzer
from speed_analyzer import SpeedAnalyzer
from hit_zone_analyzer import HitZoneAnalyzer


def analyze_performance(config_path, video_path, num_frames=50):
    """
    详细性能分析
    
    Args:
        config_path: 配置文件路径
        video_path: 视频路径
        num_frames: 分析帧数
    """
    print("=" * 80)
    print("🔍 详细性能分析")
    print("=" * 80)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n📹 视频信息:")
    print(f"   FPS: {fps}")
    print(f"   分析帧数: {num_frames}")
    
    # 初始化组件
    print(f"\n🚀 初始化组件...")
    
    # ROI 管理器
    roi_manager = ROIManager(config)
    
    # 姿态估计
    pose_module = PoseEstimator(config['yolo_pose_model_path'], config, roi_manager)
    
    # 统一检测
    unified_config = config.get('unified_detection', {})
    unified_detector = YOLO26nUnifiedDetector(
        unified_config.get('model_path', 'yolo26n.mlpackage'),
        unified_config,
        roi_manager
    )
    ball_module = BallDetectionWrapper(unified_detector)
    racket_module = RacketDetectionWrapper(unified_detector)
    
    # 分析器
    swing_analyzer = FullSwingAnalyzer(config)
    speed_analyzer = SpeedAnalyzer(fps=fps)
    hit_zone_analyzer = HitZoneAnalyzer(config)
    
    print("✅ 组件初始化完成")
    
    # 性能统计
    timings = defaultdict(list)
    
    print(f"\n⏱️  开始性能分析...")
    print("-" * 80)
    
    for frame_num in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.time()
        
        # 1. ROI 提取
        t1 = time.time()
        if roi_manager.is_roi_set:
            detection_frame, roi_offset = roi_manager.extract_roi_region(frame)
        else:
            detection_frame = frame
            roi_offset = (0, 0)
        timings['roi_extraction'].append((time.time() - t1) * 1000)
        
        # 2. 姿态检测
        t2 = time.time()
        pose_results = pose_module.get_keypoints(frame)
        timings['pose_detection'].append((time.time() - t2) * 1000)
        
        # 3. 球和球拍检测（统一）
        t3 = time.time()
        ball_positions = ball_module.predict_ball(detection_frame)
        racket_detections = racket_module.detect_rackets(detection_frame)
        timings['unified_detection'].append((time.time() - t3) * 1000)
        
        # 4. 坐标转换
        t4 = time.time()
        if roi_offset != (0, 0):
            if ball_positions:
                ball_positions = roi_manager.adjust_detection_coordinates(ball_positions, roi_offset, "ball")
            if racket_detections:
                racket_detections = roi_manager.adjust_detection_coordinates(racket_detections, roi_offset, "racket")
        timings['coordinate_transform'].append((time.time() - t4) * 1000)
        
        # 5. 挥拍分析（跳过，简化分析）
        t5 = time.time()
        # if pose_results:
        #     swing_analyzer.update(pose_results, frame_num)
        timings['swing_analysis'].append((time.time() - t5) * 1000)
        
        # 6. 可视化
        t6 = time.time()
        display_frame = frame.copy()
        if pose_results:
            display_frame = pose_module.draw_keypoints(display_frame, pose_results)
        timings['visualization'].append((time.time() - t6) * 1000)
        
        # 7. 其他处理
        t7 = time.time()
        # 模拟其他处理（速度分析、击球点分析等）
        timings['other_processing'].append((time.time() - t7) * 1000)
        
        # 总时间
        frame_time = (time.time() - frame_start) * 1000
        timings['total_frame'].append(frame_time)
        
        if (frame_num + 1) % 10 == 0:
            print(f"   处理帧 {frame_num + 1}/{num_frames} - {frame_time:.1f}ms")
    
    cap.release()
    
    # 分析结果
    print("\n" + "=" * 80)
    print("📊 性能分析结果")
    print("=" * 80)
    
    print(f"\n{'阶段':<25} {'平均时间':<12} {'最小':<10} {'最大':<10} {'占比':<10}")
    print("-" * 80)
    
    total_avg = np.mean(timings['total_frame'])
    
    stages = [
        ('ROI 提取', 'roi_extraction'),
        ('姿态检测', 'pose_detection'),
        ('统一检测 (球+球拍)', 'unified_detection'),
        ('坐标转换', 'coordinate_transform'),
        ('挥拍分析', 'swing_analysis'),
        ('可视化', 'visualization'),
        ('其他处理', 'other_processing'),
        ('总计', 'total_frame'),
    ]
    
    for name, key in stages:
        if key in timings:
            avg = np.mean(timings[key])
            min_t = np.min(timings[key])
            max_t = np.max(timings[key])
            percentage = (avg / total_avg * 100) if key != 'total_frame' else 100
            
            print(f"{name:<25} {avg:>10.2f}ms {min_t:>8.2f}ms {max_t:>8.2f}ms {percentage:>8.1f}%")
    
    # FPS 计算
    avg_frame_time = np.mean(timings['total_frame'])
    fps_result = 1000 / avg_frame_time
    
    print("\n" + "=" * 80)
    print("🎯 性能指标")
    print("=" * 80)
    print(f"   平均帧时间: {avg_frame_time:.2f}ms")
    print(f"   理论 FPS: {fps_result:.2f}")
    print(f"   实际 FPS: ~{fps_result * 0.95:.2f} (考虑开销)")
    
    # 瓶颈分析
    print("\n" + "=" * 80)
    print("🔥 性能瓶颈分析")
    print("=" * 80)
    
    bottlenecks = []
    for name, key in stages[:-1]:  # 排除总计
        if key in timings:
            avg = np.mean(timings[key])
            percentage = (avg / total_avg * 100)
            if percentage > 10:
                bottlenecks.append((name, avg, percentage))
    
    bottlenecks.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, avg, percentage) in enumerate(bottlenecks, 1):
        print(f"   {i}. {name}: {avg:.2f}ms ({percentage:.1f}%)")
    
    # 优化建议
    print("\n" + "=" * 80)
    print("💡 优化建议")
    print("=" * 80)
    
    pose_avg = np.mean(timings['pose_detection'])
    det_avg = np.mean(timings['unified_detection'])
    vis_avg = np.mean(timings['visualization'])
    
    if pose_avg > 10:
        print(f"   1. 姿态检测 ({pose_avg:.1f}ms)")
        print(f"      - 考虑降低输入分辨率")
        print(f"      - 尝试 ANE 计算单元")
        print(f"      - 减少检测频率（每2帧检测一次）")
    
    if det_avg > 5:
        print(f"   2. 统一检测 ({det_avg:.1f}ms)")
        print(f"      - 已使用 YOLO26n，性能较好")
        print(f"      - 可考虑降低置信度阈值减少后处理")
    
    if vis_avg > 5:
        print(f"   3. 可视化 ({vis_avg:.1f}ms)")
        print(f"      - 减少绘制元素")
        print(f"      - 使用更简单的绘制方式")
        print(f"      - 考虑降低输出分辨率")
    
    print(f"\n   4. 总体优化:")
    print(f"      - 当前 FPS: {fps_result:.2f}")
    print(f"      - 目标 FPS: 15-20")
    print(f"      - 需要提升: {(15/fps_result - 1)*100:.1f}%")
    
    return timings


if __name__ == "__main__":
    import sys
    
    config_path = "configs/yolo26_tennis_config.yaml"
    video_path = "data/16.10.mp4"
    num_frames = 50
    
    if len(sys.argv) > 1:
        num_frames = int(sys.argv[1])
    
    timings = analyze_performance(config_path, video_path, num_frames)
    
    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)
