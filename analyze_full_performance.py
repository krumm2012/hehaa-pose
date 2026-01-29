#!/usr/bin/env python3
"""
完整性能分析 - 找出所有隐藏开销
分析为什么理论 16.7ms/帧 vs 实际 105ms/帧
"""

import cv2
import yaml
import time
import numpy as np
from collections import defaultdict
import sys

# 导入所有模块
from pose_estimator import PoseEstimator
from yolo26n_unified_detector import YOLO26nUnifiedDetector, BallDetectionWrapper, RacketDetectionWrapper
from roi_manager import ROIManager
from full_swing_analyzer import FullSwingAnalyzer
from speed_analyzer import SpeedAnalyzer
from hit_zone_analyzer import HitZoneAnalyzer


def full_performance_analysis(config_path, video_path, output_path, num_frames=100):
    """
    完整性能分析 - 模拟真实 main.py 的所有操作
    """
    print("=" * 80)
    print("🔍 完整性能分析 - 找出所有隐藏开销")
    print("=" * 80)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\n📹 视频信息:")
    print(f"   分辨率: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   分析帧数: {num_frames}")
    
    # 初始化组件
    print(f"\n🚀 初始化组件...")
    
    roi_manager = ROIManager(config)
    pose_module = PoseEstimator(config['yolo_pose_model_path'], config, roi_manager)
    
    unified_config = config.get('unified_detection', {})
    unified_detector = YOLO26nUnifiedDetector(
        unified_config.get('model_path', 'yolo26n.mlpackage'),
        unified_config,
        roi_manager
    )
    ball_module = BallDetectionWrapper(unified_detector)
    racket_module = RacketDetectionWrapper(unified_detector)
    
    swing_analyzer = FullSwingAnalyzer(config)
    speed_analyzer = SpeedAnalyzer(fps=fps)
    hit_zone_analyzer = HitZoneAnalyzer(config)
    
    print("✅ 组件初始化完成")
    
    # 性能统计
    timings = defaultdict(list)
    
    print(f"\n⏱️  开始完整性能分析...")
    print("-" * 80)
    
    prev_ball_pos = None
    
    total_start = time.time()
    
    for frame_num in range(num_frames):
        frame_start = time.time()
        
        # 1. 读取帧
        t1 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        timings['1_read_frame'].append((time.time() - t1) * 1000)
        
        # 2. 复制显示帧
        t2 = time.time()
        display_frame = frame.copy()
        timings['2_copy_frame'].append((time.time() - t2) * 1000)
        
        # 3. ROI 提取
        t3 = time.time()
        if roi_manager.is_roi_set:
            detection_frame, roi_offset = roi_manager.extract_roi_region(frame)
        else:
            detection_frame = frame
            roi_offset = (0, 0)
        timings['3_roi_extraction'].append((time.time() - t3) * 1000)
        
        # 4. 姿态检测
        t4 = time.time()
        pose_results = pose_module.get_keypoints(frame)
        timings['4_pose_detection'].append((time.time() - t4) * 1000)
        
        # 5. 球和球拍检测
        t5 = time.time()
        ball_positions = ball_module.predict_ball(detection_frame)
        racket_detections = racket_module.detect_rackets(detection_frame)
        timings['5_unified_detection'].append((time.time() - t5) * 1000)
        
        # 6. 坐标转换
        t6 = time.time()
        if roi_offset != (0, 0):
            if ball_positions:
                ball_positions = roi_manager.adjust_detection_coordinates(ball_positions, roi_offset, "ball")
            if racket_detections:
                racket_detections = roi_manager.adjust_detection_coordinates(racket_detections, roi_offset, "racket")
        timings['6_coordinate_transform'].append((time.time() - t6) * 1000)
        
        # 7. 球处理
        t7 = time.time()
        ball_position = None
        if ball_positions and len(ball_positions) > 0:
            if isinstance(ball_positions[0], list) and len(ball_positions[0]) >= 2:
                ball_position = ball_positions[0][:2]
        timings['7_ball_processing'].append((time.time() - t7) * 1000)
        
        # 8. 速度分析
        t8 = time.time()
        if ball_position and prev_ball_pos:
            ball_speed_kmh = speed_analyzer.calculate_ball_speed(prev_ball_pos, tuple(ball_position))
        prev_ball_pos = tuple(ball_position) if ball_position else prev_ball_pos
        timings['8_speed_analysis'].append((time.time() - t8) * 1000)
        
        # 9. 击球点分析（简化）
        t9 = time.time()
        # if ball_position and racket_detections:
        #     hit_zone_analyzer.analyze_hit(ball_position, racket_detections, frame_num)
        timings['9_hit_analysis'].append((time.time() - t9) * 1000)
        
        # 10. 绘制姿态
        t10 = time.time()
        if pose_results:
            display_frame = pose_module.draw_keypoints(display_frame, pose_results)
        timings['10_draw_pose'].append((time.time() - t10) * 1000)
        
        # 11. 绘制球
        t11 = time.time()
        if ball_position:
            cv2.circle(display_frame, (int(ball_position[0]), int(ball_position[1])), 
                      5, (0, 255, 0), -1)
        timings['11_draw_ball'].append((time.time() - t11) * 1000)
        
        # 12. 绘制球拍
        t12 = time.time()
        if racket_detections:
            for det in racket_detections:
                if 'box' in det:
                    box = det['box']
                    cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), 
                                (255, 0, 0), 2)
        timings['12_draw_racket'].append((time.time() - t12) * 1000)
        
        # 13. 绘制 ROI
        t13 = time.time()
        if roi_manager.is_roi_set:
            display_frame = roi_manager.draw_roi_boundary(display_frame)
        timings['13_draw_roi'].append((time.time() - t13) * 1000)
        
        # 14. 添加文字信息
        t14 = time.time()
        cv2.putText(display_frame, f"Frame: {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        timings['14_draw_text'].append((time.time() - t14) * 1000)
        
        # 15. 写入视频
        t15 = time.time()
        out.write(display_frame)
        timings['15_write_video'].append((time.time() - t15) * 1000)
        
        # 总时间
        frame_time = (time.time() - frame_start) * 1000
        timings['99_total_frame'].append(frame_time)
        
        if (frame_num + 1) % 25 == 0:
            print(f"   处理帧 {frame_num + 1}/{num_frames} - {frame_time:.1f}ms")
    
    total_time = time.time() - total_start
    
    cap.release()
    out.release()
    
    # 分析结果
    print("\n" + "=" * 80)
    print("📊 详细性能分析结果")
    print("=" * 80)
    
    print(f"\n{'阶段':<30} {'平均':<10} {'最小':<10} {'最大':<10} {'占比':<10}")
    print("-" * 80)
    
    total_avg = np.mean(timings['99_total_frame'])
    
    stages = [
        ('1. 读取帧', '1_read_frame'),
        ('2. 复制帧', '2_copy_frame'),
        ('3. ROI 提取', '3_roi_extraction'),
        ('4. 姿态检测', '4_pose_detection'),
        ('5. 统一检测', '5_unified_detection'),
        ('6. 坐标转换', '6_coordinate_transform'),
        ('7. 球处理', '7_ball_processing'),
        ('8. 速度分析', '8_speed_analysis'),
        ('9. 击球点分析', '9_hit_analysis'),
        ('10. 绘制姿态', '10_draw_pose'),
        ('11. 绘制球', '11_draw_ball'),
        ('12. 绘制球拍', '12_draw_racket'),
        ('13. 绘制 ROI', '13_draw_roi'),
        ('14. 绘制文字', '14_draw_text'),
        ('15. 写入视频', '15_write_video'),
        ('', ''),
        ('总计', '99_total_frame'),
    ]
    
    for name, key in stages:
        if not name:
            print("-" * 80)
            continue
            
        if key in timings:
            avg = np.mean(timings[key])
            min_t = np.min(timings[key])
            max_t = np.max(timings[key])
            percentage = (avg / total_avg * 100) if key != '99_total_frame' else 100
            
            print(f"{name:<30} {avg:>8.2f}ms {min_t:>8.2f}ms {max_t:>8.2f}ms {percentage:>8.1f}%")
    
    # 总结
    print("\n" + "=" * 80)
    print("🎯 性能总结")
    print("=" * 80)
    
    actual_fps = num_frames / total_time
    theoretical_fps = 1000 / total_avg
    
    print(f"\n实际总时间: {total_time:.2f}秒")
    print(f"处理帧数: {num_frames}")
    print(f"实际 FPS: {actual_fps:.2f}")
    print(f"平均帧时间: {total_avg:.2f}ms")
    print(f"理论 FPS: {theoretical_fps:.2f}")
    
    # 找出最大开销
    print("\n" + "=" * 80)
    print("🔥 最大开销排名")
    print("=" * 80)
    
    costs = []
    for name, key in stages[:-2]:  # 排除空行和总计
        if key in timings:
            avg = np.mean(timings[key])
            percentage = (avg / total_avg * 100)
            costs.append((name, avg, percentage))
    
    costs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, avg, percentage) in enumerate(costs[:10], 1):
        print(f"   {i}. {name:<30} {avg:>8.2f}ms ({percentage:>5.1f}%)")
    
    return timings


if __name__ == "__main__":
    config_path = "configs/yolo26_tennis_config.yaml"
    video_path = "data/16.10.mp4"
    output_path = "/tmp/test_output.mp4"
    
    num_frames = 100
    if len(sys.argv) > 1:
        num_frames = int(sys.argv[1])
    
    timings = full_performance_analysis(config_path, video_path, output_path, num_frames)
    
    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)
