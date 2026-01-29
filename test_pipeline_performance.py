#!/usr/bin/env python3
"""
测试 I/O Pipeline 性能提升
对比有无 Pipeline 的性能差异
"""

import cv2
import yaml
import time
import numpy as np
from pose_estimator import PoseEstimator
from yolo26n_unified_detector import YOLO26nUnifiedDetector, BallDetectionWrapper, RacketDetectionWrapper
from roi_manager import ROIManager
from video_io_pipeline import VideoIOPipeline


def test_without_pipeline(config_path, video_path, num_frames=100):
    """测试不使用 Pipeline"""
    print("\n" + "=" * 60)
    print("📊 测试1: 不使用 Pipeline (顺序 I/O)")
    print("=" * 60)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化
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
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建输出
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('/tmp/test_no_pipeline.mp4', fourcc, fps, (width, height))
    
    start_time = time.time()
    
    for frame_num in range(num_frames):
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        pose_results = pose_module.get_keypoints(frame)
        ball_positions = ball_module.predict_ball(frame)
        racket_detections = racket_module.detect_rackets(frame)
        
        # 处理
        display_frame = frame.copy()
        if pose_results:
            display_frame = pose_module.draw_keypoints(display_frame, pose_results)
        
        # 写入
        out.write(display_frame)
        
        if (frame_num + 1) % 25 == 0:
            print(f"   处理帧 {frame_num + 1}/{num_frames}")
    
    total_time = time.time() - start_time
    
    cap.release()
    out.release()
    
    fps_result = num_frames / total_time
    
    print(f"\n结果:")
    print(f"   总时间: {total_time:.2f}秒")
    print(f"   平均 FPS: {fps_result:.2f}")
    
    return fps_result


def test_with_pipeline(config_path, video_path, num_frames=100):
    """测试使用 Pipeline"""
    print("\n" + "=" * 60)
    print("📊 测试2: 使用 I/O Pipeline (并行 I/O)")
    print("=" * 60)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化
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
    
    # 获取视频信息
    cap_temp = cv2.VideoCapture(video_path)
    fps = cap_temp.get(cv2.CAP_PROP_FPS)
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_temp.release()
    
    # 使用 Pipeline
    with VideoIOPipeline(
        input_path=video_path,
        output_path='/tmp/test_with_pipeline.mp4',
        fps=fps,
        frame_size=(width, height),
        buffer_size=3
    ) as pipeline:
        
        frame_count = 0
        while frame_count < num_frames:
            # 获取帧
            item = pipeline.get_frame()
            if item is None:
                break
            
            frame_num, frame = item
            
            # 检测
            pose_results = pose_module.get_keypoints(frame)
            ball_positions = ball_module.predict_ball(frame)
            racket_detections = racket_module.detect_rackets(frame)
            
            # 处理
            display_frame = frame.copy()
            if pose_results:
                display_frame = pose_module.draw_keypoints(display_frame, pose_results)
            
            # 提交结果
            pipeline.put_frame(frame_num, display_frame)
            
            frame_count += 1
            if frame_count % 25 == 0:
                print(f"   处理帧 {frame_count}/{num_frames}")
    
    # Pipeline 会在 __exit__ 时打印统计
    
    # 从 Pipeline 获取 FPS
    total_time = time.time() - pipeline.start_time
    fps_result = frame_count / total_time
    
    return fps_result


def main():
    print("=" * 60)
    print("🧪 I/O Pipeline 性能测试")
    print("=" * 60)
    
    config_path = "configs/yolo26_tennis_config.yaml"
    video_path = "data/16.10.mp4"
    num_frames = 100
    
    # 测试1: 不使用 Pipeline
    fps_no_pipeline = test_without_pipeline(config_path, video_path, num_frames)
    
    # 测试2: 使用 Pipeline
    fps_with_pipeline = test_with_pipeline(config_path, video_path, num_frames)
    
    # 对比
    print("\n" + "=" * 60)
    print("📈 性能对比")
    print("=" * 60)
    print(f"   不使用 Pipeline: {fps_no_pipeline:.2f} FPS")
    print(f"   使用 Pipeline:   {fps_with_pipeline:.2f} FPS")
    print(f"   提升: {(fps_with_pipeline / fps_no_pipeline - 1) * 100:.1f}%")
    print(f"   加速比: {fps_with_pipeline / fps_no_pipeline:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
