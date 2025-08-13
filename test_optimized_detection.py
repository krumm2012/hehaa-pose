#!/usr/bin/env python3
"""
测试优化后的人脸检测过滤效果
"""

import cv2
import yaml
import numpy as np
import time
from advanced_face_detector import AdvancedFaceDetector

def load_config():
    """加载配置文件"""
    with open('configs/default_config.yaml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def test_detection_with_filtering():
    """测试带过滤的人脸检测"""
    print("🎾 测试优化后的人脸检测过滤系统")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    
    # 初始化检测器
    detector = AdvancedFaceDetector(config)
    
    # 显示配置信息
    print("📋 当前过滤配置:")
    size_constraints = config.get('advanced_face_detection', {}).get('face_size_constraints', {})
    quality_filters = config.get('advanced_face_detection', {}).get('quality_filters', {})
    
    print(f"   最大人脸尺寸: {size_constraints.get('max_face_size', 200)}px")
    print(f"   最小人脸尺寸: {size_constraints.get('min_face_size', 50)}px")
    print(f"   最小置信度: {quality_filters.get('min_confidence', 0.75)}")
    print(f"   每帧最大检测数: {quality_filters.get('max_detections_per_frame', 2)}")
    print(f"   启用NMS: {quality_filters.get('enable_nms', True)}")
    
    # 获取检测器信息
    info = detector.get_detection_info()
    print(f"\n🔧 可用检测方法: {info['detection_methods']}")
    
    # 加载测试视频
    video_path = config.get('video_input_path', 'data/input_video.mp4')
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return
    
    frame_count = 0
    total_detections_before = 0
    total_detections_after = 0
    processing_times = []
    
    print(f"\n🎥 开始处理视频: {video_path}")
    print("正在分析前100帧...")
    
    while frame_count < 100:  # 只处理前100帧
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 检测人脸
        start_time = time.time()
        faces = detector.detect_faces(frame)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        # 统计
        total_detections_after += len(faces)
        
        # 每10帧显示进度
        if frame_count % 10 == 0:
            print(f"  处理帧 {frame_count}/100, 检测到 {len(faces)} 个人脸")
        
        # 可视化前几帧的检测结果
        if frame_count <= 5:
            test_frame = frame.copy()
            for i, face in enumerate(faces):
                bbox = face['bbox']
                confidence = face.get('confidence', 0)
                method = face.get('method', 'unknown')
                
                # 绘制边界框
                cv2.rectangle(test_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # 添加信息标签
                label = f"{method}: {confidence:.2f}"
                cv2.putText(test_frame, label, (bbox[0], bbox[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 保存测试结果
            cv2.imwrite(f'test_frame_{frame_count}_filtered.jpg', test_frame)
    
    cap.release()
    
    # 显示统计结果
    stats = detector.detection_stats
    print(f"\n📊 检测统计结果 (前{frame_count}帧):")
    print(f"   总检测数: {total_detections_after}")
    print(f"   平均每帧: {total_detections_after/frame_count:.2f}")
    print(f"   平均处理时间: {np.mean(processing_times)*1000:.2f}ms")
    
    print(f"\n🔍 过滤统计:")
    print(f"   尺寸过滤: {stats['size_filtered']}")
    print(f"   质量过滤: {stats['quality_filtered']}")
    print(f"   位置过滤: {stats['position_filtered']}")
    print(f"   总过滤数: {stats['filtered_out']}")
    
    print(f"\n🎯 检测方法统计:")
    for method, count in stats.items():
        if method not in ['filtered_out', 'size_filtered', 'quality_filtered', 'position_filtered']:
            if count > 0:
                print(f"   {method}: {count}")
    
    # 计算过滤效率
    total_raw_detections = total_detections_after + stats['filtered_out']
    if total_raw_detections > 0:
        filter_rate = (stats['filtered_out'] / total_raw_detections) * 100
        print(f"\n✨ 过滤效率: {filter_rate:.1f}% ({stats['filtered_out']}/{total_raw_detections})")
    
    print(f"\n✅ 测试完成! 优化后的检测结果已保存到 test_frame_*_filtered.jpg")

def test_size_comparison():
    """测试不同尺寸限制的效果"""
    print("\n" + "=" * 50)
    print("🔬 测试不同尺寸限制的效果")
    
    config = load_config()
    
    # 测试不同的最大尺寸设置
    max_sizes = [400, 300, 200, 150, 100]
    
    for max_size in max_sizes:
        print(f"\n测试最大尺寸限制: {max_size}px")
        
        # 修改配置
        test_config = config.copy()
        if 'advanced_face_detection' not in test_config:
            test_config['advanced_face_detection'] = {}
        if 'face_size_constraints' not in test_config['advanced_face_detection']:
            test_config['advanced_face_detection']['face_size_constraints'] = {}
        
        test_config['advanced_face_detection']['face_size_constraints']['max_face_size'] = max_size
        
        # 初始化检测器
        detector = AdvancedFaceDetector(test_config)
        
        # 处理几帧测试
        video_path = config.get('video_input_path', 'data/input_video.mp4')
        cap = cv2.VideoCapture(video_path)
        
        if cap.isOpened():
            total_detections = 0
            frames_tested = 10
            
            for i in range(frames_tested):
                ret, frame = cap.read()
                if not ret:
                    break
                
                faces = detector.detect_faces(frame)
                total_detections += len(faces)
            
            cap.release()
            
            avg_detections = total_detections / frames_tested
            filter_stats = detector.detection_stats
            
            print(f"   平均每帧检测数: {avg_detections:.2f}")
            print(f"   尺寸过滤数: {filter_stats['size_filtered']}")
            print(f"   总过滤数: {filter_stats['filtered_out']}")

if __name__ == "__main__":
    test_detection_with_filtering()
    test_size_comparison() 