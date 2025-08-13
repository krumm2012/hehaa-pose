#!/usr/bin/env python3
"""
测试严格过滤效果 - 模拟大量误检测场景
"""

import cv2
import yaml
import numpy as np
from advanced_face_detector import AdvancedFaceDetector

def create_test_frame_with_false_detections():
    """创建包含各种尺寸检测框的测试帧"""
    # 创建一个测试图像 (1920x1080)
    test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    test_frame.fill(128)  # 灰色背景
    
    # 模拟各种检测结果
    mock_detections = [
        # 正常尺寸的人脸 (应该保留)
        {'bbox': [400, 300, 480, 380], 'confidence': 0.85, 'method': 'test'},
        {'bbox': [800, 400, 880, 480], 'confidence': 0.90, 'method': 'test'},
        
        # 过大的检测框 (应该被过滤)
        {'bbox': [100, 100, 350, 350], 'confidence': 0.80, 'method': 'test'},  # 250x250
        {'bbox': [600, 200, 900, 500], 'confidence': 0.85, 'method': 'test'},  # 300x300
        
        # 过小的检测框 (应该被过滤)
        {'bbox': [1000, 300, 1030, 330], 'confidence': 0.80, 'method': 'test'},  # 30x30
        {'bbox': [1200, 400, 1235, 435], 'confidence': 0.75, 'method': 'test'},  # 35x35
        
        # 低置信度 (应该被过滤)
        {'bbox': [500, 600, 580, 680], 'confidence': 0.65, 'method': 'test'},
        {'bbox': [700, 700, 780, 780], 'confidence': 0.70, 'method': 'test'},
        
        # 边缘位置 (可能被过滤)
        {'bbox': [10, 10, 90, 90], 'confidence': 0.80, 'method': 'test'},
        {'bbox': [1830, 990, 1910, 1070], 'confidence': 0.80, 'method': 'test'},
        
        # 错误宽高比 (应该被过滤)
        {'bbox': [1100, 500, 1200, 550], 'confidence': 0.80, 'method': 'test'},  # 100x50, 2:1
        {'bbox': [1300, 600, 1350, 700], 'confidence': 0.80, 'method': 'test'},  # 50x100, 1:2
    ]
    
    return test_frame, mock_detections

def test_filtering_effectiveness():
    """测试过滤效果"""
    print("🎾 测试严格过滤效果")
    print("=" * 50)
    
    # 加载配置
    with open('configs/default_config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # 初始化检测器
    detector = AdvancedFaceDetector(config)
    
    # 创建测试数据
    test_frame, mock_detections = create_test_frame_with_false_detections()
    
    print(f"📊 测试数据:")
    print(f"   输入检测数: {len(mock_detections)}")
    
    # 分析每个检测
    print(f"\n🔍 检测分析:")
    for i, detection in enumerate(mock_detections):
        bbox = detection['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        confidence = detection['confidence']
        
        print(f"   {i+1:2d}. 尺寸: {width}x{height}, 置信度: {confidence:.2f}")
    
    # 应用过滤器
    filtered_detections = detector._filter_faces(mock_detections, test_frame.shape)
    
    print(f"\n📈 过滤结果:")
    print(f"   过滤后检测数: {len(filtered_detections)}")
    print(f"   过滤率: {(len(mock_detections) - len(filtered_detections))/len(mock_detections)*100:.1f}%")
    
    # 显示过滤统计
    stats = detector.detection_stats
    print(f"\n🔍 详细过滤统计:")
    print(f"   尺寸过滤: {stats['size_filtered']}")
    print(f"   质量过滤: {stats['quality_filtered']}")
    print(f"   位置过滤: {stats['position_filtered']}")
    print(f"   总过滤数: {stats['filtered_out']}")
    
    # 显示保留的检测
    print(f"\n✅ 保留的检测:")
    for i, detection in enumerate(filtered_detections):
        bbox = detection['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        confidence = detection['confidence']
        
        print(f"   {i+1}. 尺寸: {width}x{height}, 置信度: {confidence:.2f}")
    
    # 可视化结果
    visualize_filtering_results(test_frame, mock_detections, filtered_detections)

def visualize_filtering_results(frame, original_detections, filtered_detections):
    """可视化过滤结果"""
    # 创建对比图像
    result_frame = frame.copy()
    
    # 绘制原始检测 (红色)
    for detection in original_detections:
        bbox = detection['bbox']
        cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        
        # 添加标签
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        label = f"{width}x{height}"
        cv2.putText(result_frame, label, (bbox[0], bbox[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 绘制过滤后的检测 (绿色)
    for detection in filtered_detections:
        bbox = detection['bbox']
        cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
        
        # 添加标签
        confidence = detection['confidence']
        label = f"OK: {confidence:.2f}"
        cv2.putText(result_frame, label, (bbox[0], bbox[3] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 添加图例
    cv2.putText(result_frame, "Red: Original detections", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(result_frame, "Green: Filtered detections", (50, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 保存结果
    cv2.imwrite('filtering_test_result.jpg', result_frame)
    print(f"\n🖼️  过滤对比图已保存到: filtering_test_result.jpg")

def test_progressive_filtering():
    """测试渐进式过滤效果"""
    print(f"\n{'='*50}")
    print("🔬 渐进式过滤测试")
    
    # 加载配置
    with open('configs/default_config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # 测试不同的置信度阈值
    confidence_thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    test_frame, mock_detections = create_test_frame_with_false_detections()
    
    print(f"\n📊 不同置信度阈值的过滤效果:")
    print(f"{'阈值':<6} {'保留数':<8} {'过滤率':<8}")
    print("-" * 24)
    
    for threshold in confidence_thresholds:
        # 修改配置
        test_config = config.copy()
        test_config['advanced_face_detection']['quality_filters']['min_confidence'] = threshold
        
        # 重新初始化检测器
        detector = AdvancedFaceDetector(test_config)
        
        # 应用过滤
        filtered = detector._filter_faces(mock_detections, test_frame.shape)
        filter_rate = (len(mock_detections) - len(filtered))/len(mock_detections)*100
        
        print(f"{threshold:<6.2f} {len(filtered):<8} {filter_rate:<8.1f}%")

if __name__ == "__main__":
    test_filtering_effectiveness()
    test_progressive_filtering()