#!/usr/bin/env python3
"""
测试ROI预处理功能 - 验证ROI区域提取和坐标转换
"""

import cv2
import yaml
from roi_manager import ROIManager
import numpy as np

def test_roi_preprocessing():
    print("🧪 测试ROI预处理功能")
    print("=" * 50)
    
    # 加载配置
    with open("configs/roi_enabled_config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化ROI管理器
    print("🎯 初始化ROI管理器...")
    roi_manager = ROIManager(config)
    
    # 加载ROI配置
    if roi_manager.load_roi_config("configs/roi_config.yaml"):
        print("✅ ROI配置加载成功")
        print(f"   ROI点数: {len(roi_manager.roi_points)}")
        print(f"   ROI坐标: {roi_manager.roi_points}")
    else:
        print("❌ ROI配置加载失败")
        return
    
    # 读取测试帧
    video_path = config['video_input_path']
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ 无法读取视频帧")
        return
    
    frame_height, frame_width = frame.shape[:2]
    print(f"📹 原始帧尺寸: {frame_width}x{frame_height}")
    
    # 测试ROI掩码生成
    print("\n🎭 测试ROI掩码生成...")
    roi_mask = roi_manager.get_roi_mask(frame.shape[:2])
    if roi_mask is not None:
        print(f"✅ ROI掩码生成成功: {roi_mask.shape}")
        roi_area = np.sum(roi_mask == 255)
        total_area = frame_width * frame_height
        roi_percentage = (roi_area / total_area) * 100
        print(f"   ROI面积: {roi_area} 像素 ({roi_percentage:.1f}%)")
    else:
        print("❌ ROI掩码生成失败")
        return
    
    # 测试ROI边界框获取
    print("\n📦 测试ROI边界框...")
    roi_bbox = roi_manager.get_roi_bounding_box()
    if roi_bbox:
        x1, y1, x2, y2 = roi_bbox
        print(f"✅ ROI边界框: ({x1}, {y1}) - ({x2}, {y2})")
        print(f"   边界框尺寸: {x2-x1}x{y2-y1}")
        
        # 提取ROI区域
        roi_cropped = frame[y1:y2, x1:x2]
        print(f"   裁剪区域尺寸: {roi_cropped.shape}")
        
        # 计算性能提升
        original_pixels = frame_width * frame_height
        cropped_pixels = roi_cropped.shape[0] * roi_cropped.shape[1]
        reduction_percentage = (1 - cropped_pixels / original_pixels) * 100
        print(f"   计算量减少: {reduction_percentage:.1f}%")
        
    else:
        print("❌ ROI边界框获取失败")
        return
    
    # 测试坐标转换
    print("\n🔄 测试坐标转换...")
    roi_offset = (x1, y1)
    
    # 模拟检测结果
    test_detections = {
        'ball': [[50, 60], [100, 120]],  # 两个球的坐标
        'pose': [[[30, 40, 0.9], [50, 80, 0.8]]],  # 一个人的关键点
        'racket': [{'box': [70, 90, 120, 140]}]  # 一个球拍边界框
    }
    
    for detection_type, detections in test_detections.items():
        print(f"   测试 {detection_type} 坐标转换:")
        print(f"     原始: {detections}")
        adjusted = roi_manager.adjust_detection_coordinates(detections, roi_offset, detection_type)
        print(f"     调整后: {adjusted}")
    
    # 可视化测试
    print("\n🎨 生成可视化测试图片...")
    
    # 创建对比图
    comparison_frame = np.hstack([
        cv2.resize(frame, (640, 360)),  # 原始帧
        cv2.resize(roi_cropped, (640, 360))  # ROI裁剪帧
    ])
    
    # 在原始帧上绘制ROI边界
    display_frame = frame.copy()
    display_frame = roi_manager.draw_roi(display_frame, show_fill=True)
    
    # 添加文字说明
    cv2.putText(comparison_frame, "Original Frame", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison_frame, "ROI Cropped", (650, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison_frame, f"Size reduction: {reduction_percentage:.1f}%", 
               (10, comparison_frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 保存测试图片
    cv2.imwrite("roi_preprocessing_test.jpg", comparison_frame)
    cv2.imwrite("roi_boundary_visualization.jpg", display_frame)
    
    print("✅ 测试图片已保存:")
    print("   - roi_preprocessing_test.jpg (原始vs裁剪对比)")
    print("   - roi_boundary_visualization.jpg (ROI边界可视化)")
    
    # 性能预期
    print(f"\n📈 性能优化预期:")
    print(f"   - 检测计算量减少: {reduction_percentage:.1f}%")
    print(f"   - 预期速度提升: {1/(1-reduction_percentage/100):.1f}x")
    print(f"   - ROI区域占比: {roi_percentage:.1f}%")
    
    print("\n✅ ROI预处理功能测试完成!")

if __name__ == "__main__":
    test_roi_preprocessing()
