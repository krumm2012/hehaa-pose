# test_roi_simple.py
"""
简单的ROI测试脚本 - 验证ROI功能是否正常
"""

import cv2
import yaml
import os
from roi_manager import ROIManager

def test_roi_functionality():
    print("🎯 ROI功能简单测试")
    print("=" * 40)
    
    # 创建基础配置
    config = {
        'roi_settings': {
            'enabled': True,
            'interactive_selection': True,
            'auto_load_config': False,  # 强制不加载配置，直接交互式选择
            'visualization': {
                'show_roi_boundary': True,
                'show_roi_fill': True,
                'show_roi_points': True,
                'highlight_detections': True,
                'show_roi_stats': True
            }
        }
    }
    
    # 检查视频文件
    video_path = "data/input_video.mp4"
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return False
    
    # 初始化ROI管理器
    print("🎯 初始化ROI管理器...")
    roi_manager = ROIManager(config)
    
    # 读取视频第一帧
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ 无法读取视频第一帧")
        return False
    
    print(f"✅ 成功读取视频帧: {frame.shape}")
    
    # 交互式ROI选择
    print("\n🎯 开始ROI交互式选择...")
    print("操作说明:")
    print("1. 按顺序点击4个点定义兴趣区域")
    print("2. 点击完4个点后按'c'确认")
    print("3. 按'r'重置，按'q'取消")
    
    selected_points = roi_manager.interactive_roi_selection(frame.copy(), "ROI测试选择")
    
    if not selected_points or len(selected_points) != 4:
        print("⚠️ ROI选择取消或无效")
        return False
    
    print(f"✅ ROI选择完成: {selected_points}")
    
    # 测试ROI功能
    print("\n🧪 测试ROI功能...")
    
    # 测试点在ROI内判断
    test_points = [
        (100, 100),
        (frame.shape[1]//2, frame.shape[0]//2),  # 图像中心点
        (frame.shape[1]-100, frame.shape[0]-100)
    ]
    
    for i, point in enumerate(test_points):
        is_inside = roi_manager.is_point_in_roi(point)
        print(f"测试点 {i+1}: {point} -> {'在ROI内' if is_inside else '在ROI外'}")
    
    # 测试ROI绘制
    print("\n🎨 测试ROI可视化...")
    test_frame = frame.copy()
    test_frame = roi_manager.draw_roi(test_frame, True)
    
    # 添加测试点
    for point in test_points:
        is_inside = roi_manager.is_point_in_roi(point)
        color = (0, 255, 0) if is_inside else (0, 0, 255)
        cv2.circle(test_frame, point, 10, color, -1)
        cv2.putText(test_frame, f"({point[0]},{point[1]})", 
                   (point[0]+15, point[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 显示结果
    cv2.putText(test_frame, "ROI Test - Green: Inside, Red: Outside", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(test_frame, "Press any key to continue...", 
               (10, test_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("ROI测试结果", test_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 获取ROI统计信息
    roi_stats = roi_manager.get_roi_stats()
    print(f"\n📊 ROI统计信息:")
    print(f"面积: {roi_stats.get('roi_area', 0):.0f} 像素²")
    print(f"边界框: {roi_stats.get('roi_bbox', {})}")
    print(f"周长: {roi_stats.get('roi_perimeter', 0):.0f} 像素")
    
    # 保存ROI配置
    roi_manager.save_roi_config("configs/test_roi_config.yaml")
    print("✅ ROI配置已保存到 configs/test_roi_config.yaml")
    
    print("\n🎉 ROI功能测试完成！")
    return True

if __name__ == "__main__":
    test_roi_functionality()
