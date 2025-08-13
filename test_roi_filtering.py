#!/usr/bin/env python3
"""
快速测试ROI过滤是否真的生效
"""

import cv2
import yaml
from roi_manager import ROIManager
from ball_tracker import BallTracker

def test_roi_filtering():
    print("🧪 测试ROI过滤功能")
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
        print(f"   - ROI设置状态: {roi_manager.is_roi_set}")
        print(f"   - ROI点数: {len(roi_manager.roi_points)}")
    else:
        print("❌ ROI配置加载失败")
        return
    
    # 初始化球检测器
    print("🎾 初始化球检测器...")
    ball_tracker = BallTracker(config.get('tracknet_model_path', None), config, roi_manager)
    print(f"   - 球检测器ROI管理器: {ball_tracker.roi_manager is not None}")
    print(f"   - 球检测器ROI设置状态: {ball_tracker.roi_manager.is_roi_set if ball_tracker.roi_manager else 'None'}")
    
    # 读取一帧进行测试
    print("📹 读取测试帧...")
    video_path = config['video_input_path']
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ 无法读取视频帧")
        return
    
    print(f"✅ 视频帧读取成功: {frame.shape}")
    
    # 进行球检测
    print("🔍 开始球检测...")
    print("-" * 30)
    
    ball_positions = ball_tracker.predict_ball(frame)
    
    print("-" * 30)
    print(f"🎯 检测结果: {len(ball_positions) if ball_positions else 0} 个球")
    
    if ball_positions:
        for i, pos in enumerate(ball_positions):
            x, y = pos[:2]  # 取前两个坐标
            is_in_roi = roi_manager.is_point_in_roi((x, y))
            print(f"   球 {i+1}: ({x}, {y}) - {'✅ ROI内' if is_in_roi else '❌ ROI外'}")
    
    print("\n🧪 ROI过滤测试完成!")

if __name__ == "__main__":
    test_roi_filtering()
