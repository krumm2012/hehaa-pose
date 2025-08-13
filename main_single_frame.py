#!/usr/bin/env python3
"""
运行main.py但只处理前几帧，重点观察ROI效果
"""

import cv2
import yaml
from pose_estimator import PoseEstimator
from ball_tracker import BallTracker
from racket_detector import RacketDetector
from full_swing_analyzer import FullSwingAnalyzer
from head_replacement_processor import HeadReplacementProcessor
from roi_manager import ROIManager
from enhanced_motion_capture import EnhancedMotionCapture
import os
import numpy as np
import time

def load_config(config_path="configs/default_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main_single_frame(config_path="configs/roi_enabled_config.yaml", frame_index: int = 0):
    print("🎯 单帧ROI测试main.py")
    print("=" * 50)
    
    # 加载配置
    config = load_config(config_path)
    video_path = config['video_input_path']
    
    print(f"📹 视频路径: {video_path}")
    
    # 🎯 初始化ROI管理器
    print("🎯 初始化ROI管理器...")
    roi_manager = ROIManager(config)
    print("✅ ROI管理器创建完成")
    
    print(f"ROI设置状态: {config.get('roi_settings', {})}")
    
    # 检查是否需要ROI交互式选择
    roi_settings = config.get('roi_settings', {})
    if roi_settings.get('enabled', False):
        if roi_settings.get('auto_load_config', True):
            roi_config_path = roi_settings.get('roi_config_path', 'configs/roi_config.yaml')
            if os.path.exists(roi_config_path):
                if roi_manager.load_roi_config(roi_config_path):
                    print("✅ ROI配置已从文件加载")
                else:
                    print("⚠️ ROI配置文件存在但未启用或无效")
    else:
        print("🎯 ROI功能未启用")

    # 显示最终ROI状态
    print(f"🎯 ROI最终状态: {'已设置' if roi_manager.is_roi_set else '未设置'}")
    if roi_manager.is_roi_set:
        roi_stats = roi_manager.get_roi_stats()
        print(f"   ROI面积: {roi_stats.get('roi_area', 0):.0f} 像素²")
        print(f"   ROI点数: {len(roi_manager.roi_points)}")

    # 初始化组件
    print("🤖 初始化姿势估计模块...")
    pose_module = PoseEstimator(config['yolo_pose_model_path'], config, roi_manager)
    print("✅ 姿势估计模块初始化完成")
    
    print("🎾 初始化球追踪模块...")
    ball_module = BallTracker(config.get('tracknet_model_path', None), config, roi_manager)
    print("✅ 球追踪模块初始化完成")
    
    print("🏓 初始化球拍检测模块...")
    racket_module = RacketDetector(config['racket_yolo_model_path'], config, roi_manager)
    print("✅ 球拍检测模块初始化完成")
    
    print(f"🎬 开始处理第{frame_index}帧...")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频文件")
        return
    
    # 跳转并读取指定帧
    if frame_index and frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        print(f"❌ 无法读取第{frame_index}帧")
        cap.release()
        return
    
    frame_height, frame_width = frame.shape[:2]
    print(f"📐 帧尺寸: {frame_width}x{frame_height}")
    
    # 处理指定帧
    print("\n" + "="*50)
    print(f"🎬 处理第{frame_index}帧")
    print("="*50)
    
    # 球检测
    print("🎾 开始球检测...")
    ball_positions = ball_module.predict_ball(frame)
    print(f"🎾 球检测结果: {len(ball_positions) if ball_positions else 0} 个球")
    
    # 姿态检测
    print("🤖 开始姿态检测...")
    pose_results = pose_module.get_keypoints(frame)
    print(f"🤖 姿态检测结果: {len(pose_results) if pose_results else 0} 个人")
    
    # 球拍检测
    print("🏓 开始球拍检测...")
    racket_results = racket_module.detect_rackets(frame)
    print(f"🏓 球拍检测结果: {len(racket_results) if racket_results else 0} 个球拍")
    
    # 显示检测结果位置
    if ball_positions:
        print("⚽ 球位置:")
        for i, pos in enumerate(ball_positions):
            x, y = pos[:2]
            is_in_roi = roi_manager.is_point_in_roi((x, y)) if roi_manager.is_roi_set else True
            roi_status = "✅ ROI内" if is_in_roi else "❌ ROI外"
            print(f"   球 {i+1}: ({x}, {y}) - {roi_status}")
    
    if pose_results:
        print("🤖 人物位置:")
        for i, keypoints in enumerate(pose_results):
            if isinstance(keypoints, list) and len(keypoints) > 0:
                # 计算人物中心点（使用鼻子或躯干中心）
                nose = keypoints[0]  # 鼻子
                if len(nose) >= 3 and nose[2] > 0.5:  # 置信度检查
                    x, y = int(nose[0]), int(nose[1])
                    is_in_roi = roi_manager.is_point_in_roi((x, y)) if roi_manager.is_roi_set else True
                    roi_status = "✅ ROI内" if is_in_roi else "❌ ROI外"
                    print(f"   人 {i+1}: ({x}, {y}) - {roi_status}")
                else:
                    print(f"   人 {i+1}: 关键点置信度不足")
    
    if racket_results:
        print("🏓 球拍位置:")
        for i, racket in enumerate(racket_results):
            if len(racket) >= 4:  # x1, y1, x2, y2
                x = int((racket[0] + racket[2]) / 2)  # 中心点x
                y = int((racket[1] + racket[3]) / 2)  # 中心点y
                is_in_roi = roi_manager.is_point_in_roi((x, y)) if roi_manager.is_roi_set else True
                roi_status = "✅ ROI内" if is_in_roi else "❌ ROI外"
                print(f"   球拍 {i+1}: ({x}, {y}) - {roi_status}")
    
    # 创建显示帧
    display_frame = frame.copy()
    
    # 🎯 绘制ROI
    if roi_manager.is_roi_set and roi_settings.get('visualization', {}).get('show_roi_boundary', True):
        display_frame = roi_manager.draw_roi(display_frame, 
                                           roi_settings.get('visualization', {}).get('show_roi_fill', True))
        
        # 高亮ROI内的检测结果
        if roi_settings.get('visualization', {}).get('highlight_detections', True):
            if ball_positions:
                display_frame = roi_manager.highlight_roi_detections(display_frame, ball_positions, "ball")
            if racket_results:
                display_frame = roi_manager.highlight_roi_detections(display_frame, racket_results, "racket")
    
    # 🎯 显示ROI状态信息
    if roi_manager.is_roi_set:
        roi_info = f"ROI: Active ({len(roi_manager.roi_points)} points)"
        cv2.putText(display_frame, roi_info, (10, frame_height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 显示检测统计
        detection_info = f"Poses:{len(pose_results) if pose_results else 0} Balls:{len(ball_positions) if ball_positions else 0} Rackets:{len(racket_results) if racket_results else 0}"
        cv2.putText(display_frame, detection_info, (10, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 窗口与保存
    cv2.namedWindow("ROI Single Frame Test", cv2.WINDOW_NORMAL)
    cv2.imshow("ROI Single Frame Test", display_frame)
    out_dir = "frames_70_90_analysis"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"single_frame_{frame_index:03d}_original.jpg"), frame)
    cv2.imwrite(os.path.join(out_dir, f"single_frame_{frame_index:03d}_debug.jpg"), display_frame)
    
    print(f"\n🎨 显示第{frame_index}帧处理结果，按任意键关闭...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cap.release()
    print("✅ 单帧测试完成!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="单帧ROI测试与球检测")
    parser.add_argument("--config", "-c", default="configs/roi_enabled_config.yaml", help="配置文件路径")
    parser.add_argument("--frame", type=int, default=0, help="要分析的帧号(从0开始)")
    args = parser.parse_args()
    main_single_frame(args.config, args.frame)
