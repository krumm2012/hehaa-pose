# demo_roi_detection.py
"""
ROI兴趣区域检测演示脚本
演示4点描线的兴趣区检测功能，包括：
1. 交互式ROI选择
2. ROI内的人物、球拍、球检测
3. 增强的动作捕捉
4. 实时可视化显示
"""

import cv2
import yaml
import os
import sys
import numpy as np
from roi_manager import ROIManager
from enhanced_motion_capture import EnhancedMotionCapture
from pose_estimator import PoseEstimator
from ball_tracker import BallTracker
from racket_detector import RacketDetector

def load_config(config_path="configs/default_config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_demo_config():
    """创建演示用的配置"""
    demo_config = {
        # 基础配置
        'video_input_path': "data/input_video.mp4",
        'yolo_pose_model_path': "models/yolov8n-pose.pt",
        'racket_yolo_model_path': "yolov8s.pt",
        
        # ROI配置
        'roi_settings': {
            'enabled': True,
            'interactive_selection': True,
            'auto_load_config': True,
            'roi_config_path': "configs/roi_demo_config.yaml",
            'visualization': {
                'show_roi_boundary': True,
                'show_roi_fill': True,
                'show_roi_points': True,
                'highlight_detections': True,
                'show_roi_stats': True
            }
        },
        
        # 动作捕捉配置
        'roi_motion_capture': {
            'enabled': True,
            'pose_history_length': 30,
            'ball_history_length': 50,
            'racket_history_length': 30,
            'thresholds': {
                'significant_movement': 15,
                'rapid_movement': 30,
                'pose_change_threshold': 20,
                'swing_velocity_threshold': 25
            }
        },
        
        # 检测参数
        'pose_confidence_threshold': 0.5,
        'ball_confidence_threshold': 0.75,
        'racket_confidence_threshold': 0.4,
        
        # 其他必要配置
        'use_boundary': False,
        'use_mask_zones': False,
        'static_ball_movement_threshold_px': 5,
        'static_ball_frames_threshold': 8,
        'min_ball_radius': 3,
        'max_ball_radius': 30,
        'ball_detection_debug': {'enabled': False}
    }
    return demo_config

def main():
    print("🎯 ROI兴趣区域检测演示")
    print("=" * 50)
    
    # 检查视频文件
    config = create_demo_config()
    video_path = config['video_input_path']
    
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        print("请将测试视频放置在 data/input_video.mp4")
        return
    
    # 初始化视频捕获
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"📹 视频信息: {frame_width}x{frame_height}, {fps}FPS")
    
    # 初始化ROI管理器
    print("🎯 初始化ROI管理器...")
    roi_manager = ROIManager(config)
    
    # ROI交互式选择
    ret, first_frame = cap.read()
    if not ret:
        print("❌ 无法读取视频第一帧")
        return
    
    print("\n🎯 ROI选择说明:")
    print("1. 请按顺序点击4个点来定义网球场兴趣区域")
    print("2. 建议选择网球场的主要活动区域")
    print("3. 按顺序点击四个角点（可以是任意四边形）")
    print("4. 点击完4个点后按'c'确认，按'r'重置，按'q'取消")
    
    selected_points = roi_manager.interactive_roi_selection(first_frame.copy(), "ROI选择 - 网球场兴趣区域")
    
    if not selected_points or len(selected_points) != 4:
        print("⚠️ ROI选择取消，将处理整个画面")
    else:
        print(f"✅ ROI选择完成: {selected_points}")
        # 保存ROI配置
        roi_manager.save_roi_config("configs/roi_demo_config.yaml")
    
    # 初始化检测模块
    print("🤖 初始化检测模块...")
    
    try:
        pose_module = PoseEstimator(config['yolo_pose_model_path'], config, roi_manager)
        print("✅ 姿态检测模块初始化完成")
    except Exception as e:
        print(f"⚠️ 姿态检测模块初始化失败: {e}")
        pose_module = None
    
    try:
        ball_module = BallTracker(None, config, roi_manager)
        print("✅ 球检测模块初始化完成")
    except Exception as e:
        print(f"⚠️ 球检测模块初始化失败: {e}")
        ball_module = None
    
    try:
        racket_module = RacketDetector(config['racket_yolo_model_path'], config, roi_manager)
        print("✅ 球拍检测模块初始化完成")
    except Exception as e:
        print(f"⚠️ 球拍检测模块初始化失败: {e}")
        racket_module = None
    
    # 初始化增强动作捕捉
    motion_capture = None
    if config.get('roi_motion_capture', {}).get('enabled', False):
        try:
            motion_capture = EnhancedMotionCapture(config)
            print("✅ 增强动作捕捉模块初始化完成")
        except Exception as e:
            print(f"⚠️ 增强动作捕捉模块初始化失败: {e}")
    
    print("\n🎬 开始视频处理...")
    print("控制键:")
    print("  空格键: 暂停/播放")
    print("  'q'键: 退出")
    print("  's'键: 保存当前帧")
    print("  'r'键: 重新选择ROI")
    
    # 重置视频到开始
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_num = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("📹 视频播放完毕")
                break
        
        display_frame = frame.copy()
        
        # 姿态检测
        pose_results = []
        if pose_module:
            try:
                pose_results = pose_module.get_keypoints(display_frame)
                if pose_results:
                    display_frame = pose_module.draw_keypoints(display_frame, pose_results)
            except Exception as e:
                print(f"姿态检测错误: {e}")
        
        # 球检测
        ball_positions = []
        if ball_module:
            try:
                ball_positions = ball_module.predict_ball(display_frame)
                if ball_positions:
                    for ball_pos in ball_positions:
                        cv2.circle(display_frame, (int(ball_pos[0]), int(ball_pos[1])), 8, (0, 255, 0), 3)
            except Exception as e:
                print(f"球检测错误: {e}")
        
        # 球拍检测
        racket_results = []
        if racket_module:
            try:
                racket_results = racket_module.detect_rackets(display_frame)
                if racket_results:
                    for racket in racket_results:
                        box = racket['box']
                        cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (255, 165, 0), 3)
            except Exception as e:
                print(f"球拍检测错误: {e}")
        
        # ROI增强动作捕捉
        if motion_capture and roi_manager.is_roi_set:
            try:
                motion_analysis = motion_capture.analyze_roi_motion(
                    pose_results, ball_positions, racket_results, frame_num
                )
                display_frame = motion_capture.draw_motion_analysis(display_frame, motion_analysis)
            except Exception as e:
                print(f"动作捕捉错误: {e}")
        
        # 绘制ROI
        if roi_manager.is_roi_set:
            display_frame = roi_manager.draw_roi(display_frame, True)
            
            # 高亮ROI内的检测结果
            if ball_positions:
                display_frame = roi_manager.highlight_roi_detections(display_frame, ball_positions, "ball")
            if racket_results:
                display_frame = roi_manager.highlight_roi_detections(display_frame, racket_results, "racket")
        
        # 显示统计信息
        info_y = 30
        cv2.putText(display_frame, f"Frame: {frame_num}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 30
        
        if roi_manager.is_roi_set:
            roi_stats = roi_manager.get_roi_stats()
            cv2.putText(display_frame, f"ROI Area: {roi_stats.get('roi_area', 0):.0f}px²", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            info_y += 25
        
        cv2.putText(display_frame, f"Poses: {len(pose_results)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
        info_y += 25
        
        cv2.putText(display_frame, f"Balls: {len(ball_positions)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += 25
        
        cv2.putText(display_frame, f"Rackets: {len(racket_results)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # 显示控制提示
        control_info = "SPACE: Pause/Play | Q: Quit | S: Save | R: Re-select ROI"
        if paused:
            control_info = "PAUSED - " + control_info
        
        cv2.putText(display_frame, control_info, 
                   (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示视频
        cv2.imshow("ROI 兴趣区域检测演示", display_frame)
        
        # 处理按键
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        
        if key == ord('q'):
            print("🛑 用户退出")
            break
        elif key == ord(' '):  # 空格键暂停/播放
            paused = not paused
            print(f"⏸️ {'暂停' if paused else '播放'}")
        elif key == ord('s'):  # 保存当前帧
            save_path = f"roi_demo_frame_{frame_num:04d}.jpg"
            cv2.imwrite(save_path, display_frame)
            print(f"💾 当前帧已保存: {save_path}")
        elif key == ord('r'):  # 重新选择ROI
            print("🎯 重新选择ROI...")
            selected_points = roi_manager.interactive_roi_selection(frame.copy(), "重新选择ROI")
            if selected_points and len(selected_points) == 4:
                print(f"✅ ROI重新选择完成: {selected_points}")
                roi_manager.save_roi_config("configs/roi_demo_config.yaml")
            else:
                print("⚠️ ROI重新选择取消")
        
        if not paused:
            frame_num += 1
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n📊 演示完成统计:")
    if roi_manager.is_roi_set:
        final_stats = roi_manager.get_roi_stats()
        print(f"ROI面积: {final_stats.get('roi_area', 0):.0f} 像素²")
        print(f"ROI边界框: {final_stats.get('roi_bbox', {})}")
    
    if motion_capture:
        motion_stats = motion_capture.motion_stats
        print(f"姿态检测: {motion_stats.get('total_pose_detections', 0)} 次")
        print(f"球检测: {motion_stats.get('total_ball_detections', 0)} 次")
        print(f"球拍检测: {motion_stats.get('total_racket_detections', 0)} 次")
        print(f"挥拍事件: {motion_stats.get('swing_events', 0)} 次")
    
    print("🎯 ROI兴趣区域检测演示结束")

if __name__ == "__main__":
    main()
