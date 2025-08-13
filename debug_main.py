#!/usr/bin/env python3
"""
调试版main.py - 逐步显示初始化过程
"""

import cv2
import yaml
import os
import sys
import time

def load_config(config_path="configs/default_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def debug_main(config_path="configs/roi_enabled_config.yaml"):
    print("🐛 调试版main.py启动")
    print("=" * 50)
    
    # 1. 加载配置
    print("📁 步骤1: 加载配置...")
    config = load_config(config_path)
    print("✅ 配置加载完成")
    
    # 2. 检查视频
    print("\n📹 步骤2: 检查视频文件...")
    video_path = config['video_input_path']
    print(f"   视频路径: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    print("✅ 视频文件存在")
    
    # 3. 测试视频读取
    print("\n🎬 步骤3: 测试视频读取...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频文件")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取视频第一帧")
        cap.release()
        return
    
    frame_height, frame_width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ 视频读取成功:")
    print(f"   - 分辨率: {frame_width}x{frame_height}")
    print(f"   - FPS: {fps}")
    print(f"   - 总帧数: {total_frames}")
    cap.release()
    
    # 4. ROI初始化
    print("\n🎯 步骤4: ROI管理器初始化...")
    try:
        from roi_manager import ROIManager
        roi_manager = ROIManager(config)
        print("✅ ROI管理器创建成功")
        
        # ROI配置检查
        roi_settings = config.get('roi_settings', {})
        print(f"   ROI启用: {roi_settings.get('enabled', False)}")
        
        if roi_settings.get('enabled', False):
            # 尝试加载ROI配置
            roi_config_path = roi_settings.get('roi_config_path', 'configs/roi_config.yaml')
            if os.path.exists(roi_config_path):
                print(f"   正在加载ROI配置: {roi_config_path}")
                if roi_manager.load_roi_config(roi_config_path):
                    print("✅ ROI配置加载成功")
                    print(f"   - ROI设置状态: {roi_manager.is_roi_set}")
                    print(f"   - ROI点数: {len(roi_manager.roi_points)}")
                else:
                    print("⚠️ ROI配置加载失败")
            else:
                print(f"⚠️ ROI配置文件不存在: {roi_config_path}")
        else:
            print("   ROI功能未启用")
            
    except Exception as e:
        print(f"❌ ROI初始化失败: {e}")
        return
    
    # 5. 检测器初始化
    print("\n🤖 步骤5: 检测器初始化...")
    
    # 5.1 姿态检测器
    print("   5.1 初始化姿态检测器...")
    try:
        from pose_estimator import PoseEstimator
        pose_module = PoseEstimator(config['yolo_pose_model_path'], config, roi_manager)
        print("   ✅ 姿态检测器初始化成功")
    except Exception as e:
        print(f"   ❌ 姿态检测器初始化失败: {e}")
        return
    
    # 5.2 球检测器
    print("   5.2 初始化球检测器...")
    try:
        from ball_tracker import BallTracker
        ball_module = BallTracker(config.get('tracknet_model_path', None), config, roi_manager)
        print("   ✅ 球检测器初始化成功")
    except Exception as e:
        print(f"   ❌ 球检测器初始化失败: {e}")
        print(f"   错误详情: {str(e)}")
        # 继续运行，球检测器失败不影响ROI演示
        ball_module = None
    
    # 5.3 球拍检测器
    print("   5.3 初始化球拍检测器...")
    try:
        from racket_detector import RacketDetector
        racket_module = RacketDetector(config['racket_yolo_model_path'], config, roi_manager)
        print("   ✅ 球拍检测器初始化成功")
    except Exception as e:
        print(f"   ❌ 球拍检测器初始化失败: {e}")
        racket_module = None
    
    # 6. ROI演示
    print("\n🎯 步骤6: ROI功能演示...")
    if roi_manager.is_roi_set:
        print("✅ ROI已设置，开始演示...")
        
        # 重新打开视频
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        
        if ret:
            # 绘制ROI
            display_frame = frame.copy()
            display_frame = roi_manager.draw_roi(display_frame, show_fill=True)
            
            # 添加状态信息
            cv2.putText(display_frame, "ROI Debug Demo - Press any key to close", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(display_frame, f"ROI: Active ({len(roi_manager.roi_points)} points)", 
                       (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 显示窗口
            cv2.namedWindow("ROI Debug Demo", cv2.WINDOW_NORMAL)
            cv2.imshow("ROI Debug Demo", display_frame)
            
            print("🎨 ROI演示窗口已打开，按任意键关闭...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        cap.release()
        
    else:
        print("⚠️ ROI未设置，跳过演示")
    
    print("\n✅ 调试完成!")

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/roi_enabled_config.yaml"
    debug_main(config_path)
