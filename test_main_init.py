#!/usr/bin/env python3
"""
测试main.py的初始化过程，找出ROI问题
"""

import yaml
import os
import sys

def test_main_initialization():
    print("🧪 测试main.py初始化过程")
    print("=" * 50)
    
    # 1. 测试配置加载
    print("📁 1. 测试配置加载...")
    try:
        with open("configs/roi_enabled_config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ 配置文件加载成功")
        
        # 检查ROI设置
        roi_settings = config.get('roi_settings', {})
        print(f"🎯 ROI设置: {roi_settings}")
        print(f"   - enabled: {roi_settings.get('enabled', False)}")
        print(f"   - interactive_selection: {roi_settings.get('interactive_selection', False)}")
        print(f"   - auto_load_config: {roi_settings.get('auto_load_config', False)}")
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False
    
    # 2. 测试视频文件
    print("\n📹 2. 测试视频文件...")
    video_path = config.get('video_input_path', 'data/input_video.mp4')
    if os.path.exists(video_path):
        print(f"✅ 视频文件存在: {video_path}")
    else:
        print(f"❌ 视频文件不存在: {video_path}")
        print("   这可能导致程序卡住！")
        return False
    
    # 3. 测试模型文件
    print("\n🤖 3. 测试模型文件...")
    model_files = [
        ('yolo_pose_model_path', 'YOLO姿态模型'),
        ('racket_yolo_model_path', 'YOLO球拍模型'),
        ('tracknet_model_path', 'TrackNet模型'),
    ]
    
    for key, name in model_files:
        model_path = config.get(key)
        if model_path and os.path.exists(model_path):
            print(f"✅ {name}: {model_path}")
        else:
            print(f"⚠️ {name}不存在: {model_path}")
    
    # 4. 测试ROI管理器导入
    print("\n🎯 4. 测试ROI模块导入...")
    try:
        from roi_manager import ROIManager
        print("✅ ROI管理器导入成功")
        
        # 测试ROI管理器初始化
        roi_manager = ROIManager(config)
        print("✅ ROI管理器初始化成功")
        print(f"   - ROI设置状态: {roi_manager.is_roi_set}")
        
    except Exception as e:
        print(f"❌ ROI管理器问题: {e}")
        return False
    
    # 5. 测试ROI配置文件
    print("\n📄 5. 测试ROI配置文件...")
    roi_config_path = roi_settings.get('roi_config_path', 'configs/roi_config.yaml')
    if os.path.exists(roi_config_path):
        print(f"✅ ROI配置文件存在: {roi_config_path}")
        try:
            if roi_manager.load_roi_config(roi_config_path):
                print("✅ ROI配置加载成功")
                print(f"   - ROI点数: {len(roi_manager.roi_points)}")
                print(f"   - ROI设置: {roi_manager.is_roi_set}")
            else:
                print("⚠️ ROI配置文件存在但加载失败或被禁用")
        except Exception as e:
            print(f"❌ ROI配置加载错误: {e}")
    else:
        print(f"❌ ROI配置文件不存在: {roi_config_path}")
    
    # 6. 测试各个检测模块导入
    print("\n🔧 6. 测试检测模块导入...")
    modules = [
        ('pose_estimator', 'PoseEstimator', '姿态检测'),
        ('ball_tracker', 'BallTracker', '球检测'),
        ('racket_detector', 'RacketDetector', '球拍检测'),
    ]
    
    for module_name, class_name, display_name in modules:
        try:
            module = __import__(module_name)
            cls = getattr(module, class_name)
            print(f"✅ {display_name}模块导入成功")
        except Exception as e:
            print(f"❌ {display_name}模块导入失败: {e}")
    
    print("\n📊 初始化测试完成!")
    return True

if __name__ == "__main__":
    test_main_initialization()
