# diagnose_roi_integration.py
"""
ROI集成问题全面诊断脚本
分析ROI功能在整个系统中的集成状态和问题
"""

import cv2
import yaml
import os
from roi_manager import ROIManager
from ball_tracker import BallTracker
from pose_estimator import PoseEstimator
from racket_detector import RacketDetector

def diagnose_roi_integration():
    print("🔍 ROI集成问题全面诊断")
    print("=" * 60)
    
    # 1. 检查配置文件
    print("\n📁 1. 配置文件检查")
    print("-" * 30)
    
    config_files = [
        "configs/roi_enabled_config.yaml",
        "configs/roi_config.yaml"
    ]
    
    configs = {}
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            configs[config_file] = config
            print(f"✅ {config_file}: 加载成功")
            
            # 检查ROI相关配置
            if 'roi_settings' in config:
                roi_settings = config['roi_settings']
                print(f"   ROI启用: {roi_settings.get('enabled', False)}")
                print(f"   交互选择: {roi_settings.get('interactive_selection', False)}")
                print(f"   自动加载: {roi_settings.get('auto_load_config', False)}")
            
            if 'roi_enabled' in config:
                print(f"   ROI启用状态: {config.get('roi_enabled', False)}")
                print(f"   ROI点数: {len(config.get('roi_points', []))}")
                
        except Exception as e:
            print(f"❌ {config_file}: 加载失败 - {e}")
    
    # 使用主配置
    main_config = configs.get("configs/roi_enabled_config.yaml", {})
    
    # 2. 检查ROI管理器初始化
    print(f"\n🎯 2. ROI管理器初始化检查")
    print("-" * 30)
    
    try:
        roi_manager = ROIManager(main_config)
        print("✅ ROI管理器初始化成功")
        
        # 尝试加载ROI配置
        roi_config_path = "configs/roi_config.yaml"
        if os.path.exists(roi_config_path):
            load_success = roi_manager.load_roi_config(roi_config_path)
            print(f"{'✅' if load_success else '❌'} ROI配置加载: {load_success}")
            print(f"   ROI设置状态: {roi_manager.is_roi_set}")
            if roi_manager.is_roi_set:
                print(f"   ROI点数: {len(roi_manager.roi_points)}")
                print(f"   ROI坐标: {roi_manager.roi_points}")
                stats = roi_manager.get_roi_stats()
                print(f"   ROI面积: {stats.get('roi_area', 0):.0f} 像素²")
        else:
            print(f"❌ ROI配置文件不存在: {roi_config_path}")
            
    except Exception as e:
        print(f"❌ ROI管理器初始化失败: {e}")
        return
    
    # 3. 检查检测模块初始化
    print(f"\n🤖 3. 检测模块初始化检查")
    print("-" * 30)
    
    detection_modules = {}
    
    # BallTracker
    try:
        ball_tracker = BallTracker(None, main_config, roi_manager)
        detection_modules['ball_tracker'] = ball_tracker
        print("✅ BallTracker初始化成功")
        print(f"   ROI管理器引用: {'✅' if ball_tracker.roi_manager else '❌'}")
        print(f"   ROI设置状态: {'✅' if ball_tracker.roi_manager and ball_tracker.roi_manager.is_roi_set else '❌'}")
    except Exception as e:
        print(f"❌ BallTracker初始化失败: {e}")
    
    # PoseEstimator  
    try:
        pose_estimator = PoseEstimator("models/yolov8n-pose.pt", main_config, roi_manager)
        detection_modules['pose_estimator'] = pose_estimator
        print("✅ PoseEstimator初始化成功")
        print(f"   ROI管理器引用: {'✅' if pose_estimator.roi_manager else '❌'}")
        print(f"   ROI设置状态: {'✅' if pose_estimator.roi_manager and pose_estimator.roi_manager.is_roi_set else '❌'}")
    except Exception as e:
        print(f"❌ PoseEstimator初始化失败: {e}")
    
    # RacketDetector
    try:
        racket_detector = RacketDetector("yolov8s.pt", main_config, roi_manager)
        detection_modules['racket_detector'] = racket_detector
        print("✅ RacketDetector初始化成功")
        print(f"   ROI管理器引用: {'✅' if racket_detector.roi_manager else '❌'}")
        print(f"   ROI设置状态: {'✅' if racket_detector.roi_manager and racket_detector.roi_manager.is_roi_set else '❌'}")
    except Exception as e:
        print(f"❌ RacketDetector初始化失败: {e}")
    
    # 4. 检查视频文件和帧处理
    print(f"\n📹 4. 视频处理检查")
    print("-" * 30)
    
    video_path = main_config.get('video_input_path', 'data/input_video.mp4')
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ 无法读取视频帧")
        return
    
    print(f"✅ 视频处理: {frame.shape}")
    
    # 5. 测试ROI过滤功能
    print(f"\n🧪 5. ROI过滤功能测试")
    print("-" * 30)
    
    if roi_manager.is_roi_set:
        # 测试点在ROI内判断
        test_points = [
            (100, 100),
            (1000, 500),
            (1500, 700),
            (frame.shape[1]//2, frame.shape[0]//2)
        ]
        
        for point in test_points:
            is_inside = roi_manager.is_point_in_roi(point)
            print(f"   点 {point}: {'✅ 在ROI内' if is_inside else '❌ 在ROI外'}")
        
        # 测试检测过滤
        if 'ball_tracker' in detection_modules:
            try:
                print(f"\n   球检测过滤测试:")
                # 模拟一些球检测结果
                mock_balls = [(100, 100), (1000, 500), (2000, 1000)]
                filtered_balls = roi_manager.filter_detections_by_roi(mock_balls, "ball")
                print(f"   原始球数: {len(mock_balls)}")
                print(f"   过滤后球数: {len(filtered_balls)}")
                print(f"   过滤后位置: {filtered_balls}")
            except Exception as e:
                print(f"   ❌ 球检测过滤测试失败: {e}")
    else:
        print("❌ ROI未设置，无法测试过滤功能")
    
    # 6. 检查调试日志配置
    print(f"\n📝 6. 调试日志配置检查")
    print("-" * 30)
    
    ball_debug_config = main_config.get('ball_detection_debug', {})
    print(f"球检测调试启用: {ball_debug_config.get('enabled', False)}")
    print(f"HSV检测日志: {ball_debug_config.get('log_hsv_detection', False)}")
    print(f"边界检查日志: {ball_debug_config.get('log_boundary_check', False)}")
    print(f"ROI过滤日志: {ball_debug_config.get('log_roi_filtering', 'Not configured')}")
    
    # 7. 检查可视化配置
    print(f"\n🎨 7. ROI可视化配置检查")
    print("-" * 30)
    
    roi_settings = main_config.get('roi_settings', {})
    visualization = roi_settings.get('visualization', {})
    
    print(f"显示ROI边界: {visualization.get('show_roi_boundary', False)}")
    print(f"显示ROI填充: {visualization.get('show_roi_fill', False)}")
    print(f"显示ROI角点: {visualization.get('show_roi_points', False)}")
    print(f"高亮检测结果: {visualization.get('highlight_detections', False)}")
    print(f"显示ROI统计: {visualization.get('show_roi_stats', False)}")
    
    # 8. 提供问题诊断和建议
    print(f"\n💡 8. 问题诊断和建议")
    print("-" * 30)
    
    issues_found = []
    recommendations = []
    
    # 检查ROI是否设置
    if not roi_manager.is_roi_set:
        issues_found.append("ROI未设置")
        recommendations.append("运行 python test_roi_simple.py 设置ROI")
    
    # 检查调试日志
    if not ball_debug_config.get('enabled', False):
        issues_found.append("球检测调试日志未启用")
        recommendations.append("在配置文件中设置 ball_detection_debug.enabled: true")
    
    # 检查可视化设置
    if not any(visualization.values()):
        issues_found.append("ROI可视化设置全部关闭")
        recommendations.append("启用至少一个ROI可视化选项")
    
    # 检查检测模块ROI引用
    for module_name, module in detection_modules.items():
        if not hasattr(module, 'roi_manager') or not module.roi_manager:
            issues_found.append(f"{module_name} 缺少ROI管理器引用")
            recommendations.append(f"确保 {module_name} 初始化时传入roi_manager参数")
    
    if issues_found:
        print("🚨 发现的问题:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print(f"\n💡 建议解决方案:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("✅ 未发现明显问题，ROI集成应该正常工作")
    
    # 9. 生成测试报告
    print(f"\n📊 9. 测试报告摘要")
    print("-" * 30)
    
    total_checks = 8
    passed_checks = 0
    
    if configs:
        passed_checks += 1
    if roi_manager.is_roi_set:
        passed_checks += 1
    if detection_modules:
        passed_checks += len(detection_modules) * 0.33
    if os.path.exists(video_path):
        passed_checks += 1
    if not issues_found:
        passed_checks += 2
    
    success_rate = (passed_checks / total_checks) * 100
    print(f"检查通过率: {success_rate:.1f}%")
    print(f"ROI设置状态: {'✅ 已设置' if roi_manager.is_roi_set else '❌ 未设置'}")
    print(f"检测模块数: {len(detection_modules)}/3")
    print(f"发现问题数: {len(issues_found)}")
    
    if success_rate >= 80:
        print("🎉 ROI集成状态良好！")
    elif success_rate >= 60:
        print("⚠️ ROI集成存在一些问题，需要修复")
    else:
        print("🚨 ROI集成存在严重问题，需要全面检查")
    
    print("\n" + "=" * 60)
    print("🔍 诊断完成")

if __name__ == "__main__":
    diagnose_roi_integration()
