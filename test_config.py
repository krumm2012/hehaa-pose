#!/usr/bin/env python3
import yaml
import os

def test_config():
    """测试配置文件加载和参数设置"""
    
    config_path = 'configs/default_config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print('✅ 配置文件加载成功')
        
        # 检查人脸检测配置
        advanced_config = config.get('advanced_face_detection', {})
        face_constraints = advanced_config.get('face_size_constraints', {})
        quality_filters = advanced_config.get('quality_filters', {})
        detector_control = advanced_config.get('detector_control', {})
        
        print(f'📊 人脸检测参数设置:')
        print(f'   - 最小置信度: {quality_filters.get("min_confidence", "未设置")}')
        print(f'   - 最大人脸尺寸: {face_constraints.get("max_face_size", "未设置")}px')
        print(f'   - 最小人脸尺寸: {face_constraints.get("min_face_size", "未设置")}px') 
        print(f'   - 每帧最大检测数: {quality_filters.get("max_detections_per_frame", "未设置")}')
        print(f'   - 宽高比范围: {face_constraints.get("min_aspect_ratio", "未设置")}-{face_constraints.get("max_aspect_ratio", "未设置")}')
        print(f'   - OpenCV质量过滤: {quality_filters.get("enable_opencv_quality_filter", "未设置")}')
        
        print(f'\n🔧 检测器启用状态:')
        print(f'   - 正面人脸检测器: {"✅" if detector_control.get("enable_front_face", True) else "❌"}')
        print(f'   - 备选正面检测器: {"✅" if detector_control.get("enable_front_face_alt", True) else "❌"}')
        print(f'   - 第二备选检测器: {"✅" if detector_control.get("enable_front_face_alt2", True) else "❌"}')
        print(f'   - 侧脸检测器: {"✅" if detector_control.get("enable_profile_face", False) else "❌"}')
        print(f'   - 全身检测器: {"✅" if detector_control.get("enable_full_body", False) else "❌"}')
        
        return True
    else:
        print('❌ 配置文件不存在')
        return False

if __name__ == "__main__":
    test_config() 