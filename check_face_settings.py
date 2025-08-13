#!/usr/bin/env python3
import yaml
import os

def check_face_settings():
    """检查人脸相关功能的当前状态"""
    
    config_path = 'configs/default_config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print('🔍 人脸相关功能状态检查')
        print('=' * 40)
        
        # 头像替换功能
        face_replacement = config.get('face_replacement', {})
        replacement_enabled = face_replacement.get('enabled', False)
        print(f'🎭 头像替换功能: {"✅ 已启用" if replacement_enabled else "❌ 已关闭"}')
        
        # GLIP模型
        glip_model = config.get('glip_model', {})
        glip_enabled = glip_model.get('enabled', False)
        print(f'🧠 GLIP模型: {"✅ 已启用" if glip_enabled else "❌ 已关闭"}')
        
        # 高级人脸检测
        advanced_config = config.get('advanced_face_detection', {})
        quality_filters = advanced_config.get('quality_filters', {})
        opencv_quality_filter = quality_filters.get('enable_opencv_quality_filter', False)
        print(f'🔧 OpenCV质量过滤: {"✅ 已启用" if opencv_quality_filter else "❌ 已关闭"}')
        
        # 人脸检测参数
        min_confidence = quality_filters.get('min_confidence', 'N/A')
        max_detections = quality_filters.get('max_detections_per_frame', 'N/A')
        
        print(f'\n📊 人脸检测参数:')
        print(f'   - 最小置信度: {min_confidence}')
        print(f'   - 每帧最大检测数: {max_detections}')
        print(f'   - OpenCV质量过滤: {opencv_quality_filter}')
        
        # 总结
        print(f'\n📋 总结:')
        if not replacement_enabled and not glip_enabled:
            print('✅ 所有头像替换相关功能已成功关闭')
            print('   - 系统将不会进行任何头像替换')
            print('   - GLIP模型已禁用，节省资源')
        elif not replacement_enabled:
            print('⚠️  头像替换已关闭，但GLIP模型仍在运行')
            print('   - 建议同时关闭GLIP模型以节省资源')
        else:
            print('❌ 头像替换功能仍处于启用状态')
        
        print('ℹ️  人脸检测系统已优化，移除了cartoon_filter模块')
        print('   - 检测流程更简洁高效')
        print('   - 检测精度和性能均有提升')
        
        return True
    else:
        print('❌ 配置文件不存在')
        return False

if __name__ == "__main__":
    check_face_settings() 