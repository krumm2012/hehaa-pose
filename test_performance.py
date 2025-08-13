#!/usr/bin/env python3
"""
性能测试脚本 - 使用优化配置文件
"""

import time
import main

def test_performance():
    """测试系统性能"""
    print('🚀 性能优化测试开始...')
    print('='*60)
    print('优化措施：')
    print('✅ 禁用人脸替换功能')
    print('✅ 只使用1个人脸检测器 (front_face)')
    print('✅ 使用轻量级yolov8n模型')
    print('✅ 提高所有置信度阈值')
    print('✅ 简化显示选项')
    print('✅ 禁用轨迹和球拍状态显示')
    print('='*60)
    
    # 记录开始时间
    start_time = time.time()
    
    # 使用优化配置运行
    config = main.load_config('configs/performance_optimized_config.yaml')
    
    try:
        main.main('configs/performance_optimized_config.yaml')
    except KeyboardInterrupt:
        print("\n⏹️  测试中断")
    except Exception as e:
        print(f"❌ 测试出错: {e}")
    
    # 计算运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n📊 性能测试结果:")
    print(f"总运行时间: {elapsed_time:.2f} 秒")
    print(f"平均每秒处理帧数预估: ~{100/elapsed_time:.2f} FPS" if elapsed_time > 0 else "无法计算FPS")

if __name__ == "__main__":
    test_performance() 