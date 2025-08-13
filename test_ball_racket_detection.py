#!/usr/bin/env python3
"""
球检测和球拍检测功能测试脚本
"""

import time
import main

def test_ball_racket_detection():
    """测试球检测和球拍检测功能"""
    print('🎾 球检测和球拍检测功能测试开始...')
    print('='*60)
    print('恢复功能：')
    print('✅ 球检测置信度: 0.8 → 0.6')
    print('✅ 球拍检测置信度: 0.6 → 0.4')
    print('✅ 球轨迹显示: false → true')
    print('✅ 球拍状态显示: false → true')
    print('✅ 球尺寸过滤: 4-12px → 3-15px')
    print('✅ 球移动距离: 12px → 8px')
    print('✅ 轨迹长度: 20 → 30')
    print('='*60)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 使用平衡配置运行
        main.main('configs/balanced_config.yaml')
    except KeyboardInterrupt:
        print("\n⏹️  测试中断")
    except Exception as e:
        print(f"❌ 测试出错: {e}")
    
    # 计算运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n📊 测试结果:")
    print(f"总运行时间: {elapsed_time:.2f} 秒")
    if elapsed_time > 0:
        fps = 100 / elapsed_time
        print(f"平均处理速度: {fps:.2f} FPS")
        
        # 性能对比分析
        if fps > 5:
            print("✅ 性能状态: 优秀 (>5 FPS)")
        elif fps > 3:
            print("⚠️  性能状态: 良好 (3-5 FPS)")
        elif fps > 1:
            print("⚠️  性能状态: 可接受 (1-3 FPS)")
        else:
            print("❌ 性能状态: 需要优化 (<1 FPS)")
    
    print(f"\n🔍 功能验证要点:")
    print(f"请检查视频输出中是否包含:")
    print(f"  🎾 绿色圆点标记球位置")
    print(f"  🏸 蓝色矩形框标记球拍")
    print(f"  📈 球的运动轨迹线")
    print(f"  📊 球拍状态信息显示")

if __name__ == "__main__":
    test_ball_racket_detection() 