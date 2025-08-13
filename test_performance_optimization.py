#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎾 网球分析器性能测试脚本
比较不同配置文件的处理速度和效果
"""

import os
import sys
import time
import subprocess
import yaml
from pathlib import Path

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return None

def run_performance_test(config_path, test_duration=30):
    """运行性能测试"""
    print(f"\n🚀 开始测试配置文件: {config_path}")
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    
    # 检查输入视频是否存在
    config = load_config(config_path)
    if not config:
        return None
    
    video_path = config.get('video_input_path', 'data/input_video.mp4')
    if not os.path.exists(video_path):
        print(f"❌ 输入视频不存在: {video_path}")
        return None
    
    # 启动测试进程
    start_time = time.time()
    try:
        # 使用subprocess启动程序
        process = subprocess.Popen(
            [sys.executable, 'main.py', '--config', config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待指定时间或进程结束
        try:
            stdout, stderr = process.communicate(timeout=test_duration)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 分析输出
        fps_info = extract_fps_info(stdout)
        
        return {
            'config_path': config_path,
            'duration': duration,
            'fps': fps_info.get('fps', 0),
            'total_frames': fps_info.get('total_frames', 0),
            'success': process.returncode == 0,
            'stdout': stdout,
            'stderr': stderr
        }
        
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        return None

def extract_fps_info(output):
    """从输出中提取FPS信息"""
    fps = 0
    total_frames = 0
    
    lines = output.split('\n')
    for line in lines:
        if '平均处理速度:' in line:
            try:
                fps = float(line.split(':')[1].strip().replace(' FPS', ''))
            except:
                pass
        elif '共处理' in line and '帧' in line:
            try:
                total_frames = int(line.split('共处理')[1].split('帧')[0].strip())
            except:
                pass
    
    return {'fps': fps, 'total_frames': total_frames}

def compare_configs():
    """比较不同配置文件的性能"""
    configs_to_test = [
        'configs/comprehensive_tennis_config.yaml',
        'configs/performance_optimized_config.yaml'
    ]
    
    print("🎾 网球分析器性能测试")
    print("=" * 50)
    
    results = []
    
    for config_path in configs_to_test:
        if os.path.exists(config_path):
            result = run_performance_test(config_path, test_duration=60)
            if result:
                results.append(result)
        else:
            print(f"⚠️  配置文件不存在: {config_path}")
    
    # 显示结果
    print("\n📊 性能测试结果")
    print("=" * 50)
    
    for result in results:
        config_name = os.path.basename(result['config_path'])
        print(f"\n📁 配置文件: {config_name}")
        print(f"⏱️  测试时长: {result['duration']:.1f}秒")
        print(f"🎯 处理速度: {result['fps']:.2f} FPS")
        print(f"📹 处理帧数: {result['total_frames']}帧")
        print(f"✅ 执行状态: {'成功' if result['success'] else '失败'}")
        
        if result['fps'] > 0:
            print(f"🚀 性能评级: {'优秀' if result['fps'] >= 5 else '良好' if result['fps'] >= 3 else '一般'}")
    
    # 性能对比
    if len(results) >= 2:
        print("\n📈 性能对比")
        print("=" * 50)
        
        fps_values = [r['fps'] for r in results if r['fps'] > 0]
        if len(fps_values) >= 2:
            max_fps = max(fps_values)
            min_fps = min(fps_values)
            improvement = ((max_fps - min_fps) / min_fps) * 100
            
            print(f"🏆 最高FPS: {max_fps:.2f}")
            print(f"📉 最低FPS: {min_fps:.2f}")
            print(f"📈 性能提升: {improvement:.1f}%")
            
            if improvement > 20:
                print("🎉 性能优化效果显著!")
            elif improvement > 10:
                print("👍 性能优化效果良好")
            else:
                print("📊 性能差异较小")

def quick_test():
    """快速测试当前配置"""
    print("⚡ 快速性能测试")
    print("=" * 30)
    
    # 测试当前使用的配置文件
    current_config = 'configs/comprehensive_tennis_config.yaml'
    
    if os.path.exists(current_config):
        print(f"🎯 测试当前配置: {current_config}")
        result = run_performance_test(current_config, test_duration=30)
        
        if result:
            print(f"\n📊 测试结果:")
            print(f"   FPS: {result['fps']:.2f}")
            print(f"   帧数: {result['total_frames']}")
            print(f"   状态: {'✅ 成功' if result['success'] else '❌ 失败'}")
            
            if result['fps'] < 3:
                print("\n💡 建议使用性能优化配置:")
                print("   python main.py --config configs/performance_optimized_config.yaml")
        else:
            print("❌ 测试失败")
    else:
        print(f"❌ 配置文件不存在: {current_config}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_configs()
    else:
        quick_test()


