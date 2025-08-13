#!/usr/bin/env python3
"""
性能基准测试 - 对比原版和优化版的处理速度
"""

import time
import subprocess
import sys
import os
import cv2
import psutil

def get_video_info(video_path):
    """获取视频基本信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    info = {
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    cap.release()
    return info

def monitor_system_resources():
    """监控系统资源使用情况"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_used_gb': psutil.virtual_memory().used / (1024**3)
    }

def run_benchmark_test(script_name, config_path, test_frames=100):
    """运行性能测试"""
    print(f"\n🧪 测试 {script_name}")
    print("=" * 50)
    
    # 获取系统资源使用情况（开始）
    start_resources = monitor_system_resources()
    print(f"📊 开始资源使用: CPU {start_resources['cpu_percent']:.1f}%, "
          f"内存 {start_resources['memory_percent']:.1f}% ({start_resources['memory_used_gb']:.1f}GB)")
    
    # 修改配置以限制处理帧数（测试用）
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行脚本（限制时间避免测试过长）
        cmd = [sys.executable, script_name, config_path]
        print(f"🚀 执行命令: {' '.join(cmd)}")
        
        # 设置超时时间
        timeout = 60  # 60秒超时
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            end_time = time.time()
            
            # 获取系统资源使用情况（结束）
            end_resources = monitor_system_resources()
            
            execution_time = end_time - start_time
            
            print(f"✅ 执行完成")
            print(f"⏱️  执行时间: {execution_time:.2f}秒")
            print(f"📊 结束资源使用: CPU {end_resources['cpu_percent']:.1f}%, "
                  f"内存 {end_resources['memory_percent']:.1f}% ({end_resources['memory_used_gb']:.1f}GB)")
            
            # 分析输出中的性能信息
            if "平均处理速度" in stdout:
                lines = stdout.split('\n')
                for line in lines:
                    if "平均处理速度" in line or "FPS" in line:
                        print(f"📈 {line.strip()}")
            
            if stderr:
                print(f"⚠️ 错误输出: {stderr}")
            
            return {
                'success': True,
                'execution_time': execution_time,
                'start_resources': start_resources,
                'end_resources': end_resources,
                'stdout': stdout,
                'stderr': stderr
            }
            
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"⏰ 测试超时 ({timeout}秒)")
            return {
                'success': False,
                'execution_time': timeout,
                'error': 'timeout'
            }
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return {
            'success': False,
            'execution_time': time.time() - start_time,
            'error': str(e)
        }

def main():
    """主性能测试函数"""
    print("🎯 网球分析性能基准测试")
    print("=" * 60)
    
    # 检查视频文件
    video_path = "data/input_video.mp4"
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 获取视频信息
    video_info = get_video_info(video_path)
    if video_info:
        print(f"📹 视频信息:")
        print(f"   - 分辨率: {video_info['width']}x{video_info['height']}")
        print(f"   - 帧率: {video_info['fps']:.1f} FPS")
        print(f"   - 总帧数: {video_info['frame_count']}")
        print(f"   - 预计时长: {video_info['frame_count']/video_info['fps']:.1f}秒")
    
    # 系统信息
    print(f"\n💻 系统信息:")
    print(f"   - CPU核心数: {psutil.cpu_count()}")
    print(f"   - 总内存: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    print(f"   - Python版本: {sys.version}")
    
    # 测试用例
    test_cases = [
        {
            'name': '原版main.py',
            'script': 'main.py',
            'config': 'configs/roi_enabled_config.yaml'
        },
        {
            'name': '优化版main_optimized.py',
            'script': 'main_optimized.py',
            'config': 'configs/performance_optimized_config.yaml'
        }
    ]
    
    results = {}
    
    # 运行测试
    for test_case in test_cases:
        if os.path.exists(test_case['script']) and os.path.exists(test_case['config']):
            results[test_case['name']] = run_benchmark_test(
                test_case['script'], 
                test_case['config']
            )
        else:
            print(f"⚠️ 跳过 {test_case['name']}: 文件不存在")
            results[test_case['name']] = {'success': False, 'error': 'file_not_found'}
    
    # 性能对比报告
    print("\n📊 性能对比报告")
    print("=" * 60)
    
    successful_tests = {k: v for k, v in results.items() if v.get('success', False)}
    
    if len(successful_tests) >= 2:
        test_names = list(successful_tests.keys())
        original = successful_tests[test_names[0]]
        optimized = successful_tests[test_names[1]]
        
        time_improvement = ((original['execution_time'] - optimized['execution_time']) / original['execution_time']) * 100
        
        print(f"🚀 性能改进:")
        print(f"   - 原版执行时间: {original['execution_time']:.2f}秒")
        print(f"   - 优化版执行时间: {optimized['execution_time']:.2f}秒")
        print(f"   - 性能提升: {time_improvement:.1f}%")
        
        if time_improvement > 0:
            print(f"✅ 优化版速度提升 {time_improvement:.1f}%")
        else:
            print(f"⚠️ 优化版性能未达预期")
    
    else:
        print("⚠️ 需要至少两个成功的测试来进行对比")
    
    # 优化建议
    print(f"\n💡 优化建议:")
    print(f"   1. 如果CPU使用率低，可以增加并行线程数")
    print(f"   2. 如果内存使用率高，可以增加跳帧设置")
    print(f"   3. 如果处理速度仍然慢，考虑:")
    print(f"      - 降低视频分辨率")
    print(f"      - 关闭不必要的可视化选项")
    print(f"      - 使用更小的AI模型")
    print(f"      - 启用无头模式（不显示窗口）")

if __name__ == "__main__":
    main()
