#!/usr/bin/env python3
"""
测试不同计算单元的性能
比较 CPU、GPU、ANE 和 ALL 的推理速度
"""

import cv2
import time
import coremltools as ct
from PIL import Image
import numpy as np


def test_compute_unit(model_path, compute_unit, frame, num_runs=10):
    """测试指定计算单元的性能"""
    print(f"\n{'='*60}")
    print(f"测试计算单元: {compute_unit}")
    print(f"{'='*60}")
    
    # 加载模型
    print(f"加载模型...")
    model = ct.models.MLModel(model_path, compute_units=compute_unit)
    
    # 预处理
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (640, 640))
    pil_img = Image.fromarray(resized)
    
    # 预热
    print(f"预热模型...")
    for _ in range(3):
        _ = model.predict({'image': pil_img})
    
    # 测试
    print(f"运行 {num_runs} 次推理...")
    times = []
    for i in range(num_runs):
        start = time.time()
        result = model.predict({'image': pil_img})
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # 转换为毫秒
        
        if i % 5 == 0:
            print(f"  [{i+1}/{num_runs}] {elapsed*1000:.1f}ms")
    
    # 统计
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    
    print(f"\n📊 统计结果:")
    print(f"   平均时间: {avg_time:.1f}ms")
    print(f"   最小时间: {min_time:.1f}ms")
    print(f"   最大时间: {max_time:.1f}ms")
    print(f"   标准差:   {std_time:.1f}ms")
    
    return {
        'compute_unit': str(compute_unit),
        'avg': avg_time,
        'min': min_time,
        'max': max_time,
        'std': std_time
    }


def main():
    print("🧪 Core ML 计算单元性能测试")
    print("="*60)
    
    # 加载测试图像
    model_path = "yolo26n.mlpackage"
    video_path = "data/16.10.mp4"
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ 无法读取视频")
        return
    
    print(f"✅ 测试图像: {frame.shape[1]}x{frame.shape[0]}")
    print(f"✅ 模型路径: {model_path}")
    
    # 测试不同计算单元
    compute_units = [
        ('CPU_ONLY', ct.ComputeUnit.CPU_ONLY),
        ('CPU_AND_GPU', ct.ComputeUnit.CPU_AND_GPU),
        ('CPU_AND_NE', ct.ComputeUnit.CPU_AND_NE),
        ('ALL', ct.ComputeUnit.ALL)
    ]
    
    results = []
    for name, unit in compute_units:
        try:
            result = test_compute_unit(model_path, unit, frame, num_runs=20)
            result['name'] = name
            results.append(result)
        except Exception as e:
            print(f"❌ 测试 {name} 失败: {e}")
    
    # 对比结果
    print(f"\n{'='*60}")
    print("🏆 性能对比")
    print(f"{'='*60}")
    print(f"\n{'计算单元':<20} {'平均时间':<12} {'最小时间':<12} {'标准差':<12}")
    print("-"*60)
    
    for r in results:
        print(f"{r['name']:<20} {r['avg']:>10.1f}ms {r['min']:>10.1f}ms {r['std']:>10.1f}ms")
    
    # 找出最快的
    if results:
        fastest = min(results, key=lambda x: x['avg'])
        print(f"\n🥇 最快: {fastest['name']} ({fastest['avg']:.1f}ms)")
        
        # 计算加速比
        cpu_result = next((r for r in results if r['name'] == 'CPU_ONLY'), None)
        if cpu_result:
            print(f"\n📈 相对 CPU 的加速比:")
            for r in results:
                if r['name'] != 'CPU_ONLY':
                    speedup = cpu_result['avg'] / r['avg']
                    print(f"   {r['name']:<20} {speedup:.2f}x")
    
    print(f"\n{'='*60}")
    print("✅ 测试完成")
    print(f"{'='*60}")
    
    # 推荐
    print(f"\n💡 推荐配置:")
    if fastest['name'] == 'CPU_AND_NE':
        print(f"   compute_units: \"ANE\"  # Apple Neural Engine 最快")
    elif fastest['name'] == 'CPU_AND_GPU':
        print(f"   compute_units: \"GPU\"  # GPU (Metal) 最快")
    elif fastest['name'] == 'ALL':
        print(f"   compute_units: \"ALL\"  # 自动选择最优")
    else:
        print(f"   compute_units: \"CPU\"  # CPU 最快")


if __name__ == "__main__":
    main()
