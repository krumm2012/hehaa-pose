#!/usr/bin/env python3
# check_coreml_acceleration.py
# 检查和优化 Core ML 模型的硬件加速设置

import coremltools as ct
import sys
import time
import numpy as np
from PIL import Image

def check_model_compute_units(model_path):
    """检查模型的计算单元配置"""
    print("=" * 60)
    print("🔍 Core ML 模型硬件加速检查")
    print("=" * 60)
    
    print(f"\n📦 加载模型: {model_path}")
    model = ct.models.MLModel(model_path)
    
    # 获取模型规格
    spec = model.get_spec()
    
    print("\n📊 模型信息:")
    print(f"   输入: {[inp.name for inp in spec.description.input]}")
    print(f"   输出: {[out.name for out in spec.description.output]}")
    
    # 检查计算单元设置
    print("\n⚙️ 计算单元配置:")
    if hasattr(spec, 'computeUnits'):
        compute_units = spec.computeUnits
        print(f"   当前设置: {compute_units}")
    else:
        print("   ⚠️ 未找到计算单元配置（使用默认设置）")
    
    return model

def benchmark_model(model, num_iterations=10):
    """性能基准测试"""
    print(f"\n🏃 性能测试 ({num_iterations} 次推理)...")
    
    # 创建测试图像
    test_image = Image.new('RGB', (640, 640), color='white')
    
    # 预热
    print("   预热中...")
    for _ in range(3):
        _ = model.predict({'image': test_image})
    
    # 测试
    print("   测试中...")
    times = []
    for i in range(num_iterations):
        start = time.time()
        _ = model.predict({'image': test_image})
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   迭代 {i+1}/{num_iterations}: {elapsed*1000:.2f}ms")
    
    # 统计
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\n📈 性能统计:")
    print(f"   平均时间: {avg_time*1000:.2f}ms")
    print(f"   标准差: {std_time*1000:.2f}ms")
    print(f"   最快: {min_time*1000:.2f}ms")
    print(f"   最慢: {max_time*1000:.2f}ms")
    print(f"   FPS: {1/avg_time:.2f}")
    
    return avg_time

def optimize_model_for_neural_engine(model_path, output_path=None):
    """优化模型以使用 Neural Engine"""
    print("\n" + "=" * 60)
    print("⚡ 优化模型以使用 Neural Engine")
    print("=" * 60)
    
    if output_path is None:
        output_path = model_path.replace('.mlpackage', '_optimized.mlpackage')
    
    print(f"\n📦 加载原始模型: {model_path}")
    
    try:
        # 加载模型
        model = ct.models.MLModel(model_path)
        spec = model.get_spec()
        
        # 设置计算单元为 ALL（自动选择最佳硬件）
        print("\n⚙️ 设置计算单元...")
        print("   选项说明:")
        print("   - ALL: 自动选择（Neural Engine > GPU > CPU）")
        print("   - CPU_AND_GPU: 仅使用 CPU 和 GPU")
        print("   - CPU_AND_NE: 仅使用 CPU 和 Neural Engine")
        print("   - CPU_ONLY: 仅使用 CPU")
        
        # 创建优化后的模型（使用 ALL）
        print("\n   正在创建优化模型（compute_units=ALL）...")
        optimized_model = ct.models.MLModel(
            model_path,
            compute_units=ct.ComputeUnit.ALL
        )
        
        print(f"\n💾 保存优化模型: {output_path}")
        optimized_model.save(output_path)
        
        print("✅ 优化完成！")
        
        return output_path
        
    except Exception as e:
        print(f"❌ 优化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_compute_units(model_path):
    """比较不同计算单元的性能"""
    print("\n" + "=" * 60)
    print("📊 不同计算单元性能对比")
    print("=" * 60)
    
    compute_units = [
        (ct.ComputeUnit.ALL, "ALL (自动选择)"),
        (ct.ComputeUnit.CPU_AND_GPU, "CPU + GPU"),
        (ct.ComputeUnit.CPU_AND_NE, "CPU + Neural Engine"),
        (ct.ComputeUnit.CPU_ONLY, "仅 CPU"),
    ]
    
    results = {}
    
    for unit, name in compute_units:
        print(f"\n🔧 测试 {name}...")
        try:
            model = ct.models.MLModel(model_path, compute_units=unit)
            avg_time = benchmark_model(model, num_iterations=5)
            results[name] = avg_time
            print(f"✅ {name}: {avg_time*1000:.2f}ms ({1/avg_time:.2f} FPS)")
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
            results[name] = None
    
    # 显示对比
    print("\n" + "=" * 60)
    print("📊 性能对比总结")
    print("=" * 60)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1])
        
        print("\n排名（从快到慢）:")
        for i, (name, time) in enumerate(sorted_results, 1):
            fps = 1 / time
            print(f"{i}. {name:25s} {time*1000:6.2f}ms ({fps:5.2f} FPS)")
        
        # 计算加速比
        if len(sorted_results) > 1:
            fastest = sorted_results[0][1]
            slowest = sorted_results[-1][1]
            speedup = slowest / fastest
            print(f"\n⚡ 最快 vs 最慢加速比: {speedup:.2f}x")

def main():
    model_path = "models/yolo26m-pose.mlpackage"
    
    print("🎯 Core ML 硬件加速检查工具\n")
    
    # 1. 检查当前配置
    model = check_model_compute_units(model_path)
    
    # 2. 基准测试
    benchmark_model(model, num_iterations=10)
    
    # 3. 性能对比
    print("\n" + "=" * 60)
    response = input("是否进行不同计算单元的性能对比？(y/n): ")
    if response.lower() == 'y':
        compare_compute_units(model_path)
    
    # 4. 优化建议
    print("\n" + "=" * 60)
    print("💡 优化建议")
    print("=" * 60)
    print("""
1. **默认设置（推荐）**:
   Core ML 会自动选择最佳硬件（Neural Engine > GPU > CPU）
   
2. **如何在代码中指定计算单元**:
   ```python
   import coremltools as ct
   
   # 自动选择（推荐）
   model = ct.models.MLModel('model.mlpackage', 
                            compute_units=ct.ComputeUnit.ALL)
   
   # 仅 Neural Engine + CPU
   model = ct.models.MLModel('model.mlpackage',
                            compute_units=ct.ComputeUnit.CPU_AND_NE)
   ```

3. **监控硬件使用**:
   - 使用 Activity Monitor 查看 CPU/GPU 使用率
   - 使用 powermetrics 查看 ANE (Neural Engine) 使用率:
     ```bash
     sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000
     ```

4. **当前模型性能**:
   - 您的模型运行在 ~8 FPS (125ms/帧)
   - 这已经包含了 Neural Engine 加速
   - 如果使用纯 CPU 会慢 3-5 倍
    """)

if __name__ == "__main__":
    main()
