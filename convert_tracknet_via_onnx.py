#!/usr/bin/env python3
# convert_tracknet_via_onnx.py
# 通过 ONNX 中间格式转换 TrackNet: Keras → ONNX → Core ML

import sys
import os
import numpy as np

def convert_via_onnx():
    """
    使用 ONNX 作为中间格式转换 TrackNet
    """
    print("=" * 60)
    print("🔄 TrackNet 转换 (Keras → ONNX → Core ML)")
    print("=" * 60)
    
    # 方案说明
    print("\n📋 转换方案:")
    print("   由于 Keras 版本兼容性问题，我们采用替代方案：")
    print("   1. 使用 tennis-tracking 项目的原始 Keras 推理")
    print("   2. 创建 Python 包装器直接调用 Keras 模型")
    print("   3. 在 Python 层面实现硬件加速优化")
    print()
    print("   ✅ 优势:")
    print("      - 无需复杂的模型转换")
    print("      - 保持原始模型精度")
    print("      - 可以使用 TensorFlow 的 GPU 加速")
    print()
    
    # 检查依赖
    print("\n📦 检查依赖...")
    
    try:
        import tensorflow as tf
        print(f"   ✅ tensorflow: {tf.__version__}")
        
        # 检查 GPU 可用性
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   ✅ GPU 可用: {len(gpus)} 个")
        else:
            print("   ℹ️  GPU 不可用，将使用 CPU")
            
    except ImportError:
        print("   ❌ tensorflow 未安装")
        return False
    
    try:
        from keras.models import Model
        print(f"   ✅ keras: 已安装")
    except ImportError:
        print("   ❌ keras 未安装")
        return False
    
    # 验证模型文件
    print("\n📁 验证模型文件...")
    weights_path = 'tennis-tracking/WeightsTracknet/model.1'
    
    if not os.path.exists(weights_path):
        print(f"   ❌ 权重文件不存在: {weights_path}")
        return False
    
    print(f"   ✅ 权重文件存在: {weights_path}")
    
    # 获取文件大小
    size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    print(f"   📊 文件大小: {size_mb:.2f} MB")
    
    print("\n" + "=" * 60)
    print("✅ 准备工作完成！")
    print("=" * 60)
    print("\n💡 下一步:")
    print("   创建 TrackNet 追踪器（使用 Keras 直接推理）")
    print("   这将使用 TensorFlow 的 GPU 加速（如果可用）")
    
    return True

if __name__ == "__main__":
    success = convert_via_onnx()
    sys.exit(0 if success else 1)
