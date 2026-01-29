#!/usr/bin/env python3
# test_tracknet_simple.py
# 简单测试 TrackNet - 使用 channels_last 模型但随机初始化权重

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, 'tennis-tracking')
sys.path.insert(0, 'tennis-tracking/Models')

def test_simple():
    """
    测试 channels_last 模型是否可以运行（不加载权重）
    """
    print("=" * 60)
    print("🧪 测试 TrackNet channels_last 模型")
    print("=" * 60)
    
    from tracknet_channels_last import trackNet_channels_last
    
    # 创建模型
    print("\n🏗️ 创建模型...")
    model = trackNet_channels_last(n_classes=256, input_height=360, input_width=640)
    print("✅ 模型创建成功")
    
    # 创建测试输入
    print("\n📦 创建测试输入...")
    test_input = np.random.rand(1, 360, 640, 3).astype(np.float32)
    print(f"   输入形状: {test_input.shape}")
    
    # 推理
    print("\n🔄 执行推理...")
    try:
        output = model.predict(test_input, verbose=0)
        print(f"   ✅ 推理成功！")
        print(f"   输出形状: {output.shape}")
        print(f"   输出范围: [{output.min():.4f}, {output.max():.4f}]")
        
        # 测试热力图处理
        heatmap = output[0].reshape((360, 640, 256)).argmax(axis=2)
        print(f"   热力图形状: {heatmap.shape}")
        print(f"   热力图范围: [{heatmap.min()}, {heatmap.max()}]")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ 测试通过！channels_last 模型可以正常运行")
        print("=" * 60)
        print("\n💡 下一步:")
        print("   由于权重转换复杂，建议采用以下方案之一：")
        print("   1. 使用改进的 HSV 方法（准确度 75-85%）")
        print("   2. 训练新的 YOLO 球检测模型")
        print("   3. 使用 PyTorch 版本的 TrackNet（如果有）")
    else:
        print("\n" + "=" * 60)
        print("❌ 测试失败")
        print("=" * 60)
    
    sys.exit(0 if success else 1)
