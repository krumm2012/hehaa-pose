#!/usr/bin/env python3
# convert_weights_format.py
# 将 channels_first 权重转换为 channels_last 格式

import sys
import os
import numpy as np

sys.path.insert(0, 'tennis-tracking')
sys.path.insert(0, 'tennis-tracking/Models')

def convert_weights():
    """
    将 TrackNet 权重从 channels_first 转换为 channels_last
    """
    print("=" * 60)
    print("🔄 转换 TrackNet 权重格式")
    print("=" * 60)
    
    from tracknet import trackNet
    from tracknet_channels_last import trackNet_channels_last
    
    # 1. 加载原始模型（channels_first）
    print("\n📦 加载原始模型（channels_first）...")
    model_cf = trackNet(n_classes=256, input_height=360, input_width=640)
    model_cf.load_weights('tennis-tracking/WeightsTracknet/model.1')
    print("✅ 原始模型加载成功")
    
    # 2. 创建新模型（channels_last）
    print("\n🏗️ 创建新模型（channels_last）...")
    model_cl = trackNet_channels_last(n_classes=256, input_height=360, input_width=640)
    print("✅ 新模型创建成功")
    
    # 3. 转换权重
    print("\n🔄 转换权重...")
    print(f"   原始模型层数: {len(model_cf.layers)}")
    print(f"   新模型层数: {len(model_cl.layers)}")
    
    # 创建层名称映射
    conv_layers_cf = [l for l in model_cf.layers if 'conv2d' in l.name]
    conv_layers_cl = [l for l in model_cl.layers if 'conv2d' in l.name]
    bn_layers_cf = [l for l in model_cf.layers if 'batch_normalization' in l.name]
    bn_layers_cl = [l for l in model_cl.layers if 'batch_normalization' in l.name]
    
    print(f"   Conv2D 层数: {len(conv_layers_cf)} -> {len(conv_layers_cl)}")
    print(f"   BatchNorm 层数: {len(bn_layers_cf)} -> {len(bn_layers_cl)}")
    
    # 转换 Conv2D 层
    print("\n   转换 Conv2D 层...")
    for i, (layer_cf, layer_cl) in enumerate(zip(conv_layers_cf, conv_layers_cl)):
        weights = layer_cf.get_weights()
        if len(weights) == 0:
            continue
        
        kernel = weights[0]
        bias = weights[1] if len(weights) > 1 else None
        
        # Keras 在加载 HDF5 权重时已经自动转换为 (h, w, in_ch, out_ch) 格式
        # 所以我们可以直接复制权重
        layer_cl.set_weights(weights)
        print(f"      ✓ {layer_cf.name}: {kernel.shape} -> {layer_cl.name}")
    
    # 转换 BatchNormalization 层
    print("\n   转换 BatchNormalization 层...")
    for i, (layer_cf, layer_cl) in enumerate(zip(bn_layers_cf, bn_layers_cl)):
        weights = layer_cf.get_weights()
        if len(weights) > 0:
            layer_cl.set_weights(weights)
            print(f"      ✓ {layer_cf.name} -> {layer_cl.name}")
    
    print(f"\n✅ 转换完成！")
    
    # 4. 保存新模型
    output_path = 'models/tracknet_channels_last.h5'
    print(f"\n💾 保存新模型: {output_path}")
    os.makedirs('models', exist_ok=True)
    model_cl.save_weights(output_path)
    print("✅ 保存成功")
    
    return model_cl

if __name__ == "__main__":
    model = convert_weights()
    
    print("\n" + "=" * 60)
    print("✅ 权重转换完成！")
    print("=" * 60)
    print("\n💡 下一步:")
    print("   使用新模型进行测试")
