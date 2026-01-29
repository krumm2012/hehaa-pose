#!/usr/bin/env python3
# convert_tracknet_to_coreml.py
# 将 TrackNet Keras 模型转换为 Core ML 格式

import sys
import os
import numpy as np

def convert_tracknet_to_coreml():
    """
    将 TrackNet Keras 模型转换为 Core ML
    """
    print("=" * 60)
    print("🔄 TrackNet → Core ML 转换")
    print("=" * 60)
    
    # 1. 检查依赖
    print("\n📦 检查依赖...")
    try:
        import coremltools as ct
        print(f"   ✅ coremltools: {ct.__version__}")
    except ImportError:
        print("   ❌ coremltools 未安装")
        print("   💡 运行: pip install coremltools")
        return None
    
    try:
        import tensorflow as tf
        print(f"   ✅ tensorflow: {tf.__version__}")
    except ImportError:
        print("   ❌ tensorflow 未安装")
        print("   💡 运行: pip install tensorflow")
        return None
    
    try:
        from keras.models import Model
        print(f"   ✅ keras: 已安装")
    except ImportError:
        print("   ❌ keras 未安装")
        print("   💡 运行: pip install keras")
        return None
    
    # 2. 加载 TrackNet 模型架构
    print("\n🏗️ 加载 TrackNet 模型架构...")
    
    # 添加 tennis-tracking 到路径
    sys.path.insert(0, 'tennis-tracking')
    sys.path.insert(0, 'tennis-tracking/Models')
    
    try:
        from tracknet import trackNet
        
        # 创建模型
        print("   创建模型架构...")
        model = trackNet(n_classes=256, input_height=360, input_width=640)
        print("   ✅ 模型架构创建成功")
        
    except Exception as e:
        print(f"   ❌ 加载模型架构失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 3. 加载权重
    print("\n⚖️ 加载模型权重...")
    weights_path = 'tennis-tracking/WeightsTracknet/model.1'
    
    if not os.path.exists(weights_path):
        print(f"   ❌ 权重文件不存在: {weights_path}")
        return None
    
    try:
        # Keras 3 不支持旧格式，需要使用 legacy 模式
        import tensorflow.keras.backend as K
        
        # 尝试使用 legacy 加载
        print("   尝试使用 legacy 格式加载...")
        
        # 方法1: 使用 h5py 直接加载
        try:
            import h5py
            with h5py.File(weights_path, 'r') as f:
                if 'layer_names' in f.attrs:
                    print("   检测到 HDF5 格式权重")
                    # 使用 Keras 的 load_weights_from_hdf5_group
                    from tensorflow.python.keras.saving import hdf5_format
                    hdf5_format.load_weights_from_hdf5_group(f, model.layers)
                    print("   ✅ 权重加载成功（HDF5 格式）")
                else:
                    raise ValueError("不是有效的 Keras HDF5 权重文件")
        except Exception as e1:
            print(f"   HDF5 加载失败: {e1}")
            
            # 方法2: 降级到 Keras 2.x
            print("   尝试安装 Keras 2.x...")
            import subprocess
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'keras==2.15.0', '--quiet'], check=True)
            
            # 重新导入
            import importlib
            import keras
            importlib.reload(keras)
            
            # 重新创建模型
            from tracknet import trackNet
            model = trackNet(n_classes=256, input_height=360, input_width=640)
            model.load_weights(weights_path)
            print("   ✅ 权重加载成功（Keras 2.x）")
            
    except Exception as e:
        print(f"   ❌ 权重加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 4. 转换为 Core ML
    print("\n🔄 转换为 Core ML...")
    print("   这可能需要几分钟时间...")
    
    try:
        # 定义输入形状
        # TrackNet 使用 channels_first: (batch, channels, height, width)
        input_shape = (1, 3, 360, 640)
        
        # 转换
        coreml_model = ct.convert(
            model,
            inputs=[ct.TensorType(
                name="image",
                shape=input_shape,
                dtype=np.float32
            )],
            outputs=[ct.TensorType(name="heatmap")],
            compute_units=ct.ComputeUnit.ALL,  # 使用所有硬件（ANE/GPU/CPU）
            minimum_deployment_target=ct.target.macOS13
        )
        
        print("   ✅ 转换成功")
        
    except Exception as e:
        print(f"   ❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 5. 添加元数据
    print("\n📝 添加模型元数据...")
    coreml_model.author = "TrackNet (Converted to Core ML)"
    coreml_model.license = "Research Use"
    coreml_model.short_description = "TrackNet - Tennis Ball Tracking Network"
    coreml_model.version = "1.0"
    
    # 添加输入/输出描述
    spec = coreml_model.get_spec()
    spec.description.input[0].shortDescription = "Input image (3, 360, 640)"
    spec.description.output[0].shortDescription = "Ball position heatmap (230400, 256)"
    
    # 6. 保存模型
    output_path = "models/tracknet.mlpackage"
    print(f"\n💾 保存 Core ML 模型: {output_path}")
    
    # 确保目录存在
    os.makedirs("models", exist_ok=True)
    
    try:
        coreml_model.save(output_path)
        print("   ✅ 保存成功")
    except Exception as e:
        print(f"   ❌ 保存失败: {e}")
        return None
    
    # 7. 验证模型
    print("\n🔍 验证模型...")
    try:
        # 重新加载验证
        loaded_model = ct.models.MLModel(output_path)
        spec = loaded_model.get_spec()
        
        print(f"   输入: {[inp.name for inp in spec.description.input]}")
        print(f"   输出: {[out.name for out in spec.description.output]}")
        
        # 获取文件大小
        import subprocess
        result = subprocess.run(['du', '-sh', output_path], capture_output=True, text=True)
        size = result.stdout.split()[0]
        print(f"   模型大小: {size}")
        
        print("   ✅ 模型验证成功")
        
    except Exception as e:
        print(f"   ❌ 验证失败: {e}")
        return None
    
    return coreml_model

def main():
    """主函数"""
    model = convert_tracknet_to_coreml()
    
    if model:
        print("\n" + "=" * 60)
        print("✅ TrackNet 转换完成！")
        print("=" * 60)
        print("\n📍 输出位置: models/tracknet.mlpackage")
        print("\n💡 下一步:")
        print("   1. 创建 TrackNet 追踪器")
        print("   2. 集成到现有系统")
        print("   3. 运行性能测试")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ 转换失败")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
