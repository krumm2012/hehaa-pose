#!/usr/bin/env python3
"""
inspect_yolo26_model.py
检查 YOLO26 Core ML 模型的输入输出格式
"""

import coremltools as ct
import numpy as np


def inspect_model(model_path):
    """检查模型规格"""
    print(f"\n🔍 检查模型: {model_path}")
    print("="*60)
    
    # 加载模型
    try:
        model = ct.models.MLModel(model_path)
        print("✅ 模型加载成功\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 获取规格
    spec = model.get_spec()
    
    # 输入信息
    print("📥 输入信息:")
    print("-"*60)
    for i, input_desc in enumerate(spec.description.input):
        print(f"\n输入 {i+1}: {input_desc.name}")
        
        if input_desc.type.HasField('imageType'):
            img_type = input_desc.type.imageType
            print(f"  类型: Image")
            print(f"  宽度: {img_type.width}")
            print(f"  高度: {img_type.height}")
            print(f"  颜色空间: {img_type.colorSpace}")
        
        elif input_desc.type.HasField('multiArrayType'):
            array_type = input_desc.type.multiArrayType
            print(f"  类型: MultiArray")
            print(f"  形状: {list(array_type.shape)}")
            print(f"  数据类型: {array_type.dataType}")
    
    # 输出信息
    print("\n\n📤 输出信息:")
    print("-"*60)
    for i, output_desc in enumerate(spec.description.output):
        print(f"\n输出 {i+1}: {output_desc.name}")
        
        if output_desc.type.HasField('multiArrayType'):
            array_type = output_desc.type.multiArrayType
            print(f"  类型: MultiArray")
            print(f"  形状: {list(array_type.shape)}")
            print(f"  数据类型: {array_type.dataType}")
        
        elif output_desc.type.HasField('dictionaryType'):
            print(f"  类型: Dictionary")
    
    # 元数据
    print("\n\n📋 模型元数据:")
    print("-"*60)
    metadata = spec.description.metadata
    if metadata.author:
        print(f"  作者: {metadata.author}")
    if metadata.shortDescription:
        print(f"  描述: {metadata.shortDescription}")
    if metadata.versionString:
        print(f"  版本: {metadata.versionString}")
    
    # 尝试测试推理
    print("\n\n🧪 测试推理:")
    print("-"*60)
    try:
        # 创建测试输入
        input_name = spec.description.input[0].name
        input_desc = spec.description.input[0]
        
        if input_desc.type.HasField('imageType'):
            # 图像输入
            from PIL import Image
            width = input_desc.type.imageType.width
            height = input_desc.type.imageType.height
            
            # 创建随机图像
            test_image = Image.fromarray(
                np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            )
            
            print(f"  创建测试图像: {width}x{height}")
            
            # 推理
            predictions = model.predict({input_name: test_image})
            
            print(f"  ✅ 推理成功")
            print(f"\n  输出键: {list(predictions.keys())}")
            
            for key, value in predictions.items():
                if isinstance(value, np.ndarray):
                    print(f"    {key}:")
                    print(f"      形状: {value.shape}")
                    print(f"      数据类型: {value.dtype}")
                    print(f"      范围: [{value.min():.4f}, {value.max():.4f}]")
                elif isinstance(value, dict):
                    print(f"    {key}: Dictionary (长度={len(value)})")
                else:
                    print(f"    {key}: {type(value)}")
        
        else:
            print("  ⚠️ 非图像输入，跳过测试推理")
    
    except Exception as e:
        print(f"  ❌ 测试推理失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "yolo26n.mlpackage"
    
    inspect_model(model_path)
