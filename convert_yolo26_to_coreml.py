#!/usr/bin/env python3
"""
convert_yolo26_to_coreml.py
将 YOLO26 PyTorch 模型转换为 Core ML 格式
"""

from ultralytics import YOLO
import os


def convert_yolo26_to_coreml(model_path, output_name=None):
    """
    转换 YOLO26 模型到 Core ML
    
    Args:
        model_path: PyTorch 模型路径 (.pt)
        output_name: 输出文件名（可选）
    """
    print(f"\n🔄 转换 YOLO26 模型到 Core ML")
    print("="*60)
    print(f"📥 输入模型: {model_path}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 加载模型
    print(f"\n⏳ 加载模型...")
    try:
        model = YOLO(model_path)
        print(f"✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 导出为 Core ML
    print(f"\n⏳ 导出为 Core ML 格式...")
    print(f"   这可能需要几分钟时间...")
    
    try:
        # 使用 Ultralytics 的 export 方法
        # format='coreml' 会自动转换为 .mlpackage 格式
        success = model.export(
            format='coreml',
            imgsz=640,  # 输入尺寸
            nms=True,   # 包含 NMS
            half=False  # 使用 FP32（更好的兼容性）
        )
        
        print(f"\n✅ 转换成功！")
        print(f"📤 输出文件: {success}")
        
        # 重命名（如果指定了输出名称）
        if output_name and success != output_name:
            import shutil
            if os.path.exists(output_name):
                shutil.rmtree(output_name)
            shutil.move(success, output_name)
            print(f"📝 已重命名为: {output_name}")
        
        return success
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='转换 YOLO26 到 Core ML')
    parser.add_argument('--model', type=str, default='yolo26m.pt',
                       help='PyTorch 模型路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件名（可选）')
    
    args = parser.parse_args()
    
    # 转换
    result = convert_yolo26_to_coreml(args.model, args.output)
    
    if result:
        print(f"\n🎉 转换完成！")
        print(f"\n📋 使用方法:")
        print(f"   from yolo26_detector import YOLO26Detector")
        print(f"   detector = YOLO26Detector('{result}', config)")
        print(f"\n" + "="*60)
    else:
        print(f"\n❌ 转换失败")


if __name__ == "__main__":
    main()
