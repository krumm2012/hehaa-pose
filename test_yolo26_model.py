#!/usr/bin/env python3
# test_yolo26_model.py
# 快速测试 YOLO26-pose 模型加载和推理

import sys
import os

def test_coremltools():
    """测试 coremltools 是否安装"""
    try:
        import coremltools as ct
        print(f"✅ coremltools 已安装，版本: {ct.__version__}")
        return True
    except ImportError:
        print("❌ coremltools 未安装")
        print("💡 请运行: pip install coremltools")
        return False

def test_model_loading():
    """测试模型加载"""
    try:
        import coremltools as ct
        model_path = "models/yolo26m-pose.mlpackage"
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        print(f"📦 加载模型: {model_path}")
        model = ct.models.MLModel(model_path)
        print("✅ 模型加载成功")
        
        # 显示模型信息
        spec = model.get_spec()
        print("\n📊 模型信息:")
        print(f"   输入: {[inp.name for inp in spec.description.input]}")
        print(f"   输出: {[out.name for out in spec.description.output]}")
        
        # 显示输入规格
        for inp in spec.description.input:
            print(f"\n   输入 '{inp.name}':")
            if hasattr(inp.type, 'imageType'):
                print(f"      类型: 图像")
                print(f"      尺寸: {inp.type.imageType.width}x{inp.type.imageType.height}")
            elif hasattr(inp.type, 'multiArrayType'):
                print(f"      类型: 多维数组")
                print(f"      形状: {list(inp.type.multiArrayType.shape)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def test_pose_estimator():
    """测试姿态估计器"""
    try:
        import numpy as np
        import cv2
        
        # 创建测试图像
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        print("\n🤖 测试姿态估计器...")
        from pose_estimator import create_pose_estimator
        
        config = {
            'pose_confidence_threshold': 0.5,
            'pose_keypoint_confidence': 0.3,
            'dominant_hand': 'right',
            'two_hand_wrist_distance_max_px': 50
        }
        
        estimator = create_pose_estimator('models/yolo26m-pose.mlpackage', config)
        print("✅ 姿态估计器创建成功")
        
        # 测试推理（使用空图像）
        print("🔄 测试推理...")
        keypoints = estimator.get_keypoints(test_image)
        print(f"✅ 推理成功，检测到 {len(keypoints)} 个人")
        
        return True
        
    except Exception as e:
        print(f"❌ 姿态估计器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🎯 YOLO26-pose 模型测试")
    print("=" * 60)
    
    # 测试1: coremltools
    print("\n[测试 1/3] 检查 coremltools...")
    if not test_coremltools():
        print("\n⚠️ 请先安装 coremltools:")
        print("   pip install coremltools")
        return 1
    
    # 测试2: 模型加载
    print("\n[测试 2/3] 测试模型加载...")
    if not test_model_loading():
        return 1
    
    # 测试3: 姿态估计器
    print("\n[测试 3/3] 测试姿态估计器...")
    if not test_pose_estimator():
        return 1
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    print("\n💡 下一步:")
    print("   python3 main.py --config configs/yolo26_tennis_config.yaml --input 'video.mp4'")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
