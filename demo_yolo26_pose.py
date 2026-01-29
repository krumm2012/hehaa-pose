#!/usr/bin/env python3
# demo_yolo26_pose.py
# 使用真实图像演示 YOLO26-pose 姿态检测

import cv2
import numpy as np
import sys
import os

def create_test_image():
    """创建一个包含人形的测试图像"""
    # 创建白色背景
    img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    
    # 绘制一个简单的人形（用于测试）
    # 头部
    cv2.circle(img, (320, 150), 30, (0, 0, 0), -1)
    
    # 躯干
    cv2.rectangle(img, (290, 180), (350, 350), (0, 0, 0), -1)
    
    # 左臂
    cv2.rectangle(img, (250, 200), (290, 220), (0, 0, 0), -1)
    cv2.rectangle(img, (210, 220), (250, 240), (0, 0, 0), -1)
    
    # 右臂
    cv2.rectangle(img, (350, 200), (390, 220), (0, 0, 0), -1)
    cv2.rectangle(img, (390, 220), (430, 240), (0, 0, 0), -1)
    
    # 左腿
    cv2.rectangle(img, (290, 350), (310, 450), (0, 0, 0), -1)
    cv2.rectangle(img, (290, 450), (310, 550), (0, 0, 0), -1)
    
    # 右腿
    cv2.rectangle(img, (330, 350), (350, 450), (0, 0, 0), -1)
    cv2.rectangle(img, (330, 450), (350, 550), (0, 0, 0), -1)
    
    return img

def main():
    print("=" * 60)
    print("🎯 YOLO26-pose 姿态检测演示")
    print("=" * 60)
    
    # 检查是否有输入图像
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            return 1
        
        print(f"\n📸 加载图像: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ 无法读取图像: {image_path}")
            return 1
    else:
        print("\n📸 使用测试图像（简单人形）")
        print("💡 提示: 可以传入真实图像路径作为参数")
        print("   例如: python3 demo_yolo26_pose.py path/to/image.jpg")
        frame = create_test_image()
    
    print(f"   图像尺寸: {frame.shape[1]}x{frame.shape[0]}")
    
    # 创建姿态估计器
    print("\n🤖 初始化 YOLO26-pose 姿态估计器...")
    from pose_estimator import create_pose_estimator
    
    config = {
        'pose_confidence_threshold': 0.3,  # 降低阈值以便更容易检测
        'pose_keypoint_confidence': 0.2,
        'dominant_hand': 'right',
        'two_hand_wrist_distance_max_px': 50
    }
    
    try:
        estimator = create_pose_estimator('models/yolo26m-pose.mlpackage', config)
        print("✅ 姿态估计器初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return 1
    
    # 进行姿态检测
    print("\n🔄 进行姿态检测...")
    keypoints = estimator.get_keypoints(frame)
    
    print(f"✅ 检测完成，检测到 {len(keypoints)} 个人")
    
    # 显示检测结果
    if keypoints:
        for i, person_kpts in enumerate(keypoints):
            print(f"\n👤 人物 {i+1}:")
            detected_kpts = [name for name, pt in person_kpts.items() if pt is not None]
            print(f"   检测到的关键点: {len(detected_kpts)}/17")
            print(f"   关键点: {', '.join(detected_kpts[:5])}{'...' if len(detected_kpts) > 5 else ''}")
            
            # 分类挥拍类型
            swing_type = estimator.classify_swing([person_kpts])
            print(f"   挥拍类型: {swing_type}")
    else:
        print("\n⚠️ 未检测到人物")
        print("💡 提示:")
        print("   1. 尝试使用包含人物的真实图像")
        print("   2. 降低 pose_confidence_threshold")
        print("   3. 确保图像质量良好")
    
    # 绘制关键点
    print("\n🎨 绘制检测结果...")
    result_frame = estimator.draw_keypoints(frame.copy(), keypoints)
    
    # 保存结果
    output_path = "yolo26_pose_demo_output.jpg"
    cv2.imwrite(output_path, result_frame)
    print(f"✅ 结果已保存: {output_path}")
    
    # 如果是测试图像，也保存原图
    if len(sys.argv) <= 1:
        cv2.imwrite("yolo26_pose_demo_input.jpg", frame)
        print(f"✅ 输入图像已保存: yolo26_pose_demo_input.jpg")
    
    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("=" * 60)
    print(f"\n💡 查看结果: open {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
