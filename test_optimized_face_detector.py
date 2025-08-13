#!/usr/bin/env python3
# test_optimized_face_detector.py
"""
测试优化后的人脸检测器
"""

import cv2
import numpy as np
import yaml
import logging
from advanced_face_detector import AdvancedFaceDetector

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """加载配置文件"""
    try:
        with open('configs/default_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"加载配置失败: {e}")
        return {}

def test_with_sample_video():
    """使用示例视频测试人脸检测器"""
    # 加载配置
    config = load_config()
    
    # 初始化检测器
    detector = AdvancedFaceDetector(config)
    
    # 检查检测器状态
    detection_info = detector.get_detection_info()
    print("\n🔍 检测器状态:")
    print(f"   增强OpenCV可用: {detection_info['enhanced_opencv_available']}")
    print(f"   MTCNN可用: {detection_info['mtcnn_available']}")
    print(f"   MTCNN已加载: {detection_info['mtcnn_loaded']}")
    print(f"   GLIP可用: {detection_info['glip_available']}")
    print(f"   GLIP已加载: {detection_info['glip_loaded']}")
    print(f"   检测方法: {detection_info['detection_methods']}")
    
    # 测试视频路径
    video_path = config.get('video_input_path', 'data/input_video.mp4')
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return
            
        frame_count = 0
        total_detections = 0
        detection_methods_used = {}
        
        print(f"\n🎬 开始处理视频: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 检测人脸
            faces = detector.detect_faces(frame)
            total_detections += len(faces)
            
            # 统计检测方法使用情况
            for face in faces:
                method = face.get('method', 'unknown')
                detection_methods_used[method] = detection_methods_used.get(method, 0) + 1
            
            # 在帧上绘制检测结果
            for face in faces:
                bbox = face['bbox']
                confidence = face.get('confidence', 0)
                method = face.get('method', 'unknown')
                
                x1, y1, x2, y2 = bbox
                
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加标签
                label = f"{method}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 显示当前帧信息
            if frame_count % 10 == 0:
                print(f"   处理帧 {frame_count}, 检测到 {len(faces)} 个人脸")
            
            # 限制处理前50帧
            if frame_count >= 50:
                break
        
        cap.release()
        
        # 输出统计信息
        print(f"\n📊 处理结果:")
        print(f"   总帧数: {frame_count}")
        print(f"   总检测数: {total_detections}")
        print(f"   平均每帧检测数: {total_detections/frame_count:.2f}")
        print(f"   检测方法使用: {detection_methods_used}")
        
        # 获取详细统计
        final_stats = detector.get_detection_info()
        print(f"\n🔧 检测器统计:")
        for method, count in final_stats['detection_stats'].items():
            if count > 0:
                print(f"   {method}: {count}")
        
        print("\n✅ 测试完成!")
        
    except Exception as e:
        logging.error(f"视频处理失败: {e}")

def test_with_sample_image():
    """使用示例图片测试人脸检测器"""
    # 加载配置
    config = load_config()
    
    # 初始化检测器
    detector = AdvancedFaceDetector(config)
    
    # 创建测试图像（模拟场景）
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\n🖼️ 测试图像检测:")
    
    # 检测人脸
    faces = detector.detect_faces(test_image)
    
    print(f"   检测结果: {len(faces)} 个人脸")
    
    for i, face in enumerate(faces):
        print(f"   人脸 {i+1}: bbox={face['bbox']}, confidence={face.get('confidence', 0):.2f}, method={face.get('method', 'unknown')}")

if __name__ == "__main__":
    print("🎾 优化后的人脸检测器测试")
    print("="*50)
    
    # 测试示例图像
    test_with_sample_image()
    
    # 测试示例视频（如果存在）
    test_with_sample_video() 