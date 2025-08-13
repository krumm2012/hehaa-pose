#!/usr/bin/env python3
# test_debug_face_detector.py
"""
调试版人脸检测器测试 - 显示详细的检测和过滤信息
"""

import cv2
import numpy as np
import yaml
import logging
from advanced_face_detector import AdvancedFaceDetector

# 配置日志显示调试信息
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    """加载配置文件"""
    try:
        with open('configs/default_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"加载配置失败: {e}")
        return {}

def debug_enhanced_opencv_detector(detector, frame):
    """调试增强OpenCV检测器的原始检测结果"""
    print("\n🔍 调试增强OpenCV检测器:")
    
    if detector.enhanced_opencv_detector is None:
        print("   ❌ 增强OpenCV检测器未初始化")
        return
    
    # 获取原始检测结果（绕过过滤）
    opencv_detector = detector.enhanced_opencv_detector
    
    # 预处理图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.equalizeHist(gray)
    
    all_raw_detections = []
    
    # 测试每个检测器
    for detector_name, cascade in opencv_detector.detectors.items():
        try:
            # 使用宽松的参数进行检测
            faces = cascade.detectMultiScale(
                enhanced_gray,
                scaleFactor=1.05,
                minNeighbors=3,  # 降低邻居数
                minSize=(20, 20),  # 更小的最小尺寸
                maxSize=(500, 500),  # 更大的最大尺寸
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            print(f"   {detector_name}: 检测到 {len(faces)} 个原始候选")
            
            for i, (x, y, w, h) in enumerate(faces):
                detection = {
                    'bbox': [x, y, x + w, y + h],
                    'width': w,
                    'height': h,
                    'aspect_ratio': w / h,
                    'area': w * h,
                    'method': detector_name
                }
                all_raw_detections.append(detection)
                print(f"     候选 {i+1}: 位置=({x},{y}), 尺寸={w}x{h}, 宽高比={w/h:.2f}, 面积={w*h}")
                
        except Exception as e:
            print(f"   检测器 {detector_name} 失败: {e}")
    
    print(f"\n   总原始检测数: {len(all_raw_detections)}")
    return all_raw_detections

def debug_filtering_process(detector, faces, frame_shape):
    """调试过滤过程"""
    print(f"\n🛡️ 调试过滤过程 (输入: {len(faces)} 个检测):")
    
    if not faces:
        print("   没有检测结果需要过滤")
        return []
    
    frame_h, frame_w = frame_shape[:2]
    
    # 获取过滤参数
    size_constraints = detector.size_constraints
    quality_filters = detector.quality_filters
    position_constraints = detector.position_constraints
    
    print(f"   帧尺寸: {frame_w}x{frame_h}")
    print(f"   尺寸约束: {size_constraints.get('min_face_size', 30)}-{size_constraints.get('max_face_size', 300)}px")
    print(f"   宽高比约束: {size_constraints.get('min_aspect_ratio', 0.5)}-{size_constraints.get('max_aspect_ratio', 2.0)}")
    print(f"   最小置信度: {quality_filters.get('min_confidence', 0.6)}")
    
    passed_faces = []
    filter_reasons = []
    
    for i, face in enumerate(faces):
        bbox = face['bbox']
        x1, y1, x2, y2 = bbox
        face_width = x2 - x1
        face_height = y2 - y1
        aspect_ratio = face_width / face_height
        confidence = face.get('confidence', 0.8)  # 假设置信度
        
        reasons = []
        
        # 检查各种过滤条件
        min_size = size_constraints.get('min_face_size', 30)
        max_size = size_constraints.get('max_face_size', 300)
        
        if face_width < min_size or face_height < min_size:
            reasons.append(f"尺寸太小({face_width}x{face_height} < {min_size})")
        
        if face_width > max_size or face_height > max_size:
            reasons.append(f"尺寸太大({face_width}x{face_height} > {max_size})")
        
        min_ratio = size_constraints.get('min_aspect_ratio', 0.5)
        max_ratio = size_constraints.get('max_aspect_ratio', 2.0)
        
        if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
            reasons.append(f"宽高比异常({aspect_ratio:.2f} 不在 {min_ratio}-{max_ratio})")
        
        min_confidence = quality_filters.get('min_confidence', 0.6)
        if confidence < min_confidence:
            reasons.append(f"置信度太低({confidence:.2f} < {min_confidence})")
        
        # 边界检查
        if x1 < 0 or y1 < 0 or x2 >= frame_w or y2 >= frame_h:
            reasons.append("超出边界")
        
        if reasons:
            filter_reasons.append(f"   检测 {i+1}: {reasons}")
        else:
            passed_faces.append(face)
            print(f"   ✅ 检测 {i+1}: 尺寸={face_width}x{face_height}, 宽高比={aspect_ratio:.2f}, 置信度={confidence:.2f} - 通过")
    
    print(f"\n   过滤结果: {len(passed_faces)} / {len(faces)} 通过")
    
    if filter_reasons:
        print("   被过滤的检测:")
        for reason in filter_reasons:
            print(f"   ❌ {reason}")
    
    return passed_faces

def test_debug_video():
    """调试模式测试视频"""
    config = load_config()
    detector = AdvancedFaceDetector(config)
    
    # 显示配置信息
    print("\n⚙️ 当前配置:")
    print(f"   最小人脸尺寸: {detector.size_constraints.get('min_face_size', 30)}px")
    print(f"   最大人脸尺寸: {detector.size_constraints.get('max_face_size', 300)}px")
    print(f"   最小置信度: {detector.quality_filters.get('min_confidence', 0.6)}")
    print(f"   宽高比范围: {detector.size_constraints.get('min_aspect_ratio', 0.5)}-{detector.size_constraints.get('max_aspect_ratio', 2.0)}")
    
    video_path = config.get('video_input_path', 'data/input_video.mp4')
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return
        
        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n🎬 调试视频: {video_path}")
        print(f"   总帧数: {total_frames}")
        print(f"   帧率: {fps:.2f} FPS")
        
        # 计算最后10帧的起始位置
        start_frame = max(0, total_frames - 10)
        
        print(f"\n🔍 开始调试最后10帧 (帧 {start_frame + 1} 到 {total_frames})")
        
        # 跳转到最后10帧的开始位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = 0
        current_frame_num = start_frame
        
        while frame_count < 10:
            ret, frame = cap.read()
            if not ret:
                print("到达视频末尾")
                break
            
            current_frame_num += 1
            frame_count += 1
            
            print(f"\n" + "="*60)
            print(f"🖼️ 调试帧 {current_frame_num} (倒数第 {total_frames - current_frame_num + 1} 帧)")
            print(f"   帧尺寸: {frame.shape[1]}x{frame.shape[0]}")
            print(f"   时间戳: {current_frame_num / fps:.2f}s")
            
            # 调试原始检测
            raw_detections = debug_enhanced_opencv_detector(detector, frame)
            
            # 如果有原始检测，调试过滤过程
            if raw_detections:
                filtered_detections = debug_filtering_process(detector, raw_detections, frame.shape)
                
                # 测试完整的检测流程
                print(f"\n🔄 完整检测流程结果:")
                official_faces = detector.detect_faces(frame)
                print(f"   最终检测数: {len(official_faces)}")
                
                for i, face in enumerate(official_faces):
                    bbox = face['bbox']
                    confidence = face.get('confidence', 0)
                    method = face.get('method', 'unknown')
                    print(f"   人脸 {i+1}: bbox={bbox}, confidence={confidence:.2f}, method={method}")
            else:
                print("   没有原始检测结果")
                # 即使没有原始检测也测试完整流程
                official_faces = detector.detect_faces(frame)
                print(f"\n🔄 完整检测流程结果: {len(official_faces)} 个人脸")
        
        cap.release()
        
        # 显示检测器统计
        stats = detector.get_detection_info()
        print(f"\n📊 最后10帧检测器统计:")
        for key, value in stats['detection_stats'].items():
            if value > 0:
                print(f"   {key}: {value}")
        
    except Exception as e:
        logging.error(f"调试失败: {e}")

def test_debug_video_first_frames():
    """调试前5帧（保留原有功能）"""
    config = load_config()
    detector = AdvancedFaceDetector(config)
    
    # 显示配置信息
    print("\n⚙️ 当前配置:")
    print(f"   最小人脸尺寸: {detector.size_constraints.get('min_face_size', 30)}px")
    print(f"   最大人脸尺寸: {detector.size_constraints.get('max_face_size', 300)}px")
    print(f"   最小置信度: {detector.quality_filters.get('min_confidence', 0.6)}")
    print(f"   宽高比范围: {detector.size_constraints.get('min_aspect_ratio', 0.5)}-{detector.size_constraints.get('max_aspect_ratio', 2.0)}")
    
    video_path = config.get('video_input_path', 'data/input_video.mp4')
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return
        
        print(f"\n🎬 调试视频前5帧: {video_path}")
        
        # 只处理前5帧进行详细调试
        for frame_num in range(5):
            ret, frame = cap.read()
            if not ret:
                break
            
            print(f"\n" + "="*60)
            print(f"🖼️ 调试帧 {frame_num + 1}")
            print(f"   帧尺寸: {frame.shape[1]}x{frame.shape[0]}")
            
            # 调试原始检测
            raw_detections = debug_enhanced_opencv_detector(detector, frame)
            
            # 如果有原始检测，调试过滤过程
            if raw_detections:
                filtered_detections = debug_filtering_process(detector, raw_detections, frame.shape)
                
                # 测试完整的检测流程
                print(f"\n🔄 完整检测流程结果:")
                official_faces = detector.detect_faces(frame)
                print(f"   最终检测数: {len(official_faces)}")
                
                for i, face in enumerate(official_faces):
                    bbox = face['bbox']
                    confidence = face.get('confidence', 0)
                    method = face.get('method', 'unknown')
                    print(f"   人脸 {i+1}: bbox={bbox}, confidence={confidence:.2f}, method={method}")
            else:
                print("   没有原始检测结果")
        
        cap.release()
        
        # 显示检测器统计
        stats = detector.get_detection_info()
        print(f"\n📊 检测器统计:")
        for key, value in stats['detection_stats'].items():
            if value > 0:
                print(f"   {key}: {value}")
        
    except Exception as e:
        logging.error(f"调试失败: {e}")

def test_debug_video_range(start_frame, end_frame):
    """调试指定帧范围"""
    config = load_config()
    detector = AdvancedFaceDetector(config)
    
    # 显示配置信息
    print("\n⚙️ 当前配置:")
    print(f"   最小人脸尺寸: {detector.size_constraints.get('min_face_size', 30)}px")
    print(f"   最大人脸尺寸: {detector.size_constraints.get('max_face_size', 300)}px")
    print(f"   最小置信度: {detector.quality_filters.get('min_confidence', 0.6)}")
    print(f"   宽高比范围: {detector.size_constraints.get('min_aspect_ratio', 0.5)}-{detector.size_constraints.get('max_aspect_ratio', 2.0)}")
    
    video_path = config.get('video_input_path', 'data/input_video.mp4')
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n🎬 调试视频: {video_path}")
        print(f"   总帧数: {total_frames}")
        print(f"   帧率: {fps:.2f} FPS")
        
        # 确保帧范围合理
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_frames - 1))
        
        print(f"\n🔍 开始调试帧范围 {start_frame + 1} 到 {end_frame + 1} (共 {end_frame - start_frame + 1} 帧)")
        
        # 跳转到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame_num = start_frame
        
        while current_frame_num <= end_frame:
            ret, frame = cap.read()
            if not ret:
                print("到达视频末尾")
                break
            
            current_frame_num += 1
            
            print(f"\n" + "="*60)
            print(f"🖼️ 调试帧 {current_frame_num} (范围内第 {current_frame_num - start_frame} 帧)")
            print(f"   帧尺寸: {frame.shape[1]}x{frame.shape[0]}")
            print(f"   时间戳: {current_frame_num / fps:.2f}s")
            
            # 特别关注侧脸检测
            raw_detections = debug_enhanced_opencv_detector_with_profile_focus(detector, frame)
            
            # 如果有原始检测，调试过滤过程
            if raw_detections:
                filtered_detections = debug_filtering_process(detector, raw_detections, frame.shape)
                
                # 测试完整的检测流程
                print(f"\n🔄 完整检测流程结果:")
                official_faces = detector.detect_faces(frame)
                print(f"   最终检测数: {len(official_faces)}")
                
                for i, face in enumerate(official_faces):
                    bbox = face['bbox']
                    confidence = face.get('confidence', 0)
                    method = face.get('method', 'unknown')
                    print(f"   人脸 {i+1}: bbox={bbox}, confidence={confidence:.2f}, method={method}")
                    
                    # 分析检测到的人脸类型
                    face_width = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
                    face_height = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
                    aspect_ratio = face_width / face_height if face_height > 0 else 0
                    
                    if 'profile' in method.lower():
                        print(f"       -> 🎯 侧脸检测成功! 尺寸={face_width}x{face_height}, 宽高比={aspect_ratio:.2f}")
                    else:
                        print(f"       -> 正脸检测: 尺寸={face_width}x{face_height}, 宽高比={aspect_ratio:.2f}")
            else:
                print("   没有原始检测结果")
                # 即使没有原始检测也测试完整流程
                official_faces = detector.detect_faces(frame)
                print(f"\n🔄 完整检测流程结果: {len(official_faces)} 个人脸")
                
            # 每5帧显示一次间隔
            if (current_frame_num - start_frame) % 5 == 0:
                print(f"\n--- 已处理 {current_frame_num - start_frame + 1} 帧 ---")
        
        cap.release()
        
        # 显示检测器统计
        stats = detector.get_detection_info()
        print(f"\n📊 帧 {start_frame + 1}-{end_frame + 1} 检测器统计:")
        for key, value in stats['detection_stats'].items():
            if value > 0:
                print(f"   {key}: {value}")
        
        # 分析侧脸检测问题
        print(f"\n🎯 侧脸检测分析:")
        opencv_info = stats.get('opencv_detector_info', {})
        available_detectors = opencv_info.get('available_detectors', [])
        print(f"   可用检测器: {available_detectors}")
        
        if 'profile_face' in available_detectors:
            print("   ✅ profile_face 检测器已加载")
        else:
            print("   ❌ profile_face 检测器未加载")
            
        print(f"   💡 建议: 如果侧脸检测效果不佳，可能需要:")
        print(f"       1. 调整 profile_face 检测器参数")
        print(f"       2. 降低侧脸检测的置信度阈值")
        print(f"       3. 放宽宽高比约束以适应侧脸形状")
        
    except Exception as e:
        logging.error(f"调试失败: {e}")

def debug_enhanced_opencv_detector_with_profile_focus(detector, frame):
    """调试增强OpenCV检测器，特别关注侧脸检测"""
    print("\n🔍 调试增强OpenCV检测器 (重点关注侧脸):")
    
    if detector.enhanced_opencv_detector is None:
        print("   ❌ 增强OpenCV检测器未初始化")
        return []
    
    # 获取原始检测结果（绕过过滤）
    opencv_detector = detector.enhanced_opencv_detector
    
    # 预处理图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.equalizeHist(gray)
    
    all_raw_detections = []
    
    # 测试每个检测器，特别关注profile_face
    for detector_name, cascade in opencv_detector.detectors.items():
        try:
            # 对于侧脸检测器使用更宽松的参数
            if 'profile' in detector_name.lower():
                print(f"\n   🎯 特别测试侧脸检测器: {detector_name}")
                # 更宽松的参数用于侧脸检测
                faces = cascade.detectMultiScale(
                    enhanced_gray,
                    scaleFactor=1.03,      # 更小的缩放因子
                    minNeighbors=2,        # 更少的邻居要求
                    minSize=(20, 20),      # 更小的最小尺寸
                    maxSize=(300, 300),    # 更大的最大尺寸
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            else:
                # 使用标准参数
                faces = cascade.detectMultiScale(
                    enhanced_gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(20, 20),
                    maxSize=(500, 500),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            
            print(f"   {detector_name}: 检测到 {len(faces)} 个原始候选")
            
            for i, (x, y, w, h) in enumerate(faces):
                detection = {
                    'bbox': [x, y, x + w, y + h],
                    'width': w,
                    'height': h,
                    'aspect_ratio': w / h,
                    'area': w * h,
                    'method': detector_name
                }
                all_raw_detections.append(detection)
                
                # 特别标注侧脸检测
                if 'profile' in detector_name.lower():
                    print(f"     🎯 侧脸候选 {i+1}: 位置=({x},{y}), 尺寸={w}x{h}, 宽高比={w/h:.2f}, 面积={w*h}")
                else:
                    print(f"     候选 {i+1}: 位置=({x},{y}), 尺寸={w}x{h}, 宽高比={w/h:.2f}, 面积={w*h}")
                
        except Exception as e:
            print(f"   检测器 {detector_name} 失败: {e}")
    
    # 统计侧脸检测情况
    profile_detections = [d for d in all_raw_detections if 'profile' in d['method'].lower()]
    front_detections = [d for d in all_raw_detections if 'profile' not in d['method'].lower()]
    
    print(f"\n   📊 检测统计:")
    print(f"   总原始检测数: {len(all_raw_detections)}")
    print(f"   侧脸检测数: {len(profile_detections)}")
    print(f"   正脸检测数: {len(front_detections)}")
    
    return all_raw_detections

def test_debug_specific_frames(frame_list):
    """调试指定的多个帧"""
    config = load_config()
    detector = AdvancedFaceDetector(config)
    
    # 显示配置信息
    print("\n⚙️ 当前配置:")
    print(f"   最小人脸尺寸: {detector.size_constraints.get('min_face_size', 30)}px")
    print(f"   最大人脸尺寸: {detector.size_constraints.get('max_face_size', 300)}px")
    print(f"   最小置信度: {detector.quality_filters.get('min_confidence', 0.6)}")
    print(f"   宽高比范围: {detector.size_constraints.get('min_aspect_ratio', 0.5)}-{detector.size_constraints.get('max_aspect_ratio', 2.0)}")
    
    video_path = config.get('video_input_path', 'data/input_video.mp4')
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n🎬 调试视频: {video_path}")
        print(f"   总帧数: {total_frames}")
        print(f"   帧率: {fps:.2f} FPS")
        print(f"\n🔍 分析指定帧: {frame_list}")
        
        # 按帧号排序
        sorted_frames = sorted([f - 1 for f in frame_list])  # 转换为0基索引
        
        detection_summary = {}
        
        for frame_idx in sorted_frames:
            frame_num = frame_idx + 1  # 显示时转回1基索引
            
            if frame_idx >= total_frames:
                print(f"\n❌ 帧 {frame_num} 超出视频范围 (总帧数: {total_frames})")
                continue
            
            # 跳转到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"\n❌ 无法读取帧 {frame_num}")
                continue
            
            print(f"\n" + "="*60)
            print(f"🖼️ 分析帧 {frame_num}")
            print(f"   帧尺寸: {frame.shape[1]}x{frame.shape[0]}")
            print(f"   时间戳: {frame_num / fps:.2f}s")
            
            # 调试原始检测
            raw_detections = debug_enhanced_opencv_detector_with_profile_focus(detector, frame)
            
            # 过滤检测
            if raw_detections:
                filtered_detections = debug_filtering_process(detector, raw_detections, frame.shape)
                
                # 完整检测流程
                print(f"\n🔄 完整检测流程结果:")
                official_faces = detector.detect_faces(frame)
                print(f"   最终检测数: {len(official_faces)}")
                
                # 保存检测结果摘要
                detection_summary[frame_num] = {
                    'raw_count': len(raw_detections),
                    'filtered_count': len(filtered_detections),
                    'final_count': len(official_faces),
                    'faces': []
                }
                
                for i, face in enumerate(official_faces):
                    bbox = face['bbox']
                    confidence = face.get('confidence', 0)
                    method = face.get('method', 'unknown')
                    face_width = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
                    face_height = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
                    aspect_ratio = face_width / face_height if face_height > 0 else 0
                    
                    face_info = {
                        'bbox': bbox,
                        'size': f"{face_width}x{face_height}",
                        'aspect_ratio': aspect_ratio,
                        'confidence': confidence,
                        'method': method
                    }
                    detection_summary[frame_num]['faces'].append(face_info)
                    
                    print(f"   人脸 {i+1}: 位置=({bbox[0]},{bbox[1]}), 尺寸={face_width}x{face_height}, 宽高比={aspect_ratio:.2f}, 置信度={confidence:.2f}, 方法={method}")
                    
                    if 'profile' in method.lower():
                        print(f"       -> 🎯 侧脸检测! ")
            else:
                print("   没有原始检测结果")
                official_faces = detector.detect_faces(frame)
                detection_summary[frame_num] = {
                    'raw_count': 0,
                    'filtered_count': 0,
                    'final_count': len(official_faces),
                    'faces': []
                }
                print(f"\n🔄 完整检测流程结果: {len(official_faces)} 个人脸")
        
        cap.release()
        
        # 显示总体分析摘要
        print(f"\n" + "="*80)
        print(f"📊 指定帧检测分析摘要")
        print(f"="*80)
        
        total_raw = sum(info['raw_count'] for info in detection_summary.values())
        total_filtered = sum(info['filtered_count'] for info in detection_summary.values())
        total_final = sum(info['final_count'] for info in detection_summary.values())
        
        print(f"\n🔢 整体统计:")
        print(f"   分析帧数: {len(detection_summary)}")
        print(f"   原始检测总数: {total_raw}")
        print(f"   过滤后总数: {total_filtered}")
        print(f"   最终检测总数: {total_final}")
        if total_raw > 0:
            print(f"   过滤通过率: {(total_filtered/total_raw)*100:.1f}%")
            print(f"   最终通过率: {(total_final/total_raw)*100:.1f}%")
        
        print(f"\n📋 逐帧详情:")
        for frame_num in sorted([int(f) for f in detection_summary.keys()]):
            info = detection_summary[frame_num]
            status = "✅" if info['final_count'] > 0 else "❌"
            print(f"   帧 {frame_num:2d}: {status} 原始={info['raw_count']:2d} -> 过滤={info['filtered_count']:2d} -> 最终={info['final_count']} 个人脸")
            
            # 显示检测到的人脸详情
            for i, face in enumerate(info['faces']):
                method_type = "侧脸" if 'profile' in face['method'].lower() else "正脸"
                print(f"           人脸{i+1}: {face['size']}, 宽高比={face['aspect_ratio']:.2f}, 置信度={face['confidence']:.2f} ({method_type})")
        
        # 分析检测模式
        print(f"\n🎯 检测模式分析:")
        frames_with_detection = [f for f, info in detection_summary.items() if info['final_count'] > 0]
        frames_without_detection = [f for f, info in detection_summary.items() if info['final_count'] == 0]
        
        if frames_with_detection:
            print(f"   ✅ 成功检测帧: {sorted(frames_with_detection)} ({len(frames_with_detection)}/{len(detection_summary)})")
        if frames_without_detection:
            print(f"   ❌ 无检测帧: {sorted(frames_without_detection)} ({len(frames_without_detection)}/{len(detection_summary)})")
        
        # 检测方法分析
        all_methods = []
        profile_count = 0
        front_count = 0
        for info in detection_summary.values():
            for face in info['faces']:
                all_methods.append(face['method'])
                if 'profile' in face['method'].lower():
                    profile_count += 1
                else:
                    front_count += 1
        
        if all_methods:
            print(f"\n🔬 检测方法统计:")
            method_counts = {}
            for method in all_methods:
                method_counts[method] = method_counts.get(method, 0) + 1
            
            for method, count in sorted(method_counts.items()):
                print(f"   {method}: {count} 次")
            
            print(f"\n📊 人脸类型分布:")
            print(f"   正脸检测: {front_count} 次")
            print(f"   侧脸检测: {profile_count} 次")
            
        # 尺寸分析
        all_sizes = []
        for info in detection_summary.values():
            for face in info['faces']:
                size_parts = face['size'].split('x')
                if len(size_parts) == 2:
                    width = int(size_parts[0])
                    height = int(size_parts[1])
                    all_sizes.append((width, height))
        
        if all_sizes:
            widths = [s[0] for s in all_sizes]
            heights = [s[1] for s in all_sizes]
            print(f"\n📏 人脸尺寸分析:")
            print(f"   宽度范围: {min(widths)}-{max(widths)}px, 平均: {sum(widths)/len(widths):.1f}px")
            print(f"   高度范围: {min(heights)}-{max(heights)}px, 平均: {sum(heights)/len(heights):.1f}px")
        
    except Exception as e:
        logging.error(f"调试失败: {e}")

if __name__ == "__main__":
    print("🎾 调试版人脸检测器测试")
    print("="*60)
    
    # 询问用户要调试哪部分
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "first":
            print("🔍 调试前5帧...")
            test_debug_video_first_frames()
        elif sys.argv[1] == "profile":
            print("🔍 调试45-60帧 (侧脸重点分析)...")
            test_debug_video_range(44, 59)  # 转换为0基索引
        elif sys.argv[1] == "frames":
            # 支持多个指定帧分析
            if len(sys.argv) > 2:
                try:
                    frame_list = [int(f) for f in sys.argv[2:]]
                    print(f"🔍 分析指定帧: {frame_list}...")
                    test_debug_specific_frames(frame_list)
                except ValueError:
                    print("❌ 无效的帧号参数")
                    print("用法: python test_debug_face_detector.py frames 50 45 46 56 61 62 79 71")
            else:
                print("❌ 缺少帧号参数")
                print("用法: python test_debug_face_detector.py frames 50 45 46 56 61 62 79 71")
        elif len(sys.argv) >= 3 and sys.argv[1] == "range":
            try:
                start = int(sys.argv[2]) - 1  # 转换为0基索引
                end = int(sys.argv[3]) - 1 if len(sys.argv) > 3 else start + 15
                print(f"🔍 调试帧范围 {start+1}-{end+1}...")
                test_debug_video_range(start, end)
            except ValueError:
                print("❌ 无效的帧范围参数")
                print("用法: python test_debug_face_detector.py range <起始帧> <结束帧>")
        else:
            print("🔍 调试最后10帧...")
            test_debug_video()
    else:
        print("🔍 调试最后10帧...")
        test_debug_video() 