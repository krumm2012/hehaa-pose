#!/usr/bin/env python3
"""
球拍检测测试脚本
单独测试 YOLO26n 的球拍检测功能
"""

import cv2
import yaml
import numpy as np
from yolo26n_unified_detector import YOLO26nUnifiedDetector


def test_racket_detection(config_path, video_path, output_path, max_frames=100):
    """
    测试球拍检测
    
    Args:
        config_path: 配置文件路径
        video_path: 输入视频路径
        output_path: 输出视频路径
        max_frames: 最大处理帧数
    """
    print("=" * 80)
    print("🎾 球拍检测测试")
    print("=" * 80)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取统一检测配置
    unified_config = config.get('unified_detection', {})
    
    # 初始化检测器
    print(f"\n🚀 初始化检测器...")
    detector = YOLO26nUnifiedDetector(
        unified_config.get('model_path', 'yolo26n.mlpackage'),
        unified_config,
        roi_manager=None
    )
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 视频信息:")
    print(f"   分辨率: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   总帧数: {total_frames}")
    print(f"   处理帧数: {min(max_frames, total_frames)}")
    
    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 统计信息
    stats = {
        'total_frames': 0,
        'frames_with_racket': 0,
        'total_rackets': 0,
        'racket_areas': [],
        'racket_aspect_ratios': [],
        'racket_confidences': []
    }
    
    print(f"\n⏱️  开始处理...")
    print("-" * 80)
    
    frame_num = 0
    while frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        _, racket_detections, _ = detector.detect_unified(frame)
        
        # 绘制结果
        display_frame = frame.copy()
        
        # 统计
        stats['total_frames'] += 1
        if racket_detections:
            stats['frames_with_racket'] += 1
            stats['total_rackets'] += len(racket_detections)
        
        # 绘制每个检测到的球拍
        for i, racket in enumerate(racket_detections):
            box = racket['box']
            conf = racket['confidence']
            
            # 获取额外信息
            area = racket.get('area', 0)
            aspect_ratio = racket.get('aspect_ratio', 0)
            
            # 记录统计
            stats['racket_areas'].append(area)
            stats['racket_aspect_ratios'].append(aspect_ratio)
            stats['racket_confidences'].append(conf)
            
            # 绘制边界框
            x1, y1, x2, y2 = box
            
            # 根据置信度选择颜色
            if conf >= 0.7:
                color = (0, 255, 0)  # 绿色 - 高置信度
            elif conf >= 0.5:
                color = (0, 255, 255)  # 黄色 - 中等置信度
            else:
                color = (0, 165, 255)  # 橙色 - 低置信度
            
            # 绘制框
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
            
            # 计算框的尺寸
            box_width = x2 - x1
            box_height = y2 - y1
            
            # 绘制信息
            info_lines = [
                f"Racket #{i+1}",
                f"Conf: {conf:.2f}",
                f"Size: {box_width}x{box_height}",
                f"Area: {area}",
                f"Ratio: {aspect_ratio:.2f}"
            ]
            
            # 绘制文字背景
            y_offset = y1 - 10
            for line in info_lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display_frame, 
                            (x1, y_offset - text_size[1] - 5),
                            (x1 + text_size[0] + 5, y_offset + 5),
                            color, -1)
                cv2.putText(display_frame, line, (x1 + 2, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                y_offset -= (text_size[1] + 10)
        
        # 绘制帧信息
        info_text = f"Frame: {frame_num + 1}/{max_frames} | Rackets: {len(racket_detections)}"
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 绘制配置信息
        config_text = f"Threshold: {unified_config.get('racket_confidence_threshold', 0.3)}"
        cv2.putText(display_frame, config_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 写入视频
        out.write(display_frame)
        
        # 进度显示
        if (frame_num + 1) % 25 == 0:
            print(f"   处理帧 {frame_num + 1}/{max_frames} - 检测到 {len(racket_detections)} 个球拍")
        
        frame_num += 1
    
    # 释放资源
    cap.release()
    out.release()
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("📊 检测统计")
    print("=" * 80)
    
    print(f"\n总帧数: {stats['total_frames']}")
    print(f"检测到球拍的帧数: {stats['frames_with_racket']} ({stats['frames_with_racket']/stats['total_frames']*100:.1f}%)")
    print(f"总检测数: {stats['total_rackets']}")
    print(f"平均每帧球拍数: {stats['total_rackets']/stats['total_frames']:.2f}")
    
    if stats['racket_confidences']:
        print(f"\n置信度统计:")
        print(f"   平均: {np.mean(stats['racket_confidences']):.3f}")
        print(f"   最小: {np.min(stats['racket_confidences']):.3f}")
        print(f"   最大: {np.max(stats['racket_confidences']):.3f}")
        print(f"   中位数: {np.median(stats['racket_confidences']):.3f}")
    
    if stats['racket_areas']:
        print(f"\n面积统计 (像素²):")
        print(f"   平均: {np.mean(stats['racket_areas']):.0f}")
        print(f"   最小: {np.min(stats['racket_areas']):.0f}")
        print(f"   最大: {np.max(stats['racket_areas']):.0f}")
        print(f"   中位数: {np.median(stats['racket_areas']):.0f}")
    
    if stats['racket_aspect_ratios']:
        print(f"\n宽高比统计:")
        print(f"   平均: {np.mean(stats['racket_aspect_ratios']):.2f}")
        print(f"   最小: {np.min(stats['racket_aspect_ratios']):.2f}")
        print(f"   最大: {np.max(stats['racket_aspect_ratios']):.2f}")
        print(f"   中位数: {np.median(stats['racket_aspect_ratios']):.2f}")
    
    # 分析建议
    print("\n" + "=" * 80)
    print("💡 分析建议")
    print("=" * 80)
    
    if stats['racket_confidences']:
        avg_conf = np.mean(stats['racket_confidences'])
        min_conf = np.min(stats['racket_confidences'])
        
        if avg_conf < 0.6:
            print(f"\n⚠️  平均置信度较低 ({avg_conf:.2f})")
            print(f"   建议: 降低 racket_confidence_threshold 到 {min_conf - 0.05:.2f}")
        elif min_conf > 0.7:
            print(f"\n✅ 置信度很高 (最小 {min_conf:.2f})")
            print(f"   可以提高 racket_confidence_threshold 到 {min_conf:.2f} 以减少误检")
    
    if stats['racket_areas']:
        max_area = np.max(stats['racket_areas'])
        image_area = width * height
        
        if max_area > image_area / 4:
            print(f"\n⚠️  检测到过大的边界框 ({max_area:.0f} 像素²)")
            print(f"   建议: 检查是否为误检")
        
        min_area = np.min(stats['racket_areas'])
        if min_area < 1000:
            print(f"\n⚠️  检测到过小的边界框 ({min_area:.0f} 像素²)")
            print(f"   建议: 可能需要降低 min_area 阈值")
    
    if stats['racket_aspect_ratios']:
        ratios = stats['racket_aspect_ratios']
        unusual_ratios = [r for r in ratios if r < 0.5 or r > 5.0]
        
        if unusual_ratios:
            print(f"\n⚠️  检测到异常宽高比: {unusual_ratios}")
            print(f"   建议: 检查这些检测是否为误检")
    
    print(f"\n✅ 输出视频已保存: {output_path}")
    print("=" * 80)
    
    return stats


if __name__ == "__main__":
    import sys
    
    # 默认参数
    config_path = "configs/yolo26_tennis_config.yaml"
    video_path = "data/16.10.mp4"
    output_path = "data/racket_detection_test.mp4"
    max_frames = 250  # 处理所有帧
    
    # 命令行参数
    if len(sys.argv) > 1:
        max_frames = int(sys.argv[1])
    
    # 运行测试
    stats = test_racket_detection(config_path, video_path, output_path, max_frames)
    
    print("\n🎉 测试完成！")
    print(f"请查看输出视频: {output_path}")
