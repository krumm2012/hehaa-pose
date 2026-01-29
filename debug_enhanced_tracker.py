#!/usr/bin/env python3
"""
debug_enhanced_tracker.py
调试增强版追踪器，可视化检测过程
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from ball_tracker_enhanced import BallTrackerEnhanced


def debug_detection(video_path, output_path, max_frames=10):
    """调试检测过程，保存中间结果"""
    
    print("\n🔍 调试增强版追踪器")
    print("="*60)
    
    # 创建追踪器
    tracker = BallTrackerEnhanced({
        'enable_kalman': True,
        'enable_multi_colorspace': True,
        'enable_adaptive_threshold': True
    })
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\n📹 视频信息:")
    print(f"   分辨率: {width}x{height}")
    print(f"   FPS: {fps}")
    
    # 创建输出目录
    debug_dir = 'output/debug_frames'
    os.makedirs(debug_dir, exist_ok=True)
    
    frame_num = 0
    
    while cap.isOpened() and frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        print(f"\n{'='*60}")
        print(f"帧 {frame_num}")
        print(f"{'='*60}")
        
        # 缩小图像
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        small_frame = cv2.GaussianBlur(small_frame, (3, 3), 0)
        
        # 检测各个颜色空间
        hsv_mask = tracker._detect_hsv(small_frame, adaptive=True)
        lab_mask = tracker._detect_lab(small_frame)
        ycrcb_mask = tracker._detect_ycrcb(small_frame)
        
        # 融合
        combined_mask = tracker._fuse_masks(hsv_mask, lab_mask, ycrcb_mask)
        
        # 形态学操作
        clean_mask = tracker._morphology_operations(combined_mask)
        
        # 统计
        hsv_pixels = np.sum(hsv_mask > 0)
        lab_pixels = np.sum(lab_mask > 0)
        ycrcb_pixels = np.sum(ycrcb_mask > 0)
        combined_pixels = np.sum(combined_mask > 0)
        clean_pixels = np.sum(clean_mask > 0)
        
        print(f"\n像素统计:")
        print(f"   HSV:      {hsv_pixels:6d} 像素")
        print(f"   LAB:      {lab_pixels:6d} 像素")
        print(f"   YCrCb:    {ycrcb_pixels:6d} 像素")
        print(f"   融合后:   {combined_pixels:6d} 像素")
        print(f"   清理后:   {clean_pixels:6d} 像素")
        
        # 圆检测
        circles = cv2.HoughCircles(
            clean_mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=10,
            minRadius=3,
            maxRadius=30
        )
        
        if circles is not None:
            print(f"\n检测到 {len(circles[0])} 个圆:")
            for i, circle in enumerate(circles[0][:5]):  # 只显示前5个
                x, y, r = circle
                print(f"   圆 {i+1}: 中心=({x:.1f}, {y:.1f}), 半径={r:.1f}")
        else:
            print(f"\n未检测到圆")
        
        # 保存调试图像
        # 创建可视化
        vis_height = small_frame.shape[0]
        vis_width = small_frame.shape[1]
        
        # 转换掩码为彩色
        hsv_vis = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
        lab_vis = cv2.cvtColor(lab_mask, cv2.COLOR_GRAY2BGR)
        ycrcb_vis = cv2.cvtColor(ycrcb_mask, cv2.COLOR_GRAY2BGR)
        combined_vis = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        clean_vis = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
        
        # 在原图上绘制检测结果
        result_frame = small_frame.copy()
        if circles is not None:
            for circle in circles[0]:
                x, y, r = circle
                cv2.circle(result_frame, (int(x), int(y)), int(r), (0, 0, 255), 2)
                cv2.circle(result_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        # 拼接图像 (2行3列)
        row1 = np.hstack([small_frame, hsv_vis, lab_vis])
        row2 = np.hstack([ycrcb_vis, combined_vis, clean_vis])
        row3 = np.hstack([result_frame, result_frame.copy(), result_frame.copy()])
        
        debug_img = np.vstack([row1, row2, row3])
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_img, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(debug_img, 'HSV', (vis_width + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(debug_img, 'LAB', (vis_width*2 + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(debug_img, 'YCrCb', (10, vis_height + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(debug_img, 'Combined', (vis_width + 10, vis_height + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(debug_img, 'Clean', (vis_width*2 + 10, vis_height + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(debug_img, 'Result', (10, vis_height*2 + 30), font, 1, (255, 255, 255), 2)
        
        # 保存
        debug_path = f"{debug_dir}/frame_{frame_num:03d}.jpg"
        cv2.imwrite(debug_path, debug_img)
        print(f"\n💾 调试图像已保存: {debug_path}")
    
    cap.release()
    
    print(f"\n{'='*60}")
    print(f"✅ 调试完成！共处理 {frame_num} 帧")
    print(f"📁 调试图像保存在: {debug_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='调试增强版追踪器')
    parser.add_argument('--video', type=str, default='data/16.10.mp4', help='视频路径')
    parser.add_argument('--output', type=str, default='output/debug_result.mp4', help='输出路径')
    parser.add_argument('--max-frames', type=int, default=10, help='最大帧数')
    
    args = parser.parse_args()
    
    debug_detection(args.video, args.output, args.max_frames)
