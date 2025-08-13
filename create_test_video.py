#!/usr/bin/env python3
"""
创建一个简单的测试视频用于演示网球分析系统
"""

import cv2
import numpy as np
import os

def create_test_video():
    """创建一个简单的测试视频，包含模拟的网球和球员"""
    
    # 视频参数
    width, height = 1280, 720
    fps = 30
    duration_seconds = 10
    total_frames = fps * duration_seconds
    
    # 确保data目录存在
    os.makedirs('data', exist_ok=True)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('data/input_video.mp4', fourcc, fps, (width, height))
    
    print(f"正在创建测试视频: data/input_video.mp4")
    print(f"视频参数: {width}x{height}, {fps}fps, {duration_seconds}秒")
    
    for frame_num in range(total_frames):
        # 创建绿色背景（模拟网球场）
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (34, 139, 34)  # 森林绿
        
        # 添加网球场线条
        # 中线
        cv2.line(frame, (width//2, 0), (width//2, height), (255, 255, 255), 3)
        # 边界线
        cv2.rectangle(frame, (100, 100), (width-100, height-100), (255, 255, 255), 3)
        
        # 模拟移动的网球
        ball_radius = 8
        
        # 球的运动轨迹（从左到右，带有弧线）
        progress = frame_num / total_frames
        ball_x = int(200 + (width - 400) * progress)
        ball_y = int(height//2 - 100 * np.sin(progress * np.pi * 2))
        
        # 绘制网球（黄绿色）
        cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 255, 255), -1)
        cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 0, 0), 2)
        
        # 添加一些静止的球（作为对比）
        static_balls = [(300, 500), (900, 400), (1100, 600)]
        for static_x, static_y in static_balls:
            cv2.circle(frame, (static_x, static_y), ball_radius, (0, 255, 255), -1)
            cv2.circle(frame, (static_x, static_y), ball_radius, (0, 0, 0), 2)
        
        # 模拟简单的人形图案（球员）
        player_x, player_y = 640, 500
        
        # 头
        cv2.circle(frame, (player_x, player_y - 60), 25, (255, 200, 150), -1)
        # 身体
        cv2.rectangle(frame, (player_x - 15, player_y - 35), (player_x + 15, player_y + 40), (0, 0, 255), -1)
        # 手臂
        cv2.line(frame, (player_x - 15, player_y - 20), (player_x - 40, player_y), (255, 200, 150), 8)
        cv2.line(frame, (player_x + 15, player_y - 20), (player_x + 40, player_y), (255, 200, 150), 8)
        # 腿
        cv2.line(frame, (player_x - 10, player_y + 40), (player_x - 20, player_y + 80), (0, 0, 255), 8)
        cv2.line(frame, (player_x + 10, player_y + 40), (player_x + 20, player_y + 80), (0, 0, 255), 8)
        
        # 球拍
        racket_x = player_x + 40 + int(10 * np.sin(progress * np.pi * 4))
        racket_y = player_y - int(5 * np.cos(progress * np.pi * 4))
        cv2.ellipse(frame, (racket_x, racket_y), (15, 25), 0, 0, 360, (139, 69, 19), 3)
        cv2.line(frame, (racket_x, racket_y + 25), (racket_x, racket_y + 60), (101, 67, 33), 5)
        
        # 添加文本信息
        cv2.putText(frame, f"Tennis Analysis Test Video - Frame {frame_num}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Moving Ball vs Static Balls Demo", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 写入帧
        out.write(frame)
        
        # 显示进度
        if frame_num % 30 == 0:
            print(f"创建进度: {frame_num}/{total_frames} ({progress*100:.1f}%)")
    
    # 释放资源
    out.release()
    print(f"✅ 测试视频创建完成: data/input_video.mp4")
    print(f"视频包含：")
    print(f"  - 1个移动的网球（黄色）")
    print(f"  - 3个静止的网球（用于对比）")
    print(f"  - 1个模拟的球员和球拍")
    print(f"  - 网球场背景")

if __name__ == "__main__":
    create_test_video()


