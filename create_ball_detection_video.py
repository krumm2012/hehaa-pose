#!/usr/bin/env python3
"""
🎾 网球检测视频生成器
批量生成带有球检测标识的输出视频
支持无显示模式，适合服务器环境使用
"""

import cv2
import yaml
import numpy as np
import os
import argparse
import time
from datetime import datetime
from ball_tracker import BallTracker

def load_config(config_path="configs/noise_filtered_ball_config.yaml"):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        print(f"✅ 配置加载成功: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return None

def draw_legend(frame):
    """绘制颜色标识图例"""
    height, width = frame.shape[:2]
    
    # 图例位置
    legend_x = 20
    legend_y = height - 200
    legend_width = 300
    legend_height = 160
    
    # 绘制半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), 
                  (50, 50, 50), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # 绘制边框
    cv2.rectangle(frame, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), 
                  (255, 255, 255), 2)
    
    # 图例标题
    cv2.putText(frame, "Ball Detection Results", (legend_x + 10, legend_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 运动球图例 - 红色
    cv2.circle(frame, (legend_x + 30, legend_y + 60), 12, (0, 0, 255), -1)
    cv2.circle(frame, (legend_x + 30, legend_y + 60), 15, (0, 0, 255), 2)
    cv2.putText(frame, "Moving Ball", (legend_x + 60, legend_y + 67), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 静止球图例 - 蓝色
    cv2.circle(frame, (legend_x + 30, legend_y + 95), 12, (255, 100, 0), -1)
    cv2.circle(frame, (legend_x + 30, legend_y + 95), 15, (255, 100, 0), 2)
    cv2.putText(frame, "Static Ball", (legend_x + 60, legend_y + 102), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    
    # 轨迹图例 - 绿色
    cv2.line(frame, (legend_x + 20, legend_y + 125), (legend_x + 40, legend_y + 125), (0, 255, 0), 4)
    cv2.putText(frame, "Ball Trajectory", (legend_x + 60, legend_y + 132), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

def add_timestamp_watermark(frame, timestamp_text):
    """添加时间戳水印"""
    height, width = frame.shape[:2]
    
    # 时间戳位置（右上角）
    timestamp_x = width - 300
    timestamp_y = 30
    
    # 绘制半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (timestamp_x - 10, timestamp_y - 25), 
                  (timestamp_x + 280, timestamp_y + 10), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
    
    # 绘制时间戳文字
    cv2.putText(frame, timestamp_text, (timestamp_x, timestamp_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def create_ball_detection_video(input_video, output_video, config_path, show_progress=True, add_timestamp=True):
    """
    创建带球检测标识的视频
    
    Args:
        input_video: 输入视频路径
        output_video: 输出视频路径
        config_path: 配置文件路径
        show_progress: 是否显示进度
        add_timestamp: 是否添加时间戳
    """
    
    print(f"🚀 开始生成球检测视频...")
    print(f"📹 输入: {input_video}")
    print(f"💾 输出: {output_video}")
    
    # 加载配置
    config = load_config(config_path)
    if not config:
        return False
    
    # 更新配置中的视频路径
    config['video_input_path'] = input_video
    config['video_output_path'] = output_video
    
    # 初始化球检测模块
    print(f"🔧 初始化球检测模块...")
    ball_tracker = BallTracker(config.get('tracknet_model_path', None), config)
    
    # 打开输入视频
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"❌ 无法打开输入视频: {input_video}")
        return False
    
    # 获取视频信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 视频信息: {frame_width}x{frame_height}, {fps}FPS, {total_frames}帧")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    if not video_writer.isOpened():
        print(f"❌ 无法创建输出视频文件: {output_video}")
        cap.release()
        return False
    
    # 处理统计变量
    frame_count = 0
    static_count = 0
    moving_count = 0
    start_time = time.time()
    
    print(f"\n🎬 开始处理视频帧...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 执行球检测
        detected_balls = ball_tracker.predict_ball(frame)
        
        # 应用高级处理获取运动球
        moving_balls = ball_tracker.advanced_ball_processing(detected_balls, frame_count)
        
        # 创建输出帧
        output_frame = frame.copy()
        
        # 绘制轨迹（绿色）
        output_frame = ball_tracker.draw_trajectory(output_frame)
        
        # 绘制静止球（蓝色）
        output_frame = ball_tracker.draw_static_balls(output_frame)
        
        # 绘制运动球（红色）
        for ball in moving_balls:
            # 处理不同的数据格式
            if isinstance(ball, dict):
                center = (int(ball['x']), int(ball['y']))
                radius = int(ball.get('radius', 10))
            else:
                # 假设是 (x, y, radius) 元组格式
                center = (int(ball[0]), int(ball[1]))
                radius = int(ball[2]) if len(ball) > 2 else 10
            
            # 绘制实心圆
            cv2.circle(output_frame, center, radius, (0, 0, 255), -1)
            # 绘制边框
            cv2.circle(output_frame, center, radius + 3, (0, 0, 255), 2)
            # 添加标签
            cv2.putText(output_frame, "MOVING", (center[0] - 30, center[1] - radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 绘制边界框（如果启用）
        if config.get('draw_boundary', False):
            boundary_color = (0, 255, 255)  # 黄色边界
            cv2.rectangle(output_frame, 
                         (config['boundary_x1'], config['boundary_y1']), 
                         (config['boundary_x2'], config['boundary_y2']), 
                         boundary_color, 2)
            cv2.putText(output_frame, "DETECTION AREA", 
                       (config['boundary_x1'], config['boundary_y1'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, boundary_color, 1)
        
        # 计算统计信息
        current_static = 0
        for ball_id, static_info in ball_tracker.static_ball_candidates.items():
            if static_info['frames_still'] >= config['static_ball_frames_threshold']:
                current_static += 1
        
        current_moving = len(moving_balls)
        
        # 更新计数
        static_count = max(static_count, current_static)
        if current_moving > 0:
            moving_count += 1
        
        # 绘制图例
        output_frame = draw_legend(output_frame)
        
        # 显示统计信息
        info_y = 30
        info_x = frame_width - 300
        cv2.putText(output_frame, f"Frame: {frame_count}/{total_frames}", 
                   (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Static Balls: {current_static}", 
                   (info_x, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        cv2.putText(output_frame, f"Moving Balls: {current_moving}", 
                   (info_x, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(output_frame, f"Detections: {len(detected_balls)}", 
                   (info_x, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 添加时间戳（如果启用）
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_frame = add_timestamp_watermark(output_frame, f"Generated: {timestamp}")
        
        # 添加处理状态
        cv2.putText(output_frame, "PROCESSING", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 写入输出视频
        video_writer.write(output_frame)
        
        # 显示进度
        if show_progress and frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            progress = (frame_count / total_frames) * 100
            eta = (elapsed_time / frame_count) * (total_frames - frame_count)
            print(f"📈 进度: {progress:.1f}% ({frame_count}/{total_frames}) - ETA: {eta:.1f}s")
    
    # 清理资源
    cap.release()
    video_writer.release()
    
    # 检查输出文件
    if os.path.exists(output_video):
        file_size = os.path.getsize(output_video) / (1024 * 1024)  # MB
        total_time = time.time() - start_time
        print(f"\n✅ 视频生成成功!")
        print(f"📁 输出文件: {output_video}")
        print(f"📏 文件大小: {file_size:.1f} MB")
        print(f"⏱️  处理时间: {total_time:.1f}秒")
        print(f"🎯 处理速度: {frame_count/total_time:.1f} fps")
        
        # 显示检测统计
        print(f"\n📊 检测统计:")
        print(f"🔵 最大静止球数量: {static_count}")
        print(f"🔴 运动球检测帧数: {moving_count}")
        print(f"📈 运动球识别率: {moving_count/max(1, frame_count)*100:.1f}%")
        
        if len(ball_tracker.tracked_balls_history) > 0:
            print(f"🟢 轨迹总长度: {len(ball_tracker.tracked_balls_history)} 点")
            
            # 计算轨迹总距离
            total_distance = 0
            if len(ball_tracker.tracked_balls_history) > 1:
                for i in range(1, len(ball_tracker.tracked_balls_history)):
                    p1 = np.array(ball_tracker.tracked_balls_history[i-1]['coords'])
                    p2 = np.array(ball_tracker.tracked_balls_history[i]['coords'])
                    total_distance += np.linalg.norm(p2 - p1)
            print(f"📏 轨迹总距离: {total_distance:.1f}px")
        
        return True
    else:
        print(f"❌ 视频生成失败: {output_video}")
        return False

def main():
    """主函数 - 支持命令行参数"""
    parser = argparse.ArgumentParser(description='🎾 网球检测视频生成器')
    parser.add_argument('-i', '--input', default='data/input_video.mp4', 
                       help='输入视频路径')
    parser.add_argument('-o', '--output', default='data/ball_detection_output.mp4', 
                       help='输出视频路径')
    parser.add_argument('-c', '--config', default='configs/noise_filtered_ball_config.yaml', 
                       help='配置文件路径')
    parser.add_argument('--no-progress', action='store_true', 
                       help='不显示进度信息')
    parser.add_argument('--no-timestamp', action='store_true', 
                       help='不添加时间戳水印')
    
    args = parser.parse_args()
    
    try:
        success = create_ball_detection_video(
            input_video=args.input,
            output_video=args.output,
            config_path=args.config,
            show_progress=not args.no_progress,
            add_timestamp=not args.no_timestamp
        )
        
        if success:
            print(f"\n🎉 任务完成! 输出视频已保存到: {args.output}")
        else:
            print(f"\n❌ 任务失败!")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断处理")
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main() 