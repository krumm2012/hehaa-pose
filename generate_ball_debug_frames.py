#!/usr/bin/env python3
"""
指定帧编号球检测调试图像生成脚本
基于default_config.yaml生成带球识别框的指定帧图像
"""

import cv2
import yaml
import os
import argparse
from ball_tracker import BallTracker

def load_config(config_path="configs/default_config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_debug_config(config):
    """设置调试配置"""
    if 'ball_detection_debug' not in config:
        config['ball_detection_debug'] = {}
    
    # 强制启用所有调试功能
    config['ball_detection_debug'].update({
        'enabled': True,
        'log_hsv_detection': True,
        'log_boundary_check': True,
        'log_size_filtering': True,
        'log_static_filtering': False,
        'log_advanced_processing': False,
        'log_simulation': False,
        'save_debug_frames': True,
        'debug_frame_interval': 1,  # 每帧都保存
        'max_log_detections': 20
    })
    
    return config

def generate_debug_frame(frame, detected_balls, frame_num, config, output_dir="debug_frames"):
    """生成带球检测框的调试图像"""
    debug_frame = frame.copy()
    
    # 绘制检测到的球
    for i, ball in enumerate(detected_balls):
        x, y = int(ball[0]), int(ball[1])
        # 绘制球的圆圈
        cv2.circle(debug_frame, (x, y), 15, (0, 255, 0), 3)
        cv2.circle(debug_frame, (x, y), 3, (0, 255, 0), -1)  # 中心点
        
        # 标注球编号
        cv2.putText(debug_frame, f"Ball {i+1}", (x-25, y-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 标注坐标
        cv2.putText(debug_frame, f"({x},{y})", (x-30, y+40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 绘制边界（如果启用）
    if config.get('use_boundary', False):
        x1 = config['boundary_x1']
        y1 = config['boundary_y1']
        x2 = config['boundary_x2']
        y2 = config['boundary_y2']
        
        # 绘制边界矩形
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        
        # 标注边界信息
        cv2.putText(debug_frame, f"BOUNDARY ({x1},{y1})-({x2},{y2})", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # 添加帧信息
    info_texts = [
        f"Frame: {frame_num}",
        f"Balls Detected: {len(detected_balls)}",
        f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
        f"Config: default_config.yaml"
    ]
    
    for i, text in enumerate(info_texts):
        cv2.putText(debug_frame, text, (10, 30 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # 添加黑色背景让文字更清晰
        cv2.putText(debug_frame, text, (10, 30 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(debug_frame, text, (10, 30 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存图像
    filename = f"{output_dir}/ball_frame_{frame_num:04d}_debug.jpg"
    cv2.imwrite(filename, debug_frame)
    
    return filename

def process_specific_frames(frame_numbers, config_path="configs/default_config.yaml", output_dir="debug_frames"):
    """处理指定的帧编号"""
    print(f"🎯 球检测调试图像生成器")
    print(f"📄 配置文件: {config_path}")
    print(f"🎬 目标帧编号: {frame_numbers}")
    print(f"📁 输出目录: {output_dir}")
    print("="*70)
    
    # 加载配置
    config = load_config(config_path)
    config = setup_debug_config(config)
    
    # 显示配置信息
    print("🔧 球检测配置:")
    print(f"   球尺寸范围: {config.get('min_ball_radius', 'N/A')}-{config.get('max_ball_radius', 'N/A')}px")
    print(f"   边界检查: {config.get('use_boundary', False)}")
    if config.get('use_boundary', False):
        print(f"   边界范围: ({config.get('boundary_x1', 0)},{config.get('boundary_y1', 0)})-({config.get('boundary_x2', 0)},{config.get('boundary_y2', 0)})")
    print("="*70)
    
    # 初始化球检测器
    ball_tracker = BallTracker(config.get('tracknet_model_path', None), config)
    
    # 打开视频
    video_path = config['video_input_path']
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 视频信息:")
    print(f"   总帧数: {total_frames}")
    print(f"   帧率: {fps:.2f} FPS")
    print(f"   时长: {total_frames/fps:.2f} 秒")
    print("="*70)
    
    generated_files = []
    
    try:
        for frame_num in sorted(frame_numbers):
            if frame_num < 1 or frame_num > total_frames:
                print(f"⚠️  帧 {frame_num} 超出范围 (1-{total_frames})，跳过")
                continue
            
            print(f"\n🎯 处理帧 {frame_num}")
            print("-" * 50)
            
            # 跳转到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # OpenCV帧编号从0开始
            ret, frame = cap.read()
            
            if not ret:
                print(f"❌ 无法读取帧 {frame_num}")
                continue
            
            # 重置帧计数器以获得正确的日志输出
            ball_tracker.frame_counter = frame_num - 1
            
            # 球检测
            detected_balls = ball_tracker.predict_ball(frame)
            
            # 生成调试图像
            output_file = generate_debug_frame(frame, detected_balls, frame_num, config, output_dir)
            generated_files.append(output_file)
            
            print(f"✅ 帧 {frame_num}: 检测到 {len(detected_balls)} 个球")
            for i, ball in enumerate(detected_balls):
                print(f"   球 {i+1}: 位置 ({ball[0]}, {ball[1]})")
            print(f"📸 已保存: {output_file}")
    
    except KeyboardInterrupt:
        print("\n⏹️  处理被用户中断")
    except Exception as e:
        print(f"\n❌ 处理出错: {e}")
    finally:
        cap.release()
    
    # 显示结果总结
    print("\n" + "="*70)
    print("📊 处理结果总结")
    print("="*70)
    print(f"目标帧数: {len(frame_numbers)}")
    print(f"成功生成: {len(generated_files)}")
    print(f"输出目录: {output_dir}")
    print("\n🖼️  生成的调试图像:")
    for file in generated_files:
        print(f"   {file}")
    
    return generated_files

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成指定帧编号的球检测调试图像')
    parser.add_argument('frames', nargs='+', type=int, help='要处理的帧编号 (例如: 5 10 25 50)')
    parser.add_argument('--config', '-c', default='configs/default_config.yaml', 
                       help='配置文件路径 (默认: configs/default_config.yaml)')
    parser.add_argument('--output', '-o', default='debug_frames', 
                       help='输出目录 (默认: debug_frames)')
    
    args = parser.parse_args()
    
    # 验证帧编号
    if not args.frames:
        print("❌ 请指定至少一个帧编号")
        return
    
    if any(f <= 0 for f in args.frames):
        print("❌ 帧编号必须大于0")
        return
    
    # 处理帧
    generated_files = process_specific_frames(args.frames, args.config, args.output)
    
    if generated_files:
        print(f"\n🎉 成功生成 {len(generated_files)} 个调试图像！")
    else:
        print(f"\n😟 没有成功生成任何图像")

if __name__ == "__main__":
    main() 