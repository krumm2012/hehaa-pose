#!/usr/bin/env python3
# frame_extractor.py
# 用于从视频中提取特定帧并保存为PNG图像

import cv2
import os
import argparse
from pathlib import Path
import yaml
import re

def load_config(config_path="configs/default_config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def timestamp_to_seconds(timestamp):
    """将时间戳字符串转换为秒数
    
    支持的格式:
    - HH:MM:SS
    - MM:SS
    - SS
    """
    if not timestamp:
        return 0
    
    # 尝试匹配不同的时间格式
    match = re.match(r'^(\d+):(\d+):(\d+)$', timestamp)  # HH:MM:SS
    if match:
        h, m, s = map(int, match.groups())
        return h * 3600 + m * 60 + s
    
    match = re.match(r'^(\d+):(\d+)$', timestamp)  # MM:SS
    if match:
        m, s = map(int, match.groups())
        return m * 60 + s
    
    match = re.match(r'^(\d+)$', timestamp)  # SS
    if match:
        return int(match.group(1))
    
    raise ValueError(f"不支持的时间戳格式: {timestamp}")

def extract_frames_by_timestamp(video_path, timestamps, output_dir=None, video_name=None):
    """根据时间戳从视频中提取帧
    
    Args:
        video_path: 视频文件路径
        timestamps: 时间戳列表，格式为 "HH:MM:SS", "MM:SS" 或 "SS"
        output_dir: 输出目录
        video_name: 视频名称
    
    Returns:
        成功提取的帧数
    """
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return 0
    
    # 创建输出目录
    if output_dir is None:
        video_dir = os.path.dirname(video_path)
        output_dir = os.path.join(video_dir, "frames")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果未提供视频名称，则使用文件名
    if video_name is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"视频信息 - 路径: {video_path}, FPS: {fps:.2f}, 总帧数: {total_frames}, 时长: {duration:.2f}秒")
    
    # 提取帧
    extracted_count = 0
    
    for timestamp in timestamps:
        try:
            # 将时间戳转换为秒数
            seconds = timestamp_to_seconds(timestamp)
            
            # 检查时间戳是否在视频范围内
            if seconds > duration:
                print(f"警告: 时间戳 {timestamp} ({seconds}秒) 超出视频时长 {duration:.2f}秒")
                continue
            
            # 计算帧位置
            frame_position = int(seconds * fps)
            
            # 设置帧位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print(f"错误: 无法读取时间戳 {timestamp} 处的帧")
                continue
            
            # 生成输出文件名 (使用时间戳和帧号)
            output_filename = f"{video_name}_time_{timestamp.replace(':', '_')}_frame_{frame_position:04d}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存帧为PNG
            cv2.imwrite(output_path, frame)
            print(f"已保存时间戳 {timestamp} 的帧: {output_path}")
            
            extracted_count += 1
            
        except ValueError as e:
            print(f"错误: {str(e)}")
    
    # 释放资源
    cap.release()
    
    return extracted_count

def extract_frames(video_path, frame_ids, output_dir=None, video_name=None):
    """从视频中提取特定帧并保存为PNG图像
    
    Args:
        video_path: 视频文件路径
        frame_ids: 要提取的帧ID列表
        output_dir: 输出目录，如果为None则使用视频所在目录的"frames"子目录
        video_name: 视频名称，用于生成输出文件名，如果为None则使用视频文件名
    
    Returns:
        成功提取的帧数
    """
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return 0
    
    # 创建输出目录
    if output_dir is None:
        video_dir = os.path.dirname(video_path)
        output_dir = os.path.join(video_dir, "frames")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果未提供视频名称，则使用文件名
    if video_name is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 按照升序排序帧ID
    frame_ids = sorted(frame_ids)
    max_frame_id = max(frame_ids) if frame_ids else 0
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"警告: 无法确定视频总帧数: {video_path}")
        # 如果无法获取总帧数，尝试估算
        if fps > 0:
            duration = 60 * 60  # 假设最大为1小时
            total_frames = int(fps * duration)
        else:
            total_frames = 100000  # 一个很大的值
    
    print(f"视频信息 - 路径: {video_path}, FPS: {fps:.2f}, 总帧数: {total_frames}")
    
    if max_frame_id >= total_frames:
        print(f"警告: 请求的帧ID {max_frame_id} 超出视频总帧数 {total_frames}")
    
    # 提取帧
    extracted_count = 0
    current_frame = 0
    
    for frame_id in frame_ids:
        # 检查帧ID是否有效
        if frame_id < 0 or frame_id >= total_frames:
            print(f"跳过无效的帧ID: {frame_id}")
            continue
        
        # 如果当前帧号小于目标帧号，则跳转到目标帧
        if current_frame < frame_id:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            current_frame = frame_id
        
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print(f"错误: 无法读取帧 {frame_id}")
            break
        
        # 生成输出文件名
        output_filename = f"{video_name}_frame_{frame_id:04d}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 保存帧为PNG
        cv2.imwrite(output_path, frame)
        print(f"已保存: {output_path}")
        
        extracted_count += 1
        current_frame += 1
    
    # 释放资源
    cap.release()
    
    return extracted_count

def extract_from_all_videos(directory, frame_ids=None, timestamps=None, output_dir=None, recursive=False):
    """从目录中的所有视频文件中提取特定帧
    
    Args:
        directory: 视频目录
        frame_ids: 要提取的帧ID列表
        timestamps: 要提取的时间戳列表
        output_dir: 输出目录，如果为None则为每个视频创建一个子目录
        recursive: 是否递归搜索子目录
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos_processed = 0
    frames_extracted = 0
    
    # 如果指定了输出目录，确保它存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有视频文件
    if recursive:
        video_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(root, file))
    else:
        video_files = [os.path.join(directory, f) for f in os.listdir(directory)
                      if os.path.isfile(os.path.join(directory, f)) and 
                      any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        print(f"在目录 {directory} 中未找到视频文件")
        return 0, 0
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    for video_path in video_files:
        print(f"\n处理视频: {video_path}")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 为每个视频创建输出目录
        video_output_dir = output_dir
        if not video_output_dir:
            video_dir = os.path.dirname(video_path)
            video_output_dir = os.path.join(video_dir, "frames", video_name)
            os.makedirs(video_output_dir, exist_ok=True)
        
        # 提取帧（按帧ID和时间戳）
        extracted = 0
        if frame_ids:
            extracted += extract_frames(video_path, frame_ids, video_output_dir, video_name)
        if timestamps:
            extracted += extract_frames_by_timestamp(video_path, timestamps, video_output_dir, video_name)
        
        if extracted > 0:
            videos_processed += 1
            frames_extracted += extracted
    
    return videos_processed, frames_extracted

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从视频中提取特定帧并保存为PNG图像')
    
    # 添加参数
    parser.add_argument('--video', '-v', type=str, help='单个视频文件路径')
    parser.add_argument('--dir', '-d', type=str, help='视频文件目录')
    parser.add_argument('--recursive', '-r', action='store_true', help='递归搜索子目录中的视频')
    parser.add_argument('--output', '-o', type=str, help='输出目录')
    parser.add_argument('--frames', '-f', type=str, default='21,37,62,87', 
                        help='要提取的帧ID，用逗号分隔，例如 "21,37,62,87"')
    parser.add_argument('--timestamps', '-t', type=str,
                        help='要提取的时间戳，用逗号分隔，例如 "00:01:30,00:02:45"')
    parser.add_argument('--config', '-c', type=str, default='configs/default_config.yaml', 
                        help='配置文件路径')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    # 解析帧ID
    frame_ids = None
    if args.frames:
        try:
            frame_ids = [int(frame_id.strip()) for frame_id in args.frames.split(',')]
            if not frame_ids:
                print("警告: 未提供有效的帧ID")
        except ValueError:
            print(f"错误: 无效的帧ID格式: {args.frames}")
            return
    
    # 解析时间戳
    timestamps = None
    if args.timestamps:
        timestamps = [ts.strip() for ts in args.timestamps.split(',')]
        if not timestamps:
            print("警告: 未提供有效的时间戳")
    
    # 确保至少有一种提取方式
    if not frame_ids and not timestamps:
        print("错误: 必须提供帧ID或时间戳")
        return
    
    # 加载配置（用于将来可能的扩展功能）
    config = load_config(args.config)
    
    # 检查输入参数
    if args.video:
        # 处理单个视频
        if not os.path.exists(args.video):
            print(f"错误: 视频文件不存在: {args.video}")
            return
        
        extracted = 0
        if frame_ids:
            print(f"从视频中提取帧ID {frame_ids}: {args.video}")
            extracted += extract_frames(args.video, frame_ids, args.output)
        
        if timestamps:
            print(f"从视频中提取时间戳 {timestamps}: {args.video}")
            extracted += extract_frames_by_timestamp(args.video, timestamps, args.output)
        
        print(f"成功提取 {extracted} 帧")
    
    elif args.dir:
        # 处理目录中的所有视频
        if not os.path.isdir(args.dir):
            print(f"错误: 目录不存在: {args.dir}")
            return
        
        print(f"从目录 {args.dir} 中的视频提取帧")
        if frame_ids:
            print(f"帧ID: {frame_ids}")
        if timestamps:
            print(f"时间戳: {timestamps}")
        
        videos_processed, frames_extracted = extract_from_all_videos(
            args.dir, frame_ids, timestamps, args.output, args.recursive)
        
        print(f"\n总结: 处理了 {videos_processed} 个视频，提取了 {frames_extracted} 帧")
    
    else:
        print("错误: 必须指定视频文件 (--video) 或视频目录 (--dir)")

if __name__ == "__main__":
    main() 