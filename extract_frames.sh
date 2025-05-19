#!/bin/bash
# extract_frames.sh - 快速提取指定帧的脚本

# 创建输出目录
OUTPUT_DIR="./extracted_frames"
mkdir -p "$OUTPUT_DIR"

# 提取所有视频中指定的帧 (ID: 21, 37, 62, 87)
echo "正在从所有视频中提取指定帧..."
python frame_extractor.py --dir data --recursive --frames 21,37,62,87 --output "$OUTPUT_DIR"

# 也可以提取指定时间戳处的帧（取消下面的注释使用）
# echo "正在从所有视频中提取指定时间戳的帧..."
# python frame_extractor.py --dir player-video --recursive --timestamps "00:00:14,00:00:15,00:00:16" --output "$OUTPUT_DIR"

echo "所有帧已提取到: $OUTPUT_DIR"
echo "完成!" 