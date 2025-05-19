#!/bin/bash
# extract_timestamps.sh - 提取特定时间戳帧的脚本

# 创建输出目录
OUTPUT_DIR="./extracted_timestamps"
mkdir -p "$OUTPUT_DIR"

# 提取所有视频中指定时间戳的帧
echo "正在从所有视频中提取指定时间戳的帧..."
python frame_extractor.py --dir player-video --recursive \
    --timestamps "16:04:13,16:04:14,16:04:15,16:04:16" \
    --output "$OUTPUT_DIR"

echo "所有帧已提取到: $OUTPUT_DIR"
echo "完成!" 