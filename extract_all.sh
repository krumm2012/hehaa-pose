#!/bin/bash
# extract_all.sh - 提取所有指定帧和时间戳的脚本

# 创建输出目录
OUTPUT_DIR="./extracted_all"
mkdir -p "$OUTPUT_DIR"

# 提取所有视频中指定帧ID的帧
echo "=== 正在提取指定帧ID的帧... ==="
python frame_extractor.py --dir player-video --recursive \
    --frames 21,37,62,87 \
    --output "$OUTPUT_DIR/by_frame_id"

# 提取所有视频中指定时间戳的帧
echo ""
echo "=== 正在提取指定时间戳的帧... ==="
python frame_extractor.py --dir player-video --recursive \
    --timestamps "16:04:13,16:04:14,16:04:15,16:04:16" \
    --output "$OUTPUT_DIR/by_timestamp"

echo ""
echo "所有帧已提取到: $OUTPUT_DIR"
echo "  - 按帧ID提取: $OUTPUT_DIR/by_frame_id"
echo "  - 按时间戳提取: $OUTPUT_DIR/by_timestamp"
echo "完成!" 