#!/bin/bash

# Project Cleanup Script for tennis_analyzer
# Usage: ./cleanup.sh [--dry-run]

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "--- DRY RUN MODE: No files will be deleted ---"
fi

# Directories to remove recursively
DIRS=(
    "__pycache__"
    "output"
    "test_output"
    "clips"
    "extracted_frames"
    "debug_frames"
    "false_detection_analysis"
    "frames_70_90_analysis"
    "debug_frames_compare_210_220"
    "debug_frames_compare_210_220_tuned"
)

# File patterns to remove
FILE_PATTERNS=(
    "*.log"
    "debug_frame_*.jpg"
    "frame_*_analysis.jpg"
    "roi_boundary_visualization.jpg"
    "*_test_result.jpg"
    "roi_preprocessing_test.jpg"
    "yolo26_pose_demo_*.jpg"
    "adjusted_params_analysis.csv"
    "comprehensive_predict_analysis.csv"
    "*.pyc"
    "filtering_test_result.jpg"
    "mask_zone_*-test_result.jpg"
    "logs_frame_*.txt"
)

echo "Cleaning up project artifacts..."

# Clean Directories
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY-RUN] Would remove directory: $dir"
        else
            echo "Removing directory: $dir"
            rm -rf "$dir"
        fi
    fi
done

# Clean Files
for pattern in "${FILE_PATTERNS[@]}"; do
    # Using find to handle patterns safely
    find . -maxdepth 1 -name "$pattern" -type f | while read -r file; do
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY-RUN] Would remove file: $file"
        else
            echo "Removing file: $file"
            rm "$file"
        fi
    done
done

# Specifically also handle nested __pycache__
if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] Would remove all nested __pycache__ directories"
else
    echo "Removing all nested __pycache__ directories..."
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
fi

echo "Cleanup complete."
