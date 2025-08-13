#!/usr/bin/env python3
"""
导出全视频指标可视化图表，并自动截图关键帧对比。

输出：
- data/analysis_results/metrics.csv
- data/analysis_results/plots/detections_per_frame.png
- data/analysis_results/plots/ball_xy_over_frames.png
- data/analysis_results/keyframes/frame_XXX.jpg
"""

import os
import csv
import cv2
import yaml
import math
import numpy as np
from typing import Dict, Tuple, List
import matplotlib

# 使用无界面后端，防止GUI依赖
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roi_manager import ROIManager
from ball_tracker import BallTracker


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def analyze_video(config_path: str,
                  keyframe_targets: List[int] = None,
                  sample_keyframes: int = 12) -> None:
    config = load_config(config_path)

    # 目录准备
    out_root = os.path.join("data", "analysis_results")
    plots_dir = os.path.join(out_root, "plots")
    keyframes_dir = os.path.join(out_root, "keyframes")
    ensure_dir(out_root)
    ensure_dir(plots_dir)
    ensure_dir(keyframes_dir)

    # ROI设置（仅加载，不交互）
    roi_settings = config.get("roi_settings", {})
    roi_settings["interactive_selection"] = False
    roi_settings["auto_load_config"] = True
    config["roi_settings"] = roi_settings

    # 检测偏好：不强制仅最大球，以便统计数量；但保存截图时优先画第一个球
    config.setdefault("ball_detection_strategy", {})
    config["ball_detection_strategy"]["prefer_largest_ball"] = False

    cap = cv2.VideoCapture(config["video_input_path"])
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {config['video_input_path']}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    roi_manager = ROIManager(config)
    if roi_settings.get("auto_load_config", True):
        roi_cfg_path = roi_settings.get("roi_config_path", "configs/roi_config.yaml")
        if os.path.exists(roi_cfg_path):
            roi_manager.load_roi_config(roi_cfg_path)

    # 禁用模拟回退，避免每帧都出现“虚拟球”导致误统计
    config_mod = dict(config)
    config_mod["allow_simulation_fallback"] = False
    tracker = BallTracker(config_mod.get("tracknet_model_path", None), config_mod, roi_manager)

    metrics_rows = [("frame", "num_balls", "ball_x", "ball_y")]
    first_ball_history: List[Tuple[int, int]] = []

    # 逐帧统计
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detection_frame = frame
        roi_offset = (0, 0)
        if roi_manager.is_roi_set:
            bbox = roi_manager.get_roi_bounding_box()
            if bbox:
                x1, y1, x2, y2 = bbox
                detection_frame = frame[y1:y2, x1:x2]
                roi_offset = (x1, y1)

        detections = tracker.predict_ball(detection_frame)
        if roi_offset != (0, 0) and detections:
            detections = roi_manager.adjust_detection_coordinates(detections, roi_offset, "ball")

        ball_x = ball_y = ""
        if detections:
            ball_x, ball_y = int(detections[0][0]), int(detections[0][1])
            first_ball_history.append((ball_x, ball_y))
        else:
            first_ball_history.append((np.nan, np.nan))

        metrics_rows.append((frame_idx, len(detections), ball_x, ball_y))

        # 关键帧保存：如果在目标帧或后续筛选中入选，则保存覆盖
        frame_idx += 1

    cap.release()

    # 写出CSV
    csv_path = os.path.join(out_root, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(metrics_rows)
    print(f"✅ 指标CSV已保存: {csv_path}")

    # 绘图
    data = np.array(metrics_rows[1:], dtype=object)
    frames = data[:, 0].astype(int)
    num_balls = data[:, 1].astype(int)
    ball_x = np.array([np.nan if v == "" else float(v) for v in data[:, 2]])
    ball_y = np.array([np.nan if v == "" else float(v) for v in data[:, 3]])

    # 1) 每帧检测数量
    plt.figure(figsize=(10, 4))
    plt.plot(frames, num_balls, label="Detections per Frame")
    plt.xlabel("Frame")
    plt.ylabel("#Balls")
    plt.title("Ball Detections per Frame")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot1 = os.path.join(plots_dir, "detections_per_frame.png")
    plt.tight_layout()
    plt.savefig(plot1, dpi=160)
    plt.close()
    print(f"✅ 图表已保存: {plot1}")

    # 2) 球中心坐标随帧变化
    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(frames, ball_x, color="tab:blue")
    ax1.set_ylabel("ball_x")
    ax1.grid(True, alpha=0.3)
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(frames, ball_y, color="tab:orange")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("ball_y")
    ax2.grid(True, alpha=0.3)
    plot2 = os.path.join(plots_dir, "ball_xy_over_frames.png")
    plt.tight_layout()
    plt.savefig(plot2, dpi=160)
    plt.close()
    print(f"✅ 图表已保存: {plot2}")

    # 关键帧选择：
    candidates = []
    for i, n in zip(frames, num_balls):
        if n >= 1:
            candidates.append(int(i))
    # 合并固定目标帧
    keyframe_set = set()
    if keyframe_targets:
        keyframe_set.update([k for k in keyframe_targets if 0 <= k < len(frames)])
    if candidates:
        step = max(1, len(candidates) // max(1, sample_keyframes))
        subsampled = candidates[::step][:sample_keyframes]
        keyframe_set.update(subsampled)
    keyframes = sorted(list(keyframe_set))

    # 保存关键帧截图
    if keyframes:
        cap2 = cv2.VideoCapture(config["video_input_path"])
        for idx in keyframes:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap2.read()
            if not ok:
                continue
            # 画圆点（如果CSV里有坐标）
            row = metrics_rows[idx + 1]  # 偏移1行为表头
            if row[2] != "" and row[3] != "":
                cx, cy = int(row[2]), int(row[3])
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"F{idx}", (cx + 12, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            out_path = os.path.join(keyframes_dir, f"frame_{idx:03d}.jpg")
            cv2.imwrite(out_path, frame)
        cap2.release()
        print(f"✅ 关键帧已保存到: {keyframes_dir} (共{len(keyframes)}张)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="导出指标图表与关键帧截图")
    parser.add_argument("--config", "-c", default="configs/comprehensive_tennis_config.yaml", help="配置文件路径")
    parser.add_argument("--start", type=int, default=70, help="可选：关键帧目标起始帧")
    parser.add_argument("--end", type=int, default=90, help="可选：关键帧目标结束帧")
    args = parser.parse_args()
    targets = list(range(args.start, args.end + 1)) if args.start >= 0 and args.end >= args.start else []
    analyze_video(args.config, targets)


