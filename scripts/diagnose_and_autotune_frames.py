#!/usr/bin/env python3
"""
诊断并自动微调指定帧的球检测：
- 输出四宫格：原图(ROI裁剪)、HSV掩膜、轮廓叠加、最终检测
- 自动尝试一组更宽松的参数，选择质量最高的检测作为建议

保存位置：frames_70_90_analysis/diagnose_frame_XXX.jpg
"""

import os
import cv2
import yaml
import numpy as np
from typing import Dict, List, Tuple

from roi_manager import ROIManager
from ball_tracker import BallTracker


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compute_hsv_mask(roi_frame: np.ndarray, cfg: Dict, process_w: int = 640, process_h: int = 360) -> np.ndarray:
    """按当前配置计算清理后的HSV掩膜（缩放回ROI尺寸便于可视化）。"""
    resized = cv2.resize(roi_frame, (process_w, process_h))
    resized = cv2.GaussianBlur(resized, (3, 3), 0.5)
    resized = cv2.medianBlur(resized, 3)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    lh = int(cfg.get('hsv_lower_hue', 20))
    uh = int(cfg.get('hsv_upper_hue', 70))
    ls = int(cfg.get('hsv_lower_sat', 55))
    us = int(cfg.get('hsv_upper_sat', 255))
    lv = int(cfg.get('hsv_lower_val', 60))
    uv = int(cfg.get('hsv_upper_val', 255))
    lower = np.array([lh, ls, lv], dtype=np.uint8)
    upper = np.array([uh, us, uv], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # 形态学与面积清理（与BallTracker对齐）
    kernel_tiny = np.ones((1, 1), np.uint8)
    kernel_small = np.ones((2, 2), np.uint8)
    kernel_medium = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_tiny, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # 面积阈值按缩放比换算
    scale_factor = process_w / roi_frame.shape[1]
    min_area_cfg = float(cfg.get('min_ball_area', 120))
    max_area_cfg = float(cfg.get('max_ball_area', 2000))
    min_area = int(max(1, min_area_cfg * (scale_factor ** 2)))
    max_area = int(max_area_cfg * (scale_factor ** 2))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    for c in contours:
        a = cv2.contourArea(c)
        if min_area <= a <= max_area:
            cv2.drawContours(clean_mask, [c], -1, 255, -1)
    mask = cv2.dilate(clean_mask, kernel_medium, iterations=1)

    # 缩放回ROI尺寸以便显示
    mask_vis = cv2.resize(mask, (roi_frame.shape[1], roi_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_vis


def draw_contours_overlay(roi_frame: np.ndarray, mask_vis: np.ndarray) -> np.ndarray:
    overlay = roi_frame.copy()
    contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def tile_four(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    def to_bgr(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    a, b, c, d = map(to_bgr, [a, b, c, d])
    h = max(a.shape[0], b.shape[0], c.shape[0], d.shape[0])
    w = max(a.shape[1], b.shape[1], c.shape[1], d.shape[1])
    def pad(img):
        dy = max(0, h - img.shape[0])
        dx = max(0, w - img.shape[1])
        return cv2.copyMakeBorder(img, 0, dy, 0, dx, cv2.BORDER_CONSTANT)
    a, b, c, d = map(pad, [a, b, c, d])
    top = cv2.hconcat([a, b])
    bottom = cv2.hconcat([c, d])
    grid = cv2.vconcat([top, bottom])
    return grid


def diagnose_frames(config_path: str, frames: List[int]) -> None:
    cfg = load_config(config_path)
    out_dir = 'frames_70_90_analysis'
    ensure_dir(out_dir)

    # 禁用模拟回退
    cfg_local = dict(cfg)
    cfg_local['allow_simulation_fallback'] = False

    # 初始化
    cap = cv2.VideoCapture(cfg_local['video_input_path'])
    roi = ROIManager(cfg_local)
    if cfg_local.get('roi_settings', {}).get('auto_load_config', True):
        roi_cfg = cfg_local.get('roi_settings', {}).get('roi_config_path', 'configs/roi_config.yaml')
        if os.path.exists(roi_cfg):
            roi.load_roi_config(roi_cfg)
    bt = BallTracker(cfg_local.get('tracknet_model_path', None), cfg_local, roi)

    for fidx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            print(f'❌ 读取帧 {fidx} 失败')
            continue

        detection_frame = frame
        offset = (0, 0)
        if roi.is_roi_set:
            bbox = roi.get_roi_bounding_box()
            if bbox:
                x1, y1, x2, y2 = bbox
                detection_frame = frame[y1:y2, x1:x2]
                offset = (x1, y1)

        # 生成四宫格各部分
        mask = compute_hsv_mask(detection_frame, cfg_local)
        contours_overlay = draw_contours_overlay(detection_frame, mask)
        dets = bt.predict_ball(detection_frame)
        det_vis = frame.copy()
        if offset != (0, 0) and dets:
            dets = roi.adjust_detection_coordinates(dets, offset, 'ball')
        for (x, y) in dets or []:
            cv2.circle(det_vis, (int(x), int(y)), 12, (0, 255, 0), 3)
            cv2.circle(det_vis, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(det_vis, f'F{fidx}', (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        grid = tile_four(detection_frame, mask, contours_overlay, det_vis)
        out_path = os.path.join(out_dir, f'diagnose_frame_{fidx:03d}.jpg')
        cv2.imwrite(out_path, grid)
        print(f'✅ 诊断图已保存: {out_path}  检测数: {len(dets) if dets else 0}')

    cap.release()


def autotune_for_frames(config_path: str, frames: List[int]) -> None:
    """尝试一组更宽松的参数组合，挑选质量最高的检测并给出建议。"""
    base = load_config(config_path)
    base['allow_simulation_fallback'] = False
    candidates = [
        {"hsv_lower_sat": 50, "hsv_lower_val": 55, "hough_param2": 7, "min_ball_area": 100},
        {"hsv_lower_sat": 50, "hsv_lower_val": 50, "hough_param2": 6, "min_ball_area": 90},
        {"hsv_lower_sat": 45, "hsv_lower_val": 55, "hough_param2": 7, "min_ball_area": 90},
    ]

    cap = cv2.VideoCapture(base['video_input_path'])
    roi = ROIManager(base)
    if base.get('roi_settings', {}).get('auto_load_config', True):
        roi_cfg = base.get('roi_settings', {}).get('roi_config_path', 'configs/roi_config.yaml')
        if os.path.exists(roi_cfg):
            roi.load_roi_config(roi_cfg)

    for fidx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            print(f'❌ 读取帧 {fidx} 失败')
            continue

        best = None  # (quality, params, (x,y))
        for p in candidates:
            cfg_try = dict(base)
            cfg_try.update(p)
            bt = BallTracker(cfg_try.get('tracknet_model_path', None), cfg_try, roi)
            det_frame = frame
            offset = (0, 0)
            if roi.is_roi_set:
                bbox = roi.get_roi_bounding_box()
                if bbox:
                    x1, y1, x2, y2 = bbox
                    det_frame = frame[y1:y2, x1:x2]
                    offset = (x1, y1)
            dets = bt.predict_ball(det_frame)
            if offset != (0, 0) and dets:
                dets = roi.adjust_detection_coordinates(dets, offset, 'ball')
            if dets:
                x, y = int(dets[0][0]), int(dets[0][1])
                q = bt._evaluate_ball_quality(frame, x, y, max(base.get('min_ball_radius', 18), 6))
                if (best is None) or (q > best[0]):
                    best = (q, p, (x, y))

        if best:
            q, params, (x, y) = best
            print(f"🎯 帧{fidx}: 推荐参数 {params}, 质量={q:.2f}, 位置=({x},{y})")
        else:
            print(f"⚠️ 帧{fidx}: 候选参数均未检测到球")

    cap.release()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='诊断并自动微调指定帧的球检测')
    parser.add_argument('-c', '--config', default='configs/comprehensive_tennis_config.yaml')
    parser.add_argument('--frames', nargs='+', type=int, required=True)
    parser.add_argument('--autotune', action='store_true')
    args = parser.parse_args()
    diagnose_frames(args.config, args.frames)
    if args.autotune:
        autotune_for_frames(args.config, args.frames)


