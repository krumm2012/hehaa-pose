#!/usr/bin/env python3
"""
从指定帧采样网球的HSV颜色值，并给出建议的HSV范围。

输出：
- data/analysis_results/ball_hsv_samples.csv
"""

import os
import csv
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


def wide_hsv_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 非常宽的黄绿范围，尽量覆盖
    lower = np.array([15, 35, 45], dtype=np.uint8)
    upper = np.array([80, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def detect_or_guess_center(frame: np.ndarray, roi_mgr: ROIManager, cfg: Dict) -> Tuple[int, int]:
    cfg_try = dict(cfg)
    cfg_try['allow_simulation_fallback'] = False
    bt = BallTracker(cfg_try.get('tracknet_model_path', None), cfg_try, roi_mgr)
    det = frame
    offset = (0, 0)
    if roi_mgr.is_roi_set:
        bbox = roi_mgr.get_roi_bounding_box()
        if bbox:
            x1, y1, x2, y2 = bbox
            det = frame[y1:y2, x1:x2]
            offset = (x1, y1)
    balls = bt.predict_ball(det)
    if offset != (0, 0) and balls:
        balls = roi_mgr.adjust_detection_coordinates(balls, offset, 'ball')
    if balls:
        x, y = int(balls[0][0]), int(balls[0][1])
        return x, y

    # 回退：宽掩膜+最大连通域
    det2 = det if offset == (0, 0) else frame[offset[1]:offset[1]+det.shape[0], offset[0]:offset[0]+det.shape[1]]
    mask = wide_hsv_mask(det2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(c)
        cx, cy = int(cx), int(cy)
        if offset != (0, 0):
            cx += offset[0]
            cy += offset[1]
        return cx, cy
    return -1, -1


def sample_hsv_around(frame_bgr: np.ndarray, cx: int, cy: int, radius: int = 12) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if cx < 0 or cy < 0:
        return np.empty((0, 3), dtype=np.uint8)
    x1, x2 = max(0, cx - radius), min(w, cx + radius + 1)
    y1, y2 = max(0, cy - radius), min(h, cy + radius + 1)
    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # 圆形mask 仅取圆内像素
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.circle(mask, (min(radius, mask.shape[1]-1), min(radius, mask.shape[0]-1)), radius, 1, -1)
    pixels = hsv[mask.astype(bool)]
    return pixels


def analyze_frames(config_path: str, frames: List[int]) -> None:
    cfg = load_config(config_path)
    out_dir = os.path.join('data', 'analysis_results')
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, 'ball_hsv_samples.csv')

    cap = cv2.VideoCapture(cfg['video_input_path'])
    roi = ROIManager(cfg)
    if cfg.get('roi_settings', {}).get('auto_load_config', True):
        roi_cfg = cfg.get('roi_settings', {}).get('roi_config_path', 'configs/roi_config.yaml')
        if os.path.exists(roi_cfg):
            roi.load_roi_config(roi_cfg)

    rows = [("frame", "n", "h_mean", "h_p05", "h_p95", "s_mean", "s_p05", "s_p95", "v_mean", "v_p05", "v_p95")]

    all_hsv = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        cx, cy = detect_or_guess_center(frame, roi, cfg)
        pixels = sample_hsv_around(frame, cx, cy, radius=12)
        if pixels.size == 0:
            rows.append((f, 0, '', '', '', '', '', '', '', '', ''))
            continue
        h, s, v = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        all_hsv.append(pixels)
        def p(a, q):
            return float(np.percentile(a, q))
        rows.append((
            f, len(pixels),
            float(np.mean(h)), p(h, 5), p(h, 95),
            float(np.mean(s)), p(s, 5), p(s, 95),
            float(np.mean(v)), p(v, 5), p(v, 95)
        ))

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    print(f"✅ HSV样本CSV: {csv_path}")

    if all_hsv:
        all_hsv = np.vstack(all_hsv)
        h, s, v = all_hsv[:, 0], all_hsv[:, 1], all_hsv[:, 2]
        rec = {
            'hsv_lower_hue': int(max(0, np.percentile(h, 3) - 2)),
            'hsv_upper_hue': int(min(179, np.percentile(h, 97) + 2)),
            'hsv_lower_sat': int(max(0, np.percentile(s, 3) - 5)),
            'hsv_lower_val': int(max(0, np.percentile(v, 3) - 5)),
        }
        print(f"🎯 建议HSV下界: H>={rec['hsv_lower_hue']}, S>={rec['hsv_lower_sat']}, V>={rec['hsv_lower_val']}")
        print(f"🎯 建议HSV上界: H<={rec['hsv_upper_hue']}")
    cap.release()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='采集网球HSV样本并给出建议范围')
    parser.add_argument('-c', '--config', default='configs/comprehensive_tennis_config.yaml')
    parser.add_argument('--frames', nargs='+', type=int, required=True)
    args = parser.parse_args()
    analyze_frames(args.config, args.frames)


