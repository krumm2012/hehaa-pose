#!/usr/bin/env python3
"""
详细分析指定帧检测到的球的具体参数：圆度、大小、质量等
用法：python analyze_frame_183_detailed.py --frame 183
"""

import cv2
import yaml
from ball_tracker import BallTracker
from roi_manager import ROIManager
import numpy as np
import argparse

def analyze_frame(frame_id: int):
    print("🔍 详细分析球检测参数")
    print("=" * 60)

    # 加载配置
    config_path = 'configs/comprehensive_tennis_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 初始化组件
    roi_manager = ROIManager(config)
    roi_manager.load_roi_config('configs/roi_config.yaml')

    ball_tracker = BallTracker(config.get('tracknet_model_path', None), config, roi_manager)

    # 打开视频
    video_path = config['video_input_path']
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return

    # 跳转到目标帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if not ret:
        print(f"❌ 无法读取第{frame_id}帧")
        return

    print(f"📹 成功读取第{frame_id}帧，尺寸: {frame.shape}")

    # ROI处理
    roi_cropped_frame = frame
    roi_offset = (0, 0)
    if roi_manager.is_roi_set:
        roi_bbox = roi_manager.get_roi_bounding_box()
        if roi_bbox:
            x1, y1, x2, y2 = roi_bbox
            roi_offset = (x1, y1)
            roi_cropped_frame = frame[y1:y2, x1:x2]
            print(f"📘 ROI边界框: ({x1}, {y1}) - ({x2}, {y2})")
            print(f"📐 ROI偏移量: {roi_offset}")
            print(f"📘 ROI裁剪帧尺寸: {roi_cropped_frame.shape}")

    # 获取HSV检测结果
    try:
        hsv_detections = ball_tracker._detect_with_hsv(roi_cropped_frame)
        print(f"\n📊 HSV检测结果: {len(hsv_detections)} 个候选球")
    except Exception as e:
        print(f"❌ HSV检测失败: {e}")
        return

    # 详细分析每个候选球
    print(f"\n🎾 === 详细球检测参数分析（帧{frame_id}）===")

    for i, detection in enumerate(hsv_detections):
        if len(detection) >= 2:
            x, y = detection[:2]

            # 转换坐标
            original_x = x + roi_offset[0]
            original_y = y + roi_offset[1]

            print(f"\n🎯 候选球 {i+1}:")
            print(f"  📍 ROI裁剪帧坐标: ({x:.0f}, {y:.0f})")
            print(f"  📍 原始帧坐标: ({original_x:.0f}, {original_y:.0f})")

            # 检查ROI
            in_roi = roi_manager.is_point_in_roi((original_x, original_y))
            print(f"  🟩 ROI内: {'✅' if in_roi else '❌'}")

            # 计算边缘距离
            edge_distance = min(x, y, roi_cropped_frame.shape[1] - x, roi_cropped_frame.shape[0] - y)
            print(f"  📏 边缘距离: {edge_distance:.0f}px")

            # 质量/圆度（用固定半径估计）
            est_radius = 15
            try:
                quality_score = ball_tracker._evaluate_ball_quality(roi_cropped_frame, int(x), int(y), est_radius)
            except Exception:
                quality_score = 0.0
            try:
                circularity_score = ball_tracker._check_ball_circularity(roi_cropped_frame, int(x), int(y), est_radius)
            except Exception:
                circularity_score = 0.0

            print(f"  🎯 质量分数: {quality_score:.3f}")
            print(f"  🔵 圆度分数: {circularity_score:.3f}")
            print(f"  📐 估算半径: {est_radius}px")

    # 测试完整的predict_ball方法
    final_detections = ball_tracker.predict_ball(roi_cropped_frame)
    print(f"\n🏁 predict_ball最终输出: {len(final_detections)} 个球")

    # 当前阈值展示
    print(f"\n⚙️ === 当前阈值 ===")
    print(f"  尺寸: 半径{config.get('min_ball_radius', 5)}-{config.get('max_ball_radius', 45)}px, 面积≥{config.get('min_ball_area', 0)}")
    print(f"  质量阈值: {config.get('noise_filter_quality_threshold', 0.3)}  圆度阈值: {config.get('noise_filter_circularity_threshold', 0.5)}")
    print(f"  静态球: 移动≥{config.get('static_ball_movement_threshold_px', 4)}px, 帧数≥{config.get('static_ball_frames_threshold', 10)}")

    # HSV覆盖率
    hsv_frame = cv2.cvtColor(roi_cropped_frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([config.get('hsv_lower_hue', 20), config.get('hsv_lower_sat', 40), config.get('hsv_lower_val', 40)])
    upper_hsv = np.array([config.get('hsv_upper_hue', 70), config.get('hsv_upper_sat', 255), config.get('hsv_upper_val', 255)])
    hsv_mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    hsv_pixel_count = cv2.countNonZero(hsv_mask)
    total_pixels = hsv_frame.shape[0] * hsv_frame.shape[1]
    hsv_coverage = (hsv_pixel_count / total_pixels) * 100

    print(f"\n🎨 HSV覆盖率: {hsv_coverage:.2f}% (匹配像素 {hsv_pixel_count}/{total_pixels})")

    # 保存输出
    debug_image_path = f"debug_frame_{frame_id}_detailed_analysis.jpg"
    hsv_mask_path = f"debug_frame_{frame_id}_detailed_hsv_mask.jpg"

    debug = roi_cropped_frame.copy()
    for i, detection in enumerate(hsv_detections):
        if len(detection) >= 2:
            x, y = int(detection[0]), int(detection[1])
            cv2.circle(debug, (x, y), 15, (0, 255, 0), 2)
            cv2.putText(debug, f"Ball{i+1}", (x+18, y-18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imwrite(debug_image_path, debug)
    cv2.imwrite(hsv_mask_path, hsv_mask)
    print(f"💾 已保存: {debug_image_path}, {hsv_mask_path}")

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', '-f', type=int, default=183, help='要分析的帧号，默认183')
    args = parser.parse_args()
    analyze_frame(args.frame)
