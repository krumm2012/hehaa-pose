import os
import cv2
import yaml
from typing import Dict, Tuple

from roi_manager import ROIManager
from ball_tracker import BallTracker


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def main(config_path: str = 'configs/comprehensive_tennis_config.yaml',
         start_frame: int = 70,
         end_frame: int = 90) -> None:
    config = load_config(config_path)

    # Disable any interactive ROI selection for analysis
    roi_settings = config.get('roi_settings', {})
    roi_settings['interactive_selection'] = False
    roi_settings['auto_load_config'] = True
    config['roi_settings'] = roi_settings

    # Analysis preferences: keep all balls (not only largest), use static-filter-only
    config.setdefault('ball_detection_strategy', {})
    config['ball_detection_strategy']['prefer_largest_ball'] = False
    config['ball_tracking_enabled'] = False
    config['static_filter_only'] = True

    # Make debug frames save every frame into analysis folder
    config.setdefault('ball_detection_debug', {})
    config['ball_detection_debug']['save_debug_frames'] = True
    config['ball_detection_debug']['debug_frame_interval'] = 1
    config['ball_detection_debug']['debug_frames_path'] = 'frames_70_90_analysis/'
    # Keep HSV logs if desired
    config['ball_detection_debug']['log_hsv_detection'] = True

    video_path = config['video_input_path']
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    # Init ROI
    roi_manager = ROIManager(config)
    if roi_settings.get('auto_load_config', True):
        roi_cfg_path = roi_settings.get('roi_config_path', 'configs/roi_config.yaml')
        if os.path.exists(roi_cfg_path):
            roi_manager.load_roi_config(roi_cfg_path)

    # Init BallTracker
    ball_tracker = BallTracker(config.get('tracknet_model_path', None), config, roi_manager)

    # Prepare output dir
    out_dir = 'frames_70_90_analysis'
    ensure_dir(out_dir)

    # Jump to start_frame efficiently (read and discard)
    current_idx = 0
    while current_idx < start_frame:
        ok, _ = cap.read()
        if not ok:
            print(f"提前结束：视频不足 {start_frame} 帧")
            cap.release()
            return
        current_idx += 1

    # Iterate frames in range
    for frame_idx in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            print(f"在第 {frame_idx} 帧读帧失败，提前结束")
            break

        display = frame.copy()
        roi_offset: Tuple[int, int] = (0, 0)

        # ROI crop for detection
        detection_frame = frame
        if roi_manager.is_roi_set:
            bbox = roi_manager.get_roi_bounding_box()
            if bbox:
                x1, y1, x2, y2 = bbox
                detection_frame = frame[y1:y2, x1:x2]
                roi_offset = (x1, y1)

        detections = ball_tracker.predict_ball(detection_frame)

        # Map coords back to original frame if ROI used
        if roi_offset != (0, 0) and detections:
            detections = roi_manager.adjust_detection_coordinates(detections, roi_offset, 'ball')

        # Draw all detections on output image
        for (x, y) in detections:
            cv2.circle(display, (int(x), int(y)), 8, (0, 255, 0), 2)
            cv2.circle(display, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Save images
        cv2.imwrite(os.path.join(out_dir, f'frame_{frame_idx:03d}_original.jpg'), frame)
        cv2.imwrite(os.path.join(out_dir, f'frame_{frame_idx:03d}_debug.jpg'), display)

        # Print per-frame summary
        print(f"帧 {frame_idx}: 检测到 {len(detections)} 个球 -> {[(int(x), int(y)) for (x,y) in detections]}")

    cap.release()
    print("分析完成。结果图像已保存到 frames_70_90_analysis/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='分析70-90帧所有球的检测')
    parser.add_argument('-c', '--config', default='configs/comprehensive_tennis_config.yaml', help='配置文件路径')
    parser.add_argument('--start', type=int, default=70, help='起始帧（含）')
    parser.add_argument('--end', type=int, default=90, help='结束帧（含）')
    args = parser.parse_args()
    main(args.config, args.start, args.end)


