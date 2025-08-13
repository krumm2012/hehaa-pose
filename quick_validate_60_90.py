#!/usr/bin/env python3
"""
快速验证60–90帧：统计每帧HSV候选与最终检测数量
"""
import cv2, yaml, numpy as np
from ball_tracker import BallTracker
from roi_manager import ROIManager

START, END = 60, 90

def main():
    with open('configs/comprehensive_tennis_config.yaml','r',encoding='utf-8') as f:
        config = yaml.safe_load(f)
    roi = ROIManager(config)
    roi.load_roi_config('configs/roi_config.yaml')
    bt = BallTracker(config.get('tracknet_model_path', None), config, roi)

    cap = cv2.VideoCapture(config['video_input_path'])
    if not cap.isOpened():
        print('❌ 无法打开视频'); return

    print('帧, HSV候选数, 最终球数')
    for frame_id in range(START, END+1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret: break
        # ROI裁剪
        det = frame
        if roi.is_roi_set:
            bbox = roi.get_roi_bounding_box()
            if bbox:
                x1,y1,x2,y2=bbox
                det = frame[y1:y2, x1:x2]
        try:
            hsv_cands = bt._detect_with_hsv(det)
        except Exception:
            hsv_cands = []
        final = bt.predict_ball(det)
        print(f"{frame_id}, {len(hsv_cands)}, {len(final)}")
    cap.release()

if __name__ == '__main__':
    main()
