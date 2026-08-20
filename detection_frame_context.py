#!/usr/bin/env python3
"""
检测帧上下文：
- ROI 预处理（裁剪与检测帧选择）
- 检测结果坐标回填（ROI 坐标 -> 原图坐标）
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class DetectionFrameContext:
    frame_num: int
    frame: Any
    detection_frame: Any
    pose_detection_frame: Any
    roi_offset: Tuple[int, int]
    pose_use_roi: bool
    roi_manager: Any
    config: Dict

    @classmethod
    def build(cls, frame_num: int, frame: Any, roi_manager: Any, config: Dict) -> "DetectionFrameContext":
        roi_cropped_frame = frame
        roi_offset = (0, 0)
        log_frame_context = bool(
            config.get("roi_settings", {})
            .get("logging", {})
            .get("log_frame_context", False)
        )

        if roi_manager.is_roi_set:
            roi_bbox = roi_manager.get_roi_bounding_box()
            if roi_bbox:
                x1, y1, x2, y2 = roi_bbox
                margin = int(config.get("roi_settings", {}).get("crop_margin", 12))
                x1_expanded = max(0, x1 - margin)
                y1_expanded = max(0, y1 - margin)
                x2_expanded = min(frame.shape[1], x2 + margin)
                y2_expanded = min(frame.shape[0], y2 + margin)

                roi_offset = (x1_expanded, y1_expanded)
                roi_cropped_frame = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
                if log_frame_context and frame_num % 30 == 0:
                    print(f"🎯 [帧{frame_num}] ROI区域提取(含边距{margin}px): {roi_cropped_frame.shape} at offset {roi_offset}")

        detection_frame = roi_cropped_frame if roi_cropped_frame is not None else frame

        pose_use_roi = config.get("pose_estimation_debug", {}).get("use_roi_detection", False)
        pose_detection_frame = detection_frame if pose_use_roi and detection_frame is not None else frame
        if log_frame_context and frame_num % 30 == 0:
            if pose_use_roi:
                print(f"🤖 [帧{frame_num}] 开始姿态检测（ROI模式），检测区域: {pose_detection_frame.shape}")
            else:
                print(f"🤖 [帧{frame_num}] 开始姿态检测（全帧模式），检测区域: {pose_detection_frame.shape}")

        return cls(
            frame_num=frame_num,
            frame=frame,
            detection_frame=detection_frame,
            pose_detection_frame=pose_detection_frame,
            roi_offset=roi_offset,
            pose_use_roi=pose_use_roi,
            roi_manager=roi_manager,
            config=config,
        )

    def adjust_detections(
        self,
        pose_results: List,
        ball_positions: List,
        racket_detections: List,
        object_coordinates_are_full_frame: bool = False,
    ) -> Tuple[List, List, List]:
        # 姿态回填：仅在姿态也走 ROI 检测时回填
        if self.pose_use_roi and self.roi_offset != (0, 0) and pose_results:
            pose_results = self.roi_manager.adjust_detection_coordinates(pose_results, self.roi_offset, "pose")

        # 球回填
        if (
            not object_coordinates_are_full_frame
            and self.roi_offset != (0, 0)
            and ball_positions
        ):
            ball_positions = self.roi_manager.adjust_detection_coordinates(ball_positions, self.roi_offset, "ball")

        # 可选：过滤 ROI 外球
        if self.roi_manager.is_roi_set:
            roi_cfg = self.config.get("roi_settings", {})
            if roi_cfg.get("filter_balls_outside_roi", True):
                ball_positions = self.roi_manager.filter_detections_by_roi(ball_positions or [], "ball")

        # 球拍回填
        if (
            not object_coordinates_are_full_frame
            and self.roi_offset != (0, 0)
            and racket_detections
        ):
            racket_detections = self.roi_manager.adjust_detection_coordinates(racket_detections, self.roi_offset, "racket")

        return pose_results, ball_positions, racket_detections
