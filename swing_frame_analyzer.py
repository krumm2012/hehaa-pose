#!/usr/bin/env python3
"""
共享的帧级挥拍分析器
统一管理挥拍分类与挥拍指标计算，支持 stride 缓存复用。
"""

from typing import Dict, List, Optional, Tuple

from full_swing_analyzer import FullSwingAnalyzer
from pose_estimator_yolo26 import PoseEstimatorYOLO26


class SwingFrameAnalyzer:
    """帧级挥拍分析编排器（可在 main.py 和 main_pipe.py 复用）"""

    def __init__(
        self,
        config: Dict,
        frame_dimensions: Tuple[int, int],
        analysis_stride: int = 1,
    ):
        self.config = config
        self.frame_dimensions = frame_dimensions
        self.analysis_stride = max(1, int(analysis_stride))
        self.swing_analyzer = FullSwingAnalyzer(config)
        self._cached_label = "No Pose"
        self._cached_metrics: Dict = {}

    def _compute(
        self,
        poses: List[Dict],
        racket_list: List[Dict],
        ball_pos: Optional[Tuple[float, float]],
    ) -> Tuple[str, Dict]:
        if not poses:
            return "No Pose", {}

        swing_label = PoseEstimatorYOLO26.classify_swing_static(poses[0], self.config)
        detailed_data = self.swing_analyzer.analyze_swing_components(
            poses, racket_list, ball_pos, self.frame_dimensions
        )
        return swing_label, detailed_data

    def analyze(
        self,
        frame_id: int,
        poses: List[Dict],
        racket_list: List[Dict],
        ball_pos: Optional[Tuple[float, float]],
    ) -> Tuple[str, Dict]:
        """按 stride 复用缓存结果，降低每帧计算成本"""
        if not poses:
            self._cached_label = "No Pose"
            self._cached_metrics = {}
            return self._cached_label, self._cached_metrics

        if frame_id % self.analysis_stride == 0:
            self._cached_label, self._cached_metrics = self._compute(
                poses=poses, racket_list=racket_list, ball_pos=ball_pos
            )
        return self._cached_label, self._cached_metrics

    def analyze_force(
        self,
        frame_id: int,
        poses: List[Dict],
        racket_list: List[Dict],
        ball_pos: Optional[Tuple[float, float]],
    ) -> Tuple[str, Dict]:
        """强制计算当前帧，并刷新缓存"""
        if not poses:
            self._cached_label = "No Pose"
            self._cached_metrics = {}
            return self._cached_label, self._cached_metrics

        self._cached_label, self._cached_metrics = self._compute(
            poses=poses, racket_list=racket_list, ball_pos=ball_pos
        )
        return self._cached_label, self._cached_metrics
