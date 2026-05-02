#!/usr/bin/env python3
"""
共享的帧级业务处理器（不包含 UI 与 I/O）。
负责把检测结果转换为可消费的业务分析结果。
"""

from typing import Dict, List, Optional, Tuple

from swing_frame_analyzer import SwingFrameAnalyzer


class FrameProcessor:
    """检测后融合与分析层：swing/speed/hit 统一输出"""

    def __init__(
        self,
        config: Dict,
        frame_dimensions: Tuple[int, int],
        fps: float,
        analysis_stride: int = 1,
        speed_analyzer=None,
        hit_zone_analyzer=None,
        enable_speed: bool = False,
        enable_hit_zone: bool = False,
    ):
        self.config = config
        self.fps = fps
        self.speed_analyzer = speed_analyzer
        self.hit_zone_analyzer = hit_zone_analyzer
        self.enable_speed = bool(enable_speed and speed_analyzer is not None)
        self.enable_hit_zone = bool(enable_hit_zone and hit_zone_analyzer is not None)
        self.prev_ball_pos: Optional[Tuple[float, float]] = None
        self.swing_analyzer = SwingFrameAnalyzer(
            config=config,
            frame_dimensions=frame_dimensions,
            analysis_stride=analysis_stride,
        )

    @staticmethod
    def _normalize_ball_pos(
        ball_position: Optional[Tuple[float, float]] = None,
        ball_candidates: Optional[List] = None,
    ) -> Optional[Tuple[float, float]]:
        if ball_position is not None:
            return float(ball_position[0]), float(ball_position[1])

        if not ball_candidates:
            return None

        first = ball_candidates[0]
        if isinstance(first, dict) and first.get("position"):
            return float(first["position"][0]), float(first["position"][1])
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            return float(first[0]), float(first[1])
        return None

    @staticmethod
    def _first_racket_box(racket_detections: List[Dict]) -> Optional[Tuple[float, float, float, float]]:
        if not racket_detections:
            return None
        first = racket_detections[0]
        if isinstance(first, dict) and first.get("box"):
            x1, y1, x2, y2 = first["box"]
            return float(x1), float(y1), float(x2), float(y2)
        return None

    def process(
        self,
        frame_id: int,
        poses: List[Dict],
        racket_detections: List[Dict],
        ball_position: Optional[Tuple[float, float]] = None,
        ball_candidates: Optional[List] = None,
    ) -> Dict:
        norm_ball_pos = self._normalize_ball_pos(ball_position=ball_position, ball_candidates=ball_candidates)
        swing_type, phase_metrics = self.swing_analyzer.analyze(
            frame_id=frame_id,
            poses=poses or [],
            racket_list=racket_detections or [],
            ball_pos=norm_ball_pos,
        )

        speed_kmh = None
        if self.enable_speed and norm_ball_pos is not None:
            if self.prev_ball_pos is not None:
                speed_kmh = self.speed_analyzer.calculate_ball_speed(self.prev_ball_pos, norm_ball_pos)
            self.prev_ball_pos = norm_ball_pos
        elif norm_ball_pos is not None:
            self.prev_ball_pos = norm_ball_pos

        hit_analysis = None
        if self.enable_hit_zone and norm_ball_pos is not None and racket_detections:
            racket_box = self._first_racket_box(racket_detections)
            if racket_box is not None:
                hit_analysis = self.hit_zone_analyzer.analyze_hit_zone(
                    ball_pos=norm_ball_pos,
                    racket_bbox=racket_box,
                )

        return {
            "swing_type": swing_type,
            "phase_metrics": phase_metrics,
            "ball_position": norm_ball_pos,
            "speed_kmh": speed_kmh,
            "hit_analysis": hit_analysis,
        }

    def force_swing_analysis(
        self,
        frame_id: int,
        poses: List[Dict],
        racket_detections: List[Dict],
        ball_position: Optional[Tuple[float, float]] = None,
    ) -> Tuple[str, Dict]:
        norm_ball_pos = self._normalize_ball_pos(ball_position=ball_position)
        return self.swing_analyzer.analyze_force(
            frame_id=frame_id,
            poses=poses or [],
            racket_list=racket_detections or [],
            ball_pos=norm_ball_pos,
        )

    def build_frame_record(
        self,
        frame_id: int,
        swing_type: str,
        ball_position: Optional[Tuple[float, float]],
        racket_detections: List[Dict],
        poses: List[Dict],
        phase_metrics: Dict,
    ) -> Dict:
        return {
            "frame_id": frame_id,
            "timestamp": round(frame_id / self.fps, 3) if self.fps else 0.0,
            "swing_type": swing_type,
            "ball": [ball_position[0], ball_position[1]] if ball_position is not None else None,
            "rackets": racket_detections,
            "pose": poses[0] if poses else None,
            "metrics": phase_metrics,
        }
