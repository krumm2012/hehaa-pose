#!/usr/bin/env python3
"""
精彩瞬间检测与产出服务。
将 main.py 中高光检测、触发与素材输出逻辑收敛到独立模块。
"""

import json
import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class HighlightService:
    def __init__(self, highlights_cfg: Dict, fps: float):
        self.enabled = bool(highlights_cfg.get("enabled", True))
        self.output_dir = highlights_cfg.get("output_dir", "data/highlights")
        self.hit_distance_factor = float(highlights_cfg.get("hit_distance_factor", 1.5))
        self.racket_radius_factor = float(highlights_cfg.get("racket_radius_factor", 1.0))
        self.distance_scale = float(highlights_cfg.get("distance_scale", 1.0))
        self.cooldown_frames = int(highlights_cfg.get("cooldown_frames", 12))
        self.max_highlights = int(highlights_cfg.get("max_highlights", 50))
        self.pre_frames = int(highlights_cfg.get("pre_frames", 5))
        self.post_frames = int(highlights_cfg.get("post_frames", 15))
        self.save_center_image = bool(highlights_cfg.get("save_center_image", True))
        self.annotate_center_image = bool(highlights_cfg.get("annotate_center_image", True))
        self.racket_selection_mode = str(highlights_cfg.get("racket_selection", "nearest")).lower()
        self.min_inside_frames = int(highlights_cfg.get("min_inside_frames", 1))
        self.min_relative_threshold_ratio = float(highlights_cfg.get("min_relative_threshold_ratio", 1.2))
        self.min_exit_increase_px = float(highlights_cfg.get("min_exit_increase_px", 20.0))
        self.min_speed_px_per_frame = float(highlights_cfg.get("min_speed_px_per_frame", 8.0))
        self.min_enter_decrease_px = float(highlights_cfg.get("min_enter_decrease_px", 20.0))
        self.max_consecutive_inside_allowed = int(highlights_cfg.get("max_inside_frames", 2))
        self.ball_radius_px = float(highlights_cfg.get("ball_radius_px", 10))
        self.fps = fps if fps > 0 else 25.0

        self.last_hit_frame = -10**9
        self.highlight_count = 0
        self.prev2_dist = None
        self.prev_dist = None
        self.prev_effective_racket_radius = None
        self.prev_threshold = None
        self.prev_display_snapshot = None
        self.prev_frame_num_snapshot = None
        self.consecutive_inside_count = 0

        self.recent_frames = deque(maxlen=60)
        self.pending_clips: List[Dict] = []

        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _save_highlight_frame(frame: np.ndarray, out_dir: str, frame_num: int, tag: str = "hit") -> str:
        os.makedirs(out_dir, exist_ok=True)
        filename = f"highlight_{frame_num:06d}_{tag}.jpg"
        path = os.path.join(out_dir, filename)
        cv2.imwrite(path, frame)
        return path

    @staticmethod
    def _save_highlight_clip(frames_map: Dict[int, np.ndarray], out_dir: str, center_frame_num: int, fps: float, tag: str = "hit") -> str:
        os.makedirs(out_dir, exist_ok=True)
        ordered_nums = sorted(frames_map.keys())
        if not ordered_nums:
            return ""
        h, w = frames_map[ordered_nums[0]].shape[:2]
        filename = f"highlight_{center_frame_num:06d}_{tag}.mp4"
        path = os.path.join(out_dir, filename)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), fps if fps > 0 else 25, (w, h))
        for fn in ordered_nums:
            writer.write(frames_map[fn])
        writer.release()
        return path

    def add_recent_frame(self, frame_num: int, frame: np.ndarray) -> None:
        if not self.enabled:
            return
        self.recent_frames.append((frame_num, frame.copy()))

    def process_pending_clips(self, frame_num: int, frame: np.ndarray) -> None:
        if not self.enabled or not self.pending_clips:
            return
        still_pending = []
        for task in self.pending_clips:
            if frame_num <= task["end"]:
                task["frames"][frame_num] = frame.copy()
                still_pending.append(task)
            else:
                clip_path = self._save_highlight_clip(task["frames"], self.output_dir, task["center"], self.fps, tag="hit")
                print(f"🎞️ 精彩瞬间短视频已保存: {clip_path}")
        self.pending_clips = still_pending

    def _select_racket(self, ball_position: Tuple[float, float], racket_detections: List[Dict]) -> Optional[Dict]:
        selected_racket = None
        if self.racket_selection_mode == "nearest":
            bx, by = ball_position
            best_dist = float("inf")
            for racket in racket_detections:
                if not isinstance(racket, dict) or "box" not in racket:
                    continue
                x1, y1, x2, y2 = racket["box"]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    selected_racket = racket
        else:
            max_area = -1
            for racket in racket_detections:
                if not isinstance(racket, dict) or "box" not in racket:
                    continue
                x1, y1, x2, y2 = racket["box"]
                area = max(1, (x2 - x1) * (y2 - y1))
                if area > max_area:
                    max_area = area
                    selected_racket = racket
        return selected_racket

    @staticmethod
    def _recent_ball_speed(ball_module) -> float:
        try:
            history = getattr(ball_module, "tracked_balls_history", None)
            if history and len(history) >= 2:
                p1 = np.array(history[-1]["coords"], dtype=float)
                p0 = np.array(history[-2]["coords"], dtype=float)
                return float(np.linalg.norm(p1 - p0))
        except Exception:
            pass
        return 0.0

    def process_frame(
        self,
        frame_num: int,
        display_frame: np.ndarray,
        ball_position: Optional[Tuple[float, float]],
        racket_detections: List[Dict],
        ball_module,
        pose_results: List[Dict],
        frame_processor,
    ) -> None:
        if (
            not self.enabled
            or ball_position is None
            or not racket_detections
            or self.highlight_count >= self.max_highlights
        ):
            return

        bx, by = float(ball_position[0]), float(ball_position[1])
        selected_racket = self._select_racket((bx, by), racket_detections)
        if not selected_racket:
            return

        x1, y1, x2, y2 = selected_racket["box"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        rw = max(1.0, (x2 - x1))
        rh = max(1.0, (y2 - y1))
        effective_racket_radius = max(rw, rh) / 2.0
        impact_threshold = (
            (effective_racket_radius * self.racket_radius_factor + self.ball_radius_px * self.hit_distance_factor)
            * self.distance_scale
        )

        dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
        inside_now = dist <= impact_threshold
        if frame_num % 5 == 0:
            print(f"[HL] frame={frame_num} dist={dist:.2f} thr={impact_threshold:.2f} inside={inside_now}")

        if inside_now:
            self.consecutive_inside_count = min(self.max_consecutive_inside_allowed, self.consecutive_inside_count + 1)
            if self.prev_display_snapshot is None or (self.prev_dist is not None and dist < self.prev_dist):
                self.prev_display_snapshot = display_frame.copy()
                self.prev_frame_num_snapshot = frame_num
        else:
            self.consecutive_inside_count = 0

        local_min_triggered = (
            self.prev_dist is not None
            and self.prev2_dist is not None
            and self.prev_dist
            <= ((self.prev_effective_racket_radius or effective_racket_radius) * self.racket_radius_factor + self.ball_radius_px * self.hit_distance_factor)
            * self.distance_scale
            and self.prev_dist <= self.prev2_dist
            and dist >= self.prev_dist
            and self.prev_display_snapshot is not None
            and (frame_num - self.last_hit_frame) >= self.cooldown_frames
        )
        if local_min_triggered:
            passed = True
            if self.consecutive_inside_count < self.min_inside_frames:
                passed = False
            if not (self.prev_threshold is not None and self.prev_dist <= self.prev_threshold * self.min_relative_threshold_ratio):
                passed = False
            if (dist - self.prev_dist) < self.min_exit_increase_px:
                passed = False
            if (self.prev2_dist - self.prev_dist) < self.min_enter_decrease_px:
                passed = False
            if self._recent_ball_speed(ball_module) < self.min_speed_px_per_frame:
                passed = False

            if not passed:
                self.prev_display_snapshot = None
                self.prev_frame_num_snapshot = None
            else:
                clip_frames = {}
                for fn, fr in list(self.recent_frames):
                    if self.prev_frame_num_snapshot - self.pre_frames <= fn <= self.prev_frame_num_snapshot:
                        clip_frames[fn] = fr.copy()
                self.pending_clips.append(
                    {
                        "start": self.prev_frame_num_snapshot + 1,
                        "end": self.prev_frame_num_snapshot + self.post_frames,
                        "center": self.prev_frame_num_snapshot,
                        "frames": clip_frames,
                    }
                )

                if self.save_center_image and self.prev_display_snapshot is not None:
                    snapshot = self.prev_display_snapshot.copy()
                    if self.annotate_center_image:
                        cv2.circle(snapshot, (int(bx), int(by)), 8, (0, 0, 255), -1)
                        cv2.circle(snapshot, (int(cx), int(cy)), 8, (255, 0, 0), -1)
                        cv2.putText(
                            snapshot,
                            f"HIT dist={self.prev_dist:.1f}",
                            (int(bx) + 10, int(by) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                    img_path = self._save_highlight_frame(snapshot, self.output_dir, self.prev_frame_num_snapshot, tag="hit")
                    print(f"📸 精彩瞬间中心帧已保存: {img_path}")

                try:
                    base_name = f"highlight_{self.prev_frame_num_snapshot:06d}_hit"
                    base_path = os.path.join(self.output_dir, base_name)
                    try:
                        _, analysis = frame_processor.force_swing_analysis(
                            frame_id=frame_num,
                            poses=pose_results if pose_results else [],
                            racket_detections=racket_detections if racket_detections else [],
                            ball_position=(bx, by),
                        )
                    except Exception as exc:
                        print(f"分析组件计算失败: {exc}")
                        analysis = {}

                    payload = {
                        "frame_center": int(self.prev_frame_num_snapshot),
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "analysis": analysis,
                    }
                    with open(f"{base_path}.analysis.json", "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    print(f"📝 分析JSON已保存: {base_path}.analysis.json")

                    if self.prev_display_snapshot is not None:
                        phase_tags = [
                            ("prep", "准备"),
                            ("turn", "转身"),
                            ("drop", "降拍"),
                            ("swing", "挥拍"),
                            ("foot", "步伐"),
                        ]
                        for tag, zh in phase_tags:
                            out_img = self.prev_display_snapshot.copy()
                            try:
                                cv2.putText(out_img, zh, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                            except Exception:
                                pass
                            cv2.imwrite(f"{base_path}_{tag}.jpg", out_img)
                        print(f"🖼️ 阶段占位截图已保存: {base_path}_[prep|turn|drop|swing|foot].jpg")
                except Exception as exc:
                    print(f"生成分析侧车文件失败: {exc}")

                self.last_hit_frame = self.prev_frame_num_snapshot
                self.highlight_count += 1
                print(
                    f"⭐ 精彩瞬间-击球(最小距离): 计划保存短视频（前{self.pre_frames}后{self.post_frames}），"
                    f"中心帧 {self.prev_frame_num_snapshot} (total={self.highlight_count})"
                )
                self.prev_display_snapshot = None
                self.prev_frame_num_snapshot = None

        self.prev2_dist = self.prev_dist
        self.prev_dist = dist
        self.prev_effective_racket_radius = effective_racket_radius
        self.prev_threshold = impact_threshold
