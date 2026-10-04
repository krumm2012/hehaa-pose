"""
dual_view_renderer.py
──────────────────────
算法 2.0 双视角同步渲染器：
负责在 Side-by-Side 画面上绘制：
1. 正面视角与背面视角的骨骼连线、关节热点与置信度指示
2. 特殊自愈标记（正面遮挡时从镜面拾取的关键点显示高亮提示）
3. 网球生物力学 HUD 仪表盘：
   - 击球类型（Forehand / Backhand / Two-Handed Backhand）
   - 抗侧身塌陷转肩角 (Robust Shoulder Turn)
   - 后背引拍深度 (Takeback Depth) 与肩胛收缩度 (Scapular Pinch)
   - 击球真实性校验结果
4. 导出高清对比视频与动作复盘帧。
"""
from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from dual_pose_estimator import DualPoseResult
from dual_view_biomechanics import Keypoint
from dual_view_manager import DualViewFrame

logger = logging.getLogger("DualViewRenderer")

# COCO 骨骼连线定义 (起点, 终点)
COCO_SKELETON_PAIRS = [
    # 头部五官
    ("left_eye", "right_eye"),
    ("left_eye", "nose"),
    ("right_eye", "nose"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    # 上肢与躯干
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # 下肢
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# 颜色定义 (BGR)
COLOR_FRONT_BONE = (0, 255, 128)      # 亮绿
COLOR_BACK_BONE = (0, 200, 255)       # 暖黄/橙
COLOR_HEALED_BONE = (255, 0, 255)     # 品红（遮挡自愈特权色）
COLOR_JOINT = (0, 0, 255)             # 红色关节点
COLOR_JOINT_HEALED = (255, 255, 0)    # 青色关节点


class DualViewRenderer:
    """
    双视角画中画与骨骼可视化渲染器
    """

    def __init__(
        self,
        show_hud: bool = True,
        show_skeleton: bool = True,
        line_thickness: int = 2,
        point_radius: int = 4,
        mask_backview_eyes: bool = True,
        eye_mask_style: str = "bar",  # "bar" | "mosaic"
    ):
        self.show_hud = show_hud
        self.show_skeleton = show_skeleton
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.mask_backview_eyes = mask_backview_eyes
        self.eye_mask_style = eye_mask_style
        self._prev_eye_boxes: List[Tuple[int, int, int, int]] = []
        self._eye_hold_counter: int = 0
        self._prev_racket_box: Optional[Tuple[int, int, int, int]] = None
        self._racket_hold_counter: int = 0
        self._font_mgr = None

    @property
    def font_mgr(self):
        if self._font_mgr is None:
            from font_manager import FontManager
            self._font_mgr = FontManager()
        return self._font_mgr

    def draw_skeleton(
        self,
        img: np.ndarray,
        pose_dict: Dict[str, Keypoint],
        is_back_view: bool = False,
    ) -> np.ndarray:
        """在单个视角画面上绘制人体骨骼与关键点。"""
        canvas = img.copy()

        # Low-confidence joints remain missing, never completed into solid bones.
        pose_dict = {name: kp for name, kp in pose_dict.items()
                     if kp.conf >= .5 and kp.confidence_source != 'unavailable'
                     and all(math.isfinite(v) for v in (kp.x, kp.y, kp.conf))}
        # 1. 绘制连线
        bone_color = COLOR_BACK_BONE if is_back_view else COLOR_FRONT_BONE

        for p1_name, p2_name in COCO_SKELETON_PAIRS:
            kp1 = pose_dict.get(p1_name)
            kp2 = pose_dict.get(p2_name)
            if kp1 is not None and kp2 is not None:
                # 若包含自愈恢复的关键点，连线高亮显示
                cur_color = COLOR_HEALED_BONE if (kp1.recovered_from_mirror or kp2.recovered_from_mirror) else bone_color
                pt1 = (int(round(kp1.x)), int(round(kp1.y)))
                pt2 = (int(round(kp2.x)), int(round(kp2.y)))
                inferred = kp1.recovered_from_mirror or kp2.recovered_from_mirror
                cached = not kp1.observed or not kp2.observed
                if inferred or cached:
                    color = COLOR_HEALED_BONE if inferred else (140, 140, 140)
                    length = max(1, int(math.dist(pt1, pt2)))
                    for offset in range(0, length, 12):
                        a = offset / length
                        b = min(offset + 6, length) / length
                        start = tuple(round(x+(y-x)*a) for x,y in zip(pt1,pt2))
                        end = tuple(round(x+(y-x)*b) for x,y in zip(pt1,pt2))
                        cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
                else:
                    cv2.line(canvas, pt1, pt2, cur_color, self.line_thickness, cv2.LINE_AA)

        # 2. 绘制关节点
        for name, kp in pose_dict.items():
            pt = (int(round(kp.x)), int(round(kp.y)))
            pt_color = COLOR_JOINT_HEALED if kp.recovered_from_mirror else COLOR_JOINT
            if not kp.observed or kp.recovered_from_mirror:
                pt_color = COLOR_JOINT_HEALED if kp.recovered_from_mirror else (140,140,140)
            cv2.circle(canvas, pt, self.point_radius, pt_color,
                       -1 if kp.observed and not kp.recovered_from_mirror else 1, cv2.LINE_AA)
            cv2.circle(canvas, pt, self.point_radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas

    def draw_hud(
        self,
        sbs_canvas: np.ndarray,
        pose_result: DualPoseResult,
        event_label: Optional[str] = None,
        coaching_text: Optional[str] = None,
    ) -> np.ndarray:
        """在拼接画面顶部与底部叠加网球动力学生物力学仪表盘。"""
        canvas = sbs_canvas.copy()
        h, w = canvas.shape[:2]
        bio = pose_result.biomechanics
        shot = bio.shot_classification

        # 1. 顶部半透明背景条 (根据是否有教练建议自适应高度)
        top_bar_h = 82 if coaching_text else 52
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (20, 20, 22), -1)
        # 底部信息条
        cv2.rectangle(overlay, (0, h - 46), (w, h), (20, 20, 22), -1)
        cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, canvas)

        # 2. 视角标题 (左右两端对齐，绝不挤占中央区域)
        # 左侧正面标题
        cv2.putText(canvas, "FRONT VIEW", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2, cv2.LINE_AA)
        # 右侧背面标题 (靠右端对齐)
        back_text = "BACK VIEW (MIRROR FLIPPED)"
        (bw, bh), _ = cv2.getTextSize(back_text, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
        cv2.putText(
            canvas,
            back_text,
            (w - bw - 20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        # 3. 击球分类核心指标 (居中胶囊徽章设计)
        if event_label is not None:
            shot_text = event_label.upper()
            if "SHADOW SWING" in shot_text or "MISSED" in shot_text:
                shot_color = (0, 165, 255)      # 醒目橙黄色：空挥/未触球
                pill_bg = (35, 28, 20)
            elif "FOREHAND" in shot_text:
                shot_color = (0, 255, 128)      # 亮绿：正手
                pill_bg = (20, 35, 25)
            elif "BACKHAND" in shot_text:
                shot_color = (0, 215, 255)      # 青黄：反手
                pill_bg = (20, 32, 38)
            else:
                shot_color = (180, 180, 180)
                pill_bg = (30, 30, 30)
        else:
            shot_color = (0, 255, 255) if shot.is_two_handed else (0, 255, 0)
            shot_text = f"{shot.shot_type.upper()} ({shot.confidence * 100:.0f}%)"
            pill_bg = (25, 30, 25)

        (text_w, text_h), _ = cv2.getTextSize(shot_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        center_x = w // 2
        badge_pad_x = 16
        bx1 = max(170, center_x - text_w // 2 - badge_pad_x)
        bx2 = min(w - bw - 30, center_x + text_w // 2 + badge_pad_x)
        by1 = 8
        by2 = 40

        # 胶囊徽章底色与边框
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), pill_bg, -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), shot_color, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            shot_text,
            (center_x - text_w // 2, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            shot_color,
            2,
            cv2.LINE_AA,
        )

        # 4. 实时智能教练建议 (居中第二行，防重叠提示框)
        if coaching_text:
            coach_banner = f"💡 COACH: {coaching_text}"
            coach_pill_w = max(280, len(coach_banner) * 14 + 30)
            cx1 = max(100, center_x - coach_pill_w // 2)
            cx2 = min(w - 100, center_x + coach_pill_w // 2)
            cv2.rectangle(canvas, (cx1, 48), (cx2, 74), (28, 28, 34), -1)
            cv2.rectangle(canvas, (cx1, 48), (cx2, 74), (70, 70, 85), 1, cv2.LINE_AA)
            canvas = self.font_mgr.put_text_with_font(
                canvas,
                coach_banner,
                (cx1 + 12, 51),
                font_scale=0.55,
                color=(0, 240, 255),
                thickness=1,
            )

        # 5. 底部生物力学指标详情
        # 转肩角与 X-Factor
        turn_text = f"Width-angle proxy: {bio.robust_shoulder_turn_deg:.1f}" if bio.robust_shoulder_turn_deg is not None else "Width-angle proxy: N/A"
        if bio.shoulder_hip_separation_deg is not None:
            turn_text += f" | 2D shoulder-hip: {bio.shoulder_hip_separation_deg:.1f} deg"
        cv2.putText(canvas, turn_text, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

        # 后背引拍深度与自愈状态及 3D 相对深度
        depth = f"{bio.takeback_depth_ratio:.2f}x" if bio.takeback_depth_ratio is not None else "N/A"
        width = f"{bio.scapular_retraction_ratio:.2f}x" if bio.scapular_retraction_ratio is not None else "N/A"
        back_text = f"Wrist-offset proxy: {depth} | Shoulder-width ratio: {width}"
        if bio.occlusion_healed_points:
            back_text += f" | Healed: {','.join(bio.occlusion_healed_points)}"
        cv2.putText(
            canvas,
            back_text,
            (w // 2 + 20, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def apply_eye_privacy_mask(
        self,
        b_img: np.ndarray,
        detected_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> np.ndarray:
        """
        在背面视角 (Mirror view) 上对出现的人脸眼睛区域进行隐私遮挡。
        支持黑色隐私遮挡条 (bar) 与马赛克 (mosaic)。
        包含帧间平滑与两帧防闪烁保持机制。
        """
        canvas = b_img.copy()
        h, w = canvas.shape[:2]

        boxes_to_draw: List[Tuple[int, int, int, int]] = []
        if detected_boxes:
            # 当前帧检测到眼睛遮挡框
            current_boxes = list(detected_boxes)
            # 若上一帧存在对应框，进行 EMA 平滑处理 (alpha=0.7)
            if self._prev_eye_boxes and len(self._prev_eye_boxes) == len(current_boxes):
                smoothed = []
                for (cx1, cy1, cx2, cy2), (px1, py1, px2, py2) in zip(current_boxes, self._prev_eye_boxes):
                    sx1 = int(round(0.7 * cx1 + 0.3 * px1))
                    sy1 = int(round(0.7 * cy1 + 0.3 * py1))
                    sx2 = int(round(0.7 * cx2 + 0.3 * px2))
                    sy2 = int(round(0.7 * cy2 + 0.3 * py2))
                    smoothed.append((sx1, sy1, sx2, sy2))
                boxes_to_draw = smoothed
            else:
                boxes_to_draw = current_boxes

            self._prev_eye_boxes = list(boxes_to_draw)
            self._eye_hold_counter = 2
        elif self._eye_hold_counter > 0 and self._prev_eye_boxes:
            # 帧间短暂丢失时的平滑保持 (最多保持 2 帧)
            self._eye_hold_counter -= 1
            boxes_to_draw = list(self._prev_eye_boxes)
        else:
            self._prev_eye_boxes = []
            self._eye_hold_counter = 0

        # 绘制隐私遮挡
        for (x1, y1, x2, y2) in boxes_to_draw:
            bx1 = max(0, min(w - 1, x1))
            bx2 = max(0, min(w, x2))
            by1 = max(0, min(h - 1, y1))
            by2 = max(0, min(h, y2))
            if bx2 <= bx1 or by2 <= by1:
                continue

            if self.eye_mask_style == "mosaic":
                roi = canvas[by1:by2, bx1:bx2]
                rw, rh = bx2 - bx1, by2 - by1
                if rw > 4 and rh > 4:
                    small = cv2.resize(roi, (max(2, rw // 8), max(2, rh // 4)), interpolation=cv2.INTER_NEAREST)
                    mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
                    canvas[by1:by2, bx1:bx2] = mosaic
                    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (180, 180, 180), 1, cv2.LINE_AA)
            else:
                # 经典隐私条 (Solid Charcoal Redaction Bar with subtle 1px border)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (200, 200, 200), 1, cv2.LINE_AA)

        return canvas

    def draw_impact_telemetry_card(
        self,
        sbs_canvas: np.ndarray,
        card_data: Dict[str, Any],
        opacity: float = 0.88,
    ) -> np.ndarray:
        """
        在画面中央/指定位置渲染击球瞬间特写遥测卡片 (Impact Telemetry Card)
        包含第一、第二、第三梯队拓展的高级网球生物力学指标
        """
        canvas = sbs_canvas.copy()
        h, w = canvas.shape[:2]

        card_w = min(500, w - 40)
        card_h = 220
        x1 = (w - card_w) // 2
        y1 = 80  # 紧接在顶部 HUD 下方居中浮动，位于左右机位交界处
        x2 = x1 + card_w
        y2 = y1 + card_h

        overlay = canvas.copy()
        # 半透明深黑底板 (高级磨砂质感)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (16, 18, 24), -1)
        # 顶部标题栏背景
        cv2.rectangle(overlay, (x1, y1), (x2, y1 + 38), (28, 35, 48), -1)
        cv2.addWeighted(overlay, opacity, canvas, 1.0 - opacity, 0, canvas)

        # 发光外边框 (高科技青蓝色)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (x1, y1 + 38), (x2, y1 + 38), (0, 180, 210), 1, cv2.LINE_AA)

        # 标题栏文本
        title_text = "2D OBSERVATIONS"
        canvas = self.font_mgr.put_text_with_font(
            canvas,
            title_text,
            (x1 + 16, y1 + 8),
            font_scale=0.45,
            color=(0, 240, 255),
            thickness=2,
        )

        score = card_data.get("swing_score")
        grade = str(card_data.get("swing_grade", "N/A"))
        grade_colors = {
            "PRO": (0, 215, 255),          # 金黄色
            "ADVANCED": (0, 255, 128),     # 翡翠绿
            "INTERMEDIATE": (0, 200, 255), # 暖橙
            "DEVELOPING": (200, 200, 200), # 灰白
        }
        badge_color = grade_colors.get(grade, (0, 240, 255))
        score_label = "COACH" if (card_data.get("practice_score") or {}).get("method") == "coach_manual" else "VISIBLE"
        score_badge = f"{score_label}: {score:.1f}" if score is not None else f"{score_label}: N/A"
        canvas = self.font_mgr.put_text_with_font(
            canvas,
            score_badge,
            (x2 - 190, y1 + 8),
            font_scale=0.55,
            color=badge_color,
            thickness=2,
        )

        # 4 行遥测核心指标
        speed_px_s = card_data.get("racket_speed_px_s")
        brush_deg = card_data.get("brush_angle_deg")
        drop_ratio = card_data.get("drop_depth_ratio")
        foot_angle = card_data.get("foot_line_angle_deg")
        stance_str = f"{foot_angle:.0f} deg (2D)" if foot_angle is not None else "N/A"
        hip_rise_px = card_data.get("hip_rise_px")
        seq_text = str(card_data.get("kinematic_sequence_text") or "未观测")

        drive_display = f"{float(hip_rise_px):.0f}px (2D)" if hip_rise_px is not None else str(card_data.get("hip_evidence_text") or "N/A")
        drop_display = f"{float(drop_ratio):.2f}x" if drop_ratio is not None else "N/A"

        items = [
            ("RACKET SPEED", (f"{speed_px_s:.0f} px/s (2D centre)" if speed_px_s is not None else "N/A - uncalibrated"), (0, 255, 180)),
            ("PATH & RISE", (f"{brush_deg:+.1f} deg | Rise: {drop_display}" if brush_deg is not None else str(card_data.get("brush_evidence_text") or "观测证据不足") + f" | Rise: {drop_display}"), (0, 220, 255)),
            ("ANKLE & HIP", f"{stance_str} | Rise: {drive_display}", (255, 230, 100)),
            ("2D PEAK ORDER", f"{seq_text}", (255, 180, 255)),
        ]

        row_y = y1 + 48
        for label, val_str, val_col in items:
            canvas = self.font_mgr.put_text_with_font(
                canvas,
                label,
                (x1 + 16, row_y),
                font_scale=0.48,
                color=(170, 180, 195),
                thickness=1,
            )
            canvas = self.font_mgr.put_text_with_font(
                canvas,
                val_str,
                (x1 + 160, row_y),
                font_scale=0.50,
                color=val_col,
                thickness=1,
            )
            row_y += 38

        return canvas

    def render_dual_frame(
        self,
        dual_frame: DualViewFrame,
        pose_result: DualPoseResult,
        event_label: Optional[str] = None,
        coaching_text: Optional[str] = None,
        ball_trail: Optional[List[Tuple[float, float]]] = None,
        racket_box: Optional[Tuple[float, float, float, float]] = None,
        mask_back_eyes: Optional[bool] = None,
        telemetry_card: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        全量渲染单帧双视角画面：
        1. 骨骼绘制
        2. 运动球轨迹拖尾与球拍框绘制（正面视角）
        3. 背面机位人脸眼睛隐私遮蔽
        4. Side-by-Side 拼接
        5. 生物力学 HUD 叠加
        6. 击球瞬间特写遥测卡片叠加 (可选)
        """
        f_img = dual_frame.front_frame.copy()
        b_img = dual_frame.back_frame.copy()

        # 绘制球轨迹 (正面视角局部映射)
        if ball_trail:
            mapped_trail = []
            for pt in ball_trail:
                if pt is not None:
                    # 将原图坐标映射至正面局部视口
                    fx, fy = dual_frame.front_info.map_from_original(pt[0], pt[1])
                    if -50 <= fx <= f_img.shape[1] + 50 and -50 <= fy <= f_img.shape[0] + 50:
                        mapped_trail.append((int(round(fx)), int(round(fy))))
            
            # 绘制连续轨迹光效
            num_pts = len(mapped_trail)
            for i in range(1, num_pts):
                p_prev = mapped_trail[i - 1]
                p_curr = mapped_trail[i]
                alpha_factor = i / max(1, num_pts)
                line_w = max(1, int(round(1 + alpha_factor * 3)))
                # 渐变荧光黄绿
                cv2.line(f_img, p_prev, p_curr, (0, int(220 * alpha_factor + 35), int(255 * alpha_factor)), line_w, cv2.LINE_AA)

            # 绘制当前最新球点
            if mapped_trail:
                curr_pt = mapped_trail[-1]
                cv2.circle(f_img, curr_pt, 6, (0, 255, 230), -1, cv2.LINE_AA)
                cv2.circle(f_img, curr_pt, 7, (255, 255, 255), 1, cv2.LINE_AA)

        # 绘制球拍边界框 (正面视角局部映射，带帧间平滑与自愈保持)
        box_to_draw = None
        if racket_box is not None and len(racket_box) >= 4:
            rx1, ry1, rx2, ry2 = racket_box[:4]
            fx1, fy1 = dual_frame.front_info.map_from_original(rx1, ry1)
            fx2, fy2 = dual_frame.front_info.map_from_original(rx2, ry2)
            px1, py1 = int(round(min(fx1, fx2))), int(round(min(fy1, fy2)))
            px2, py2 = int(round(max(fx1, fx2))), int(round(max(fy1, fy2)))
            if px2 > 0 and py2 > 0 and px1 < f_img.shape[1] and py1 < f_img.shape[0]:
                curr_box = (px1, py1, px2, py2)
                if self._prev_racket_box is not None:
                    sx1 = int(round(0.75 * px1 + 0.25 * self._prev_racket_box[0]))
                    sy1 = int(round(0.75 * py1 + 0.25 * self._prev_racket_box[1]))
                    sx2 = int(round(0.75 * px2 + 0.25 * self._prev_racket_box[2]))
                    sy2 = int(round(0.75 * py2 + 0.25 * self._prev_racket_box[3]))
                    box_to_draw = (sx1, sy1, sx2, sy2)
                else:
                    box_to_draw = curr_box
                self._prev_racket_box = box_to_draw
                self._racket_hold_counter = 2
        elif self._racket_hold_counter > 0 and self._prev_racket_box is not None:
            self._racket_hold_counter -= 1
            box_to_draw = self._prev_racket_box

        if box_to_draw is not None:
            px1, py1, px2, py2 = box_to_draw
            cv2.rectangle(f_img, (px1, py1), (px2, py2), (255, 220, 0), 2, cv2.LINE_AA)
            cv2.putText(
                f_img,
                "RACKET",
                (px1, max(18, py1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 220, 0),
                1,
                cv2.LINE_AA,
            )

        if self.show_skeleton:
            # 严格依据当前正面视口几何投影映射骨骼（消除多进程跨帧视口漂移导致的骨骼错位）
            front_pose_to_draw = {}
            source_front = getattr(pose_result, "fused_pose_orig", None) or getattr(pose_result, "front_pose_orig", None)
            if source_front and getattr(dual_frame, "front_info", None) is not None:
                for name, kp in source_front.items():
                    fx, fy = dual_frame.front_info.map_from_original(kp.x, kp.y)
                    front_pose_to_draw[name] = replace(kp, x=fx, y=fy)
            else:
                front_pose_to_draw = pose_result.fused_pose_local

            f_img = self.draw_skeleton(f_img, front_pose_to_draw, is_back_view=False)

            # 在背面绘制背面视角关键点（严格依据当前背面视口几何投影映射）
            back_pose_to_draw = {}
            source_back = getattr(pose_result, "back_pose_orig", None)
            if source_back and getattr(dual_frame, "back_info", None) is not None:
                for name, kp in source_back.items():
                    bx, by = dual_frame.back_info.map_from_original(kp.x, kp.y)
                    back_pose_to_draw[name] = replace(kp, x=bx, y=by)
            else:
                back_pose_to_draw = pose_result.back_pose_local

            b_img = self.draw_skeleton(b_img, back_pose_to_draw, is_back_view=True)

        # 在背面视角画面上叠加人脸眼睛隐私遮挡（在骨骼绘制之后，确保完整遮蔽）
        should_mask_eyes = self.mask_backview_eyes if mask_back_eyes is None else mask_back_eyes
        if should_mask_eyes:
            back_eyes = getattr(pose_result, "back_view_eyes", [])
            b_img = self.apply_eye_privacy_mask(b_img, back_eyes)

        # 拼接左右画面
        sbs = np.hstack([f_img, b_img])

        if self.show_hud:
            sbs = self.draw_hud(sbs, pose_result, event_label=event_label, coaching_text=coaching_text)

        if telemetry_card is not None:
            sbs = self.draw_impact_telemetry_card(sbs, telemetry_card)

        return sbs
