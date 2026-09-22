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
                cv2.line(canvas, pt1, pt2, cur_color, self.line_thickness, cv2.LINE_AA)

        # 2. 绘制关节点
        for name, kp in pose_dict.items():
            pt = (int(round(kp.x)), int(round(kp.y)))
            pt_color = COLOR_JOINT_HEALED if kp.recovered_from_mirror else COLOR_JOINT
            cv2.circle(canvas, pt, self.point_radius, pt_color, -1, cv2.LINE_AA)
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
        turn_text = f"Turn: {bio.robust_shoulder_turn_deg:.1f} deg"
        if bio.shoulder_hip_separation_deg is not None:
            turn_text += f" | X-Factor: {bio.shoulder_hip_separation_deg:.1f} deg"
        cv2.putText(canvas, turn_text, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

        # 后背引拍深度与自愈状态
        back_text = f"Takeback: {bio.takeback_depth_ratio * 100:.1f}% | Scapular: {bio.scapular_retraction_ratio:.2f}"
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

    def render_dual_frame(
        self,
        dual_frame: DualViewFrame,
        pose_result: DualPoseResult,
        event_label: Optional[str] = None,
        coaching_text: Optional[str] = None,
        ball_trail: Optional[List[Tuple[float, float]]] = None,
        racket_box: Optional[Tuple[float, float, float, float]] = None,
        mask_back_eyes: Optional[bool] = None,
    ) -> np.ndarray:
        """
        全量渲染单帧双视角画面：
        1. 骨骼绘制
        2. 运动球轨迹拖尾与球拍框绘制（正面视角）
        3. 背面机位人脸眼睛隐私遮蔽
        4. Side-by-Side 拼接
        5. 生物力学 HUD 叠加
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

        # 绘制球拍边界框 (正面视角局部映射)
        if racket_box is not None and len(racket_box) >= 4:
            rx1, ry1, rx2, ry2 = racket_box[:4]
            fx1, fy1 = dual_frame.front_info.map_from_original(rx1, ry1)
            fx2, fy2 = dual_frame.front_info.map_from_original(rx2, ry2)
            px1, py1 = int(round(min(fx1, fx2))), int(round(min(fy1, fy2)))
            px2, py2 = int(round(max(fx1, fx2))), int(round(max(fy1, fy2)))
            if px2 > 0 and py2 > 0 and px1 < f_img.shape[1] and py1 < f_img.shape[0]:
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
            # 在正面绘制自愈后的完整姿态
            f_img = self.draw_skeleton(f_img, pose_result.fused_pose_local, is_back_view=False)
            # 在背面绘制背面视角关键点
            b_img = self.draw_skeleton(b_img, pose_result.back_pose_local, is_back_view=True)

        # 在背面视角画面上叠加人脸眼睛隐私遮挡（在骨骼绘制之后，确保完整遮蔽）
        should_mask_eyes = self.mask_backview_eyes if mask_back_eyes is None else mask_back_eyes
        if should_mask_eyes:
            back_eyes = getattr(pose_result, "back_view_eyes", [])
            b_img = self.apply_eye_privacy_mask(b_img, back_eyes)

        # 拼接左右画面
        sbs = np.hstack([f_img, b_img])

        if self.show_hud:
            sbs = self.draw_hud(sbs, pose_result, event_label=event_label, coaching_text=coaching_text)

        return sbs
