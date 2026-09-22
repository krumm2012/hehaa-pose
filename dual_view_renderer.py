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
    ):
        self.show_hud = show_hud
        self.show_skeleton = show_skeleton
        self.line_thickness = line_thickness
        self.point_radius = point_radius
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
        top_bar_h = 96 if coaching_text else 72
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (20, 20, 20), -1)
        # 底部信息条
        cv2.rectangle(overlay, (0, h - 50), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        # 2. 标题文字
        # 左侧正面标题
        cv2.putText(canvas, "FRONT VIEW", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        # 右侧背面标题 (w // 2 处)
        cv2.putText(
            canvas,
            "BACK VIEW (MIRROR FLIPPED)",
            (w // 2 + 20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        # 3. 击球分类核心指标 (居中显示)
        if event_label is not None:
            shot_text = event_label.upper()
            if "FOREHAND" in shot_text:
                shot_color = (0, 255, 128)
            elif "BACKHAND" in shot_text:
                shot_color = (0, 215, 255)
            else:
                shot_color = (180, 180, 180)
        else:
            shot_color = (0, 255, 255) if shot.is_two_handed else (0, 255, 0)
            shot_text = f"{shot.shot_type.upper()} ({shot.confidence * 100:.0f}%)"

        (text_w, text_h), _ = cv2.getTextSize(shot_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(
            canvas,
            shot_text,
            ((w - text_w) // 2, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            shot_color,
            2,
            cv2.LINE_AA,
        )

        # 4. 实时智能教练建议 (若提供)
        if coaching_text:
            coach_banner = f"COACH: {coaching_text}"
            canvas = self.font_mgr.put_text_with_font(
                canvas,
                coach_banner,
                (w // 2 - 200, 68),
                font_scale=0.55,
                color=(0, 255, 255),
                thickness=1,
            )

        # 5. 底部生物力学指标详情
        # 转肩角与 X-Factor
        turn_text = f"Turn: {bio.robust_shoulder_turn_deg:.1f} deg"
        if bio.shoulder_hip_separation_deg is not None:
            turn_text += f" | X-Factor: {bio.shoulder_hip_separation_deg:.1f} deg"
        cv2.putText(canvas, turn_text, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 后背引拍深度与自愈状态
        back_text = f"Takeback Depth: {bio.takeback_depth_ratio * 100:.1f}% | Scapular: {bio.scapular_retraction_ratio:.2f}"
        if bio.occlusion_healed_points:
            back_text += f" | Healed: {','.join(bio.occlusion_healed_points)}"
        cv2.putText(
            canvas,
            back_text,
            (w // 2 + 20, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def render_dual_frame(
        self,
        dual_frame: DualViewFrame,
        pose_result: DualPoseResult,
        event_label: Optional[str] = None,
        coaching_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        全量渲染单帧双视角画面：
        1. 骨骼绘制
        2. Side-by-Side 拼接
        3. 生物力学 HUD 叠加
        """
        f_img = dual_frame.front_frame.copy()
        b_img = dual_frame.back_frame.copy()

        if self.show_skeleton:
            # 在正面绘制自愈后的完整姿态
            f_img = self.draw_skeleton(f_img, pose_result.fused_pose_local, is_back_view=False)
            # 在背面绘制背面视角关键点
            b_img = self.draw_skeleton(b_img, pose_result.back_pose_local, is_back_view=True)

        # 拼接左右画面
        sbs = np.hstack([f_img, b_img])

        if self.show_hud:
            sbs = self.draw_hud(sbs, pose_result, event_label=event_label, coaching_text=coaching_text)

        return sbs
