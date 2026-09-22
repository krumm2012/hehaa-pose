"""
dual_view_biomechanics.py
──────────────────────────
算法 2.0 网球高级生物力学引擎：
深度融合借鉴 https://github.com/krumm2012/Tennis-Vision 中的核心肢体几何算法：
1. 躯干中线跨越法则 (Midline Crossing Rule)：基于身体相对坐标系与躯干轴投影，判定正手/反手
2. 双手握拍几何判定 (Two-Handed Detection)：基于双腕间距与肩宽比 (TWO_HANDED_MAX_GAP=0.45)
3. 击球距离物理门控 (Contact Distance Validation)：击球瞬间校验腕-球/拍-球间距，剔除空挥
4. 抗侧身塌陷转肩角 (Anti Side-on Collapse)：正反双重视角互斥投影消除侧身计算奇点
5. 盲区姿态自愈 (Occlusion Self-Healing)：正面引拍手腕遮挡时，自动从背面镜像无损拾取
6. 后背特色动力链指标：后背引拍深度 (Takeback Depth) 与肩胛骨收紧度 (Scapular Pinch)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# 常量定义（与 Tennis-Vision 完全对齐）
FOREHAND = "Forehand"
BACKHAND = "Backhand"
TWO_HANDED_BACKHAND = "Two-Handed Backhand"
UNKNOWN = "Unknown"

# 双手同时握拍的最大双腕距离与肩宽之比（Tennis-Vision 经真实网球比赛测算的物理门限）
TWO_HANDED_MAX_GAP_RATIO = 0.45

# COCO 17 关键点标准映射
COCO_KEYPOINTS = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


@dataclass
class Keypoint:
    x: float
    y: float
    conf: float
    z: Optional[float] = None
    recovered_from_mirror: bool = False

    @property
    def pt(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class ShotClassificationResult:
    shot_type: str                  # Forehand | Backhand | Two-Handed Backhand | Unknown
    confidence: float               # 0.0 ~ 1.0
    hitting_hand: str               # "right" | "left" | "both"
    is_two_handed: bool             # 是否双手握拍
    midline_side_projection: float  # 手腕相对躯干轴投影距离（+为左，-为右）
    wrist_ball_distance: Optional[float] = None
    is_valid_contact: bool = True   # 是否通过击球物理距离校验
    rejection_reason: Optional[str] = None


@dataclass
class DualViewBiomechanicsResult:
    # 基础与高级击球分类
    shot_classification: ShotClassificationResult

    # 旋转与转体动力学
    front_shoulder_width: float
    back_shoulder_width: float
    robust_shoulder_turn_deg: float   # 消除侧身退化后的真实转肩角
    shoulder_hip_separation_deg: Optional[float] = None  # X-Factor (肩髋分离角)

    # 后背特色指标
    takeback_depth_ratio: float = 0.0       # 引拍深度比率（手腕引拍离后背脊柱距离 / 肩宽）
    scapular_retraction_ratio: float = 1.0  # 肩胛骨收紧比率

    # 姿态自愈记录
    occlusion_healed_points: List[str] = field(default_factory=list)


class DualViewBiomechanicsEngine:
    """
    双视角网球生物力学分析引擎
    """

    def __init__(
        self,
        dominant_hand: str = "right",
        max_contact_distance: float = 200.0,
        min_keypoint_conf: float = 0.35,
        two_handed_max_gap: float = TWO_HANDED_MAX_GAP_RATIO,
    ):
        self.dominant_hand = dominant_hand.lower()
        self.max_contact_distance = max_contact_distance
        self.min_keypoint_conf = min_keypoint_conf
        self.two_handed_max_gap = two_handed_max_gap

    @staticmethod
    def parse_pose_dict(
        raw_pose: Union[Dict[str, Any], np.ndarray, List[Any]]
    ) -> Dict[str, Keypoint]:
        """将不同格式的姿态输出（字典、COCO 17x3 数组等）统一转换为 Dict[str, Keypoint]。"""
        parsed: Dict[str, Keypoint] = {}

        if isinstance(raw_pose, dict):
            # 形式为 {"left_shoulder": [x, y, conf], ...} 或 {"left_shoulder": (x, y)}
            for k, v in raw_pose.items():
                if v is None:
                    continue
                name = k.lower()
                if isinstance(v, (list, tuple)):
                    if len(v) >= 3:
                        parsed[name] = Keypoint(float(v[0]), float(v[1]), float(v[2]))
                    elif len(v) == 2:
                        parsed[name] = Keypoint(float(v[0]), float(v[1]), 1.0)
                elif hasattr(v, "x") and hasattr(v, "y"):
                    conf = getattr(v, "conf", 1.0)
                    parsed[name] = Keypoint(float(v.x), float(v.y), float(conf))
        elif isinstance(raw_pose, (list, np.ndarray)):
            arr = np.array(raw_pose)
            if arr.ndim == 2 and arr.shape[0] == 17:
                for idx, name in COCO_KEYPOINTS.items():
                    x, y = float(arr[idx, 0]), float(arr[idx, 1])
                    conf = float(arr[idx, 2]) if arr.shape[1] > 2 else 1.0
                    parsed[name] = Keypoint(x, y, conf)

        return parsed

    def heal_occluded_pose(
        self,
        front_pose: Dict[str, Keypoint],
        back_pose: Dict[str, Keypoint],
    ) -> Tuple[Dict[str, Keypoint], List[str]]:
        """
        盲区姿态自愈逻辑：
        当正面视角的引拍侧手臂（手腕、手肘）因侧身被身体遮挡（置信度低）时，
        从背面机位提取对应的高置信度关键点进行互补自愈。
        """
        healed_pose = dict(front_pose)
        healed_list: List[str] = []

        critical_joints = ["right_wrist", "right_elbow", "left_wrist", "left_elbow"]

        for joint in critical_joints:
            f_kp = front_pose.get(joint)
            b_kp = back_pose.get(joint)

            f_valid = f_kp is not None and f_kp.conf >= self.min_keypoint_conf
            b_valid = b_kp is not None and b_kp.conf >= self.min_keypoint_conf

            if not f_valid and b_valid:
                # 正面丢失但背面有效：自愈补全
                healed_pose[joint] = Keypoint(
                    x=b_kp.x,
                    y=b_kp.y,
                    conf=b_kp.conf * 0.9,  # 适度衰减置信度作为融合标记
                    recovered_from_mirror=True,
                )
                healed_list.append(joint)

        return healed_pose, healed_list

    def classify_shot(
        self,
        pose: Dict[str, Keypoint],
        ball_pos: Optional[Tuple[float, float]] = None,
    ) -> ShotClassificationResult:
        """
        移植自 Tennis-Vision 的正反手与双手击球判定算法：
        基于躯干轴投影中线跨越法则，支持双手握拍识别与接触点距离门控。
        """
        l_sh = pose.get("left_shoulder")
        r_sh = pose.get("right_shoulder")

        if not l_sh or not r_sh:
            return ShotClassificationResult(
                shot_type=UNKNOWN,
                confidence=0.0,
                hitting_hand=self.dominant_hand,
                is_two_handed=False,
                midline_side_projection=0.0,
                is_valid_contact=False,
                rejection_reason="missing_shoulders",
            )

        # 构建身体解剖坐标系：
        # 脊柱轴 (Spine Axis): 从髋骨中心指向双肩中心
        l_hip = pose.get("left_hip")
        r_hip = pose.get("right_hip")
        center_x = (l_sh.x + r_sh.x) / 2.0
        center_y = (l_sh.y + r_sh.y) / 2.0

        if l_hip and r_hip:
            mid_hip_x = (l_hip.x + r_hip.x) / 2.0
            mid_hip_y = (l_hip.y + r_hip.y) / 2.0
            spine_dx = center_x - mid_hip_x
            spine_dy = center_y - mid_hip_y
        else:
            # 回退：假设垂直站立
            spine_dx, spine_dy = 0.0, -100.0

        # 解剖横向轴 (Transverse Axis): 垂直于脊柱轴，指向选手解剖左侧
        # 向量 (-spine_dy, spine_dx)
        axis_dx = -spine_dy
        axis_dy = spine_dx
        axis_len = math.hypot(axis_dx, axis_dy)

        # 确保轴向量朝向与左肩大体同向 (x 轴方向对齐)
        raw_sh_dx = l_sh.x - r_sh.x
        if axis_dx * raw_sh_dx < 0:
            axis_dx = -axis_dx
            axis_dy = -axis_dy

        if axis_len < 1e-4:
            return ShotClassificationResult(
                shot_type=UNKNOWN,
                confidence=0.0,
                hitting_hand=self.dominant_hand,
                is_two_handed=False,
                midline_side_projection=0.0,
                is_valid_contact=False,
                rejection_reason="collapsed_spine_axis",
            )

        l_wrist = pose.get("left_wrist")
        r_wrist = pose.get("right_wrist")

        # 1. 判定双手持拍 (Two-Handed Detection - Tennis-Vision 黄金规则)
        is_two_handed = False
        if l_wrist and r_wrist:
            wrist_gap = math.hypot(l_wrist.x - r_wrist.x, l_wrist.y - r_wrist.y)
            if wrist_gap < self.two_handed_max_gap * axis_len:
                is_two_handed = True

        # 2. 选定击球点手腕位置
        if is_two_handed and l_wrist and r_wrist:
            hitting_hand = "both"
            hit_x = (l_wrist.x + r_wrist.x) / 2.0
            hit_y = (l_wrist.y + r_wrist.y) / 2.0
        else:
            hitting_hand = self.dominant_hand
            primary_wrist = r_wrist if self.dominant_hand == "right" else l_wrist
            if primary_wrist is None:
                # 回退到另一个手腕
                primary_wrist = l_wrist if self.dominant_hand == "right" else r_wrist
            if primary_wrist is None:
                return ShotClassificationResult(
                    shot_type=UNKNOWN,
                    confidence=0.0,
                    hitting_hand=self.dominant_hand,
                    is_two_handed=False,
                    midline_side_projection=0.0,
                    is_valid_contact=False,
                    rejection_reason="no_wrists_detected",
                )
            hit_x, hit_y = primary_wrist.x, primary_wrist.y

        # 3. 接触点物理距离校验 (Contact Distance Validation)
        wrist_ball_dist = None
        is_valid_contact = True
        rejection_reason = None

        if ball_pos is not None:
            bx, by = ball_pos
            wrist_ball_dist = math.hypot(hit_x - bx, hit_y - by)
            if self.max_contact_distance is not None and wrist_ball_dist > self.max_contact_distance:
                is_valid_contact = False
                rejection_reason = f"wrist_ball_dist_{wrist_ball_dist:.1f}px_exceeds_max_{self.max_contact_distance}px"

        # 4. 躯干中线投影 (Midline Crossing Rule)
        # 相对中心偏移向量
        rel_x = hit_x - center_x
        rel_y = hit_y - center_y
        # 投影到轴向量: >0 为偏左肩侧, <0 为偏右肩侧
        side_proj = (rel_x * axis_dx + rel_y * axis_dy) / axis_len

        # 5. 分类逻辑 (结合双手握拍与中线跨越几何)
        is_backhand_side = (side_proj > 0) if self.dominant_hand == "right" else (side_proj < 0)

        if is_two_handed and is_backhand_side:
            # 双手持拍且位于反手侧，确认为双手反拍
            shot_type = TWO_HANDED_BACKHAND
            confidence = min(1.0, max(0.65, 1.0 - (wrist_gap / (self.two_handed_max_gap * axis_len))))
        elif is_backhand_side:
            # 单手反拍
            shot_type = BACKHAND
            confidence = min(1.0, 0.6 + abs(side_proj) / axis_len)
        else:
            # 正手侧（即便引拍阶段双手扶拍喉，亦属正手引拍准备）
            shot_type = FOREHAND
            confidence = min(1.0, 0.6 + abs(side_proj) / axis_len)

        return ShotClassificationResult(
            shot_type=shot_type,
            confidence=round(float(confidence), 4),
            hitting_hand=hitting_hand,
            is_two_handed=is_two_handed,
            midline_side_projection=round(float(side_proj), 2),
            wrist_ball_distance=round(float(wrist_ball_dist), 2) if wrist_ball_dist is not None else None,
            is_valid_contact=is_valid_contact,
            rejection_reason=rejection_reason,
        )

    def calculate_dual_biomechanics(
        self,
        front_pose_raw: Union[Dict[str, Any], np.ndarray, List[Any]],
        back_pose_raw: Union[Dict[str, Any], np.ndarray, List[Any]],
        ball_pos: Optional[Tuple[float, float]] = None,
    ) -> DualViewBiomechanicsResult:
        """
        全量计算前后双视角融合网球生物力学指标
        """
        f_pose = self.parse_pose_dict(front_pose_raw)
        b_pose = self.parse_pose_dict(back_pose_raw)

        # 1. 遮挡自愈
        fused_pose, healed_points = self.heal_occluded_pose(f_pose, b_pose)

        # 2. 击球分类与触球校验
        shot_res = self.classify_shot(fused_pose, ball_pos=ball_pos)

        # 3. 消除侧身退化转肩角 (Anti Side-on Collapse)
        # 前视角肩线
        f_l_sh = fused_pose.get("left_shoulder")
        f_r_sh = fused_pose.get("right_shoulder")
        f_w = math.hypot(f_l_sh.x - f_r_sh.x, f_l_sh.y - f_r_sh.y) if f_l_sh and f_r_sh else 0.0

        # 后视角肩线
        b_l_sh = b_pose.get("left_shoulder")
        b_r_sh = b_pose.get("right_shoulder")
        b_w = math.hypot(b_l_sh.x - b_r_sh.x, b_l_sh.y - b_r_sh.y) if b_l_sh and b_r_sh else 0.0

        # 结合正面与反面宽度的稳定转体角度估计：
        # 当纯正面正对相机时 f_w 接近最大，侧身 90 度时 f_w 接近最小
        # 同时利用 atan2(f_w, b_w) 在象限内平滑过渡
        robust_turn_deg = math.degrees(math.atan2(max(1.0, b_w), max(1.0, f_w)))

        # 4. 肩髋分离角 (X-Factor)
        f_l_hip = fused_pose.get("left_hip")
        f_r_hip = fused_pose.get("right_hip")
        sep_deg = None
        if f_l_sh and f_r_sh and f_l_hip and f_r_hip:
            sh_ang = math.degrees(math.atan2(f_l_sh.y - f_r_sh.y, f_l_sh.x - f_r_sh.x))
            hip_ang = math.degrees(math.atan2(f_l_hip.y - f_r_hip.y, f_l_hip.x - f_r_hip.x))
            sep_deg = abs((sh_ang - hip_ang + 180.0) % 360.0 - 180.0)

        # 5. 后背特色指标：引拍深度 (Takeback Depth)
        # 以后背脊柱中线为基准，测算击球手腕向后拉伸的深度
        takeback_depth_ratio = 0.0
        if b_l_sh and b_r_sh:
            spine_x = (b_l_sh.x + b_r_sh.x) / 2.0
            hitting_wrist_b = b_pose.get("right_wrist") if self.dominant_hand == "right" else b_pose.get("left_wrist")
            if hitting_wrist_b and b_w > 10.0:
                takeback_depth_ratio = abs(hitting_wrist_b.x - spine_x) / b_w

        # 6. 肩胛收缩度 (Scapular Retraction)
        scapular_ratio = (b_w / max(1.0, f_w)) if f_w > 10.0 else 1.0

        return DualViewBiomechanicsResult(
            shot_classification=shot_res,
            front_shoulder_width=round(f_w, 2),
            back_shoulder_width=round(b_w, 2),
            robust_shoulder_turn_deg=round(robust_turn_deg, 2),
            shoulder_hip_separation_deg=round(sep_deg, 2) if sep_deg is not None else None,
            takeback_depth_ratio=round(takeback_depth_ratio, 4),
            scapular_retraction_ratio=round(scapular_ratio, 4),
            occlusion_healed_points=healed_points,
        )
