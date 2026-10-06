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
from observation_policy import finite_number, finite_point

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
    observed: bool = True
    source_frame_id: Optional[int] = None
    confidence_source: str = "provided"

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
    is_valid_contact: Optional[bool] = None   # 是否通过击球物理距离校验
    rejection_reason: Optional[str] = None


@dataclass
class DualViewBiomechanicsResult:
    # 基础与高级击球分类
    shot_classification: ShotClassificationResult

    # 旋转与转体动力学
    front_shoulder_width: float
    back_shoulder_width: float
    robust_shoulder_turn_deg: Optional[float]  # Uncalibrated shoulder-width projection proxy
    shoulder_hip_separation_deg: Optional[float] = None  # X-Factor (肩髋分离角)

    # 后背特色指标与 3D 相对深度
    takeback_depth_ratio: Optional[float] = None
    scapular_retraction_ratio: Optional[float] = None
    relative_depth_z: Optional[float] = None  # 双机位前后尺度视差拟合的相对 3D 深度比率

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

        def add(name, x, y, conf, **metadata):
            xy, score = finite_point((x, y)), finite_number(conf)
            if xy is not None and score is not None and 0 <= score <= 1:
                parsed[name] = Keypoint(xy[0], xy[1], score, **metadata)

        if isinstance(raw_pose, dict):
            # 形式为 {"left_shoulder": [x, y, conf], ...} 或 {"left_shoulder": (x, y)}
            for k, v in raw_pose.items():
                if v is None:
                    continue
                name = k.lower()
                if isinstance(v, (list, tuple)):
                    if len(v) >= 3:
                        add(name, v[0], v[1], v[2])
                    elif len(v) == 2:
                        add(name, v[0], v[1], 1.0, confidence_source='legacy_xy_unverified')
                elif hasattr(v, "x") and hasattr(v, "y"):
                    conf = getattr(v, "conf", 1.0)
                    add(name, v.x, v.y, conf, z=getattr(v, 'z', None),
                        observed=getattr(v, 'observed', True),
                        recovered_from_mirror=getattr(v, 'recovered_from_mirror', False),
                        source_frame_id=getattr(v, 'source_frame_id', None),
                        confidence_source=getattr(v, 'confidence_source', 'provided'))
        elif isinstance(raw_pose, (list, np.ndarray)):
            arr = np.array(raw_pose)
            if arr.ndim == 2 and arr.shape[0] == 17 and arr.shape[1] >= 2:
                for idx, name in COCO_KEYPOINTS.items():
                    x, y = arr[idx, 0], arr[idx, 1]
                    conf = arr[idx, 2] if arr.shape[1] > 2 else 1.0
                    add(name, x, y, conf, confidence_source='provided' if arr.shape[1] > 2 else 'legacy_xy_unverified')

        return parsed

    def _measurement_pose(self, pose):
        return {name: point for name, point in pose.items()
                if point.observed is True and not point.recovered_from_mirror
                and point.confidence_source != 'unavailable'
                and finite_point(point.pt) is not None
                and finite_number(point.conf) is not None
                and self.min_keypoint_conf <= point.conf <= 1.}

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

        # 计算双机位身体基准参考（以双肩中心与跨度为尺度进行解剖归一化对齐）
        f_l_sh = front_pose.get("left_shoulder")
        f_r_sh = front_pose.get("right_shoulder")
        b_l_sh = back_pose.get("left_shoulder")
        b_r_sh = back_pose.get("right_shoulder")

        has_front_sh = (
            f_l_sh is not None
            and f_r_sh is not None
            and f_l_sh.conf >= self.min_keypoint_conf
            and f_r_sh.conf >= self.min_keypoint_conf
        )
        has_back_sh = (
            b_l_sh is not None
            and b_r_sh is not None
            and b_l_sh.conf >= self.min_keypoint_conf
            and b_r_sh.conf >= self.min_keypoint_conf
        )

        anchors = (f_l_sh, f_r_sh, b_l_sh, b_r_sh)
        if not (has_front_sh and has_back_sh) or not all(
            kp.observed and not kp.recovered_from_mirror and kp.conf >= .5
            and all(math.isfinite(v) for v in (kp.x, kp.y, kp.conf)) for kp in anchors
        ):
            return healed_pose, healed_list
        frame_ids = {kp.source_frame_id for kp in anchors if kp.source_frame_id is not None}
        if len(frame_ids) > 1:
            return healed_pose, healed_list
        if has_front_sh and has_back_sh:
            f_cx = (f_l_sh.x + f_r_sh.x) / 2.0
            f_cy = (f_l_sh.y + f_r_sh.y) / 2.0
            b_cx = (b_l_sh.x + b_r_sh.x) / 2.0
            b_cy = (b_l_sh.y + b_r_sh.y) / 2.0
            f_w = math.hypot(f_l_sh.x - f_r_sh.x, f_l_sh.y - f_r_sh.y)
            b_w = math.hypot(b_l_sh.x - b_r_sh.x, b_l_sh.y - b_r_sh.y)

            # 躯干垂直轴尺度对齐（抗引拍侧身转肩时双肩投影横向塌陷）
            f_l_hip = front_pose.get("left_hip")
            f_r_hip = front_pose.get("right_hip")
            b_l_hip = back_pose.get("left_hip")
            b_r_hip = back_pose.get("right_hip")
            hip_anchors = (f_l_hip, f_r_hip, b_l_hip, b_r_hip)
            has_hips = all(
                kp is not None and getattr(kp, "observed", True) and not getattr(kp, "recovered_from_mirror", False)
                and getattr(kp, "conf", 0.0) >= 0.5
                and all(math.isfinite(v) for v in (kp.x, kp.y, kp.conf))
                for kp in hip_anchors
            )
            torso_scale = None
            if has_hips:
                hip_frame_ids = {kp.source_frame_id for kp in hip_anchors if kp.source_frame_id is not None}
                if not frame_ids or hip_frame_ids == frame_ids:
                    f_hip_cx = (f_l_hip.x + f_r_hip.x) / 2.0
                    f_hip_cy = (f_l_hip.y + f_r_hip.y) / 2.0
                    b_hip_cx = (b_l_hip.x + b_r_hip.x) / 2.0
                    b_hip_cy = (b_l_hip.y + b_r_hip.y) / 2.0
                    f_torso_h = math.hypot(f_cx - f_hip_cx, f_cy - f_hip_cy)
                    b_torso_h = math.hypot(b_cx - b_hip_cx, b_cy - b_hip_cy)
                    if f_torso_h >= 18 and b_torso_h >= 18:
                        torso_scale = f_torso_h / b_torso_h

            if min(f_w, b_w) < 12:
                if torso_scale is not None:
                    scale = torso_scale
                else:
                    return healed_pose, healed_list
            else:
                scale = f_w / b_w
                if torso_scale is not None and (scale < 0.35 or scale > 2.8):
                    scale = torso_scale
        else:
            scale = 1.0
            f_cx, f_cy, b_cx, b_cy = 0.0, 0.0, 0.0, 0.0

        for joint in critical_joints:
            f_kp = front_pose.get(joint)
            b_kp = back_pose.get(joint)

            f_valid = f_kp is not None and f_kp.conf >= self.min_keypoint_conf
            b_valid = (b_kp is not None and b_kp.conf >= max(.5, self.min_keypoint_conf)
                       and b_kp.observed and not b_kp.recovered_from_mirror
                       and all(math.isfinite(v) for v in (b_kp.x, b_kp.y, b_kp.conf))
                       and (not frame_ids or b_kp.source_frame_id in frame_ids))

            if not f_valid and b_valid:
                # 正面丢失但背面有效：以身体解剖尺度进行归一化映射自愈补全
                if has_front_sh and has_back_sh:
                    mapped_x = f_cx + (b_kp.x - b_cx) * scale
                    mapped_y = f_cy + (b_kp.y - b_cy) * scale
                else:
                    mapped_x = b_kp.x
                    mapped_y = b_kp.y

                healed_pose[joint] = Keypoint(
                    x=round(float(mapped_x), 2),
                    y=round(float(mapped_y), 2),
                    conf=round(float(b_kp.conf * 0.9), 3),  # 适度衰减置信度作为融合标记
                    recovered_from_mirror=True,
                    observed=False, source_frame_id=b_kp.source_frame_id, confidence_source=b_kp.confidence_source,
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
        pose = self._measurement_pose(pose)
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
        is_valid_contact = None
        rejection_reason = None

        if ball_pos is not None:
            is_valid_contact = True
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
        f_pose = self._measurement_pose(self.parse_pose_dict(front_pose_raw))
        b_pose = self._measurement_pose(self.parse_pose_dict(back_pose_raw))

        # 1. 遮挡自愈
        fused_pose, healed_points = self.heal_occluded_pose(f_pose, b_pose)

        # 2. 击球分类与触球校验
        shot_res = self.classify_shot(f_pose, ball_pos=ball_pos)

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
        robust_turn_deg = math.degrees(math.atan2(b_w, f_w)) if f_w > 15 and b_w > 15 else None

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
        takeback_depth_ratio = None
        if b_l_sh and b_r_sh:
            spine_x = (b_l_sh.x + b_r_sh.x) / 2.0
            hitting_wrist_b = b_pose.get("right_wrist") if self.dominant_hand == "right" else b_pose.get("left_wrist")
            if hitting_wrist_b and b_w > 20.0:
                raw_ratio = abs(hitting_wrist_b.x - spine_x) / b_w
                takeback_depth_ratio = min(2.5, max(0.0, raw_ratio))

        # 6. 肩胛收缩度 (Scapular Retraction)
        scapular_ratio = min(3.0, b_w / f_w) if f_w > 15 and b_w > 15 else None

        # 7. 双重视角视差拟合与相对 3D 深度比率推算 (Relative 3D Depth Ratio)
        relative_depth_z = round(float(f_w / max(1.0, b_w)), 3) if (f_w > 15.0 and b_w > 15.0) else None

        return DualViewBiomechanicsResult(
            shot_classification=shot_res,
            front_shoulder_width=round(f_w, 2),
            back_shoulder_width=round(b_w, 2),
            robust_shoulder_turn_deg=round(robust_turn_deg, 2) if robust_turn_deg is not None else None,
            shoulder_hip_separation_deg=round(sep_deg, 2) if sep_deg is not None else None,
            takeback_depth_ratio=round(takeback_depth_ratio, 4) if takeback_depth_ratio is not None else None,
            scapular_retraction_ratio=round(scapular_ratio, 4) if scapular_ratio is not None else None,
            relative_depth_z=relative_depth_z,
            occlusion_healed_points=healed_points,
        )


def map_mirror_racket_to_front(
    mirror_racket_box: Union[List[float], Tuple[float, float, float, float]],
    front_pose: Dict[str, Any],
    back_pose: Dict[str, Any],
) -> Optional[Tuple[float, float, float, float]]:
    """
    引拍阶段球拍镜面互补映射：
    当正面视角的引拍侧肢体/躯干遮挡球拍时，利用背面镜面视点中清晰检出的手持球拍，
    结合镜中空间距离缩放（深度映射）与水平镜像几何反转，高精度映射至正面选手的解剖学持拍位置。

    Args:
        mirror_racket_box: 原图全局坐标系下的镜中球拍包围盒 (x1, y1, x2, y2)
        front_pose: 正面选手关键点字典 (Keypoint 或包含 x, y 的字典/元组)
        back_pose: 镜中背面选手关键点字典 (Keypoint 或包含 x, y 的字典/元组)

    Returns:
        映射至正面选手空间的包围盒 (fx1, fy1, fx2, fy2) 原图像素坐标，若几何基准不足则返回 None
    """
    if not mirror_racket_box or len(mirror_racket_box) < 4:
        return None
    if not front_pose or not back_pose:
        return None

    def _extract_pt(pose: Dict[str, Any], name: str) -> Optional[Tuple[float, float]]:
        val = pose.get(name)
        if val is None:
            return None
        if isinstance(val, (tuple, list)) and len(val) >= 2:
            if len(val) >= 3 and val[2] is not None and val[2] < 0.20:
                return None
            return float(val[0]), float(val[1])
        if hasattr(val, "x") and hasattr(val, "y"):
            conf = getattr(val, "conf", 1.0)
            if conf is not None and conf < 0.20:
                return None
            return float(val.x), float(val.y)
        if isinstance(val, dict) and "x" in val and "y" in val:
            conf = val.get("conf") or val.get("confidence", 1.0)
            if conf is not None and conf < 0.20:
                return None
            return float(val["x"]), float(val["y"])
        return None

    f_l_sh = _extract_pt(front_pose, "left_shoulder")
    f_r_sh = _extract_pt(front_pose, "right_shoulder")
    f_l_hip = _extract_pt(front_pose, "left_hip")
    f_r_hip = _extract_pt(front_pose, "right_hip")

    b_l_sh = _extract_pt(back_pose, "left_shoulder")
    b_r_sh = _extract_pt(back_pose, "right_shoulder")
    b_l_hip = _extract_pt(back_pose, "left_hip")
    b_r_hip = _extract_pt(back_pose, "right_hip")

    # 计算双肩与双髋中点
    f_sh = ((f_l_sh[0] + f_r_sh[0]) / 2.0, (f_l_sh[1] + f_r_sh[1]) / 2.0) if (f_l_sh and f_r_sh) else (f_l_sh or f_r_sh)
    f_hip = ((f_l_hip[0] + f_r_hip[0]) / 2.0, (f_l_hip[1] + f_r_hip[1]) / 2.0) if (f_l_hip and f_r_hip) else (f_l_hip or f_r_hip)

    b_sh = ((b_l_sh[0] + b_r_sh[0]) / 2.0, (b_l_sh[1] + b_r_sh[1]) / 2.0) if (b_l_sh and b_r_sh) else (b_l_sh or b_r_sh)
    b_hip = ((b_l_hip[0] + b_r_hip[0]) / 2.0, (b_l_hip[1] + b_r_hip[1]) / 2.0) if (b_l_hip and b_r_hip) else (b_l_hip or b_r_hip)

    # 躯干中心与垂直高度
    if f_sh is not None and f_hip is not None:
        f_center = ((f_sh[0] + f_hip[0]) / 2.0, (f_sh[1] + f_hip[1]) / 2.0)
        f_torso_h = math.hypot(f_sh[0] - f_hip[0], f_sh[1] - f_hip[1])
    elif f_sh is not None and f_l_sh and f_r_sh:
        f_center = f_sh
        f_torso_h = math.hypot(f_l_sh[0] - f_r_sh[0], f_l_sh[1] - f_r_sh[1]) * 1.35
    elif f_hip is not None and f_l_hip and f_r_hip:
        f_center = f_hip
        f_torso_h = math.hypot(f_l_hip[0] - f_r_hip[0], f_l_hip[1] - f_r_hip[1]) * 1.45
    else:
        return None

    if b_sh is not None and b_hip is not None:
        b_center = ((b_sh[0] + b_hip[0]) / 2.0, (b_sh[1] + b_hip[1]) / 2.0)
        b_torso_h = math.hypot(b_sh[0] - b_hip[0], b_sh[1] - b_hip[1])
    elif b_sh is not None and b_l_sh and b_r_sh:
        b_center = b_sh
        b_torso_h = math.hypot(b_l_sh[0] - b_r_sh[0], b_l_sh[1] - b_r_sh[1]) * 1.35
    elif b_hip is not None and b_l_hip and b_r_hip:
        b_center = b_hip
        b_torso_h = math.hypot(b_l_hip[0] - b_r_hip[0], b_l_hip[1] - b_r_hip[1]) * 1.45
    else:
        return None

    if f_torso_h < 10.0 or b_torso_h < 10.0:
        return None

    # 空间深度尺度比例
    scale = max(0.4, min(3.0, f_torso_h / b_torso_h))

    bx1, by1, bx2, by2 = mirror_racket_box[:4]
    bx_c = (bx1 + bx2) / 2.0
    by_c = (by1 + by2) / 2.0
    bw = max(10.0, bx2 - bx1)
    bh = max(10.0, by2 - by1)

    dx_back = bx_c - b_center[0]
    dy_back = by_c - b_center[1]

    # 镜中空间距离缩放与全局全景几何投影对齐：
    # 原图全景画面为同一视点，平面镜垂直于地面，在原图像素空间中物体的横向偏移矢量方向 (dx) 与其镜中虚像完全同向；
    # 空间偏移向量 (dx_back, dy_back) 经视差尺度缩放后直接映射至正面选手的空间位置：
    dx_front = dx_back * scale
    dy_front = dy_back * scale

    fx_c = f_center[0] + dx_front
    fy_c = f_center[1] + dy_front
    fw = bw * scale
    fh = bh * scale

    # 解剖学合理性距离校验：若正面机位已明确检出手腕，校验映射球拍中心是否在解剖合理范围
    f_r_wrist = _extract_pt(front_pose, "right_wrist")
    f_l_wrist = _extract_pt(front_pose, "left_wrist")
    wrists = [pt for pt in (f_r_wrist, f_l_wrist) if pt is not None]
    if wrists:
        min_wrist_dist = min(math.hypot(fx_c - w[0], fy_c - w[1]) for w in wrists)
        max_allowed_dist = max(180.0, f_torso_h * 1.6)
        if min_wrist_dist > max_allowed_dist:
            return None

    fx1 = fx_c - fw / 2.0
    fy1 = fy_c - fh / 2.0
    fx2 = fx_c + fw / 2.0
    fy2 = fy_c + fh / 2.0

    return (round(float(fx1), 2), round(float(fy1), 2), round(float(fx2), 2), round(float(fy2), 2))
