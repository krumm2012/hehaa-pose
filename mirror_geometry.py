"""
mirror_geometry.py
──────────────────
算法 2.0 双视角镜面平面单应性与几何反射变换模型：
利用室内训练仓“正面摄像头 + 后墙平镜”的光学空间关系，
将镜中背面视角的球拍与关键点高精度映射至正面选手解剖坐标空间。

物理光学模型：
1. 镜面反射几何：
   后墙平镜位于地面深度 Y_wall（Court 02 实测 ~6.2m），相机光心位于 Y=0，选手位于 Y_player (~4.68m)。
   在平镜中，选手的虚像位于深度 Y_virtual = 2 * Y_wall - Y_player = 7.72m。
   相机视角下的横向光学放大率理论值为 k_x = Y_virtual / Y_player = 7.72 / 4.68 ≈ 1.6496。
2. 视角俯仰与各向异性尺度：
   由于相机高挂（H_cam ≈ 3.3m）并带有俯角（~35°），像面垂直方向发生透视压缩，
   导致像面纵向尺度 k_y ≈ 1.27 与横向尺度 k_x ≈ 1.65 存在 ~30% 的显著各向异性（Anisotropy）。
   传统的单一体宽/身长标量映射会严重低估横向位移；本模块通过解剖躯干平面仿射（Planar Affine）
   与地面单应性（Ground Homography）显式解耦 k_x 与 k_y，彻底杜绝转体/下蹲时的投影畸变。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class PlanarAffineResult:
    """解剖躯干平面仿射映射解算结果。"""
    matrix: np.ndarray             # 2x3 仿射变换矩阵
    scale_x: float                 # 横向缩放比例 (sx)
    scale_y: float                 # 纵向缩放比例 (sy)
    rotation_deg: float            # 旋转/倾角偏差 (度)
    determinant: float             # 变换行列式 det(A)
    rmse: float                    # 拟合残差 RMSE (像素)
    num_points: int                # 参与拟合的解剖对应点数
    status: str                    # 'optimal' | 'fallback' | 'invalid'


@dataclass
class Torso3DKinematics:
    """3D 躯干动力学绝对旋转与立体力学指标。"""
    shoulder_yaw_deg: Optional[float] = None    # 3D 绝对转肩偏航角 (-180° ~ 180°, 0° = 正面朝向相机, 90° = 侧身引拍)
    shoulder_pitch_deg: Optional[float] = None  # 3D 肩部俯仰角 (前倾 / 后仰)
    shoulder_roll_deg: Optional[float] = None   # 3D 肩部侧倾角 (右肩高/低)
    hip_yaw_deg: Optional[float] = None         # 3D 骨盆绝对偏航角
    hip_pitch_deg: Optional[float] = None       # 3D 骨盆俯仰角
    x_factor_3d_deg: Optional[float] = None     # 真实 3D X-Factor (肩髋空间绝对分离角: |shoulder_yaw - hip_yaw|)
    relative_depth_z: Optional[float] = None    # 虚实视差解算的相对 3D 深度比率
    confidence: float = 0.0                     # 解算置信度
    status: str = "invalid"                     # 'optimal' | 'fallback' | 'invalid'


def extract_keypoint_xy(kp: Any, min_conf: float = 0.20) -> Optional[Tuple[float, float]]:
    """安全提取姿态关键点的 (x, y) 坐标，置信度不足时返回 None。"""
    if kp is None:
        return None
    if isinstance(kp, (list, tuple)) and len(kp) >= 2:
        if len(kp) >= 3 and kp[2] is not None and float(kp[2]) < min_conf:
            return None
        return float(kp[0]), float(kp[1])
    if hasattr(kp, "x") and hasattr(kp, "y"):
        conf = getattr(kp, "conf", 1.0)
        if conf is not None and float(conf) < min_conf:
            return None
        return float(kp.x), float(kp.y)
    if isinstance(kp, dict) and "x" in kp and "y" in kp:
        conf = kp.get("conf") if kp.get("conf") is not None else kp.get("confidence", 1.0)
        if conf is not None and float(conf) < min_conf:
            return None
        return float(kp["x"]), float(kp["y"])
    return None


def estimate_torso_planar_affine(
    front_pose: Dict[str, Any],
    back_pose: Dict[str, Any],
    min_conf: float = 0.20,
) -> Optional[PlanarAffineResult]:
    """
    利用正面与镜中背面姿态的躯干骨骼点（双肩、双髋）估算最优 2D 仿射变换矩阵。

    解算点对：
    - left_shoulder  <-> left_shoulder
    - right_shoulder <-> right_shoulder
    - left_hip       <-> left_hip
    - right_hip      <-> right_hip
    """
    if not front_pose or not back_pose:
        return None

    pts_front = []
    pts_back = []
    landmarks = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")

    for name in landmarks:
        f_pt = extract_keypoint_xy(front_pose.get(name), min_conf=min_conf)
        b_pt = extract_keypoint_xy(back_pose.get(name), min_conf=min_conf)
        if f_pt is not None and b_pt is not None:
            pts_front.append(f_pt)
            pts_back.append(b_pt)

    if len(pts_front) < 3:
        return None

    src_pts = np.array(pts_back, dtype=np.float32)
    dst_pts = np.array(pts_front, dtype=np.float32)

    # 几何退化与近共线防护：点集构成的包围盒不可过于狭窄或退化
    src_w = float(np.ptp(src_pts[:, 0]))
    src_h = float(np.ptp(src_pts[:, 1]))
    if src_w < 5.0 or src_h < 5.0 or (src_w * src_h) < 60.0:
        return None

    # 求解 2x3 仿射变换：dst = M * [src, 1]^T
    M, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None:
        return None

    # 分解尺度与旋转
    a11, a12 = M[0, 0], M[0, 1]
    a21, a22 = M[1, 0], M[1, 1]
    det = a11 * a22 - a12 * a21

    # 物理保真性约束：
    # 1. 在全景同视点平镜反射下，横纵手性保持同向 (det > 0)，绝不可发生手性反转
    # 2. 尺度因子必须处于合理物理区间 [0.35, 3.2]
    if det <= 0.05:
        return None

    scale_x = math.hypot(a11, a12)
    scale_y = math.hypot(a21, a22)
    if not (0.35 <= scale_x <= 3.2 and 0.35 <= scale_y <= 3.2):
        return None

    # 仿射变换条件数检查，防止局部极端各向异性畸变
    try:
        s_vals = np.linalg.svd(M[:2, :2], compute_uv=False)
        if s_vals[-1] < 1e-4 or (s_vals[0] / s_vals[-1]) > 20.0:
            return None
    except Exception:
        return None

    rotation_rad = math.atan2(a21, a11)
    rotation_deg = math.degrees(rotation_rad)

    # 计算对应点重投影残差 RMSE
    transformed = (M @ np.hstack([src_pts, np.ones((len(src_pts), 1))]).T).T
    errors = np.linalg.norm(transformed - dst_pts, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return PlanarAffineResult(
        matrix=M,
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        rotation_deg=float(rotation_deg),
        determinant=float(det),
        rmse=round(rmse, 2),
        num_points=len(pts_front),
        status="optimal",
    )


def extract_ground_mirror_homography(
    ground_calibration: Optional[Dict[str, Any]]
) -> Optional[np.ndarray]:
    """
    从地面标定元数据中构建地面诱导镜面单应性矩阵 H_mirror_to_front = H_front^-1 @ H_back。
    """
    if not ground_calibration or not isinstance(ground_calibration, dict):
        return None
    views = ground_calibration.get("views") or {}
    front_view = views.get("front") or {}
    back_view = views.get("back") or {}
    h_f = front_view.get("H")
    h_b = back_view.get("H")
    if not h_f or not h_b:
        return None

    try:
        from ground_reference import validate_homography_matrix
        v_f, _, _, _ = validate_homography_matrix(h_f)
        v_b, _, _, _ = validate_homography_matrix(h_b)
        if not (v_f and v_b):
            return None

        hf_mat = np.array(h_f, dtype=np.float64).reshape(3, 3)
        hb_mat = np.array(h_b, dtype=np.float64).reshape(3, 3)
        inv_hf = np.linalg.inv(hf_mat)
        h_comp = inv_hf @ hb_mat
        norm = np.linalg.norm(h_comp)
        if norm > 1e-9:
            h_comp = h_comp / norm

        v_c, _, _, _ = validate_homography_matrix(h_comp)
        if not v_c:
            return None
        return h_comp
    except Exception:
        return None


def map_mirror_point_planar(
    point: Tuple[float, float],
    affine_result: Optional[PlanarAffineResult] = None,
    ground_homography: Optional[np.ndarray] = None,
) -> Optional[Tuple[float, float]]:
    """
    将镜中单点 (bx, by) 经平面反射单应性/仿射矩阵映射回正面像面坐标。
    优先使用解剖躯干仿射矩阵（贴近选手自身空间平面）；
    次选地面单应性矩阵。
    """
    bx, by = float(point[0]), float(point[1])

    if affine_result is not None and affine_result.matrix is not None:
        M = affine_result.matrix
        fx = M[0, 0] * bx + M[0, 1] * by + M[0, 2]
        fy = M[1, 0] * bx + M[1, 1] * by + M[1, 2]
        if -2000.0 <= fx <= 8000.0 and -2000.0 <= fy <= 8000.0:
            return float(fx), float(fy)
        return None

    if ground_homography is not None:
        vec = np.array([bx, by, 1.0], dtype=np.float64)
        mapped = ground_homography @ vec
        if abs(mapped[2]) >= 1e-4:
            fx = mapped[0] / mapped[2]
            fy = mapped[1] / mapped[2]
            if -2000.0 <= fx <= 8000.0 and -2000.0 <= fy <= 8000.0:
                return float(fx), float(fy)

    return None


def map_mirror_box_planar(
    mirror_racket_box: Union[List[float], Tuple[float, float, float, float]],
    front_pose: Dict[str, Any],
    back_pose: Dict[str, Any],
    ground_calibration: Optional[Dict[str, Any]] = None,
    method: str = "planar_affine",  # 'planar_affine' | 'ground_homography' | 'torso_scale'
    min_kp_conf: float = 0.20,
) -> Tuple[Optional[Tuple[float, float, float, float]], Dict[str, Any]]:
    """
    核心入口：执行镜中球拍的高级平面反射单应性投影。

    Args:
        mirror_racket_box: 原图全局坐标系下的镜中球拍包围盒 (bx1, by1, bx2, by2)
        front_pose: 正面选手关键点字典
        back_pose: 镜中背面选手关键点字典
        ground_calibration: 可选的地面标定字典 (含 views.front.H 与 views.back.H)
        method: 期望映射算法
        min_kp_conf: 关键点有效置信度阈值

    Returns:
        (mapped_box, diagnostics)
    """
    diagnostics: Dict[str, Any] = {
        "method_requested": method,
        "method_used": "none",
        "scale_x": 1.0,
        "scale_y": 1.0,
        "affine_rmse": None,
        "rotation_deg": None,
    }

    if not mirror_racket_box or len(mirror_racket_box) < 4:
        return None, diagnostics
    if not front_pose or not back_pose:
        return None, diagnostics

    bx1, by1, bx2, by2 = [float(v) for v in mirror_racket_box[:4]]
    bx_c = (bx1 + bx2) / 2.0
    by_c = (by1 + by2) / 2.0
    bw = max(10.0, bx2 - bx1)
    bh = max(10.0, by2 - by1)

    # 1. 尝试解剖躯干仿射变换 (Planar Affine)
    affine_res = None
    if method in ("planar_affine", "auto"):
        affine_res = estimate_torso_planar_affine(front_pose, back_pose, min_conf=min_kp_conf)

    # 2. 尝试地面单应性复合矩阵 (Ground Homography)
    ground_h = None
    if method in ("ground_homography", "auto") and ground_calibration:
        ground_h = extract_ground_mirror_homography(ground_calibration)

    mapped_center = None
    scale_x, scale_y = 1.0, 1.0

    if affine_res is not None:
        mapped_center = map_mirror_point_planar((bx_c, by_c), affine_result=affine_res)
        scale_x = affine_res.scale_x
        scale_y = affine_res.scale_y
        diagnostics.update({
            "method_used": "planar_affine",
            "scale_x": round(scale_x, 3),
            "scale_y": round(scale_y, 3),
            "affine_rmse": affine_res.rmse,
            "rotation_deg": round(affine_res.rotation_deg, 2),
            "affine_num_points": affine_res.num_points,
        })
    elif ground_h is not None and method == "ground_homography":
        mapped_center = map_mirror_point_planar((bx_c, by_c), ground_homography=ground_h)
        # 用原点附近的局部梯度估计尺度
        p0 = map_mirror_point_planar((bx_c, by_c), ground_homography=ground_h)
        px = map_mirror_point_planar((bx_c + 10.0, by_c), ground_homography=ground_h)
        py = map_mirror_point_planar((bx_c, by_c + 10.0), ground_homography=ground_h)
        if p0 and px and py:
            scale_x = math.hypot(px[0] - p0[0], px[1] - p0[1]) / 10.0
            scale_y = math.hypot(py[0] - p0[0], py[1] - p0[1]) / 10.0
        else:
            scale_x, scale_y = 1.65, 1.27
        diagnostics.update({
            "method_used": "ground_homography",
            "scale_x": round(scale_x, 3),
            "scale_y": round(scale_y, 3),
        })

    if mapped_center is None:
        return None, diagnostics

    fx_c, fy_c = mapped_center
    fw = bw * scale_x
    fh = bh * scale_y

    # 解剖学合理性门控：若正面检出手腕，校验映射球拍中心距手腕是否在生理允许范围
    f_rwrist = extract_keypoint_xy(front_pose.get("right_wrist"), min_conf=0.15)
    f_lwrist = extract_keypoint_xy(front_pose.get("left_wrist"), min_conf=0.15)
    wrists = [pt for pt in (f_rwrist, f_lwrist) if pt is not None]
    if wrists:
        min_wrist_d = min(math.hypot(fx_c - w[0], fy_c - w[1]) for w in wrists)
        diagnostics["wrist_distance_px"] = round(min_wrist_d, 1)
        # 允许最大生理距离为 190px（正常网球拍全长约 68cm，握把到拍框中心约 40-50cm，像面 ~80-120px）
        if min_wrist_d > 190.0:
            diagnostics["rejection_reason"] = "exceeds_anatomical_wrist_proximity"
            return None, diagnostics

    fx1 = fx_c - fw / 2.0
    fy1 = fy_c - fh / 2.0
    fx2 = fx_c + fw / 2.0
    fy2 = fy_c + fh / 2.0

    mapped_box = (round(float(fx1), 2), round(float(fy1), 2), round(float(fx2), 2), round(float(fy2), 2))
    return mapped_box, diagnostics


def estimate_torso_3d_kinematics(
    front_pose: Dict[str, Any],
    back_pose: Dict[str, Any],
    k_x: float = 1.65,
    k_y: float = 1.27,
    min_conf: float = 0.20,
) -> Torso3DKinematics:
    """
    基于正面机位与后墙镜面虚拟机位的双重视角视差几何，
    解算人体躯干在 3D 欧氏坐标系下的绝对旋转动力学指标：
    1. 转肩偏航角 (Shoulder Yaw)
    2. 肩部俯仰角 (Shoulder Pitch)
    3. 肩部侧倾角 (Shoulder Roll)
    4. 骨盆偏航角 (Hip Yaw)
    5. 真实 3D X-Factor (肩髋三维绝对分离角)
    6. 虚实视差相对深度 Z (Relative Depth Z)
    """
    if not front_pose or not back_pose:
        return Torso3DKinematics(status="invalid")

    # 提取正面与背面肩部关键点
    f_l_sh = extract_keypoint_xy(front_pose.get("left_shoulder"), min_conf=min_conf)
    f_r_sh = extract_keypoint_xy(front_pose.get("right_shoulder"), min_conf=min_conf)
    b_l_sh = extract_keypoint_xy(back_pose.get("left_shoulder"), min_conf=min_conf)
    b_r_sh = extract_keypoint_xy(back_pose.get("right_shoulder"), min_conf=min_conf)

    # 提取正面与背面髋部关键点
    f_l_hp = extract_keypoint_xy(front_pose.get("left_hip"), min_conf=min_conf)
    f_r_hp = extract_keypoint_xy(front_pose.get("right_hip"), min_conf=min_conf)
    b_l_hp = extract_keypoint_xy(back_pose.get("left_hip"), min_conf=min_conf)
    b_r_hp = extract_keypoint_xy(back_pose.get("right_hip"), min_conf=min_conf)

    confs = []
    for p_dict in (front_pose, back_pose):
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            v = p_dict.get(name)
            if v is not None:
                c = getattr(v, "conf", None) or (v.get("conf") if isinstance(v, dict) else (v[2] if isinstance(v, (list, tuple)) and len(v) >= 3 else None))
                if c is not None:
                    confs.append(float(c))
    mean_conf = round(float(np.mean(confs)), 3) if confs else 0.50

    has_front_sh = f_l_sh is not None and f_r_sh is not None
    has_back_sh = b_l_sh is not None and b_r_sh is not None
    has_front_hp = f_l_hp is not None and f_r_hp is not None
    has_back_hp = b_l_hp is not None and b_r_hp is not None

    if not has_front_sh:
        return Torso3DKinematics(confidence=mean_conf, status="invalid")

    # 1. 前视角肩部投影
    dx_sf = f_r_sh[0] - f_l_sh[0]
    dy_sf = f_r_sh[1] - f_l_sh[1]
    w_sf = math.hypot(dx_sf, dy_sf)
    if w_sf < 5.0:
        return Torso3DKinematics(confidence=mean_conf, status="invalid")

    # 肩部侧倾角 (Roll)
    shoulder_roll_deg = round(math.degrees(math.atan2(dy_sf, max(1e-4, abs(dx_sf)))), 2)

    # 2. 结合背部视角计算 3D 转肩偏航角 (Yaw)
    shoulder_yaw_deg = None
    rel_depth_z = None
    if has_back_sh:
        dx_sb = b_r_sh[0] - b_l_sh[0]
        dy_sb = b_r_sh[1] - b_l_sh[1]
        w_sb = math.hypot(dx_sb, dy_sb)
        w_sb_norm = w_sb * k_x

        w_max = max(w_sf, w_sb_norm, 15.0)
        x_ratio = max(-1.0, min(1.0, dx_sf / w_max))
        # 深度梯度：背部相对展开程度
        depth_gradient = (w_sb_norm - w_sf) / w_max
        y_sign = 1.0 if depth_gradient >= -0.05 else -1.0
        y_ratio = y_sign * math.sqrt(max(0.0, 1.0 - x_ratio ** 2))

        yaw_rad = math.atan2(y_ratio, x_ratio)
        shoulder_yaw_deg = round(math.degrees(yaw_rad), 2)
        rel_depth_z = round(float(w_sf / max(1.0, w_sb)), 3) if w_sb > 0 else None
    else:
        # 仅有单机位时的启发式降级
        shoulder_yaw_deg = round(math.degrees(math.acos(max(-1.0, min(1.0, dx_sf / max(w_sf, 15.0))))), 2)

    # 3. 计算 3D 骨盆偏航角 (Hip Yaw)
    hip_yaw_deg = None
    if has_front_hp:
        dx_hf = f_r_hp[0] - f_l_hp[0]
        dy_hf = f_r_hp[1] - f_l_hp[1]
        w_hf = math.hypot(dx_hf, dy_hf)
        if has_back_hp and w_hf >= 5.0:
            dx_hb = b_r_hp[0] - b_l_hp[0]
            dy_hb = b_r_hp[1] - b_l_hp[1]
            w_hb = math.hypot(dx_hb, dy_hb)
            w_hb_norm = w_hb * k_x
            w_hmax = max(w_hf, w_hb_norm, 15.0)
            hx_ratio = max(-1.0, min(1.0, dx_hf / w_hmax))
            h_gradient = (w_hb_norm - w_hf) / w_hmax
            hy_sign = 1.0 if h_gradient >= -0.05 else -1.0
            hy_ratio = hy_sign * math.sqrt(max(0.0, 1.0 - hx_ratio ** 2))
            hip_yaw_deg = round(math.degrees(math.atan2(hy_ratio, hx_ratio)), 2)
        elif w_hf >= 5.0:
            hip_yaw_deg = round(math.degrees(math.acos(max(-1.0, min(1.0, dx_hf / max(w_hf, 15.0))))), 2)

    # 4. 躯干俯仰角 (Pitch)
    shoulder_pitch_deg = None
    if has_front_hp:
        mid_sh = ((f_l_sh[0] + f_r_sh[0]) / 2.0, (f_l_sh[1] + f_r_sh[1]) / 2.0)
        mid_hp = ((f_l_hp[0] + f_r_hp[0]) / 2.0, (f_l_hp[1] + f_r_hp[1]) / 2.0)
        trunk_dx = mid_sh[0] - mid_hp[0]
        trunk_dy = mid_hp[1] - mid_sh[1]  # 向上为正
        if trunk_dy > 10.0:
            pitch_rad = math.atan2(trunk_dx, trunk_dy)
            shoulder_pitch_deg = round(math.degrees(pitch_rad), 2)

    # 5. 真实 3D X-Factor (肩髋三维空间分离角)
    x_factor_3d_deg = None
    if shoulder_yaw_deg is not None and hip_yaw_deg is not None:
        diff = (shoulder_yaw_deg - hip_yaw_deg + 180.0) % 360.0 - 180.0
        x_factor_3d_deg = round(abs(diff), 2)

    status = "optimal" if (has_back_sh and has_back_hp) else "fallback"

    return Torso3DKinematics(
        shoulder_yaw_deg=shoulder_yaw_deg,
        shoulder_pitch_deg=shoulder_pitch_deg,
        shoulder_roll_deg=shoulder_roll_deg,
        hip_yaw_deg=hip_yaw_deg,
        x_factor_3d_deg=x_factor_3d_deg,
        relative_depth_z=rel_depth_z,
        confidence=mean_conf,
        status=status,
    )

