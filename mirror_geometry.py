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
        hf_mat = np.array(h_f, dtype=np.float64).reshape(3, 3)
        hb_mat = np.array(h_b, dtype=np.float64).reshape(3, 3)
        inv_hf = np.linalg.inv(hf_mat)
        h_comp = inv_hf @ hb_mat
        norm = np.linalg.norm(h_comp)
        if norm > 1e-9:
            h_comp = h_comp / norm
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
        return float(fx), float(fy)

    if ground_homography is not None:
        vec = np.array([bx, by, 1.0], dtype=np.float64)
        mapped = ground_homography @ vec
        if abs(mapped[2]) > 1e-9:
            fx = mapped[0] / mapped[2]
            fy = mapped[1] / mapped[2]
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
