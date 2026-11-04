# -*- coding: utf-8 -*-
"""
dual_view_dataset_miner.py
──────────────────────────
双视图合并与跨视角数据集扩充挖掘引擎 (Dual-View Merging & Dataset Supplementation Engine)：
利用 Front View (实像) 与 Mirror Back View (镜面虚像) 的平面仿射/各向异性躯干几何约束：
1. 找回因单侧手腕漏检而被丢弃的原生 7 点球拍候选框 (Rescued Native 7-Point Racket Candidates)；
2. 跨视角重投影自动生成遮挡态 (v=1) 7 点球拍标签 (Cross-View Occluded v=1 Racket Labels)；
3. 双向补全正背面缺失的手腕/手肘关键点 (Bidirectional Limb Joint Healing)；
4. 挖掘正反重投影分歧难例帧 (Cross-View Disagreement Hard Examples)。

输出的 7 点球拍标签严格遵循 yolo26pose-racket 规范：
0: butt_cap, 1: grip_center, 2: throat, 3: sweet_spot, 4: head_top, 5: frame_left, 6: frame_right
flip_idx: [0, 1, 2, 3, 4, 6, 5]
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from joint_racket_2d_analysis import (
    BODY_POINTS,
    RACKET_POINTS,
    analyze_frames,
     summarize,
)

RACKET_FLIP_IDX = [0, 1, 2, 3, 4, 6, 5]


def format_yolo_pose_racket_label(
    keypoints_xy: Dict[str, Optional[Sequence[float]]],
    image_size: Sequence[int],
    default_visibility: int = 2,
    padding_ratio: float = 0.12,
) -> Optional[str]:
    """
    将像素坐标系下的 7 点球拍关键点转换为标准 YOLO-Pose 归一化标签行：
    `0 <x_c> <y_c> <w> <h> <kpt0_x> <kpt0_y> <v0> ... <kpt6_x> <kpt6_y> <v6>`
    """
    img_w, img_h = float(image_size[0]), float(image_size[1])
    if img_w <= 0 or img_h <= 0:
        return None

    valid_pts: List[Tuple[float, float]] = []
    kpt_tokens: List[str] = []

    for name in RACKET_POINTS:
        pt = keypoints_xy.get(name)
        if (
            pt is not None
            and len(pt) >= 2
            and math.isfinite(float(pt[0]))
            and math.isfinite(float(pt[1]))
        ):
            kx = max(0.0, min(1.0, float(pt[0]) / img_w))
            ky = max(0.0, min(1.0, float(pt[1]) / img_h))
            v = int(pt[2]) if len(pt) >= 3 and pt[2] in (0, 1, 2) else int(default_visibility)
            valid_pts.append((kx, ky))
            kpt_tokens.append(f"{kx:.6f} {ky:.6f} {v}")
        else:
            kpt_tokens.append("0.000000 0.000000 0")

    # 至少需要 4 个有效关键点才能构建可靠球拍训练样本
    if len(valid_pts) < 4:
        return None

    xs = [p[0] for p in valid_pts]
    ys = [p[1] for p in valid_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max(0.01, max_x - min_x)
    span_y = max(0.01, max_y - min_y)
    pad_x = span_x * padding_ratio
    pad_y = span_y * padding_ratio

    x1 = max(0.0, min_x - pad_x)
    y1 = max(0.0, min_y - pad_y)
    x2 = min(1.0, max_x + pad_x)
    y2 = min(1.0, max_y + pad_y)

    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    bw = max(0.005, x2 - x1)
    bh = max(0.005, y2 - y1)

    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} " + " ".join(kpt_tokens)


def mine_session_dual_view_data(
    session_id: str,
    rows: List[Dict[str, Any]],
    image_size: Sequence[int],
    events: Optional[List[Dict[str, Any]]] = None,
    source_video: Optional[str] = None,
    max_pseudo_label_rmse: float = 12.0,
    disagreement_threshold_px: float = 25.0,
) -> Dict[str, Any]:
    """
    在单个 Session 的全量帧上执行双视图合并，并提取 4 类高价值补充数据集记录。
    """
    frames, cadence = analyze_frames(rows, image_size)
    stats = summarize(frames, events or [])
    fusion_summary = stats.get("dual_view_fusion") or {}

    # 建立事件帧索引，便于标记样本所属的击球阶段 (swing_window / contact_window)
    event_Lookup: Dict[int, int] = {}
    contact_lookup: Dict[int, int] = {}
    for ev in events or []:
        eid = int(ev.get("event_id", -1))
        sf = int(ev.get("start_frame", -1))
        ef = int(ev.get("end_frame", -1))
        cf = int(ev.get("contact_frame", -1))
        for f_idx in range(sf, ef + 1):
            event_Lookup[f_idx] = eid
        for f_idx in range(cf - 3, cf + 4):
            contact_lookup[f_idx] = eid

    rescued_racket_samples: List[Dict[str, Any]] = []
    projected_occluded_racket_samples: List[Dict[str, Any]] = []
    healed_body_samples: List[Dict[str, Any]] = []
    disagreement_hard_examples: List[Dict[str, Any]] = []

    for frame in frames:
        fid = frame["frame_id"]
        fusion = frame.get("dual_view_fusion") or {}
        rmse = fusion.get("torso_affine_rmse_px")
        status = fusion.get("torso_transform_status")
        in_event_id = event_Lookup.get(fid)
        near_contact_id = contact_lookup.get(fid)

        # 1. 挖掘找回的原生 7 点球拍候选样本 (v=2) & 跨视角重投影遮挡球拍样本 (v=1)
        for view in ("front", "back"):
            prov = (fusion.get("racket_provenance") or {}).get(view)
            merged_rkt = (fusion.get("merged_racket") or {}).get(view) or {}
            if prov in ("healed_wrist_candidate_rescue", "relaxed_threshold_candidate_rescue"):
                yolo_line = format_yolo_pose_racket_label(merged_rkt, image_size, default_visibility=2)
                if yolo_line:
                    rescued_racket_samples.append({
                        "session_id": session_id,
                        "source_video": source_video,
                        "frame_id": fid,
                        "source_seconds": frame.get("source_seconds"),
                        "view": view,
                        "provenance": prov,
                        "rescued_candidate_index": (fusion.get("rescued_candidate_index") or {}).get(view),
                        "torso_affine_rmse_px": rmse,
                        "event_id": in_event_id,
                        "near_contact_event_id": near_contact_id,
                        "keypoints_xy": merged_rkt,
                        "yolo_pose_label": yolo_line,
                    })
            elif prov in ("back_projected_to_front", "front_projected_to_back"):
                if rmse is not None and rmse <= max_pseudo_label_rmse:
                    # 被遮挡或远景缺失视角使用 v=1 (Labeled but occluded/transferred)
                    yolo_line = format_yolo_pose_racket_label(merged_rkt, image_size, default_visibility=1)
                    if yolo_line:
                        projected_occluded_racket_samples.append({
                            "session_id": session_id,
                            "source_video": source_video,
                            "frame_id": fid,
                            "source_seconds": frame.get("source_seconds"),
                            "target_view": view,
                            "provenance": prov,
                            "torso_affine_rmse_px": rmse,
                            "torso_transform_status": status,
                            "event_id": in_event_id,
                            "near_contact_event_id": near_contact_id,
                            "keypoints_xy": merged_rkt,
                            "yolo_pose_label": yolo_line,
                        })

        # 2. 挖掘正背面双向补全的手腕/手肘姿态样本
        healed_joints_in_frame = {}
        for view in ("front", "back"):
            b_prov = (fusion.get("body_provenance") or {}).get(view) or {}
            m_body = (fusion.get("merged_body") or {}).get(view) or {}
            healed = {
                j: m_body[j]
                for j in ("right_wrist", "left_wrist", "right_elbow", "left_elbow")
                if b_prov.get(j) in ("back_projected_to_front", "front_projected_to_back")
                and m_body.get(j) is not None
            }
            if healed:
                healed_joints_in_frame[view] = healed

        if healed_joints_in_frame and rmse is not None and rmse <= max_pseudo_label_rmse:
            healed_body_samples.append({
                "session_id": session_id,
                "source_video": source_video,
                "frame_id": fid,
                "source_seconds": frame.get("source_seconds"),
                "torso_affine_rmse_px": rmse,
                "torso_transform_status": status,
                "event_id": in_event_id,
                "near_contact_event_id": near_contact_id,
                "healed_views": healed_joints_in_frame,
            })

        # 3. 挖掘跨视角重投影分歧难例 (Hard Examples)
        joint_errs = fusion.get("joint_reprojection_errors_px") or {}
        ss_err = fusion.get("sweet_spot_reprojection_error_px")
        max_j_err = max(joint_errs.values(), default=0.0)
        if max_j_err >= disagreement_threshold_px or (ss_err is not None and ss_err >= disagreement_threshold_px):
            disagreement_hard_examples.append({
                "session_id": session_id,
                "source_video": source_video,
                "frame_id": fid,
                "source_seconds": frame.get("source_seconds"),
                "torso_affine_rmse_px": rmse,
                "joint_reprojection_errors_px": joint_errs,
                "sweet_spot_reprojection_error_px": ss_err,
                "event_id": in_event_id,
                "near_contact_event_id": near_contact_id,
            })

    return {
        "session_id": session_id,
        "total_frames": len(frames),
        "fusion_summary": fusion_summary,
        "counts": {
            "rescued_native_racket_7pt": len(rescued_racket_samples),
            "projected_occluded_racket_v1": len(projected_occluded_racket_samples),
            "healed_body_limb_frames": len(healed_body_samples),
            "disagreement_hard_examples": len(disagreement_hard_examples),
        },
        "samples": {
            "rescued_racket": rescued_racket_samples,
            "projected_occluded_racket": projected_occluded_racket_samples,
            "healed_body": healed_body_samples,
            "hard_examples": disagreement_hard_examples,
        },
    }
