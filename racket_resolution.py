"""
Racket Resolution and Dual-View Mirror Recovery Module.

Decouples racket detection filtering, mirror-view candidate association,
wrist anatomical proximity gating, and temporal decay smoothing from the
main pipeline inference loop into a pure, testable, configurable module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from dual_view_biomechanics import map_mirror_racket_to_front


@dataclass
class RacketResolutionConfig:
    """Configuration for racket candidate filtering and dual-view recovery."""

    # 镜面区域回退边界 (x_min, y_min, x_max, y_max)
    mirror_fallback_zone: Tuple[float, float, float, float] = (680.0, 0.0, 1580.0, 620.0)
    # 镜中背面球拍候选与后背人体关键点的最大欧氏距离门限 (px)
    back_racket_dist_threshold: float = 250.0
    # 正面机位持拍手腕/手肘的欧氏距离合理门限 (px)
    wrist_proximity_threshold: float = 420.0
    # 镜面区域内被认定为前景真实球拍的严格手腕距离门限 (px, 超过则判定为镜中虚像而非前景直接持拍)
    mirror_overlap_wrist_threshold: float = 180.0
    # 镜面区域外无手腕强约束时的距离惩罚 (px)
    mirror_penalty_dist: float = 200.0
    # 挥拍遮挡/模糊允许的最大时序自愈帧数
    max_missing_smooth_frames: int = 2
    # 时序自愈衰减置信度
    decay_confidence: float = 0.5
    # 镜中补偿球拍置信度缩放系数
    recovered_confidence_scale: float = 0.85
    # 关键点有效性置信度阈值
    min_kp_conf_wrist: float = 0.20
    min_kp_conf_body: float = 0.20

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "RacketResolutionConfig":
        if not d:
            return cls()
        sub = d.get("racket_resolution", d)
        zone = sub.get("mirror_fallback_zone", (680.0, 0.0, 1580.0, 620.0))
        if isinstance(zone, (list, tuple)) and len(zone) == 4:
            zone_tup = (float(zone[0]), float(zone[1]), float(zone[2]), float(zone[3]))
        else:
            zone_tup = (680.0, 0.0, 1580.0, 620.0)

        return cls(
            mirror_fallback_zone=zone_tup,
            back_racket_dist_threshold=float(sub.get("back_racket_dist_threshold", 250.0)),
            wrist_proximity_threshold=float(sub.get("wrist_proximity_threshold", 420.0)),
            mirror_overlap_wrist_threshold=float(sub.get("mirror_overlap_wrist_threshold", 180.0)),
            mirror_penalty_dist=float(sub.get("mirror_penalty_dist", 200.0)),
            max_missing_smooth_frames=int(sub.get("max_missing_smooth_frames", 2)),
            decay_confidence=float(sub.get("decay_confidence", 0.5)),
            recovered_confidence_scale=float(sub.get("recovered_confidence_scale", 0.85)),
            min_kp_conf_wrist=float(sub.get("min_kp_conf_wrist", 0.20)),
            min_kp_conf_body=float(sub.get("min_kp_conf_body", 0.20)),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RacketResolutionConfig":
        p = Path(path)
        if not p.is_file():
            return cls()
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)
        except Exception:
            return cls()


@dataclass
class RacketResolutionResult:
    """Output bundle for resolved racket in a single frame."""

    rackets: List[Dict[str, Any]]
    racket_box: Optional[List[float]]
    is_racket_recovered: bool
    back_racket_box: Optional[List[float]]
    status: str  # 'observed' | 'recovered' | 'decayed' | 'missing'
    missing_count: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class RacketResolver:
    """Stateful racket resolver maintaining temporal smoothing across frames."""

    def __init__(self, config: Optional[RacketResolutionConfig] = None):
        self.config = config or RacketResolutionConfig()
        self.last_racket_box: Optional[List[float]] = None
        self.last_racket_entry: Optional[Dict[str, Any]] = None
        self.racket_missing_count: int = 0

    def reset(self) -> None:
        self.last_racket_box = None
        self.last_racket_entry = None
        self.racket_missing_count = 0

    def is_in_mirror(
        self,
        rx_c: float,
        ry_c: float,
        frame_w: int,
        frame_h: int,
        dual_view_mgr: Optional[Any] = None,
    ) -> bool:
        if dual_view_mgr is not None and hasattr(dual_view_mgr, "is_point_in_mirror"):
            return bool(dual_view_mgr.is_point_in_mirror(rx_c, ry_c, frame_w, frame_h))
        x_min, y_min, x_max, y_max = self.config.mirror_fallback_zone
        return (x_min <= rx_c <= x_max) and (y_min <= ry_c <= y_max)

    def resolve(
        self,
        raw_rackets: Optional[List[Dict[str, Any]]],
        front_pose: Optional[Dict[str, Any]] = None,
        back_pose: Optional[Dict[str, Any]] = None,
        fused_pose: Optional[Dict[str, Any]] = None,
        frame_idx: int = 0,
        source_time: Optional[Dict[str, Any]] = None,
        frame_size: Tuple[int, int] = (1920, 1080),
        dual_view_mgr: Optional[Any] = None,
        racket_tracker: Optional[Any] = None,
        detector: Optional[Any] = None,
        base_diagnostics: Optional[Dict[str, Any]] = None,
        pose_res: Optional[Any] = None,
    ) -> RacketResolutionResult:
        """
        核心解算方法：综合正面模型检测、时序跟踪、镜中虚像关联与衰减平滑。
        """
        if pose_res is not None:
            if front_pose is None:
                front_pose = getattr(pose_res, "front_pose_orig", None)
            if back_pose is None:
                back_pose = getattr(pose_res, "back_pose_orig", None)
            if fused_pose is None:
                fused_pose = getattr(pose_res, "fused_pose_orig", None)

        front_pose = front_pose or {}
        back_pose = back_pose or {}
        fused_pose = fused_pose or {}

        racket_diagnostics = dict(base_diagnostics or {})
        frame_w, frame_h = frame_size

        # ---------------------------------------------------------
        # A. 优先识别镜中背面选手持拍候选 (Back View Mirror Racket)
        # ---------------------------------------------------------
        back_racket_entry = None
        back_racket_box: Optional[List[float]] = None

        back_body_pts = [
            kp for name, kp in back_pose.items()
            if ("wrist" in name or "elbow" in name or "shoulder" in name or "hip" in name)
            and getattr(kp, "conf", 0.0) >= self.config.min_kp_conf_body
        ]

        for r in (raw_rackets or []):
            r_box = r.get("box")
            if not r_box:
                continue
            rx_c = (r_box[0] + r_box[2]) / 2.0
            ry_c = (r_box[1] + r_box[3]) / 2.0
            in_mirror = self.is_in_mirror(rx_c, ry_c, frame_w, frame_h, dual_view_mgr)

            if in_mirror:
                if back_body_pts:
                    dist_b = min(
                        ((rx_c - getattr(w, "x", w[0] if isinstance(w, (list, tuple)) else 0.0)) ** 2
                         + (ry_c - getattr(w, "y", w[1] if isinstance(w, (list, tuple)) else 0.0)) ** 2) ** 0.5
                        for w in back_body_pts
                    )
                    if dist_b <= self.config.back_racket_dist_threshold:
                        if back_racket_entry is None or dist_b < back_racket_entry[0]:
                            back_racket_entry = (dist_b, r)
                elif back_racket_entry is None:
                    back_racket_entry = (999.0, r)

        if back_racket_entry is not None:
            back_racket_box = list(back_racket_entry[1]["box"])

        # ---------------------------------------------------------
        # B. 前景候选送入 RacketTemporalTracker 进行时序连续性初选
        # ---------------------------------------------------------
        if racket_tracker is not None:
            tracker_points = {
                name: {
                    "x": getattr(kp, "x", kp[0] if isinstance(kp, (list, tuple)) else 0.0),
                    "y": getattr(kp, "y", kp[1] if isinstance(kp, (list, tuple)) else 0.0),
                    "confidence": getattr(kp, "conf", 1.0),
                    "observed": getattr(kp, "observed", True),
                    "source_frame_id": getattr(kp, "source_frame_id", frame_idx),
                    "recovered_from_mirror": getattr(kp, "recovered_from_mirror", False),
                    "confidence_source": getattr(kp, "confidence_source", "model"),
                }
                for name, kp in front_pose.items()
            }
            selected, temporal_diagnostics = racket_tracker.select(
                raw_rackets or [],
                tracker_points,
                frame_idx,
                source_time,
                (frame_w, frame_h),
            )
            candidates_to_eval = [selected] if selected else []
            racket_diagnostics["temporal_tracking"] = temporal_diagnostics
            if detector is not None:
                racket_diagnostics["model_candidates"] = dict(
                    (getattr(detector, "last_parse_diagnostics", {}) or {}).get("racket") or {}
                )
        else:
            candidates_to_eval = list(raw_rackets or [])

        # ---------------------------------------------------------
        # C. 关联手腕位置并防误判后墙纯镜面虚影
        # ---------------------------------------------------------
        wrists = [
            kp for name, kp in front_pose.items()
            if "wrist" in name and getattr(kp, "conf", 0.0) >= self.config.min_kp_conf_wrist
        ]
        if not wrists:
            wrists = [
                kp for name, kp in front_pose.items()
                if "elbow" in name and getattr(kp, "conf", 0.0) >= self.config.min_kp_conf_wrist
            ]

        valid_rackets = []
        for r in candidates_to_eval:
            r_box = r.get("box")
            if not r_box:
                continue
            rx_c = (r_box[0] + r_box[2]) / 2.0
            ry_c = (r_box[1] + r_box[3]) / 2.0
            in_mirror = self.is_in_mirror(rx_c, ry_c, frame_w, frame_h, dual_view_mgr)

            if wrists:
                dist = min(
                    ((rx_c - getattr(w, "x", w[0] if isinstance(w, (list, tuple)) else 0.0)) ** 2
                     + (ry_c - getattr(w, "y", w[1] if isinstance(w, (list, tuple)) else 0.0)) ** 2) ** 0.5
                    for w in wrists
                )
                if in_mirror:
                    # 落在镜面区域内部：若作为前景真实持拍，必须极其靠近手腕（选手遮挡镜面）
                    # 否则判定为镜中虚像（由后背机位处理），绝不可误当成正面持拍
                    if dist <= self.config.mirror_overlap_wrist_threshold:
                        valid_rackets.append((dist, r))
                else:
                    if dist <= self.config.wrist_proximity_threshold:
                        valid_rackets.append((dist, r))
                    else:
                        valid_rackets.append((dist + self.config.mirror_penalty_dist, r))
            else:
                if not in_mirror:
                    valid_rackets.append((0.0, r))

        # ---------------------------------------------------------
        # D. 状态机仲裁与镜中自愈补偿 (State Machine Arbitration)
        # ---------------------------------------------------------
        is_racket_recovered = False
        status = "missing"

        if valid_rackets:
            valid_rackets.sort(key=lambda x: x[0])
            racket = [{**item[1], "observed": True, "source_frame_id": frame_idx} for item in valid_rackets]
            racket_box = valid_rackets[0][1].get("box")
            self.last_racket_box = racket_box
            self.last_racket_entry = racket[0]
            self.racket_missing_count = 0
            is_racket_recovered = False
            status = "observed"

        elif back_racket_box is not None:
            # 正面引拍躯干遮挡球拍：利用镜中背面球拍进行空间投影补偿自愈
            mapped_front_box = map_mirror_racket_to_front(
                back_racket_box,
                front_pose or fused_pose,
                back_pose,
            )
            if mapped_front_box is not None:
                base_conf = float(back_racket_entry[1].get("confidence", 0.6)) if back_racket_entry else 0.6
                recovered_racket = {
                    "box": list(mapped_front_box),
                    "confidence": round(base_conf * self.config.recovered_confidence_scale, 3),
                    "observed": False,
                    "recovered_from_mirror": True,
                    "source_frame_id": frame_idx,
                    "source_mirror_box": list(back_racket_box),
                }
                racket = [recovered_racket]
                racket_box = list(mapped_front_box)
                self.last_racket_box = racket_box
                self.last_racket_entry = racket[0]
                self.racket_missing_count = 0
                is_racket_recovered = True
                status = "recovered"
            elif (
                self.last_racket_box is not None
                and self.racket_missing_count < self.config.max_missing_smooth_frames
                and wrists
            ):
                self.racket_missing_count += 1
                racket_box = list(self.last_racket_box)
                decayed = dict(self.last_racket_entry) if self.last_racket_entry else {"box": racket_box, "confidence": self.config.decay_confidence}
                decayed["observed"] = False
                racket = [decayed]
                status = "decayed"
            else:
                self.racket_missing_count += 1
                racket = []
                racket_box = None
                status = "missing"

        elif (
            self.last_racket_box is not None
            and self.racket_missing_count < self.config.max_missing_smooth_frames
            and wrists
        ):
            # 挥拍动作模糊短时平滑自愈（最多保持 max_missing_smooth_frames 帧）
            self.racket_missing_count += 1
            racket_box = list(self.last_racket_box)
            decayed = dict(self.last_racket_entry) if self.last_racket_entry else {"box": racket_box, "confidence": self.config.decay_confidence}
            decayed["observed"] = False
            racket = [decayed]
            status = "decayed"

        else:
            self.racket_missing_count += 1
            racket = []
            racket_box = None
            status = "missing"

        racket_diagnostics["resolution_status"] = status
        racket_diagnostics["is_racket_recovered"] = is_racket_recovered
        racket_diagnostics["missing_count"] = self.racket_missing_count
        racket_diagnostics["back_racket_box"] = back_racket_box

        return RacketResolutionResult(
            rackets=racket,
            racket_box=racket_box,
            is_racket_recovered=is_racket_recovered,
            back_racket_box=back_racket_box,
            status=status,
            missing_count=self.racket_missing_count,
            diagnostics=racket_diagnostics,
        )
