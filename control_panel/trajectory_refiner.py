"""Per-swing trajectory gating, PCHIP interpolation, Savitzky-Golay smoothing, and contact re-alignment."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    PchipInterpolator = None  # type: ignore[assignment]
    savgol_filter = None  # type: ignore[assignment]
    _HAS_SCIPY = False


_KP_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

MAX_PLAUSIBLE_SMOOTH_SPEED_PX_S = 6000.0


def _angle_3pt(a: Any, b: Any, c: Any) -> float:
    if a is None or b is None or c is None:
        return float("nan")
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    nba = float(np.linalg.norm(ba))
    nbc = float(np.linalg.norm(bc))
    if nba < 1e-6 or nbc < 1e-6:
        return float("nan")
    cos_v = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_v)))


def _pchip_fill_short_gaps(t: np.ndarray, y: np.ndarray, max_gap: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Fill NaN gaps of length <= max_gap using shape-preserving PCHIP (or linear fallback)."""
    arr = np.asarray(y, dtype=float).copy()
    t_arr = np.asarray(t, dtype=float)
    valid = np.isfinite(arr)
    interp_mask = np.zeros(len(arr), dtype=bool)
    if int(valid.sum()) < 2:
        return arr, interp_mask

    if _HAS_SCIPY and PchipInterpolator is not None:
        interp_fn = PchipInterpolator(t_arr[valid], arr[valid], extrapolate=False)
    else:
        t_valid = t_arr[valid]
        y_valid = arr[valid]
        interp_fn = lambda q: np.interp(q, t_valid, y_valid)  # noqa: E731

    n = len(arr)
    i = 0
    while i < n:
        if not valid[i]:
            j = i
            while j < n and not valid[j]:
                j += 1
            gap_len = j - i
            if i > 0 and j < n and gap_len <= max_gap:
                arr[i:j] = interp_fn(t_arr[i:j])
                interp_mask[i:j] = True
            i = j
        else:
            i += 1
    return arr, interp_mask


def _smooth_valid_segments(y: np.ndarray, win: int = 5, poly: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay filter on contiguous finite segments."""
    arr = np.asarray(y, dtype=float).copy()
    out = arr.copy()
    n = len(arr)
    i = 0
    while i < n:
        if np.isfinite(arr[i]):
            j = i
            while j < n and np.isfinite(arr[j]):
                j += 1
            seg_len = j - i
            w = min(win, seg_len if seg_len % 2 == 1 else seg_len - 1)
            if _HAS_SCIPY and savgol_filter is not None and w >= poly + 2 and w >= 3:
                out[i:j] = savgol_filter(arr[i:j], window_length=w, polyorder=poly)
            elif w >= 3:
                kernel = np.ones(w, dtype=float) / float(w)
                pad = w // 2
                padded = np.pad(arr[i:j], (pad, pad), mode="edge")
                out[i:j] = np.convolve(padded, kernel, mode="valid")
            i = j
        else:
            i += 1
    return out


def _hampel_reject_2d(pts: np.ndarray, half_win: int = 3, n_sigmas: float = 3.2, min_tol_px: float = 95.0) -> np.ndarray:
    """Reject 1-2 frame consecutive teleport outliers using sliding-window median + MAD."""
    n = len(pts)
    rej = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(pts[i]).all():
            continue
        lo = max(0, i - half_win)
        hi = min(n, i + half_win + 1)
        win_pts = pts[lo:hi]
        valid = np.isfinite(win_pts).all(axis=1)
        if int(valid.sum()) < 3:
            continue
        v_pts = win_pts[valid]
        med = np.median(v_pts, axis=0)
        dists = np.linalg.norm(v_pts - med, axis=1)
        mad = float(np.median(dists))
        sigma = 1.4826 * mad
        dist_i = float(np.linalg.norm(pts[i] - med))
        if dist_i > max(min_tol_px, n_sigmas * sigma):
            rej[i] = True
    return rej


def _json_num_list(arr: np.ndarray, ndigits: int = 1, cap: Optional[float] = None) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for v in arr:
        if v is None or not np.isfinite(v):
            out.append(None)
        else:
            val = float(v)
            if cap is not None and val > cap:
                val = cap
            out.append(round(val, ndigits))
    return out


def _round_opt(v: Any, ndigits: int = 1) -> Optional[float]:
    if v is None:
        return None
    try:
        fv = float(v)
        if not math.isfinite(fv):
            return None
        return round(fv, ndigits)
    except (TypeError, ValueError):
        return None


class SessionTrajectoryCache:
    """Caches parsed per-frame light summaries from *_frames.jsonl for fast swing diagnostics."""

    def __init__(self, max_sessions: int = 6):
        self._lock = threading.Lock()
        self._max_sessions = max_sessions
        # session_dir_str -> (sig, {frame_id: light_frame_dict})
        self._cache: Dict[str, Tuple[Tuple[int, int], Dict[int, Dict[str, Any]]]] = {}
        self._order: List[str] = []

    def load_frames_index(self, session_dir: Path) -> Dict[int, Dict[str, Any]]:
        jsonl_files = sorted(session_dir.glob("*_frames.jsonl"))
        if not jsonl_files:
            return {}
        target = jsonl_files[0]
        try:
            st = target.stat()
            sig = (int(st.st_mtime_ns), int(st.st_size))
        except OSError:
            return {}

        key = str(target.resolve())
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and cached[0] == sig:
                return cached[1]

        index: Dict[int, Dict[str, Any]] = {}
        try:
            with open(target, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    fid = d.get("frame_id")
                    if fid is None:
                        continue
                    fid_int = int(fid)
                    pose_raw = d.get("pose") or {}
                    pose_light = {
                        k: pose_raw[k][:2]
                        for k in _KP_NAMES
                        if isinstance(pose_raw.get(k), (list, tuple))
                        and len(pose_raw[k]) >= 2
                        and pose_raw[k][0] is not None
                        and pose_raw[k][1] is not None
                    }
                    ball_raw = d.get("ball")
                    ball_pt = (
                        [float(ball_raw[0]), float(ball_raw[1])]
                        if isinstance(ball_raw, (list, tuple))
                        and len(ball_raw) >= 2
                        and ball_raw[0] is not None
                        and ball_raw[1] is not None
                        else None
                    )
                    rkt_box = None
                    rkts = d.get("rackets") or []
                    if rkts and isinstance(rkts[0], dict) and rkts[0].get("box"):
                        b = rkts[0]["box"]
                        if len(b) >= 4:
                            rkt_box = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                    index[fid_int] = {
                        "ts": float(d.get("timestamp") if d.get("timestamp") is not None else fid_int / 25.0),
                        "pose": pose_light,
                        "ball": ball_pt,
                        "rkt_box": rkt_box,
                    }
        except OSError:
            return {}

        with self._lock:
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            self._cache[key] = (sig, index)
            while len(self._order) > self._max_sessions:
                oldest = self._order.pop(0)
                self._cache.pop(oldest, None)
        return index


def refine_swing_trajectory(
    event_row: Dict[str, Any],
    frames_index: Dict[int, Dict[str, Any]],
    *,
    dominant_hand: str = "right",
    include_series: bool = True,
) -> Dict[str, Any]:
    """Fit smooth per-frame curves and compute confidence-tiered repaired metrics for one swing."""
    eid = int(event_row["event_id"])
    sf = int(event_row.get("start_frame") or 0)
    cf = int(event_row.get("contact_frame") if event_row.get("contact_frame") is not None else sf)
    ef = int(event_row.get("end_frame") if event_row.get("end_frame") is not None else max(sf, cf))
    stroke_type = str(event_row.get("stroke_type") or "")

    payload = event_row.get("payload")
    if payload is None and event_row.get("payload_json"):
        try:
            payload = json.loads(event_row["payload_json"])
        except Exception:
            payload = {}
    payload = payload or {}

    raw_metrics = {
        m["key"]: m.get("value")
        for m in (payload.get("analysis_metrics") or [])
        if isinstance(m, dict) and m.get("key")
    }
    contact_status = str(payload.get("contact_status") or "unknown")
    ball_win_ratio = float((payload.get("quality_flags") or {}).get("ball_contact_window_ratio") or 0.0)

    hitting_side = "left" if str(dominant_hand or "right").strip().lower().startswith("l") else "right"
    sh_key = f"{hitting_side}_shoulder"
    el_key = f"{hitting_side}_elbow"
    wr_key = f"{hitting_side}_wrist"

    w_start = max(0, min(sf, cf - 22))
    w_end = max(ef, cf + 16)
    fids = [fid for fid in range(w_start, w_end + 1) if fid in frames_index]
    n = len(fids)

    if n < 3:
        return {
            "available": False,
            "reason": "该会话缺少逐帧轨迹文件 (*_frames.jsonl) 或帧数不足",
            "event_id": eid,
            "session_id": event_row.get("session_id"),
            "stroke_type": stroke_type,
            "tier": "B",
            "weight": 0.5,
            "old_contact_frame": cf,
            "refined_contact_frame": cf,
            "shift_frames": 0,
            "rejected_count": 0,
            "interpolated_count": 0,
            "summary": {
                "arm_extension": {"raw": _round_opt(raw_metrics.get("arm_extension")), "refined": _round_opt(raw_metrics.get("arm_extension"))},
                "preparation_knee_flexion": {"raw": _round_opt(raw_metrics.get("preparation_knee_flexion")), "refined": _round_opt(raw_metrics.get("preparation_knee_flexion"))},
                "shoulder_turn_change": {"raw": _round_opt(raw_metrics.get("shoulder_turn_change")), "refined": _round_opt(raw_metrics.get("shoulder_turn_change"))},
                "racket_max_speed": {"raw": _round_opt(raw_metrics.get("racket_max_speed"), 0), "refined": _round_opt(event_row.get("racket_max_speed_px_s"), 0)},
            },
        }

    ts = np.array([float(frames_index[fid]["ts"]) for fid in fids], dtype=float)
    for k in range(1, n):
        if ts[k] <= ts[k - 1]:
            ts[k] = ts[k - 1] + 0.04

    raw_kps = {k: np.full((n, 2), np.nan) for k in _KP_NAMES}
    raw_ball = np.full((n, 2), np.nan)
    raw_rkt = np.full((n, 2), np.nan)
    raw_rkt_area = np.full(n, np.nan)

    for i, fid in enumerate(fids):
        fr = frames_index[fid]
        pose = fr["pose"]
        for k in _KP_NAMES:
            pt = pose.get(k)
            if pt is not None:
                raw_kps[k][i] = [float(pt[0]), float(pt[1])]
        if fr["ball"] is not None:
            raw_ball[i] = fr["ball"]
        if fr["rkt_box"] is not None:
            b = fr["rkt_box"]
            raw_rkt[i] = [0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])]
            raw_rkt_area[i] = abs((b[2] - b[0]) * (b[3] - b[1]))

    # 1. Smooth body keypoints
    sm_kps: Dict[str, np.ndarray] = {}
    for k in _KP_NAMES:
        xf, _ = _pchip_fill_short_gaps(ts, raw_kps[k][:, 0], max_gap=3)
        yf, _ = _pchip_fill_short_gaps(ts, raw_kps[k][:, 1], max_gap=3)
        sm_kps[k] = np.column_stack([_smooth_valid_segments(xf, 5, 2), _smooth_valid_segments(yf, 5, 2)])

    sh_mid = 0.5 * (sm_kps["left_shoulder"] + sm_kps["right_shoulder"])
    hip_mid = 0.5 * (sm_kps["left_hip"] + sm_kps["right_hip"])
    torso_dists = np.linalg.norm(sh_mid - hip_mid, axis=1)
    torso_l = float(np.nanmedian(torso_dists)) if np.isfinite(torso_dists).any() else 130.0

    # Raw & smoothed angles
    raw_arm = np.array([
        _angle_3pt(
            raw_kps[sh_key][i] if np.isfinite(raw_kps[sh_key][i]).all() else None,
            raw_kps[el_key][i] if np.isfinite(raw_kps[el_key][i]).all() else None,
            raw_kps[wr_key][i] if np.isfinite(raw_kps[wr_key][i]).all() else None,
        )
        for i in range(n)
    ])
    sm_arm = np.array([
        _angle_3pt(
            sm_kps[sh_key][i] if np.isfinite(sm_kps[sh_key][i]).all() else None,
            sm_kps[el_key][i] if np.isfinite(sm_kps[el_key][i]).all() else None,
            sm_kps[wr_key][i] if np.isfinite(sm_kps[wr_key][i]).all() else None,
        )
        for i in range(n)
    ])

    def _knee_series(kp_dict: Dict[str, np.ndarray]) -> np.ndarray:
        out = np.full(n, np.nan)
        for i in range(n):
            angs = []
            for side in ("left", "right"):
                h, kn, an = kp_dict[f"{side}_hip"][i], kp_dict[f"{side}_knee"][i], kp_dict[f"{side}_ankle"][i]
                if np.isfinite(h).all() and np.isfinite(kn).all() and np.isfinite(an).all():
                    a = _angle_3pt(h, kn, an)
                    if np.isfinite(a):
                        angs.append(a)
            if angs:
                out[i] = 180.0 - float(np.mean(angs))
        return out

    raw_knee = _knee_series(raw_kps)
    sm_knee = _knee_series(sm_kps)

    raw_sh_vec = raw_kps["right_shoulder"] - raw_kps["left_shoulder"]
    raw_sh_ang = np.degrees(np.arctan2(raw_sh_vec[:, 1], raw_sh_vec[:, 0]))
    sm_sh_vec = sm_kps["right_shoulder"] - sm_kps["left_shoulder"]
    sm_sh_ang = np.degrees(np.arctan2(sm_sh_vec[:, 1], sm_sh_vec[:, 0]))

    # 2. Racket raw speed & multi-stage physical gating + Hampel MAD + PCHIP + SG
    raw_rkt_spd = np.full(n, np.nan)
    for i in range(1, n):
        if np.isfinite(raw_rkt[i]).all() and np.isfinite(raw_rkt[i - 1]).all():
            dt = ts[i] - ts[i - 1]
            if dt > 1e-4:
                raw_rkt_spd[i] = float(np.linalg.norm(raw_rkt[i] - raw_rkt[i - 1]) / dt)

    gated_rkt = raw_rkt.copy()
    rejected_mask = np.zeros(n, dtype=bool)
    med_area = float(np.nanmedian(raw_rkt_area)) if np.isfinite(raw_rkt_area).any() else 12000.0
    max_wrist_dist = 1.8 * torso_l

    for i in range(n):
        if not np.isfinite(gated_rkt[i]).all():
            continue
        wr = sm_kps[wr_key][i]
        too_far = np.isfinite(wr).all() and float(np.linalg.norm(gated_rkt[i] - wr)) > max_wrist_dist
        area_explode = np.isfinite(raw_rkt_area[i]) and raw_rkt_area[i] > max(3.5 * med_area, 45000.0)
        if too_far or area_explode:
            gated_rkt[i] = [np.nan, np.nan]
            rejected_mask[i] = True

    # Hampel MAD 2D rejection (catches 1-2 frame consecutive mirror/shoe glitches)
    hampel_rej = _hampel_reject_2d(gated_rkt, half_win=3, n_sigmas=3.2, min_tol_px=110.0)
    gated_rkt[hampel_rej] = [np.nan, np.nan]
    rejected_mask |= hampel_rej

    # Single-frame teleport jump rejection (>200 px/frame)
    valid_idxs = [i for i in range(n) if np.isfinite(gated_rkt[i]).all()]
    for pos, i in enumerate(valid_idxs):
        d_prev = float(np.linalg.norm(gated_rkt[i] - gated_rkt[valid_idxs[pos - 1]])) / max(1, i - valid_idxs[pos - 1]) if pos > 0 else 0.0
        d_next = float(np.linalg.norm(gated_rkt[valid_idxs[pos + 1]] - gated_rkt[i])) / max(1, valid_idxs[pos + 1] - i) if pos + 1 < len(valid_idxs) else 0.0
        if (d_prev > 200.0 and d_next > 200.0) or (pos == 0 and d_next > 240.0) or (pos == len(valid_idxs) - 1 and d_prev > 240.0):
            gated_rkt[i] = [np.nan, np.nan]
            rejected_mask[i] = True

    rx_f, interp_x = _pchip_fill_short_gaps(ts, gated_rkt[:, 0], max_gap=4)
    ry_f, interp_y = _pchip_fill_short_gaps(ts, gated_rkt[:, 1], max_gap=4)
    interp_mask = interp_x | interp_y
    rx_s = _smooth_valid_segments(rx_f, 5, 2)
    ry_s = _smooth_valid_segments(ry_f, 5, 2)

    sm_rkt_spd = np.full(n, np.nan)
    for i in range(1, n - 1):
        if np.isfinite(rx_s[i - 1]) and np.isfinite(rx_s[i + 1]):
            dt = ts[i + 1] - ts[i - 1]
            if dt > 1e-4:
                spd_val = float(math.hypot(rx_s[i + 1] - rx_s[i - 1], ry_s[i + 1] - ry_s[i - 1]) / dt)
                if spd_val <= MAX_PLAUSIBLE_SMOOTH_SPEED_PX_S:
                    sm_rkt_spd[i] = spd_val

    # Wrist speed for fallback contact anchoring
    wr_spd = np.full(n, np.nan)
    for i in range(1, n - 1):
        if np.isfinite(sm_kps[wr_key][i - 1]).all() and np.isfinite(sm_kps[wr_key][i + 1]).all():
            dt = ts[i + 1] - ts[i - 1]
            if dt > 1e-4:
                wr_spd[i] = float(np.linalg.norm(sm_kps[wr_key][i + 1] - sm_kps[wr_key][i - 1]) / dt)

    # 3. Guarded Multi-Cue Contact Re-Alignment
    cf_idx = fids.index(cf) if cf in fids else min(n - 1, max(0, int(np.argmin(np.abs(np.asarray(fids) - cf)))))
    search_lo = max(1, cf_idx - 3)
    search_hi = min(n - 2, cf_idx + 5)

    best_ball_idx: Optional[int] = None
    best_ball_dist = 1e9
    for i in range(search_lo, search_hi + 1):
        if np.isfinite(raw_ball[i]).all() and np.isfinite(sm_kps[wr_key][i]).all():
            d = float(np.linalg.norm(raw_ball[i] - sm_kps[wr_key][i]))
            if d < best_ball_dist and d < 2.6 * torso_l:
                best_ball_dist = d
                best_ball_idx = i

    wr_slice = wr_spd[search_lo : search_hi + 1]
    wr_peak_idx = int(search_lo + int(np.nanargmax(wr_slice))) if np.isfinite(wr_slice).any() else cf_idx

    if best_ball_idx is not None:
        # Arm-extension guard: do not drag an already-aligned arm-extension peak into follow-through fold
        arm_at_orig = sm_arm[cf_idx]
        arm_at_cand = sm_arm[best_ball_idx]
        if (
            best_ball_idx > cf_idx
            and np.isfinite(arm_at_orig)
            and np.isfinite(arm_at_cand)
            and arm_at_orig >= 155.0
            and arm_at_cand < arm_at_orig - 18.0
        ):
            refined_cf_idx = cf_idx
            anchor_cue = "contact_arm_peak_guarded"
        else:
            refined_cf_idx = best_ball_idx
            anchor_cue = "ball_proximity"
    else:
        refined_cf_idx = wr_peak_idx
        anchor_cue = "wrist_peak"
    refined_cf = fids[refined_cf_idx]

    # 4. Confidence tier
    local_rej = int(rejected_mask[max(0, refined_cf_idx - 4) : min(n, refined_cf_idx + 5)].sum())
    if contact_status == "unknown" and ball_win_ratio < 0.25 and best_ball_idx is None:
        tier = "C"
        weight = 0.0
    elif best_ball_idx is not None and local_rej <= 1:
        tier = "A"
        weight = 1.0
    else:
        tier = "B"
        weight = 0.5

    # 5. Fixed pre-contact window metrics
    c_win = sm_arm[max(0, refined_cf_idx - 1) : min(n, refined_cf_idx + 2)]
    ref_arm = float(np.nanmedian(c_win)) if np.isfinite(c_win).any() else float("nan")

    k_lo = max(0, refined_cf_idx - 15)
    k_hi = max(k_lo + 1, refined_cf_idx - 1)
    k_win = sm_knee[k_lo:k_hi]
    ref_knee = float(np.nanpercentile(k_win[np.isfinite(k_win)], 90)) if np.isfinite(k_win).any() else float("nan")

    t_lo = max(0, refined_cf_idx - 20)
    t_hi = min(n, refined_cf_idx + 1)
    t_win = sm_sh_ang[t_lo:t_hi]
    ref_turn = float(np.nanmax(t_win) - np.nanmin(t_win)) if int(np.isfinite(t_win).sum()) >= 2 else float("nan")

    s_lo = max(0, refined_cf_idx - 6)
    s_hi = min(n, refined_cf_idx + 7)
    s_win = sm_rkt_spd[s_lo:s_hi]
    ref_spd = float(np.nanmax(s_win)) if np.isfinite(s_win).any() else float("nan")

    result: Dict[str, Any] = {
        "available": True,
        "event_id": eid,
        "session_id": event_row.get("session_id"),
        "display_time": event_row.get("display_time"),
        "stroke_type": stroke_type,
        "tier": tier,
        "weight": weight,
        "anchor_cue": anchor_cue,
        "old_contact_frame": cf,
        "refined_contact_frame": refined_cf,
        "shift_frames": int(refined_cf - cf),
        "prep_window_frames": [int(fids[k_lo]), int(fids[min(n - 1, k_hi)])],
        "rejected_count": int(rejected_mask.sum()),
        "interpolated_count": int(interp_mask.sum()),
        "summary": {
            "arm_extension": {
                "raw": _round_opt(raw_metrics.get("arm_extension")),
                "refined": _round_opt(ref_arm),
            },
            "preparation_knee_flexion": {
                "raw": _round_opt(raw_metrics.get("preparation_knee_flexion")),
                "refined": _round_opt(ref_knee),
            },
            "shoulder_turn_change": {
                "raw": _round_opt(raw_metrics.get("shoulder_turn_change")),
                "refined": _round_opt(ref_turn),
            },
            "racket_max_speed": {
                "raw": _round_opt(raw_metrics.get("racket_max_speed"), 0),
                "refined": _round_opt(ref_spd, 0),
            },
        },
    }

    if include_series:
        result["series"] = {
            "frame_ids": [int(x) for x in fids],
            "raw_racket_speed": _json_num_list(raw_rkt_spd, 0, cap=6500.0),
            "smooth_racket_speed": _json_num_list(sm_rkt_spd, 0),
            "rejected_mask": [bool(x) for x in rejected_mask],
            "interp_mask": [bool(x) for x in interp_mask],
            "raw_arm": _json_num_list(raw_arm, 1),
            "smooth_arm": _json_num_list(sm_arm, 1),
            "raw_knee": _json_num_list(raw_knee, 1),
            "smooth_knee": _json_num_list(sm_knee, 1),
            "raw_shoulder": _json_num_list(raw_sh_ang, 1),
            "smooth_shoulder": _json_num_list(sm_sh_ang, 1),
        }

    return result
