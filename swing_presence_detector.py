#!/usr/bin/env python3
"""
独立视频挥拍检测器
用于快速筛选训练视频：判定视频是否包含挥拍动作。
"""

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

SWING_LABELS = {"Forehand", "Backhand", "Two-Handed Backhand"}


def load_config(config_path: str) -> Dict:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dominant_hand(config: Dict) -> str:
    swing_cfg = config.get("swing_analysis", {})
    return swing_cfg.get("dominant_hand", config.get("dominant_hand", "right"))


def _build_label_runs(labels: List[str]) -> List[Tuple[int, int, str]]:
    if not labels:
        return []
    runs: List[Tuple[int, int, str]] = []
    start = 0
    cur = labels[0]
    for idx in range(1, len(labels)):
        if labels[idx] != cur:
            runs.append((start, idx - 1, cur))
            start = idx
            cur = labels[idx]
    runs.append((start, len(labels) - 1, cur))
    return runs


def _smooth_short_label_runs(labels: List[str], min_stable_run: int) -> List[str]:
    if len(labels) < 3:
        return list(labels)
    stable_min = max(1, int(min_stable_run))
    smoothed = list(labels)
    while True:
        changed = False
        runs = _build_label_runs(smoothed)
        for idx, (start, end, run_label) in enumerate(runs):
            run_len = end - start + 1
            if run_len >= stable_min:
                continue
            if idx == 0 or idx == len(runs) - 1:
                continue
            prev_label = runs[idx - 1][2]
            next_label = runs[idx + 1][2]
            if prev_label == next_label and prev_label != run_label:
                for pos in range(start, end + 1):
                    smoothed[pos] = prev_label
                changed = True
        if not changed:
            break
    return smoothed


def _majority_vote_labels(labels: List[str], window: int) -> List[str]:
    if not labels:
        return []
    w = max(1, int(window))
    if w <= 1:
        return list(labels)
    if w % 2 == 0:
        w += 1
    half = w // 2
    voted = []
    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        cnt: Dict[str, int] = {}
        for x in labels[lo:hi]:
            cnt[x] = cnt.get(x, 0) + 1
        voted.append(max(cnt.items(), key=lambda kv: kv[1])[0])
    return voted


def detect_swing_presence(
    events: List[Dict],
    min_swing_frames: int = 12,
    min_consecutive: int = 3,
    min_motion_px: float = 6.0,
    max_motion_px: float = 120.0,
    min_event_frames: int = 3,
    max_break_frames: int = 1,
    type_switch_min_frames: int = 2,
    label_vote_window: int = 5,
    entry_motion_px: Optional[float] = None,
    exit_motion_px: Optional[float] = None,
    min_peak_motion_px: Optional[float] = None,
    min_core_frames: int = 2,
    min_rearm_frames: int = 4,
    post_event_cooldown_frames: int = 6,
    motion_smooth_window: int = 3,
    entry_confirm_frames: int = 2,
    exit_confirm_frames: int = 2,
    max_rearm_hold_frames: int = 14,
    enable_event_two_handed_override: bool = False,
    two_handed_event_min_frames: int = 3,
    two_handed_event_ratio: float = 0.22,
    return_frame_trace: bool = False,
) -> Tuple[bool, Dict]:
    swing_frame_count = 0
    qualified_swing_frames = 0
    consecutive = 0
    max_consecutive = 0
    swing_type_counts = {k: 0 for k in sorted(SWING_LABELS)}
    swing_events: List[Dict[str, Any]] = []
    swing_event_type_counts = {k: 0 for k in sorted(SWING_LABELS)}
    frame_trace = [] if return_frame_trace else None

    entry_thr = (
        float(entry_motion_px)
        if entry_motion_px is not None
        else float(min_motion_px) + 1.5
    )
    exit_thr = (
        float(exit_motion_px)
        if exit_motion_px is not None
        else max(0.0, float(min_motion_px) * 0.7)
    )
    peak_thr = (
        float(min_peak_motion_px)
        if min_peak_motion_px is not None
        else max(entry_thr + 1.0, float(min_motion_px) + 2.0)
    )

    raw_labels = [str(e.get("label", "No Pose")) for e in events]
    frame_to_raw_label = {
        int(e.get("frame", idx)): str(e.get("label", "No Pose"))
        for idx, e in enumerate(events)
    }
    voted_labels = _majority_vote_labels(raw_labels, label_vote_window)
    stable_labels = _smooth_short_label_runs(voted_labels, type_switch_min_frames)

    frame_to_event: Dict[int, int] = {}
    frame_to_stable_label: Dict[int, str] = {}
    in_event = False
    event_start = -1
    event_end = -1
    event_peak_motion = 0.0
    event_qualified_frames = 0
    event_core_frames = 0
    event_label_counts = {k: 0 for k in sorted(SWING_LABELS)}
    event_frames: List[int] = []
    break_count = 0
    cooldown_left = 0
    rearm_low_count = int(min_rearm_frames)
    rearm_hold_count = 0
    armed = True
    entry_streak = 0
    exit_streak = 0
    entry_buffer: List[Dict[str, Any]] = []

    smooth_w = max(1, int(motion_smooth_window))
    motion_hist: List[float] = []
    smooth_motion_by_idx: List[float] = []
    for event in events:
        m = float(event.get("motion_px", 0.0))
        motion_hist.append(m)
        tail = motion_hist[-smooth_w:]
        smooth_motion_by_idx.append(sorted(tail)[len(tail) // 2])

    def _finalize_event() -> bool:
        nonlocal in_event, event_start, event_end, event_peak_motion
        nonlocal event_qualified_frames, event_core_frames, event_label_counts, event_frames, break_count
        nonlocal cooldown_left, armed, rearm_low_count, rearm_hold_count
        accepted = False
        if not in_event:
            return accepted
        if (
            event_qualified_frames >= int(min_event_frames)
            and event_core_frames >= int(min_core_frames)
            and event_peak_motion >= float(peak_thr)
        ):
            event_raw_label_counts = {k: 0 for k in sorted(SWING_LABELS)}
            for fr in event_frames:
                raw = frame_to_raw_label.get(int(fr))
                if raw in SWING_LABELS:
                    event_raw_label_counts[raw] += 1

            dominant_type = max(event_label_counts.items(), key=lambda kv: kv[1])[0]
            bh_frames = event_raw_label_counts.get("Backhand", 0)
            twoh_frames = event_raw_label_counts.get("Two-Handed Backhand", 0)
            twoh_ratio = (
                float(twoh_frames) / float(max(1, twoh_frames + bh_frames))
                if (twoh_frames + bh_frames) > 0
                else 0.0
            )
            if (
                bool(enable_event_two_handed_override)
                and
                twoh_frames >= int(two_handed_event_min_frames)
                and twoh_ratio >= float(two_handed_event_ratio)
            ):
                dominant_type = "Two-Handed Backhand"

            swing_events.append(
                {
                    "start_frame": int(event_start),
                    "end_frame": int(event_end),
                    "qualified_frames": int(event_qualified_frames),
                    "core_frames": int(event_core_frames),
                    "peak_motion_px": round(float(event_peak_motion), 4),
                    "dominant_type": dominant_type,
                    "type_counts": {k: int(v) for k, v in event_label_counts.items()},
                    "raw_type_counts": {k: int(v) for k, v in event_raw_label_counts.items()},
                }
            )
            event_id = len(swing_events)
            for frame_id in event_frames:
                frame_to_event[frame_id] = event_id
            swing_event_type_counts[dominant_type] += 1
            cooldown_left = int(post_event_cooldown_frames)
            armed = False
            rearm_low_count = 0
            rearm_hold_count = 0
            accepted = True

        in_event = False
        event_start = -1
        event_end = -1
        event_peak_motion = 0.0
        event_qualified_frames = 0
        event_core_frames = 0
        event_label_counts = {k: 0 for k in sorted(SWING_LABELS)}
        event_frames = []
        break_count = 0
        return accepted

    for idx, event in enumerate(events):
        frame_id = int(event.get("frame", -1))
        raw_label = str(event.get("label", "No Pose"))
        label = stable_labels[idx]
        motion_px = float(event.get("motion_px", 0.0))
        smooth_motion_px = float(smooth_motion_by_idx[idx])
        frame_to_stable_label[frame_id] = label
        is_swing = label in SWING_LABELS
        is_raw_swing = raw_label in SWING_LABELS
        if is_raw_swing:
            swing_frame_count += 1

        in_motion_range = smooth_motion_px >= min_motion_px and smooth_motion_px <= max_motion_px
        is_qualified = is_swing and in_motion_range
        if is_qualified:
            qualified_swing_frames += 1
            consecutive += 1
            swing_type_counts[label] += 1
            if consecutive > max_consecutive:
                max_consecutive = consecutive
        else:
            consecutive = 0

        low_motion = (not is_swing) or smooth_motion_px < exit_thr or smooth_motion_px > max_motion_px
        if low_motion:
            rearm_low_count += 1
        else:
            rearm_low_count = 0

        if not in_event and not armed:
            rearm_hold_count += 1
        elif armed:
            rearm_hold_count = 0

        if not armed and cooldown_left <= 0:
            if rearm_low_count >= int(min_rearm_frames) or rearm_hold_count >= int(max_rearm_hold_frames):
                armed = True
                rearm_hold_count = 0
        if cooldown_left > 0:
            cooldown_left -= 1

        decision = "none"
        if in_event:
            if is_qualified:
                event_end = frame_id
                event_qualified_frames += 1
                event_peak_motion = max(event_peak_motion, smooth_motion_px)
                if smooth_motion_px >= peak_thr:
                    event_core_frames += 1
                event_label_counts[label] += 1
                event_frames.append(frame_id)
                break_count = 0
                exit_streak = 0
                decision = "event_keep"
            else:
                break_count += 1
                exit_streak += 1
                decision = "event_break"
                if break_count > int(max_break_frames) and exit_streak >= int(exit_confirm_frames):
                    accepted = _finalize_event()
                    decision = "event_close_accept" if accepted else "event_close_reject"
                    exit_streak = 0
        else:
            if armed and is_qualified and smooth_motion_px >= entry_thr:
                entry_streak += 1
                entry_buffer.append(
                    {
                        "frame": frame_id,
                        "label": label,
                        "motion_px": smooth_motion_px,
                    }
                )
                decision = "entry_candidate"
            else:
                entry_streak = 0
                entry_buffer = []
            if entry_streak >= int(entry_confirm_frames):
                start_record = entry_buffer[0]
                in_event = True
                event_start = int(start_record["frame"])
                event_end = frame_id
                event_qualified_frames = len(entry_buffer)
                event_peak_motion = max(float(r["motion_px"]) for r in entry_buffer)
                event_core_frames = sum(1 for r in entry_buffer if float(r["motion_px"]) >= peak_thr)
                for r in entry_buffer:
                    event_label_counts[str(r["label"])] += 1
                event_frames = [int(r["frame"]) for r in entry_buffer]
                break_count = 0
                exit_streak = 0
                entry_streak = 0
                entry_buffer = []
                decision = "event_open"

        if frame_trace is not None:
            phase = "idle"
            if in_event:
                if event_core_frames > 0 and smooth_motion_px >= peak_thr:
                    phase = "core"
                elif event_core_frames > 0:
                    phase = "follow_through"
                else:
                    phase = "entry"
            elif not armed and cooldown_left > 0:
                phase = "cooldown"
            elif not armed:
                phase = "rearm"
            frame_trace.append(
                {
                    "frame": frame_id,
                    "raw_label": raw_label,
                    "label": label,
                    "motion_px": round(motion_px, 4),
                    "motion_smooth_px": round(smooth_motion_px, 4),
                    "is_swing_label": bool(is_swing),
                    "in_motion_range": bool(in_motion_range),
                    "qualified": bool(is_qualified),
                    "consecutive_qualified": int(consecutive),
                    "armed": bool(armed),
                    "cooldown_left": int(max(0, cooldown_left)),
                    "in_event": bool(in_event),
                    "rearm_hold_count": int(rearm_hold_count),
                    "phase": phase,
                    "decision": decision,
                }
            )

    _finalize_event()

    has_swing = (
        qualified_swing_frames >= int(min_swing_frames)
        and max_consecutive >= int(min_consecutive)
    )
    stats = {
        "swing_frame_count": swing_frame_count,
        "qualified_swing_frames": qualified_swing_frames,
        "max_consecutive_qualified": max_consecutive,
        "swing_type_counts": swing_type_counts,
        "swing_event_count": len(swing_events),
        "swing_events": swing_events,
        "swing_event_type_counts": swing_event_type_counts,
        "thresholds": {
            "min_swing_frames": int(min_swing_frames),
            "min_consecutive": int(min_consecutive),
            "min_motion_px": float(min_motion_px),
            "max_motion_px": float(max_motion_px),
            "min_event_frames": int(min_event_frames),
            "max_break_frames": int(max_break_frames),
            "type_switch_min_frames": int(type_switch_min_frames),
            "label_vote_window": int(max(1, label_vote_window)),
            "entry_motion_px": float(entry_thr),
            "exit_motion_px": float(exit_thr),
            "min_peak_motion_px": float(peak_thr),
            "min_core_frames": int(min_core_frames),
            "min_rearm_frames": int(min_rearm_frames),
            "post_event_cooldown_frames": int(post_event_cooldown_frames),
            "motion_smooth_window": int(smooth_w),
            "entry_confirm_frames": int(entry_confirm_frames),
            "exit_confirm_frames": int(exit_confirm_frames),
            "max_rearm_hold_frames": int(max_rearm_hold_frames),
            "enable_event_two_handed_override": bool(enable_event_two_handed_override),
            "two_handed_event_min_frames": int(two_handed_event_min_frames),
            "two_handed_event_ratio": float(two_handed_event_ratio),
        },
    }
    if frame_trace is not None:
        for row in frame_trace:
            frame_id = int(row["frame"])
            row["event_id"] = frame_to_event.get(frame_id)
            row["stable_label"] = frame_to_stable_label.get(frame_id)
        stats["frame_trace"] = frame_trace
    return has_swing, stats


def _extract_dominant_wrist(
    pose_results: List[Dict],
    dominant_hand: str,
) -> Optional[Tuple[float, float]]:
    if not pose_results:
        return None
    first = pose_results[0]
    if not isinstance(first, dict):
        return None
    key = "right_wrist" if dominant_hand == "right" else "left_wrist"
    pt = first.get(key)
    if not pt:
        return None
    return float(pt[0]), float(pt[1])


def _infer_two_handed_backhand_from_pose(
    pose_results: List[Dict],
    dominant_hand: str,
    mirror_view: bool,
    wrist_distance_threshold_px: float,
) -> bool:
    if not pose_results:
        return False
    first = pose_results[0]
    if not isinstance(first, dict):
        return False
    lw = first.get("left_wrist")
    rw = first.get("right_wrist")
    ls = first.get("left_shoulder")
    rs = first.get("right_shoulder")
    if not all([lw, rw, ls, rs]):
        return False

    dx = float(lw[0]) - float(rw[0])
    dy = float(lw[1]) - float(rw[1])
    wrist_dist = float((dx * dx + dy * dy) ** 0.5)
    if wrist_dist > float(wrist_distance_threshold_px):
        return False

    body_center_x = (float(ls[0]) + float(rs[0])) / 2.0
    avg_wrist_x = (float(lw[0]) + float(rw[0])) / 2.0
    if not mirror_view:
        is_backhand_side = (
            (dominant_hand == "right" and avg_wrist_x < body_center_x)
            or (dominant_hand == "left" and avg_wrist_x > body_center_x)
        )
    else:
        is_backhand_side = (
            (dominant_hand == "right" and avg_wrist_x > body_center_x)
            or (dominant_hand == "left" and avg_wrist_x < body_center_x)
        )
    return bool(is_backhand_side)


def process_video_for_swing_presence(
    video_path: str,
    pose_module,
    config: Dict,
    sample_every_n_frames: int = 2,
    min_swing_frames: int = 12,
    min_consecutive: int = 3,
    min_motion_px: float = 6.0,
    max_motion_px: float = 120.0,
    min_event_frames: int = 3,
    max_break_frames: int = 1,
    type_switch_min_frames: int = 2,
    label_vote_window: int = 5,
    entry_motion_px: Optional[float] = None,
    exit_motion_px: Optional[float] = None,
    min_peak_motion_px: Optional[float] = None,
    min_core_frames: int = 2,
    min_rearm_frames: int = 4,
    post_event_cooldown_frames: int = 6,
    motion_smooth_window: int = 3,
    entry_confirm_frames: int = 2,
    exit_confirm_frames: int = 2,
    max_rearm_hold_frames: int = 14,
    enable_event_two_handed_override: bool = False,
    two_handed_event_min_frames: int = 3,
    two_handed_event_ratio: float = 0.22,
    enable_pose_two_handed_boost: bool = False,
    two_handed_wrist_distance_px: Optional[float] = None,
    return_frame_trace: bool = False,
    max_frames: int = 0,
) -> Dict:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    dominant_hand = get_dominant_hand(config)
    swing_cfg = config.get("swing_analysis", {})
    mirror_view = bool(swing_cfg.get("mirror_view", False))
    two_handed_thresh = (
        float(two_handed_wrist_distance_px)
        if two_handed_wrist_distance_px is not None
        else float(swing_cfg.get("two_hand_wrist_distance_max_px", 60))
    )
    frame_idx = 0
    sampled_frames = 0
    prev_wrist = None
    events: List[Dict] = []
    sample_stride = max(1, int(sample_every_n_frames))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    try:
        while True:
            if max_frames > 0 and frame_idx >= max_frames:
                break

            if sample_stride > 1 and (frame_idx % sample_stride) != 0:
                # 非采样帧仅 grab，不做完整解码，降低批量扫描开销
                if not cap.grab():
                    break
                frame_idx += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            pose_results = pose_module.get_keypoints(frame)
            label = pose_module.classify_swing(pose_results) if pose_results else "No Pose"
            if label == "No Person":
                label = "No Pose"
            # 事件级识别更依赖双手结构，遇到可疑反手时优先提升为双反
            if bool(enable_pose_two_handed_boost) and label in {"Backhand", "Forehand", "Two-Handed Backhand"}:
                if _infer_two_handed_backhand_from_pose(
                    pose_results=pose_results,
                    dominant_hand=dominant_hand,
                    mirror_view=mirror_view,
                    wrist_distance_threshold_px=two_handed_thresh,
                ):
                    label = "Two-Handed Backhand"

            wrist = _extract_dominant_wrist(pose_results, dominant_hand)
            motion_px = 0.0
            if prev_wrist and wrist:
                dx = wrist[0] - prev_wrist[0]
                dy = wrist[1] - prev_wrist[1]
                motion_px = float((dx * dx + dy * dy) ** 0.5)
            # 关键点丢失时重置，避免跨长间隔误算超大位移
            prev_wrist = wrist if wrist else None

            events.append(
                {
                    "frame": frame_idx,
                    "label": label,
                    "motion_px": motion_px,
                }
            )
            sampled_frames += 1
            frame_idx += 1
    finally:
        cap.release()

    has_swing, stats = detect_swing_presence(
        events,
        min_swing_frames=min_swing_frames,
        min_consecutive=min_consecutive,
        min_motion_px=min_motion_px,
        max_motion_px=max_motion_px,
        min_event_frames=min_event_frames,
        max_break_frames=max_break_frames,
        type_switch_min_frames=type_switch_min_frames,
        label_vote_window=label_vote_window,
        entry_motion_px=entry_motion_px,
        exit_motion_px=exit_motion_px,
        min_peak_motion_px=min_peak_motion_px,
        min_core_frames=min_core_frames,
        min_rearm_frames=min_rearm_frames,
        post_event_cooldown_frames=post_event_cooldown_frames,
        motion_smooth_window=motion_smooth_window,
        entry_confirm_frames=entry_confirm_frames,
        exit_confirm_frames=exit_confirm_frames,
        max_rearm_hold_frames=max_rearm_hold_frames,
        enable_event_two_handed_override=enable_event_two_handed_override,
        two_handed_event_min_frames=two_handed_event_min_frames,
        two_handed_event_ratio=two_handed_event_ratio,
        return_frame_trace=return_frame_trace,
    )
    return {
        "video_path": video_path,
        "has_swing": has_swing,
        "total_frames": total_frames,
        "sampled_frames": sampled_frames,
        "fps": fps,
        "stats": stats,
    }


def _collect_videos(input_dir: str, recursive: bool) -> List[str]:
    exts = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
    files = []
    if recursive:
        for root, _, names in os.walk(input_dir):
            for name in names:
                if os.path.splitext(name)[1].lower() in exts:
                    files.append(os.path.join(root, name))
    else:
        for name in os.listdir(input_dir):
            path = os.path.join(input_dir, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in exts:
                files.append(path)
    return sorted(files)


def _write_csv(results: List[Dict], output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True) if os.path.dirname(output_csv) else None
    headers = [
        "video_path",
        "has_swing",
        "total_frames",
        "sampled_frames",
        "swing_frame_count",
        "qualified_swing_frames",
        "max_consecutive_qualified",
        "swing_event_count",
        "forehand_events",
        "backhand_events",
        "two_handed_backhand_events",
        "forehand_count",
        "backhand_count",
        "two_handed_backhand_count",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for item in results:
            counts = item["stats"]["swing_type_counts"]
            event_counts = item["stats"].get("swing_event_type_counts", {})
            writer.writerow(
                {
                    "video_path": item["video_path"],
                    "has_swing": item["has_swing"],
                    "total_frames": item["total_frames"],
                    "sampled_frames": item["sampled_frames"],
                    "swing_frame_count": item["stats"]["swing_frame_count"],
                    "qualified_swing_frames": item["stats"]["qualified_swing_frames"],
                    "max_consecutive_qualified": item["stats"]["max_consecutive_qualified"],
                    "swing_event_count": item["stats"].get("swing_event_count", 0),
                    "forehand_events": event_counts.get("Forehand", 0),
                    "backhand_events": event_counts.get("Backhand", 0),
                    "two_handed_backhand_events": event_counts.get("Two-Handed Backhand", 0),
                    "forehand_count": counts.get("Forehand", 0),
                    "backhand_count": counts.get("Backhand", 0),
                    "two_handed_backhand_count": counts.get("Two-Handed Backhand", 0),
                }
            )


def main():
    from pose_estimator import PoseEstimator

    parser = argparse.ArgumentParser(description="独立视频挥拍检测器（有/无挥拍）")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", "-i", help="单个视频文件路径")
    source.add_argument("--input-dir", help="批量视频目录路径")

    parser.add_argument("--recursive", action="store_true", help="批量模式下递归扫描子目录")
    parser.add_argument("--config", "-c", default="configs/yolo26_tennis_config.yaml", help="配置文件路径")
    parser.add_argument("--model", default=None, help="覆盖姿态模型路径")
    parser.add_argument("--sample-every-n-frames", type=int, default=2, help="采样步长，默认2")
    parser.add_argument("--min-swing-frames", type=int, default=12, help="判定最少挥拍帧")
    parser.add_argument("--min-consecutive", type=int, default=3, help="判定最少连续挥拍帧")
    parser.add_argument("--min-motion-px", type=float, default=6.0, help="手腕最小位移阈值")
    parser.add_argument("--max-motion-px", type=float, default=120.0, help="手腕最大位移阈值（抑制关键点跳变）")
    parser.add_argument("--min-event-frames", type=int, default=3, help="统计挥拍次数时单次挥拍最少有效帧")
    parser.add_argument("--max-break-frames", type=int, default=1, help="单次挥拍内部允许的最大中断帧数")
    parser.add_argument("--type-switch-min-frames", type=int, default=2, help="类型切换至少持续多少有效帧才算新挥拍")
    parser.add_argument("--label-vote-window", type=int, default=5, help="逐帧标签时间窗投票（奇数，>=1）")
    parser.add_argument("--entry-motion-px", type=float, default=None, help="动作进入阈值（默认=min_motion_px+1.5）")
    parser.add_argument("--exit-motion-px", type=float, default=None, help="动作退出阈值（默认=min_motion_px*0.7）")
    parser.add_argument("--min-peak-motion-px", type=float, default=None, help="一次挥拍必须达到的峰值位移阈值")
    parser.add_argument("--min-core-frames", type=int, default=2, help="一次挥拍中峰值核心帧最少数量")
    parser.add_argument("--min-rearm-frames", type=int, default=4, help="两次挥拍之间重新武装所需低运动帧数")
    parser.add_argument("--post-event-cooldown-frames", type=int, default=6, help="每次挥拍结束后的抑制帧数（抑制随挥重复计数）")
    parser.add_argument("--motion-smooth-window", type=int, default=3, help="逐帧位移中值平滑窗口")
    parser.add_argument("--entry-confirm-frames", type=int, default=2, help="连续满足进入阈值多少帧才开新事件")
    parser.add_argument("--exit-confirm-frames", type=int, default=2, help="连续满足退出条件多少帧才结束事件")
    parser.add_argument("--max-rearm-hold-frames", type=int, default=14, help="rearm 最长等待帧数，超过后强制允许下一次事件")
    parser.add_argument("--enable-event-two-handed-override", action="store_true", help="启用事件级双反重标（默认关闭）")
    parser.add_argument("--two-handed-event-min-frames", type=int, default=3, help="事件判双反所需最少双反标签帧")
    parser.add_argument("--two-handed-event-ratio", type=float, default=0.22, help="事件判双反所需双反占比（相对反手+双反）")
    parser.add_argument("--enable-pose-two-handed-boost", action="store_true", help="启用基于关键点的逐帧双反增强（默认关闭）")
    parser.add_argument("--two-handed-wrist-distance-px", type=float, default=None, help="覆盖双反手腕距离阈值（逐帧标签增强）")
    parser.add_argument("--debug-frame-trace", action="store_true", help="在JSON中输出逐帧判定轨迹")
    parser.add_argument("--max-frames", type=int, default=0, help="最大处理帧数，0表示不限制")
    parser.add_argument("--output-json", default=None, help="输出 JSON 文件路径")
    parser.add_argument("--output-csv", default=None, help="批量模式输出 CSV 文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["yolo_pose_model_path"] = args.model
    model_path = config["yolo_pose_model_path"]

    print(f"加载姿态模型: {model_path}")
    pose_module = PoseEstimator(model_path, config, roi_manager=None)

    targets = [args.input] if args.input else _collect_videos(args.input_dir, args.recursive)
    if not targets:
        raise RuntimeError("未找到可处理的视频文件")

    results = []
    for idx, video in enumerate(targets, start=1):
        result = process_video_for_swing_presence(
            video,
            pose_module=pose_module,
            config=config,
            sample_every_n_frames=max(1, args.sample_every_n_frames),
            min_swing_frames=max(1, args.min_swing_frames),
            min_consecutive=max(1, args.min_consecutive),
            min_motion_px=max(0.0, args.min_motion_px),
            max_motion_px=max(args.min_motion_px, args.max_motion_px),
            min_event_frames=max(1, args.min_event_frames),
            max_break_frames=max(0, args.max_break_frames),
            type_switch_min_frames=max(1, args.type_switch_min_frames),
            label_vote_window=max(1, args.label_vote_window),
            entry_motion_px=args.entry_motion_px,
            exit_motion_px=args.exit_motion_px,
            min_peak_motion_px=args.min_peak_motion_px,
            min_core_frames=max(1, args.min_core_frames),
            min_rearm_frames=max(1, args.min_rearm_frames),
            post_event_cooldown_frames=max(0, args.post_event_cooldown_frames),
            motion_smooth_window=max(1, args.motion_smooth_window),
            entry_confirm_frames=max(1, args.entry_confirm_frames),
            exit_confirm_frames=max(1, args.exit_confirm_frames),
            max_rearm_hold_frames=max(1, args.max_rearm_hold_frames),
            enable_event_two_handed_override=bool(args.enable_event_two_handed_override),
            two_handed_event_min_frames=max(1, args.two_handed_event_min_frames),
            two_handed_event_ratio=max(0.0, min(1.0, args.two_handed_event_ratio)),
            enable_pose_two_handed_boost=bool(args.enable_pose_two_handed_boost),
            two_handed_wrist_distance_px=args.two_handed_wrist_distance_px,
            return_frame_trace=bool(args.debug_frame_trace),
            max_frames=max(0, args.max_frames),
        )
        results.append(result)
        print(
            f"[{idx}/{len(targets)}] {video} -> "
            f"has_swing={result['has_swing']} "
            f"(events={result['stats'].get('swing_event_count', 0)}, "
            f"qualified={result['stats']['qualified_swing_frames']}, "
            f"max_consecutive={result['stats']['max_consecutive_qualified']})"
        )

    summary = {
        "total_videos": len(results),
        "videos_with_swing": sum(1 for r in results if r["has_swing"]),
        "videos_without_swing": sum(1 for r in results if not r["has_swing"]),
        "results": results,
    }

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"JSON结果已保存: {args.output_json}")

    if args.output_csv:
        _write_csv(results, args.output_csv)
        print(f"CSV结果已保存: {args.output_csv}")

    print(
        f"完成: {summary['total_videos']} 个视频, "
        f"有挥拍 {summary['videos_with_swing']} 个, "
        f"无挥拍 {summary['videos_without_swing']} 个"
    )


if __name__ == "__main__":
    main()
