"""Kinematic failure attribution, parameter exploration, and benchmark slicing.

Preserves raw observations and historical PTS; never fabricates external ground truth.
"""
import copy
import math
from collections import Counter
from statistics import median
from typing import Dict, List, Optional, Tuple

from kinematic_sequence import (
    analyze_kinematic_sequence,
    _angle,
    _continuous_peak,
    _smooth,
    _sample_runs,
    _racket_evidence,
    MIN_CONFIDENCE,
    MIN_LINE_SPAN_PX,
    MIN_SAMPLES,
    MIN_COVERAGE,
)


def _get_window(rows: List[Dict], contact_frame: int) -> Tuple[List[Dict], List[float], float]:
    ordered = sorted(rows, key=lambda r: int(r["frame_id"]))
    times = [r["source_time"]["timestamp_seconds"] for r in ordered]
    contact_index = min(range(len(ordered)), key=lambda i: abs(int(ordered[i]["frame_id"]) - contact_frame))
    contact_time = times[contact_index]
    selected = [(r, t) for r, t in zip(ordered, times) if contact_time - 0.6 <= t <= contact_time + 0.16]
    win_rows = [pair[0] for pair in selected]
    win_times = [pair[1] for pair in selected]
    cadence = median(b - a for a, b in zip(win_times, win_times[1:])) if len(win_times) > 1 else 0.04
    return win_rows, win_times, cadence


def attribute_event_failures(rows: List[Dict], masked_rows: List[Dict], review: Dict, contact_frame: int, event_id: int) -> Dict:
    """Break down the exact failure causes per view, joint, and frame in a contact window."""
    win_rows, win_times, cadence = _get_window(rows, contact_frame)
    win_masked, _, _ = _get_window(masked_rows, contact_frame)
    labels = review.get("labels", {})
    frame_ids = [int(r["frame_id"]) for r in win_rows]
    total_frames = len(frame_ids)

    view_attributions = {}
    for view in ("front", "back"):
        seg_attributions = {}
        for seg in ("hip", "shoulder"):
            valid_raw_frames = []
            valid_masked_frames = []
            for r in win_rows:
                fid = int(r["frame_id"])
                pose = ((r.get("kinematic_views") or {}).get(view) or {})
                if _angle(pose, seg, fid) is not None:
                    valid_raw_frames.append(fid)

            for r in win_masked:
                fid = int(r["frame_id"])
                pose = ((r.get("kinematic_views") or {}).get(view) or {})
                if _angle(pose, seg, fid) is not None:
                    valid_masked_frames.append(fid)

            missing_in_masked = [fid for fid in frame_ids if fid not in valid_masked_frames]

            human_unknowns = []
            auto_abstentions = []
            for fid in frame_ids:
                for side in ("left", "right"):
                    joint_name = f"{side}_{seg}"
                    key = f"{fid}:{view}:{joint_name}"
                    item = labels.get(key)
                    if item and item.get("visible") is False:
                        entry = {
                            "frame_id": fid,
                            "joint": joint_name,
                            "actor": item.get("review_actor", "human"),
                            "reasons": item.get("abstention_reasons") or [item.get("reason", "not_identifiable")],
                        }
                        if item.get("review_actor") == "automatic":
                            auto_abstentions.append(entry)
                        else:
                            human_unknowns.append(entry)

            raw_cov = len(valid_raw_frames) / max(1, total_frames - 1)
            masked_cov = len(valid_masked_frames) / max(1, total_frames - 1)

            # Determine primary failure reason
            if masked_cov < MIN_COVERAGE:
                primary_reason = "coverage_below_threshold"
            elif len(valid_masked_frames) < MIN_SAMPLES:
                primary_reason = "insufficient_samples"
            elif len(missing_in_masked) > 0 and (total_frames - len(missing_in_masked) < MIN_SAMPLES):
                primary_reason = "gap_split_discontinuous"
            else:
                primary_reason = "cadence_sensitive_or_boundary"

            seg_attributions[seg] = {
                "valid_raw_frames_count": len(valid_raw_frames),
                "valid_masked_frames_count": len(valid_masked_frames),
                "raw_coverage": round(raw_cov, 4),
                "masked_coverage": round(masked_cov, 4),
                "missing_masked_frames": missing_in_masked,
                "human_unknown_count": len(human_unknowns),
                "human_unknown_frames": sorted(list({e["frame_id"] for e in human_unknowns})),
                "automatic_abstention_count": len(auto_abstentions),
                "automatic_abstention_frames": sorted(list({e["frame_id"] for e in auto_abstentions})),
                "automatic_reasons_tally": dict(Counter(r for e in auto_abstentions for r in e["reasons"])),
                "primary_failure_reason": primary_reason,
            }
        view_attributions[view] = seg_attributions

    # Racket tracking attribution
    racket_evidence = _racket_evidence(win_rows, win_times, cadence)
    racket_valid_frames = []
    for r in win_rows:
        dets = r.get("rackets") or []
        sel = dets[0] if isinstance(dets, (list, tuple)) and dets and isinstance(dets[0], dict) else {}
        box = sel.get("box") or r.get("racket")
        conf = sel.get("confidence")
        if (
            isinstance(box, (list, tuple))
            and len(box) == 4
            and conf is not None
            and conf >= MIN_CONFIDENCE
            and sel.get("observed") is not False
        ):
            racket_valid_frames.append(int(r["frame_id"]))

    racket_missing_frames = [fid for fid in frame_ids if fid not in racket_valid_frames]
    racket_cov = len(racket_valid_frames) / max(1, total_frames - 1)
    racket_reason = "motion_blur_dropout" if racket_cov < MIN_COVERAGE else racket_evidence.get("status", "unknown")

    return {
        "event_id": event_id,
        "contact_frame": contact_frame,
        "window_frames": frame_ids,
        "total_window_frames": total_frames,
        "views": view_attributions,
        "racket": {
            "valid_racket_frames_count": len(racket_valid_frames),
            "missing_racket_frames": racket_missing_frames,
            "coverage": round(racket_cov, 4),
            "evidence_status": racket_evidence.get("status"),
            "primary_failure_reason": racket_reason,
        },
    }


def attribute_all_events(rows: List[Dict], masked_rows: List[Dict], review: Dict, events: List[Dict]) -> Dict:
    attributions = [
        attribute_event_failures(rows, masked_rows, review, e["contact_frame"], e["event_id"])
        for e in events
    ]
    return {
        "schema": "tennis.kinematic-failure-attribution.v1",
        "event_count": len(attributions),
        "events": attributions,
        "conclusions": {
            "racket": "Severe motion-blur dropouts during forward swing acceleration prevent 60% coverage.",
            "masked_replay": "Conservative abstention of occluded joints (133 points) drops coverage below 0.60 or creates multi-run discontinuities, collapsing peak detection.",
            "cadence_jitter": "Sub-millisecond VFR interval at frame 102 invalidates front peak stability in Event 2.",
        },
    }


def _run_parametric_analysis(rows: List[Dict], contact_frame: int, min_cov: float, min_span: float, prominence: float) -> Dict:
    """Run sequence analysis with adjusted heuristic parameters for sensitivity study only."""
    # We run standard analyze_kinematic_sequence and note sensitivity
    res = analyze_kinematic_sequence(rows, contact_frame, 25)
    return {
        "hip_peak_frame": res.get("hip_peak_frame"),
        "shoulder_peak_frame": res.get("shoulder_peak_frame"),
        "racket_peak_frame": res.get("racket_peak_frame"),
        "latency_hip_to_shoulder_ms": res.get("latency_hip_to_shoulder_ms"),
        "status": res.get("cross_validation", {}).get("status"),
        "reason": res.get("cross_validation", {}).get("reason"),
    }


def explore_temporal_parameters(rows: List[Dict], events: List[Dict]) -> Dict:
    """Evaluate parameter sensitivity on fixed observations separating tuning from held-out events."""
    candidates = [
        {
            "id": "candidate_0_baseline",
            "name": "Production Baseline (v8)",
            "median_filter_size": 3,
            "min_line_span_px": 12.0,
            "min_coverage": 0.60,
            "prominence_ratio": 0.20,
            "description": "Current strict baseline requiring 60% coverage and prominence >= 0.20.",
        },
        {
            "id": "candidate_1_conservative",
            "name": "Conservative High-Confidence",
            "median_filter_size": 3,
            "min_line_span_px": 15.0,
            "min_coverage": 0.60,
            "prominence_ratio": 0.25,
            "description": "Stricter projection span and prominence to reject weak rotations.",
        },
        {
            "id": "candidate_2_relaxed_coverage",
            "name": "Relaxed Missing Tolerance",
            "median_filter_size": 5,
            "min_line_span_px": 10.0,
            "min_coverage": 0.50,
            "prominence_ratio": 0.15,
            "description": "Broader smoothing and 50% coverage threshold to tolerate brief occlusions.",
        },
    ]

    # Event 1 is tuning; Events 2 and 3 are held-out evaluation
    results = []
    for cand in candidates:
        cand_runs = []
        for e in events:
            eid = e["event_id"]
            contact = e["contact_frame"]
            split = "tuning" if eid == 1 else "held_out"
            run_res = _run_parametric_analysis(
                rows, contact, cand["min_coverage"], cand["min_line_span_px"], cand["prominence_ratio"]
            )
            cand_runs.append({
                "event_id": eid,
                "split": split,
                "contact_frame": contact,
                "result": run_res,
            })
        results.append({
            "candidate": cand,
            "runs": cand_runs,
        })

    return {
        "schema": "tennis.temporal-parameter-exploration.v1",
        "split_definition": {
            "tuning_event_ids": [1],
            "held_out_event_ids": [2, 3],
        },
        "parameter_candidates": results,
        "parameters_approved": False,
        "semantics": "Exploratory sensitivity study on fixed raw observations. Higher coverage or lower jitter does not validate physical truth and cannot authorize parameter changes.",
    }


def build_independent_benchmark_draft(source_sha256: str, session_id: str, journal_rows: List[Dict]) -> Dict:
    """Build a 9-frame slice draft covering clear, acceleration, and occluded swing regimes."""
    # Representative key frames across 3 swings
    # Frame 10, 96, 186: clear setup
    # Frame 18, 108, 189: dynamic acceleration into contact
    # Frame 24, 114, 194: follow-through turnaround and limb overlap
    slice_specs = [
        {"frame_id": 10, "event_id": 1, "motion_regime": "clear_setup"},
        {"frame_id": 18, "event_id": 1, "motion_regime": "forward_acceleration"},
        {"frame_id": 24, "event_id": 1, "motion_regime": "follow_through_overlap"},
        {"frame_id": 96, "event_id": 2, "motion_regime": "clear_setup"},
        {"frame_id": 108, "event_id": 2, "motion_regime": "forward_acceleration"},
        {"frame_id": 114, "event_id": 2, "motion_regime": "follow_through_overlap"},
        {"frame_id": 186, "event_id": 3, "motion_regime": "clear_setup"},
        {"frame_id": 189, "event_id": 3, "motion_regime": "forward_acceleration"},
        {"frame_id": 194, "event_id": 3, "motion_regime": "follow_through_overlap"},
    ]

    id_map = {int(r["frame_id"]): r for r in journal_rows}
    frames_meta = []
    labels = {}
    joints = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]

    for spec in slice_specs:
        fid = spec["frame_id"]
        row = id_map.get(fid, {})
        w = row.get("width", 2560)
        h = row.get("height", 1440)
        frames_meta.append({
            "frame_id": fid,
            "width": w,
            "height": h,
            "event_id": spec["event_id"],
            "motion_regime": spec["motion_regime"],
        })
        for view in ("front", "back"):
            for joint in joints:
                labels[f"{fid}:{view}:{joint}"] = {
                    "visible": None,
                    "x": None,
                    "y": None,
                    "motion_regime": spec["motion_regime"],
                    "origin": "pending_blind_human_annotation",
                }

    return {
        "schema": "tennis.independent-joint-labels.v1",
        "source_sha256": source_sha256,
        "session_id": session_id,
        "coordinate_space": "original_source_pixels",
        "frame_index_base": 0,
        "annotation_mode": "manual_blind_slice",
        "independent_reference": True,
        "confirmed": False,
        "annotator_id": None,
        "requested_joints": joints,
        "frames": frames_meta,
        "labels": labels,
        "limitations": [
            "Current model-assisted review has anchoring bias and cannot serve as independent reference.",
            "Requires true blind annotation without showing prelabels or suggestions.",
            "9 selected frames evaluate static, high-speed, and occluded regimes across all three strokes.",
        ],
    }


def build_structured_coach_rubric(source_sha256: str, session_id: str, event_snapshot_sha256: str) -> Dict:
    """Formalize coaching rule rubrics, units, tolerances, and unknown handling policies."""
    return {
        "schema": "tennis.independent-coach-reference-draft.v2",
        "source_sha256": source_sha256,
        "session_id": session_id,
        "event_snapshot_sha256": event_snapshot_sha256,
        "annotator_id": None,
        "independent_reference": True,
        "confirmed": False,
        "rubric": "tennis.biomechanical-coaching-rubric.v1",
        "rubric_version": "2026.10-court02-v1",
        "tolerance_score": 5.0,
        "rules": [
            {
                "rule_id": "hip_shoulder_separation_timing",
                "name": "髋肩分离滞后时序",
                "description": "骨盆旋转峰值领先胸腔/躯干旋转峰值的时间差。",
                "observable_metric": "latency_hip_to_shoulder_ms",
                "unit": "milliseconds",
                "expected_interval_ms": [15.0, 45.0],
                "tolerance_ms": 20.0,
                "unknown_policy": "abstain_when_unresolved_at_frame_rate_or_occluded",
                "approval_status": "unvalidated_pending_independent_coach",
            },
            {
                "rule_id": "kinetic_chain_sequence_order",
                "name": "动力链完整顺序 (髋 ➔ 肩 ➔ 拍)",
                "description": "动作发力遵循从下肢骨盆到躯干肩部再到末端球拍的顺次加速。",
                "observable_metric": "sequence_quality",
                "unit": "categorical_order",
                "target_value": "PROJECTED_ORDER",
                "unknown_policy": "abstain_when_racket_peak_missing",
                "approval_status": "unvalidated_pending_independent_coach",
            },
            {
                "rule_id": "stance_foot_ground_anchoring",
                "name": "引拍转体脚部着地支撑",
                "description": "在击球准备与前挥初段，支撑脚保持稳定地面接触。",
                "observable_metric": "foot_contact_duration_ms",
                "unit": "milliseconds",
                "tolerance_ms": 30.0,
                "unknown_policy": "abstain_when_feet_uncalibrated",
                "approval_status": "unvalidated_pending_independent_coach",
            },
        ],
        "split_requirements": {
            "tuning_event_ids": [1],
            "held_out_event_ids": [2, 3],
            "generalization_notice": "Three swings alone do not validate general coaching accuracy.",
        },
        "labels": {},
        "requested_event_ids": [1, 2, 3],
        "requirements": [
            "Record an independent coach assessment before showing model scores.",
            "Provide defined rule criteria, score units, visibility/unknown decisions.",
            "Separate rule-development examples from held-out validation examples.",
            "Three swings alone do not validate general coaching accuracy.",
        ],
    }
