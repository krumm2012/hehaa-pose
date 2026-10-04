#!/usr/bin/env python3
"""Event-level stroke classification from extracted swing features."""

from __future__ import annotations

from collections import Counter
import math
from statistics import median
from typing import Dict, List, Optional

from metric_source_windows import MetricWindowContext


SWING_TYPES = {"Forehand", "Backhand", "Two-Handed Backhand"}
POLICY_VERSION = "player_camera_swing_v3_source_windows"


def _classification_windows(rows, contact_frame):
    """Separate bounded stroke and contact candidates, with disclosed clocks.

    Invalid stream/legacy clocks retain source-frame candidate heuristics. They
    never authorize measured timing, exposure, accuracy or technical scoring.
    """
    context = MetricWindowContext(rows)

    def select(before, after, legacy_before, legacy_after):
        request = {'anchor_frame_id': contact_frame, 'before_seconds': before,
                   'after_seconds': after}
        reasons = list(context.reasons)
        basis = context.basis
        qualified = context.qualified and contact_frame is not None
        identity_invalid = 'invalid_or_duplicate_source_frame_identity' in reasons
        if contact_frame is None:
            selected = list(rows)
            qualified = False
            reasons.append('contact_anchor_not_supplied')
        elif (identity_invalid or type(contact_frame) is not int
              or contact_frame < 0 or contact_frame not in context.by_id):
            selected = []
            qualified = False
            reasons.append('missing_or_invalid_contact_anchor')
        elif context.timed:
            window = context.around(contact_frame, before=before, after=after)
            selected = [context.by_id[fid] for fid in window.frame_ids]
        else:
            selected = [context.by_id[fid] for fid in context.ids
                        if contact_frame - legacy_before <= fid <= contact_frame + legacy_after]
            basis = 'source_frame_offsets_unverified'
            request['legacy_candidate_offsets'] = [-legacy_before, legacy_after]
        if not qualified:
            reasons.append('candidate_window_time_unverified')
        ids = [r.get('frame_id') for r in selected]
        evidence = {'policy_version': POLICY_VERSION, 'basis': basis,
            'source_time_qualified': qualified, 'requested_window': request,
            'source_frame_ids': ids, 'sample_count': len(selected),
            'observation_frame_gaps': [[a, b] for a, b in zip(ids, ids[1:])
                if type(a) is int and type(b) is int and b != a + 1],
            'reasons': list(dict.fromkeys(reasons)),
            'actual_time_range_seconds': ([context.times[ids[0]], context.times[ids[-1]]]
                if ids and context.timed and all(fid in context.times for fid in ids) else None),
            'accuracy_validated': False, 'sensor_exposure_verified': False,
            'coach_eligible': False,
            'coverage_semantics': 'retained_sample_availability_not_temporal_coverage'}
        return selected, evidence

    stroke_rows, stroke_window = select(.56, .08, 14, 2)
    contact_rows, contact_window = select(.12, .12, 3, 3)
    return stroke_rows, contact_rows, stroke_window, contact_window


def _dominant_hand(event_features: List[Dict]) -> str:
    hands = [
        str(feature.get("dominant_hand") or "").lower()
        for feature in event_features
        if str(feature.get("dominant_hand") or "").lower() in {"left", "right"}
    ]
    return Counter(hands).most_common(1)[0][0] if hands else "right"


def _camera_context(event_features: List[Dict]) -> Dict:
    scores = [
        float(feature["camera_facing_score"])
        for feature in event_features
        if feature.get("camera_facing_score") is not None
    ]
    if len(scores) < 3:
        return {
            "view": "unknown",
            "confidence": 0.0,
            "evidence_frames": len(scores),
            "right_side_projects_to": "unknown",
        }

    center = float(median(scores))
    direction = -1 if center < 0 else 1
    agreement = sum(1 for score in scores if score * direction > 0.20) / len(scores)
    confidence = min(1.0, abs(center)) * agreement
    if abs(center) < 0.35 or agreement < 0.65:
        view = "side_or_uncertain"
        projection = "unknown"
    elif center < 0:
        view = "facing_player"
        projection = "screen_left"
    else:
        view = "behind_player"
        projection = "screen_right"
    return {
        "view": view,
        "confidence": round(float(confidence), 4),
        "evidence_frames": len(scores),
        "median_facing_score": round(center, 4),
        "orientation_agreement": round(float(agreement), 4),
        "right_side_projects_to": projection,
    }


def _swing_side_context(event_features: List[Dict], dominant_hand: str, camera: Dict) -> Dict:
    view = camera.get("view")
    if view not in {"facing_player", "behind_player"}:
        return {
            "side": "unknown",
            "confidence": 0.0,
            "evidence_frames": 0,
            "forehand_side_frames": 0,
            "backhand_side_frames": 0,
        }

    # A right-handed player's forehand is on anatomical right.  Its screen
    # projection flips when the player turns from facing to facing away from
    # the camera.  Left-handed players use the inverse anatomical side.
    expected_forehand_sign = -1.0 if view == "facing_player" else 1.0
    if dominant_hand == "left":
        expected_forehand_sign *= -1.0

    forehand_frames = 0
    backhand_frames = 0
    for feature in event_features:
        normalized = feature.get("active_wrist_x_offset_body_width")
        offset = normalized if normalized is not None else feature.get("active_wrist_x_offset")
        if offset is None:
            continue
        offset = float(offset)
        dead_zone = 0.10 if normalized is not None else 1.0
        if abs(offset) < dead_zone:
            continue
        if offset * expected_forehand_sign > 0:
            forehand_frames += 1
        else:
            backhand_frames += 1

    evidence_frames = forehand_frames + backhand_frames
    if evidence_frames < 3:
        side = "unknown"
        confidence = 0.0
    else:
        forehand_ratio = forehand_frames / evidence_frames
        backhand_ratio = backhand_frames / evidence_frames
        if forehand_ratio >= 0.58:
            side = "forehand"
            confidence = forehand_ratio
        elif backhand_ratio >= 0.58:
            side = "backhand"
            confidence = backhand_ratio
        else:
            side = "uncertain"
            confidence = max(forehand_ratio, backhand_ratio)
    return {
        "side": side,
        "confidence": round(float(confidence), 4),
        "evidence_frames": evidence_frames,
        "forehand_side_frames": forehand_frames,
        "backhand_side_frames": backhand_frames,
    }


def _two_hand_ratio(event_features: List[Dict], pixel_threshold: float) -> tuple[float, int]:
    evidence_frames = 0
    close_frames = 0
    for feature in event_features:
        normalized = feature.get("two_hand_distance_body_width")
        distance = normalized if normalized is not None else feature.get("two_hand_distance")
        if distance is None:
            continue
        evidence_frames += 1
        threshold = 0.82 if normalized is not None else float(pixel_threshold)
        if float(distance) <= threshold:
            close_frames += 1
    return close_frames / max(1, evidence_frames), evidence_frames


def classify_swing_event(
    event_features: List[Dict],
    two_hand_distance_px: float = 95.0,
    two_hand_min_ratio: float = 0.28,
    contact_frame: Optional[int] = None,
) -> Dict:
    """Classify a complete swing event using event-level evidence.

    Frame labels are treated as one signal, not as the source of truth.
    Two-handed backhand needs sustained close-hand evidence within the event.
    """
    event_features, contact_features, stroke_window, contact_window = _classification_windows(
        event_features, contact_frame)

    label_counts = Counter(f.get("raw_swing_type", "Unknown") for f in event_features)
    swing_label_counts = {k: label_counts.get(k, 0) for k in sorted(SWING_TYPES)}
    total_swing_labels = max(1, sum(swing_label_counts.values()))

    dominant_hand = _dominant_hand(event_features)
    camera = _camera_context(event_features)
    swing_side = _swing_side_context(event_features, dominant_hand, camera)
    two_hand_ratio, two_hand_evidence_frames = _two_hand_ratio(
        event_features,
        two_hand_distance_px,
    )
    label_two_hand_ratio = swing_label_counts.get("Two-Handed Backhand", 0) / total_swing_labels
    backhand_support_ratio = (
        swing_label_counts.get("Backhand", 0) + swing_label_counts.get("Two-Handed Backhand", 0)
    ) / total_swing_labels

    label_forehand_ratio = swing_label_counts.get("Forehand", 0) / total_swing_labels
    label_backhand_ratio = backhand_support_ratio

    # Dual-view biomechanics evidence (virtual rear camera + front camera fusion)
    # 结合挥拍动态权重与解剖中线跨越法则
    dv_stroke_weights = Counter()
    total_dv_weight = 0.0
    dv_two_handed_weight = 0.0
    dv_two_handed_count = 0

    for f in event_features:
        st = f.get("dual_view_stroke_type")
        if st in SWING_TYPES and float(f.get("dual_view_confidence") or 0) >= 0.55:
            w = min(30.0, max(1.0, float(f.get("wrist_speed") or 1.0))) * float(f["dual_view_confidence"])
            dv_stroke_weights[st] += w
            total_dv_weight += w
            if f.get("dual_view_is_two_handed") is True:
                dv_two_handed_count += 1
                dv_two_handed_weight += w

    dv_evidence_count = sum(
        1
        for f in event_features
        if f.get("dual_view_stroke_type") in SWING_TYPES and float(f.get("dual_view_confidence") or 0) >= 0.55
    )

    stroke_type, confidence, decision_rule = "Unknown", 0.0, "insufficient_classification_evidence"
    side = swing_side["side"]
    if dv_evidence_count >= 3:
        top_stroke, top_weight = dv_stroke_weights.most_common(1)[0]
        dv_ratio = top_weight / max(1.0, total_dv_weight)

        if dv_ratio >= 0.55:
            if top_stroke == "Two-Handed Backhand" or (
                top_stroke == "Backhand"
                and (
                    dv_two_handed_count / max(1, dv_evidence_count) >= 0.35
                    or two_hand_ratio >= 0.35
                )
            ):
                stroke_type = "Two-Handed Backhand"
                confidence = dv_ratio * sum(float(f.get("dual_view_confidence") or 0) for f in event_features if f.get("dual_view_stroke_type") in SWING_TYPES and float(f.get("dual_view_confidence") or 0) >= .55) / dv_evidence_count
                decision_rule = "dual_view_two_handed_backhand"
            else:
                stroke_type = top_stroke
                confidence = dv_ratio * sum(float(f.get("dual_view_confidence") or 0) for f in event_features if f.get("dual_view_stroke_type") in SWING_TYPES and float(f.get("dual_view_confidence") or 0) >= .55) / dv_evidence_count
                decision_rule = "dual_view_transverse_projection"
        elif side == "forehand":
            stroke_type = "Forehand"
            confidence = float(swing_side["confidence"]) * 0.60 + 0.30
            decision_rule = "dual_view_fallback_forehand"
        elif side == "backhand":
            stroke_type = "Backhand"
            confidence = float(swing_side["confidence"]) * 0.60 + 0.30
            decision_rule = "dual_view_fallback_backhand"
    elif side == "forehand":
        stroke_type = "Forehand"
        confidence = (
            float(swing_side["confidence"]) * 0.60
            + float(camera["confidence"]) * 0.20
            + label_forehand_ratio * 0.20
        )
        decision_rule = "camera_normalized_forehand_side"
    elif side == "backhand":
        if (
            two_hand_ratio >= max(0.38, two_hand_min_ratio)
            and backhand_support_ratio >= 0.20
        ) or label_two_hand_ratio >= max(0.45, two_hand_min_ratio):
            stroke_type = "Two-Handed Backhand"
            confidence = max(two_hand_ratio, label_two_hand_ratio)
            decision_rule = "camera_normalized_backhand_with_two_hand_support"
        else:
            stroke_type = "Backhand"
            confidence = (
                float(swing_side["confidence"]) * 0.65
                + float(camera["confidence"]) * 0.20
                + label_backhand_ratio * 0.15
            )
            decision_rule = "camera_normalized_backhand_side"
    elif (
        label_two_hand_ratio >= max(0.50, two_hand_min_ratio)
        or (
            two_hand_ratio >= max(0.50, two_hand_min_ratio)
            and swing_label_counts.get("Backhand", 0) > swing_label_counts.get("Forehand", 0)
        )
    ):
        stroke_type = "Two-Handed Backhand"
        confidence = max(two_hand_ratio, label_two_hand_ratio)
        decision_rule = "label_fallback_two_hand"
    elif swing_label_counts.get("Forehand", 0) > 0 and swing_label_counts.get("Forehand", 0) >= (
        swing_label_counts.get("Backhand", 0)
        + swing_label_counts.get("Two-Handed Backhand", 0)
    ):
        stroke_type = "Forehand"
        confidence = label_forehand_ratio
        decision_rule = "label_fallback_forehand"
    elif label_backhand_ratio > 0:
        stroke_type = "Backhand"
        confidence = label_backhand_ratio
        decision_rule = "label_fallback_backhand"

    # Contact & Shadow swing analysis (物理触球与空挥判定)
    ball_pts = [f["ball"] for f in contact_features if f.get("ball") is not None]
    has_ball_in_event = len(ball_pts) >= 2 or any(f.get("has_ball") for f in contact_features)

    min_ball_distance = None
    closest_contact_feature = None
    for f in contact_features:
        d = f.get("ball_racket_distance")
        if d is None and f.get("ball") is not None and f.get("wrist") is not None:
            bx, by = f["ball"]
            wx, wy = f["wrist"]
            d = ((bx - wx) ** 2 + (by - wy) ** 2) ** 0.5
        if type(d) in (int, float) and math.isfinite(d) and d >= 0:
            if min_ball_distance is None or d < min_ball_distance:
                min_ball_distance = d
                closest_contact_feature = f

    # 轨迹反弹检验 (Trajectory Rebound Detection)
    has_trajectory_rebound = False
    if len(ball_pts) >= 3:
        y_diffs = [ball_pts[i][1] - ball_pts[i - 1][1] for i in range(1, len(ball_pts))]
        has_negative = any(dy < -10 for dy in y_diffs)
        has_positive = any(dy > 10 for dy in y_diffs)
        if has_negative and has_positive:
            has_trajectory_rebound = True

    # A distant rebound is not contact evidence; non-detection is not proof of shadow.
    is_shadow_swing = False
    is_valid_contact = min_ball_distance is not None and min_ball_distance <= 180.0
    contact_status = "candidate" if is_valid_contact else "unknown"
    if stroke_type == "Forehand" and two_hand_ratio >= .6 and side != "forehand":
        stroke_type, confidence, decision_rule = "Unknown", 0.0, "two_hand_direction_conflict"
    if len(event_features) < 3 or confidence < .55:
        stroke_type, confidence, decision_rule = "Unknown", 0.0, "insufficient_classification_evidence"
        if len(event_features) < 3:
            stroke_window['reasons'].append('insufficient_classification_samples')

    return {
        "stroke_type": stroke_type,
        "confidence": round(float(confidence), 4),
        "is_shadow_swing": is_shadow_swing,
        "is_valid_contact": is_valid_contact,
        "contact_status": contact_status,
        "min_ball_distance": round(float(min_ball_distance), 2) if min_ball_distance is not None else None,
        "evidence": {
            "label_counts": swing_label_counts,
            "two_hand_ratio": round(float(two_hand_ratio), 4),
            "two_hand_evidence_frames": int(two_hand_evidence_frames),
            "label_two_hand_ratio": round(float(label_two_hand_ratio), 4),
            "backhand_support_ratio": round(float(backhand_support_ratio), 4),
            "backhand_side_frames": int(swing_side["backhand_side_frames"]),
            "forehand_side_frames": int(swing_side["forehand_side_frames"]),
            "classification_context": {
                "policy_version": POLICY_VERSION,
                "window_evidence": stroke_window,
                "confidence_semantics": "uncalibrated_heuristic_evidence_not_accuracy_probability",
                "player": {
                    "dominant_hand": dominant_hand,
                    "source": "configured",
                },
                "camera": camera,
                "swing": {
                    **swing_side,
                    "two_hand_ratio": round(float(two_hand_ratio), 4),
                    "two_hand_evidence_frames": int(two_hand_evidence_frames),
                },
                "dual_view": {
                    "evidence_frames": int(dv_evidence_count),
                    "two_handed_frames": int(dv_two_handed_count),
                } if dv_evidence_count > 0 else None,
                "contact_analysis": {
                    "window_evidence": contact_window,
                    "observation_policy": "fresh_selected_ball_v1; legacy_XY_unverified",
                    "fresh_ball_frame_count": sum(f.get('ball_provenance_status') == 'fresh_model_observation' for f in contact_features),
                    "legacy_unverified_ball_frame_count": sum(f.get('ball') is not None and f.get('ball_provenance_status') != 'fresh_model_observation' for f in contact_features),
                    "closest_evidence_frame": (closest_contact_feature or {}).get('frame_id'),
                    "closest_geometry": (closest_contact_feature or {}).get('contact_geometry') or 'legacy_geometry_unverified',
                    "distance_threshold_px": 180.0,
                    "threshold_accuracy_validated": False,
                    "has_ball": bool(has_ball_in_event),
                    "is_shadow_swing": bool(is_shadow_swing),
                    "is_valid_contact": bool(is_valid_contact),
                    "contact_status": contact_status,
                    "min_ball_distance": round(float(min_ball_distance), 2) if min_ball_distance is not None else None,
                    "has_trajectory_rebound": bool(has_trajectory_rebound),
                    "trajectory_rebound_semantics": "image_vertical_steps_not_verified_contact_or_bounce",
                },
                "decision_rule": decision_rule,
            },
        },
    }
