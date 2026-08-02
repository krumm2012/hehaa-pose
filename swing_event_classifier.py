#!/usr/bin/env python3
"""Event-level stroke classification from extracted swing features."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Dict, List


SWING_TYPES = {"Forehand", "Backhand", "Two-Handed Backhand"}


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
) -> Dict:
    """Classify a complete swing event using event-level evidence.

    Frame labels are treated as one signal, not as the source of truth.
    Two-handed backhand needs sustained close-hand evidence within the event.
    """
    if not event_features:
        return {
            "stroke_type": "Unknown",
            "confidence": 0.0,
            "evidence": {},
        }

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
    side = swing_side["side"]
    if side == "forehand":
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
    elif swing_label_counts.get("Forehand", 0) >= (
        swing_label_counts.get("Backhand", 0)
        + swing_label_counts.get("Two-Handed Backhand", 0)
    ):
        stroke_type = "Forehand"
        confidence = label_forehand_ratio
        decision_rule = "label_fallback_forehand"
    else:
        stroke_type = "Backhand"
        confidence = label_backhand_ratio
        decision_rule = "label_fallback_backhand"

    return {
        "stroke_type": stroke_type,
        "confidence": round(float(confidence), 4),
        "evidence": {
            "label_counts": swing_label_counts,
            "two_hand_ratio": round(float(two_hand_ratio), 4),
            "two_hand_evidence_frames": int(two_hand_evidence_frames),
            "label_two_hand_ratio": round(float(label_two_hand_ratio), 4),
            "backhand_support_ratio": round(float(backhand_support_ratio), 4),
            "backhand_side_frames": int(swing_side["backhand_side_frames"]),
            "forehand_side_frames": int(swing_side["forehand_side_frames"]),
            "classification_context": {
                "policy_version": "player_camera_swing_v1",
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
                "decision_rule": decision_rule,
            },
        },
    }
