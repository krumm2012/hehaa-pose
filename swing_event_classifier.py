#!/usr/bin/env python3
"""Event-level stroke classification from extracted swing features."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List


SWING_TYPES = {"Forehand", "Backhand", "Two-Handed Backhand"}


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

    close_two_hand_frames = 0
    backhand_side_frames = 0
    forehand_side_frames = 0
    for feature in event_features:
        dist = feature.get("two_hand_distance")
        offset = feature.get("active_wrist_x_offset")
        if dist is not None and float(dist) <= float(two_hand_distance_px):
            close_two_hand_frames += 1
        if offset is not None:
            if float(offset) > 0:
                forehand_side_frames += 1
            elif float(offset) < 0:
                backhand_side_frames += 1

    two_hand_ratio = close_two_hand_frames / max(1, len(event_features))
    label_two_hand_ratio = swing_label_counts.get("Two-Handed Backhand", 0) / total_swing_labels
    backhand_support_ratio = (
        swing_label_counts.get("Backhand", 0) + swing_label_counts.get("Two-Handed Backhand", 0)
    ) / total_swing_labels

    if (two_hand_ratio >= two_hand_min_ratio and backhand_support_ratio >= 0.20) or label_two_hand_ratio >= two_hand_min_ratio:
        stroke_type = "Two-Handed Backhand"
        confidence = max(two_hand_ratio, label_two_hand_ratio)
    elif swing_label_counts.get("Forehand", 0) >= swing_label_counts.get("Backhand", 0):
        stroke_type = "Forehand"
        confidence = swing_label_counts.get("Forehand", 0) / total_swing_labels
    else:
        stroke_type = "Backhand"
        confidence = swing_label_counts.get("Backhand", 0) / total_swing_labels

    return {
        "stroke_type": stroke_type,
        "confidence": round(float(confidence), 4),
        "evidence": {
            "label_counts": swing_label_counts,
            "two_hand_ratio": round(float(two_hand_ratio), 4),
            "label_two_hand_ratio": round(float(label_two_hand_ratio), 4),
            "backhand_support_ratio": round(float(backhand_support_ratio), 4),
            "backhand_side_frames": int(backhand_side_frames),
            "forehand_side_frames": int(forehand_side_frames),
        },
    }
