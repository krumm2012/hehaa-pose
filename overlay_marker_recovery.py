"""Recover analyzer-owned hollow ball markers and racket boxes from video frames."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import cv2
import numpy as np


def recover_overlay_detections(frame, config: Optional[Dict] = None) -> Dict:
    """Return overlay-derived ball/racket candidates without model inference.

    Recovery is opt-in.  It only recognizes the analyzer's known marker colors
    and closed geometry, so ordinary court lines and pose strokes stay out of
    the candidate stream.  Returned detections retain explicit provenance and
    can pass through the existing temporal selectors with model detections.
    """
    options = dict(config or {})
    result = {
        "balls": [],
        "rackets": [],
        "diagnostics": {
            "enabled": bool(options.get("overlay_marker_recovery_enabled", False)),
            "ball_candidates": 0,
            "racket_candidates": 0,
        },
    }
    if not result["diagnostics"]["enabled"] or frame is None or frame.size == 0:
        return result

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ball_mask = cv2.inRange(hsv, (20, 120, 120), (40, 255, 255))
    result["balls"] = _ball_outline_candidates(ball_mask)

    # Current analyzer racket boxes use BGR (255, 128, 0), approximately H105.
    racket_masks = [cv2.inRange(hsv, (95, 120, 100), (115, 255, 255))]
    if options.get("overlay_legacy_green_racket_enabled", False):
        # Older exported videos used green racket rectangles.  Closed-shape
        # validation below rejects open pose skeleton strokes and long ROI lines.
        racket_masks.append(cv2.inRange(hsv, (48, 140, 100), (75, 255, 255)))
    result["rackets"] = _racket_box_candidates(racket_masks, frame.shape[:2])
    result["diagnostics"]["ball_candidates"] = len(result["balls"])
    result["diagnostics"]["racket_candidates"] = len(result["rackets"])
    return result


def _contours(mask) -> List:
    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _ball_outline_candidates(mask) -> List[Dict]:
    candidates = []
    for contour in _contours(mask):
        x, y, width, height = cv2.boundingRect(contour)
        # The analyzer owns a 14px-radius marker with a thick contrast ring;
        # encoded bounds are roughly 24-40px.  Smaller yellow disks are likely
        # real stationary tennis balls and must remain subject to model/tracker
        # evidence rather than being promoted as trusted overlays.
        if not (24 <= width <= 52 and 24 <= height <= 52):
            continue
        aspect = width / max(1.0, float(height))
        if not 0.68 <= aspect <= 1.48:
            continue
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area < 70.0 or perimeter <= 0.0:
            continue
        circularity = 4.0 * math.pi * area / max(1e-6, perimeter * perimeter)
        if circularity < 0.48:
            continue
        candidates.append(
            {
                "position": [x + width / 2.0, y + height / 2.0],
                "box": [float(x), float(y), float(x + width), float(y + height)],
                "confidence": 0.99,
                "radius": max(1, int(round((width + height) / 4.0))),
                "source": "overlay_ball_outline",
            }
        )
    return sorted(candidates, key=lambda item: item["position"])


def _racket_box_candidates(masks: List, frame_shape) -> List[Dict]:
    frame_height, frame_width = frame_shape
    candidates = []
    for mask in masks:
        for contour in _contours(mask):
            x, y, width, height = cv2.boundingRect(contour)
            if width < 28 or height < 20:
                continue
            if width > frame_width * 0.45 or height > frame_height * 0.45:
                continue
            area = float(cv2.contourArea(contour))
            box_area = float(width * height)
            if box_area < 500.0 or area / max(1.0, box_area) < 0.68:
                continue
            aspect = height / max(1.0, float(width))
            if not 0.20 <= aspect <= 5.0:
                continue
            candidate = {
                "box": [int(x), int(y), int(x + width), int(y + height)],
                "confidence": 0.99,
                "class_name": "tennis racket",
                "area": int(box_area),
                "aspect_ratio": round(aspect, 2),
                "source": "overlay_racket_box",
            }
            if not any(_box_iou(candidate["box"], kept["box"]) >= 0.70 for kept in candidates):
                candidates.append(candidate)
    return sorted(candidates, key=lambda item: item["box"])


def _box_iou(left, right) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    return intersection / max(1e-6, left_area + right_area - intersection)
