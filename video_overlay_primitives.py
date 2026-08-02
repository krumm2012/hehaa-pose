"""Shared non-destructive drawing primitives for analysis video overlays."""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2


BALL_MARKER_RADIUS = 14
BALL_MARKER_COLOR = (0, 255, 255)
BALL_MARKER_CONTRAST_COLOR = (20, 20, 20)


def draw_ball_outline(
    frame,
    center: Sequence[float],
    *,
    radius: int = BALL_MARKER_RADIUS,
    color: Tuple[int, int, int] = BALL_MARKER_COLOR,
) -> None:
    """Draw a visible hollow marker while preserving pixels over the tennis ball."""
    if frame is None or center is None or len(center) < 2:
        return
    location = (int(round(float(center[0]))), int(round(float(center[1]))))
    safe_radius = max(4, int(radius))
    cv2.circle(
        frame,
        location,
        safe_radius,
        BALL_MARKER_CONTRAST_COLOR,
        5,
        cv2.LINE_AA,
    )
    cv2.circle(frame, location, safe_radius, color, 2, cv2.LINE_AA)
