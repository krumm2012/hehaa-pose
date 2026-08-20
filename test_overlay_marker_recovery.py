import unittest

import cv2
import numpy as np

from overlay_marker_recovery import recover_overlay_detections
from video_overlay_primitives import draw_ball_outline


class OverlayMarkerRecoveryTests(unittest.TestCase):
    def test_recovers_current_ball_outline_and_racket_box(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        draw_ball_outline(frame, (180, 220))
        cv2.rectangle(frame, (320, 120), (440, 210), (255, 128, 0), 2)

        recovered = recover_overlay_detections(
            frame,
            {"overlay_marker_recovery_enabled": True},
        )

        self.assertEqual(len(recovered["balls"]), 1)
        self.assertAlmostEqual(recovered["balls"][0]["position"][0], 180, delta=2)
        self.assertAlmostEqual(recovered["balls"][0]["position"][1], 220, delta=2)
        self.assertEqual(recovered["balls"][0]["source"], "overlay_ball_outline")
        self.assertEqual(len(recovered["rackets"]), 1)
        self.assertEqual(recovered["rackets"][0]["source"], "overlay_racket_box")

    def test_ignores_long_roi_lines_and_open_pose_strokes(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.line(frame, (20, 300), (620, 300), (0, 255, 255), 3)
        cv2.line(frame, (80, 40), (90, 260), (0, 255, 0), 4)
        cv2.line(frame, (90, 150), (180, 210), (0, 255, 0), 4)

        recovered = recover_overlay_detections(
            frame,
            {
                "overlay_marker_recovery_enabled": True,
                "overlay_legacy_green_racket_enabled": True,
            },
        )

        self.assertEqual(recovered["balls"], [])
        self.assertEqual(recovered["rackets"], [])

    def test_ignores_small_filled_yellow_tennis_ball(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (160, 120), 7, (0, 255, 255), -1, cv2.LINE_AA)

        recovered = recover_overlay_detections(
            frame,
            {"overlay_marker_recovery_enabled": True},
        )

        self.assertEqual(recovered["balls"], [])

    def test_disabled_recovery_is_a_noop(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        draw_ball_outline(frame, (80, 60))

        recovered = recover_overlay_detections(frame, {})

        self.assertEqual(recovered["balls"], [])
        self.assertEqual(recovered["rackets"], [])


if __name__ == "__main__":
    unittest.main()
