import unittest

import numpy as np

from video_overlay_primitives import BALL_MARKER_RADIUS, draw_ball_outline


class BallOutlineOverlayTests(unittest.TestCase):
    def test_preserves_ball_center_and_draws_only_an_outer_ring(self):
        frame = np.full((80, 80, 3), (34, 170, 92), dtype=np.uint8)
        original = frame.copy()

        draw_ball_outline(frame, (40, 40))

        np.testing.assert_array_equal(frame[35:46, 35:46], original[35:46, 35:46])
        self.assertFalse(
            np.array_equal(
                frame[40, 40 + BALL_MARKER_RADIUS],
                original[40, 40 + BALL_MARKER_RADIUS],
            )
        )

    def test_handles_fractional_coordinates_near_frame_edge(self):
        frame = np.zeros((24, 24, 3), dtype=np.uint8)

        draw_ball_outline(frame, (1.6, 2.4))

        np.testing.assert_array_equal(frame[2, 2], np.zeros(3, dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
