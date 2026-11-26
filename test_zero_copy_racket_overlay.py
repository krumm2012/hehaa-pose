import unittest
import cv2
import numpy as np

from racket_pose_detector import KEYPOINT_NAMES, SKELETON_EDGES, draw_racket_pose


def _legacy_full_frame_draw(canvas, observation):
    if not observation or observation.get("status") != "observed":
        return canvas
    overlay = canvas.copy()
    for candidate in observation.get("candidates", []):
        pts = candidate.get("keypoints") or {}
        for a, b in SKELETON_EDGES:
            p, q = pts.get(KEYPOINT_NAMES[a], {}), pts.get(KEYPOINT_NAMES[b], {})
            if p.get("observed") and q.get("observed"):
                cv2.line(
                    overlay,
                    (round(p["x"]), round(p["y"])),
                    (round(q["x"]), round(q["y"])),
                    (0, 210, 245),
                    1,
                    cv2.LINE_AA,
                )
        for name, p in pts.items():
            if p.get("observed"):
                r = 3 if name == "sweet_spot" else 2
                cv2.circle(
                    overlay,
                    (round(p["x"]), round(p["y"])),
                    r,
                    (0, 0, 240) if name == "sweet_spot" else (0, 210, 245),
                    -1,
                    cv2.LINE_AA,
                )
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
    return canvas


def _sample_observation():
    return {
        "status": "observed",
        "candidates": [
            {
                "keypoints": {
                    "butt_cap": {"x": 200.2, "y": 300.4, "observed": True},
                    "grip_center": {"x": 210.5, "y": 285.1, "observed": True},
                    "throat": {"x": 222.0, "y": 265.0, "observed": True},
                    "sweet_spot": {"x": 240.8, "y": 235.6, "observed": True},
                    "head_top": {"x": 260.0, "y": 205.0, "observed": True},
                    "frame_left": {"x": 225.0, "y": 230.0, "observed": True},
                    "frame_right": {"x": 255.0, "y": 240.0, "observed": True},
                }
            }
        ],
    }


class ZeroCopyRacketOverlayTests(unittest.TestCase):
    def test_roi_blending_matches_full_frame_legacy_and_preserves_outside_pixels(self):
        rng = np.random.default_rng(42)
        base = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
        obs = _sample_observation()

        expected = _legacy_full_frame_draw(base.copy(), obs)
        actual = draw_racket_pose(base.copy(), obs)

        np.testing.assert_array_equal(actual, expected)
        # Outside padded bbox (~192..268 x 197..309) must remain strictly untouched
        np.testing.assert_array_equal(actual[:150, :], base[:150, :])
        np.testing.assert_array_equal(actual[:, 350:], base[:, 350:])

    def test_coord_mapper_draws_front_view_only_and_skips_out_of_bounds_back_view(self):
        front = np.full((720, 540, 3), 30, dtype=np.uint8)
        back = np.full((720, 540, 3), 30, dtype=np.uint8)
        obs = {
            "status": "observed",
            "candidates": [
                {
                    "keypoints": {
                        "butt_cap": {"x": 1400.0, "y": 700.0, "observed": True},
                        "grip_center": {"x": 1420.0, "y": 670.0, "observed": True},
                        "throat": {"x": 1440.0, "y": 640.0, "observed": True},
                        "sweet_spot": {"x": 1465.0, "y": 600.0, "observed": True},
                        "head_top": {"x": 1490.0, "y": 560.0, "observed": True},
                        "frame_left": {"x": 1445.0, "y": 600.0, "observed": True},
                        "frame_right": {"x": 1485.0, "y": 600.0, "observed": True},
                    }
                }
            ],
        }

        def front_mapper(x, y):
            return (x - 1152.0) * (540.0 / 768.0), (y - 115.0) * (720.0 / 1109.0)

        def back_mapper(x, y):
            return (x - 1431.0) * (540.0 / 349.0), (y - 0.0) * (720.0 / 465.0)

        out_front = draw_racket_pose(front.copy(), obs, coord_mapper=front_mapper)
        out_back = draw_racket_pose(back.copy(), obs, coord_mapper=back_mapper)

        self.assertFalse(np.array_equal(out_front, front))
        # Candidate is below back mirror crop (y=560..700 -> y_view=867..1083 >= 720), so back is untouched
        np.testing.assert_array_equal(out_back, back)

    def test_non_observed_status_is_noop(self):
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        out = draw_racket_pose(canvas, {"status": "missing", "candidates": []})
        np.testing.assert_array_equal(out, canvas)


if __name__ == "__main__":
    unittest.main()
