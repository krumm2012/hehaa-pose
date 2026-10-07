"""
Unit tests for production robustness against degenerate homographies,
ill-conditioned matrices, singularity horizon crossings, and unphysical speed spikes.
"""
import math
import unittest
import numpy as np

from ground_reference import fit_homography, map_point, validate_homography_matrix
from image_motion_measurements import racket_physical_velocity
from mirror_geometry import (
    extract_ground_mirror_homography,
    estimate_torso_planar_affine,
    map_mirror_point_planar,
)
from swing_motion_features import extract_motion_features


def make_source_frame(frame_id, x, y, timestamp_seconds, racket_height_m=None):
    f = {
        "frame_id": frame_id,
        "timestamp": timestamp_seconds,
        "source_time": {
            "schema_version": "tennis.source-time.v1",
            "source_kind": "video_file",
            "source_frame_id": frame_id,
            "timestamp_seconds": timestamp_seconds,
            "basis": "media_pts",
            "quality": "reported",
        },
        "pose": {
            "left_shoulder": [100.0, 100.0],
            "right_shoulder": [140.0, 100.0],
            "left_hip": [100.0, 180.0],
            "right_hip": [140.0, 180.0],
            "right_wrist": [x - 40.0, y],
        },
        "racket": [x - 10.0, y - 10.0, x + 10.0, y + 10.0],
    }
    if racket_height_m is not None:
        f["racket_height_m"] = racket_height_m
    return f


class HomographyRobustnessTests(unittest.TestCase):
    def setUp(self):
        # Well-conditioned 100px = 1.0m homography
        self.img_pts = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
        self.world_pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        self.H_good = fit_homography(self.img_pts, self.world_pts)

    def test_validate_homography_matrix_well_conditioned(self):
        is_valid, cond, det, status = validate_homography_matrix(self.H_good)
        self.assertTrue(is_valid)
        self.assertLess(cond, 1e4)
        self.assertGreater(abs(det), 1e-8)
        self.assertEqual(status, "well_conditioned")

    def test_validate_homography_matrix_detects_singular_and_ill_conditioned(self):
        # 1. Singular matrix (det = 0)
        H_singular = np.zeros((3, 3), dtype=float)
        is_valid, _, _, status = validate_homography_matrix(H_singular)
        self.assertFalse(is_valid)
        self.assertIn("singular", status)

        # 2. NaN or Inf elements
        H_nan = np.array([[1.0, 0.0, np.nan], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        is_valid, _, _, status = validate_homography_matrix(H_nan)
        self.assertFalse(is_valid)
        self.assertEqual(status, "non_finite_elements")

        # 3. Near-collinear / ill-conditioned matrix
        H_ill = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.00000001, 3.0],
            [0.0, 0.0, 1.0]
        ], dtype=float)
        is_valid, cond, _, status = validate_homography_matrix(H_ill, max_condition_number=1e5)
        self.assertFalse(is_valid)
        self.assertTrue("ill_conditioned" in status or "singular" in status)

    def test_racket_physical_velocity_degrades_on_ill_conditioned_matrix(self):
        f0 = make_source_frame(0, 10.0, 20.0, 0.0)
        f1 = make_source_frame(1, 50.0, 20.0, 0.04)

        # Pass singular homography
        H_bad = np.zeros((3, 3))
        mps, kmh, status = racket_physical_velocity(
            f0, f1, [10.0, 20.0], [50.0, 20.0], homography=H_bad
        )
        self.assertIsNone(mps)
        self.assertIsNone(kmh)
        self.assertTrue(status.startswith("degraded_ill_conditioned_homography"))

    def test_racket_physical_velocity_degrades_on_horizon_singularity(self):
        f0 = make_source_frame(0, 10.0, 20.0, 0.0)
        f1 = make_source_frame(1, 50.0, 20.0, 0.04)

        # Create homography where denominator w = H[2] @ [x, y, 1] = 0 around y=20
        # H[2] = [0.0, -0.05, 1.0] -> for y=20, w = 1 - 1 = 0
        H_horizon = np.array([
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, -0.05, 1.0]
        ])
        mps, kmh, status = racket_physical_velocity(
            f0, f1, [10.0, 20.0], [50.0, 20.0], homography=H_horizon
        )
        self.assertIsNone(mps)
        self.assertIsNone(kmh)
        self.assertTrue("degraded_homography_mapping_failed" in status or "degraded_ill_conditioned_homography" in status)

    def test_racket_physical_velocity_degrades_on_out_of_bounds_projection(self):
        f0 = make_source_frame(0, 10.0, 20.0, 0.0)
        f1 = make_source_frame(1, 50.0, 20.0, 0.04)

        # Scale is 10.0m per pixel -> 50px = 500m (outside [-100m, 100m])
        H_huge = np.diag([10.0, 10.0, 1.0])
        mps, kmh, status = racket_physical_velocity(
            f0, f1, [10.0, 20.0], [50.0, 20.0], homography=H_huge
        )
        self.assertIsNone(mps)
        self.assertIsNone(kmh)
        self.assertTrue(status.startswith("degraded_projection_out_of_bounds"))

    def test_racket_physical_velocity_degrades_on_unphysical_displacement(self):
        f0 = make_source_frame(0, 10.0, 20.0, 0.0)
        # Shift 600 pixels in 0.04s. At 0.01m/px scale, dist = 6.0m (> 4.0m threshold)
        # 6.0m in 0.04s is 150 m/s = 540 km/h (unphysical jump)
        f1 = make_source_frame(1, 610.0, 20.0, 0.04)
        mps, kmh, status = racket_physical_velocity(
            f0, f1, [10.0, 20.0], [610.0, 20.0], homography=self.H_good
        )
        self.assertIsNone(mps)
        self.assertIsNone(kmh)
        self.assertTrue(status.startswith("degraded_unphysical_displacement"))

    def test_mirror_geometry_rejects_collinear_torso_pose(self):
        # Front pose: normal
        front_pose = {
            "left_shoulder": [100.0, 100.0],
            "right_shoulder": [140.0, 100.0],
            "left_hip": [100.0, 180.0],
            "right_hip": [140.0, 180.0],
        }
        # Back pose: completely collinear (all y coords nearly identical, w < 5px)
        back_pose_collinear = {
            "left_shoulder": [100.0, 100.0],
            "right_shoulder": [101.0, 100.0],
            "left_hip": [102.0, 100.0],
            "right_hip": [103.0, 100.0],
        }
        affine = estimate_torso_planar_affine(front_pose, back_pose_collinear)
        self.assertIsNone(affine)

    def test_extract_ground_mirror_homography_rejects_degraded_views(self):
        # Calibration where front view H is singular
        bad_cal = {
            "views": {
                "front": {"H": np.zeros((3, 3)).tolist()},
                "back": {"H": self.H_good},
            }
        }
        h_comp = extract_ground_mirror_homography(bad_cal)
        self.assertIsNone(h_comp)

    def test_map_mirror_point_planar_rejects_extreme_out_of_bounds(self):
        # Ground homography that blows point up to (100000, 100000)
        H_blowup = np.diag([1000.0, 1000.0, 1.0])
        mapped = map_mirror_point_planar((50.0, 50.0), ground_homography=H_blowup)
        self.assertIsNone(mapped)

    def test_motion_features_pipeline_fallback_to_pixel_speed_when_homography_degraded(self):
        f0 = make_source_frame(0, 10.0, 20.0, 0.0)
        f1 = make_source_frame(1, 50.0, 20.0, 0.04)

        # Pass singular H
        features = extract_motion_features([f0, f1], homography=np.zeros((3, 3)))
        # Pixel velocity is still measured stably! (40px / 0.04s = 1000 px/s)
        self.assertAlmostEqual(features[1]["racket_speed_px_s"], 1000.0, places=1)
        # Metric speed gracefully falls back to None, status reflects degradation
        self.assertIsNone(features[1]["racket_head_speed_kmh"])
        self.assertIsNone(features[1]["racket_speed_mps"])
        self.assertTrue("degraded" in features[1]["racket_speed_calibration_status"])


if __name__ == "__main__":
    unittest.main()
