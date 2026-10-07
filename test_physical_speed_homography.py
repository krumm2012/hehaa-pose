import math
import unittest
from ground_reference import fit_homography
from image_motion_measurements import racket_physical_velocity
from swing_motion_features import extract_motion_features
from swing_biomechanics import _calculate_extended_tier_biomechanics, aggregate_event_biomechanics


def source_frame(i, x, y, t):
    return {
        "frame_id": i,
        "timestamp": t,
        "source_time": {
            "schema_version": "tennis.source-time.v1",
            "source_kind": "video_file",
            "source_frame_id": i,
            "timestamp_seconds": t,
            "basis": "media_pts",
            "quality": "reported",
        },
        "pose": {
            "left_shoulder": [100, 100],
            "right_shoulder": [140, 100],
            "left_hip": [100, 180],
            "right_hip": [140, 180],
        },
        "racket": [x - 10, y - 10, x + 10, y + 10],
    }


class PhysicalSpeedHomographyTests(unittest.TestCase):
    def setUp(self):
        # Calibrated 4 points: 100px = 1.0m (0.01m per pixel scale)
        self.img_pts = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
        self.world_pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        self.H = fit_homography(self.img_pts, self.world_pts)

    def test_uncalibrated_speed_returns_none(self):
        f0 = source_frame(0, 10, 20, 0.0)
        f1 = source_frame(1, 50, 20, 0.04)
        mps, kmh, status = racket_physical_velocity(f0, f1, [10, 20], [50, 20], homography=None)
        self.assertIsNone(mps)
        self.assertIsNone(kmh)
        self.assertEqual(status, "uncalibrated")

        features = extract_motion_features([f0, f1], homography=None)
        self.assertIsNone(features[-1]["racket_head_speed_kmh"])
        self.assertIsNone(features[-1]["racket_speed_mps"])
        self.assertEqual(features[-1]["racket_speed_calibration_status"], "uncalibrated")

    def test_calibrated_homography_exact_metric_speed(self):
        # 40 pixels displacement in 0.04s = 1000 px/s
        # With 0.01m/px scale, 40 px = 0.40m
        # Metric speed = 0.40m / 0.04s = 10.0 m/s = 36.0 km/h
        f0 = source_frame(0, 10, 20, 0.0)
        f1 = source_frame(1, 50, 20, 0.04)
        mps, kmh, status = racket_physical_velocity(f0, f1, [10, 20], [50, 20], homography=self.H)
        self.assertAlmostEqual(mps, 10.0, places=2)
        self.assertAlmostEqual(kmh, 36.0, places=2)
        self.assertEqual(status, "homography_ground_calibrated")

    def test_extract_motion_features_with_passed_homography(self):
        f0 = source_frame(0, 10, 20, 0.0)
        f1 = source_frame(1, 50, 20, 0.04)
        features = extract_motion_features([f0, f1], homography=self.H)
        self.assertAlmostEqual(features[1]["racket_speed_mps"], 10.0, places=2)
        self.assertAlmostEqual(features[1]["racket_head_speed_kmh"], 36.0, places=2)
        self.assertEqual(features[1]["racket_speed_calibration_status"], "homography_ground_calibrated")

    def test_extract_motion_features_auto_discovers_ground_calibration_from_frames(self):
        cal = {
            "views": {
                "front": {
                    "H": self.H,
                }
            }
        }
        f0 = source_frame(0, 10, 20, 0.0)
        f0["ground_calibration"] = cal
        f1 = source_frame(1, 50, 20, 0.04)
        f1["ground_calibration"] = cal

        features = extract_motion_features([f0, f1])
        self.assertAlmostEqual(features[1]["racket_speed_mps"], 10.0, places=2)
        self.assertAlmostEqual(features[1]["racket_head_speed_kmh"], 36.0, places=2)

    def test_biomechanics_incorporates_calibrated_speed(self):
        f0 = source_frame(0, 10, 20, 0.0)
        f1 = source_frame(1, 50, 20, 0.04)
        features = extract_motion_features([f0, f1], homography=self.H)

        bio = _calculate_extended_tier_biomechanics(
            features, start_frame=0, contact_frame=1, end_frame=1, body_width=40.0
        )
        rkt = bio["racket_head_speed"]
        self.assertEqual(rkt["status"], "ground_homography_calibrated")
        self.assertEqual(rkt["observability"], "ground_plane_projected_speed")
        self.assertAlmostEqual(rkt["contact_kmh"], 36.0, places=1)
        self.assertAlmostEqual(rkt["max_kmh"], 36.0, places=1)
        self.assertGreater(rkt["confidence"], 0.8)

    def test_nonconsecutive_and_invalid_timestamps_refuse_speed(self):
        f0 = source_frame(0, 10, 20, 0.0)
        f2 = source_frame(2, 50, 20, 0.08)  # Gap: frame 0 to frame 2
        mps, kmh, status = racket_physical_velocity(f0, f2, [10, 20], [50, 20], homography=self.H)
        self.assertIsNone(mps)
        self.assertIsNone(kmh)
        self.assertEqual(status, "nonconsecutive_observations")

        f_rev = source_frame(1, 50, 20, -0.04)  # Negative or backward time
        mps_rev, _, status_rev = racket_physical_velocity(f0, f_rev, [10, 20], [50, 20], homography=self.H)
        self.assertIsNone(mps_rev)
        self.assertEqual(status_rev, "invalid_source_time")


if __name__ == "__main__":
    unittest.main()
