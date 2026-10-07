import math
import unittest
from ground_reference import fit_homography, map_point, map_point_at_height
from image_motion_measurements import racket_physical_velocity
from kinematic_smoothing import (
    clamp_velocity_step,
    clamp_centripetal_speed,
    wrist_racket_geometric_consistency,
    MAX_TANGENTIAL_ACCEL_MPS2,
    MAX_CENTRIPETAL_ACCEL_MPS2,
)
from swing_motion_features import extract_motion_features
from swing_biomechanics import _calculate_extended_tier_biomechanics, aggregate_event_biomechanics


def source_frame(i, x, y, t, racket_height_m=None):
    f = {
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
            "right_wrist": [x - 40, y],
        },
        "racket": [x - 10, y - 10, x + 10, y + 10],
    }
    if racket_height_m is not None:
        f["racket_height_m"] = racket_height_m
    return f


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

    def test_height_compensated_debiasing(self):
        # At elevation h = 0.96m, with camera height H_cam = 2.40m:
        # Perspective dilation debias factor = 1.0 - 0.96 / 2.40 = 0.60
        # Ground distance 0.40m debiases to 0.40 * 0.60 = 0.24m
        # Debiased speed = 0.24m / 0.04s = 6.0 m/s = 21.6 km/h
        f0 = source_frame(0, 10, 20, 0.0, racket_height_m=0.96)
        f1 = source_frame(1, 50, 20, 0.04, racket_height_m=0.96)
        mps, kmh, status = racket_physical_velocity(
            f0, f1, [10, 20], [50, 20], homography=self.H, height_m=0.96, camera_height_m=2.4
        )
        self.assertAlmostEqual(mps, 6.0, places=2)
        self.assertAlmostEqual(kmh, 21.6, places=2)
        self.assertEqual(status, "homography_height_debiased")

    def test_map_point_at_height_removes_dilation(self):
        pt_img = [50.0, 50.0]
        # At ground level (h=0), map_point_at_height matches map_point exactly
        pt_ground = map_point(self.H, pt_img)
        pt_h0 = map_point_at_height(self.H, pt_img, height_m=0.0)
        self.assertAlmostEqual(pt_ground[0], pt_h0[0], places=3)
        self.assertAlmostEqual(pt_ground[1], pt_h0[1], places=3)

        # At h = 1.2m, H_cam = 2.4m, scale = 1 - 1.2/2.4 = 0.50
        pt_h12 = map_point_at_height(self.H, pt_img, height_m=1.2, camera_height_m=2.4)
        self.assertAlmostEqual(pt_h12[0], pt_ground[0] * 0.5, places=3)
        self.assertAlmostEqual(pt_h12[1], pt_ground[1] * 0.5, places=3)

    def test_centripetal_clamping_prevents_anomalous_spikes(self):
        # Human max centripetal speed ceiling for R=1.2m is sqrt(1050 * 1.2) ≈ 35.5 m/s (~127.8 km/h)
        normal_speed = 25.0
        clamped_normal, was_clamped = clamp_centripetal_speed(normal_speed, radius_m=1.2)
        self.assertEqual(clamped_normal, 25.0)
        self.assertFalse(was_clamped)

        spike_speed = 80.0  # Impossible 288 km/h racket jump
        clamped_spike, was_clamped_spike = clamp_centripetal_speed(spike_speed, radius_m=1.2)
        self.assertTrue(was_clamped_spike)
        self.assertLess(clamped_spike, 36.0)

    def test_clamp_velocity_step(self):
        # At dt=0.04s, max acceleration 300 m/s^2 allows max delta = 12.0 m/s
        v_prev = 10.0
        v_ok = 18.0  # delta = 8.0 <= 12.0
        clamped_ok, changed_ok = clamp_velocity_step(v_prev, v_ok, dt=0.04, max_accel=300.0)
        self.assertEqual(clamped_ok, 18.0)
        self.assertFalse(changed_ok)

        v_spike = 40.0  # delta = 30.0 > 12.0
        clamped_spike, changed_spike = clamp_velocity_step(v_prev, v_spike, dt=0.04, max_accel=300.0)
        self.assertEqual(clamped_spike, 22.0)  # 10.0 + 12.0
        self.assertTrue(changed_spike)

    def test_wrist_racket_geometric_consistency(self):
        wrist = (100.0, 100.0)
        racket_ok = (150.0, 100.0)  # dist = 50px ∈ [15, 180]
        corrected_ok, adjusted_ok = wrist_racket_geometric_consistency(racket_ok, wrist)
        self.assertEqual(corrected_ok, racket_ok)
        self.assertFalse(adjusted_ok)

        racket_drift = (400.0, 100.0)  # dist = 300px > 180px
        prev_w = (90.0, 100.0)
        prev_r = (140.0, 100.0)
        corrected_drift, adjusted_drift = wrist_racket_geometric_consistency(
            racket_drift, wrist, prev_racket_point=prev_r, prev_wrist_point=prev_w
        )
        self.assertTrue(adjusted_drift)
        # Predicted follows wrist dx = +10px: 140 + 10 = 150px
        self.assertEqual(corrected_drift, (150.0, 100.0))

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

    def test_biomechanics_reflects_height_debiased_status(self):
        f0 = source_frame(0, 10, 20, 0.0, racket_height_m=0.96)
        f1 = source_frame(1, 50, 20, 0.04, racket_height_m=0.96)
        features = extract_motion_features([f0, f1], homography=self.H)

        bio = _calculate_extended_tier_biomechanics(
            features, start_frame=0, contact_frame=1, end_frame=1, body_width=40.0
        )
        rkt = bio["racket_head_speed"]
        self.assertEqual(rkt["status"], "homography_height_debiased")
        self.assertEqual(rkt["observability"], "height_debiased_projected_speed")
        self.assertAlmostEqual(rkt["contact_kmh"], 21.6, places=1)
        self.assertAlmostEqual(rkt["max_kmh"], 21.6, places=1)
        self.assertGreater(rkt["confidence"], 0.85)

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
