import unittest

from pose_estimator_yolo26 import PoseEstimatorYOLO26


def _make_estimator():
    est = PoseEstimatorYOLO26.__new__(PoseEstimatorYOLO26)
    est.keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    est.pose_smoothing_enabled = True
    est.pose_smoothing_alpha = 0.5
    est.pose_smoothing_max_jump_px = 20.0
    est.pose_smoothing_hold_missing_frames = 2
    est.pose_smoothing_reset_frames = 3
    est.pose_smoothing_min_valid_points = 1
    est._smoothed_keypoints = {}
    est._missing_counts = {name: 0 for name in est.keypoint_names}
    est._no_person_frames = 0
    return est


class PoseTemporalSmoothingTests(unittest.TestCase):
    def test_clamps_large_jump(self):
        est = _make_estimator()
        f1 = [{"right_wrist": (100, 100)}]
        f2 = [{"right_wrist": (200, 100)}]
        out1 = est._apply_temporal_smoothing(f1)
        out2 = est._apply_temporal_smoothing(f2)
        self.assertEqual(out1[0]["right_wrist"], (100, 100))
        # jump 100px is clamped to 20px then EMA alpha=0.5 => +10
        self.assertEqual(out2[0]["right_wrist"], (110, 100))

    def test_holds_missing_for_short_gap(self):
        est = _make_estimator()
        est._apply_temporal_smoothing([{"left_wrist": (300, 400), "right_wrist": (320, 410)}])
        out_missing = est._apply_temporal_smoothing([{"left_wrist": None, "right_wrist": (321, 411)}])
        self.assertEqual(out_missing[0]["left_wrist"], (300, 400))

    def test_resets_after_no_person_frames(self):
        est = _make_estimator()
        est._apply_temporal_smoothing([{"left_wrist": (50, 60)}])
        est._apply_temporal_smoothing([])
        est._apply_temporal_smoothing([])
        est._apply_temporal_smoothing([])
        self.assertEqual(est._smoothed_keypoints, {})


if __name__ == "__main__":
    unittest.main()
