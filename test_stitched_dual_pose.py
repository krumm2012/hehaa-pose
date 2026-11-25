import unittest
from types import SimpleNamespace
import numpy as np

from dual_pose_estimator import DualPoseEstimator
from dual_view_manager import DualViewCropInfo, DualViewFrame
from pose_estimator_yolo26 import PoseEstimatorYOLO26


class _FakeCoreMLModel:
    def __init__(self, output_tensor: np.ndarray):
        self.output_tensor = output_tensor
        self.calls = 0
        self.last_image = None

    def predict(self, inputs):
        self.calls += 1
        self.last_image = inputs["image"]
        return {"var_out": self.output_tensor}


def _make_det(bbox, kp_map, conf=0.95):
    row = np.zeros(6 + 17 * 3, dtype=np.float32)
    row[0:4] = bbox
    row[4] = conf
    row[5] = 0.0
    names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    for idx, name in enumerate(names):
        if name in kp_map:
            x, y, c = kp_map[name]
            row[6 + idx * 3] = x
            row[6 + idx * 3 + 1] = y
            row[6 + idx * 3 + 2] = c
    return row


class StitchedDualPoseTests(unittest.TestCase):
    def _make_stub_estimator(self, fake_model):
        est = PoseEstimatorYOLO26.__new__(PoseEstimatorYOLO26)
        est.model = fake_model
        est.input_size = (640, 640)
        est.confidence_threshold = 0.4
        est.keypoint_confidence = 0.3
        est.preprocess_mode = "letterbox"
        est.pose_smoothing_enabled = False
        est.roi_manager = None
        est.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]
        return est

    def test_stitched_single_pass_maps_left_and_right_halves_with_subpixel_precision(self):
        # front: 1080x1440 (3:4) -> slot_w=316, in_h=640 -> gain=316/1080, rw=316, rh=421, ox=0, oy=109
        # back:  1080x1440 (3:4) -> slot_x0=324, slot_w=316 -> gain=316/1080, rw=316, rh=421, ox=324, oy=109
        front_frame = np.zeros((1440, 1080, 3), dtype=np.uint8)
        back_frame = np.zeros((1440, 1080, 3), dtype=np.uint8)
        gain = 316.0 / 1080.0
        oy = (640 - round(1440 * gain)) // 2

        # Front shoulder at canonical (270.5, 360.25) in (540, 720) -> source (541.0, 720.5)
        fx_canvas = 0 + 541.0 * gain
        fy_canvas = oy + 720.5 * gain
        # Back shoulder at canonical (200.25, 300.5) in (540, 720) -> source (400.5, 601.0)
        bx_canvas = 324 + 400.5 * gain
        by_canvas = oy + 601.0 * gain

        det_front = _make_det(
            [fx_canvas - 40, fy_canvas - 40, fx_canvas + 40, fy_canvas + 80],
            {"left_shoulder": (fx_canvas, fy_canvas, 0.92)},
        )
        det_back = _make_det(
            [bx_canvas - 40, by_canvas - 40, bx_canvas + 40, by_canvas + 80],
            {"right_shoulder": (bx_canvas, by_canvas, 0.88)},
        )
        out = np.stack([det_front, det_back], axis=0)[None, :, :]
        fake = _FakeCoreMLModel(out)
        est = self._make_stub_estimator(fake)

        front_list, back_list = est.get_keypoints_from_stitched_views(
            front_frame, (540, 720), back_frame, (540, 720), gap_px=8
        )

        self.assertEqual(fake.calls, 1)
        self.assertEqual(len(front_list), 1)
        self.assertEqual(len(back_list), 1)
        fls = front_list[0]["left_shoulder"]
        brs = back_list[0]["right_shoulder"]
        self.assertAlmostEqual(fls[0], 270.5, places=3)
        self.assertAlmostEqual(fls[1], 360.25, places=3)
        self.assertAlmostEqual(fls[2], 0.92, places=3)
        self.assertAlmostEqual(brs[0], 200.25, places=3)
        self.assertAlmostEqual(brs[1], 300.5, places=3)
        self.assertAlmostEqual(brs[2], 0.88, places=3)

    def test_dual_pose_estimator_stitched_preserves_backview_anatomical_guard(self):
        # Construct DualPoseEstimator in mock mode, then wire a stub Core ML stitched model
        dpe = DualPoseEstimator(backend="mock", stitched_single_pass=True)
        dpe.backend = "coreml"

        # Back view has X-crossed hips (left_shoulder > right_shoulder, but left_hip < right_hip)
        front_cand = {
            "left_shoulder": (320.0, 200.0, 0.95),
            "right_shoulder": (220.0, 200.0, 0.95),
            "left_hip": (310.0, 380.0, 0.95),
            "right_hip": (230.0, 380.0, 0.95),
        }
        back_cand_crossed = {
            "left_shoulder": (320.0, 200.0, 0.95),
            "right_shoulder": (220.0, 200.0, 0.95),
            "left_hip": (230.0, 380.0, 0.95),   # inverted!
            "right_hip": (310.0, 380.0, 0.95),  # inverted!
        }

        calls = {"n": 0}
        def _fake_stitched(ff, fsize, bf, bsize):
            calls["n"] += 1
            return [front_cand], [back_cand_crossed]

        dpe.model_front = SimpleNamespace(get_keypoints_from_stitched_views=_fake_stitched)
        dpe.model_back = dpe.model_front

        crop = DualViewCropInfo(bbox_orig=(0, 0, 540, 720), crop_size=(540, 720), view_size=(540, 720))
        frame = np.zeros((720, 540, 3), dtype=np.uint8)
        dvf = DualViewFrame(
            frame_id=1,
            original_frame=frame,
            front_frame=frame,
            back_frame=frame,
            front_info=crop,
            back_info=crop,
            is_point_in_mirror_fn=lambda x, y: True,
        )

        res = dpe.estimate_dual_pose(dvf)
        self.assertEqual(calls["n"], 1)
        # Anatomical consistency guard must uncross the inverted back hips
        self.assertGreater(res.back_pose_local["left_hip"].x, res.back_pose_local["right_hip"].x)


if __name__ == "__main__":
    unittest.main()
