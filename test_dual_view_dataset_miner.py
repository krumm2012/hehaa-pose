# -*- coding: utf-8 -*-
import copy
import unittest

from dual_view_biomechanics import DualViewBiomechanicsEngine
from dual_view_dataset_miner import (
    RACKET_FLIP_IDX,
    format_yolo_pose_racket_label,
    mine_session_dual_view_data,
)
from joint_racket_2d_analysis import BODY_POINTS, RACKET_POINTS, analyze_frames, summarize
from mirror_geometry import (
    estimate_cross_view_torso_transform,
    map_racket_keypoints_cross_view,
)


def _make_point(x=100.0, y=100.0, fid=0, conf=0.9, observed=True):
    return {
        "x": float(x),
        "y": float(y),
        "confidence": float(conf),
        "observed": bool(observed),
        "source_frame_id": int(fid),
    }


def _make_dual_row(fid=0, t=0.0):
    row = {
        "frame_id": fid,
        "pose_observation_coordinate_space": "original_source_pixels",
        "source_time": {
            "schema_version": "tennis.source-time.v1",
            "source_kind": "video_file",
            "source_frame_id": fid,
            "basis": "media_pts",
            "quality": "reported",
            "timestamp_seconds": float(t),
        },
        "pose_observations": {"front": {}, "back": {}},
        "racket_pose": {
            "frame_id": fid,
            "coordinate_space": "original_source_pixels",
            "status": "observed",
            "candidates": [],
        },
    }
    # Front torso centered at (1500, 500), width 100, height 130
    front_coords = {
        "left_shoulder": (1550.0, 435.0),
        "right_shoulder": (1450.0, 435.0),
        "left_hip": (1545.0, 565.0),
        "right_hip": (1455.0, 565.0),
        "left_elbow": (1580.0, 490.0),
        "right_elbow": (1410.0, 490.0),
        "left_wrist": (1600.0, 520.0),
        "right_wrist": (1380.0, 510.0),
        "left_knee": (1550.0, 680.0),
        "right_knee": (1450.0, 680.0),
        "left_ankle": (1555.0, 800.0),
        "right_ankle": (1445.0, 800.0),
    }
    # Back torso in mirror (scaled down by ~1.65x horizontally, ~1.27x vertically)
    back_coords = {
        "left_shoulder": (1480.0, 120.0),
        "right_shoulder": (1420.0, 120.0),
        "left_hip": (1477.0, 222.0),
        "right_hip": (1423.0, 222.0),
        "left_elbow": (1498.0, 163.0),
        "right_elbow": (1396.0, 163.0),
        "left_wrist": (1510.0, 187.0),
        "right_wrist": (1378.0, 179.0),
        "left_knee": (1480.0, 312.0),
        "right_knee": (1420.0, 312.0),
        "left_ankle": (1483.0, 406.0),
        "right_ankle": (1417.0, 406.0),
    }
    for name in BODY_POINTS:
        fx, fy = front_coords[name]
        bx, by = back_coords[name]
        row["pose_observations"]["front"][name] = _make_point(fx, fy, fid)
        row["pose_observations"]["back"][name] = _make_point(bx, by, fid)

    # Add Front and Back 7-point racket candidates
    f_cand = {
        "source": "racket_pose_model",
        "source_frame_id": fid,
        "observed": True,
        "confidence": 0.92,
        "keypoints": {
            "butt_cap": _make_point(1385.0, 515.0, fid),
            "grip_center": _make_point(1378.0, 508.0, fid),
            "throat": _make_point(1355.0, 490.0, fid),
            "sweet_spot": _make_point(1325.0, 465.0, fid),
            "head_top": _make_point(1295.0, 440.0, fid),
            "frame_left": _make_point(1315.0, 480.0, fid),
            "frame_right": _make_point(1335.0, 450.0, fid),
        },
    }
    b_cand = {
        "source": "racket_pose_model",
        "source_frame_id": fid,
        "observed": True,
        "confidence": 0.88,
        "keypoints": {
            "butt_cap": _make_point(1381.0, 183.0, fid),
            "grip_center": _make_point(1376.0, 177.0, fid),
            "throat": _make_point(1362.0, 163.0, fid),
            "sweet_spot": _make_point(1344.0, 143.0, fid),
            "head_top": _make_point(1326.0, 124.0, fid),
            "frame_left": _make_point(1338.0, 155.0, fid),
            "frame_right": _make_point(1350.0, 132.0, fid),
        },
    }
    row["racket_pose"]["candidates"] = [f_cand, b_cand]
    return row


class DualViewMergingAndMiningTests(unittest.TestCase):
    def test_bidirectional_torso_transform_and_side_on_fallback(self):
        r = _make_dual_row(0)
        f_pose = r["pose_observations"]["front"]
        b_pose = r["pose_observations"]["back"]

        t_b2f = estimate_cross_view_torso_transform(b_pose, f_pose, direction="back_to_front")
        t_f2b = estimate_cross_view_torso_transform(f_pose, b_pose, direction="front_to_back")
        self.assertIsNotNone(t_b2f)
        self.assertIsNotNone(t_f2b)
        self.assertEqual(t_b2f.status, "optimal")
        self.assertAlmostEqual(t_b2f.scale_x * t_f2b.scale_x, 1.0, delta=0.05)

        # Simulate side-on shoulder/hip collapse (< 5px width) -> fallback to anisotropic spine scale
        collapsed_back = copy.deepcopy(b_pose)
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            collapsed_back[name]["x"] = 1450.0
        t_fb = estimate_cross_view_torso_transform(collapsed_back, f_pose, direction="back_to_front")
        self.assertIsNotNone(t_fb)
        self.assertEqual(t_fb.status, "fallback")
        self.assertGreater(t_fb.scale_x, t_fb.scale_y * 1.2)

    def test_cross_view_7pt_racket_mapping(self):
        r = _make_dual_row(0)
        f_pose = r["pose_observations"]["front"]
        b_pose = r["pose_observations"]["back"]
        b_kpts = r["racket_pose"]["candidates"][1]["keypoints"]

        mapped_front, tf = map_racket_keypoints_cross_view(
            b_kpts, b_pose, f_pose, direction="back_to_front"
        )
        self.assertIsNotNone(tf)
        self.assertEqual(len(mapped_front), 7)
        # Projected sweet_spot should land close to actual front sweet_spot (1325, 465)
        ss_x, ss_y = mapped_front["sweet_spot"]
        self.assertAlmostEqual(ss_x, 1325.0, delta=6.0)
        self.assertAlmostEqual(ss_y, 465.0, delta=6.0)

    def test_bidirectional_pose_healing_in_biomechanics_engine(self):
        engine = DualViewBiomechanicsEngine()
        r = _make_dual_row(0)
        f_raw = copy.deepcopy(r["pose_observations"]["front"])
        b_raw = copy.deepcopy(r["pose_observations"]["back"])

        # Occlude front right_wrist and drop back left_wrist
        f_raw["right_wrist"]["confidence"] = 0.10
        b_raw["left_wrist"]["confidence"] = 0.10

        f_parsed = engine.parse_pose_dict({k: (v["x"], v["y"], v["confidence"]) for k, v in f_raw.items()})
        b_parsed = engine.parse_pose_dict({k: (v["x"], v["y"], v["confidence"]) for k, v in b_raw.items()})

        merged = engine.heal_bidirectional_poses(f_parsed, b_parsed)
        self.assertIn("right_wrist", merged["front_healed_joints"])
        self.assertIn("left_wrist", merged["back_healed_joints"])
        self.assertTrue(merged["front_pose"]["right_wrist"].recovered_from_mirror)
        self.assertTrue(merged["back_pose"]["left_wrist"].recovered_from_mirror)

    def test_rescued_racket_when_back_wrist_is_missing(self):
        # Back view has a valid 7-point racket candidate, but back right_wrist and left_wrist dropped out.
        # Single-view fails association with 'no_qualified_wrist', while dual-view fusion rescues it!
        r = _make_dual_row(0)
        r["pose_observations"]["back"]["right_wrist"]["confidence"] = 0.15
        r["pose_observations"]["back"]["left_wrist"]["confidence"] = 0.15

        frames, _ = analyze_frames([r], (2560, 1440))
        f0 = frames[0]
        # Single-view back racket is None
        self.assertIsNone(f0["racket"]["back"]["sweet_spot"])
        self.assertEqual(f0["associations"]["back"]["reason"], "no_qualified_wrist")

        # Dual-view fusion heals back wrist from front and rescues native candidate #1
        fusion = f0["dual_view_fusion"]
        self.assertEqual(fusion["body_provenance"]["back"]["right_wrist"], "front_projected_to_back")
        self.assertEqual(fusion["rescued_candidate_index"]["back"], 1)
        self.assertEqual(fusion["racket_provenance"]["back"], "healed_wrist_candidate_rescue")
        self.assertEqual(fusion["merged_racket"]["back"]["sweet_spot"], [1344.0, 143.0])

    def test_cross_view_occluded_racket_projection_and_yolo_label_mining(self):
        # Frame 0: Back wrist missing -> rescues native back racket
        r0 = _make_dual_row(0, 0.0)
        r0["pose_observations"]["back"]["right_wrist"]["confidence"] = 0.10
        r0["pose_observations"]["back"]["left_wrist"]["confidence"] = 0.10

        # Frame 1: Front racket occluded during takeback (only Back candidate present)
        r1 = _make_dual_row(1, 0.04)
        r1["racket_pose"]["candidates"] = [r1["racket_pose"]["candidates"][1]]

        res = mine_session_dual_view_data(
            session_id="unit_test_session",
            rows=[r0, r1],
            image_size=(2560, 1440),
            events=[{"event_id": 1, "start_frame": 0, "end_frame": 1, "contact_frame": 1}],
        )
        self.assertEqual(res["counts"]["rescued_native_racket_7pt"], 1)
        self.assertEqual(res["counts"]["projected_occluded_racket_v1"], 1)
        self.assertGreaterEqual(res["counts"]["healed_body_limb_frames"], 1)

        # Verify YOLO-Pose label format: 0 xc yc w h + 7 * (kx ky v) = 26 tokens
        v1_sample = res["samples"]["projected_occluded_racket"][0]
        self.assertEqual(v1_sample["target_view"], "front")
        tokens = v1_sample["yolo_pose_label"].split()
        self.assertEqual(len(tokens), 5 + 7 * 3)
        self.assertEqual(tokens[0], "0")
        # Visibility flag for occluded projected points must be '1'
        self.assertEqual(tokens[7], "1")
        self.assertEqual(RACKET_FLIP_IDX, [0, 1, 2, 3, 4, 6, 5])


if __name__ == "__main__":
    unittest.main()
