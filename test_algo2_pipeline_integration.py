import unittest

from swing_motion_features import extract_motion_features
from swing_event_classifier import classify_swing_event
from swing_biomechanics import aggregate_event_biomechanics
from local_realtime_coach import LocalRealtimeCoach


class Algo2PipelineIntegrationTests(unittest.TestCase):
    def test_swing_motion_features_extracts_dual_view_and_healed_pose(self):
        frames = [
            {
                "frame_id": 10,
                "timestamp": 1.0,
                "pose": {"right_wrist": [100, 200], "left_shoulder": [80, 100], "right_shoulder": [120, 100]},
                "healed_pose": {"right_wrist": [105, 205], "left_shoulder": [80, 100], "right_shoulder": [120, 100]},
                "dual_view_biomechanics": {
                    "shoulder_turn": {"shoulder_turn_deg": 88.5},
                    "takeback_depth": {"takeback_depth_ratio": 0.42},
                    "scapular_retraction": {"scapular_retraction_ratio": 0.28},
                    "shot_classification": {"stroke_type": "Forehand", "is_two_handed": False},
                    "contact_distance_gate": {"is_valid_contact": True},
                },
            }
        ]
        features = extract_motion_features(frames, dominant_hand="right")
        self.assertEqual(len(features), 1)
        feat = features[0]
        # Should use healed_pose coordinates for wrist
        self.assertEqual(feat["wrist"], (105.0, 205.0))
        self.assertAlmostEqual(feat["robust_shoulder_turn_deg"], 88.5)
        self.assertAlmostEqual(feat["takeback_depth_ratio"], 0.42)
        self.assertAlmostEqual(feat["scapular_retraction_ratio"], 0.28)
        self.assertEqual(feat["dual_view_stroke_type"], "Forehand")
        self.assertFalse(feat["dual_view_is_two_handed"])
        self.assertTrue(feat["dual_view_contact_valid"])

    def test_swing_event_classifier_prioritizes_dual_view(self):
        event_features = [
            {
                "dual_view_stroke_type": "Two-Handed Backhand",
                "dual_view_is_two_handed": True,
                "active_wrist_x_offset_body_width": -0.25,
                "two_hand_distance_body_width": 0.35,
                "raw_swing_type": "Backhand",
                "dominant_hand": "right",
            }
            for _ in range(5)
        ]
        result = classify_swing_event(event_features)
        self.assertEqual(result["stroke_type"], "Two-Handed Backhand")
        self.assertGreaterEqual(result["confidence"], 0.90)
        self.assertEqual(
            result["evidence"]["classification_context"]["decision_rule"],
            "dual_view_two_handed_backhand",
        )
        self.assertIn("dual_view", result["evidence"]["classification_context"])
        self.assertEqual(
            result["evidence"]["classification_context"]["dual_view"]["evidence_frames"],
            5,
        )

    def test_swing_biomechanics_dual_view_aggregation(self):
        frames = [
            {
                "frame_id": i,
                "pose": {
                    "left_shoulder": [90, 100],
                    "right_shoulder": [150, 100],
                    "left_hip": [95, 200],
                    "right_hip": [145, 200],
                },
            }
            for i in range(5)
        ]
        features = [
            {
                "frame_id": i,
                "has_pose": True,
                "contact_score": 0.8 if i == 3 else 0.1,
                "robust_shoulder_turn_deg": 92.0 + i,
                "takeback_depth_ratio": 0.30 + i * 0.05,
                "scapular_retraction_ratio": 0.15 + i * 0.03,
                "arm_extension_deg": 140.0,
            }
            for i in range(5)
        ]
        event = {
            "start_frame": 0,
            "contact_frame": 3,
            "peak_frame": 3,
            "end_frame": 4,
            "quality_flags": {"pose_frame_ratio": 1.0},
        }

        result = aggregate_event_biomechanics(event, frames, features)
        self.assertEqual(result["schema_version"], "dual_view_2d_v1")
        metrics = result["metrics"]

        # Shoulder turn should use robust dual-view anti-collapse metric
        self.assertTrue(metrics["shoulder_turn"]["coach_eligible"])
        self.assertEqual(metrics["shoulder_turn"]["observability"], "dual_view_anti_collapse")
        self.assertEqual(metrics["shoulder_turn"]["unit"], "deg_360")

        # Takeback depth and scapular retraction should be present
        self.assertIn("takeback_depth", metrics)
        self.assertIn("scapular_retraction", metrics)
        self.assertTrue(metrics["takeback_depth"]["coach_eligible"])
        self.assertTrue(metrics["scapular_retraction"]["coach_eligible"])
        self.assertEqual(metrics["takeback_depth"]["observability"], "dual_view_mirror_projection")
        # Peak value between frame 0 and frame 3: 0.30 + 3 * 0.05 = 0.45
        self.assertAlmostEqual(metrics["takeback_depth"]["value"], 0.45)

    def test_local_realtime_coach_gives_dual_view_guidance(self):
        coach = LocalRealtimeCoach(max_suggestions=3, min_confidence=0.45)
        # Event with shallow takeback and low scapular retraction
        event = {
            "event_id": 99,
            "confidence": 0.92,
            "quality_flags": {"warnings": [], "pose_frame_ratio": 1.0},
            "phase_counts": {"backswing": 10, "follow_through": 10},
            "biomechanics": {
                "schema_version": "dual_view_2d_v1",
                "metrics": {
                    "takeback_depth": {
                        "value": 0.22,  # < threshold 0.35
                        "unit": "ratio",
                        "confidence": 0.85,
                        "coach_eligible": True,
                    },
                    "scapular_retraction": {
                        "value": 0.12,  # < threshold 0.20
                        "unit": "ratio",
                        "confidence": 0.82,
                        "coach_eligible": True,
                    },
                    "arm_extension": {
                        "value": 155.0,  # OK
                        "unit": "deg_2d",
                        "confidence": 0.90,
                        "coach_eligible": True,
                    },
                },
            },
        }

        advices = coach.advise_all(event)
        codes = [a["code"] for a in advices]
        self.assertIn("limited_takeback_depth", codes)
        self.assertIn("limited_scapular_retraction", codes)

        for advice in advices:
            self.assertLessEqual(len(advice["message"]), 15)
            if advice["code"] == "limited_takeback_depth":
                self.assertEqual(advice["message"], "充分展开后背引拍")
            elif advice["code"] == "limited_scapular_retraction":
                self.assertEqual(advice["message"], "转肩蓄力拉开后背")

    def test_contact_and_shadow_swing_classification(self):
        # 1. 真实触球事件 (带反弹与近距离)
        contact_features = [
            {"frame_id": i, "ball": (100.0, 200.0 - i * 10), "racket": (100.0, 150.0), "ball_racket_distance": abs(50.0 - i * 10)}
            for i in range(3)
        ]
        # 添加反弹
        contact_features.extend([
            {"frame_id": 3 + j, "ball": (95.0, 170.0 + j * 15), "racket": (100.0, 150.0), "ball_racket_distance": 20.0 + j * 15}
            for j in range(3)
        ])
        res_contact = classify_swing_event(contact_features, contact_frame=3)
        self.assertFalse(res_contact["is_shadow_swing"])
        self.assertTrue(res_contact["is_valid_contact"])

        # 2. 空挥/未触球事件 (球距离拍子远且无反弹)
        shadow_features = [
            {"frame_id": i, "ball": (100.0, 200.0 - i * 20), "racket": (400.0, 150.0), "ball_racket_distance": 320.0}
            for i in range(5)
        ]
        res_shadow = classify_swing_event(shadow_features, contact_frame=2)
        self.assertTrue(res_shadow["is_shadow_swing"])
        self.assertFalse(res_shadow["is_valid_contact"])
        self.assertGreater(res_shadow["min_ball_distance"], 180.0)

    def test_dual_view_renderer_renders_ball_trail_and_racket_box(self):
        import numpy as np
        from dual_view_renderer import DualViewRenderer
        from dual_view_manager import DualViewCropInfo, DualViewFrame
        from dual_pose_estimator import DualPoseResult
        from dual_view_biomechanics import DualViewBiomechanicsResult, ShotClassificationResult

        renderer = DualViewRenderer(show_hud=True, show_skeleton=False)
        front_img = np.zeros((720, 540, 3), dtype=np.uint8)
        back_img = np.zeros((720, 540, 3), dtype=np.uint8)
        orig_img = np.zeros((1440, 2560, 3), dtype=np.uint8)

        front_info = DualViewCropInfo(
            bbox_orig=(100, 100, 1100, 1300),
            crop_size=(1000, 1200),
            view_size=(540, 720),
            is_horizontally_flipped=False,
        )
        back_info = DualViewCropInfo(
            bbox_orig=(100, 100, 640, 820),
            crop_size=(540, 720),
            view_size=(540, 720),
            is_horizontally_flipped=True,
        )
        dual_frame = DualViewFrame(
            frame_id=1,
            original_frame=orig_img,
            front_frame=front_img,
            back_frame=back_img,
            front_info=front_info,
            back_info=back_info,
        )
        pose_res = DualPoseResult(
            frame_id=1,
            front_pose_local={},
            back_pose_local={},
            front_pose_orig={},
            back_pose_orig={},
            fused_pose_local={},
            fused_pose_orig={},
            biomechanics=DualViewBiomechanicsResult(
                shot_classification=ShotClassificationResult(
                    shot_type="Forehand",
                    confidence=0.9,
                    hitting_hand="right",
                    is_two_handed=False,
                    midline_side_projection=-1.0,
                ),
                front_shoulder_width=100.0,
                back_shoulder_width=100.0,
                robust_shoulder_turn_deg=45.0,
                takeback_depth_ratio=0.85,
                scapular_retraction_ratio=0.45,
            ),
        )

        ball_trail = [(500.0, 600.0), (520.0, 580.0), (540.0, 560.0)]
        racket_box = (500.0, 520.0, 600.0, 620.0)

        out_canvas = renderer.render_dual_frame(
            dual_frame,
            pose_res,
            event_label="FOREHAND (SHADOW SWING)",
            coaching_text="未触及球，注意盯球击球点",
            ball_trail=ball_trail,
            racket_box=racket_box,
        )
        self.assertEqual(out_canvas.shape, (720, 1080, 3))

    def test_backview_eye_privacy_masking(self):
        """测试背面视角人脸眼睛隐私遮挡机制与防闪烁平滑。"""
        import numpy as np
        from dual_view_renderer import DualViewRenderer
        from dual_pose_estimator import DualPoseEstimator

        renderer = DualViewRenderer(mask_backview_eyes=True, eye_mask_style="bar")
        b_img = np.full((720, 540, 3), 200, dtype=np.uint8)

        # 1. 模拟眼睛区域并在其上绘制亮色
        eye_box = (200, 300, 280, 330)
        # 初始帧应用隐私条
        masked = renderer.apply_eye_privacy_mask(b_img, [eye_box])
        # 验证隐私条内部已被覆盖为深灰色 (20, 20, 20)
        center_pixel = masked[315, 240]
        self.assertEqual(center_pixel[0], 20)
        self.assertEqual(center_pixel[1], 20)
        self.assertEqual(center_pixel[2], 20)

        # 2. 模拟下一帧检测暂时丢失（验证 2 帧保持机制生效）
        masked_held = renderer.apply_eye_privacy_mask(b_img, [])
        self.assertEqual(masked_held[315, 240, 0], 20)

        # 3. 验证姿态提取模块对背面人脸眼睛提取与背面朝向过滤
        estimator = DualPoseEstimator(backend="mock")
        # 前景人脸（双眼可见）
        kpts_face = [{"left_eye": (250, 600), "right_eye": (210, 600), "nose": (230, 620)}]
        boxes = estimator._extract_eye_boxes_from_kpts_list(kpts_face, h=720, w=540)
        self.assertEqual(len(boxes), 1)
        bx1, by1, bx2, by2 = boxes[0]
        self.assertTrue(bx1 < 230 < bx2)
        self.assertTrue(by1 < 600 < by2)

        # 镜中转身击球背向人像（眼睛不可见，无误遮挡）
        kpts_back = [{"left_eye": None, "right_eye": None, "nose": None, "left_ear": (250, 300)}]
        boxes_back = estimator._extract_eye_boxes_from_kpts_list(kpts_back, h=720, w=540)
        self.assertEqual(len(boxes_back), 0)


if __name__ == "__main__":
    unittest.main()

