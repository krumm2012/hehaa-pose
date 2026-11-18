#!/usr/bin/env python3
"""
test_algo2_realtime_pipeline.py
───────────────────────────────
单元与集成测试：验证算法 2.0 生产实时化主工程对接 (P0)
1. 命令行参数与配置解析 (--algo2-dual-view / --algo2-config)
2. FrameProcessor 的 healed_pose 与 dual_view_biomechanics 注入
3. RealtimeSwingRuntime 的击球特写遥测卡片生成与 Active Display 状态机管理
4. 端到端模拟数据流集成测试
"""

import unittest
from unittest.mock import MagicMock
from pathlib import Path

from frame_processor import FrameProcessor
from main_pipe import build_argument_parser, MultiprocessPipeline
from realtime_swing_runtime import RealtimeSwingRuntime, build_impact_telemetry_card
from realtime_swing_pipeline import RealtimeSwingEventEngine


class Algo2RealtimePipelineTests(unittest.TestCase):
    """测试算法 2.0 实时管道对接组件"""

    def test_cli_argument_parsing(self):
        """测试 CLI 参数解析对 --algo2-dual-view 与 --dual-view 的支持"""
        parser = build_argument_parser()
        args = parser.parse_args(["--algo2-dual-view", "--algo2-config", "configs/dual_view_config.yaml"])
        self.assertTrue(args.algo2_dual_view)
        self.assertEqual(args.algo2_config, "configs/dual_view_config.yaml")

        # 测试短别名 --dual-view
        args_alias = parser.parse_args(["--dual-view"])
        self.assertTrue(args_alias.algo2_dual_view)

    def test_multiprocess_pipeline_init_algo2_defaults(self):
        """测试 MultiprocessPipeline 在 algo2_dual_view 开启时的自动化行为"""
        pipeline = MultiprocessPipeline(
            config_path="configs/yolo26_tennis_config.yaml",
            algo2_dual_view=True,
            no_save_video=True,
        )
        self.assertTrue(pipeline.algo2_dual_view)
        self.assertTrue(pipeline.realtime_swing_events)
        self.assertTrue(pipeline.realtime_coach)
        self.assertFalse(pipeline.config.get("pipeline_perf", {}).get("dual_view_enabled", True))

    def test_frame_processor_build_frame_record_algo2(self):
        """测试 FrameProcessor.build_frame_record 对自愈姿态和双视角生物力学的接收与封装"""
        fp = FrameProcessor(
            config={},
            frame_dimensions=(720, 1280),
            fps=25.0,
        )
        mock_pose = {"nose": (640, 360)}
        mock_healed = {"nose": (640, 360), "right_wrist": (700, 300)}
        mock_dv_bio = {
            "shoulder_turn": {"shoulder_turn_deg": 42.5},
            "takeback_depth": {"takeback_depth_ratio": 1.62},
        }
        rec = fp.build_frame_record(
            frame_id=10,
            swing_type="Forehand",
            ball_position=(650.0, 320.0),
            racket_detections=[{"box": [680, 280, 750, 350], "confidence": 0.85}],
            poses=[mock_pose],
            phase_metrics={"shoulder_turn": {"angle": 42.5}},
            healed_pose=mock_healed,
            dual_view_biomechanics=mock_dv_bio,
            racket=(680, 280, 750, 350),
        )
        self.assertEqual(rec["frame_id"], 10)
        self.assertEqual(rec["healed_pose"], mock_healed)
        self.assertEqual(rec["dual_view_biomechanics"], mock_dv_bio)
        self.assertEqual(rec["racket"], [680, 280, 750, 350])

    def test_build_impact_telemetry_card(self):
        """测试 build_impact_telemetry_card 对第一、二、三梯队指标与评分的规范抽取"""
        event_dict = {
            "stroke_type": "Forehand",
            "confidence": 0.95,
            "biomechanics": {
                "extended_biomechanics": {
                    "swing_quality_score": {
                        "overall_score": 88.5,
                        "grade": "ADVANCED",
                    },
                    "racket_head_speed": {
                        "contact_kmh": 76.2,
                        "max_kmh": 82.0,
                    },
                    "brush_angle": {
                        "low_to_high_angle_deg": 24.5,
                        "drop_depth_ratio": 0.42,
                    },
                    "stance": {
                        "stance_type": "Semi-Open",
                    },
                    "leg_drive": {
                        "drive_ratio": 0.18,
                    },
                    "kinematic_sequence": {
                        "sequence_quality": "OPTIMAL",
                    },
                }
            }
        }
        card = build_impact_telemetry_card(event_dict)
        self.assertEqual(card["stroke_type"], "Forehand")
        self.assertIsNone(card["swing_score"])
        self.assertEqual(card["practice_score"]["status"], "insufficient_evidence")
        self.assertIsNone(card["swing_grade"])
        self.assertEqual(card["racket_speed_kmh"], 76.2)
        self.assertIsNone(card["brush_angle_deg"])
        self.assertEqual(card["stance_type"], "Semi-Open")
        self.assertIn("二维参考", card["kinematic_sequence_text"])
        self.assertIn("未验证", card["kinematic_sequence_text"])

    def test_realtime_runtime_active_display_state(self):
        """测试 RealtimeSwingRuntime 的 get_active_display_state 在无事件与触发事件时的状态变迁与过期控制"""
        mock_engine = MagicMock()
        mock_engine.fps = 25.0
        mock_output = MagicMock()

        runtime = RealtimeSwingRuntime(
            engine=mock_engine,
            output=mock_output,
        )

        # 初始无事件状态
        state0 = runtime.get_active_display_state(current_frame_id=0)
        self.assertFalse(state0["active"])
        self.assertEqual(state0["event_label"], "READY STANCE")
        self.assertEqual(state0["coaching_text"], "")
        self.assertIsNone(state0["telemetry_card"])

        # 模拟发布一个挥拍事件 (击球点 50，发布点 55)
        test_event = {
            "event_id": 1,
            "stroke_type": "Forehand",
            "confidence": 0.96,
            "start_frame": 30,
            "end_frame": 60,
            "contact_frame": 50,
            "emitted_at_frame": 55,
            "coach_advices": [{"message": "挥拍时手臂再舒展", "confidence": 0.90}],
            "biomechanics": {
                "extended_biomechanics": {
                    "swing_quality_score": {"overall_score": 85.0, "grade": "ADVANCED"},
                    "racket_head_speed": {"contact_kmh": 72.0, "max_kmh": 78.0},
                    "brush_angle": {"low_to_high_angle_deg": 20.0, "drop_depth_ratio": 0.35},
                    "stance": {"stance_type": "Semi-Open"},
                    "leg_drive": {"drive_ratio": 0.15},
                    "kinematic_sequence": {"sequence_quality": "OPTIMAL"},
                }
            }
        }

        runtime._publish([test_event])

        # 在保持窗口期内 (如第 65 帧，未满 55 + 25 = 80 帧) 查询
        state_active = runtime.get_active_display_state(current_frame_id=65)
        self.assertTrue(state_active["active"])
        self.assertIn("FOREHAND", state_active["event_label"])
        self.assertEqual(state_active["coaching_text"], "挥拍时手臂再舒展")
        self.assertIsNotNone(state_active["telemetry_card"])
        self.assertIsNone(state_active["telemetry_card"]["swing_grade"])

        # 超过过期帧后查询 (如第 85 帧)
        state_expired = runtime.get_active_display_state(current_frame_id=85)
        self.assertFalse(state_expired["active"])
        self.assertEqual(state_expired["event_label"], "READY STANCE")
        self.assertEqual(state_expired["coaching_text"], "")
        self.assertIsNone(state_expired["telemetry_card"])

        runtime.close()

    def test_back_view_adaptive_cadence_skips_idle_and_wakes_on_motion_or_occlusion(self):
        import numpy as np
        from dual_pose_estimator import DualPoseEstimator
        from dual_view_biomechanics import Keypoint
        from dual_view_manager import DualViewCropInfo, DualViewFrame

        est = DualPoseEstimator.__new__(DualPoseEstimator)
        est.conf_threshold = 0.35
        from dual_view_biomechanics import DualViewBiomechanicsEngine
        est.biomech_engine = DualViewBiomechanicsEngine()
        est.backend = "coreml"
        est.concurrent = False
        est.compute_units = "ALL"
        est.back_view_adaptive_cadence = True
        est.back_idle_max_skip = 2
        est.active_wrist_motion_px = 10.0
        est.model_front = object()
        est.model_back = object()
        est._pool = None
        est._last_back_eyes = []
        est._last_valid_back_pose = {}
        est._back_missing_count = 0
        est._has_back_orientation_anchor = False
        est._last_valid_front_pose = {}
        est._front_missing_count = 0
        est._cached_back_pose_local = {}
        est._cached_back_eyes = []
        est._prev_front_pose_for_gate = {}
        est._front_active_hold = 0
        est._back_skipped_streak = 0
        est._last_front_occluded = True
        est.back_view_inferred_count = 0
        est.back_view_skipped_count = 0

        def make_pose(wrist_x=200.0, drop_wrist=False):
            pts = {
                "left_shoulder": Keypoint(180, 150, 0.9),
                "right_shoulder": Keypoint(240, 150, 0.9),
                "left_elbow": Keypoint(160, 210, 0.9),
                "right_elbow": Keypoint(260, 210, 0.9),
                "left_wrist": Keypoint(150, 260, 0.9),
                "right_wrist": Keypoint(wrist_x, 260, 0.9),
                "left_hip": Keypoint(190, 280, 0.9),
                "right_hip": Keypoint(230, 280, 0.9),
            }
            if drop_wrist:
                pts.pop("right_wrist")
            return pts

        back_calls = []
        front_seq = [
            make_pose(200.0),  # frame 0: cold start -> runs back
            make_pose(201.0),  # frame 1: idle -> skips back
            make_pose(202.0),  # frame 2: idle -> skips back
            make_pose(202.0),  # frame 3: max_skip reached -> runs back
            make_pose(260.0),  # frame 4: fast wrist jump (+58px) -> immediately wakes back!
        ]

        def fake_predict(img, is_back_view=False, **kwargs):
            fid = est._source_frame_id
            if is_back_view:
                back_calls.append(fid)
                return make_pose(210.0)
            return front_seq[fid]

        est._predict_single_view = fake_predict
        dummy_img = np.zeros((720, 540, 3), dtype=np.uint8)
        dummy_orig = np.zeros((1440, 2560, 3), dtype=np.uint8)
        info = DualViewCropInfo(bbox_orig=(0, 0, 540, 720), crop_size=(540, 720), view_size=(540, 720))
        for fid in range(len(front_seq)):
            df = DualViewFrame(frame_id=fid, original_frame=dummy_orig, front_frame=dummy_img, back_frame=dummy_img, front_info=info, back_info=info)
            est.estimate_dual_pose(df)

        self.assertEqual(back_calls, [0, 3, 4])
        self.assertEqual(est.back_view_inferred_count, 3)
        self.assertEqual(est.back_view_skipped_count, 2)


if __name__ == "__main__":
    unittest.main()
