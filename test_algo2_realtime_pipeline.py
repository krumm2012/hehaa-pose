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
            "stroke_type": "forehand",
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
        self.assertEqual(card["stroke_type"], "forehand")
        self.assertEqual(card["swing_score"], 88.5)
        self.assertEqual(card["swing_grade"], "ADVANCED")
        self.assertEqual(card["racket_speed_kmh"], 76.2)
        self.assertEqual(card["brush_angle_deg"], 24.5)
        self.assertEqual(card["stance_type"], "Semi-Open")
        self.assertIn("腿 -> 髋 -> 肩 -> 拍", card["kinematic_sequence_text"])

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
        self.assertEqual(state_active["telemetry_card"]["swing_grade"], "ADVANCED")

        # 超过过期帧后查询 (如第 85 帧)
        state_expired = runtime.get_active_display_state(current_frame_id=85)
        self.assertFalse(state_expired["active"])
        self.assertEqual(state_expired["event_label"], "READY STANCE")
        self.assertEqual(state_expired["coaching_text"], "")
        self.assertIsNone(state_expired["telemetry_card"])

        runtime.close()


if __name__ == "__main__":
    unittest.main()
