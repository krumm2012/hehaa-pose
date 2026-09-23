"""
test_swing_report_builder.py
────────────────────────────
Unit tests for swing_report_builder:
1. 5-dimension SVG biomechanical quality radar chart generation
2. Performance tier badge & overall score meter mapping
3. Kinematic sequence latency timeline bars
4. Impact freeze snapshot & telemetry rendering
5. Backward compatibility with standard legacy events
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from swing_report_builder import (
    _build_radar_svg,
    build_report_payload,
    default_report_path,
    render_report_html,
    write_report_html,
)


class SwingReportBuilderTests(unittest.TestCase):
    def test_build_radar_svg_empty(self):
        self.assertEqual(_build_radar_svg({}), "")
        self.assertEqual(_build_radar_svg(None), "")

    def test_build_radar_svg_with_subscores(self):
        sub_scores = {
            "shoulder_turn": 88.0,
            "takeback": 82.0,
            "arm_extension": 85.0,
            "racket_speed": 80.0,
            "leg_drive": 86.0,
        }
        svg = _build_radar_svg(sub_scores)
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))
        self.assertIn('class="radar-svg"', svg)
        self.assertIn("转肩 88", svg)
        self.assertIn("引拍 82", svg)
        self.assertIn("延展 85", svg)
        self.assertIn("挥速 80", svg)
        self.assertIn("蹬地 86", svg)
        self.assertIn("<polygon", svg)
        self.assertIn("<circle", svg)

    def test_build_report_payload_with_algo2_biomechanics(self):
        with TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            frame_json = tmp_root / "test_frames.json"
            event_json = tmp_root / "test_frames_swing_events.json"
            coach_json = tmp_root / "test_frames_coach_dataset.json"

            frame_json.write_text('{"video_info": {"fps": 25.0}, "frames": [{}, {}]}', encoding="utf-8")
            event_data = {
                "events": [
                    {
                        "event_id": 1,
                        "stroke_type": "FOREHAND",
                        "start_frame": 10,
                        "contact_frame": 36,
                        "peak_frame": 40,
                        "end_frame": 55,
                        "biomechanics": {
                            "swing_score": 85.5,
                            "swing_grade": "PRO",
                            "metrics": {
                                "swing_quality_score": {
                                    "overall_score": 85.5,
                                    "grade": "PRO",
                                    "sub_scores": {
                                        "shoulder_turn": 90.0,
                                        "takeback": 84.0,
                                        "arm_extension": 88.0,
                                        "racket_speed": 82.0,
                                        "leg_drive": 83.5,
                                    },
                                },
                                "kinematic_sequence": {
                                    "sequence_quality": "OPTIMAL",
                                    "details": {
                                        "sequence_quality": "OPTIMAL",
                                        "latency_hip_to_shoulder_ms": 42.0,
                                        "latency_shoulder_to_racket_ms": 55.0,
                                    },
                                },
                                "racket_speed": {
                                    "contact_kmh": 86.4,
                                    "max_kmh": 93.1,
                                },
                                "brush_angle": {
                                    "low_to_high_angle_deg": 22.5,
                                    "drop_depth_ratio": 1.18,
                                },
                                "stance": {"stance_type": "Semi-Open Stance"},
                                "leg_drive": {"drive_ratio": 1.15},
                            },
                        },
                        "coach_advice": [
                            {"code": "PERFECT_CHAIN", "message": "动力学链传递流畅，保持节奏", "confidence": 0.95}
                        ],
                    }
                ]
            }
            event_json.write_text(str(event_data).replace("'", '"'), encoding="utf-8")

            payload = build_report_payload(str(frame_json), str(event_json), str(coach_json))
            self.assertEqual(len(payload["events"]), 1)
            ev = payload["events"][0]
            self.assertEqual(ev["event_id"], 1)
            self.assertEqual(ev["swing_score"], 85.5)
            self.assertEqual(ev["swing_grade"], "PRO")
            self.assertIn("sub_scores", ev["swing_quality_score"])
            self.assertEqual(ev["kinematic_sequence"]["sequence_quality"], "OPTIMAL")

            # Render HTML and verify biomechanical elements
            report_path = tmp_root / "test_report.html"
            write_report_html(payload, str(report_path))
            html = report_path.read_text(encoding="utf-8")

            # Tier badge & score meter
            self.assertIn("tier-pro", html)
            self.assertIn("PRO · 职业级", html)
            self.assertIn("85.5分", html)

            # Radar chart
            self.assertIn("5维生物力学质量雷达", html)
            self.assertIn("class=\"radar-svg\"", html)
            self.assertIn("转肩 90", html)

            # Kinematic sequence
            self.assertIn("动力学链传递", html)
            self.assertIn("seq-optimal", html)
            self.assertIn("42.0 ms", html)
            self.assertIn("55.0 ms", html)

            # Telemetry grid
            self.assertIn("86.4 / 93.1 km/h", html)
            self.assertIn("+22.5°", html)
            self.assertIn("下潜 1.18x", html)
            self.assertIn("Semi-Open Stance", html)
            self.assertIn("蹬地 1.15x", html)

            # Coach advice
            self.assertIn("动力学链传递流畅", html)

    def test_render_report_html_with_impact_freeze_thumbnail(self):
        with TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            frame_json = tmp_root / "clip.json"
            event_json = tmp_root / "clip_swing_events.json"
            report_html = tmp_root / "clip_swing_report.html"

            # Create dummy impact freeze image
            freeze_img = tmp_root / "algo2_verified_frame_36_impact_freeze.jpg"
            freeze_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)

            frame_json.write_text('{"video_info": {"fps": 25.0}, "frames": []}', encoding="utf-8")
            event_data = {
                "events": [
                    {
                        "event_id": 1,
                        "stroke_type": "FOREHAND",
                        "start_frame": 10,
                        "contact_frame": 36,
                        "end_frame": 50,
                        "swing_score": 78.0,
                        "swing_grade": "ADVANCED",
                    }
                ]
            }
            event_json.write_text(str(event_data).replace("'", '"'), encoding="utf-8")

            payload = build_report_payload(str(frame_json), str(event_json))
            write_report_html(payload, str(report_html))
            html = report_html.read_text(encoding="utf-8")

            self.assertIn("tier-advanced", html)
            self.assertIn("ADVANCED · 进阶级", html)
            self.assertIn("algo2_verified_frame_36_impact_freeze.jpg", html)
            self.assertIn("击球瞬间定格特写 (第 36 帧)", html)

    def test_legacy_backward_compatibility(self):
        with TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            frame_json = tmp_root / "legacy.json"
            event_json = tmp_root / "legacy_swing_events.json"
            report_html = tmp_root / "legacy_report.html"

            frame_json.write_text('{"video_info": {"fps": 25.0}, "frames": []}', encoding="utf-8")
            event_data = {
                "events": [
                    {
                        "event_id": 1,
                        "stroke_type": "Backhand",
                        "start_frame": 5,
                        "contact_frame": 15,
                        "end_frame": 25,
                        "overall_score": 0.65,
                    }
                ]
            }
            event_json.write_text(str(event_data).replace("'", '"'), encoding="utf-8")

            payload = build_report_payload(str(frame_json), str(event_json))
            write_report_html(payload, str(report_html))
            html = report_html.read_text(encoding="utf-8")

            self.assertIn("<video", html)
            self.assertIn("Event 1 · Backhand", html)
            # Legacy score displayed cleanly
            self.assertIn("score 65", html)
            self.assertIn('data-field="start_frame"', html)


if __name__ == "__main__":
    unittest.main()
