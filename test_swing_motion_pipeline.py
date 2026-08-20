import unittest
from pathlib import Path
import json
import tempfile

from swing_event_classifier import classify_swing_event
from swing_event_analyzer import analyze_frame_records
from swing_event_segmenter import _select_peak_indices, segment_swing_events
from swing_event_video_renderer import (
    build_coach_lookup,
    build_event_lookup,
    default_coach_json,
    default_event_json,
    default_output_video,
    resolve_osd_state,
)
from swing_motion_features import extract_motion_features
from swing_coach_data_collector import build_coach_dataset, default_coach_output_path
from swing_report_builder import build_report_payload, default_report_path, write_report_html


class SwingMotionFeatureTests(unittest.TestCase):
    def test_extracts_pose_ball_racket_motion_features(self):
        frames = [
            {
                "frame_id": 0,
                "timestamp": 0.0,
                "swing_type": "Forehand",
                "ball": [100, 100],
                "rackets": [{"box": [90, 90, 110, 130], "confidence": 0.9}],
                "pose": {
                    "right_wrist": [10, 10],
                    "left_wrist": [30, 10],
                    "right_shoulder": [10, 0],
                    "right_elbow": [10, 5],
                    "left_shoulder": [0, 0],
                    "left_hip": [0, 20],
                    "right_hip": [10, 20],
                },
            },
            {
                "frame_id": 1,
                "timestamp": 0.04,
                "swing_type": "Forehand",
                "ball": [102, 100],
                "rackets": [{"box": [100, 90, 120, 130], "confidence": 0.9}],
                "pose": {
                    "right_wrist": [18, 10],
                    "left_wrist": [34, 10],
                    "right_shoulder": [10, 0],
                    "right_elbow": [14, 5],
                    "left_shoulder": [0, 0],
                    "left_hip": [0, 20],
                    "right_hip": [10, 20],
                },
            },
        ]

        features = extract_motion_features(frames)

        self.assertEqual(len(features), 2)
        self.assertEqual(features[1]["wrist_speed"], 8.0)
        self.assertEqual(features[1]["racket_speed"], 10.0)
        self.assertEqual(features[1]["two_hand_distance"], 16.0)
        self.assertGreater(features[1]["contact_score"], 0.0)


class SwingEventSegmentationTests(unittest.TestCase):
    def _feature(self, frame, wrist_speed, label="Forehand", two_hand_distance=140.0):
        return {
            "frame_id": frame,
            "raw_swing_type": label,
            "wrist_speed": wrist_speed,
            "wrist_accel": 0.0,
            "racket_speed": wrist_speed * 0.8,
            "contact_score": 0.2,
            "two_hand_distance": two_hand_distance,
        }

    def test_follow_through_is_merged_into_one_event(self):
        speeds = [0, 2, 8, 14, 18, 12, 5, 2, 0, 6, 7, 3, 0]
        features = [self._feature(i, speed) for i, speed in enumerate(speeds)]

        result = segment_swing_events(
            features,
            min_peak_energy=10.0,
            active_energy=5.0,
            min_event_frames=5,
            max_internal_gap=2,
            min_event_gap=6,
        )

        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["stroke_type"], "Forehand")

    def test_separates_distinct_events_after_gap(self):
        speeds = [0, 11, 16, 11, 0, 0, 0, 0, 0, 10, 16, 11, 0]
        features = [self._feature(i, speed) for i, speed in enumerate(speeds)]

        result = segment_swing_events(
            features,
            min_peak_energy=10.0,
            active_energy=5.0,
            min_event_frames=3,
            max_internal_gap=1,
            min_event_gap=3,
        )

        self.assertEqual(len(result["events"]), 2)

    def test_peak_suppression_does_not_chain_distinct_swings_through_noise(self):
        candidates = [
            (10, 100.0),
            (35, 60.0),
            (60, 90.0),
        ]

        selected = _select_peak_indices(candidates, peak_min_distance=40)

        self.assertEqual(selected, [10, 60])

    def test_peak_segmentation_refines_start_to_preparation_onset(self):
        features = []
        for frame in range(100):
            if frame < 20:
                speed = 2.0
                shoulder_turn = 90.0
            elif frame < 46:
                speed = 11.0 + (frame - 20) * 0.25
                shoulder_turn = 95.0 + (frame - 20) * 1.2
            elif frame <= 60:
                speed = 20.0 + (frame - 46) * 2.5
                shoulder_turn = 125.0
            else:
                speed = max(2.0, 34.0 - (frame - 60) * 1.5)
                shoulder_turn = 95.0
            feature = self._feature(frame, speed)
            feature.update(
                {
                    "timestamp": frame / 25.0,
                    "shoulder_turn_deg": shoulder_turn,
                    "contact_score": 0.8 if frame == 60 else 0.0,
                }
            )
            features.append(feature)

        result = segment_swing_events(
            features,
            min_peak_energy=20.0,
            active_energy=5.5,
            min_event_frames=8,
            min_event_gap=18,
        )

        self.assertEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertLessEqual(abs(event["start_frame"] - 19), 2)
        self.assertNotEqual(event["start_frame"], event["peak_frame"] - 21)
        self.assertEqual(event["evidence"]["start_boundary"]["mode"], "quiet_onset")

        event_trace = [
            row for row in result["frame_trace"] if row["event_id"] == event["event_id"]
        ]
        self.assertNotIn(
            "follow_through",
            [row["phase"] for row in event_trace if row["frame"] < event["contact_frame"]],
        )
        self.assertNotIn(
            "backswing",
            [row["phase"] for row in event_trace if row["frame"] > event["contact_frame"]],
        )

    def test_continuous_swings_expose_recovery_ready_transition(self):
        features = []
        for frame in range(130):
            if frame < 12:
                speed = 2.0
            elif frame <= 30:
                speed = 12.0 + (30 - abs(30 - frame)) * 1.5
            elif frame < 66:
                speed = max(12.0, 28.0 - (frame - 30) * 0.45)
            elif frame <= 90:
                speed = 12.0 + (frame - 66) * 1.8
            else:
                speed = max(2.0, 38.0 - (frame - 90) * 1.2)
            feature = self._feature(frame, speed)
            feature.update(
                {
                    "timestamp": frame / 25.0,
                    "shoulder_turn_deg": None,
                    "contact_score": 0.8 if frame in {30, 90} else 0.0,
                }
            )
            features.append(feature)

        result = segment_swing_events(
            features,
            min_peak_energy=20.0,
            active_energy=5.5,
            min_event_frames=8,
            min_event_gap=18,
        )

        self.assertEqual(len(result["events"]), 2)
        second = result["events"][1]
        self.assertEqual(
            second["evidence"]["start_boundary"]["mode"],
            "recovery_ready_transition",
        )
        start_trace = next(
            row for row in result["frame_trace"] if row["frame"] == second["start_frame"]
        )
        self.assertEqual(start_trace["phase"], "recovery_ready_transition")

    def test_fast_onset_inside_last_three_tenths_before_peak_is_not_backdated(self):
        features = []
        for frame in range(80):
            if frame <= 22:
                speed = 2.0
            elif frame <= 30:
                speed = 10.0 + (frame - 23) * 6.0
            else:
                speed = max(2.0, 36.0 - (frame - 30) * 2.0)
            feature = self._feature(frame, speed)
            feature.update(
                {
                    "timestamp": frame / 25.0,
                    "shoulder_turn_deg": None,
                    "contact_score": 0.8 if frame == 30 else 0.0,
                }
            )
            features.append(feature)

        result = segment_swing_events(
            features,
            min_peak_energy=20.0,
            active_energy=5.5,
            min_event_frames=8,
            min_event_gap=18,
        )

        self.assertEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertLessEqual(abs(event["start_frame"] - 22), 2)
        self.assertEqual(event["evidence"]["start_boundary"]["mode"], "quiet_onset")
        self.assertEqual(event["evidence"]["start_boundary"]["onset_signal"], "energy")

    def test_timestamp_spacing_preserves_distinct_peaks_across_dropped_frames(self):
        features = []
        for frame in range(201):
            if 40 <= frame <= 79:
                continue
            speed = 2.0
            if 15 <= frame <= 30:
                speed = 10.0 + (frame - 15) * 2.0
            elif 30 < frame < 40:
                speed = max(12.0, 40.0 - (frame - 30) * 2.0)
            elif 80 <= frame <= 90:
                speed = 12.0 + (frame - 80) * 3.0
            elif 90 < frame <= 110:
                speed = max(2.0, 42.0 - (frame - 90) * 2.0)
            feature = self._feature(frame, speed)
            feature.update(
                {
                    "timestamp": frame / 25.0,
                    "shoulder_turn_deg": None,
                    "contact_score": 0.8 if frame in {30, 90} else 0.0,
                }
            )
            features.append(feature)

        result = segment_swing_events(
            features,
            min_peak_energy=20.0,
            active_energy=5.5,
            min_event_frames=8,
            min_event_gap=18,
        )

        self.assertEqual(len(result["events"]), 2)
        self.assertEqual(result["events"][0]["peak_frame"], 30)
        self.assertLessEqual(abs(result["events"][1]["peak_frame"] - 90), 1)

    def test_screen_left_true_right_forehand_blocks_follow_through_two_hand_override(self):
        features = []
        for frame in range(90):
            label = "Forehand"
            speed = 0.0
            two_hand_distance = 220.0
            right_wrist_offset = -90.0
            if 36 <= frame <= 44:
                label = "Backhand"
                speed = 80.0 if frame == 40 else 30.0
            if 45 <= frame <= 68:
                label = "Two-Handed Backhand"
                two_hand_distance = 70.0
                speed = 8.0
            features.append(
                {
                    "frame_id": frame,
                    "timestamp": frame / 25.0,
                    "raw_swing_type": label,
                    "wrist_speed": speed,
                    "wrist_accel": 0.0,
                    "racket_speed": speed * 0.5,
                    "contact_score": 0.0,
                    "two_hand_distance": two_hand_distance,
                    "active_wrist_x_offset": right_wrist_offset,
                }
            )

        result = segment_swing_events(
            features,
            min_peak_energy=20.0,
            active_energy=6.0,
            min_event_frames=8,
            max_internal_gap=2,
            min_event_gap=18,
        )

        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["stroke_type"], "Forehand")


class SwingEventClassifierTests(unittest.TestCase):
    def test_classifies_two_handed_backhand_from_sustained_close_hands(self):
        event_features = [
            {"raw_swing_type": "Backhand", "two_hand_distance": 70, "active_wrist_x_offset": -20},
            {"raw_swing_type": "Backhand", "two_hand_distance": 72, "active_wrist_x_offset": -22},
            {"raw_swing_type": "Forehand", "two_hand_distance": 74, "active_wrist_x_offset": -25},
        ]

        result = classify_swing_event(event_features, two_hand_distance_px=95, two_hand_min_ratio=0.5)

        self.assertEqual(result["stroke_type"], "Two-Handed Backhand")

    def test_camera_normalization_overrides_noisy_follow_through_labels(self):
        event_features = []
        for frame in range(12):
            event_features.append(
                {
                    "dominant_hand": "right",
                    "raw_swing_type": "Two-Handed Backhand" if frame >= 7 else "Forehand",
                    "camera_facing_score": -0.98,
                    "active_wrist_x_offset_body_width": -1.1,
                    "two_hand_distance_body_width": 0.55 if frame >= 7 else 1.2,
                }
            )

        result = classify_swing_event(event_features)

        self.assertEqual(result["stroke_type"], "Forehand")
        context = result["evidence"]["classification_context"]
        self.assertEqual(context["player"]["dominant_hand"], "right")
        self.assertEqual(context["camera"]["view"], "facing_player")
        self.assertEqual(context["swing"]["side"], "forehand")
        self.assertEqual(context["decision_rule"], "camera_normalized_forehand_side")

    def test_same_forehand_maps_to_screen_right_from_behind_player(self):
        event_features = [
            {
                "dominant_hand": "right",
                "raw_swing_type": "Backhand",
                "camera_facing_score": 0.96,
                "active_wrist_x_offset_body_width": 0.9,
                "two_hand_distance_body_width": 1.1,
            }
            for _ in range(8)
        ]

        result = classify_swing_event(event_features)

        self.assertEqual(result["stroke_type"], "Forehand")
        context = result["evidence"]["classification_context"]
        self.assertEqual(context["camera"]["view"], "behind_player")
        self.assertEqual(context["camera"]["right_side_projects_to"], "screen_right")

    def test_camera_normalized_backhand_requires_two_hand_support(self):
        event_features = [
            {
                "dominant_hand": "right",
                "raw_swing_type": "Backhand",
                "camera_facing_score": -0.97,
                "active_wrist_x_offset_body_width": 0.75,
                "two_hand_distance_body_width": 0.55,
            }
            for _ in range(8)
        ]

        result = classify_swing_event(event_features)

        self.assertEqual(result["stroke_type"], "Two-Handed Backhand")
        self.assertEqual(
            result["evidence"]["classification_context"]["swing"]["side"],
            "backhand",
        )


class SwingEventAnalyzerTests(unittest.TestCase):
    def test_analyzes_frames_with_auditable_outputs(self):
        frames = []
        for idx, x in enumerate([10, 20, 35, 50, 65, 80, 88, 92]):
            frames.append(
                {
                    "frame_id": idx,
                    "timestamp": idx / 25.0,
                    "swing_type": "Forehand",
                    "ball": [x + 5, 100],
                    "rackets": [{"box": [x, 90, x + 10, 110], "confidence": 0.9}],
                    "pose": {
                        "right_wrist": [x, 100],
                        "left_wrist": [x - 80, 100],
                        "right_shoulder": [20, 80],
                        "right_elbow": [x - 10, 92],
                        "left_shoulder": [0, 80],
                        "left_hip": [0, 140],
                        "right_hip": [20, 140],
                    },
                }
            )

        analysis = analyze_frame_records(
            frames,
            min_peak_energy=8.0,
            active_energy=6.0,
            min_event_frames=3,
            max_internal_gap=1,
            min_event_gap=3,
        )

        self.assertEqual(analysis["summary"]["total_frames"], 8)
        self.assertGreaterEqual(analysis["summary"]["swing_event_count"], 1)
        self.assertEqual(len(analysis["features"]), 8)
        self.assertEqual(len(analysis["frame_trace"]), 8)
        self.assertEqual(
            analysis["summary"]["session_quality"]["event_count"],
            analysis["summary"]["swing_event_count"],
        )

    def test_stable_fixture_uses_screen_left_as_true_right_forehand(self):
        frames = []
        frame_id = 0

        def append_frame(x, label, contact_evidence):
            nonlocal frame_id
            ball_x = x + 5 if contact_evidence else x + 500
            frames.append(
                {
                    "frame_id": frame_id,
                    "timestamp": frame_id / 25.0,
                    "swing_type": label,
                    "ball": [ball_x, 100],
                    "rackets": [
                        {"box": [x, 90, x + 10, 110], "confidence": 0.9}
                    ],
                    "pose": {
                        "right_wrist": [x, 100],
                        "left_wrist": [x - 80, 100],
                        "right_shoulder": [20, 80],
                        "right_elbow": [x - 10, 92],
                        "left_shoulder": [0, 80],
                        "left_hip": [0, 140],
                        "right_hip": [20, 140],
                    },
                }
            )
            frame_id += 1

        for _ in range(3):
            for x in [10, 20, 35, 50, 65, 80, 88, 92]:
                append_frame(x, "Forehand", True)
            append_frame(10, "Ready", False)
            for _ in range(49):
                append_frame(10, "Ready", False)

        analysis = analyze_frame_records(
            frames,
            min_peak_energy=8.0,
            active_energy=6.0,
            min_event_frames=3,
            max_internal_gap=1,
            min_event_gap=3,
        )

        self.assertEqual(analysis["summary"]["swing_event_count"], 3)
        self.assertEqual(analysis["summary"]["swing_event_type_counts"], {"Forehand": 3})

    def test_event_quality_flags_tolerate_intermittent_ball_detection(self):
        frames = []
        for idx, x in enumerate([10, 25, 45, 70, 92, 112, 128, 138]):
            pose = {
                "right_wrist": [x, 110],
                "left_wrist": [x + 90, 110],
                "right_shoulder": [40, 80],
                "right_elbow": [x - 8, 98],
                "left_shoulder": [0, 80],
                "left_hip": [0, 150],
                "right_hip": [40, 150],
            }
            frames.append(
                {
                    "frame_id": idx,
                    "timestamp": idx / 25.0,
                    "swing_type": "Forehand",
                    "ball": [x + 4, 120] if idx not in {2, 3, 4} else None,
                    "rackets": [{"box": [x, 100, x + 10, 130], "confidence": 0.9}] if idx != 5 else [],
                    "pose": pose if idx != 3 else {},
                    "detection_diagnostics": {
                        "rejections": {"static_hard_mask": 2 if idx == 3 else 0, "upper_mirror_unsupported": 1 if idx == 4 else 0},
                        "continuity_disabled": idx == 4,
                    },
                }
            )

        analysis = analyze_frame_records(
            frames,
            min_peak_energy=8.0,
            active_energy=6.0,
            min_event_frames=3,
            max_internal_gap=1,
            min_event_gap=3,
        )

        event = analysis["events"][0]
        flags = event["quality_flags"]
        self.assertLess(flags["ball_frame_ratio"], 1.0)
        self.assertLess(flags["pose_frame_ratio"], 1.0)
        self.assertNotIn("ball_track_gaps", flags["warnings"])
        self.assertGreater(flags["ball_contact_window_ratio"], 0.0)
        self.assertIn("pose_gaps", flags["warnings"])
        self.assertEqual(flags["diagnostic_rejection_counts"]["static_hard_mask"], 2)
        self.assertEqual(flags["diagnostic_rejection_counts"]["upper_mirror_unsupported"], 1)

    def test_event_quality_flags_warn_when_ball_evidence_is_absent(self):
        frames = []
        for idx, x in enumerate([10, 25, 45, 70, 92, 112, 128, 138]):
            frames.append(
                {
                    "frame_id": idx,
                    "timestamp": idx / 25.0,
                    "swing_type": "Forehand",
                    "ball": None,
                    "rackets": [
                        {
                            "box": [x, 100, x + 10, 130],
                            "confidence": 0.9,
                        }
                    ],
                    "pose": {
                        "right_wrist": [x, 110],
                        "left_wrist": [x + 90, 110],
                        "right_shoulder": [40, 80],
                        "right_elbow": [x - 8, 98],
                        "left_shoulder": [0, 80],
                        "left_hip": [0, 150],
                        "right_hip": [40, 150],
                    },
                }
            )

        analysis = analyze_frame_records(
            frames,
            min_peak_energy=8.0,
            active_energy=6.0,
            min_event_frames=3,
            max_internal_gap=1,
            min_event_gap=3,
        )

        flags = analysis["events"][0]["quality_flags"]
        self.assertEqual(flags["ball_frame_ratio"], 0.0)
        self.assertEqual(flags["ball_contact_window_ratio"], 0.0)
        self.assertIn("ball_track_gaps", flags["warnings"])


class SwingCoachDataCollectorTests(unittest.TestCase):
    def test_builds_coach_dataset_with_event_key_metrics(self):
        frames = []
        for idx, x in enumerate([10, 20, 35, 50, 65, 80, 88, 92]):
            frames.append(
                {
                    "frame_id": idx,
                    "timestamp": idx / 25.0,
                    "swing_type": "Forehand",
                    "ball": [x + 4, 120],
                    "rackets": [{"box": [x, 100, x + 10, 130], "confidence": 0.9}],
                    "pose": {
                        "right_wrist": [x, 120],
                        "left_wrist": [x + 80, 120],
                        "right_shoulder": [20, 80],
                        "left_shoulder": [0, 80],
                        "right_elbow": [x - 8, 108],
                        "left_hip": [0, 150],
                        "right_hip": [20, 150],
                        "left_knee": [0, 190],
                        "right_knee": [22, 190],
                        "left_ankle": [0, 230],
                        "right_ankle": [28, 230],
                    },
                }
            )
        analysis = analyze_frame_records(
            frames,
            min_peak_energy=8.0,
            active_energy=6.0,
            min_event_frames=3,
            max_internal_gap=1,
            min_event_gap=3,
        )

        dataset = build_coach_dataset({"video_info": {"fps": 25, "path": "sample.mp4"}, "frames": frames}, analysis)

        self.assertEqual(dataset["metadata"]["video_path"], "sample.mp4")
        self.assertGreaterEqual(len(dataset["events"]), 1)
        event = dataset["events"][0]
        self.assertIn("contact_frame", event["frames"])
        self.assertIn("racket_speed_at_contact", event["racket"])
        self.assertIn("contact_point_relative_to_body", event["body"])
        self.assertIn("phase_durations_frames", event["timing"])
        self.assertIn("overall_score", event["scores"])
        self.assertIn("estimated_spin", event["ball"])
        self.assertIn("landing_point", event["ball"])
        self.assertIn("racket_face_angle_deg", event["racket"])
        self.assertIn("weight_transfer", event["body"])
        self.assertIn("recovery_time_frames", event["timing"])
        self.assertIn("data_quality", event)
        self.assertIn("missing_fields", event["data_quality"])
        self.assertIn("quality_flags", event)
        self.assertIn("event_quality_flags", event["data_quality"])
        self.assertIsNotNone(event["racket"]["low_to_high_ratio"])
        self.assertIsNotNone(event["racket"]["swing_path_type"])
        self.assertIsNotNone(event["body"]["weight_transfer"])
        self.assertIsNotNone(event["body"]["balance_state"])
        self.assertIsNotNone(event["body"]["stance_type"])
        self.assertIsNotNone(event["body"]["unit_turn_quality"])
        self.assertIsNone(event["body"]["late_contact"])
        self.assertEqual(
            event["body"]["late_contact_reason"],
            "front_back_depth_not_observable_single_view",
        )
        self.assertIn("coach_calibration", event)
        self.assertIn(event["scores"]["status"], {"calibrated", "insufficient_evidence"})
        self.assertIsNotNone(event["timing"]["recovery_time_frames"])
        self.assertIsNotNone(event["timing"]["tempo_consistency"])

        trace_less_analysis = dict(analysis)
        trace_less_analysis["frame_trace"] = []
        trace_less_dataset = build_coach_dataset(
            {"video_info": {"fps": 25, "path": "sample.mp4"}, "frames": frames},
            trace_less_analysis,
        )
        trace_less_event = trace_less_dataset["events"][0]
        self.assertEqual(
            trace_less_event["timing"]["phase_durations_frames"],
            analysis["events"][0]["phase_counts"],
        )
        self.assertGreater(
            trace_less_event["scores"]["preparation_score"],
            0.0,
        )

    def test_default_coach_output_path(self):
        self.assertEqual(default_coach_output_path("data/output_video.json"), "data/output_video_coach_dataset.json")

    def test_phase_one_estimates_direction_interpolated_peak_and_racket_lag(self):
        frames = []
        features = []
        frame_trace = []
        for frame_id in range(7):
            pose = {
                "right_wrist": [100 + frame_id * 2, 120],
                "left_wrist": [180, 120],
                "right_shoulder": [120, 80],
                "left_shoulder": [80, 80],
                "right_elbow": [105, 105],
                "left_hip": [80, 160],
                "right_hip": [120, 160],
                "left_knee": [80, 200],
                "right_knee": [120, 200],
                "left_ankle": [80, 240],
                "right_ankle": [122, 240],
            }
            frames.append({"frame_id": frame_id, "timestamp": frame_id / 25.0, "pose": pose})
            racket_center = None if frame_id == 3 else (90 + frame_id * 10, 130 - frame_id * 4)
            features.append(
                {
                    "frame_id": frame_id,
                    "timestamp": frame_id / 25.0,
                    "raw_swing_type": "Forehand",
                    "has_pose": True,
                    "wrist": (100 + frame_id * 2, 120),
                    "racket_center": racket_center,
                    "ball": (200 + frame_id * 15, 180 - abs(frame_id - 4) * 8),
                    "wrist_speed": 8.0,
                    "racket_speed": 12.0,
                    "racket_accel": 2.0,
                    "ball_speed": 10.0,
                    "ball_racket_distance": 35.0,
                    "contact_score": 0.8 if frame_id == 2 else 0.1,
                    "two_hand_distance": 80.0,
                    "active_wrist_x_offset": -30.0,
                    "arm_extension_deg": 150.0,
                    "shoulder_turn_deg": 85.0,
                    "hip_shoulder_sep_deg": 12.0,
                }
            )
            frame_trace.append(
                {
                    "frame": frame_id,
                    "event_id": 1,
                    "phase": "forward_swing" if frame_id <= 3 else "follow_through",
                    "motion_energy": 20.0,
                    "raw_swing_type": "Forehand",
                }
            )
        event_analysis = {
            "events": [
                {
                    "event_id": 1,
                    "start_frame": 0,
                    "end_frame": 6,
                    "duration_frames": 7,
                    "peak_frame": 3,
                    "stroke_type": "Forehand",
                    "confidence": 0.9,
                    "evidence": {},
                }
            ],
            "features": features,
            "frame_trace": frame_trace,
        }

        dataset = build_coach_dataset({"video_info": {"fps": 25, "path": "sample.mp4"}, "frames": frames}, event_analysis)
        event = dataset["events"][0]

        self.assertEqual(event["ball"]["shot_direction"], "screen_right")
        self.assertGreater(event["ball"]["shot_direction_confidence"], 0.0)
        self.assertEqual(event["ball"]["shot_direction_space"], "screen")
        self.assertIn("bounce_confidence", event["ball"])
        self.assertEqual(event["racket"]["racket_center_at_peak_source"], "interpolated")
        self.assertGreater(event["racket"]["racket_center_at_peak_confidence"], 0.0)
        self.assertIsNotNone(event["racket"]["racket_lag_at_contact"])
        self.assertGreater(event["racket"]["racket_lag_confidence"], 0.0)


class SwingReportBuilderTests(unittest.TestCase):
    def test_default_report_path(self):
        self.assertEqual(default_report_path("data/output_video.json"), "data/output_video_swing_report.html")

    def test_writes_standalone_video_json_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame_json = root / "sample.json"
            event_json = root / "sample_swing_events.json"
            coach_json = root / "sample_coach_dataset.json"
            evaluation_json = root / "sample_swing_evaluation.json"
            video_path = root / "sample_swing_annotated.mp4"
            report_path = root / "sample_swing_report.html"
            video_path.write_bytes(b"fake mp4")
            frame_json.write_text(
                json.dumps({"video_info": {"path": "sample.mp4", "fps": 25}, "frames": [{"frame_id": 0}]}),
                encoding="utf-8",
            )
            event_json.write_text(
                json.dumps(
                    {
                        "summary": {"swing_event_count": 1, "swing_event_type_counts": {"Forehand": 1}},
                        "events": [
                            {
                                "event_id": 1,
                                "start_frame": 0,
                                "end_frame": 5,
                                "peak_frame": 3,
                                "stroke_type": "Forehand",
                                "confidence": 0.8,
                                "evidence": {
                                    "classification_context": {
                                        "player": {"dominant_hand": "right"},
                                        "camera": {"view": "facing_player"},
                                        "swing": {"side": "forehand"},
                                    }
                                },
                                "quality_flags": {"warnings": ["ball_track_gaps"], "review_recommended": True},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            coach_json.write_text(
                json.dumps(
                    {
                        "summary": {"event_count": 1},
                        "events": [
                            {
                                "event_id": 1,
                                "scores": {"overall_score": 0.62},
                                "diagnosis_tags": ["low_contact_confidence"],
                                "data_quality": {"pose_frame_ratio": 0.75},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            evaluation_json.write_text(
                json.dumps(
                    {
                        "summary": {
                            "stroke_type_accuracy": 0.5,
                            "contact_accuracy": 1.0,
                            "manual_review_count": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )

            payload = build_report_payload(str(frame_json), str(event_json), str(coach_json), str(video_path))
            write_report_html(payload, str(report_path))
            html = report_path.read_text(encoding="utf-8")

        self.assertIn("<video", html)
        self.assertIn("sample_swing_annotated.mp4", html)
        self.assertIn("ball_track_gaps", html)
        self.assertIn("球员 右手 · 机位 球员面向相机 · 分类证据 正手侧", html)
        self.assertIn("low_contact_confidence", html)
        self.assertIn("annotation-stroke", html)
        self.assertIn("annotation-valid-hit", html)
        self.assertIn('data-field="needs_review" checked', html)
        self.assertIn("swing_manual_annotations_v2", html)
        self.assertIn('data-field="start_frame"', html)
        self.assertIn('data-field="contact_frame"', html)
        self.assertIn('data-field="end_frame"', html)
        self.assertIn('id="timeline-review-complete"', html)
        self.assertIn('id="annotation-readiness"', html)
        self.assertIn("updateAnnotationReadiness", html)
        self.assertIn('id="add-missed-event"', html)
        self.assertIn("addMissedEvent", html)
        self.assertIn("downloadAnnotations", html)
        self.assertIn("annotation-import-file", html)
        self.assertIn("applyImportedAnnotations", html)
        self.assertIn("importAnnotations", html)
        self.assertIn("Evaluation Summary", html)
        self.assertIn("stroke accuracy 50%", html)
        self.assertIn("contact accuracy 100%", html)
        self.assertIn('id="event-timeline"', html)
        self.assertIn('id="frame-scrubber"', html)
        self.assertIn("renderTimeline", html)
        self.assertIn("seekFrame", html)
        self.assertIn("会话质量与漂移", html)
        self.assertIn("session-dashboard", html)
        self.assertEqual(payload["session_quality"]["event_count"], 1)


class SwingEventVideoRendererTests(unittest.TestCase):
    def test_builds_event_lookup_and_default_paths(self):
        analysis = {
            "events": [{"event_id": 1, "start_frame": 10, "end_frame": 20, "stroke_type": "Forehand"}],
            "frame_trace": [{"frame": 12, "event_id": 1, "phase": "forward_swing"}],
        }

        events, traces = build_event_lookup(analysis)

        self.assertEqual(events[1]["stroke_type"], "Forehand")
        self.assertEqual(traces[12]["event"]["event_id"], 1)
        self.assertEqual(default_event_json("data/output_video.json"), "data/output_video_swing_events.json")
        self.assertEqual(default_coach_json("data/output_video.json"), "data/output_video_coach_dataset.json")
        self.assertEqual(default_output_video("data/output_video.json"), "data/output_video_swing_annotated.mp4")

    def test_builds_coach_lookup(self):
        dataset = {"events": [{"event_id": 2, "frames": {"contact_frame": 41}}]}

        lookup = build_coach_lookup(dataset)

        self.assertEqual(lookup[2]["frames"]["contact_frame"], 41)

    def test_resolves_osd_state_with_coach_milestones_and_raw_conflict(self):
        event = {"event_id": 1, "start_frame": 25, "end_frame": 77, "peak_frame": 46, "stroke_type": "Forehand"}
        trace = {
            "frame": 46,
            "phase": "forward_swing",
            "raw_swing_type": "Backhand",
            "motion_energy": 79.5,
            "contact_score": 0.0,
        }
        coach_event = {
            "frames": {"contact_frame": 41, "contact_confidence": 0.5561},
            "ball": {"shot_direction": "screen_right", "contact_confidence": 0.5561, "bounce_frame": 46},
            "racket": {"racket_lag_at_contact": 132.0},
        }

        state = resolve_osd_state(46, trace, event, coach_event)

        self.assertEqual(state["event_type"], "Forehand")
        self.assertEqual(state["milestone"], "PEAK")
        self.assertEqual(state["motion_phase"], "forward_swing")
        self.assertEqual(state["model_raw_label"], "Backhand")
        self.assertTrue(state["raw_label_conflict"])
        self.assertEqual(state["contact_frame"], 41)
        self.assertEqual(state["bounce_frame"], 46)
        self.assertEqual(state["shot_direction"], "screen_right")
        self.assertEqual(state["racket_lag_at_contact"], 132.0)

    def test_resolves_contact_milestone_without_hiding_motion_phase(self):
        event = {"event_id": 1, "start_frame": 25, "end_frame": 77, "peak_frame": 46, "stroke_type": "Forehand"}
        trace = {"frame": 41, "phase": "ready", "raw_swing_type": "Forehand", "contact_score": 0.5561}
        coach_event = {"frames": {"contact_frame": 41, "peak_frame": 46}}

        state = resolve_osd_state(41, trace, event, coach_event)

        self.assertEqual(state["milestone"], "CONTACT")
        self.assertEqual(state["motion_phase"], "ready")
        self.assertFalse(state["raw_label_conflict"])


if __name__ == "__main__":
    unittest.main()
