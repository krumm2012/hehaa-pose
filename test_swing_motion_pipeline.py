import unittest
from pathlib import Path
import json

from swing_event_classifier import classify_swing_event
from swing_event_analyzer import analyze_frame_records
from swing_event_segmenter import segment_swing_events
from swing_event_video_renderer import build_event_lookup, default_event_json, default_output_video
from swing_motion_features import extract_motion_features


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

    def test_current_fixture_uses_screen_left_as_true_right_forehand(self):
        fixture = Path("data/output_video.json")
        if not fixture.exists():
            self.skipTest("current video analysis fixture is not available")
        frames = json.loads(fixture.read_text(encoding="utf-8"))["frames"]

        analysis = analyze_frame_records(frames)

        self.assertEqual(analysis["summary"]["swing_event_count"], 3)
        self.assertEqual(analysis["summary"]["swing_event_type_counts"], {"Forehand": 3})


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
        self.assertEqual(default_output_video("data/output_video.json"), "data/output_video_swing_annotated.mp4")


if __name__ == "__main__":
    unittest.main()
