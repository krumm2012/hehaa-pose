"""Independent front/back trajectories qualify projected peak timing evidence."""

import math
import unittest

from kinematic_sequence import analyze_kinematic_sequence, serialize_kinematic_views


def frames(back_shift=0, back_confidence=.95, frame_stride=1):
    """Make mirrored views with known time-separated motion pulses."""
    result = []
    angles = {view: {joint: 175.0 for joint in ("hip", "shoulder")}
              for view in ("front", "back")}
    position = 100.0
    for i in range(26):
        views = {}
        for view in ("front", "back"):
            pose = {}
            for joint, peak in [("hip", 11), ("shoulder", 15)]:
                peak += back_shift if view == "back" else 0
                angles[view][joint] += 4 * math.exp(-((i - peak) / 1.4) ** 2)
                angle = math.radians(angles[view][joint])
                sign = -1 if view == "back" else 1
                conf = back_confidence if view == "back" else .95
                pose[f"left_{joint}"] = {"x": 500, "y": 500, "confidence": conf}
                pose[f"right_{joint}"] = {"x": 500 + sign * 100 * math.cos(angle),
                                         "y": 500 + 100 * math.sin(angle), "confidence": conf}
            views[view] = pose
        position += 20 * math.exp(-((i - 19) / 1.4) ** 2)
        result.append({"frame_id": i * frame_stride, "timestamp": i * .04,
                       "kinematic_views": views,
                       "rackets": [{"box": [position-10, 90, position+10, 110],
                                    "confidence": .95}]})
    return result


class KinematicCrossValidationTests(unittest.TestCase):
    def test_explicit_stale_joint_frame_cannot_validate_peak(self):
        rows = frames()
        for row in rows:
            for pose in row['kinematic_views'].values():
                for point in pose.values():
                    point.update(observed=True, source_frame_id=row['frame_id']-1)
        result = analyze_kinematic_sequence(rows, 20, 25)
        self.assertEqual(result['cross_validation']['status'], 'unavailable')

    def test_explicit_stale_racket_frame_cannot_validate_peak(self):
        rows = frames()
        for row in rows:
            row['rackets'][0].update(observed=True, source_frame_id=row['frame_id']-1)
        result = analyze_kinematic_sequence(rows, 20, 25)
        self.assertIsNone(result['racket_peak_frame'])

    def test_unresolved_hip_shoulder_is_reported_without_racket_peak(self):
        rows = frames()
        for row in rows:
            row['rackets'] = []
            for pose in row['kinematic_views'].values():
                for side in ('left', 'right'):
                    pose[f'{side}_shoulder'] = dict(pose[f'{side}_hip'])
        result = analyze_kinematic_sequence(rows, 20, 25)
        self.assertIsNone(result['racket_peak_frame'])
        self.assertFalse(result['pair_timing']['hip_to_shoulder']['resolved'])
        self.assertEqual(result['sequence_quality'], 'UNRESOLVED_AT_FRAME_RATE')
        self.assertIsNone(result['is_sequential'])

    def test_report_exposes_pair_range_including_zero_without_racket(self):
        from swing_report_builder import _build_kinematic_sequence_html
        page = _build_kinematic_sequence_html({
            'sequence_quality': 'UNRESOLVED_AT_FRAME_RATE',
            'latency_hip_to_shoulder_ms': 0,
            'pair_timing': {'hip_to_shoulder': {
                'latency_range_ms': [-81.8, 81.8], 'resolved': False}},
            'racket_peak_frame': None})
        self.assertIn('-81.8 ～ +81.8 ms', page)
        self.assertIn('先后难以分辨', page)
        self.assertIn('非统计置信区间', page)
        self.assertIn('不能判断完整动力链', page)

    def test_back_view_remains_available_when_front_view_is_occluded(self):
        rows = frames()
        for row in rows:
            row["kinematic_views"]["front"] = {}
        result = analyze_kinematic_sequence(rows, 20, 25)
        self.assertEqual(result["source_views"], ["back"])
        self.assertEqual(result["cross_validation"]["status"], "single_view")
        self.assertIsNotNone(result["latency_hip_to_shoulder_ms"])

    def test_sparse_sampling_does_not_produce_high_confidence(self):
        rows = frames()
        for i, row in enumerate(rows):
            row["timestamp"] = i * .1
        result = analyze_kinematic_sequence(rows, 20, 10)
        self.assertEqual(result["cross_validation"]["status"], "unavailable")
        self.assertEqual(result["evidence_confidence"], 0)

    def test_flat_motion_cannot_produce_a_trusted_peak(self):
        rows = frames()
        for row in rows:
            row["kinematic_views"] = rows[0]["kinematic_views"]
        result = analyze_kinematic_sequence(rows, 20, 25)
        self.assertEqual(result["cross_validation"]["status"], "unavailable")

    def test_event_aggregation_preserves_cross_validation_in_report_and_card(self):
        from swing_biomechanics import aggregate_event_biomechanics
        from swing_motion_features import extract_motion_features
        from swing_report_builder import _build_kinematic_sequence_html
        from realtime_swing_runtime import build_impact_telemetry_card

        rows = frames()
        event = {"start_frame": 0, "contact_frame": 20, "end_frame": 25}
        event["biomechanics"] = aggregate_event_biomechanics(
            event, rows, extract_motion_features(rows),
        )
        metric = event["biomechanics"]["metrics"]["kinematic_sequence"]
        self.assertEqual(metric["details"]["cross_validation"]["status"], "agree")
        self.assertFalse(metric["coach_eligible"])
        page = _build_kinematic_sequence_html(metric)
        self.assertIn("双视角一致", page)
        self.assertIn("正面髋—肩", page)
        self.assertIn("背面髋—肩", page)
        card = build_impact_telemetry_card(event)
        self.assertIn("双视角时序", card["kinematic_sequence_text"])

    def test_media_pts_take_precedence_over_nominal_time_and_reader_wall_clock(self):
        from reader_runtime import SourceMediaClock
        clock = SourceMediaClock(25)
        rows = frames(frame_stride=3)
        for i, row in enumerate(rows):
            row["timestamp"] = i * .12
            row["source_time"] = clock.observe(row['frame_id'], i * 40)
            row["timing"] = {"captured_at_unix_ns": 1791077322000000000 + i * 900000000}
        result = analyze_kinematic_sequence(rows, 60, 25)
        self.assertEqual(result["time_basis"], "media_pts")
        self.assertAlmostEqual(result["latency_hip_to_shoulder_ms"], 160, delta=40)

    def test_legacy_replay_ignores_decoder_speed(self):
        rows = frames()
        expected = analyze_kinematic_sequence(rows, 20, 25)
        for i, row in enumerate(rows):
            row['timing'] = {'captured_at_unix_ns': 1791077322000000000 + i * 1000000}
        actual = analyze_kinematic_sequence(rows, 20, 25)
        self.assertEqual(actual, expected)
        self.assertEqual(actual['time_quality'], 'legacy_unverified')

    def test_bad_media_pts_abstain_without_nominal_or_wall_clock_substitution(self):
        from reader_runtime import SourceMediaClock
        for bad in ['duplicate', 'backward']:
            rows = frames()
            clock = SourceMediaClock(25)
            for i, row in enumerate(rows):
                ms = i * 40 if i != 12 else (440 if bad == 'duplicate' else 400)
                row['source_time'] = clock.observe(i, ms)
            result = analyze_kinematic_sequence(rows, 20, 25)
            self.assertEqual(result['cross_validation']['reason'], 'duplicate_or_discontinuous_source_time')
            self.assertIsNone(result['latency_hip_to_shoulder_ms'])

    def test_stream_without_device_pts_abstains_even_with_valid_legacy_timestamp(self):
        from reader_runtime import SourceMediaClock
        rows = frames()
        for row in rows:
            row['source_time'] = SourceMediaClock(25, 'stream').observe(row['frame_id'])
        result = analyze_kinematic_sequence(rows, 20, 25)
        self.assertEqual(result['cross_validation']['reason'], 'source_media_time_unavailable')

    def test_estimated_fps_time_caps_evidence_and_preserves_gaps(self):
        from reader_runtime import SourceMediaClock
        rows = frames()
        for row in rows:
            row['source_time'] = SourceMediaClock(25).observe(row['frame_id'])
        result = analyze_kinematic_sequence(rows, 20, 25)
        self.assertEqual(result['time_quality'], 'estimated')
        self.assertEqual(result['cross_validation']['status'], 'agree')
        self.assertLessEqual(result['evidence_confidence'], .45)

    def test_mixed_source_time_basis_cannot_be_merged(self):
        from reader_runtime import SourceMediaClock
        rows = frames()
        clock = SourceMediaClock(25)
        for row in rows:
            row['source_time'] = clock.observe(row['frame_id'], row['frame_id'] * 40)
        rows[10]['source_time'] = SourceMediaClock(25).observe(10)
        result = analyze_kinematic_sequence(rows, 20, 25)
        self.assertEqual(result['cross_validation']['reason'], 'mixed_source_time_bases')

    def test_vfr_media_time_matches_physical_timeline_after_json_replay(self):
        import json
        from reader_runtime import SourceMediaClock
        rows = frames()
        reference = frames()
        clock = SourceMediaClock(25)
        for i, (row, oracle) in enumerate(zip(rows, reference)):
            pts_ms = i * 40 + (8 if i % 2 else 0)
            row['source_time'] = clock.observe(i, pts_ms)
            oracle['timestamp'] = pts_ms / 1000
            row['timestamp'] = i * .2  # Deliberately wrong nominal/compatibility time.
        replay = json.loads(json.dumps(rows))
        actual = analyze_kinematic_sequence(replay, 20, 25)
        expected = analyze_kinematic_sequence(reference, 20, 25)
        self.assertEqual(actual['views'], expected['views'])
        self.assertEqual(actual['latency_hip_to_shoulder_ms'], expected['latency_hip_to_shoulder_ms'])

    def test_large_occlusion_gap_abstains_instead_of_bridging_missing_angles(self):
        rows = frames()
        for row in rows[7:22]:
            row["kinematic_views"] = {"front": {}, "back": {}}
        result = analyze_kinematic_sequence(rows, 20, 25)
        self.assertEqual(result["cross_validation"]["status"], "unavailable")

    def test_coincident_peaks_are_unresolved_instead_of_optimal(self):
        rows = frames()
        for row in rows:
            for pose in row["kinematic_views"].values():
                for side in ("left", "right"):
                    pose[f"{side}_shoulder"] = dict(pose[f"{side}_hip"])
        result = analyze_kinematic_sequence(rows, 20, 25)
        self.assertEqual(result["sequence_quality"], "UNRESOLVED_AT_FRAME_RATE")
        self.assertIsNone(result["is_sequential"])

    def test_mirrored_views_agree_even_across_angle_wrap(self):
        result = analyze_kinematic_sequence(frames(), contact_frame=20, fps=25)
        self.assertEqual(result["cross_validation"]["status"], "agree")
        self.assertEqual(result["source_views"], ["front", "back"])
        self.assertEqual(result["cross_validation"]["peak_deltas_ms"], {"hip": 0, "shoulder": 0})
        self.assertAlmostEqual(result["latency_hip_to_shoulder_ms"], 160, delta=40)
        self.assertGreater(result["evidence_confidence"], .5)
        self.assertFalse(result["coach_eligible"])
        self.assertEqual(result["validation_status"], "unvalidated_2d_projection")

    def test_conflicting_view_is_not_silently_averaged(self):
        result = analyze_kinematic_sequence(frames(back_shift=5), 20, 25)
        self.assertEqual(result["cross_validation"]["status"], "disagree")
        self.assertIsNone(result["latency_hip_to_shoulder_ms"])
        self.assertIsNone(result["is_sequential"])
        self.assertEqual(result["evidence_confidence"], 0)

    def test_low_confidence_back_view_does_not_raise_confidence(self):
        paired = analyze_kinematic_sequence(frames(), 20, 25)
        single = analyze_kinematic_sequence(frames(back_confidence=.1), 20, 25)
        self.assertEqual(single["cross_validation"]["status"], "single_view")
        self.assertEqual(single["source_views"], ["front"])
        self.assertLess(single["evidence_confidence"], paired["evidence_confidence"])

    def test_timing_uses_timestamps_not_frame_number_gaps(self):
        result = analyze_kinematic_sequence(frames(frame_stride=3), 60, 25)
        self.assertAlmostEqual(result["latency_hip_to_shoulder_ms"], 160, delta=40)

    def test_racket_box_resize_does_not_create_a_motion_peak(self):
        rows = frames()
        for i, row in enumerate(rows):
            size = 10 + i * 5
            row["rackets"] = [{"box": [100-size, 100-size, 100+size, 100+size],
                               "confidence": .95}]
        result = analyze_kinematic_sequence(rows, 20, 25)
        self.assertIsNone(result["racket_peak_frame"])
        self.assertIsNone(result["latency_shoulder_to_racket_ms"])

    def test_missing_joint_confidence_cannot_be_cross_validated(self):
        rows = frames()
        for row in rows:
            for pose in row["kinematic_views"].values():
                for joint in pose.values():
                    joint.pop("confidence")
        result = analyze_kinematic_sequence(rows, 20, 25)
        self.assertEqual(result["cross_validation"]["status"], "unavailable")
        self.assertEqual(result["evidence_confidence"], 0)

    def test_serializer_preserves_independent_raw_views_and_confidence(self):
        from types import SimpleNamespace
        kp = SimpleNamespace(x=1, y=2, conf=.8)
        result = serialize_kinematic_views(SimpleNamespace(
            front_pose_orig={"left_hip": kp}, back_pose_orig={"left_hip": kp},
            fused_pose_orig={"left_hip": SimpleNamespace(x=99, y=99, conf=1)}))
        self.assertEqual(result["front"]["left_hip"]["x"], 1)
        self.assertEqual(result["back"]["left_hip"]["confidence"], .8)
