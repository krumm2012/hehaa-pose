"""Source media time must survive event/coach consumers without FPS invention."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from local_realtime_coach import LocalRealtimeCoach
from swing_coach_data_collector import build_coach_dataset
from test_swing_evidence_builder import sample_documents
from event_source_timing import analyze_event_source_timing, source_frame_navigation
from coach_evidence_policy import build_coach_decision_policy
from deepseek_evidence_view import build_deepseek_evidence_view
from swing_evidence_builder import build_swing_evidence_packet
from swing_report_builder import build_report_payload, render_report_html


def source_documents():
    frames, analysis, _ = sample_documents()
    for row, seconds in zip(frames['frames'], (0., .021, .08, .24, .40)):
        row['source_time'] = {'schema_version': 'tennis.source-time.v1',
            'source_kind': 'video_file', 'source_frame_id': row['frame_id'],
            'timestamp_seconds': seconds, 'basis': 'media_pts', 'quality': 'reported',
            'exposure_time_verified': False}
    return frames, analysis


class EventSourceTimingIntegrationTests(unittest.TestCase):
    def test_report_does_not_guess_another_video_thumbnail_and_renders_single_advice(self):
        frames, analysis = source_documents()
        with TemporaryDirectory() as tmp:
            root = Path(tmp); media = root/'media'; media.mkdir()
            source = media/'source.mp4'; frames['video_info']['path'] = str(source)
            (media/'other_video_frame_11.jpg').write_bytes(b'not matching evidence')
            analysis['events'][0]['coach_advice'] = {'message': '触球候选需确认', 'code': 'review', 'confidence': 0.}
            f, e = root/'frames.json', root/'events.json'
            f.write_text(json.dumps(frames)); e.write_text(json.dumps(analysis))
            payload = build_report_payload(str(f), str(e), video_path=str(source))
            page = render_report_html(payload, str(root/'report.html'))
            self.assertNotIn('other_video_frame_11.jpg', page)
            self.assertIn('触球候选需确认', page)
            self.assertEqual(page.count('class="coach-advice-item"'), 1)

    def test_navigation_refuses_legacy_duplicate_and_bad_time_records(self):
        frames, _ = source_documents()
        bad_lists = [[], [{**frames['frames'][0], 'source_time': None}],
                     frames['frames'] + [deepcopy(frames['frames'][0])]]
        wrong = deepcopy(frames['frames']); wrong[2]['source_time']['timestamp_seconds'] = .001
        bad_lists.append(wrong)
        for rows in bad_lists:
            self.assertEqual(source_frame_navigation(rows)['frames'], [])

    def test_navigation_preserves_known_sparse_frame_identity_without_filling_gaps(self):
        frames, _ = source_documents()
        rows = frames['frames'][::2]
        nav = source_frame_navigation(rows)
        self.assertEqual(nav['frames'], [[9, 0.], [11, .08], [13, .4]])

    def test_phase_display_escapes_labels_and_requires_phase_qualification(self):
        from swing_report_builder import _build_event_source_timing_html
        frames, analysis = source_documents()
        result = analyze_event_source_timing(analysis['events'][0], frames['frames'], analysis['frame_trace'])
        result['phase_durations_seconds'] = {'<img src=x onerror=alert(1)>': .2}
        page = _build_event_source_timing_html(result)
        self.assertNotIn('<img src=x', page)
        self.assertIn('&lt;img', page)
        result['phase_status'] = 'unavailable'
        self.assertNotIn('0.200 s', _build_event_source_timing_html(result))

    def test_report_preserves_source_time_and_does_not_apply_it_to_another_video(self):
        frames, analysis = source_documents()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source.mp4'
            frames['video_info']['path'] = str(source)
            fpath, epath = root/'frames.json', root/'events.json'
            fpath.write_text(json.dumps(frames)); epath.write_text(json.dumps(analysis))
            payload = build_report_payload(str(fpath), str(epath), video_path=str(source))
            self.assertEqual(payload['timeline']['source_time_navigation']['frames'],
                             [[9, 0.], [10, .021], [11, .08], [12, .24], [13, .4]])
            self.assertAlmostEqual(payload['events'][0]['phase_timing']['duration_seconds'], .219)
            page = render_report_html(payload, str(root/'report.html'))
            self.assertIn('候选事件媒体时长', page)
            self.assertIn('0.219 s', page)
            self.assertNotIn('safeFrame / fps', page)
            other = build_report_payload(str(fpath), str(epath), video_path=str(root/'annotated.mp4'))
            self.assertEqual(other['timeline']['source_time_navigation']['frames'], [])
            self.assertIn('selected_video_not_bound_to_source',
                          other['timeline']['source_time_navigation']['reasons'])

    def test_compact_language_evidence_preserves_source_clock_and_does_not_mutate_input(self):
        frames, analysis = source_documents()
        before = deepcopy(frames)
        coach = build_coach_dataset(frames, analysis)
        packet = build_swing_evidence_packet(frames, analysis, coach, event_id=1)
        view = build_deepseek_evidence_view(packet)
        self.assertEqual(view['frame_sequence'][0]['source_time'], frames['frames'][1]['source_time'])
        self.assertEqual(view['frame_sequence'][0]['timestamp_semantics'],
                         'compatibility_timestamp_inspect_source_time')
        self.assertEqual(frames, before)

    def test_phase_support_conserves_elapsed_time_without_inventing_last_frame_duration(self):
        frames, analysis = source_documents()
        event = analysis['events'][0]
        timing = analyze_event_source_timing(event, frames['frames'], analysis['frame_trace'])
        self.assertAlmostEqual(sum(timing['phase_durations_seconds'].values()), .219)
        self.assertEqual(timing['phase_interval_source_frames'],
                         {'forward_swing': [[10, 11]], 'contact_candidate': [[11, 12]]})
        self.assertIsNone(timing['recovery_time_seconds'])
        self.assertFalse(timing['coach_eligible'])
        self.assertFalse(timing['accuracy_validated'])
        self.assertFalse(timing['sensor_exposure_verified'])

    def test_first_zero_source_time_and_negative_peak_offset_are_retained(self):
        frames, analysis = source_documents()
        event = {**analysis['events'][0], 'start_frame': 9, 'end_frame': 13, 'peak_frame': 12}
        timing = analyze_event_source_timing(event, frames['frames'], analysis['frame_trace'])
        self.assertEqual(timing['anchors']['start_frame']['timestamp_seconds'], 0.)
        self.assertAlmostEqual(timing['duration_seconds'], .4)
        self.assertAlmostEqual(timing['peak_to_contact_offset_seconds'], -.16)

    def test_invalid_mixed_and_nonmonotonic_source_times_cannot_fall_back_to_fps(self):
        variants = [{'timestamp_seconds': float('nan')}, {'timestamp_seconds': float('inf')},
                    {'timestamp_seconds': True}, {'timestamp_seconds': -.1},
                    {'timestamp_seconds': .021}, {'timestamp_seconds': .001},
                    {'source_frame_id': 99}, {'source_frame_id': True},
                    {'basis': 'nominal_fps', 'quality': 'estimated'},
                    {'quality': 'discontinuous'}, {'source_kind': 'stream'},
                    {'schema_version': 'unknown'}]
        for changes in variants:
            with self.subTest(changes=changes):
                frames, analysis = source_documents()
                frames['frames'][2]['source_time'].update(changes)
                event = build_coach_dataset(frames, analysis)['events'][0]
                self.assertIsNone(event['timing']['duration_seconds'])
                self.assertIsNone(event['timing']['tempo_consistency'])
                self.assertTrue(event['timing']['source_time_evidence']['reasons'])

    def test_missing_anchors_and_duplicate_records_do_not_supply_time(self):
        for duplicate in (False, True):
            frames, analysis = source_documents()
            rows = frames['frames']
            if duplicate:
                rows.append(deepcopy(rows[2]))
            else:
                rows.pop(2)
            timing = analyze_event_source_timing(analysis['events'][0], rows)
            self.assertIsNone(timing['duration_seconds'])

    def test_source_frame_gap_keeps_anchor_elapsed_time_but_cannot_fill_phase_support(self):
        frames, analysis = source_documents()
        event = {**analysis['events'][0], 'contact_frame': 12, 'peak_frame': 12}
        frames['frames'].pop(2)
        timing = analyze_event_source_timing(event, frames['frames'], analysis['frame_trace'])
        self.assertAlmostEqual(timing['duration_seconds'], .219)
        self.assertEqual(timing['observation_frame_gaps'], [[10, 12]])
        self.assertIsNone(timing['phase_durations_seconds'])

    def test_missing_duplicate_or_wrong_event_trace_does_not_become_phase_time(self):
        frames, analysis = source_documents()
        traces = analysis['frame_trace']
        for bad in ([], traces + [deepcopy(traces[1])],
                    [{**row, 'event_id': 2} for row in traces]):
            timing = analyze_event_source_timing(analysis['events'][0], frames['frames'], bad)
            self.assertAlmostEqual(timing['duration_seconds'], .219)
            self.assertIsNone(timing['phase_durations_seconds'])
            self.assertTrue(timing['phase_reasons'])

    def test_language_policy_rejects_legacy_phase_diagnoses_and_tempo(self):
        metrics = {'timing': {'phase_durations_frames': {'follow_through': 2},
                    'preparation_timing_quality': 'quick', 'tempo_consistency': .1},
                   'diagnosis_tags': ['short_follow_through'], 'body': {}, 'scores': {}}
        policy = build_coach_decision_policy({'warnings': []}, metrics, {})
        self.assertIn('follow_through', policy['blocked_advice_topics'])
        self.assertIn('tempo', policy['blocked_advice_topics'])
        self.assertIn('phase_quality_from_frame_counts', policy['prohibited_claims'])
        self.assertEqual(policy['advice_candidates'], [])

    def test_actual_dataset_uses_source_time_not_declared_fps_or_receipt_clock(self):
        frames, analysis = source_documents()
        timings = []
        for fps in (10, 25, 50, 240):
            variant = deepcopy(frames)
            variant['video_info']['fps'] = fps
            for row in variant['frames']:
                row['timestamp'] = row['frame_id'] / fps
                row['timing'] = {'captured_at_unix_ns': row['frame_id'] * fps * 10**9}
            timing = build_coach_dataset(variant, analysis)['events'][0]['timing']
            self.assertAlmostEqual(timing['duration_seconds'], .24 - .021)
            self.assertAlmostEqual(timing['start_to_contact_seconds'], .08 - .021)
            self.assertAlmostEqual(timing['contact_to_end_seconds'], .24 - .08)
            timings.append(timing)
        self.assertTrue(all(value == timings[0] for value in timings))

    def test_legacy_frames_cannot_invent_measured_seconds_or_phase_scores(self):
        frames, analysis, _ = sample_documents()
        event = build_coach_dataset(frames, analysis)['events'][0]
        self.assertIsNone(event['timing']['duration_seconds'])
        self.assertIsNone(event['scores']['preparation_score'])
        self.assertIsNone(event['scores']['follow_through_score'])
        self.assertNotIn('short_follow_through', event['diagnosis_tags'])

    def test_unvalidated_frame_count_rules_cannot_issue_technical_corrections(self):
        for phases in ({}, {'backswing': 1, 'follow_through': 8},
                       {'backswing': 8, 'follow_through': 2}):
            with self.subTest(phases=phases):
                event = {'confidence': .9, 'quality_flags': {'warnings': []},
                         'phase_counts': phases}
                advice = LocalRealtimeCoach().advise_all(event)
                self.assertFalse(any(x['code'] in ('short_backswing', 'short_follow_through')
                                     for x in advice))


if __name__ == '__main__':
    unittest.main()
