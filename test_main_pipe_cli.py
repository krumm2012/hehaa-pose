import unittest
from unittest.mock import patch

from main_pipe import build_argument_parser, main_cli


class MainPipeCliTests(unittest.TestCase):
    def test_no_save_video_flag_is_supported(self):
        parser = build_argument_parser()

        args = parser.parse_args(["--no-save-video"])

        self.assertTrue(args.no_save_video)

    def test_hdmi_output_options_are_supported(self):
        parser = build_argument_parser()

        args = parser.parse_args([
            "--hdmi-output",
            "--display-origin",
            "1512",
            "0",
        ])

        self.assertTrue(args.hdmi_output)
        self.assertEqual(args.display_origin, [1512, 0])

    def test_swing_analysis_options_are_supported(self):
        parser = build_argument_parser()

        args = parser.parse_args([
            '--analyze-swings',
            '--dominant-hand', 'left',
            '--min-peak-energy', '12.5',
            '--active-energy', '6.25',
            '--min-event-frames', '10',
            '--max-internal-gap', '4',
            '--min-event-gap', '24',
        ])

        self.assertTrue(args.analyze_swings)
        self.assertEqual(args.dominant_hand, 'left')
        self.assertEqual(args.min_peak_energy, 12.5)
        self.assertEqual(args.active_energy, 6.25)
        self.assertEqual(args.min_event_frames, 10)
        self.assertEqual(args.max_internal_gap, 4)
        self.assertEqual(args.min_event_gap, 24)

    def test_realtime_swing_output_options_are_supported(self):
        parser = build_argument_parser()

        args = parser.parse_args([
            '--live-mode',
            '--realtime-swing-events',
            '--realtime-swing-json', 'data/live/events.json',
            '--realtime-swing-html', 'data/live/index.html',
            '--realtime-swing-clips-dir', 'data/live/clips',
            '--realtime-analysis-interval', '4',
            '--realtime-settle-frames', '12',
            '--realtime-window-frames', '180',
            '--realtime-clip-workers', '2',
            '--realtime-frame-output',
            '--realtime-frame-jsonl', 'data/live/frames.jsonl',
            '--realtime-frame-snapshot-json', 'data/live/frames_latest.json',
            '--realtime-frame-snapshot-size', '120',
            '--realtime-frame-flush-interval', '4',
            '--realtime-coach',
            '--realtime-coach-max-chars', '12',
            '--realtime-coach-max-suggestions', '2',
            '--realtime-coach-min-confidence', '0.6',
            '--deepseek-coach',
            '--deepseek-model', 'deepseek-v4-flash',
            '--deepseek-base-url', 'http://127.0.0.1:9000/v1',
            '--deepseek-api-key-env', 'TEST_DEEPSEEK_KEY',
            '--deepseek-timeout-seconds', '2.5',
            '--deepseek-workers', '3',
            '--deepseek-coach-max-chars', '13',
            '--realtime-open-report',
        ])

        self.assertTrue(args.live_mode)
        self.assertTrue(args.realtime_swing_events)
        self.assertEqual(args.realtime_swing_json, 'data/live/events.json')
        self.assertEqual(args.realtime_swing_html, 'data/live/index.html')
        self.assertEqual(args.realtime_swing_clips_dir, 'data/live/clips')
        self.assertEqual(args.realtime_analysis_interval, 4)
        self.assertEqual(args.realtime_settle_frames, 12)
        self.assertEqual(args.realtime_window_frames, 180)
        self.assertEqual(args.realtime_clip_workers, 2)
        self.assertTrue(args.realtime_frame_output)
        self.assertEqual(args.realtime_frame_jsonl, 'data/live/frames.jsonl')
        self.assertEqual(args.realtime_frame_snapshot_json, 'data/live/frames_latest.json')
        self.assertEqual(args.realtime_frame_snapshot_size, 120)
        self.assertEqual(args.realtime_frame_flush_interval, 4)
        self.assertTrue(args.realtime_coach)
        self.assertEqual(args.realtime_coach_max_chars, 12)
        self.assertEqual(args.realtime_coach_max_suggestions, 2)
        self.assertEqual(args.realtime_coach_min_confidence, 0.6)
        self.assertTrue(args.deepseek_coach)
        self.assertEqual(args.deepseek_model, 'deepseek-v4-flash')
        self.assertEqual(args.deepseek_base_url, 'http://127.0.0.1:9000/v1')
        self.assertEqual(args.deepseek_api_key_env, 'TEST_DEEPSEEK_KEY')
        self.assertEqual(args.deepseek_timeout_seconds, 2.5)
        self.assertEqual(args.deepseek_workers, 3)
        self.assertEqual(args.deepseek_coach_max_chars, 13)
        self.assertTrue(args.realtime_open_report)

    @patch('main_pipe.MultiprocessPipeline')
    def test_main_cli_passes_swing_analysis_options_to_pipeline(self, pipeline_class):
        main_cli([
            '--input', 'data/example.mp4',
            '--analyze-swings',
            '--realtime-swing-events',
            '--realtime-frame-output',
            '--realtime-coach',
            '--deepseek-coach',
            '--hdmi-output',
            '--display-origin', '1512', '0',
            '--dominant-hand', 'left',
            '--min-peak-energy', '11.0',
        ])

        pipeline_class.assert_called_once()
        kwargs = pipeline_class.call_args.kwargs
        self.assertTrue(kwargs['analyze_swings'])
        self.assertTrue(kwargs['realtime_swing_events'])
        self.assertTrue(kwargs['realtime_frame_output'])
        self.assertTrue(kwargs['realtime_coach'])
        self.assertTrue(kwargs['deepseek_coach_options']['enabled'])
        self.assertTrue(kwargs['hdmi_output'])
        self.assertEqual(kwargs['display_origin'], [1512, 0])
        self.assertIsNone(kwargs['deepseek_coach_options']['model'])
        self.assertEqual(kwargs['dominant_hand'], 'left')
        self.assertEqual(kwargs['min_peak_energy'], 11.0)
        pipeline_class.return_value.run.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
