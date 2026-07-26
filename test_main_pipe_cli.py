import unittest
from unittest.mock import patch

from main_pipe import build_argument_parser, main_cli


class MainPipeCliTests(unittest.TestCase):
    def test_no_save_video_flag_is_supported(self):
        parser = build_argument_parser()

        args = parser.parse_args(["--no-save-video"])

        self.assertTrue(args.no_save_video)

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

    @patch('main_pipe.MultiprocessPipeline')
    def test_main_cli_passes_swing_analysis_options_to_pipeline(self, pipeline_class):
        main_cli([
            '--input', 'data/example.mp4',
            '--analyze-swings',
            '--dominant-hand', 'left',
            '--min-peak-energy', '11.0',
        ])

        pipeline_class.assert_called_once()
        kwargs = pipeline_class.call_args.kwargs
        self.assertTrue(kwargs['analyze_swings'])
        self.assertEqual(kwargs['dominant_hand'], 'left')
        self.assertEqual(kwargs['min_peak_energy'], 11.0)
        pipeline_class.return_value.run.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
