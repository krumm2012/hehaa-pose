"""A local regression must never silently inherit another camera's geometry."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.validate_racket_pose_pipeline import run_case


class RegressionROIPolicyTests(unittest.TestCase):
    def test_project_policy_requires_explicit_local_video_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(output=directory,roi_policy='project',roi_stream_id=None)
            with self.assertRaisesRegex(ValueError,'roi-stream-id'):
                run_case(args,Path('48.43.mp4'),'exclusive','buffered',1,0)
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_cannot_relabel_disabled_cached_measurements_as_project_roi(self):
        with tempfile.TemporaryDirectory() as directory:
            session=Path(directory)/'buffered-exclusive-s1-48.43-r0'
            session.mkdir()
            original=json.dumps({'returncode':0,'frames':250})
            (session/'measurement.json').write_text(original)
            args=SimpleNamespace(output=directory,roi_policy='project',roi_stream_id='court02-main')
            with self.assertRaisesRegex(ValueError,'new output directory'):
                run_case(args,Path('48.43.mp4'),'exclusive','buffered',1,0)
            self.assertEqual((session/'measurement.json').read_text(),original)

    def test_unknown_binding_does_not_fall_back_to_default_camera(self):
        with tempfile.TemporaryDirectory() as directory, patch('scripts.validate_racket_pose_pipeline.subprocess.Popen') as launch:
            args=SimpleNamespace(output=directory,roi_policy='project',roi_stream_id='missing-camera',save_video=True)
            with self.assertRaisesRegex(ValueError,'Requested project ROI'):
                run_case(args,Path('48.43.mp4'),'exclusive','buffered',1,0)
            launch.assert_not_called()


if __name__=='__main__':unittest.main()
