import json
import os
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from performance_metrics import FpsTracker
from pose_renderer import draw_pose_keypoints
from video_writer_backend import create_video_writer
from main_pipe import MultiprocessPipeline


class LazyImportTests(unittest.TestCase):
    def test_analyzer_process_does_not_shadow_module_path_import(self):
        """TTS setup uses Path before the optional report-open branch."""
        self.assertNotIn(
            "Path",
            MultiprocessPipeline.analyzer_process.__code__.co_varnames,
        )

    def test_main_pipe_does_not_import_model_runtimes(self):
        script = (
            "import json, sys; import main_pipe; "
            "print(json.dumps({"
            "'coremltools': 'coremltools' in sys.modules, "
            "'torch': 'torch' in sys.modules, "
            "'detector': 'yolo26n_unified_detector' in sys.modules, "
            "'pose': 'pose_estimator_yolo26' in sys.modules"
            "}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            check=True,
        )
        loaded = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(
            loaded,
            {
                "coremltools": False,
                "torch": False,
                "detector": False,
                "pose": False,
            },
        )


class FpsTrackerTests(unittest.TestCase):
    def test_first_frame_start_excludes_startup_and_reports_windows(self):
        tracker = FpsTracker()
        tracker.start(now=10.0)

        snapshot = None
        for frame_number in range(1, 101):
            snapshot = tracker.tick(now=10.0 + frame_number / 25.0)

        self.assertAlmostEqual(snapshot.cumulative_fps, 25.0)
        self.assertAlmostEqual(snapshot.window_25_fps, 25.0)
        self.assertAlmostEqual(snapshot.window_100_fps, 25.0)


class PoseRendererTests(unittest.TestCase):
    def test_draws_pose_without_importing_model_runtime(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        poses = [{
            "right_shoulder": (80, 30),
            "right_elbow": (100, 55),
            "right_wrist": (120, 75),
            "left_shoulder": (60, 30),
            "left_elbow": (45, 55),
            "left_wrist": (30, 75),
            "right_hip": (85, 75),
            "left_hip": (55, 75),
            "right_knee": (90, 100),
            "left_knee": (50, 100),
        }]

        rendered = draw_pose_keypoints(frame, poses)

        self.assertGreater(int(rendered.sum()), 0)


class VideoWriterBackendTests(unittest.TestCase):
    def test_opencv_backend_preserves_h264_video_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            writer = create_video_writer(
                output_path=output_path,
                width=64,
                height=48,
                fps=25.0,
                backend="opencv",
            )
            for value in (0, 50, 100):
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            writer.release()

            capture = cv2.VideoCapture(output_path)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            output_fps = capture.get(cv2.CAP_PROP_FPS)
            capture.release()

            self.assertEqual(writer.backend_name, "opencv")
            self.assertEqual(frame_count, 3)
            self.assertAlmostEqual(output_fps, 25.0)


if __name__ == "__main__":
    unittest.main()
