import tempfile
import unittest

import numpy as np

from highlight_service import HighlightService


class HighlightServiceTests(unittest.TestCase):
    def test_disabled_service_noops_on_recent_and_pending(self):
        service = HighlightService({"enabled": False}, fps=25.0)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        service.add_recent_frame(1, frame)
        self.assertEqual(len(service.recent_frames), 0)

        service.pending_clips = [
            {"start": 2, "end": 3, "center": 2, "frames": {2: frame.copy()}}
        ]
        service.process_pending_clips(4, frame)
        # disabled 时不处理队列
        self.assertEqual(len(service.pending_clips), 1)

    def test_process_pending_clips_flushes_finished_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = HighlightService({"enabled": True, "output_dir": tmpdir}, fps=25.0)
            frame = np.zeros((32, 32, 3), dtype=np.uint8)

            called = {"count": 0}

            def _fake_save_clip(frames_map, out_dir, center_frame_num, fps, tag="hit"):
                called["count"] += 1
                return f"{out_dir}/fake_{center_frame_num}.mp4"

            service._save_highlight_clip = _fake_save_clip
            service.pending_clips = [
                {"start": 1, "end": 1, "center": 1, "frames": {1: frame.copy()}}
            ]

            service.process_pending_clips(frame_num=2, frame=frame)
            self.assertEqual(called["count"], 1)
            self.assertEqual(len(service.pending_clips), 0)


if __name__ == "__main__":
    unittest.main()
