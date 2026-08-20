import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from detection_frame_context import DetectionFrameContext


class _FakeROIManager:
    def __init__(self):
        self.is_roi_set = True
        self.adjust_calls = []
        self.filter_calls = []

    def get_roi_mask(self, frame_shape):
        h, w = frame_shape
        return np.ones((h, w), dtype=np.uint8)

    def get_roi_bounding_box(self):
        # x1, y1, x2, y2
        return (10, 20, 30, 40)

    def adjust_detection_coordinates(self, detections, roi_offset, detection_type):
        self.adjust_calls.append((detection_type, roi_offset))
        return [f"{detection_type}_adjusted"]

    def filter_detections_by_roi(self, detections, detection_type="general"):
        self.filter_calls.append((detection_type, tuple(detections)))
        return ["ball_filtered"]


class DetectionFrameContextTests(unittest.TestCase):
    def test_frame_context_logging_is_silent_by_default(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        output = StringIO()

        with redirect_stdout(output):
            DetectionFrameContext.build(
                frame_num=30,
                frame=frame,
                roi_manager=_FakeROIManager(),
                config={"roi_settings": {"crop_margin": 5}},
            )

        self.assertEqual(output.getvalue(), "")

    def test_build_and_adjust(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        roi_manager = _FakeROIManager()
        config = {
            "roi_settings": {"crop_margin": 5, "filter_balls_outside_roi": True},
            "pose_estimation_debug": {"use_roi_detection": True},
        }

        ctx = DetectionFrameContext.build(
            frame_num=30,
            frame=frame,
            roi_manager=roi_manager,
            config=config,
        )

        self.assertEqual(ctx.roi_offset, (5, 15))
        self.assertEqual(ctx.detection_frame.shape, (30, 30, 3))
        self.assertEqual(ctx.pose_detection_frame.shape, (30, 30, 3))
        self.assertTrue(ctx.pose_use_roi)

        pose_results, ball_positions, racket_detections = ctx.adjust_detections(
            pose_results=[["pose"]],
            ball_positions=[["ball"]],
            racket_detections=[["racket"]],
        )

        self.assertEqual(pose_results, ["pose_adjusted"])
        self.assertEqual(ball_positions, ["ball_filtered"])
        self.assertEqual(racket_detections, ["racket_adjusted"])
        self.assertEqual(
            roi_manager.adjust_calls,
            [("pose", (5, 15)), ("ball", (5, 15)), ("racket", (5, 15))],
        )
        self.assertEqual(len(roi_manager.filter_calls), 1)
        self.assertEqual(roi_manager.filter_calls[0][0], "ball")


if __name__ == "__main__":
    unittest.main()
