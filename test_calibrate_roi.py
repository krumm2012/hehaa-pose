import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import yaml

from calibrate_roi import (
    fit_frame_size,
    points_to_source,
    save_roi_document,
    select_roi_points,
    update_roi_document,
    validate_roi_points,
)


class CalibrateROITests(unittest.TestCase):
    @patch("calibrate_roi.ROIManager")
    def test_default_mouse_confirmation_returns_without_second_keypress(
        self,
        roi_manager_class,
    ):
        roi_manager_class.return_value.interactive_roi_selection.return_value = [
            (100, 100),
            (900, 100),
            (1000, 700),
            (50, 700),
        ]
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        with patch(
            "calibrate_roi.cv2.waitKey",
            side_effect=AssertionError("unexpected second confirmation"),
        ):
            points = select_roi_points(
                frame,
                {},
                1280,
                800,
                confirm_preview=False,
            )

        self.assertEqual(
            points,
            [(100, 100), (900, 100), (1000, 700), (50, 700)],
        )

    @patch("calibrate_roi.ROIManager")
    def test_accepts_court02_counterclockwise_click_order(
        self,
        roi_manager_class,
    ):
        roi_manager_class.return_value.interactive_roi_selection.side_effect = [
            [(609, 154), (488, 608), (1102, 610), (971, 154)],
            [],
        ]
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        points = select_roi_points(
            frame,
            {},
            1280,
            800,
            confirm_preview=False,
        )

        self.assertEqual(
            points,
            [(609, 154), (971, 154), (1102, 610), (488, 608)],
        )

    def test_fits_2560_frame_and_restores_source_points(self):
        display_size, scale = fit_frame_size((2560, 1440), 1280, 800)

        self.assertEqual(display_size, (1280, 720))
        self.assertEqual(scale, 0.5)
        self.assertEqual(
            points_to_source(
                [(425, 65), (850, 70), (975, 660), (290, 660)],
                scale,
                (2560, 1440),
            ),
            [(850, 130), (1700, 140), (1950, 1320), (580, 1320)],
        )

    def test_updates_root_profile_and_redacts_credentials(self):
        document = {
            "roi_enabled": True,
            "stream_id": "old-id",
            "stream_source": "rtsp://10.0.0.1/old",
            "custom_field": "preserved",
        }

        updated = update_roi_document(
            document,
            "rtsp://admin:secret@192.168.1.191:554/camera/main?token=x",
            (2560, 1440),
            [(1, 2), (3, 4), (5, 6), (7, 8)],
            "court01-main",
            "Court 01",
        )

        self.assertEqual(updated["custom_field"], "preserved")
        self.assertEqual(
            updated["stream_source"],
            "rtsp://192.168.1.191:554/camera/main",
        )
        self.assertNotIn("secret", yaml.safe_dump(updated))
        self.assertEqual(updated["roi_points"][2], [5, 6])

    def test_updates_only_the_matching_stream_in_multi_court_config(self):
        document = {
            "version": "2.0",
            "streams": [
                {
                    "stream_id": "court01-main",
                    "stream_source": "rtsp://192.168.1.191/camera",
                    "roi_points": [[1, 1], [2, 1], [2, 2], [1, 2]],
                },
                {
                    "stream_id": "court02-main",
                    "stream_source": "rtsp://192.168.1.192/camera",
                    "roi_points": [[3, 3], [4, 3], [4, 4], [3, 4]],
                },
            ],
        }

        updated = update_roi_document(
            document,
            "rtsp://admin:secret@192.168.1.192/camera",
            (2560, 1440),
            [(10, 20), (30, 20), (30, 40), (10, 40)],
            "court02-main",
            "Court 02",
        )

        self.assertEqual(
            updated["streams"][0]["roi_points"],
            [[1, 1], [2, 1], [2, 2], [1, 2]],
        )
        self.assertEqual(
            updated["streams"][1]["roi_points"],
            [[10, 20], [30, 20], [30, 40], [10, 40]],
        )
        self.assertNotIn("secret", yaml.safe_dump(updated))

    def test_validates_click_order_and_minimum_area(self):
        valid, _ = validate_roi_points(
            [(100, 100), (900, 100), (1000, 700), (50, 700)],
            (1280, 720),
        )
        crossed, crossed_reason = validate_roi_points(
            [(100, 100), (900, 100), (50, 700), (1000, 700)],
            (1280, 720),
        )
        tiny, tiny_reason = validate_roi_points(
            [(10, 10), (20, 10), (20, 20), (10, 20)],
            (1280, 720),
        )

        self.assertTrue(valid)
        self.assertFalse(crossed)
        self.assertIn("顺序", crossed_reason)
        self.assertFalse(tiny)
        self.assertIn("5%", tiny_reason)

    def test_atomic_save_retains_backup(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "roi.yaml"
            path.write_text("roi_enabled: false\n", encoding="utf-8")

            backup = save_roi_document(
                path,
                {"roi_enabled": True, "roi_points": [[1, 2]]},
            )

            saved = yaml.safe_load(path.read_text(encoding="utf-8"))
            original = yaml.safe_load(backup.read_text(encoding="utf-8"))

        self.assertTrue(saved["roi_enabled"])
        self.assertFalse(original["roi_enabled"])


if __name__ == "__main__":
    unittest.main()
