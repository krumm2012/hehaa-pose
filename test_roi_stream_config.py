import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from roi_manager import ROIManager
from roi_stream_config import (
    resolve_roi_stream_profile,
    sanitize_stream_source,
)


class ROIStreamConfigTests(unittest.TestCase):
    def test_repository_config_selects_all_three_courts(self):
        config = yaml.safe_load(
            Path("configs/yolo26_tennis_config.yaml").read_text(encoding="utf-8")
        )
        roi_document = yaml.safe_load(
            Path("configs/roi_config.yaml").read_text(encoding="utf-8")
        )

        for configured in roi_document["streams"]:
            profile = resolve_roi_stream_profile(
                config,
                configured["stream_source"].replace(
                    "rtsp://",
                    "rtsp://admin:secret@",
                    1,
                ),
                (2560, 1440),
            )
            self.assertTrue(profile.enabled)
            self.assertEqual(profile.stream_id, configured["stream_id"])
            self.assertEqual(
                profile.points,
                tuple(tuple(point) for point in configured["roi_points"]),
            )
            self.assertNotIn("secret", profile.source)

    def test_legacy_roi_manager_loads_default_stream(self):
        manager = ROIManager({"roi_settings": {"enabled": True}})
        roi_document = yaml.safe_load(
            Path("configs/roi_config.yaml").read_text(encoding="utf-8")
        )
        expected = next(
            item
            for item in roi_document["streams"]
            if item.get("default", False)
        )

        loaded = manager.load_roi_config("configs/roi_config.yaml")

        self.assertTrue(loaded)
        self.assertEqual(
            manager.roi_points,
            [tuple(point) for point in expected["roi_points"]],
        )

    def test_sanitizes_credentials_and_query(self):
        source = "rtsp://admin:secret@192.168.1.191:554/camera/main?token=private"
        self.assertEqual(
            sanitize_stream_source(source),
            "rtsp://192.168.1.191:554/camera/main",
        )

    def test_matches_sanitized_rtsp_and_scales_points(self):
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "roi.yaml"
            config_path.write_text(
                """
roi_enabled: true
stream_id: court-1
stream_label: Court One
stream_source: rtsp://192.168.1.191:554/camera/main
frame_size: [2560, 1440]
roi_points:
  - [0, 0]
  - [1280, 0]
  - [2560, 1440]
  - [0, 1440]
""",
                encoding="utf-8",
            )
            profile = resolve_roi_stream_profile(
                {
                    "roi_settings": {
                        "enabled": True,
                        "auto_load_config": True,
                        "roi_config_path": str(config_path),
                    }
                },
                "rtsp://admin:secret@192.168.1.191:554/camera/main",
                (1280, 720),
            )

        self.assertTrue(profile.enabled)
        self.assertTrue(profile.matched)
        self.assertEqual(profile.source, "rtsp://192.168.1.191:554/camera/main")
        self.assertEqual(profile.points, ((0, 0), (640, 0), (1279, 719), (0, 719)))

    def test_does_not_apply_camera_roi_to_another_stream(self):
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "roi.yaml"
            config_path.write_text(
                """
roi_enabled: true
stream_source: rtsp://10.0.0.1/camera
roi_points: [[0, 0], [10, 0], [10, 10], [0, 10]]
""",
                encoding="utf-8",
            )
            profile = resolve_roi_stream_profile(
                {
                    "roi_settings": {
                        "enabled": True,
                        "roi_config_path": str(config_path),
                    }
                },
                "rtsp://10.0.0.2/camera",
                (100, 100),
            )

        self.assertFalse(profile.enabled)
        self.assertIn("No ROI profile", profile.reason)

    def test_keeps_legacy_unbound_roi_compatible(self):
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "roi.yaml"
            config_path.write_text(
                """
roi_enabled: true
roi_points: [[1, 2], [30, 2], [30, 40], [1, 40]]
""",
                encoding="utf-8",
            )
            profile = resolve_roi_stream_profile(
                {
                    "roi_settings": {
                        "enabled": True,
                        "roi_config_path": str(config_path),
                    }
                },
                "data/input.mp4",
                (100, 50),
            )

        self.assertTrue(profile.enabled)
        self.assertEqual(profile.points[0], (1, 2))


if __name__ == "__main__":
    unittest.main()
