import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from roi_manager import ROIManager
from roi_stream_config import (
    ROIStreamProfile,
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

    def test_resolves_mirror_view_and_target_stream_id_mapping(self):
        config = {
            "roi_settings": {
                "enabled": True,
                "auto_load_config": True,
                "roi_config_path": "configs/roi_config.yaml",
            }
        }
        # 1. 导入视频映射到 court01-main
        profile_c1 = resolve_roi_stream_profile(
            config,
            "/Users/krum5539/Desktop/some_uploaded_test.mp4",
            (2560, 1440),
            target_stream_id="court01-main",
        )
        self.assertTrue(profile_c1.enabled)
        self.assertTrue(profile_c1.matched)
        self.assertEqual(profile_c1.stream_id, "court01-main")
        self.assertTrue(profile_c1.has_mirror_view)
        self.assertEqual(profile_c1.mirror_reflection_roi, [0.2541, 0.1034, 0.6125, 0.4832])
        self.assertEqual(len(profile_c1.mirror_polygon), 4)
        self.assertEqual(len(profile_c1.mirror_mask_polygon), 4)
        self.assertIn("mirror_view", profile_c1.as_metadata())

        # 2. 导入视频映射到 camera04-main
        profile_c4 = resolve_roi_stream_profile(
            config,
            "/Users/krum5539/Desktop/some_other_video.mp4",
            (2560, 1440),
            target_stream_id="camera04-main",
        )
        self.assertTrue(profile_c4.enabled)
        self.assertTrue(profile_c4.matched)
        self.assertEqual(profile_c4.stream_id, "camera04-main")
        self.assertTrue(profile_c4.has_mirror_view)
        self.assertEqual(profile_c4.mirror_reflection_roi, [0.2708, 0.1018, 0.6139, 0.4642])

    def test_roi_manager_exclusion_and_multipoint_polygon(self):
        manager = ROIManager({"roi_settings": {"enabled": True}})
        # 5-point polygon
        five_points = [(100, 100), (500, 50), (900, 100), (800, 800), (200, 800)]
        success = manager.set_roi_points(five_points)
        self.assertTrue(success)
        self.assertEqual(len(manager.roi_points), 5)
        self.assertTrue(manager.is_roi_set)

        # Center point inside ROI
        self.assertTrue(manager.is_point_in_roi((500, 400)))
        # Outside ROI
        self.assertFalse(manager.is_point_in_roi((50, 50)))

        # Add mirror exclusion zone
        mirror_poly = [(400, 300), (600, 300), (600, 500), (400, 500)]
        self.assertTrue(manager.add_exclusion_polygon(mirror_poly, label="mirror_zone"))
        self.assertEqual(len(manager.exclusion_polygons), 1)

        # Center point (500, 400) is inside mirror polygon -> should be excluded!
        self.assertTrue(manager.is_point_in_exclusion((500, 400)))
        self.assertFalse(manager.is_point_in_roi((500, 400)))

        # Point (250, 400) is inside court ROI but outside mirror -> should be kept!
        self.assertFalse(manager.is_point_in_exclusion((250, 400)))
        self.assertTrue(manager.is_point_in_roi((250, 400)))

        # Filter candidate detections
        detections = [
            {"position": [250, 400], "confidence": 0.9},  # Valid court ball
            {"position": [500, 400], "confidence": 0.95}, # Mirror reflected ball
            {"position": [50, 50], "confidence": 0.8},    # Out of boundary ball
        ]
        filtered = manager.filter_detections_by_roi(detections, detection_type="ball")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["position"], [250, 400])

    def test_get_mirror_polygon_pixels_conversion(self):
        profile = resolve_roi_stream_profile(
            {
                "roi_settings": {
                    "enabled": True,
                    "auto_load_config": True,
                    "roi_config_path": "configs/roi_config.yaml",
                }
            },
            "rtsp://192.168.1.191:554/h264/ch1/main/av_stream",
            (2560, 1440),
        )
        pixels = profile.get_mirror_polygon_pixels((2560, 1440))
        self.assertIsNotNone(pixels)
        self.assertEqual(len(pixels), 4)
        # Should be scaled to ~ (0.2541 * 2560, 0.4777 * 1440) -> (650, 688)
        self.assertAlmostEqual(pixels[0][0], int(round(0.2541 * 2560)), delta=2)
        self.assertAlmostEqual(pixels[0][1], int(round(0.4777 * 1440)), delta=2)


if __name__ == "__main__":
    unittest.main()

