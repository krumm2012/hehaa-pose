import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from calibrate_mirror import MirrorCalibrationServer


class CalibrateMirrorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)

        # Create dummy video (single blank frame or minimal mp4)
        import cv2
        self.video_path = self.tmp / "dummy.mp4"
        out = cv2.VideoWriter(
            str(self.video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (640, 480),
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            out.write(frame)
        out.release()

        # Create dummy dual_view_config.yaml
        self.dual_config_path = self.tmp / "dual_view_config.yaml"
        self.dual_config_path.write_text(
            yaml.safe_dump(
                {
                    "mirror_view": {
                        "enabled": True,
                        "reflection_roi": [0.20, 0.10, 0.60, 0.50],
                        "polygon": [[0.20, 0.50], [0.20, 0.10], [0.60, 0.10], [0.60, 0.50]],
                        "mask_polygon": [],
                    }
                }
            ),
            encoding="utf-8",
        )

        # Create dummy roi_config.yaml
        self.roi_config_path = self.tmp / "roi_config.yaml"
        self.roi_config_path.write_text(
            yaml.safe_dump(
                {
                    "version": "2.0",
                    "streams": [
                        {
                            "stream_id": "court01-main",
                            "stream_label": "Court 01",
                            "stream_source": "rtsp://192.168.1.191:554/live",
                            "default": True,
                            "mirror_view": {
                                "enabled": True,
                                "reflection_roi": [0.42, 0.0, 0.81, 0.32],
                                "polygon": [[0.42, 0.30], [0.42, 0.0], [0.81, 0.0], [0.81, 0.32]],
                                "mask_polygon": [[0.64, 0.15], [0.69, 0.15], [0.69, 0.24], [0.64, 0.24]],
                            },
                        },
                        {
                            "stream_id": "court02-main",
                            "stream_label": "Court 02",
                            "stream_source": "rtsp://192.168.1.192:554/live",
                            "mirror_view": {
                                "enabled": True,
                                "reflection_roi": [0.30, 0.10, 0.70, 0.50],
                                "polygon": [[0.30, 0.50], [0.30, 0.10], [0.70, 0.10], [0.70, 0.50]],
                                "mask_polygon": [],
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        self.html_path = self.tmp / "test.html"
        self.html_path.write_text("<html></html>", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_current_config_resolves_specified_stream(self):
        server = MirrorCalibrationServer(
            video_path=self.video_path,
            config_path=self.dual_config_path,
            html_path=self.html_path,
            roi_config_path=self.roi_config_path,
        )

        # 默认匹配 default stream (court01-main)
        c1 = server.get_current_config()
        self.assertEqual(c1["stream_id"], "court01-main")
        self.assertEqual(c1["reflection_roi"], [0.42, 0.0, 0.81, 0.32])
        self.assertEqual(len(c1["available_streams"]), 2)

        # 显式查询 court02-main
        c2 = server.get_current_config(stream_id="court02-main")
        self.assertEqual(c2["stream_id"], "court02-main")
        self.assertEqual(c2["reflection_roi"], [0.30, 0.10, 0.70, 0.50])

    def test_save_config_updates_target_stream_and_dual_view(self):
        server = MirrorCalibrationServer(
            video_path=self.video_path,
            config_path=self.dual_config_path,
            html_path=self.html_path,
            roi_config_path=self.roi_config_path,
        )

        new_roi = [0.33, 0.11, 0.66, 0.44]
        new_poly = [[0.33, 0.44], [0.33, 0.11], [0.66, 0.11], [0.66, 0.44]]
        new_mask = [[0.40, 0.20], [0.50, 0.20], [0.50, 0.30], [0.40, 0.30]]

        res = server.save_config(
            reflection_roi=new_roi,
            polygon=new_poly,
            mask_polygon=new_mask,
            stream_id="court02-main",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["stream_id"], "court02-main")

        # Verify roi_config.yaml updated court02-main but kept court01-main intact
        with open(self.roi_config_path, "r", encoding="utf-8") as f:
            roi_doc = yaml.safe_load(f)

        court01 = next(s for s in roi_doc["streams"] if s["stream_id"] == "court01-main")
        self.assertEqual(court01["mirror_view"]["reflection_roi"], [0.42, 0.0, 0.81, 0.32])

        court02 = next(s for s in roi_doc["streams"] if s["stream_id"] == "court02-main")
        self.assertEqual(court02["mirror_view"]["reflection_roi"], new_roi)
        self.assertEqual(court02["mirror_view"]["polygon"], new_poly)
        self.assertEqual(court02["mirror_view"]["mask_polygon"], new_mask)

        # Verify dual_view_config.yaml updated as fallback
        with open(self.dual_config_path, "r", encoding="utf-8") as f:
            dual_doc = yaml.safe_load(f)
        self.assertEqual(dual_doc["mirror_view"]["reflection_roi"], new_roi)


if __name__ == "__main__":
    unittest.main()
