import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from run_swing_report import render_and_build_report


class RunSwingReportTests(unittest.TestCase):
    @patch("run_swing_report.write_report_html")
    @patch("run_swing_report.build_report_payload", return_value={"events": []})
    @patch("run_swing_report.render_event_video")
    def test_renders_then_builds_report_with_explicit_event_json(
        self,
        render_video,
        build_report,
        write_report,
    ):
        render_video.return_value = {
            "output_video": "/tmp/annotated.mp4",
            "record_csv": "/tmp/overlay.csv",
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frame_json = root / "sample.json"
            event_json = root / "sample_swing_events.json"
            frame_json.write_text('{"frames": []}', encoding="utf-8")
            event_json.write_text('{"events": []}', encoding="utf-8")

            result = render_and_build_report(
                str(frame_json),
                event_json=str(event_json),
                output_video=str(root / "annotated.mp4"),
                output_html=str(root / "report.html"),
            )

        render_video.assert_called_once()
        build_report.assert_called_once()
        write_report.assert_called_once()
        self.assertEqual(result["event_json"], str(event_json))
        self.assertEqual(result["report_html"], str(root / "report.html"))


if __name__ == "__main__":
    unittest.main()
