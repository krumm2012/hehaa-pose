"""Report boundaries must preserve the qualification of kinematic evidence."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from realtime_swing_pipeline import RealtimeSwingOutputManager
from swing_report_builder import render_report_html, _build_kinematic_sequence_html


class KinematicSequenceDisplayTests(unittest.TestCase):
    def test_partial_measurement_does_not_invent_a_zero_interval(self):
        page = _build_kinematic_sequence_html({
            "sequence_quality": "OPTIMAL", "coach_eligible": False, "confidence": 0,
            "latency_hip_to_shoulder_ms": None, "latency_shoulder_to_racket_ms": 120,
        })
        self.assertIn("未观测", page)
        self.assertNotIn("0.0 ms", page.replace("120.0 ms", ""))
        self.assertNotIn("seq-optimal", page)

    def test_qualified_evidence_retains_its_explicit_classification(self):
        page = _build_kinematic_sequence_html({
            "value": "OPTIMAL", "coach_eligible": True, "confidence": .9,
            "details": {"latency_hip_to_shoulder_ms": 40, "latency_shoulder_to_racket_ms": 80},
        })
        self.assertIn("seq-optimal", page)
        self.assertIn("40.0 ms", page)
        self.assertNotIn("未验证 · 需复核", page)

    def test_both_reports_label_unvalidated_sequence_as_a_projection(self):
        sequence = {
            "sequence_quality": "DISCONNECTED",
            "latency_hip_to_shoulder_ms": -240.0,
            "latency_shoulder_to_racket_ms": 280.0,
            "coach_eligible": False,
            "confidence": 0.0,
        }
        event = {"event_id": 1, "stroke_type": "Forehand", "start_frame": 9,
                 "contact_frame": 17, "end_frame": 59, "kinematic_sequence": sequence}
        with TemporaryDirectory() as td:
            manager = RealtimeSwingOutputManager.__new__(RealtimeSwingOutputManager)
            manager.output_json = Path(td) / "events.json"
            manager.output_html = Path(td) / "report.html"
            manager.preview_path = None
            manager.roi_metadata = {}
            live = manager._render_live_html({"events": [event], "summary": {}})
            offline = render_report_html({"events": [event], "paths": {}, "summary": {},
                                          "video_info": {}, "timeline": {}}, str(manager.output_html))
        for page in [live, offline]:
            with self.subTest(page="live" if page is live else "offline"):
                self.assertNotIn('seq-badge seq-disconnected', page)
                self.assertIn("二维峰值间隔", page)
                self.assertIn("未验证", page)
                self.assertIn("不代表处理耗时", page)
                self.assertIn("-240.0 ms", page)
                self.assertIn("280.0 ms", page)

    def test_candidate_racket_peak_is_shown_in_html_report(self):
        page = _build_kinematic_sequence_html({
            "sequence_quality": "UNRESOLVED_AT_FRAME_RATE", "coach_eligible": False, "confidence": 0,
            "latency_hip_to_shoulder_ms": 0.0, "latency_shoulder_to_racket_ms": None,
            "racket_candidate_peak_frame": 25, "racket_candidate_peak_speed": 2304.2,
            "candidate_latency_shoulder_to_racket_ms": 278.4,
            "racket_evidence": {"status": "discontinuous_evidence"},
        })
        self.assertIn("候选拍峰（诊断参考）：第 25 帧", page)
        self.assertIn("+278.4 ms (候选F25 @ 2304px/s)", page)


if __name__ == "__main__":
    unittest.main()
