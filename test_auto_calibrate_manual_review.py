import json
import tempfile
import unittest
from pathlib import Path

from manual_review_workflow import (
    auto_calibrate_manual_review,
    discover_session_paths,
    validate_manual_annotations,
)


def _frame(fid: int, wrist_x: float, wrist_y: float, ball=None, racket=None, swing_type="Forehand"):
    rec = {
        "frame_id": fid,
        "timestamp": round(fid / 25.0, 3),
        "swing_type": swing_type,
        "ball": ball,
        "rackets": [{"box": racket, "confidence": 0.9}] if racket else [],
        "pose": {
            "right_shoulder": [300, 200],
            "left_shoulder": [240, 200],
            "right_elbow": [320, 250],
            "right_wrist": [wrist_x, wrist_y],
            "left_wrist": [200, 260],
            "right_hip": [295, 320],
            "left_hip": [245, 320],
        },
        "metrics": {},
    }
    if racket is not None:
        rec["racket"] = racket
    return rec


class AutoCalibrateManualReviewTests(unittest.TestCase):
    def _build_session(self, root: Path):
        frames = []
        # Event 1 (frames 10..40): genuine forehand with strong wrist motion + ball contact at frame 24
        for fid in range(0, 120):
            if 16 <= fid <= 30:
                wx = 300 + (fid - 16) * 22
                wy = 260 - abs(fid - 24) * 8
                racket = [wx - 20, wy - 30, wx + 40, wy + 30]
                ball = [wx + 12, wy + 2] if fid == 24 else ([wx + 90, wy - 40] if abs(fid - 24) <= 2 else None)
                frames.append(_frame(fid, wx, wy, ball=ball, racket=racket, swing_type="Forehand"))
            elif 60 <= fid <= 72:
                # Event 2 (frames 60..72): low-energy shadow swing / noise with no ball
                wx = 300 + (fid - 60) * 2
                frames.append(_frame(fid, wx, 260, ball=None, racket=None, swing_type="Unknown"))
            else:
                frames.append(_frame(fid, 300, 260, ball=None, racket=None, swing_type="Ready"))

        events_doc = {
            "schema_version": "tennis.swing-document.v1",
            "summary": {"thresholds": {"dominant_hand": "right"}},
            "events": [
                {
                    "event_id": 1,
                    "start_frame": 10,
                    "contact_frame": 28,
                    "peak_frame": 24,
                    "end_frame": 40,
                    "stroke_type": "Unknown",
                    "confidence": 0.85,
                    "peak_energy": 25.0,
                    "quality_flags": {"warnings": []},
                },
                {
                    "event_id": 2,
                    "start_frame": 60,
                    "contact_frame": 66,
                    "peak_frame": 66,
                    "end_frame": 72,
                    "stroke_type": "Unknown",
                    "confidence": 0.4,
                    "peak_energy": 3.0,
                    "quality_flags": {"warnings": []},
                },
            ],
        }
        (root / "demo_swing_events.json").write_text(json.dumps(events_doc), encoding="utf-8")
        (root / "demo_frames.jsonl").write_text(
            "\n".join(json.dumps(r) for r in frames) + "\n", encoding="utf-8"
        )
        (root / "demo_swing_report.html").write_text("<html></html>", encoding="utf-8")
        return discover_session_paths(root), events_doc, frames

    def test_auto_calibrate_refines_contact_type_filters_shadow_and_passes_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, events_doc, frames = self._build_session(Path(tmp))
            result = auto_calibrate_manual_review(paths, {})
            summary = result["summary"]
            ann = result["annotations"]

            self.assertEqual(summary["total_events"], 2)
            self.assertEqual(summary["type_corrected_count"], 1)
            self.assertEqual(summary["filtered_invalid_count"], 1)

            ev1 = next(e for e in ann["events"] if e["source_event_id"] == 1)
            ev2 = next(e for e in ann["events"] if e["source_event_id"] == 2)

            self.assertEqual(ev1["actual_stroke_type"], "Forehand")
            self.assertEqual(ev1["frames"]["contact"], 24)
            self.assertTrue(ev1["valid_hit"])
            self.assertFalse(ev1["needs_review"])

            self.assertFalse(ev2["valid_hit"])
            self.assertFalse(ev2["count_correct"])
            self.assertFalse(ev2["needs_review"])

            # Must pass strict manual annotation validation
            validation = validate_manual_annotations(events_doc, ann, frames, event_path=paths["events"])
            self.assertTrue(validation["metrics_finalizable"])

    def test_auto_calibrate_preserves_manually_edited_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, _, _ = self._build_session(Path(tmp))
            custom_ev1 = {
                "annotation_id": "model-1",
                "source_event_id": 1,
                "predicted_stroke_type": "Unknown",
                "actual_stroke_type": "Serve",
                "count_correct": True,
                "valid_hit": True,
                "needs_review": False,
                "issue_tags": [],
                "note": "人工手改备注",
                "frames": {"start": 12, "contact": 22, "peak": 24, "end": 35},
                "contact_review": "confirmed",
                "coach_review": {"reviewer": "coach", "verdict": "supported", "observed_fact": "", "suggestion": "", "evidence": ""},
                "practice_review": {"confirmed": False, "ratings": {"preparation": None, "positioning": None, "contact": None, "coordination": None, "recovery": None}},
            }
            result = auto_calibrate_manual_review(
                paths,
                {
                    "current_annotations": {"schema_version": "swing_manual_annotations_v2", "events": [custom_ev1]},
                    "edited_event_ids": [1],
                },
            )
            self.assertEqual(result["summary"]["preserved_manual_count"], 1)
            ev1 = next(e for e in result["annotations"]["events"] if e["source_event_id"] == 1)
            self.assertEqual(ev1["actual_stroke_type"], "Serve")
            self.assertEqual(ev1["note"], "人工手改备注")
            self.assertEqual(result["item_diffs"]["1"]["status"], "preserved_manual")


if __name__ == "__main__":
    unittest.main()
