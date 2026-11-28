import json
import os
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from manual_review_llm import ReviewOutcome
from manual_review_workflow import (
    auto_calibrate_manual_review,
    discover_session_paths,
    make_handler,
    process_manual_review,
    validate_manual_annotations,
)


@contextmanager
def _run_server(handler):
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


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
            self.assertTrue(ev2["needs_review"])
            self.assertGreater(summary["boundary_adjusted_count"], 0)

            validation = validate_manual_annotations(events_doc, ann, frames, event_path=paths["events"])
            self.assertFalse(validation["metrics_finalizable"])
            state = process_manual_review(paths, ann)
            self.assertFalse(state["evaluation"]["summary"]["metrics_finalized"])

    def test_strong_swing_with_no_ball_stays_open_for_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, _, frames = self._build_session(Path(tmp))
            for frame in frames:
                frame["ball"] = None
            paths["frames"].write_text("".join(json.dumps(row) + "\n" for row in frames))
            result = auto_calibrate_manual_review(paths)
            event = result["item_diffs"]["1"]
            self.assertGreater(event["peak_energy"], 12)
            self.assertTrue(event["valid_hit"])
            self.assertTrue(event["needs_review"])

    def test_all_clear_rule_results_do_not_complete_human_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, events, _ = self._build_session(Path(tmp))
            events["events"] = events["events"][:1]
            paths["events"].write_text(json.dumps(events))
            result = auto_calibrate_manual_review(paths)
            self.assertEqual(result["summary"]["pending_review_count"], 0)
            self.assertFalse(result["annotations"]["timeline_review_complete"])
            state = process_manual_review(paths, result["annotations"])
            self.assertFalse(state["evaluation"]["summary"]["metrics_finalized"])

    def test_missing_anchors_stay_blank_and_are_not_sent_to_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, events, _ = self._build_session(Path(tmp))
            events["events"] = [events["events"][0]]
            events["events"][0].update(start_frame=None, contact_frame=None)
            paths["events"].write_text(json.dumps(events))
            result = auto_calibrate_manual_review(paths, {"use_llm": True})
            row = result["annotations"]["events"][0]
            self.assertIsNone(row["frames"]["start"])
            self.assertIsNone(row["frames"]["contact"])
            self.assertTrue(row["needs_review"])
            self.assertEqual(result["llm_review"]["status"], "no_candidates")

    def test_trim_preserves_supported_original_contact(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, events, frames = self._build_session(Path(tmp))
            events["events"][0].update(contact_frame=10, stroke_type="Forehand")
            frames[10] = _frame(10, 300, 260, ball=[384, 260], racket=[280, 230, 340, 290])
            frames[16]["ball"] = [310, 196]
            paths["events"].write_text(json.dumps(events))
            paths["frames"].write_text("".join(json.dumps(row) + "\n" for row in frames))

            result = auto_calibrate_manual_review(paths)
            row = result["annotations"]["events"][0]
            self.assertEqual(row["frames"]["contact"], 10)
            self.assertEqual(row["frames"]["start"], 10)
            self.assertLess(row["frames"]["end"], 40)
            self.assertFalse(row["needs_review"])

    def test_contact_blocked_by_manual_event_stays_blank_for_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, events, frames = self._build_session(Path(tmp))
            earlier = {**events["events"][0], "event_id": 99, "start_frame": 0, "contact_frame": 3,
                       "peak_frame": 3, "end_frame": 7}
            events["events"][0].update(contact_frame=10, stroke_type="Forehand")
            events["events"].insert(0, earlier)
            frames[10] = _frame(10, 300, 260, ball=[384, 260], racket=[280, 230, 340, 290])
            paths["events"].write_text(json.dumps(events))
            paths["frames"].write_text("".join(json.dumps(row) + "\n" for row in frames))
            protected = {"source_event_id": 99, "annotation_origin": "human",
                         "frames": {"start": 0, "contact": 3, "end": 12}}
            request = {"current_annotations": {"events": [protected]}, "use_llm": True}

            result = auto_calibrate_manual_review(paths, request)
            rows = {row["source_event_id"]: row for row in result["annotations"]["events"]}
            self.assertEqual(rows[99], protected)
            self.assertGreater(rows[1]["frames"]["start"], 12)
            self.assertIsNone(rows[1]["frames"]["contact"])
            self.assertTrue(rows[1]["needs_review"])
            self.assertIn("contact_timing", rows[1]["issue_tags"])
            self.assertIn("contact_anchor_unavailable", result["item_diffs"]["1"]["review_reasons"])
            self.assertEqual(result["llm_review"]["status"], "no_candidates")

    def test_missing_contact_observation_stays_blank_for_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, events, frames = self._build_session(Path(tmp))
            events["events"][0].update(contact_frame=24, stroke_type="Forehand")
            paths["events"].write_text(json.dumps(events))
            paths["frames"].write_text("".join(json.dumps(row) + "\n" for row in frames if row["frame_id"] != 24))

            result = auto_calibrate_manual_review(paths)
            row = result["annotations"]["events"][0]
            self.assertIsNone(row["frames"]["contact"])
            self.assertTrue(row["needs_review"])
            self.assertIn("contact_anchor_unavailable", result["item_diffs"]["1"]["review_reasons"])

    def test_contact_blocked_by_next_manual_event_stays_blank_for_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, events, _ = self._build_session(Path(tmp))
            events["events"][0].update(contact_frame=24, stroke_type="Forehand")
            paths["events"].write_text(json.dumps(events))
            protected = {"source_event_id": 2, "annotation_origin": "human",
                         "frames": {"start": 20, "contact": 66, "end": 72}}

            result = auto_calibrate_manual_review(paths, {"current_annotations": {"events": [protected]}})
            adjusted, preserved = result["annotations"]["events"]
            self.assertEqual(preserved, protected)
            self.assertLess(adjusted["frames"]["end"], 20)
            self.assertIsNone(adjusted["frames"]["contact"])
            self.assertTrue(adjusted["needs_review"])

    def test_llm_cannot_replace_supported_contact_in_low_confidence_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, events, _ = self._build_session(Path(tmp))
            events["events"][0].update(contact_frame=24, stroke_type="Forehand", confidence=0.45)
            paths["events"].write_text(json.dumps(events))

            with patch("manual_review_workflow.review_uncertain_events",
                       return_value=ReviewOutcome("no_candidates")) as review:
                result = auto_calibrate_manual_review(paths, {"use_llm": True})

            self.assertEqual(result["annotations"]["events"][0]["frames"]["contact"], 24)
            self.assertEqual(review.call_args.args[0], [])

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
                },
            )
            self.assertEqual(result["summary"]["preserved_manual_count"], 1)
            ev1 = next(e for e in result["annotations"]["events"] if e["source_event_id"] == 1)
            self.assertEqual(ev1["actual_stroke_type"], "Serve")
            self.assertEqual(ev1["note"], "人工手改备注")
            self.assertEqual(result["item_diffs"]["1"]["status"], "preserved_manual")
            self.assertFalse(result["annotations"]["timeline_review_complete"])

            paths["annotations"].write_text(json.dumps(result["annotations"]))
            prefill = {**custom_ev1, "annotation_origin": "model_prefill", "note": ""}
            rerun = auto_calibrate_manual_review(paths, {"current_annotations": {"events": [prefill]}})
            self.assertEqual(rerun["annotations"]["events"][0]["note"], "人工手改备注")

    def test_llm_draft_http_uses_only_candidates_and_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, events, _ = self._build_session(root)
            events["events"][0].update(event_id=41, stroke_type="Backhand", confidence=0.45)
            paths["events"].write_text(json.dumps(events))
            received = []
            provider_state = {"mode": "valid"}

            class Provider(BaseHTTPRequestHandler):
                def do_POST(self):
                    packet = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    received.append((self.path, packet))
                    mode = provider_state["mode"]
                    if mode == "offline":
                        self.send_error(503)
                        return
                    if mode == "stale":
                        paths["events"].write_text(json.dumps({**events, "updated": True}))
                    candidates = ["stroke:1", "contact:1"] if mode != "invalid" else ["contact:999"]
                    content = json.dumps({"reviews": [{"event_id": 1, "candidate_ids": candidates}]})
                    body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode()
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *_args):
                    pass

            with _run_server(Provider) as provider_url, patch.dict(os.environ, {"TENNIS_TEST_REVIEW_KEY": "test-key"}):
                config = root / "config.yaml"
                config.write_text(json.dumps({"manual_review": {"llm": {
                    "provider": "deepseek", "base_url": provider_url, "model": "test-review-model",
                    "api_key_env": "TENNIS_TEST_REVIEW_KEY",
                }}}))
                with _run_server(make_handler(paths, config)) as review_url:
                    def calibrate(use_llm, headers=None):
                        draft = {"events": [{"source_event_id": 41, "annotation_origin": "model_prefill",
                                              "note": "PRIVATE EDITOR NOTE"}]}
                        body = json.dumps({"current_annotations": draft, "use_llm": use_llm}).encode()
                        request = Request(review_url + "/api/manual-review/auto-calibrate", data=body,
                                          headers={"Content-Type": "application/json", **(headers or {})})
                        with urlopen(request, timeout=5) as response:
                            return json.load(response)

                    local = calibrate(False)
                    for headers in ({"Origin": "http://example.test"}, {"Host": "example.test"},
                                    {"Content-Type": "text/plain"}):
                        with self.assertRaises(HTTPError) as denied:
                            calibrate(True, headers)
                        self.assertEqual(denied.exception.code, 403)
                    self.assertEqual(received, [])
                    self.assertEqual(local["annotations"]["events"][0]["actual_stroke_type"], "Backhand")
                    suggested = calibrate(True)
                    self.assertEqual(suggested["llm_review"], {
                        "status": "reviewed", "model": "test-review-model", "candidate_count": 1, "applied_count": 1,
                    })
                    row = suggested["annotations"]["events"][0]
                    self.assertEqual(row["actual_stroke_type"], "Forehand")
                    self.assertNotEqual(row["frames"]["contact"], local["annotations"]["events"][0]["frames"]["contact"])
                    self.assertTrue(row["needs_review"])
                    self.assertEqual(row["contact_review"], "unreviewed")
                    self.assertEqual(row["llm_suggestion"]["model"], "test-review-model")
                    self.assertFalse(suggested["annotations"]["timeline_review_complete"])
                    self.assertFalse(paths["annotations"].exists())
                    self.assertEqual(received[0][0], "/chat/completions")
                    outbound = received[0][1]["messages"][-1]["content"]
                    self.assertNotIn("PRIVATE EDITOR NOTE", outbound)
                    self.assertNotIn("demo_swing", outbound)
                    self.assertEqual([case["event_id"] for case in json.loads(outbound)["cases"]], [1])

                    for mode, status in (("invalid", "invalid_response"), ("offline", "unavailable")):
                        with self.subTest(mode=mode):
                            provider_state["mode"] = mode
                            fallback = calibrate(True)
                            self.assertEqual(fallback["llm_review"]["status"], status)
                            self.assertEqual(fallback["annotations"], local["annotations"])

                    provider_state["mode"] = "stale"
                    stale = calibrate(True)
                    self.assertEqual(stale["llm_review"]["status"], "stale")
                    self.assertNotIn("annotations", stale)


if __name__ == "__main__":
    unittest.main()
