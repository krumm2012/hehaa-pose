import copy
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from practice_scoring import POLICY_VERSION, POLICY_SHA256, DIMENSIONS, score_event, score_review, grade, attach_score
from swing_biomechanics import aggregate_event_biomechanics, _calculate_extended_tier_biomechanics
from realtime_swing_runtime import build_impact_telemetry_card
from swing_report_builder import build_report_payload
from swing_report_builder import _session_dashboard_html
from practice_score_adapter import resolve_practice_score
from swing_session_summary import build_session_coaching_summary, calculate_radar_dimensions
from swing_session_quality import build_session_quality_dashboard


def evidence():
    return {"event_id": 1, "stroke_type": "Forehand", "start_frame": 0, "end_frame": 20,
            "biomechanics": {"metrics": {
                name: {"value": value, "unit": "deg_2d", "coach_eligible": True,
                       "confidence": .8, "source_frames": [5]}
                for name, value in [("arm_extension", 145), ("shoulder_turn_change", 30),
                                    ("preparation_knee_flexion", 25)]}}}


class PracticeScoringTests(unittest.TestCase):
    def test_missing_pose_and_extension_metrics_never_get_default_points(self):
        event={"start_frame": 0, "end_frame": 1, "contact_frame": 1, "quality_flags": {"pose_frame_ratio": 1}}
        bio=aggregate_event_biomechanics(event, [], [{"frame_id": 0, "has_pose": True}, {"frame_id": 1, "has_pose": True}])
        self.assertIsNone(bio["swing_score"])
        self.assertIsNone(bio["extended_biomechanics"]["racket_head_speed"]["contact_kmh"])
        self.assertIsNone(bio["extended_biomechanics"]["kinematic_sequence"]["sequence_quality"])
        self.assertIsNone(build_impact_telemetry_card({})["swing_score"])

    def test_zero_is_measured_not_replaced_with_default(self):
        for dy in [0, 5]:
            ext=_calculate_extended_tier_biomechanics([
                {"frame_id": 0, "hip_vertical_pos": 100}, {"frame_id": 1, "hip_vertical_pos": 100-dy}], 0, 1, 1, 100)
            self.assertEqual(ext["leg_drive"]["drive_ratio"], dy/100)
            self.assertIsNone(ext["swing_quality_score"]["overall_score"])
        e=evidence()
        for m in e["biomechanics"]["metrics"].values(): m["value"]=0
        self.assertIsNone(score_event(e)["score"])
        e["biomechanics"]["metrics"]={}
        self.assertIsNone(score_event(e)["score"])

    def test_missing_eligibility_and_3d_units_are_excluded(self):
        for mode in ["missing_flag", "3d"]:
            e=evidence()
            for m in e["biomechanics"]["metrics"].values():
                if mode=="3d": m["unit"]="deg_3d"
                else: m.pop("coach_eligible")
            self.assertIsNone(score_event(e)["score"])

    def test_nonfinite_boolean_and_low_confidence_rejected(self):
        for value in [math.nan, math.inf, True]:
            e=evidence()
            for m in e["biomechanics"]["metrics"].values(): m["value"]=value
            self.assertIsNone(score_event(e)["score"])
        e=evidence()
        for m in e["biomechanics"]["metrics"].values(): m["confidence"]=.1
        self.assertIsNone(score_event(e)["score"])

    def test_shadow_excluded_even_with_full_evidence(self):
        e=evidence();e["is_shadow_swing"]=True
        self.assertIsNone(score_event(e)["score"])
        e=evidence();e["evidence"]={"contact_analysis":{"has_ball":False}}
        self.assertIsNone(score_event(e)["score"])

    def test_realtime_offline_report_share_identical_score(self):
        e=evidence(); canonical=attach_score(e)
        self.assertEqual(e["swing_score"], canonical["score"])
        self.assertEqual(e["biomechanics"]["extended_biomechanics"]["swing_quality_score"]["overall_score"], canonical["score"])
        self.assertEqual(build_impact_telemetry_card(e)["practice_score"], canonical)
        with TemporaryDirectory() as td:
            root=Path(td);(root/"frames.json").write_text(json.dumps({"video_info":{"fps":25},"frames":[]}))
            (root/"events.json").write_text(json.dumps({"events":[e]}))
            report=build_report_payload(str(root/"frames.json"),str(root/"events.json"))
        self.assertEqual(report["events"][0]["practice_score"], canonical)

    def test_old_score_never_rescaled_or_promoted(self):
        for old in [0, .8, 85, 99]:
            e={"swing_score": old,"swing_grade":"PRO","coach_calibration":{"visible_technique_score":.99}}
            self.assertIsNone(score_event(e)["score"])

    def test_grade_boundaries_are_shared_and_have_no_ntrp(self):
        for x,key in [(0,"FOCUS_REQUIRED"),(39.9,"FOCUS_REQUIRED"),(40,"NEEDS_PRACTICE"),(60,"MOSTLY_COMPLETE"),(80,"TARGET_MET"),(100,"TARGET_MET")]:
            self.assertEqual(grade(x),key)
        for x in [-1, 101, math.nan, None]: self.assertIsNone(grade(x))

    def test_manual_review_requires_all_five_and_confirmation(self):
        review={"ratings":{k:4 for k in DIMENSIONS},"confirmed":False}
        self.assertIsNone(score_review(review)["score"])
        review["confirmed"]=True;self.assertEqual(score_review(review)["score"],80)
        review["ratings"]["contact"]=None;self.assertIsNone(score_review(review)["score"])
        self.assertEqual(score_review(review)["coverage"],.8)
        for bad in [0, 6, True, 2.5, "4"]:
            review["ratings"]["contact"]=bad
            with self.assertRaises(ValueError): score_review(review)

    def test_unknown_radar_and_insufficient_samples(self):
        self.assertTrue(all(v is None for v in calculate_radar_dimensions({}).values()))
        summary=build_session_coaching_summary([evidence()])
        self.assertEqual(summary["quality_metrics"]["stability_rating"],"INSUFFICIENT_EVIDENCE")
        self.assertTrue(all(v is None for v in summary["radar_averages"].values()))

    def test_different_strokes_and_machine_conditions_do_not_mix(self):
        for field,value in [("stroke_type","Backhand"),("practice_context",{"machine":{"spin":"topspin"}})]:
            first=evidence()
            first["practice_review"] = {"ratings": {key: 4 for key in DIMENSIONS}, "confirmed": True}
            second=copy.deepcopy(first);second["event_id"]=2;second[field]=value
            summary=build_session_coaching_summary([first,second])
            self.assertIsNone(summary["quality_metrics"]["average_score"])
            self.assertEqual(len(summary["score_series"]),2)
        rows=[evidence() for _ in range(6)]
        for i,row in enumerate(rows):row.update(event_id=i+1,start_frame=i*30,end_frame=i*30+20)
        rows[-1]["stroke_type"]="Backhand"
        self.assertFalse(build_session_quality_dashboard(rows)["drift"]["ready"])

    def test_vendored_viewer_policy_and_engine_match(self):
        other=Path(__file__).resolve().parent.parent/"Tennis-Vision/viewer/video_import"
        if not other.is_dir():
            self.skipTest("Cross-project parity requires both local checkouts")
        for name in ["practice_policy.json","practice_scoring.py"]:
            self.assertEqual(Path(__file__).with_name(name).read_bytes(),(other/name).read_bytes())

    def test_saved_manual_score_is_shared_without_embedded_review(self):
        e = evidence()
        e["practice_review"] = {"ratings": {key: 5 for key in DIMENSIONS}, "confirmed": True}
        saved = attach_score(e)
        e.pop("practice_review")
        self.assertEqual(resolve_practice_score(e), saved)
        self.assertEqual(build_impact_telemetry_card(e)["practice_score"], saved)
        summary = build_session_coaching_summary([e])
        dashboard = build_session_quality_dashboard([e])
        self.assertEqual(summary["quality_metrics"]["average_score"], 100)
        self.assertEqual(dashboard["quality"]["practice_score_mean_100"], 100)
        self.assertEqual(dashboard["series"][0]["scoring_method"], "coach_manual")
        self.assertIsNone(dashboard["quality"]["median_uncertainty_100"])
        self.assertEqual(summary["radar_averages"], {key: 100 for key in DIMENSIONS})
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "frames.json").write_text(json.dumps({"frames": []}))
            (root / "events.json").write_text(json.dumps({"events": [e]}))
            report = build_report_payload(str(root / "frames.json"), str(root / "events.json"))
        self.assertEqual(report["events"][0]["practice_score"], saved)
        self.assertEqual(report["session_quality"]["quality"]["practice_score_mean_100"], 100)

    def test_mixed_manual_series_do_not_average_radar(self):
        first = evidence()
        second = copy.deepcopy(first)
        second.update(event_id=2, stroke_type="Backhand")
        for e, rating in [(first, 1), (second, 5)]:
            e["practice_review"] = {"ratings": {key: rating for key in DIMENSIONS}, "confirmed": True}
        summary = build_session_coaching_summary([first, second])
        self.assertIsNone(summary["quality_metrics"]["average_score"])
        self.assertTrue(all(v is None for v in summary["radar_averages"].values()))

    def test_dashboard_uncertainty_is_on_same_scale_as_score(self):
        dashboard = build_session_quality_dashboard([evidence()])
        self.assertIsNone(dashboard['quality']['median_uncertainty_9'])
        self.assertIsNone(dashboard['quality']['median_uncertainty_100'])
        self.assertNotIn('启发式范围 ±', _session_dashboard_html(dashboard))

    def test_manual_trend_uses_manual_ratings_instead_of_automatic_evidence(self):
        rows = []
        for i in range(6):
            e = evidence()
            e.update(event_id=i+1, start_frame=i*30, end_frame=i*30+20)
            e["practice_review"] = {"ratings": {key: 5 if i < 3 else 1 for key in DIMENSIONS},
                                    "confirmed": True}
            attach_score(e)
            rows.append(e)
        dashboard = build_session_quality_dashboard(rows)
        indicator = next(row for row in dashboard["drift"]["indicators"]
                         if row["name"] == "visible_technique_score")
        self.assertEqual(indicator["status"], "declining")
        self.assertEqual(indicator["baseline"], 100)
        self.assertEqual(indicator["recent"], 20)
