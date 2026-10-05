"""Independent-view evidence for 2D peak timing, not validated 3D kinetics."""

import math
from statistics import mean, median
from typing import Dict, List, Optional, Tuple

from practice_scoring import number
from observation_policy import finite_number
from collections import Counter

POLICY_VERSION = "kinematic_cross_view_2d_v8_cadence_stability"
JOINTS = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
MIN_CONFIDENCE = .5
MIN_LINE_SPAN_PX = 12.0
MIN_SAMPLES = 6
MIN_COVERAGE = .6


def serialize_kinematic_views(pose_result, all_joints=False) -> Dict:
    """Persist raw view coordinates and confidence without copying fused joints."""
    return {
        view: {
            name: {"x": float(kp.x), "y": float(kp.y), "confidence": float(kp.conf),
                   "observed": getattr(kp, "observed", True),
                   "source_frame_id": getattr(kp, "source_frame_id", None),
                   "confidence_source": getattr(kp, "confidence_source", "provided"),
                   "recovered_from_mirror": getattr(kp, "recovered_from_mirror", False)}
            for name, kp in (getattr(pose_result, f"{view}_pose_orig", {}) or {}).items()
            if all_joints or name in JOINTS
        }
        for view in ("front", "back")
    }


def _source_frame_matches(observation: Dict, frame_id: Optional[int]) -> bool:
    """Legacy missing IDs stay unverified; explicit conflicting IDs are invalid."""
    source = observation.get("source_frame_id")
    return source is None or (type(source) is int and source == frame_id)


def _angle(pose: Dict, segment: str, frame_id: Optional[int] = None) -> Optional[Tuple[float, float]]:
    """Reject unqualified or collapsed projected segment lines."""
    points = []
    for side in ("left", "right"):
        kp = pose.get(f"{side}_{segment}")
        if not isinstance(kp, dict):
            return None
        if frame_id is not None and not _source_frame_matches(kp, frame_id):
            return None
        if kp.get("observed") is False or kp.get("recovered_from_mirror") or kp.get("confidence_source") == "unavailable":
            return None
        x, y, confidence = (finite_number(kp.get(key)) for key in ("x", "y", "confidence"))
        if x is None or y is None or confidence is None or not MIN_CONFIDENCE <= confidence <= 1:
            return None
        points.append((x, y, min(1.0, confidence)))
    left, right = points
    dx, dy = right[0] - left[0], right[1] - left[1]
    if not math.isfinite(dx) or not math.isfinite(dy) or math.hypot(dx, dy) < MIN_LINE_SPAN_PX:
        return None
    return math.degrees(math.atan2(dy, dx)), min(left[2], right[2])


def _sample_runs(samples):
    runs = []
    deltas = [b['time']-a['time'] for a,b in zip(samples,samples[1:])
              if b['time']>a['time']]
    cadence = median(deltas) if deltas else 0
    for sample in samples:
        if (not runs or sample.get('segment_id',0) != runs[-1][-1].get('segment_id',0)
                or not 0 < sample['time']-runs[-1][-1]['time'] <= 3*cadence):
            runs.append([])
        runs[-1].append(sample)
    return runs


def _smooth(samples):
    return [median(s['speed'] for s in samples[max(0,i-1):i+2])
            for i in range(len(samples))]


def _peak(samples: List[Dict], minimum_speed: float) -> Dict:
    runs = _sample_runs(samples)
    if len(runs) <= 1:
        return _continuous_peak(samples, minimum_speed)
    candidates = [_continuous_peak(run, minimum_speed) for run in runs]
    usable = [item for item in candidates if item['peak'] is not None]
    # Never choose between separate motion episodes by borrowing filter samples.
    if len(usable) != 1:
        return {'status': 'discontinuous_evidence', 'peak': None}
    return {**usable[0], 'continuous_segment_count': len(runs)}


def _continuous_peak(samples: List[Dict], minimum_speed: float) -> Dict:
    """Use a symmetric median and reject flat, broad, or boundary maxima."""
    if len(samples) < MIN_SAMPLES:
        return {"status": "insufficient_samples", "peak": None}
    smoothed = _smooth(samples)
    maximum = max(smoothed)
    prominence = (maximum - median(smoothed)) / maximum if maximum > 0 else 0
    if maximum < minimum_speed or prominence < .2:
        return {"status": "insufficient_motion", "peak": None}
    near_peak = [i for i, speed in enumerate(smoothed) if speed >= maximum * .9]
    # Treat the whole peak plateau as a timing interval, rather than pretending
    # a single frame gives exact timing. Multiple distant maxima are ambiguous.
    if 0 in near_peak or len(samples)-1 in near_peak:
        return {"status": "boundary_peak", "peak": None}
    width = samples[near_peak[-1]]["time"] - samples[near_peak[0]]["time"]
    if width > .16 + 1e-9:
        return {"status": "ambiguous_peak", "peak": None}
    tied = [i for i in near_peak if maximum-smoothed[i] <= max(1e-6, maximum*1e-6)]
    selected = tied[len(tied)//2]
    row = samples[selected]
    return {
        "status": "usable", "prominence": round(prominence, 4),
        "peak": {"frame_id": row["frame_id"], "time": row["time"],
                 "speed": round(maximum, 4),
                 "time_range": [samples[near_peak[0]]["time"], samples[near_peak[-1]]["time"]]},
    }


def _angle_rejection(pose, segment, frame_id):
    for side in ('left', 'right'):
        point = pose.get(f'{side}_{segment}')
        if not isinstance(point, dict):
            return 'joint_missing'
        if not _source_frame_matches(point, frame_id):
            return 'joint_source_frame_mismatch'
        if point.get('observed') is False or point.get('recovered_from_mirror'):
            return 'joint_not_raw_observation'
        if point.get('confidence_source') == 'unavailable':
            return 'joint_score_unavailable'
        values = [finite_number(point.get(k)) for k in ('x', 'y', 'confidence')]
        if None in values or not 0 <= values[2] <= 1:
            return 'invalid_joint_numeric'
        if values[2] < MIN_CONFIDENCE:
            return 'low_joint_score'
    return 'projected_line_too_short'


def _audit_peak(evidence, alternative, cadence):
    """Sensitivity only: the alternative never substitutes for an observed peak."""
    raw, check = evidence.get('peak'), alternative.get('peak')
    stable = (raw is not None and check is not None
              and abs(raw['time'] - check['time']) <= cadence + 1e-9
              and abs(raw['speed'] - check['speed']) <= .25 * max(raw['speed'], 1e-9))
    evidence['cadence_sensitivity'] = {
        'method': 'exclude_intervals_below_quarter_window_median',
        'accuracy_validated': False, 'stable': stable if raw else None,
        'alternative_status': alternative['status'], 'alternative_peak': check,
        'time_tolerance_seconds': cadence, 'relative_speed_tolerance': .25}
    if raw is not None and not stable:
        evidence.update(candidate_peak=raw, peak=None, status='cadence_sensitive_peak')


def _view_evidence(rows: List[Dict], times: List[float], view: str, cadence: float, minimum_dt=0.) -> Dict:
    """Differentiate each raw view independently, resetting across invalid samples."""
    result = {"status": "unavailable", "segments": {}, "quality": 0.0}
    for segment in ("hip", "shoulder"):
        previous = None
        samples = []
        confidences = []
        segment_id = 0
        rejected = Counter()
        rejected_frames = []
        valid_frames = 0
        for row, timestamp in zip(rows, times):
            pose = ((row.get("kinematic_views") or {}).get(view) or {})
            observation = _angle(pose, segment, int(row["frame_id"]))
            if observation is None:
                reason = _angle_rejection(pose, segment, int(row['frame_id']))
                rejected[reason] += 1
                rejected_frames.append({'source_frame_id': row['frame_id'], 'reason': reason})
                previous = None
                continue
            valid_frames += 1
            if previous is None:
                segment_id += 1
            angle, confidence = observation
            if previous is not None:
                old_angle, old_time, old_confidence = previous
                dt = timestamp - old_time
                if minimum_dt < dt <= 3 * cadence:
                    difference = abs((angle - old_angle + 180) % 360 - 180)
                    samples.append({"frame_id": int(row["frame_id"]), "time": timestamp,
                                    "speed": difference / dt, "segment_id": segment_id})
                    confidences.append(min(confidence, old_confidence))
                else:
                    segment_id += 1
            previous = angle, timestamp, confidence
        evidence = _peak(samples, minimum_speed=12)
        coverage = len(samples) / max(1, len(rows)-1)
        evidence.update(coverage=round(coverage, 4),
                        keypoint_quality=round(mean(confidences), 4) if confidences else 0,
                        observation_valid_frames=valid_frames,
                        observation_total_frames=len(rows), observation_rejections=dict(rejected),
                        observation_rejected_frames=rejected_frames)
        evidence["signal"] = [
            {"frame_id": sample["frame_id"], "time": sample["time"],
             "speed": round(speed, 4)}
            for run in _sample_runs(samples)
            for sample, speed in zip(run, _smooth(run))
        ]
        if coverage < MIN_COVERAGE:
            evidence.update(status="low_coverage", peak=None)
        result["segments"][segment] = evidence
    if all(s["peak"] is not None for s in result["segments"].values()):
        result["status"] = "usable"
        result["quality"] = min(s["coverage"] * s["keypoint_quality"]
                                for s in result["segments"].values())
        result["latency_hip_to_shoulder_ms"] = round(
            (result["segments"]["shoulder"]["peak"]["time"]
             - result["segments"]["hip"]["peak"]["time"]) * 1000, 1)
    return result


def _signal_agreement(front: Dict, back: Dict) -> Optional[float]:
    """Compare shapes at common frame IDs without shifting one view to force a match."""
    a = {row["frame_id"]: row["speed"] for row in front.get("signal", [])}
    b = {row["frame_id"]: row["speed"] for row in back.get("signal", [])}
    common = sorted(set(a) & set(b))
    if len(common) < MIN_SAMPLES:
        return None
    left, right = [a[key] for key in common], [b[key] for key in common]
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x-left_mean)*(y-right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x-left_mean)**2 for x in left)
                            * sum((y-right_mean)**2 for y in right))
    return max(-1.0, min(1.0, numerator / denominator)) if denominator > 1e-9 else None


def _racket_evidence(rows: List[Dict], times: List[float], cadence: float, minimum_dt=0.) -> Dict:
    """Measure detection-box centres; changing box size alone has no velocity."""
    previous = None
    samples = []
    segment_id = 0
    for row, timestamp in zip(rows, times):
        detections = row.get("rackets") or []
        selected = detections[0] if isinstance(detections, (list, tuple)) and detections and isinstance(detections[0], dict) else {}
        box = selected.get("box") or row.get("racket")
        confidence = finite_number(selected.get("confidence"))
        values = [finite_number(value) for value in box] if isinstance(box, (list, tuple)) else []
        if (not _source_frame_matches(selected, int(row["frame_id"]))
                or selected.get("observed") is False or len(values) != 4
                or None in values or confidence is None or not MIN_CONFIDENCE <= confidence <= 1
                or values[2] <= values[0] or values[3] <= values[1]):
            previous = None
            continue
        if previous is None:
            segment_id += 1
        centre = (values[0]/2+values[2]/2, values[1]/2+values[3]/2)
        if previous is not None:
            old_centre, old_time = previous
            dt = timestamp - old_time
            if minimum_dt < dt <= 3 * cadence:
                speed = math.dist(centre, old_centre) / dt
                if not math.isfinite(speed):
                    previous = None
                    continue
                samples.append({"frame_id": int(row["frame_id"]), "time": timestamp,
                                "speed": speed, "segment_id": segment_id})
            else:
                segment_id += 1
        previous = centre, timestamp
    result = _peak(samples, minimum_speed=50)
    if len(samples) / max(1, len(rows)-1) < MIN_COVERAGE:
        result.update(status="low_coverage", peak=None)
    return result


def _pair_interval(first, second, cadence):
    """Conservative lag interval; sampling allowance is not statistical CI."""
    low = second[0] - first[1] - cadence
    high = second[1] - first[0] + cadence
    return {'latency_range_ms': [round(low*1000, 2), round(high*1000, 2)],
            'resolved': low > 0 or high < 0,
            'order': 'forward' if low > 0 else 'reverse' if high < 0 else 'unresolved'}


def analyze_kinematic_sequence(frames: List[Dict], contact_frame: int, fps: float) -> Dict:
    """Cross-check per-view timing; disagreement abstains from a combined estimate."""
    result = {
        "policy_version": POLICY_VERSION, "scope": "image_plane_peak_timing_only",
        "validation_status": "unvalidated_2d_projection", "coach_eligible": False,
        "confidence": 0.0, "evidence_confidence": 0.0, "source_views": [],
        "hip_peak_frame": None, "shoulder_peak_frame": None, "racket_peak_frame": None,
        "latency_hip_to_shoulder_ms": None, "latency_shoulder_to_racket_ms": None,
        "is_sequential": None, "sequence_quality": None,
        "cross_validation": {"status": "unavailable"}, "views": {},
        "limitations": ["heuristic_evidence_quality_not_accuracy_probability",
                        "mirror_views_share_errors", "not_3d_axial_rotation_or_energy_transfer"],
        "parameters": {"min_keypoint_confidence": MIN_CONFIDENCE,
                       "min_line_span_px": MIN_LINE_SPAN_PX, "min_samples": MIN_SAMPLES,
                       "min_coverage": MIN_COVERAGE, "window_seconds": [-.6, .16],
                       "min_curve_correlation": .5, "min_peak_prominence": .2},
    }
    ordered = sorted(frames, key=lambda row: int(row["frame_id"]))
    if len(ordered) < MIN_SAMPLES or not any(row.get("kinematic_views") for row in ordered):
        result["cross_validation"]["reason"] = "independent_view_records_missing"
        return result
    # Reader receipt/processing wall clocks are latency instrumentation only.
    # Never substitute them for the media timeline, even on legacy journals.
    sources = [row.get("source_time") for row in ordered]
    if any(source is not None for source in sources):
        from analysis_data_contracts import SOURCE_TIME_SCHEMA_VERSION
        if not all(isinstance(source, dict) and source.get("schema_version") == SOURCE_TIME_SCHEMA_VERSION
                   and source.get("source_frame_id") == row["frame_id"]
                   for row, source in zip(ordered, sources)):
            result["cross_validation"]["reason"] = "incomplete_source_time_contract"
            return result
        bases = {source.get("basis") for source in sources}
        if len(bases) != 1:
            result["cross_validation"]["reason"] = "mixed_source_time_bases"
            return result
        time_basis = sources[0].get("basis")
        result["time_basis"] = time_basis
        if time_basis not in ("media_pts", "nominal_fps"):
            result["cross_validation"]["reason"] = "source_media_time_unavailable"
            return result
        allowed_quality = "reported" if time_basis == "media_pts" else "estimated"
        if any(source.get("quality") != allowed_quality for source in sources):
            result["cross_validation"]["reason"] = "duplicate_or_discontinuous_source_time"
            return result
        times = [number(source.get("timestamp_seconds")) for source in sources]
        result["time_quality"] = allowed_quality
        result["limitations"].append("media_pts_not_verified_sensor_exposure")
        if time_basis == "nominal_fps":
            result["limitations"].append("nominal_fps_assumes_uniform_source_cadence")
    else:
        times = [number(row.get("timestamp")) for row in ordered]
        time_basis = "legacy_frame_timestamps"
        result["time_quality"] = "legacy_unverified"
        result["limitations"].append("legacy_source_time_provenance_missing")
    if any(t is None or t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        result["cross_validation"]["reason"] = "invalid_source_timestamps"
        return result
    origin = times[0]
    times = [t-origin for t in times]
    contact_index = min(range(len(ordered)), key=lambda i: abs(int(ordered[i]["frame_id"])-contact_frame))
    contact_time = times[contact_index]
    selected = [(row, t) for row, t in zip(ordered, times) if contact_time-.6 <= t <= contact_time+.16]
    rows, times = [pair[0] for pair in selected], [pair[1] for pair in selected]
    if len(rows) < MIN_SAMPLES:
        result["cross_validation"]["reason"] = "window_too_short"
        return result
    cadence = median(b-a for a, b in zip(times, times[1:]))
    short_frames = [row['frame_id'] for row, a, b in zip(rows[1:], times, times[1:])
                    if b-a <= .25 * cadence]
    result['cadence_audit'] = {
        'policy': 'window_median_quarter_interval_sensitivity_v1',
        'short_interval_source_frames': short_frames,
        'min_interval_ms': round(min(b-a for a,b in zip(times,times[1:]))*1000, 3),
        'median_interval_ms': round(cadence*1000, 3),
        'short_interval_threshold_ms': round(.25*cadence*1000, 3),
        'accuracy_validated': False, 'original_timestamps_preserved': True,
        'meaning': 'heuristic_sensitivity_check_not_invalid_pts_or_exposure_proof'}
    tolerance = max(.04, 2 * cadence)
    result.update(time_basis=time_basis, sampling_interval_ms=round(cadence*1000, 2),
                  peak_time_uncertainty_ms=round(cadence*1000, 2))
    result["views"] = {view: _view_evidence(rows, times, view, cadence) for view in ("front", "back")}
    if short_frames:
        for view, evidence in result['views'].items():
            alternative = _view_evidence(rows, times, view, cadence, .25*cadence)
            for segment in ('hip', 'shoulder'):
                _audit_peak(evidence['segments'][segment], alternative['segments'][segment], cadence)
            if any(s['status'] == 'cadence_sensitive_peak' for s in evidence['segments'].values()):
                evidence.update(status='unavailable', quality=0.)
                evidence['candidate_latency_hip_to_shoulder_ms'] = evidence.pop('latency_hip_to_shoulder_ms', None)
    racket = _racket_evidence(rows, times, cadence)
    if short_frames:
        _audit_peak(racket, _racket_evidence(rows, times, cadence, .25*cadence), cadence)
    result['racket_evidence'] = racket
    usable = [view for view, evidence in result["views"].items() if evidence["status"] == "usable"]
    result["source_views"] = usable
    result["cross_validation"]["tolerance_ms"] = round(tolerance * 1000, 2)
    if cadence > .08:
        result["cross_validation"].update(status="unavailable", reason="sampling_too_sparse")
        return result
    if not usable:
        sensitive = any(s['status'] == 'cadence_sensitive_peak'
                        for v in result['views'].values() for s in v['segments'].values())
        result["cross_validation"]["reason"] = "cadence_sensitive_peak" if sensitive else "insufficient_view_evidence"
        return result
    if len(usable) == 2:
        deltas = {
            segment: abs(result["views"]["front"]["segments"][segment]["peak"]["time"]
                         - result["views"]["back"]["segments"][segment]["peak"]["time"])
            for segment in ("hip", "shoulder")
        }
        result["cross_validation"]["peak_deltas_ms"] = {k: round(v*1000, 2) for k, v in deltas.items()}
        if any(delta > tolerance + 1e-9 for delta in deltas.values()):
            result["cross_validation"].update(status="disagree", reason="view_peak_times_conflict")
            return result
        correlations = {
            segment: _signal_agreement(result["views"]["front"]["segments"][segment],
                                       result["views"]["back"]["segments"][segment])
            for segment in ("hip", "shoulder")
        }
        result["cross_validation"]["curve_correlations"] = {
            key: round(value, 4) if value is not None else None
            for key, value in correlations.items()
        }
        if any(value is None or value < .5 for value in correlations.values()):
            result["cross_validation"].update(status="disagree", reason="view_motion_curves_conflict")
            return result
        result["cross_validation"]["status"] = "agree"
        agreement = max(.5, 1-max(deltas.values())/(2*tolerance)) * min(correlations.values())
        evidence_quality = min(result["views"][v]["quality"] for v in usable) * agreement
        result["cross_validation"]["timing_agreement"] = round(agreement, 4)
    else:
        result["cross_validation"]["status"] = "single_view"
        evidence_quality = min(.45, result["views"][usable[0]]["quality"])
    peak_times = {}
    peak_ranges = {}
    for segment in ("hip", "shoulder"):
        ranges = [result['views'][view]['segments'][segment]['peak']['time_range'] for view in usable]
        peak_ranges[segment] = [min(r[0] for r in ranges), max(r[1] for r in ranges)]
        weights = [result["views"][view]["quality"] for view in usable]
        peak_times[segment] = sum(result["views"][view]["segments"][segment]["peak"]["time"] * weight
                                  for view, weight in zip(usable, weights)) / sum(weights)
        closest = min(range(len(rows)), key=lambda i: abs(times[i]-peak_times[segment]))
        result[f"{segment}_peak_frame"] = int(rows[closest]["frame_id"])
    hip_sh = peak_times["shoulder"]-peak_times["hip"]
    result["latency_hip_to_shoulder_ms"] = round(hip_sh * 1000, 1)
    time_cap = .45 if time_basis == "nominal_fps" else .9
    result["evidence_confidence"] = round(min(time_cap, evidence_quality), 4)
    # This is sampling uncertainty only, not a validated statistical interval.
    uncertainty = cadence + max(
        evidence["segments"][segment]["peak"]["time_range"][1]
        - evidence["segments"][segment]["peak"]["time_range"][0]
        for view, evidence in result["views"].items() if view in usable
        for segment in ("hip", "shoulder")
    )
    result["peak_time_uncertainty_ms"] = round(uncertainty*1000, 2)
    result['peak_time_ranges_seconds'] = peak_ranges
    result['pair_timing'] = {'hip_to_shoulder': _pair_interval(peak_ranges['hip'], peak_ranges['shoulder'], cadence)}
    # An unresolved measured pair stays unresolved even if a later segment is
    # unavailable. Racket evidence is required only for a full-chain order.
    if not result['pair_timing']['hip_to_shoulder']['resolved']:
        result["sequence_quality"] = "UNRESOLVED_AT_FRAME_RATE"
    if racket["peak"] is not None:
        racket_time = racket["peak"]["time"]
        result["racket_peak_frame"] = racket["peak"]["frame_id"]
        sh_rkt = racket_time-peak_times["shoulder"]
        result["latency_shoulder_to_racket_ms"] = round(sh_rkt*1000, 1)
        peak_ranges['racket'] = racket['peak']['time_range']
        result['pair_timing']['shoulder_to_racket'] = _pair_interval(peak_ranges['shoulder'], peak_ranges['racket'], cadence)
        if not all(pair['resolved'] for pair in result['pair_timing'].values()):
            result["sequence_quality"] = "UNRESOLVED_AT_FRAME_RATE"
        else:
            result["is_sequential"] = hip_sh > 0 and sh_rkt > 0
            result["sequence_quality"] = "PROJECTED_ORDER" if result["is_sequential"] else "PROJECTED_REVERSE_ORDER"
    return result
