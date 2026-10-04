"""Audit independent per-view SAM3D segments without calling them motion-capture truth.

Only reads the supplied dataset. Writes a separate, replayable diagnostic report.
MHR IDs follow Tennis-Vision's estimate_mirror.py and generate_multiview.py:
shoulders 5/6, hips 9/10; flipped inference restores x, retaining anatomical IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {"count": int(len(values)), "median": float(np.median(values)) if len(values) else None,
            "p90": float(np.percentile(values, 90)) if len(values) else None}


def unit_segments(joints, pair):
    segment = joints[:, pair[1]] - joints[:, pair[0]]
    length = np.linalg.norm(segment, axis=1)
    valid = np.isfinite(segment).all(axis=1) & (length > 1e-6)
    return segment / np.maximum(length[:, None], 1e-6), valid


def direction_speed(units, valid, fps):
    """Total 3D direction change, NOT axial pelvis/trunk angular velocity."""
    speed = np.full(len(units), np.nan)
    consecutive = valid[1:] & valid[:-1]
    angle = np.degrees(np.arccos(np.clip(np.sum(units[1:] * units[:-1], axis=1), -1, 1)))
    speed[1:] = np.where(consecutive, angle * fps, np.nan)
    return speed


def segment_audit(front, mirror, valid, normal, pair, fps):
    normal = np.asarray(normal, dtype=float)
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-6:
        raise ValueError("Invalid mirror normal")
    normal /= np.linalg.norm(normal)
    reflection = np.eye(3) - 2 * np.outer(normal, normal)
    a, av = unit_segments(front, pair)
    b, bv = unit_segments(mirror, pair)
    b = b @ reflection.T
    usable = np.asarray(valid, dtype=bool) & av & bv
    angles = np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))
    sa = direction_speed(a, usable, fps)
    sb = direction_speed(b, usable, fps)
    common = np.isfinite(sa) & np.isfinite(sb)
    correlation = None
    if common.sum() >= 6 and np.std(sa[common]) > 1e-6 and np.std(sb[common]) > 1e-6:
        correlation = float(np.corrcoef(sa[common], sb[common])[0, 1])
    return {"joint_pair_mhr": list(pair), "direction_difference_deg": summary(angles[usable]),
            "speed_difference_deg_per_s": summary(np.abs(sa[common] - sb[common])),
            "same_frame_direction_speed_correlation": correlation,
            "valid_frames": int(usable.sum()),
            "signals": [{"frame": int(i), "time_s": float(i / fps),
                         "direction_difference_deg": float(angles[i]),
                         "front_direction_speed_deg_per_s": float(sa[i]) if np.isfinite(sa[i]) else None,
                         "mirror_direction_speed_deg_per_s": float(sb[i]) if np.isfinite(sb[i]) else None}
                        for i in np.flatnonzero(usable)]}


def audit(result, reconstruction, source, session_manifest=None):
    result, reconstruction, source = map(Path, (result, reconstruction, source))
    meta = json.loads((result / "mesh_meta.json").read_text())
    geometry = json.loads((result / "mirror_geometry.json").read_text())
    manifest = json.loads((result / "multiview_manifest.json").read_text())
    with np.load(reconstruction, allow_pickle=False) as data:
        front, mirror = data["joints"], data["mirror_joints"]
        valid = data["mirror_valid"].astype(bool)
        native_sha = str(data["video_sha256"].item())
    identities = [sha256(source), native_sha, meta["video_sha256"], geometry["video_sha256"], manifest["source_sha256"]]
    if len(set(identities)) != 1:
        raise ValueError("Source, native reconstruction, metadata and geometry SHA256 do not match")
    fps = float(meta["fps"])
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("Invalid sampling rate")
    if front.shape != mirror.shape or front.ndim != 3 or front.shape[2] != 3 or front.shape[1] <= 10:
        raise ValueError("Invalid MHR joint arrays")
    if len(front) != int(meta["frames"]) or valid.shape != (len(front),):
        raise ValueError("Frame counts do not match")
    session_match = None
    if session_manifest:
        session = json.loads(Path(session_manifest).read_text())
        hashes = [a["sha256"] for a in session["artifacts"] if a["role"] == "source_video"]
        session_match = identities[0] in hashes
    blockers = ["No independent measured 3D reference or motion-capture trajectories in these inputs",
                "No verified world vertical / anatomical axial rotation coordinate frame",
                "No independently confirmed ball contact windows / exposure timestamp mapping",
                "Plane fit uses shoulders and hips; agreement is partly dependent on fitted geometry",
                "25 fps sampling cannot resolve sub-frame peak order" if fps == 25 else "Peak timing limited by sampling cadence"]
    if not geometry.get("measured_camera", False):
        blockers.append("Camera intrinsics and mirror pose are estimated, not measured calibration")
    return {"schema_version": "tennis.kinematic-3d-evidence-audit.v1",
            "status": "model_consistency_only_true_chain_unvalidated", "coach_eligible": False,
            "accuracy_confidence": None, "source_sha256": identities[0], "fps": fps,
            "sampling_interval_ms": 1000 / fps, "frames": len(front),
            "mirror_frames": int(valid.sum()), "source_identity_verified": True,
            "session_source_sha256_matches": session_match,
            "geometry": {k: geometry.get(k) for k in ["status", "measured_camera", "fit_joint_scope",
                           "heldout_median_px", "heldout_p90_px", "core_heldout_median_px", "core_heldout_p90_px"]},
            "segments": {name: segment_audit(front, mirror, valid, geometry["normal_camera"], pair, fps)
                         for name, pair in [("hip", (9, 10)), ("shoulder", (5, 6))]},
            "limitations": blockers,
            "inputs": {"result": str(result.resolve()), "reconstruction": str(reconstruction.resolve()),
                       "reconstruction_sha256": sha256(reconstruction), "source": str(source.resolve()),
                       "geometry_sha256": sha256(result / "mirror_geometry.json"),
                       "script_sha256": sha256(__file__)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["result", "reconstruction", "source", "output"]:
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--session-manifest", type=Path)
    args = parser.parse_args()
    report = audit(args.result, args.reconstruction, args.source, args.session_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Use a fresh output path to preserve historical evidence")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "segments": {k: {a: b for a, b in v.items() if a != "signals"}
                                   for k, v in report["segments"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
