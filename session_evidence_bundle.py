"""Build, verify, and replay one analysis session evidence bundle."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from analysis_data_contracts import utc_iso_from_ns


EVIDENCE_BUNDLE_SCHEMA_VERSION = "tennis.replay-evidence-bundle.v1"
REPLAY_RESULT_SCHEMA_VERSION = "tennis.replay-result.v1"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _artifact_entry(
    manifest_dir: Path,
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    source = Path(str(spec["path"])).expanduser().resolve()
    exists = source.is_file()
    try:
        stored_path = os.path.relpath(source, manifest_dir)
        external = source != manifest_dir and manifest_dir not in source.parents
    except ValueError:
        stored_path = str(source)
        external = True
    return {
        "role": str(spec["role"]),
        "path": stored_path,
        "external": external,
        "required_for_replay": bool(spec.get("required_for_replay", False)),
        "exists": exists,
        "bytes": source.stat().st_size if exists else None,
        "sha256": sha256_file(source) if exists else None,
    }


def write_session_evidence_manifest(
    output_path: str,
    *,
    session: Mapping[str, Any],
    status: str,
    capture: Mapping[str, Any],
    replay: Mapping[str, Any],
    artifacts: Iterable[Mapping[str, Any]],
) -> str:
    """Write a checksummed manifest for deterministic post-inference replay."""
    output = Path(output_path).expanduser().resolve()
    entries = [
        _artifact_entry(output.parent, spec)
        for spec in artifacts
        if spec.get("path")
    ]
    missing_required = sorted(
        entry["role"]
        for entry in entries
        if entry["required_for_replay"] and not entry["exists"]
    )
    required_roles = sorted(
        entry["role"] for entry in entries if entry["required_for_replay"]
    )
    created_ns = time.time_ns()
    document = {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "bundle_id": str(session.get("session_id") or output.stem),
        "created_at": utc_iso_from_ns(created_ns),
        "created_at_unix_ns": created_ns,
        "status": str(status),
        "session": dict(session),
        "capture": dict(capture),
        "replay": dict(replay),
        "artifacts": entries,
        "completeness": {
            "required_roles": required_roles,
            "missing_required_roles": missing_required,
            "replayable": bool(required_roles) and not missing_required,
        },
    }
    _atomic_write_json(output, document)
    return str(output)


def load_evidence_manifest(path: str) -> Dict[str, Any]:
    manifest = Path(path).expanduser().resolve()
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Evidence manifest must be a JSON object")
    if document.get("schema_version") != EVIDENCE_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported evidence manifest schema: {document.get('schema_version')}"
        )
    return document


def _resolve_artifact(manifest_path: Path, entry: Mapping[str, Any]) -> Path:
    value = Path(str(entry.get("path") or ""))
    return value.resolve() if value.is_absolute() else (manifest_path.parent / value).resolve()


def verify_evidence_manifest(path: str) -> Dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    document = load_evidence_manifest(str(manifest_path))
    results = []
    for entry in document.get("artifacts") or []:
        artifact = _resolve_artifact(manifest_path, entry)
        exists = artifact.is_file()
        actual_bytes = artifact.stat().st_size if exists else None
        actual_sha256 = sha256_file(artifact) if exists else None
        expected_sha256 = entry.get("sha256")
        valid = bool(
            exists
            and expected_sha256
            and actual_bytes == entry.get("bytes")
            and actual_sha256 == expected_sha256
        )
        results.append(
            {
                "role": entry.get("role"),
                "path": entry.get("path"),
                "required_for_replay": bool(entry.get("required_for_replay")),
                "exists": exists,
                "valid": valid,
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
            }
        )
    required = [row for row in results if row["required_for_replay"]]
    return {
        "manifest": str(manifest_path),
        "valid": all(row["valid"] for row in results),
        "replayable": bool(required) and all(row["valid"] for row in required),
        "artifacts": results,
    }


def _artifact_by_role(
    manifest_path: Path,
    document: Mapping[str, Any],
    role: str,
) -> Optional[Path]:
    entry = next(
        (
            item
            for item in document.get("artifacts") or []
            if item.get("role") == role
        ),
        None,
    )
    return _resolve_artifact(manifest_path, entry) if entry else None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid frame JSONL at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Frame JSONL line {line_number} must be an object")
        records.append(row)
    return records


def event_semantic_signature(event: Mapping[str, Any]) -> Dict[str, Any]:
    advices = event.get("coach_advices") or []
    if not advices and event.get("coach_advice"):
        advices = [event["coach_advice"]]
    return {
        "event_id": int(event.get("event_id") or 0),
        "start_frame": int(event.get("start_frame") or 0),
        "contact_frame": int(event.get("contact_frame") or 0),
        "peak_frame": int(event.get("peak_frame") or 0),
        "end_frame": int(event.get("end_frame") or 0),
        "stroke_type": event.get("stroke_type"),
        "coach_codes": [str(advice.get("code") or "") for advice in advices],
    }


def replay_evidence_manifest(
    manifest_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Replay FrameRecords through the realtime event engine and compare semantics."""
    source = Path(manifest_path).expanduser().resolve()
    document = load_evidence_manifest(str(source))
    verification = verify_evidence_manifest(str(source))
    if not verification["replayable"]:
        raise ValueError("Evidence bundle failed required artifact verification")

    frame_path = _artifact_by_role(source, document, "frame_journal")
    event_path = _artifact_by_role(source, document, "event_snapshot")
    if frame_path is None:
        raise ValueError("Evidence bundle has no frame_journal artifact")
    frames = _load_jsonl(frame_path)
    replay = document.get("replay") or {}
    coach_options = replay.get("coach") or {}
    coach = None
    if coach_options.get("enabled"):
        from local_realtime_coach import LocalRealtimeCoach

        coach = LocalRealtimeCoach(
            max_chars=int(coach_options.get("max_chars", 15)),
            max_suggestions=int(coach_options.get("max_suggestions", 3)),
            min_confidence=float(coach_options.get("min_confidence", 0.45)),
            thresholds=coach_options.get("thresholds") or {},
        )

    from realtime_swing_pipeline import RealtimeSwingEventEngine

    engine = RealtimeSwingEventEngine(
        fps=float(replay.get("fps") or 25.0),
        analysis_interval_frames=int(replay.get("analysis_interval_frames") or 5),
        settle_frames=int(replay.get("settle_frames") or 0),
        window_frames=int(replay.get("window_frames") or 200),
        coach=coach,
        session_metadata=document.get("session") or {},
        **(replay.get("swing_options") or {}),
    )
    for record in frames:
        engine.push_frame(record)
    engine.flush()
    replayed = engine.snapshot()
    replayed_signatures = [
        event_semantic_signature(event) for event in replayed.get("events") or []
    ]
    expected_signatures = []
    if event_path is not None and event_path.is_file():
        expected_document = json.loads(event_path.read_text(encoding="utf-8"))
        expected_signatures = [
            event_semantic_signature(event)
            for event in expected_document.get("events") or []
        ]
    result = {
        "schema_version": REPLAY_RESULT_SCHEMA_VERSION,
        "bundle_id": document.get("bundle_id"),
        "verification": verification,
        "frame_record_count": len(frames),
        "semantic_match": replayed_signatures == expected_signatures,
        "expected_event_signatures": expected_signatures,
        "replayed_event_signatures": replayed_signatures,
        "replayed_document": replayed,
    }
    if output_path:
        _atomic_write_json(Path(output_path).expanduser().resolve(), result)
    return result
