"""Accept and audit human racket crop review results; produce an immutable receipt.

Binds human decisions to session and source SHA256 without manufacturing synthetic observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_review(review_data: dict, manifest_data: dict) -> dict:
    if review_data.get("schema") != "tennis.racket-crop-review.v1":
        raise ValueError(f"Unsupported schema: {review_data.get('schema')}")
    
    session_id = manifest_data.get("session", {}).get("session_id")
    if review_data.get("session_id") != session_id:
        raise ValueError(f"Session mismatch: {review_data.get('session_id')} vs {session_id}")
    
    decisions = review_data.get("decisions", {})
    if not isinstance(decisions, dict) or len(decisions) != 20:
        raise ValueError(f"Expected exactly 20 frame decisions, got {len(decisions)}")
    
    allowed_choices = {"valid_racket", "false_positive", "gating_error", "unidentifiable"}
    for fid_str, choice in decisions.items():
        if choice not in allowed_choices:
            raise ValueError(f"Invalid decision '{choice}' for frame {fid_str}")
        
    counts = {}
    for choice in allowed_choices:
        counts[choice] = sum(1 for v in decisions.values() if v == choice)
        
    receipt = {
        "schema": "tennis.racket-crop-review-receipt.v1",
        "session_id": session_id,
        "reviewed_count": len(decisions),
        "decision_breakdown": counts,
        "contact_frames_verified": {
            "21": decisions.get("21") == "valid_racket",
            "110": decisions.get("110") == "valid_racket",
            "191": decisions.get("191") == "valid_racket",
        },
        "gating_errors_acknowledged": [fid for fid, v in decisions.items() if v == "gating_error"],
        "unidentifiable_frames_acknowledged": [fid for fid, v in decisions.items() if v == "unidentifiable"],
        "valid_racket_frames": [fid for fid, v in decisions.items() if v == "valid_racket"],
        "exported_at": review_data.get("exported_at"),
        "receipt_generated_at": None,
    }
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True, help="Path to exported review JSON")
    parser.add_argument("--manifest", required=True, help="Path to evidence manifest")
    parser.add_argument("--output", required=True, help="Path to receipt output directory")
    args = parser.parse_args()

    review_path = Path(args.review).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    review_data = json.loads(review_path.read_text(encoding="utf-8"))
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    receipt = validate_review(review_data, manifest_data)
    receipt["review_sha256"] = digest(review_path)
    receipt["manifest_sha256"] = digest(manifest_path)

    import datetime
    receipt["receipt_generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Save outputs
    (out_dir / "court02_racket_manual_review_submitted.json").write_text(
        json.dumps(review_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    (out_dir / "court02_racket_manual_review_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print(f"Receipt written to: {out_dir / 'court02_racket_manual_review_receipt.json'}")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
