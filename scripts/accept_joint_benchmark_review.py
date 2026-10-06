"""Accept and verify independent joint benchmark labels, producing cryptographic receipt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_annotation_evaluation import evaluate_joint_labels


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Accept independent joint benchmark labels")
    parser.add_argument("--labels", required=True, help="Path to independent joint labels JSON")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for receipt & report")
    args = parser.parse_args()

    labels_p = Path(args.labels)
    preds_p = Path(args.predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = json.loads(labels_p.read_text(encoding="utf-8"))
    preds = json.loads(preds_p.read_text(encoding="utf-8"))

    if labels.get("schema") != "tennis.independent-joint-labels.v1":
        raise ValueError(f"Invalid schema: {labels.get('schema')}")
    if not labels.get("confirmed"):
        raise ValueError("Labels not confirmed by annotator")
    if not labels.get("annotator_id"):
        raise ValueError("Missing annotator_id")

    eval_5px = evaluate_joint_labels(labels, preds, tolerance_px=5.0)
    eval_8px = evaluate_joint_labels(labels, preds, tolerance_px=8.0)

    # All identifiable paired samples
    all_group = [g for g in eval_5px.get("groups", []) if g.get("view") == "back" and g.get("joint") == "all"]
    # Aggregate errors
    errs = []
    for g in eval_5px.get("groups", []):
        if g.get("joint") != "all":
            for s in g.get("samples", []):
                if s.get("error_px") is not None:
                    errs.append(s["error_px"])

    errs_sorted = sorted(errs)
    med = errs_sorted[len(errs_sorted) // 2] if errs else None
    mean_err = round(sum(errs) / len(errs), 2) if errs else None
    pass_5 = round(sum(e <= 5.0 for e in errs) / len(errs) * 100, 1) if errs else None
    pass_8 = round(sum(e <= 8.0 for e in errs) / len(errs) * 100, 1) if errs else None

    visible_count = sum(1 for v in labels.get("labels", {}).values() if v.get("visible"))
    unidentifiable_count = sum(1 for v in labels.get("labels", {}).values() if not v.get("visible"))

    receipt = {
        "schema": "tennis.joint-benchmark-receipt.v1",
        "session_id": "court02_temporal_racket_20261005T161814Z_0ab2c2",
        "source_sha256": labels.get("source_sha256"),
        "annotator_id": labels.get("annotator_id"),
        "confirmed": True,
        "evaluated_frames": [f["frame_id"] for f in labels.get("frames", [])],
        "total_labels": len(labels.get("labels", {})),
        "visible_count": visible_count,
        "unidentifiable_count": unidentifiable_count,
        "evaluation_metrics": {
            "paired_count": len(errs),
            "mean_error_px": mean_err,
            "median_error_px": round(med, 2) if med is not None else None,
            "pass_rate_5px": pass_5,
            "pass_rate_8px": pass_8,
        },
        "receipt_generated_at": datetime.now(timezone.utc).isoformat(),
        "labels_sha256": digest(labels_p),
        "predictions_sha256": digest(preds_p),
        "status": "evaluated_independent_labels",
    }

    receipt_path = out_dir / "court02_independent_joint_benchmark_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Receipt generated at: {receipt_path}")
    print(f"Status: {receipt['status']}, Annotator: {receipt['annotator_id']}, Labels SHA256: {receipt['labels_sha256']}")


if __name__ == "__main__":
    main()
