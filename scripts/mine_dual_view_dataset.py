#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mine_dual_view_dataset.py
─────────────────────────
扫描历史分析审计清单 (audit.json)，批量执行正反双视图合并 (Front + Mirror Back)，
导出补充数据集 JSONL 与统计报告。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dual_view_dataset_miner import RACKET_FLIP_IDX, mine_session_dual_view_data
from joint_racket_2d_analysis import RACKET_POINTS


def run_mining(audit_path: str, output_dir: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rescued = []
    all_projected = []
    all_healed_body = []
    all_hard = []
    session_reports = []

    agg_raw = {}
    agg_merged = {}
    total_frames_all = 0

    for old in audit.get("sessions", []):
        mp = root / old["manifest"]
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        artifacts = {a["role"]: a for a in manifest["artifacts"]}
        paths = {
            role: mp.parent / artifacts[role]["path"]
            for role in ("frame_journal", "event_snapshot", "source_video")
        }
        rows = [
            json.loads(line)
            for line in paths["frame_journal"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        events = json.loads(paths["event_snapshot"].read_text(encoding="utf-8")).get("events", [])
        size = manifest["capture"]["resolution"]

        res = mine_session_dual_view_data(
            session_id=old["session"],
            rows=rows,
            image_size=size,
            events=events,
            source_video=str(paths["source_video"].resolve()),
        )

        total_frames_all += res["total_frames"]
        for k, v in res["fusion_summary"].items():
            agg_raw[k] = agg_raw.get(k, 0) + int(v["raw_frames"])
            agg_merged[k] = agg_merged.get(k, 0) + int(v["merged_frames"])

        all_rescued.extend(res["samples"]["rescued_racket"])
        all_projected.extend(res["samples"]["projected_occluded_racket"])
        all_healed_body.extend(res["samples"]["healed_body"])
        all_hard.extend(res["samples"]["hard_examples"])

        session_reports.append({
            "session_id": res["session_id"],
            "total_frames": res["total_frames"],
            "counts": res["counts"],
            "fusion_summary": res["fusion_summary"],
        })

    def _write_jsonl(filename: str, items: list) -> None:
        p = out / filename
        with p.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    _write_jsonl("rescued_racket_7pt_samples.jsonl", all_rescued)
    _write_jsonl("cross_view_occluded_racket_v1_samples.jsonl", all_projected)
    _write_jsonl("cross_view_healed_body_samples.jsonl", all_healed_body)
    _write_jsonl("cross_view_disagreement_hard_examples.jsonl", all_hard)

    overall_coverage = {}
    for k in sorted(agg_raw.keys()):
        r_n = agg_raw[k]
        m_n = agg_merged[k]
        overall_coverage[k] = {
            "raw_frames": r_n,
            "merged_frames": m_n,
            "gained_frames": m_n - r_n,
            "raw_coverage": round(r_n / max(1, total_frames_all), 4),
            "merged_coverage": round(m_n / max(1, total_frames_all), 4),
        }

    summary = {
        "schema": "tennis.dual-view-dataset-mining.v1",
        "racket_topology": list(RACKET_POINTS),
        "racket_flip_idx": RACKET_FLIP_IDX,
        "total_sessions": len(session_reports),
        "total_frames": total_frames_all,
        "total_supplemented_samples": {
            "rescued_native_racket_7pt": len(all_rescued),
            "projected_occluded_racket_v1": len(all_projected),
            "healed_body_limb_frames": len(all_healed_body),
            "disagreement_hard_examples": len(all_hard),
        },
        "overall_dual_view_coverage": overall_coverage,
        "sessions": session_reports,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine dual-view merged dataset from audit sessions.")
    parser.add_argument(
        "--audit",
        default="data/analysis_results/data_chain_audit_20261020/audit.json",
        help="Path to audit.json",
    )
    parser.add_argument(
        "--output",
        default="data/analysis_results/dual_view_merged_dataset_20261104",
        help="Output directory for mined JSONL and summary.json",
    )
    args = parser.parse_args()
    report = run_mining(args.audit, args.output)
    print(json.dumps({
        "total_frames": report["total_frames"],
        "total_supplemented_samples": report["total_supplemented_samples"],
        "overall_dual_view_coverage": report["overall_dual_view_coverage"],
    }, ensure_ascii=False, indent=2))
