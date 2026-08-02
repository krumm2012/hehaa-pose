#!/usr/bin/env python3
"""Verify and semantically replay a tennis analysis evidence bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from session_evidence_bundle import (
    replay_evidence_manifest,
    verify_evidence_manifest,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify and replay a session evidence manifest.")
    parser.add_argument("manifest", help="Path to *_evidence_manifest.json")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", help="Replay result JSON path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    verification = verify_evidence_manifest(args.manifest)
    print(
        f"valid={str(verification['valid']).lower()}"
        f" replayable={str(verification['replayable']).lower()}"
        f" artifacts={len(verification['artifacts'])}"
    )
    if not verification["replayable"]:
        return 2
    if args.verify_only:
        return 0
    output = args.output or str(
        Path(args.manifest).with_name(
            f"{Path(args.manifest).stem}_replay_result.json"
        )
    )
    result = replay_evidence_manifest(args.manifest, output)
    print(
        f"frames={result['frame_record_count']}"
        f" events={len(result['replayed_event_signatures'])}"
        f" semantic_match={str(result['semantic_match']).lower()}"
    )
    print(f"json={output}")
    return 0 if result["semantic_match"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
