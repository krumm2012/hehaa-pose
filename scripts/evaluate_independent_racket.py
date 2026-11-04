"""Evaluate explicitly reviewed held-out observations; preserve inputs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_racket_validation import evaluate

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('prediction', 'reference', 'exclusions', 'output'): p.add_argument('--'+name, required=True)
    p.add_argument('--point-tolerance-px', required=True, type=float)
    a = p.parse_args()
    paths = [Path(a.prediction), Path(a.reference), Path(a.exclusions)]
    prediction, reference, exclusions = [json.loads(path.read_text()) for path in paths]
    result = evaluate(prediction, reference, exclusions['excluded_source_sha256'], a.point_tolerance_px)
    result['input_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    with Path(a.output).open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
