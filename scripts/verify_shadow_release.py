"""Verify the released racket package without loading or executing a model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONTRACT = ROOT / 'docs/releases/640-shadow-preview-20261019.json'


def verify(workspace=ROOT, contract_path=DEFAULT_CONTRACT):
    from racket_pose_detector import KEYPOINT_NAMES, model_digest
    from racket_model_profiles import EXPERIMENT_640_PATH
    contract = json.loads(Path(contract_path).read_text())
    if (contract.get('schema') != 'tennis.shadow-release-contract.v1'
            or contract.get('channel') != 'prerelease'
            or contract.get('quality_gate_passed') is not False
            or contract.get('active_or_exclusive_allowed') is not False
            or contract.get('not_an_active_acceptance_receipt') is not True):
        raise ValueError('Unsupported or promoted shadow release contract')
    model = contract['model']
    if (model.get('path') != EXPERIMENT_640_PATH or model.get('imgsz') != 640
            or model.get('backend') != 'coreml' or model.get('compute_units') != 'ALL'
            or model.get('output_shape') != [1, 26, 8400]
            or model.get('keypoints') != list(KEYPOINT_NAMES)):
        raise ValueError('Shadow model interface differs from this release')
    workspace = Path(workspace).resolve()
    path = (workspace / model['path']).resolve()
    path.relative_to(workspace)
    if not path.is_dir():
        raise FileNotFoundError('Download and extract the release model asset first: ' + model['path'])
    actual = model_digest(path)
    if actual != model['sha256']:
        raise ValueError('Racket model hash mismatch: expected {}, got {}'.format(model['sha256'], actual))
    return {'version': contract['version'], 'model_sha256': actual,
            'model_verified': True, 'scope': 'package identity only; no accuracy or realtime acceptance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--contract', type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.workspace, args.contract), indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
