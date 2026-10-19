"""Build a model-only release asset and checksums from a frozen source commit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racket_pose_detector import model_digest
from scripts.verify_shadow_release import DEFAULT_CONTRACT, verify


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--validation-summary', required=True, type=Path)
    parser.add_argument('--stable-contract', type=Path, help='Separate stable-shadow contract; requires complete acceptance')
    args = parser.parse_args()
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=ROOT, check=True, capture_output=True)
    contract_path = args.stable_contract or DEFAULT_CONTRACT
    if args.stable_contract:
        from scripts.verify_stable_shadow_release import verify as verify_stable
        checked = verify_stable(ROOT, contract_path)
    else:
        checked = verify()
    contract = json.loads(contract_path.read_text())
    version = contract['version']
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    args.output.mkdir(parents=True, exist_ok=False)
    model = ROOT / contract['model']['path']
    files = sorted(p for p in model.rglob('*') if p.is_file() and p.name != '.DS_Store')
    files.append(model.parent / 'candidate.json')
    if any(p.is_symlink() for p in files):
        raise ValueError('Model release must contain ordinary owned files')
    archive = args.output / ('racket-pose-' + version + '.tar.gz')
    with archive.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w|') as tar:
            for path in files:
                info = tar.gettarinfo(str(path), arcname=path.relative_to(ROOT).as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mtime = 0
                info.mode = 0o644
                info.pax_headers = {}
                with path.open('rb') as stream:
                    tar.addfile(info, stream)
    expected = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    with tempfile.TemporaryDirectory() as directory:
        unpacked = Path(directory)
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            if {m.name for m in members} != set(expected) or any(not m.isfile() for m in members):
                raise ValueError('Unexpected archive members')
            for member in members:
                target = (unpacked / member.name).resolve()
                target.relative_to(unpacked.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as stream, target.open('wb') as output:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        output.write(chunk)
                if sha(target) != expected[member.name]:
                    raise ValueError('Archive file hash differs: ' + member.name)
        if model_digest(unpacked / contract['model']['path']) != checked['model_sha256']:
            raise ValueError('Extracted model identity differs')
    validation = json.loads(args.validation_summary.read_text())
    if validation.get('version') != version or validation.get('source_commit') != commit:
        raise ValueError('Validation summary must bind the frozen release commit')
    if args.stable_contract:
        from scripts.verify_stable_shadow_release import verify_validation
        verify_validation(validation, version, commit)
    (args.output / 'validation-summary.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    manifest = {'schema': 'tennis.shadow-release-artifacts.v1', 'version': version,
                'source_commit': commit, 'prerelease': not bool(args.stable_contract), 'contract': contract,
                'contract_sha256': sha(contract_path), 'archive': {'name': archive.name, 'sha256': sha(archive)},
                'model_files_sha256': expected, 'extracted_model_verified': True}
    (args.output / 'release-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    assets = sorted(p for p in args.output.iterdir() if p.is_file())
    (args.output / 'SHA256SUMS').write_text(''.join(sha(p) + '  ' + p.name + '\n' for p in assets))
    print(json.dumps({'version': version, 'commit': commit, 'model_sha256': checked['model_sha256'],
                      'assets': [p.name for p in sorted(args.output.iterdir())],
                      'archive_bytes': archive.stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
