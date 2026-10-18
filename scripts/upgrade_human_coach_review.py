#!/usr/bin/env python3
"""Create a V3 review copy; never replace a source report or human decision."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manual_review_policy import VERSION
from manual_review_workflow import discover_session_paths, process_manual_review, refresh_realtime_report


def inventory(root):
    return [{'path':str(path.relative_to(root)), 'bytes':path.stat().st_size,
             'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(root.rglob('*')) if path.is_file()]


def upgrade(source):
    source=source.expanduser().resolve()
    original_paths=discover_session_paths(source)
    original_inventory=inventory(source)
    target=source.with_name(source.name+'_review_v3_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
    shutil.copytree(source,target)
    # The copied evidence manifest describes the original analysis, not this new renderer.
    for manifest in target.glob('*evidence_manifest.json'):
        manifest.rename(manifest.with_name('source_'+manifest.name))
    paths=discover_session_paths(target)
    if paths['annotations'].exists():
        annotations=json.loads(paths['annotations'].read_text())
        annotations=deepcopy(annotations)
        annotations['review_policy_version']=VERSION
        for row in annotations['events']:
            row.setdefault('contact_review','unreviewed')
            row.setdefault('coach_review',{})
            row.setdefault('practice_review',{'ratings':{},'confirmed':False})
        process_manual_review(paths,annotations)
    refresh_realtime_report(paths)
    # Refresh is opt-in for a historical report: keep it paused when first opened.
    page=paths['report'].read_text()
    page=page.replace('let autoRefreshEnabled = true;', 'let autoRefreshEnabled = false;')
    page=page.replace("sessionStorage.getItem(refreshStorageKey) !== 'false'", "sessionStorage.getItem(refreshStorageKey) === 'true'")
    paths['report'].write_text(page)
    assert inventory(source)==original_inventory, 'Source session changed during upgrade'
    receipt={'schema_version':'tennis.human-review-upgrade.v1','review_policy_version':VERSION,
             'source_session_dir':str(source),'review_session_dir':str(target),
             'source_unchanged':True,'source_files':original_inventory,
             'upgrade_snapshot_files':inventory(target),
             'scope':'Snapshot at upgrade; later human reviews create independent immutable revisions.',
             'migration':'Existing event decisions retained; missing human contact/coach approvals remain unreviewed.'}
    (target/'review_upgrade_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return paths['report']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir',type=Path,required=True)
    print(upgrade(parser.parse_args().session_dir))
