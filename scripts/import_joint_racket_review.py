"""Import a user review and recalculate a separate human-anchor report branch."""
import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.serve_joint_racket_2d_review import validate_contact_review, review_progress
from swing_motion_features import extract_motion_features
from swing_biomechanics import enrich_events_with_biomechanics
from swing_session_quality import build_session_quality_dashboard
from swing_report_builder import build_report_payload, write_report_html


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, doc): path.write_text(json.dumps(doc, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def run(review_path, directory):
    directory = Path(directory).resolve(); review_path = Path(review_path).resolve()
    dataset = json.loads((directory/'dataset.json').read_text())
    review = json.loads(review_path.read_text())
    receipt = validate_contact_review(review, dataset)
    session = next(s for s in dataset['sessions'] if s['id'] == receipt['session_id'])
    mp = Path(session['manifest']); manifest = json.loads(mp.read_text())
    artifacts = {a['role']: a for a in manifest['artifacts']}
    paths = {k: mp.parent/artifacts[k]['path'] for k in session['hashes']}
    for role, path in paths.items():
        if sha(path) != session['hashes'][role]: raise ValueError('Source hash changed: '+role)
    # Idempotence: don't mint a duplicate receipt for an identical imported review.
    folder = directory/'human_reviews'; folder.mkdir(exist_ok=True)
    saved = None
    for candidate in folder.glob('*.json'):
        try:
            if (sha(candidate) == candidate.stem
                    and json.loads(candidate.read_text()).get('review') == review):
                saved = candidate; break
        except (OSError, ValueError): pass
    if saved is None:
        payload = {'review': review, 'receipt': receipt,
                   'received_at': datetime.now(timezone.utc).isoformat(),
                   'import_source_sha256': sha(review_path)}
        body = (json.dumps(payload, ensure_ascii=False, indent=2)+'\n').encode()
        saved = folder/(hashlib.sha256(body).hexdigest()+'.json')
        with saved.open('xb') as f: f.write(body)
    progress = review_progress(directory, dataset)
    write(directory.parent/'human_review_progress.json', progress)
    old = json.loads(paths['event_snapshot'].read_text())
    times = {f['frame_id']: f['source_seconds'] for f in session['frames']}
    changes = []
    for e in old['events']:
        item = receipt['events'][str(e['event_id'])]
        confirmed = item['status'] == 'confirmed'
        anchor = item.get('anchor_frame') if confirmed else None
        changes.append({'event_id': e['event_id'], 'model_candidate_frame': e['contact_frame'],
                        'human_status': item['status'], 'human_anchor_frame': anchor,
                        'frame_delta': anchor-e['contact_frame'] if confirmed else None,
                        'pts_delta_ms': round(1000*(times[anchor]-times[e['contact_frame']]), 3) if confirmed else None,
                        'human_interval': [item.get('start_frame'), item.get('end_frame')]})
    target = directory.parent/session['id']/('human_anchor_'+sha(review_path)[:12])
    target.mkdir(parents=True, exist_ok=True)
    write(target/'anchor_comparison.json', {'source_hashes':session['hashes'], 'reviewer':receipt['reviewer'],
          'receipt':str(saved), 'independent_reference':False, 'events':changes})
    # Only fully confirmed sessions with anchors inside frozen event boundaries
    # enter the human-anchor measurement branch. Intervals stay unresolved.
    all_confirmed = all(item['status'] == 'confirmed' for item in receipt['events'].values())
    within_events = all(e['start_frame'] <= receipt['events'][str(e['event_id'])].get('anchor_frame', -1) <= e['end_frame']
                        for e in old['events']) if all_confirmed else False
    if all_confirmed and within_events:
        rows = [json.loads(line) for line in paths['frame_journal'].read_text().splitlines() if line.strip()]
        features = extract_motion_features(rows)
        events = copy.deepcopy(old['events'])
        for event in events:
            item = receipt['events'][str(event['event_id'])]
            event['model_candidate_contact_frame'] = event['contact_frame']
            event['contact_frame'] = item['anchor_frame']
            event['contact_status'] = 'human_confirmed'
            event['human_contact_confirmation'] = {'reviewer':receipt['reviewer'], 'receipt_sha256':saved.stem,
                    'interval':[item['start_frame'],item['end_frame']], 'note':item['note'],
                    'view':item['view'], 'independent_reference':False}
            for key in ('coach_advices','coach_advice','deepseek_advice','coach_calibration','practice_review'):
                event.pop(key, None)
        events = enrich_events_with_biomechanics(events, rows, features)
        write(target/'frames.json', {'frames':rows, 'session':manifest['session'],
              'video_info':{'path':str(paths['source_video'].resolve()),'fps':manifest['capture']['fps']}})
        write(target/'events.json', {**old, 'events':events,
              'summary':{'session_quality':build_session_quality_dashboard(events)},
              'human_anchor_branch':{'receipt_sha256':saved.stem,'original_events_modified':False,
                                    'advice_recomputed':False,'accuracy_validated':False}})
        write_report_html(build_report_payload(str(target/'frames.json'),str(target/'events.json')),str(target/'report.html'))
    result = {'receipt':str(saved), 'progress':progress, 'comparison':str(target/'anchor_comparison.json'),
              'human_anchor_report':str(target/'report.html') if all_confirmed and within_events else None,
              'reanalysis_status':'human_anchor_measurements_recomputed' if all_confirmed and within_events else 'awaiting_confirmed_in_event_anchors'}
    write(target/'import_result.json', result)
    from scripts.build_contact_review_summary import build as build_summary
    build_summary(directory)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',required=True);p.add_argument('--directory',required=True)
    a=p.parse_args();run(a.review,a.directory)
