"""Receive source-bound human contact intervals without rewriting model events."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def validate_review(review, audit):
    if review.get('schema') != 'tennis.contact-racket-review.v1':
        raise ValueError('Unsupported review schema')
    for key in ('source_sha256', 'inputs_sha256', 'session_id', 'coordinate_space', 'frame_index_base'):
        if review.get(key) != audit.get(key):
            raise ValueError('Review binding mismatch: ' + key)
    if review.get('annotation_mode') != 'model_assisted_review' or review.get('independent_reference') is not False:
        raise ValueError('This board is model-assisted, not independent ground truth')
    annotator = review.get('annotator_id')
    if not isinstance(annotator, str) or not annotator.strip():
        raise ValueError('Missing annotator identity')
    events = {str(e['event_id']): e for e in audit['events'] if e['frames']}
    submitted = review.get('events')
    if not isinstance(submitted, dict) or set(submitted) != set(events):
        raise ValueError('Event identities mismatch')
    points = review.get('contact_interval_pts')
    if not isinstance(points, dict) or set(points) != set(events):
        raise ValueError('Interval PTS identities mismatch')
    receipts = []
    for event_id, event in events.items():
        item = submitted[event_id]
        if not isinstance(item, dict):
            raise ValueError('Invalid event review')
        decision = item.get('decision')
        if decision not in {'unreviewed', 'interval', 'unknown', 'window_insufficient'}:
            raise ValueError('Invalid contact decision')
        frames = {f['frame_id']: f for f in event['frames']}
        interval = [item.get('start_frame'), item.get('end_frame')]
        reported = points[event_id]
        if not isinstance(reported, dict):
            raise ValueError('Invalid interval PTS')
        if decision == 'interval':
            if any(type(fid) is not int or fid not in frames for fid in interval) or interval[0] > interval[1]:
                raise ValueError('Interval outside source window or reversed')
            for fid, name in zip(interval, ['start_seconds', 'end_seconds']):
                value = reported.get(name)
                if (type(value) not in (int, float) or not math.isfinite(value)
                        or abs(value - frames[fid]['source_pts_seconds']) > 1e-9):
                    raise ValueError('Interval PTS mismatch')
        elif any(fid is not None for fid in interval) or any(reported.get(k) is not None for k in ['start_seconds', 'end_seconds']):
            raise ValueError('Non-interval decision must not retain endpoints')
        roles = item.get('racket_roles')
        if not isinstance(roles, dict):
            raise ValueError('Invalid racket role annotations')
        annotated_roles = 0
        for key, role in roles.items():
            try:
                fid, index = map(int, key.split(':'))
            except (TypeError, ValueError, AttributeError):
                raise ValueError('Invalid racket candidate identity') from None
            if (key != f'{fid}:{index}' or fid not in frames or index < 0
                    or index >= len(frames[fid]['fresh_rackets'])):
                raise ValueError('Racket role refers to absent or stale candidate')
            if role not in {'unreviewed', 'front', 'back', 'wrong_object', 'unknown'}:
                raise ValueError('Invalid racket role')
            annotated_roles += role != 'unreviewed'
        total_roles = sum(len(f['fresh_rackets']) for f in frames.values())
        receipts.append({'event_id': int(event_id), 'decision': decision,
                         'interval_source_frames': interval,
                         'interval_pts_seconds': reported,
                         'annotated_racket_roles': annotated_roles,
                         'available_fresh_racket_candidates': total_roles,
                         'racket_roles_complete': annotated_roles == total_roles,
                         'contact_truth_verified': False})
    complete = all(r['decision'] != 'unreviewed' for r in receipts)
    if review.get('event_reviews_complete') is not complete:
        raise ValueError('Event completion flag mismatch')
    return {'schema': 'tennis.contact-racket-review-receipt.v1',
            'session_id': audit['session_id'], 'source_sha256': audit['source_sha256'],
            'inputs_sha256': audit['inputs_sha256'], 'annotator_id': annotator.strip(),
            'event_reviews_complete': complete, 'events': receipts,
            'independent_reference': False, 'accuracy_validated': False,
            'contact_truth_verified': False, 'model_events_modified': False,
            'raw_observations_modified': False,
            'semantics': 'Human-reviewed possible intervals; only explicit candidate roles are counted.'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('review', 'audit', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    submitted = Path(args.review).read_bytes()
    audited = Path(args.audit).read_bytes()
    receipt = validate_review(json.loads(submitted), json.loads(audited))
    receipt.update(submitted_sha256=hashlib.sha256(submitted).hexdigest(),
                   audit_sha256=hashlib.sha256(audited).hexdigest())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'submitted_review.json').write_bytes(submitted)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
