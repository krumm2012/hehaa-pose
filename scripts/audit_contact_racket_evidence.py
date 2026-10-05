"""Offline contact-window evidence audit; no new contact or racket truth inferred."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ball_observation_contract import measurement_ball
from event_source_timing import source_frame_navigation
from observation_policy import finite_number, qualified_front_point


def qualified_racket(candidate, frame_id, image_size):
    if not isinstance(candidate, dict) or candidate.get('observed') is not True:
        return None, 'not_fresh_observation'
    if type(candidate.get('source_frame_id')) is not int or candidate['source_frame_id'] != frame_id:
        return None, 'source_frame_mismatch'
    box = candidate.get('box')
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None, 'invalid_box'
    values = [finite_number(v) for v in box]
    if any(v is None for v in values):
        return None, 'invalid_box'
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= image_size[0] and 0 <= y1 < y2 <= image_size[1]):
        return None, 'box_outside_image_or_degenerate'
    score = finite_number(candidate.get('confidence'))
    if score is None or not 0 <= score <= 1:
        return None, 'invalid_model_score'
    return values, None


def box_point_distance(box, point):
    x1, y1, x2, y2 = box
    x, y = point[:2]
    return math.hypot(max(x1-x, 0, x-x2), max(y1-y, 0, y-y2))


def frame_evidence(row, image_size):
    fid = row['frame_id']
    ball, ball_reason = measurement_ball(row)
    if ball_reason != 'fresh_model_observation':
        ball = None
    if ball and not (0 <= ball[0] < image_size[0] and 0 <= ball[1] < image_size[1]):
        ball, ball_reason = None, 'ball_outside_image'
    rackets, rejected = [], Counter()
    for candidate in row.get('rackets') or []:
        box, reason = qualified_racket(candidate, fid, image_size)
        if reason:
            rejected[reason] += 1
        else:
            distances = {}
            if row.get('pose_observation_coordinate_space') == 'original_source_pixels':
                for view in ('front', 'back'):
                    wrists = []
                    for name in ('left_wrist', 'right_wrist'):
                        point = (row.get('pose_observations', {}).get(view) or {}).get(name)
                        if not isinstance(point, dict) or type(point.get('source_frame_id')) is not int:
                            continue
                        value, why = qualified_front_point(point, fid, minimum_score=.5)
                        if not why and 0 <= value[0] < image_size[0] and 0 <= value[1] < image_size[1]:
                            wrists.append(value)
                    if wrists:
                        distances[view] = min(box_point_distance(box, p) for p in wrists) / math.hypot(*image_size)
            rackets.append({'box': box, 'model_confidence': candidate['confidence'],
                            'wrist_to_box_distance_image_diagonal_units': distances,
                            'ball_to_box_distance_px': box_point_distance(box, ball) if ball else None,
                            'person_assignment_verified': False, 'racket_head_observed': False})
    return {'frame_id': fid, 'ball': ball, 'ball_reason': ball_reason,
            'fresh_rackets': rackets, 'rejected_rackets': dict(rejected),
            'joint_ball_racket_observation_available': bool(ball and rackets),
            'contact_verified': False}


def analyze(rows, events, image_size, radius_seconds=.24):
    if not math.isfinite(radius_seconds) or not 0 < radius_seconds <= 2:
        raise ValueError('Invalid source-time window radius')
    by_id = {r['frame_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate source frame')
    result = []
    seen = set()
    for event in events:
        event_id = event.get('event_id')
        if type(event_id) is not int or event_id < 0 or event_id in seen:
            raise ValueError('Invalid or duplicate event identity')
        seen.add(event_id)
        anchor = event.get('contact_frame')
        item = {'event_id': event_id, 'model_contact_frame': anchor, 'contact_verified': False,
                'racket_head_observed': False, 'review_mode': 'model_assisted_review'}
        if type(anchor) is not int or anchor not in by_id:
            result.append({**item, 'status': 'missing_contact_anchor', 'frames': []})
            continue
        # Validate the event records as one monotonic clock before selecting by PTS.
        selected = [r for r in rows if event['start_frame'] <= r['frame_id'] <= event['end_frame']]
        clock = source_frame_navigation(selected)
        if clock['status'] != 'reported_media_time':
            result.append({**item, 'status': 'source_time_unqualified', 'reasons': clock['reasons'], 'frames': []})
            continue
        times = dict(clock['frames'])
        if anchor not in times:
            result.append({**item, 'status': 'contact_anchor_outside_event', 'frames': []})
            continue
        ids = [fid for fid, t in clock['frames'] if abs(t-times[anchor]) <= radius_seconds]
        evidence = [{**frame_evidence(by_id[fid], image_size), 'source_pts_seconds': times[fid]} for fid in ids]
        result.append({**item, 'status': 'review_required', 'radius_seconds': radius_seconds,
                       'model_contact_pts_seconds': times[anchor], 'frames': evidence,
                       'summary': {'frames': len(ids), 'fresh_ball_frames': sum(bool(f['ball']) for f in evidence),
                                   'fresh_racket_frames': sum(bool(f['fresh_rackets']) for f in evidence),
                                   'joint_observation_frames': sum(f['joint_ball_racket_observation_available'] for f in evidence),
                                   'rejected_racket_candidates': dict(sum((Counter(f['rejected_rackets']) for f in evidence), Counter()))}})
    return {'schema': 'tennis.contact-racket-evidence-audit.v1', 'events': result,
            'image_size': list(image_size), 'coordinate_space': 'original_source_pixels', 'frame_index_base': 0,
            'accuracy_validated': False, 'runtime_modified': False,
            'semantics': 'A ball inside a detected racket box does not establish impact or racket-face/head position.'}


def main():
    parser = argparse.ArgumentParser()
    for key in ('manifest', 'journal', 'events', 'source', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--radius-seconds', type=float, default=.24,
                        help='Source PTS radius around each model contact anchor (0 < radius <= 2)')
    args = parser.parse_args()
    paths = {k: Path(getattr(args, k)) for k in ('manifest', 'journal', 'events', 'source')}
    def sha(path):
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for block in iter(lambda: file.read(1024*1024), b''):
                digest.update(block)
        return digest.hexdigest()
    hashes = {k: sha(p) for k, p in paths.items()}
    manifest = json.loads(paths['manifest'].read_text())
    document = json.loads(paths['events'].read_text())
    entries = {a['role']: a for a in manifest['artifacts']}
    if entries['event_snapshot']['sha256'] != hashes['events']:
        raise ValueError('Event snapshot hash mismatch')
    if entries['frame_journal']['sha256'] != hashes['journal']:
        raise ValueError('Journal hash mismatch')
    if document['session']['session_id'] != manifest['session']['session_id']:
        raise ValueError('Cross-session event document')
    binding = manifest['session']['ground_calibration_application']['input_binding']
    if binding['kind'] != 'video_sha256' or binding['source_id'] != hashes['source']:
        raise ValueError('Source video binding mismatch')
    rows = [json.loads(s) for s in paths['journal'].read_text().splitlines() if s.strip()]
    if any(r.get('session_id') != manifest['session']['session_id'] for r in rows):
        raise ValueError('Cross-session frame record')
    result = analyze(rows, document['events'], manifest['session']['ground_calibration']['image_size'],
                     args.radius_seconds)
    result.update(inputs_sha256=hashes, source_sha256=hashes['source'], session_id=manifest['session']['session_id'],
                  generator_sha256=sha(Path(__file__)))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    import cv2
    selected = {f['frame_id'] for e in result['events'] for f in e['frames']}
    capture = cv2.VideoCapture(str(paths['source'])); fid = 0; images = {}
    try:
        while fid <= max(selected, default=-1):
            ok, image = capture.read()
            if not ok:
                break
            if fid in selected:
                if list(image.shape[1::-1]) != result['image_size']:
                    raise ValueError('Source frame dimensions mismatch')
                target = out/f'frame_{fid}.png'
                if not cv2.imwrite(str(target), image):
                    raise OSError('Cannot write source frame')
                images[str(fid)] = sha(target)
            fid += 1
    finally:
        capture.release()
    if set(map(int, images)) != selected:
        raise ValueError('Source decode incomplete')
    result.update(frame_png_sha256=images, sequential_source_decode_verified=True)
    template = Path(__file__).with_name('contact_racket_review.html').read_text()
    result['review_template_sha256'] = hashlib.sha256(template.encode()).hexdigest()
    (out/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (out/'index.html').write_text(template)
    print(json.dumps([{'event_id': e['event_id'], 'anchor': e['model_contact_frame'], **e.get('summary', {})} for e in result['events']], ensure_ascii=False))


if __name__ == '__main__':
    main()
