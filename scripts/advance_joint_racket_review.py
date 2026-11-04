"""Recalculate fixed audit events into new reports, never rewrite source evidence."""
import argparse
import csv
import hashlib
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swing_motion_features import extract_motion_features
from swing_biomechanics import enrich_events_with_biomechanics
from swing_session_quality import build_session_quality_dashboard
from swing_report_builder import build_report_payload, write_report_html
from joint_racket_2d_live import LiveJointRacket2D
from scripts.build_joint_racket_2d_review import build as build_review


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def build(audit_path, output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    audit = json.loads(Path(audit_path).read_text())
    build_review(audit_path, output / 'contact_review')
    dataset = json.loads((output / 'contact_review/dataset.json').read_text())
    queue = []; summaries = []
    for binding in audit['sessions']:
        manifest_path = ROOT / binding['manifest']
        manifest = json.loads(manifest_path.read_text())
        artifacts = {a['role']: a for a in manifest['artifacts']}
        paths = {k: manifest_path.parent / artifacts[k]['path'] for k in ('frame_journal', 'event_snapshot', 'source_video')}
        for role, path in paths.items():
            if digest(path) != binding['hashes'][role]: raise ValueError('Changed source: ' + role)
        rows = [json.loads(line) for line in paths['frame_journal'].read_text().splitlines() if line.strip()]
        old = json.loads(paths['event_snapshot'].read_text())
        features = extract_motion_features(rows)
        events = enrich_events_with_biomechanics(old['events'], rows, features)
        # Recompute measurement only. Historical advice must not appear newly validated.
        for e in events:
            for key in ('coach_advices', 'coach_advice', 'deepseek_advice', 'coach_calibration', 'practice_review'):
                e.pop(key, None)
            e['reanalysis'] = {'boundaries': 'frozen_model_candidates', 'human_confirmed': False,
                             'advice_recomputed': False, 'historical_source_event_id': e['event_id']}
        dashboard = build_session_quality_dashboard(events)
        target = output / binding['session']; target.mkdir()
        write(target / 'frames.json', {'frames': rows, 'session': manifest['session'],
              'video_info': {'path': str(paths['source_video'].resolve()), 'fps': manifest['capture']['fps']}})
        write(target / 'events.json', {**old, 'events': events, 'summary': {'session_quality': dashboard},
              'reanalysis': {'source_hashes': binding['hashes'], 'original_event_boundaries_preserved': True,
                             'new_inference': False, 'accuracy_validated': False}})
        payload = build_report_payload(str(target / 'frames.json'), str(target / 'events.json'))
        write_report_html(payload, str(target / 'report.html'))
        live = LiveJointRacket2D(); timings = []; snapshots = []
        import cv2
        cap = cv2.VideoCapture(str(paths['source_video']))
        for row in rows:
            start = time.perf_counter()
            diagnostic = live.update(row, manifest['capture']['resolution'])
            timings.append((time.perf_counter()-start)*1000)
            if len(snapshots) < 2 and (diagnostic['active_gaps_ge_3'] or row['frame_id'] == events[0]['contact_frame']):
                cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame_id']); ok, image = cap.read()
                if ok:
                    live.draw_source(image); live.draw_status(image, diagnostic)
                    filename = f"live_2d_frame_{row['frame_id']}.jpg"
                    cv2.imwrite(str(target / filename), image); snapshots.append(filename)
        cap.release()
        session = next(s for s in dataset['sessions'] if s['id'] == binding['session'])
        draft = {'schema': 'tennis.joint-racket-contact-review.v1', 'session_id': session['id'],
                 'hashes': session['hashes'], 'annotation_mode': 'model_assisted_review',
                 'independent_reference': False, 'reviewer': '', 'events': {}}
        comparisons = []
        for before, after in zip(old['events'], events):
            eid = after['event_id']; cf = after['contact_frame']
            original = (before.get('extended_biomechanics') or {}).get('racket_head_speed') or {}
            current = (after.get('extended_biomechanics') or {}).get('racket_head_speed') or {}
            comparisons.append({'event_id': eid, 'candidate_contact_frame': cf,
                                'historical_speed': original, 'recomputed_speed': current})
            gaps = [g for g in session['statistics']['observation_gaps_ge_3'] if eid in g.get('near_contact_event_ids', []) and g['track'].startswith('front.')]
            queue.append({'session': session['id'], 'event_id': eid, 'candidate_contact_frame': cf,
                          'status': 'unreviewed', 'front_gaps_near_contact': len(gaps),
                          'priority': 'contact_gap' if gaps else 'standard'})
            draft['events'][str(eid)] = {'status': 'unreviewed', 'start_frame': None, 'end_frame': None,
                                        'anchor_frame': None, 'interval_pts': None, 'view': 'front', 'note': ''}
        write(target / 'speed_comparison.json', comparisons)
        write(target / 'contact_review_draft.json', draft)
        summaries.append({'session': session['id'], 'source_hashes': binding['hashes'],
                          'event_count': len(events), 'speed_trend': dashboard['fatigue_and_consistency'],
                          'live_2d_replay': {'frames': len(rows), 'update_max_ms': round(max(timings), 3),
                          'update_mean_ms': round(sum(timings)/len(timings), 3),
                          'scope': 'saved_journal_cpu_update_only_not_live_pipeline_acceptance'},
                          'snapshots': snapshots})
    queue.sort(key=lambda r: (r['priority'] != 'contact_gap', r['session'], r['event_id']))
    with (output / 'contact_review_queue.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(queue[0])); writer.writeheader(); writer.writerows(queue)
    write(output / 'summary.json', {'sessions': summaries, 'source_audit_sha256': digest(audit_path),
          'human_reviewed_events': 0, 'events_awaiting_human_review': len(queue),
          'independent_validation_status': 'awaiting_verified_heldout_sources_and_reference_labels',
          'active_or_exclusive_allowed': False})
    links = ''.join(f"<li>{html.escape(s['session'])} · {s['event_count']}拍 · "
                    f"<a href='{s['session']}/report.html'>重算报告</a> · "
                    f"<a href='{s['session']}/speed_comparison.json'>速度前后对照</a> · "
                    f"<a href='{s['session']}/contact_review_draft.json'>人工草稿</a>" +
                    ''.join(f" · <a href='{s['session']}/{name}'>轨迹预览</a>" for name in s['snapshots']) + '</li>' for s in summaries)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>项目推进复核</title>'
        '<style>body{font:16px/1.8 system-ui;max-width:1000px;margin:32px auto;padding:20px}li{margin:16px 0}</style>'
        '<h1>固定事件重算与人工复核</h1><p>5段 / 1250帧 / 15拍。边界及触球候选保持原值；人工确认待完成。'
        '速度带高度假设，不推断生理疲劳；本次没有重新推理或授予模型替代资格。</p>'
        '<p><a href="contact_review_queue.csv">15拍优先复核清单</a> · '
        '<a href="summary.json">重算摘要</a> · 人工界面：启动 contact_review 服务后访问本机8770端口。</p><ul>'+links+'</ul>', encoding='utf-8')
    print(json.dumps({'output': str(output), 'sessions': len(summaries), 'events': len(queue)}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--audit', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); build(a.audit, a.output)
