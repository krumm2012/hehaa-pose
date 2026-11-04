# -*- coding: utf-8 -*-
"""Build a frozen offline review from the saved audit's five source bindings."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_racket_2d_analysis import analyze_frames, summarize, summarize_event_2d_diagnostics, POLICY


def build(audit_path,output):
    root=Path(__file__).resolve().parents[1]
    audit=json.loads(Path(audit_path).read_text());output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    sessions=[];gap_rows=[]
    for old in audit['sessions']:
        mp=root/old['manifest'];manifest=json.loads(mp.read_text())
        artifacts={a['role']:a for a in manifest['artifacts']}
        paths={role:mp.parent/artifacts[role]['path'] for role in ('frame_journal','event_snapshot','source_video')}
        hashes={role:hashlib.sha256(path.read_bytes()).hexdigest() for role,path in paths.items()}
        if any(hashes[role]!=old['hashes'][role] or hashes[role]!=artifacts[role]['sha256'] for role in hashes):
            raise ValueError('Source identity changed: '+old['session'])
        rows=[json.loads(line) for line in paths['frame_journal'].read_text().splitlines() if line.strip()]
        events=json.loads(paths['event_snapshot'].read_text())['events']
        size=manifest['capture']['resolution'];frames,cadence=analyze_frames(rows,size)
        stats=summarize(frames,events,old.get('source_pts_check',{}).get('decoded_frames'))
        session={'id':old['session'],'source':str(paths['source_video'].resolve()),'hashes':hashes,
                 'manifest':str(mp.resolve()),'image_size':size,'cadence_seconds':cadence,
                 'fps':manifest['capture']['fps'],'frames':frames,'statistics':stats,
                 'events':[{**{k:e.get(k) for k in ('event_id','start_frame','end_frame','contact_frame','stroke_type','contact_status')},
                            'diagnostics_2d':summarize_event_2d_diagnostics(frames,e)} for e in events]}
        sessions.append(session)
        for gap in stats['observation_gaps_ge_3']:gap_rows.append({'session':session['id'],**gap})
    result={'schema':'tennis.joint-racket-2d-review.v1','policy_version':POLICY,'sessions':sessions,
            'association_policy':{'method':'same_frame_wrist_grip_proximity_candidates','keypoint_threshold':.5,'candidate_threshold':.25,'max_distance_image_diagonal':.05,'min_second_candidate_margin_image_diagonal':.005,'person_assignment_verified':False,'automatic_parameters_validated':False},
            'new_inference':False,'accuracy_validated':False,'coach_eligible':False,
            'source_audit_sha256':hashlib.sha256(Path(audit_path).read_bytes()).hexdigest()}
    (output/'dataset.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    with (output/'gaps_ge_3.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['session','track','start_frame','end_frame','count','start_seconds','end_seconds','pts_span_ms','priority','event_ids','near_contact_event_ids','reasons']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in gap_rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    summary={'sessions':[{k:s[k] for k in ('id','hashes','statistics')} for s in sessions],
             'observation_gap_runs_ge_3':len(gap_rows),'source_journal_missing_frames':sum(g['count'] for s in sessions for g in s['statistics']['journal_gaps'])}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    for filename in ('joint_racket_2d_review.html',):
        (output/'index.html').write_text((root/'scripts'/filename).read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps({'output':str(output),'sessions':len(sessions),'gaps_ge_3':len(gap_rows)},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--output',required=True);a=p.parse_args();build(a.audit,a.output)
