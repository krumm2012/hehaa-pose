"""Reproducible engineering audit; observation coverage is not detection accuracy."""
import argparse
import hashlib
import html
import importlib.metadata
import json
import platform
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_metric_delivery import event_analysis_metrics, scoring_blockers
from realtime_swing_pipeline import RealtimeSwingEventEngine, RealtimeSwingOutputManager
from session_evidence_bundle import load_evidence_manifest, verify_evidence_manifest


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def distribution(values):
    values = sorted(values)
    if not values: return {'count':0,'p50':None,'p95':None,'max':None}
    return {'count':len(values),'p50':round(statistics.median(values),3),
            'p95':round(values[min(len(values)-1, int((len(values)-1)*.95+.5))],3),
            'max':round(max(values),3)}


def track_coverage(rows, key):
    def observed(row):
        if key == 'ball':
            from ball_observation_contract import measurement_ball
            return measurement_ball(row)[0] is not None
        return any(d.get('observed') is True and d.get('source_frame_id') == row['frame_id'] for d in row.get('rackets') or [])
    valid = [r['frame_id'] for r in rows if observed(r)]
    runs, run = [], []
    for row in rows:
        if observed(row):
            if run: runs.append(run); run=[]
        else: run.append(row['frame_id'])
    if run: runs.append(run)
    return {'observed_count':len(valid),'denominator':len(rows),'source_frames':valid,
            'missing_runs':runs,'coverage':len(valid)/len(rows) if rows else None,
            'definition':('fresh source-matched model ball; legacy rows explicitly unverified' if any('ball_observation' in r for r in rows) else 'legacy selected ball presence; freshness unavailable') if key=='ball' else 'fresh source-matched racket box',
            'false_detection_rate':None,'accuracy_validated':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--synthetic-frames',type=int,default=5000)
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifest=Path(args.manifest).resolve();doc=load_evidence_manifest(str(manifest))
    verification=verify_evidence_manifest(str(manifest))
    if not verification['replayable']: raise ValueError('Evidence integrity verification failed')
    def artifact(role):
        a=next(a for a in doc['artifacts'] if a['role']==role)
        path=Path(a['path']);return path if path.is_absolute() else manifest.parent/path
    frame_path=artifact('frame_journal');event_path=artifact('event_snapshot')
    frames=[json.loads(line) for line in frame_path.read_text().splitlines() if line.strip()]
    old=json.loads(event_path.read_text());replay=doc.get('replay') or {}
    engine=RealtimeSwingEventEngine(fps=replay.get('fps') or 25,
        analysis_interval_frames=replay.get('analysis_interval_frames') or 5,
        settle_frames=replay.get('settle_frames') or 0, window_frames=replay.get('window_frames') or 200,
        session_metadata=doc.get('session'),execution_mode='replay',**(replay.get('swing_options') or {}))
    push_ms=[]
    for frame in frames:
        started=time.perf_counter();engine.push_frame(frame);push_ms.append((time.perf_counter()-started)*1000)
    engine.flush();current=engine.snapshot()
    differences=[]
    for event in current['events']:
        event['analysis_metrics']=event_analysis_metrics(event)
        event['scoring_blockers']=scoring_blockers(event)
        previous=next((e for e in old['events'] if e['event_id']==event['event_id']),{})
        metrics=[]
        for row in event['analysis_metrics']:
            before=next((r for r in event_analysis_metrics(previous) if r['key']==row['key']),{})
            if before.get('value')!=row['value'] or before.get('measurement_evidence')!=row['measurement_evidence']:
                metrics.append({'metric':row['key'],'before':before.get('value'),'after':row['value'],
                                'qualification':row['qualification_status']})
        contact=event['contact_frame']; selected=[r for r in frames if event['start_frame']<=r['frame_id']<=event['end_frame']]
        context=[r for r in frames if contact-15<=r['frame_id']<=contact+4]
        differences.append({'event_id':event['event_id'],'changed_metrics':metrics,
                            'event_observations':{k:track_coverage(selected,k) for k in ('ball','racket')},
                            'contact_observations':{k:track_coverage(context,k) for k in ('ball','racket')}})
    (out/'events.json').write_text(json.dumps(current,ensure_ascii=False,indent=2))
    renderer=RealtimeSwingOutputManager.__new__(RealtimeSwingOutputManager)
    renderer.output_json=out/'events.json';renderer.output_html=out/'report.html'
    renderer.roi_metadata={};renderer.preview_path=None
    t=time.perf_counter();page=renderer._render_live_html(current);render_ms=(time.perf_counter()-t)*1000
    (out/'report.html').write_text(page)
    synthetic=RealtimeSwingEventEngine(fps=25,min_peak_energy=1e9,window_frames=200,analysis_interval_frames=5)
    checkpoints=[];costs=[]
    for i in range(args.synthetic_frames):
        t=time.perf_counter();synthetic.push_frame({'frame_id':i,'timestamp':i/25,'pose':{},'rackets':[]});costs.append((time.perf_counter()-t)*1000)
        if (i+1)%1000==0:checkpoints.append({'processed':i+1,'cached_frames':len(synthetic._frames),'cost_ms':distribution(costs[-1000:])})
    runtime={}
    for binary in ('venv/bin/python','venv_yolo26/bin/python'):
        result=subprocess.run([binary,'-m','pip','freeze'],capture_output=True,text=True,check=True)
        filename=Path(binary).parts[0]+'_requirements.txt';(out/filename).write_text(result.stdout)
        runtime[binary]={'requirements_file':filename,'sha256':digest(out/filename)}
    code={p:digest(p) for p in ('swing_event_analyzer.py','swing_biomechanics.py','kinematic_sequence.py','osd_evidence.py','realtime_swing_pipeline.py','analysis_metric_delivery.py','ball_observation_contract.py','analysis_data_contracts.py','frame_processor.py','main_pipe.py','yolo26n_unified_detector.py')}
    result={'schema':'tennis.engineering-acceptance.v1','accuracy_validated':False,
            'manifest_sha256':digest(manifest),'frame_journal_sha256':digest(frame_path),
            'frame_count':len(frames),'event_count':len(current['events']),'event_comparisons':differences,
            'timing_ms':{key:distribution([r['timing'][key] for r in frames if isinstance(r.get('timing',{}).get(key),(int,float))])
                         for key in ('inference_ms','analysis_ms','capture_to_analysis_ms','render_ms','output_submit_ms','capture_to_output_submit_ms','capture_to_inference_start_ms','inference_to_analysis_start_ms')},
            'replay_push_ms':distribution(push_ms),'report_render_ms':round(render_ms,3),
            'synthetic_long_session':{'frames':args.synthetic_frames,'checkpoints':checkpoints,'scope':'empty observations; cache and repeated scan only; not detector throughput'},
            'environment':{'platform':platform.platform(),'python':sys.version,'runtime':runtime,
                           'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'code_sha256':code},
            'limits':['receipt-to-analysis is not exposure-to-display latency','display FPS and resource baseline need现场 measurement',
                      'ball freshness is unavailable in legacy selected-ball XY','missing detections are not independently labelled false negatives']}
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    original_report=next((a for a in doc['artifacts'] if a['role']=='report'),None)
    report_path=manifest.parent/Path(original_report['path']) if original_report else event_path
    old_link=os.path.relpath(report_path,out).replace(os.sep,'/')
    comparison_rows=[]
    for comparison in differences:
        changes=comparison['changed_metrics']
        for change in changes:
            comparison_rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (
                comparison['event_id'],change['metric'],change['before'],change['after'],change['qualification']))+'</tr>')
    comparison_page='''<!doctype html><meta charset="utf-8"><title>同源事件版本对照</title><style>body{font:16px system-ui;background:#111827;color:#eee;padding:24px}a{color:#67e8f9}td,th{padding:10px;border-bottom:1px solid #334155}table{border-collapse:collapse}</style><h1>同源事件 · 原始／重算对照</h1><p>冻结源帧重算；原文件保留。字段资格变化不表示真实准确性已提高，旧值不能自动升为独立真值。</p>'''
    comparison_page+=f'<p><a href="{html.escape(old_link)}">原记录</a> · <a href="report.html">当前重算报告</a> · <a href="audit.json">逐拍连续性、延迟与版本记录</a></p>'
    comparison_page+='<table><thead><tr><th>事件</th><th>指标</th><th>原值</th><th>重算值</th><th>当前资格</th></tr></thead><tbody>'+''.join(comparison_rows)+'</tbody></table><p>None表示缺失，不补零。完整资格、缺失原因与源帧见events.json。</p>'
    (out/'comparison.html').write_text(comparison_page)
    print(json.dumps({'frames':len(frames),'events':len(current['events']),'replay_push_ms':result['replay_push_ms'],'long_session':checkpoints,'report_render_ms':result['report_render_ms']},ensure_ascii=False))


if __name__=='__main__':main()
