"""Replay timing qualification into a new audit; keep historical observations intact."""
import argparse
import hashlib
import html
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinematic_sequence import analyze_kinematic_sequence, POLICY_VERSION
from swing_report_builder import _build_kinematic_sequence_html
from analysis_provenance import analysis_build_info


def main():
    p=argparse.ArgumentParser()
    for name in ('source','manifest','journal','events','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--foot-review-url',default='')
    a=p.parse_args();out=Path(a.output)
    if out.exists():
        raise FileExistsError('Use a new audit revision directory')
    paths={k:Path(getattr(a,k)) for k in ('source','manifest','journal','events')}
    hashes={k:hashlib.sha256(path.read_bytes()).hexdigest() for k,path in paths.items()}
    manifest=json.loads(paths['manifest'].read_text())
    binding=(manifest.get('session',{}).get('ground_calibration_application') or {}).get('input_binding',{})
    if binding.get('kind')!='video_sha256' or binding.get('source_id')!=hashes['source']:
        raise ValueError('Actual input video hash mismatch')
    for key, role in (('journal','frame_journal'),('events','event_snapshot')):
        entries=[x for x in manifest.get('artifacts',[]) if x.get('role')==role]
        if len(entries)!=1 or entries[0].get('sha256')!=hashes[key]:
            raise ValueError('Evidence file does not match manifest: '+key)
    rows=[json.loads(line) for line in paths['journal'].read_text().splitlines() if line.strip()]
    events=json.loads(paths['events'].read_text())['events']
    timestamps=[r['source_time']['timestamp_seconds'] for r in rows]
    dt=[(b-a)*1000 for a,b in zip(timestamps,timestamps[1:])]
    stats={'frame_count':len(rows),'min_interval_ms':min(dt),'median_interval_ms':statistics.median(dt),
           'max_interval_ms':max(dt),'short_interval_source_frames':[r['frame_id'] for r,d in zip(rows[1:],dt) if d<=statistics.median(dt)/4]}
    ffprobe=shutil.which('ffprobe')
    source_clock={'verified_sensor_exposure':False,'ffprobe_available':bool(ffprobe)}
    if ffprobe:
        decoded=json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0','-show_frames',
            '-show_entries','frame=best_effort_timestamp_time','-of','json',str(paths['source'])]))['frames']
        pts=[float(f['best_effort_timestamp_time']) for f in decoded]
        source_clock.update(decoded_frame_count=len(pts),journal_matches_file_pts=
            all(type(r['frame_id']) is int and 0<=r['frame_id']<len(pts) and abs(t-pts[r['frame_id']])<=.000002 for r,t in zip(rows,timestamps)))
    comparisons=[];cards=[]
    for event in events:
        updated=analyze_kinematic_sequence(rows,event['contact_frame'],25)
        old=event['extended_biomechanics']['kinematic_sequence']
        comparisons.append({'event_id':event['event_id'],'contact_frame':event['contact_frame'],
                            'historical':old,'audited':updated})
        cards.append('<section><h2>Event '+html.escape(str(event['event_id']))+'</h2>'
                     +'<p>历史髋 / 肩峰值帧：'+html.escape(str([old.get('hip_peak_frame'),old.get('shoulder_peak_frame')]))
                     +'；新资格结果：'+html.escape(str([updated.get('hip_peak_frame'),updated.get('shoulder_peak_frame')]))+'</p>'
                     +_build_kinematic_sequence_html(updated)+'</section>')
    report={'schema':'tennis.cadence-audit.v1','policy_version':POLICY_VERSION,'inputs_sha256':hashes,
            'analysis_build':analysis_build_info(), 'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_clock':source_clock,'interval_statistics':stats,'comparisons':comparisons,
            'accuracy_validated':False,'new_model_inference':False,'historical_inputs_modified':False}
    out.mkdir(parents=True)
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    foot=('<p><a href="'+html.escape(a.foot_review_url,quote=True)+'">打开连续帧足部接地点辅助复核</a></p>') if a.foot_review_url else ''
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>动力链时间与背面资格审计</title>
<style>body{font:16px/1.7 system-ui;background:#101827;color:#e5eef8;margin:30px auto;max-width:1200px;padding:20px}section{background:#17283a;padding:20px;margin:20px 0}a{color:#76def1}.kinematic-evidence div{margin:8px 0}.kinematic-bar-row{display:flex;gap:20px}.seq-badge{color:#ffc76b}pre{white-space:pre-wrap}</style>
<h1>动力链时间与背面资格审计</h1><p>原始 PTS、观测和历史报告保持不变。本页重放已有证据，无新增模型推理；启发式敏感性检查不证明真实曝光，也不验证技术评分。</p>
<p>短间隔定义为当前窗口中位数的四分之一；排除这些速度样本后，峰位变化超过一个中位采样间隔、峰速变化超过25%、或峰值不可用，则保留原始候选并暂停该段测量。不采用替代峰值恢复结论。参数尚待独立验证。</p>'''
    page+='<pre>'+html.escape(json.dumps(stats,ensure_ascii=False,indent=2))+'</pre>'
    page+='<p>原片 ffprobe 与日志 PTS 对照：'+html.escape(str(source_clock.get('journal_matches_file_pts','未运行')))+'</p>'
    page+=foot+''.join(cards)+'<p><a href="comparison.json">完整对照 JSON（含原始候选与敏感性结果）</a></p></html>'
    (out/'report.html').write_text(page)
    print(json.dumps({'statistics':stats,'clock_check':source_clock,'output':str(out)},ensure_ascii=False))


if __name__=='__main__':main()
