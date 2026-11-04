"""Summarize saved human decisions and candidate-relative measurements."""
import argparse
import csv
import html
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.serve_joint_racket_2d_review import saved_reviews, review_progress


def build(directory):
    directory=Path(directory).resolve();output=directory.parent
    dataset=json.loads((directory/'dataset.json').read_text())
    reviews,_=saved_reviews(directory,dataset);progress=review_progress(directory,dataset)
    rows=[];links=[]
    for session in dataset['sessions']:
        review=reviews.get(session['id']);frames={f['frame_id']:f for f in session['frames']}
        baseline=json.loads((output/session['id']/'events.json').read_text())['events']
        before={e['event_id']:e for e in baseline};branch={}
        for p in sorted((output/session['id']).glob('human_anchor_*/import_result.json')):
            result=json.loads(p.read_text());receipt=Path(result['receipt'])
            if receipt.is_file() and json.loads(receipt.read_text())['review']==review and result['human_anchor_report']:
                branch={e['event_id']:e for e in json.loads((p.parent/'events.json').read_text())['events']}
                links.append((session['id'],str((p.parent/'report.html').relative_to(output))))
        for e in session['events']:
            eid=e['event_id'];item=review['events'][str(eid)] if review else {'status':'unreviewed'}
            anchor=item.get('anchor_frame') if item['status']=='confirmed' else None
            f=frames.get(anchor);metrics=f['metrics']['front'] if f else {}
            r0=(before[eid].get('extended_biomechanics')or{}).get('racket_head_speed')or{}
            r1=(branch.get(eid,{}).get('extended_biomechanics')or{}).get('racket_head_speed')or{}
            gaps=[g for g in session['statistics']['observation_gaps_ge_3']
                  if g['track'].startswith('front.') and anchor is not None and g['start_frame']<=anchor<=g['end_frame']]
            rows.append({'session_id':session['id'],'event_id':eid,'status':item['status'],
                'model_candidate_frame':e['contact_frame'],'human_anchor_frame':anchor,
                'frame_delta':anchor-e['contact_frame'] if anchor is not None else None,
                'pts_delta_ms':round(1000*(f['source_seconds']-frames[e['contact_frame']]['source_seconds']),3) if f else None,
                'model_anchor_box_speed_kmh':r0.get('contact_kmh'),
                'human_anchor_box_speed_kmh':r1.get('contact_kmh'),
                'human_anchor_box_speed_px_s':r1.get('contact_px_s'),
                'human_anchor_sweet_spot_speed_px_s':metrics.get('sweet_spot_speed_px_s'),
                'human_anchor_wrist_speed_px_s':metrics.get('wrist_speed_px_s'),
                'human_anchor_right_elbow_deg':metrics.get('right_elbow_deg'),
                'human_anchor_ball_sweetspot_distance_px':metrics.get('ball_sweetspot_distance_px'),
                'human_anchor_front_seven_points_available':f['tracks']['front.complete.seven_points'] is not None if f else None,
                'front_gap_tracks_at_human_anchor':[g['track'] for g in gaps],
                'speed_reanalysis_status':'recomputed' if eid in branch else 'not_recomputed',
                'reviewer':review['reviewer'] if review else None,
                'independent_reference':False})
    confirmed=[r for r in rows if r['status']=='confirmed']
    summary={'schema':'tennis.contact-review-progress.v1','progress':progress,'events':rows,
             'model_assisted_comparison':{
                 'confirmed_count':len(confirmed),
                 'exact_frame_agreements':sum(r['frame_delta']==0 for r in confirmed),
                 'within_one_source_frame':sum(abs(r['frame_delta'])<=1 for r in confirmed),
                 'mean_absolute_pts_difference_ms':round(sum(abs(r['pts_delta_ms']) for r in confirmed)/len(confirmed),3) if confirmed else None,
                 'not_independent_accuracy_validation':True},
             'active_or_exclusive_allowed':False}
    (output/'human_review_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (output/'human_review_progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
    for name,selected in [('human_review_metrics.csv',rows),('remaining_contact_reviews.csv',[r for r in rows if r['status']=='unreviewed'])]:
        with (output/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(selected)
    escape=lambda x:html.escape(str(x)) if x is not None else '—'
    trs=''.join('<tr>'+''.join('<td>'+escape(v)+'</td>' for v in
        (r['session_id'],r['event_id'],r['status'],r['model_candidate_frame'],r['human_anchor_frame'],
         r['pts_delta_ms'],r['model_anchor_box_speed_kmh'],r['human_anchor_box_speed_kmh'],
         r['human_anchor_sweet_spot_speed_px_s']))+'</tr>' for r in rows)
    reportlinks=''.join(f'<li>{html.escape(s)}：<a href="{html.escape(p)}">人工锚点重算报告</a></li>' for s,p in links)
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>人工触球进度与指标对照</title><style>body{font:15px/1.7 system-ui;max-width:1280px;margin:25px auto;padding:18px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left}.scroll{overflow:auto}</style>
<h1>人工触球进度与指标对照</h1>'''+f'<p>已保存复核 {progress["reviewed_events"]}/{progress["total_events"]}；确认锚点 {len(confirmed)}。其余保持待审核，不自动填成人工确认。</p>'+'''
<p>人工锚点和原模型候选分别保存。拍框km/h包含高度假设；甜区px/s来自独立七点观测，二者不直接比较大小。“—”表示不可测或未复核。此表不是独立准确率或真实三维拍速验证。</p>
<p><a href="http://127.0.0.1:8771/">继续人工复核</a> · <a href="remaining_contact_reviews.csv">剩余待复核清单</a> · <a href="human_review_metrics.csv">全部指标CSV</a> · <a href="human_review_summary.json">来源与统计JSON</a></p>
<p>复核页选中已完成会话后，可点击“加载本会话已保存标注”恢复审核人、区间及锚点。页面上保存的收据更新后，需重新运行本汇总脚本刷新此静态报告。</p><ul>'''+reportlinks+'''</ul><div class="scroll"><table><thead><tr><th>会话</th><th>拍</th><th>人工状态</th><th>模型帧</th><th>人工帧</th><th>人工−模型 PTS(ms)</th><th>模型锚点拍框 km/h</th><th>人工锚点拍框 km/h</th><th>人工锚点甜区 px/s</th></tr></thead><tbody>'''+trs+'</tbody></table></div>'
    (output/'human_review_summary.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary['model_assisted_comparison'],ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();build(a.directory)
