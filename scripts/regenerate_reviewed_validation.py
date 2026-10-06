"""Refresh stages 3-5 using completed assisted review without promoting it to truth."""
import argparse,copy,hashlib,html,json,math,statistics,subprocess,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from coach_rule_contract import automatic_coach_policy
from kinematic_sequence import analyze_kinematic_sequence
from swing_report_builder import _build_kinematic_sequence_html


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask_unknown_joints(rows, review):
    """Remove visibility-uncertain points from a separate replay; never replace raw xy."""
    masked=copy.deepcopy(rows)
    for row in masked:
        fid=row['frame_id']
        for view in ('front','back'):
            pose=row.get('kinematic_views',{}).get(view,{})
            for joint in review['requested_joints']:
                point=review['labels'][f'{fid}:{view}:{joint}']
                if point['visible'] is False:pose.pop(joint,None)
    return masked


def summarize_review(review, predictions):
    counts=Counter();distances={'human':[],'automatic':[]};reasons=Counter()
    for key,p in review['labels'].items():
        actor='automatic' if p.get('review_actor')=='automatic' else 'human'
        counts[actor]+=1;counts[actor+'_visible' if p['visible'] else actor+'_unknown']+=1
        if not p['visible']:
            reasons.update(p.get('abstention_reasons') or [p.get('reason','not_identifiable')]);continue
        fid,view,joint=key.split(':')
        scales=predictions['samples'][fid]
        entries=[value for scale,value in scales.items() if float(scale)==review['prediction_scale']]
        if len(entries)!=1:raise ValueError('Ambiguous/missing prediction scale')
        raw=entries[0][view][joint]
        distances[actor].append(math.hypot(p['x']-raw['x'],p['y']-raw['y']))
    return {'schema':'tennis.review-completion-summary.v1','counts':dict(counts),
        'unknown_reasons':dict(reasons),'displacement_from_original_px':{
            actor:{'count':len(values),'median':statistics.median(values) if values else None,
                   'mean':statistics.mean(values) if values else None,'max':max(values) if values else None}
            for actor,values in distances.items()},
        'independent_accuracy':None,'semantics':'Correction displacement and completion; not independent joint error.'}


def peak_summary(result):
    return {key:result.get(key) for key in ('hip_peak_frame','shoulder_peak_frame','racket_peak_frame',
        'latency_hip_to_shoulder_ms','latency_shoulder_to_racket_ms','cross_validation')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binding','progress','review','predictions','coach-reference','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();binding=json.loads(Path(a.binding).read_text());paths=binding['paths']
    for name,path in paths.items():
        if digest(path)!=binding['sha256'][name]:raise ValueError('Bound input changed: '+name)
    review=json.loads(Path(a.review).read_text());pred=json.loads(Path(a.predictions).read_text())
    if review['source_sha256']!=binding['sha256']['source'] or review['prediction_sha256']!=digest(a.predictions) or pred.get('frame_journal_sha256')!=binding['sha256']['journal']:raise ValueError('Review source mismatch')
    rows=[json.loads(line) for line in Path(paths['journal']).read_text().splitlines() if line.strip()]
    expected={f"{r['frame_id']}:{view}:{joint}" for r in rows for view in ('front','back') for joint in review['requested_joints']}
    if set(review['labels'])!=expected or any(p.get('reviewed') is not True for p in review['labels'].values()):raise ValueError('Complete processed review required')
    if any(row['session_id']!=binding['session_id'] for row in rows):raise ValueError('Session mismatch')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    def write(name,doc):
        (out/name).write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write('input_binding.json',dict(binding,review_sha256=digest(a.review),prediction_sha256=digest(a.predictions)))
    write('joint_review_summary.json',summarize_review(review,pred))
    subprocess.run([sys.executable,str(Path(__file__).with_name('audit_kinematic_cadence.py')),
        '--source',paths['source'],'--manifest',paths['manifest'],'--journal',paths['journal'],
        '--events',paths['events'],'--output',str(out/'temporal_baseline')],check=True)
    masked=mask_unknown_joints(rows,review)
    events_doc=json.loads(Path(paths['events']).read_text())
    if events_doc['session']['session_id']!=binding['session_id']:raise ValueError('Event session mismatch')
    comparisons=[];cards=[]
    for event in events_doc['events']:
        contact=event['contact_frame'];baseline=analyze_kinematic_sequence(rows,contact,25)
        filtered=analyze_kinematic_sequence(masked,contact,25)
        comparisons.append({'event_id':event['event_id'],'contact_frame':contact,
            'raw_observation_replay':baseline,'review_visibility_mask_replay':filtered,
            'baseline_summary':peak_summary(baseline),'masked_summary':peak_summary(filtered)})
        cards.append('<section><h2>触球帧 '+str(contact)+'</h2><h3>原始观测重放</h3>'+_build_kinematic_sequence_html(baseline)
            +'<h3>排除不可辨认关节后的候选重放</h3>'+_build_kinematic_sequence_html(filtered)+'</section>')
    temporal={'schema':'tennis.review-masked-kinematic-comparison.v1','comparisons':comparisons,
        'excluded_joint_count':sum(not p['visible'] for p in review['labels'].values()),
        'review_sha256':digest(a.review),'source_journal_sha256':binding['sha256']['journal'],
        'raw_coordinates_replaced':False,'missing_joints_interpolated':False,'new_model_inference':False,
        'accuracy_validated':False,'parameter_optimization_accepted':False,
        'semantics':'Sensitivity to assisted visibility exclusion; not new independent measurement.'}
    write('temporal_review_comparison.json',temporal)
    (out/'temporal_review_report.html').write_text('<!doctype html><meta charset="utf-8"><title>第4项最新复核重放</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:20px}section{border:1px solid #aaa;padding:20px;margin:20px 0}</style><h1>第4项 · 最新关节复核重放</h1><p>沿用新ROI会话的原始观测和媒体PTS，排除133项不可辨认关节。未将自动优化坐标改写为原始观测；无缺失点插值。峰值不可用时保持空，不用替代峰值恢复结论。此页比较敏感性，尚未通过独立误差验证。</p>'+''.join(cards)+'<a href="temporal_review_comparison.json">完整前后对照</a>')
    reference=json.loads(Path(a.coach_reference).read_text())
    if reference['source_sha256']!=review['source_sha256'] or reference['session_id']!=binding['session_id'] or reference['event_snapshot_sha256']!=binding['sha256']['events']:raise ValueError('Coach source mismatch')
    policy=automatic_coach_policy()
    coach={'schema':'tennis.coach-validation-readiness.v1','policy':policy,
        'reference_confirmed':reference.get('confirmed') is True,'reference_label_count':len(reference.get('labels',{})),
        'requested_event_count':len(events_doc['events']),'rubric_present':bool(reference.get('rubric')),
        'rubric_version_present':bool(reference.get('rubric_version')),'accuracy_validated':False,
        'score_enabled':False,'review_completion_approves_coach_rules':False,
        'remaining_requirements':reference['requirements'],'reference_sha256':digest(a.coach_reference)}
    write('coach_validation_status.json',coach);write('coach_reference_draft.json',reference)
    progress=json.loads(Path(a.progress).read_text());progress['stages']['4'].update(
        status='review_visibility_mask_replayed_pending_independent_error_benchmark',
        excluded_joint_count=temporal['excluded_joint_count'],replayed_event_count=len(comparisons),
        report_url='temporal_review_report.html',parameter_optimization_accepted=False)
    progress['stages']['5'].update(policy=policy,reference_label_count=coach['reference_label_count'],score_enabled=False,
        report_url='coach_validation_status.json')
    write('progress.json',progress)
    summary=summarize_review(review,pred);c=summary['counts']
    event_rows=''.join('<tr><td>'+str(item['contact_frame'])+'</td><td>'+html.escape(str([item['baseline_summary']['hip_peak_frame'],item['baseline_summary']['shoulder_peak_frame'],item['baseline_summary']['racket_peak_frame']]))+'</td><td>'+html.escape(str([item['masked_summary']['hip_peak_frame'],item['masked_summary']['shoulder_peak_frame'],item['masked_summary']['racket_peak_frame']]))+'</td></tr>' for item in comparisons)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Court02 第3–5项最新更新</title><style>body{font:17px/1.8 system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#182a38}td,th{padding:12px;border-bottom:1px solid #bbb;text-align:left}a{color:#067}section{padding:15px;background:#f3f7fa;margin:18px 0}pre{white-space:pre-wrap}</style><h1>Court02 · 第3、4、5项最新更新</h1><p>2026-10-06 · 基于新ROI会话及最新复核文件重新生成。250源帧、双视角、2000个肩髋条目。此次为证据重放，没有新增模型推理。</p><section><h2>3 · 全部辅助审核已处理</h2><p>人工144项原样保留；自动1856项完成，其中1737项候选、119项留空。加上人工14项不可辨认，共133项空坐标。自动调整仅限连续观测支持、最大2像素，不插值遮挡关节。</p><p><a href="../court02_joint_optimization_20261006_v1/index.html">打开最新关节点复核</a> · <a href="joint_review_summary.json">处理计数与坐标修订量</a></p><p>独立关节点误差尚未验证；修订量不能作为准确率。</p></section><section><h2>4 · 三拍动力链已按最新留空结果重放</h2><p>排除133项不可辨认关节，保留原始坐标和媒体PTS。比较髋、肩、球拍候选峰位；None表示证据不足，保持不输出。</p><table><tr><th>触球源帧</th><th>原始重放：髋 / 肩 / 拍峰帧</th><th>排除不可辨认后：髋 / 肩 / 拍峰帧</th></tr>'''+event_rows+'''</table><p><a href="temporal_review_report.html">查看三拍前后动力链</a> · <a href="temporal_baseline/report.html">重新生成的时间基准</a></p><p>遮挡排除敏感性检查已完成；平滑、恢复和峰值参数仍待独立误差基准，未批准生产参数调整。</p></section><section><h2>5 · 教练规则与评分状态已重新核验</h2><p>获验证规则：'''+str(len(policy['validated_rule_ids']))+'；独立教练评分标签：'+str(coach['reference_label_count'])+'''。评分仍关闭，待定义规则、评分单位及独立留出验证。</p><p><a href="coach_validation_status.json">最新验证状态</a> · <a href="coach_reference_draft.json">教练评估模板</a></p></section><p>第2项继续沿用已授权ABCD / A′B′C′D′尺度映射。整体状态：辅助审核与重放已更新，独立准确率及教练评分验证尚未完成。</p><p><a href="progress.json">完整进度</a> · <a href="input_binding.json">输入与复核身份</a></p></html>'''
    (out/'index.html').write_text(page)
    print(json.dumps({'output':str(out),'counts':c,'peak_comparisons':[dict(contact_frame=i['contact_frame'],baseline=i['baseline_summary'],masked=i['masked_summary']) for i in comparisons]},ensure_ascii=False))


if __name__=='__main__':main()
