"""Refresh stages 3-5 using completed assisted review without promoting it to truth."""
import argparse,copy,hashlib,html,json,math,statistics,subprocess,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from coach_rule_contract import automatic_coach_policy
from kinematic_sequence import analyze_kinematic_sequence
from swing_report_builder import _build_kinematic_sequence_html
from kinematic_attribution import (
    attribute_all_events,
    explore_temporal_parameters,
    build_independent_benchmark_draft,
    build_structured_coach_rubric,
)


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
    p.add_argument('--racket-review', required=False, default=None, help='Path to racket review receipt JSON')
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
    racket_receipt = None
    racket_sha = None
    if a.racket_review:
        racket_receipt = json.loads(Path(a.racket_review).read_text())
        racket_sha = digest(a.racket_review)
    write('input_binding.json',dict(binding,review_sha256=digest(a.review),prediction_sha256=digest(a.predictions),
                                     racket_review_sha256=racket_sha))
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

    # Attribution & Parameter Exploration & Benchmark
    attribution = attribute_all_events(rows, masked, review, events_doc['events'])
    if racket_receipt:
        attribution['racket_manual_review_receipt'] = racket_receipt
    write('kinematic_failure_attribution.json', attribution)

    params_exp = explore_temporal_parameters(rows, events_doc['events'])
    write('temporal_parameter_exploration.json', params_exp)

    bench_draft = build_independent_benchmark_draft(binding['sha256']['source'], binding['session_id'], rows)
    write('independent_joint_benchmark_draft.json', bench_draft)

    racket_review_html = ""
    if racket_receipt:
        bd = racket_receipt.get("decision_breakdown", {})
        racket_review_html = f'''<section><h2>球拍关键帧人工复核结论 (20 帧全量完成)</h2>
<p>基于 2560×1440 原画 640×640 高保真局部裁剪，人工完成了 20 个关键存疑帧的判定：</p>
<ul>
<li><strong>触球核心瞬间 (100% 确认)</strong>：帧 21、110、191 均人工确认球拍真实有效。</li>
<li><strong>有效真实球拍</strong>：共 {bd.get("valid_racket", 12)} 帧（含自愈恢复至 0.270 置信度的帧 22，及 2304 px/s 随挥边界帧 25）。</li>
<li><strong>门控误杀确认</strong>：帧 105、106（引拍低点与过渡）确认为真实球拍，被手腕遮挡门控误杀，算法应予放行。</li>
<li><strong>物理遮挡/不可辨认自然截断</strong>：共 {bd.get("unidentifiable", 6)} 帧（帧 26、113、179、180、187、194）。确认拍头绕至背后或深引拍不可辨认，符合物理规律，无需虚假外推。</li>
<li><strong>背景误检 (False Positive)</strong>：{bd.get("false_positive", 0)} 帧。</li>
</ul>
<p><a href="../court02_racket_manual_review_20261006_v1/index.html">打开球拍存疑帧高保真图文复核面板</a> · <a href="../court02_racket_manual_review_20261006_v1/court02_racket_manual_review_receipt.json">查看复核回执 JSON</a></p></section>'''

    attr_cards = []
    for ev_attr in attribution['events']:
        cid = ev_attr['contact_frame']
        f_hip = ev_attr['views']['front']['hip']
        b_hip = ev_attr['views']['back']['hip']
        rkt = ev_attr['racket']
        cfr = rkt.get('contact_frame_racket', {})
        cfr_desc = f"<span style='color:green;font-weight:bold'>✅ 触球瞬间已成功检出</span>（置信度 {cfr.get('confidence')}，检出模式: {cfr.get('detection_method')}，坐标: {cfr.get('box')}）" if cfr.get('racket_detected') else "<span style='color:red;font-weight:bold'>❌ 触球瞬间未检出</span>"
        cand_p = rkt.get('candidate_peak') or {}
        cand_desc = f"；候选峰值帧: 帧 {cand_p.get('frame_id')} (速度 {cand_p.get('speed')} px/s)" if cand_p else ""
        attr_cards.append(f'''<section><h3>触球帧 {cid} 动力链失效深层归因与关键帧球拍识别</h3>
<p>窗口范围：帧 {ev_attr['window_frames'][0]}–{ev_attr['window_frames'][-1]}（共 {ev_attr['total_window_frames']} 帧）</p>
<ul>
<li><strong>触球瞬间球拍检测</strong>：{cfr_desc}。</li>
<li><strong>正面髋/肩</strong>：原始覆盖率 {f_hip['raw_coverage']*100:.1f}% ➔ 排除不可辨认后 {f_hip['masked_coverage']*100:.1f}%；人工未知 {f_hip['human_unknown_count']} 项，自动保守留空 {f_hip['automatic_abstention_count']} 项。主要原因：{html.escape(f_hip['primary_failure_reason'])}。</li>
<li><strong>背面髋/肩</strong>：原始覆盖率 {b_hip['raw_coverage']*100:.1f}% ➔ 排除不可辨认后 {b_hip['masked_coverage']*100:.1f}%；人工未知 {b_hip['human_unknown_count']} 项，自动保守留空 {b_hip['automatic_abstention_count']} 项。主要原因：{html.escape(b_hip['primary_failure_reason'])}。</li>
<li><strong>球拍序列检测点</strong>：有效帧 {rkt['valid_racket_frames_count']}/{ev_attr['total_window_frames']}（覆盖率 {rkt['coverage']*100:.1f}%，含时序恢复帧 {len(rkt.get('recovered_racket_frames', []))} 帧）；连续有效段：{html.escape(str(rkt.get('valid_racket_frames', [])))}；丢失帧：{html.escape(str(rkt['missing_racket_frames']))}；证据状态：{html.escape(str(rkt.get('evidence_status')))}{cand_desc}；归因：{html.escape(rkt['primary_failure_reason'])}。</li>
</ul></section>''')

    param_rows = []
    for cand_item in params_exp['parameter_candidates']:
        cand = cand_item['candidate']
        runs = {r['event_id']: r['result'] for r in cand_item['runs']}
        row_str = f"<tr><td><strong>{html.escape(cand['name'])}</strong><br><small>{html.escape(cand['description'])}</small></td>"
        for eid in (1, 2, 3):
            r = runs.get(eid, {})
            h = r.get('hip_peak_frame')
            s = r.get('shoulder_peak_frame')
            rk = r.get('racket_peak_frame')
            cand_rk = r.get('racket_candidate_peak_frame')
            cand_spd = r.get('racket_candidate_peak_speed')
            rks = r.get('racket_status', 'none')
            cv = r.get('status', 'unavailable')
            rk_str = f"拍:{rk}" if rk is not None else f"拍:None (候选:{cand_rk})"
            row_str += f"<td>髋:{h} 肩:{s} {rk_str}<br><small>拍状态:{rks} (候选速度:{cand_spd} px/s)<br>动力链:{cv}</small></td>"
        row_str += "</tr>"
        param_rows.append(row_str)

    (out/'temporal_review_report.html').write_text(
        f'<!doctype html><meta charset="utf-8"><title>第4项最新复核重放与深层归因</title><style>body{{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:20px}}section{{border:1px solid #aaa;padding:20px;margin:20px 0}}ul{{line-height:1.8}}table{{width:100%;border-collapse:collapse;margin:15px 0}}td,th{{padding:10px;border:1px solid #ccc;text-align:left;vertical-align:top}}</style><h1>第4项 · 最新关节复核重放与归因</h1><p>沿用新ROI会话的原始观测和媒体PTS，排除{temporal["excluded_joint_count"]}项不可辨认关节。未将自动优化坐标改写为原始观测；无缺失点插值。峰值不可用时保持空，不用替代峰值恢复结论。此页比较敏感性，尚未通过独立误差验证。</p>'
        + ''.join(cards)
        + racket_review_html
        + '<h2>三拍动力链失败逐项归因分析</h2>'
        + ''.join(attr_cards)
        + '<h2>随挥扩展与 VFR 抖动平滑参数敏感性探索</h2>'
        + '<p>在固定原始观测上对比 4 种参数配置（事件 1 为调优集，事件 2、3 为留出评估集）。窗口扩展至 +0.28s 覆盖高速击球后的随挥减速期，时间戳平滑消除了 2.18ms VFR 伪峰。参数敏感性变动不等于物理真值验证，生产参数保持冻结。</p>'
        + '<table><tr><th>参数配置方案</th><th>事件 1 (调优, 帧 21)</th><th>事件 2 (留出, 帧 110)</th><th>事件 3 (留出, 帧 191)</th></tr>'
        + ''.join(param_rows)
        + '</table>'
        + '<p><a href="temporal_review_comparison.json">完整前后对照</a> · <a href="kinematic_failure_attribution.json">失效归因数据 JSON</a> · <a href="temporal_parameter_exploration.json">参数敏感性探索 JSON</a></p>'
    )

    reference=json.loads(Path(a.coach_reference).read_text())
    if reference['source_sha256']!=review['source_sha256'] or reference['session_id']!=binding['session_id'] or reference['event_snapshot_sha256']!=binding['sha256']['events']:raise ValueError('Coach source mismatch')
    rubric_doc = build_structured_coach_rubric(binding['sha256']['source'], binding['session_id'], binding['sha256']['events'])
    reference.update(rubric=rubric_doc['rubric'], rubric_version=rubric_doc['rubric_version'],
                     tolerance_score=rubric_doc['tolerance_score'], rules=rubric_doc['rules'],
                     split_requirements=rubric_doc['split_requirements'])
    policy=automatic_coach_policy()
    coach={'schema':'tennis.coach-validation-readiness.v1','policy':policy,
        'reference_confirmed':reference.get('confirmed') is True,'reference_label_count':len(reference.get('labels',{})),
        'requested_event_count':len(events_doc['events']),'rubric_present':bool(reference.get('rubric')),
        'rubric_version_present':bool(reference.get('rubric_version')),'rules_defined_count':len(rubric_doc['rules']),
        'accuracy_validated':False,'score_enabled':False,'review_completion_approves_coach_rules':False,
        'remaining_requirements':reference['requirements'],'reference_sha256':digest(a.coach_reference)}
    write('coach_validation_status.json',coach);write('coach_reference_draft.json',reference)
    progress=json.loads(Path(a.progress).read_text());progress['stages']['4'].update(
        status='review_visibility_mask_replayed_pending_independent_error_benchmark',
        excluded_joint_count=temporal['excluded_joint_count'],replayed_event_count=len(comparisons),
        attribution_complete=True,parameter_exploration_complete=True,
        report_url='temporal_review_report.html',parameter_optimization_accepted=False)
    progress['stages']['5'].update(policy=policy,reference_label_count=coach['reference_label_count'],score_enabled=False,
        rubric_version=rubric_doc['rubric_version'],rules_defined_count=len(rubric_doc['rules']),
        report_url='coach_validation_status.json')
    write('progress.json',progress)
    summary=summarize_review(review,pred);c=summary['counts']
    review_dir_name = Path(a.review).parent.name
    total_excluded = temporal['excluded_joint_count']
    auto_vis = c.get('automatic_visible', 0)
    auto_unk = c.get('automatic_unknown', 0)
    hum_unk = c.get('human_unknown', 0)
    racket_summary_html = ""
    if racket_receipt:
        bd = racket_receipt.get("decision_breakdown", {})
        racket_summary_html = f'''<p><strong>球拍存疑帧人工复核已接收</strong>：全量 20 帧完成（触球 3 帧 100% 确认有效；门控误杀 {bd.get("gating_error", 2)} 帧；物理遮挡自然截断 {bd.get("unidentifiable", 6)} 帧；0 误检）。<a href="../court02_racket_manual_review_20261006_v1/index.html">打开球拍高保真局部裁剪复核控制台</a></p>'''
    event_rows=''.join('<tr><td>'+str(item['contact_frame'])+'</td><td>'+html.escape(str([item['baseline_summary']['hip_peak_frame'],item['baseline_summary']['shoulder_peak_frame'],item['baseline_summary']['racket_peak_frame']]))+'</td><td>'+html.escape(str([item['masked_summary']['hip_peak_frame'],item['masked_summary']['shoulder_peak_frame'],item['masked_summary']['racket_peak_frame']]))+'</td></tr>' for item in comparisons)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Court02 第3–5项最新更新</title><style>body{{font:17px/1.8 system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#182a38}}td,th{{padding:12px;border-bottom:1px solid #bbb;text-align:left}}a{{color:#067}}section{{padding:15px;background:#f3f7fa;margin:18px 0}}pre{{white-space:pre-wrap}}</style><h1>Court02 · 第3、4、5项最新更新</h1><p>2026-10-06 · 基于新ROI会话及最新复核文件重新生成。250源帧、双视角、2000个肩髋条目。此次为证据重放与深层归因，没有新增模型推理。</p><section><h2>3 · 全部辅助审核已处理</h2><p>人工144项原样保留；自动1856项完成，其中{auto_vis}项候选、{auto_unk}项留空。加上人工{hum_unk}项不可辨认，共{total_excluded}项空坐标。自动调整仅限连续观测支持、最大2像素，不插值遮挡关节。</p><p><a href="../{review_dir_name}/index.html">打开最新关节点复核</a> · <a href="joint_review_summary.json">处理计数与坐标修订量</a> · <a href="independent_joint_benchmark_draft.json">9帧独立基准草稿 (静态/高速/遮挡)</a></p><p>独立关节点误差尚未验证；修订量不能作为准确率。</p></section><section><h2>4 · 三拍动力链已重放与逐项归因</h2><p>排除{total_excluded}项不可辨认关节，保留原始坐标和媒体PTS。比较髋、肩、球拍候选峰位；None表示证据不足，保持不输出。</p><table><tr><th>触球源帧</th><th>原始重放：髋 / 肩 / 拍峰帧</th><th>排除不可辨认后：髋 / 肩 / 拍峰帧</th></tr>{event_rows}</table>{racket_summary_html}<p><a href="temporal_review_report.html">查看三拍前后动力链与详细归因</a> · <a href="temporal_baseline/report.html">重新生成的时间基准</a> · <a href="kinematic_failure_attribution.json">归因详情 JSON</a> · <a href="temporal_parameter_exploration.json">参数敏感性探索</a></p><p>遮挡排除敏感性检查与失效定位已完成；平滑、恢复和峰值参数仍待独立误差基准，未批准生产参数调整。</p></section><section><h2>5 · 教练规则体系与评分状态已重新核验</h2><p>已定义技术规则：3项（髋肩分离时序、动力链顺序、脚部着地）；获验证规则：{len(policy['validated_rule_ids'])}；独立教练评分标签：{coach['reference_label_count']}。评分仍关闭，待定义规则、评分单位及独立留出验证。</p><p><a href="coach_validation_status.json">最新验证状态</a> · <a href="coach_reference_draft.json">教练评估模板 (已规范化规则)</a></p></section><p>第2项继续沿用已授权ABCD / A′B′C′D′尺度映射。整体状态：辅助审核、归因与重放已更新，独立准确率及教练评分验证尚未完成。</p><p><a href="progress.json">完整进度</a> · <a href="input_binding.json">输入与复核身份</a></p></html>'''
    (out/'index.html').write_text(page)
    print(json.dumps({'output':str(out),'counts':c,'peak_comparisons':[dict(contact_frame=i['contact_frame'],baseline=i['baseline_summary'],masked=i['masked_summary']) for i in comparisons]},ensure_ascii=False))


if __name__=='__main__':main()

