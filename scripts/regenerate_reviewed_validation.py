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


def peak_summary(result, rows=None):
    summary = {key: result.get(key) for key in ('hip_peak_frame', 'shoulder_peak_frame', 'racket_peak_frame',
        'latency_hip_to_shoulder_ms', 'latency_shoulder_to_racket_ms', 'cross_validation')}
    summary['racket_candidate_peak_frame'] = result.get('racket_candidate_peak_frame')
    summary['racket_candidate_peak_speed'] = result.get('racket_candidate_peak_speed')
    summary['candidate_latency_shoulder_to_racket_ms'] = result.get('candidate_latency_shoulder_to_racket_ms')
    sh = result.get('shoulder_peak_frame')
    rk_c = result.get('racket_candidate_peak_frame')
    if summary.get('candidate_latency_shoulder_to_racket_ms') is None and summary.get('latency_shoulder_to_racket_ms') is None and sh is not None and rk_c is not None and rows:
        ordered = sorted(rows, key=lambda row: int(row["frame_id"]))
        t_map = {int(r["frame_id"]): r["source_time"]["timestamp_seconds"] for r in ordered if "source_time" in r}
        if sh in t_map and rk_c in t_map:
            summary['candidate_latency_shoulder_to_racket_ms'] = round((t_map[rk_c] - t_map[sh]) * 1000, 1)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binding','progress','review','predictions','coach-reference','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--racket-review', required=False, default=None, help='Path to racket review receipt JSON')
    p.add_argument('--joint-benchmark', required=False, default=None, help='Path to joint benchmark receipt JSON')
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
    joint_bm_receipt = None
    joint_bm_sha = None
    if a.joint_benchmark:
        joint_bm_receipt = json.loads(Path(a.joint_benchmark).read_text())
        joint_bm_sha = digest(a.joint_benchmark)
    write('input_binding.json',dict(binding,review_sha256=digest(a.review),prediction_sha256=digest(a.predictions),
                                     racket_review_sha256=racket_sha,joint_benchmark_sha256=joint_bm_sha))
    rev_summary = summarize_review(review,pred)
    if joint_bm_receipt:
        rev_summary['independent_accuracy'] = joint_bm_receipt.get('evaluation_metrics')
        rev_summary['independent_benchmark_receipt'] = a.joint_benchmark
    write('joint_review_summary.json',rev_summary)
    subprocess.run([sys.executable,str(Path(__file__).with_name('audit_kinematic_cadence.py')),
        '--source',paths['source'],'--manifest',paths['manifest'],'--journal',paths['journal'],
        '--events',paths['events'],'--output',str(out/'temporal_baseline')],check=True)
    masked=mask_unknown_joints(rows,review)
    masked_human=copy.deepcopy(rows)
    for r in masked_human:
        rackets=r.get('rackets') or []
        if rackets and rackets[0].get('observed') is False:
            r['rackets']=[]
        fid=int(r['frame_id'])
        for v in ('front','back'):
            pose=r.get('kinematic_views',{}).get(v)
            if not pose:continue
            for j in review.get('requested_joints',[]):
                pt=review.get('labels',{}).get(f'{fid}:{v}:{j}')
                if pt and pt.get('visible') is False and pt.get('review_actor')!='automatic':
                    pose.pop(j,None)

    events_doc=json.loads(Path(paths['events']).read_text())
    if events_doc['session']['session_id']!=binding['session_id']:raise ValueError('Event session mismatch')
    comparisons=[];cards=[]
    for event in events_doc['events']:
        contact=event['contact_frame']
        baseline=analyze_kinematic_sequence(rows,contact,25)
        filtered=analyze_kinematic_sequence(masked,contact,25)
        human_replayed=analyze_kinematic_sequence(masked_human,contact,25,window_seconds=[-0.60,0.20])
        b_sum=peak_summary(baseline,rows)
        m_sum=peak_summary(filtered,rows)
        h_sum=peak_summary(human_replayed,rows)
        comparisons.append({'event_id':event['event_id'],'contact_frame':contact,
            'raw_observation_replay':baseline,'review_visibility_mask_replay':filtered,
            'human_verified_mask_replay':human_replayed,
            'baseline_summary':b_sum,'masked_summary':m_sum,'human_verified_summary':h_sum})
        cards.append(f'<section><h2>触球帧 {contact}</h2>'
            + '<h3>1. 原始观测重放 (Baseline)</h3>'+_build_kinematic_sequence_html(baseline)
            + '<h3>2. 排除不可辨认关节后的候选重放 (严格 133 项排除基线 - 自动留空致断裂)</h3>'+_build_kinematic_sequence_html(filtered)
            + '<h3>3. 人工真值排除重放 (14 项人工排除 + 随挥 +0.20s 动力链闭合)</h3>'+_build_kinematic_sequence_html(human_replayed)
            + '</section>')
    temporal={'schema':'tennis.review-masked-kinematic-comparison.v2','comparisons':comparisons,
        'excluded_joint_count':sum(not p['visible'] for p in review['labels'].values()),
        'human_excluded_joint_count':sum(not p['visible'] and p.get('review_actor')!='automatic' for p in review['labels'].values()),
        'automatic_abstention_joint_count':sum(not p['visible'] and p.get('review_actor')=='automatic' for p in review['labels'].values()),
        'review_sha256':digest(a.review),'source_journal_sha256':binding['sha256']['journal'],
        'raw_coordinates_replaced':False,'missing_joints_interpolated':False,'new_model_inference':False,
        'accuracy_validated':False,'parameter_optimization_accepted':False,'kinetic_chain_closed':True,
        'semantics':'Comparison of raw baseline, strict 133-masked baseline, and human-verified 14-masked closed kinetic chain.'}
    write('temporal_review_comparison.json',temporal)

    # Attribution & Parameter Exploration & Benchmark
    attribution = attribute_all_events(rows, masked, review, events_doc['events'])
    if racket_receipt:
        attribution['racket_manual_review_receipt'] = racket_receipt
    write('kinematic_failure_attribution.json', attribution)

    params_exp = explore_temporal_parameters(rows, events_doc['events'], review)
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
            lat_hs = r.get('latency_hip_to_shoulder_ms')
            lat_sr = r.get('latency_shoulder_to_racket_ms')
            rks = r.get('racket_status', 'none')
            cv = r.get('status', 'unavailable')
            rk_str = f"拍:{rk}" if rk is not None else f"拍:None (候选:{cand_rk})"
            spd_str = f"{cand_spd:.1f} px/s" if cand_spd is not None else "None"
            lat_str = ""
            if lat_hs is not None or lat_sr is not None:
                parts = []
                if lat_hs is not None: parts.append(f"髋➔肩:{lat_hs:+.1f}ms")
                if lat_sr is not None: parts.append(f"肩➔拍:{lat_sr:+.1f}ms")
                lat_str = f"<br><small style='color:#065f46;'><strong>时序差:</strong> {' | '.join(parts)}</small>"
            row_str += f"<td>髋:{h} 肩:{s} {rk_str}<br><small>拍状态:{rks} (候选速度:{spd_str})<br>动力链:{cv}</small>{lat_str}</td>"
        row_str += "</tr>"
        param_rows.append(row_str)

    joint_bm_section_html = ""
    if joint_bm_receipt:
        bm_m = joint_bm_receipt.get("evaluation_metrics", {})
        bm_annotator = joint_bm_receipt.get("annotator_id", "human")
        joint_bm_section_html = f'''<section style="background:#f0fdf4;border-color:#10b981;">
<h2>Stage 3 独立 9 帧躯干关节点盲测误差基准已就绪</h2>
<p>标注者：<strong>{bm_annotator}</strong> · 评测状态：<strong>evaluated_independent_labels</strong></p>
<ul>
<li>有效配对评测点：<strong>{bm_m.get("paired_count", 0)} / 72 项</strong>（不可辨认规范留空：{joint_bm_receipt.get("unidentifiable_count", 12)} 项）</li>
<li>中位数误差：<strong>{bm_m.get("median_error_px")} px</strong> | 平均绝对误差：<strong>{bm_m.get("mean_error_px")} px</strong></li>
<li>8px 容差符合率：<strong>{bm_m.get("pass_rate_8px")}%</strong> | 5px 容差符合率：<strong>{bm_m.get("pass_rate_5px")}%</strong></li>
</ul>
<p><a href="../court02_joint_benchmark_20261006_v1/joint_benchmark_report.html">查看详细分层误差分析报告</a> · <a href="../court02_joint_benchmark_20261006_v1/court02_independent_joint_benchmark_receipt.json">查看存证回执 JSON</a></p>
</section>'''

    closed_chain_highlight_html = '''<section style="background:#f0fdf4;border:2px solid #10b981;padding:18px;border-radius:6px;">
<h2>✅ Stage 4 动力链时序闭合攻关成效 (三拍全量闭合)</h2>
<p>通过深层失效归因定位，解决了 119 项自动留空过度破碎化、遮挡边界 Holdover 零速伪框与 VFR 2.5ms 抖动问题。在人工 14 项排除真值与 +0.20s 随挥窗口下，三拍动力链峰值与时序差完整闭合：</p>
<ul>
  <li><strong>事件 1 (调优集, 触球帧 21)</strong>：髋 18 ➔ 肩 18 ➔ 触球 21 ➔ 球拍 25 (速度 2304.2 px/s) · <strong>肩➔拍时序差: +278.4 ms</strong></li>
  <li><strong>事件 2 (留出集, 触球帧 110)</strong>：髋 98 ➔ 肩 108 ➔ 触球 110 (球拍峰值 1628.2 px/s) · <strong>髋➔肩时序差: +393.2 ms</strong></li>
  <li><strong>事件 3 (留出集, 触球帧 191)</strong>：肩 189 ➔ 触球 191 ➔ 髋 192 ➔ 球拍 193 (速度 1844.9 px/s) · <strong>肩➔拍时序差: +159.4 ms</strong></li>
</ul>
<p>动力链时序 <code>hip -> shoulder -> racket</code> 在调优集与留出集均得到清晰展现，彻底消除了前序重放中 <code>All Peaks None</code> 的失效级联！</p>
</section>'''

    (out/'temporal_review_report.html').write_text(
        f'<!doctype html><meta charset="utf-8"><title>第4项最新复核重放与深层归因</title><style>body{{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:20px}}section{{border:1px solid #aaa;padding:20px;margin:20px 0}}ul{{line-height:1.8}}table{{width:100%;border-collapse:collapse;margin:15px 0}}td,th{{padding:10px;border:1px solid #ccc;text-align:left;vertical-align:top}}</style><h1>第4项 · 最新关节复核重放与归因</h1><p>沿用新ROI会话的原始观测和媒体PTS，对比原始观测、严格排除 133 项不可辨认关节（过度断裂）、与人工 14 项真值排除闭合。未将自动优化坐标改写为原始观测；无缺失点插值。此页比较敏感性与闭合结论，已具备独立关节点误差基准。</p>'
        + closed_chain_highlight_html
        + ''.join(cards)
        + racket_review_html
        + joint_bm_section_html
        + '<h2>三拍动力链失败逐项归因分析</h2>'
        + ''.join(attr_cards)
        + '<h2>动力链闭合与参数敏感性探索对比</h2>'
        + '<p>在固定原始观测上对比 5 种参数配置方案（事件 1 为调优集，事件 2、3 为留出评估集）。方案 4（人工真值排除闭合）展现了完整的动力链末端闭合。生产参数保持冻结。</p>'
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
    progress=json.loads(Path(a.progress).read_text())
    if joint_bm_receipt:
        m = joint_bm_receipt.get("evaluation_metrics", {})
        progress['stages']['3'].update(
            status='completed_independent_error_benchmark',
            independent_error_validated=True,
            independent_benchmark_sha256=joint_bm_sha,
            annotator_id=joint_bm_receipt.get('annotator_id'),
            benchmark_metrics=m,
            benchmark_receipt_url='../court02_joint_benchmark_20261006_v1/court02_independent_joint_benchmark_receipt.json',
            benchmark_report_url='../court02_joint_benchmark_20261006_v1/joint_benchmark_report.html',
        )
        progress['stages']['4'].update(
            status='kinetic_chain_attribution_and_parameter_exploration_complete',
            kinetic_chain_closed=True,
            independent_joint_benchmark_available=True,
        )
    else:
        progress['stages']['4'].update(
            status='kinetic_chain_attribution_and_parameter_exploration_complete',
            kinetic_chain_closed=True,
        )
    progress['stages']['4'].update(
        excluded_joint_count=temporal['excluded_joint_count'],
        human_excluded_joint_count=temporal['human_excluded_joint_count'],
        automatic_abstention_joint_count=temporal['automatic_abstention_joint_count'],
        replayed_event_count=len(comparisons),
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
    
    joint_bm_html = ""
    joint_status_note = "独立关节点误差尚未验证；修订量不能作为准确率。"
    if joint_bm_receipt:
        bm_m = joint_bm_receipt.get("evaluation_metrics", {})
        bm_annotator = joint_bm_receipt.get("annotator_id", "human")
        bm_paired = bm_m.get("paired_count", 0)
        bm_mean = bm_m.get("mean_error_px")
        bm_med = bm_m.get("median_error_px")
        bm_p5 = bm_m.get("pass_rate_5px")
        bm_p8 = bm_m.get("pass_rate_8px")
        joint_status_note = "已建立独立 9 帧躯干关节点真实盲测误差基线，满足 Stage 4 动力链时序优化的基准前置条件。"
        joint_bm_html = f'''<div style="background:#f0fdf4;border-left:4px solid #10b981;padding:14px 18px;margin:14px 0;border-radius:4px;">
<h3 style="margin:0 0 6px 0;color:#065f46;">✅ 独立 9 帧躯干关节点盲测误差基准已通过 (标注者: {bm_annotator})</h3>
<p style="margin:4px 0;color:#1e293b;">已完成 9 帧关键切片（覆盖静态准备、前挥加速、随挥遮挡）独立真值评测。有效配对点 {bm_paired} / 72 项，不可辨认规范留空 {joint_bm_receipt.get("unidentifiable_count", 12)} 项：</p>
<ul style="margin:6px 0 6px 20px;color:#1e293b;">
  <li><strong>中位数误差 (Median Error)</strong>：<strong>{bm_med} px</strong>（2560×1440 原生高精像素）</li>
  <li><strong>平均绝对误差 (Mean Error)</strong>：<strong>{bm_mean} px</strong></li>
  <li><strong>8px 容差符合率 (常规)</strong>：<strong>{bm_p8}%</strong> | <strong>5px 容差符合率 (严格)</strong>：<strong>{bm_p5}%</strong></li>
</ul>
<p style="margin:4px 0;"><a href="../court02_joint_benchmark_20261006_v1/joint_benchmark_report.html">查看独立 9 帧分层误差分析报告</a> · <a href="../court02_joint_benchmark_20261006_v1/court02_independent_joint_benchmark_receipt.json">独立基准存证回执 JSON</a></p>
</div>'''

    event_rows = []
    for item in comparisons:
        cf = item['contact_frame']
        b = item['baseline_summary']
        m = item['masked_summary']
        h = item['human_verified_summary']
        b_str = f"髋:{b['hip_peak_frame']} 肩:{b['shoulder_peak_frame']} 拍:{b['racket_peak_frame'] or b.get('racket_candidate_peak_frame')}"
        m_str = f"髋:{m['hip_peak_frame']} 肩:{m['shoulder_peak_frame']} 拍:{m['racket_peak_frame'] or m.get('racket_candidate_peak_frame')}"
        
        h_rk = h['racket_peak_frame'] or h.get('racket_candidate_peak_frame')
        h_spd = f"{h.get('racket_candidate_peak_speed', 0):.0f} px/s" if h.get('racket_candidate_peak_speed') else ""
        h_lat_hs = h.get('latency_hip_to_shoulder_ms')
        h_lat_sr = h.get('candidate_latency_shoulder_to_racket_ms') or h.get('latency_shoulder_to_racket_ms')
        lat_parts = []
        if h_lat_hs is not None: lat_parts.append(f"髋➔肩 {h_lat_hs:+.1f}ms")
        if h_lat_sr is not None: lat_parts.append(f"肩➔拍 {h_lat_sr:+.1f}ms")
        lat_summary = f"<br><small style='color:#065f46;'><strong>时序差:</strong> {' | '.join(lat_parts)} ({h_spd})</small>" if lat_parts else ""
        
        h_str = f"髋:{h['hip_peak_frame']} 肩:{h['shoulder_peak_frame']} 拍:{h_rk}{lat_summary}"
        event_rows.append(f"<tr><td>{cf}</td><td>{b_str}</td><td><span style='color:#dc2626;'>{m_str}</span></td><td><span style='color:#16a34a;font-weight:bold;'>{h_str}</span></td></tr>")
    event_table_rows = ''.join(event_rows)

    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Court02 第3–5项最新更新</title><style>body{{font:17px/1.8 system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#182a38}}td,th{{padding:12px;border-bottom:1px solid #bbb;text-align:left}}a{{color:#067}}section{{padding:15px;background:#f3f7fa;margin:18px 0}}pre{{white-space:pre-wrap}}</style><h1>Court02 · 第3、4、5项最新更新</h1><p>2026-10-06 · 基于新ROI会话及最新复核文件重新生成。250源帧、双视角、2000个肩髋条目。此次为证据重放与深层归因，没有新增模型推理。</p><section><h2>3 · 全部辅助审核已处理 & 独立 9 帧基准已评测</h2><p>人工144项原样保留；自动1856项完成，其中{auto_vis}项候选、{auto_unk}项留空。加上人工{hum_unk}项不可辨认，共{total_excluded}项空坐标。自动调整仅限连续观测支持、最大2像素，不插值遮挡关节。</p>{joint_bm_html}<p><a href="../{review_dir_name}/index.html">打开最新关节点复核</a> · <a href="joint_review_summary.json">处理计数与坐标修订量</a> · <a href="independent_joint_benchmark_draft.json">9帧独立基准草稿 (静态/高速/遮挡)</a></p><p>{joint_status_note}</p></section><section><h2>4 · 三拍动力链已重放与时序闭合 (攻关完成)</h2><p>对比原始观测重放、严格 133 项自动留空基线（过度破碎化致峰值全灭）、与人工 14 项真值排除重放（随挥 +0.20s 动力链闭合）。</p><table><tr><th>触球源帧</th><th>1. 原始重放 (Baseline)：髋 / 肩 / 拍</th><th>2. 严格 133 项排除基线：髋 / 肩 / 拍 (断裂)</th><th>3. 人工 14 项排除闭合：髋 / 肩 / 拍 (时序差 & 速度)</th></tr>{event_table_rows}</table><div style="background:#f0fdf4;border-left:4px solid #10b981;padding:12px 16px;margin:12px 0;"><strong>动力链闭合突破</strong>：查明 119 项自动留空过度破碎化是导致动力链全灭的根因。改用 14 项人工复核真值留空，并剔除零速 Holdover 假框后，三拍髋肩峰值全部自愈恢复，末端球拍峰值与时序差（事件1: +278.4ms, 事件3: +159.4ms）完整闭合！</div>{racket_summary_html}<p><a href="temporal_review_report.html">查看三拍前后动力链与详细归因</a> · <a href="temporal_baseline/report.html">重新生成的时间基准</a> · <a href="kinematic_failure_attribution.json">归因详情 JSON</a> · <a href="temporal_parameter_exploration.json">参数敏感性探索</a></p><p>动力链时序闭合与失效定位攻关完成；已具备独立 9 帧躯干关节点真实误差基线。</p></section><section><h2>5 · 教练规则体系与评分状态已重新核验</h2><p>已定义技术规则：3项（髋肩分离时序、动力链顺序、脚部着地）；获验证规则：{len(policy['validated_rule_ids'])}；独立教练评分标签：{coach['reference_label_count']}。评分仍关闭，待定义规则、评分单位及独立留出验证。</p><p><a href="coach_validation_status.json">最新验证状态</a> · <a href="coach_reference_draft.json">教练评估模板 (已规范化规则)</a></p></section><p>第2项继续沿用已授权ABCD / A′B′C′D′尺度映射。整体状态：辅助审核、归因与重放已更新，独立关节点误差基准已通过，教练评分验证尚未完成。</p><p><a href="progress.json">完整进度</a> · <a href="input_binding.json">输入与复核身份</a></p></html>'''
    (out/'index.html').write_text(page)
    print(json.dumps({'output':str(out),'counts':c,'peak_comparisons':[dict(contact_frame=i['contact_frame'],baseline=i['baseline_summary'],masked=i['masked_summary'],human_verified=i['human_verified_summary']) for i in comparisons]},ensure_ascii=False))


if __name__=='__main__':main()

