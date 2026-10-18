"""Build a local, auditable acceptance dashboard from immutable run measurements."""
import argparse
import html
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--pipeline-receipt', help='Create a new production runtime receipt after acceptance')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    groups = {}
    for name in ('final-single', 'paced-single', 'optimized-single', 'final-single-s2', 'stride-comparison', 'dual-comparison', 'fault-sidecars','shared-output-smoke-single','shared-output-smoke-single-v2','shared-output-smoke-dual','production-check-single','production-check-dual'):
        groups[name] = [read(p) for p in sorted((root/name).glob('*/measurement.json'))]
    baseline = [r for r in groups['final-single'] if r['mode'] == 'legacy']
    shadow = [r for r in groups['final-single'] if r['mode'] == 'shadow']
    offline = [r for r in groups['final-single'] if r['processing'] == 'offline']
    active = groups['final-single-s2']
    compatibility = []
    for old in baseline:
        other = next((r for r in shadow if r['source_sha256'] == old['source_sha256']), None)
        compatibility.append({'source_sha256': old['source_sha256'], 'same_event_semantics': bool(other and old['event_semantics'] == other['event_semantics'])})
    def aggregate(rows):
        return {'runs': len(rows), 'passed_runs': sum(bool(r.get('realtime_gate')) for r in rows),
                'min_output_fps': min((r.get('output_fps', 0) for r in rows), default=None),
                'max_p95_ms': max((r['timing_ms']['capture_to_output_submit_ms']['p95'] for r in rows if r.get('timing_ms')), default=None),
                'max_drop_rate': max((r.get('drop_rate', 1) for r in rows), default=None)}
    soak = read(root/'soak-final-s2/soak.json')
    numerical=read(root.parents[2]/'models/racket_pose/candidates/bn-recalibrated-v1/acceptance-release.json')
    model_ok=bool(numerical and numerical['numerical_valid'] and numerical['quality_gate_passed'])
    full_ok = len(offline) == 9 and all(r.get('frames') == r.get('unique_source_frames') == r.get('new_same_frame') == 250 and r.get('source_time_valid') and r.get('replay_semantic_match') and r['returncode'] == 0 for r in offline)
    rt_ok = len(active) == 27 and all(r.get('realtime_gate') for r in active)
    compat_ok = len(compatibility) == 9 and all(r['same_event_semantics'] for r in compatibility)
    final_smoke=groups['shared-output-smoke-dual']
    smoke_ok=rt_ok and len(final_smoke)==3 and all(r.get('realtime_gate') for r in final_smoke)
    comparisons=[]
    for run in active:
        old=next((r for r in baseline if r['source_sha256']==run['source_sha256']),None)
        if not old:continue
        pairs=[]
        for event in run['event_semantics']:
            prior=next((e for e in old['event_semantics'] if e['event_id']==event['event_id']),None)
            if prior and isinstance(event.get('contact_frame'),int) and isinstance(prior.get('contact_frame'),int):
                pairs.append({'event_id':event['event_id'],'legacy_contact_candidate_frame':prior['contact_frame'],
                              'new_contact_candidate_frame':event['contact_frame'],'frame_delta':event['contact_frame']-prior['contact_frame']})
        comparisons.append({'session':run['session'],'source_sha256':run['source_sha256'],
                            'legacy_events':len(old['event_semantics']),'new_events':len(run['event_semantics']),
                            'new_fresh_selected_frames':run['new_selected'],'candidate_anchor_comparison':pairs,
                            'scope':'Heuristic event changes, not verified physical contacts or accuracy gains'})
    summary = {'scope': 'Local engineering regression; not independent accuracy or sensor-to-photon latency',
               'default_mode': 'legacy', 'candidate_stride': 2,
               'legacy': aggregate(baseline), 'shadow': aggregate(shadow), 'active': aggregate(active),
               'source_compatibility': compatibility, 'offline_complete': full_ok,
               'same_source_comparisons': comparisons,
               'realtime_passed': rt_ok, 'compatibility_passed': compat_ok,
               'soak_passed': bool(soak and soak['passed']),
               'model_passed': model_ok,
               'final_output_ownership_smoke_passed':smoke_ok,
               'accepted': bool(model_ok and rt_ok and full_ok and compat_ok and smoke_ok and soak and soak['passed']),
               'groups': groups, 'soak': soak}
    production=groups['production-check-single']+groups['production-check-dual']
    summary['production_path_verified']=len(production)==6 and all(r.get('realtime_gate') and (r.get('sidecar') or {}).get('config',{}).get('acceptance_run') is False and (r.get('sidecar') or {}).get('config',{}).get('schedule_source')=='accepted_profile' for r in production)
    (root/'acceptance-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.pipeline_receipt:
        output=Path(args.pipeline_receipt)
        if output.exists():raise FileExistsError(output)
        profiles={}
        for name,runs in [('single',active),('dual',groups['dual-comparison'])]:
            accepted_runs=[];attempts={}
            for stride in (1,2,3):
                candidates=[r for r in runs if r['mode']=='active' and r['processing']=='realtime' and r['stride']==stride]
                attempts[str(stride)]=aggregate(candidates)
                source_repeats={}
                for r in candidates:source_repeats.setdefault(r['source_sha256'],set()).add(r['repeat'])
                if len(candidates)==27 and len(source_repeats)==9 and all(v=={0,1,2} for v in source_repeats.values()) and all(r.get('realtime_gate') for r in candidates):
                    accepted_runs=candidates;break
            passed=bool(model_ok and full_ok and compat_ok and smoke_ok and soak and soak['passed'] and accepted_runs)
            settings=(accepted_runs[0].get('sidecar') or {}).get('config',{}) if accepted_runs else {}
            measured_rates=[(r.get('sidecar') or {}).get('config',{}).get('source_fps',25.0) for r in accepted_runs]
            profiles[name]={'passed':passed,'runs':len(accepted_runs),'source_dimensions':[2560,1440],'max_source_fps':max(25.1,math.ceil(max(measured_rates,default=25.0)*10)/10),
                **{k:settings.get(k) for k in ('inference_stride','imgsz','confidence_threshold','keypoint_threshold','iou_threshold')},
                'measurements':aggregate(accepted_runs),'attempts':attempts}
        receipt={'schema':'tennis.racket-pipeline-acceptance.v1','model_sha256':numerical['model_sha256'],
                 'backend':numerical['backend'],'compute_units':numerical['compute_units'],'profiles':profiles,
                 'soak_passed':bool(soak and soak['passed']),'evidence_root':str(root),'environment':read(root/'environment.json'),
                 'scope':'Local engineering regression with report refresh, clip encoding and local Coach. Optional voice/network failure tests are separate. Not independent accuracy or exposure-to-display latency.'}
        output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    rows = []
    for group, runs in groups.items():
        for r in runs:
            session = Path(r['session'])
            label = session.name
            if (session/'report.html').exists():
                target = session/'report.html'
            elif r.get('verified_archive'):
                target = Path(r['verified_archive'])
            else:
                target = session/'measurement.json'
            href = target.resolve().relative_to(root).as_posix()
            p95 = r.get('timing_ms', {}).get('capture_to_output_submit_ms', {}).get('p95', 0)
            gate = r.get('realtime_gate') if r['processing'] == 'realtime' else r.get('frames') == r.get('new_same_frame') == 250
            historical = group == 'final-single' and r['mode'] == 'active' and r['processing'] == 'realtime'
            status = ('历史：通过' if gate else '历史：未通过') if historical else '通过' if gate else '未通过'
            rows.append(f'<tr data-group="{group}"><td>{html.escape(group)}</td><td><a href="{html.escape(href)}">{html.escape(label)}</a></td><td>{r.get("frames",0)}/250</td><td>{r.get("output_fps",0):.2f}</td><td>{p95:.1f}</td><td>{r.get("drop_rate",1):.1%}</td><td>{r.get("new_selected",0)}</td><td>{status}</td></tr>')
    cards = []
    for name in ('legacy', 'shadow', 'active'):
        a = summary[name]
        cards.append(f'<article><b>{name}</b><strong>{a["passed_runs"]}/{a["runs"]}</strong><span>实时门槛通过</span><p>最低 {a["min_output_fps"] or 0:.2f} FPS · 最差 P95 {a["max_p95_ms"] or 0:.1f} ms</p></article>')
    videos = ''
    for name, p in [('旧版', root/'final-single/realtime-legacy-s1-48.43-r0/output.mp4'), ('增强', root/'final-single-s2/realtime-active-s2-48.43-r0/output.mp4')]:
        if p.exists():videos += f'<figure><figcaption>{name} · 同源 48.43.mp4</figcaption><video controls preload="metadata" src="{p.relative_to(root).as_posix()}"></video></figure>'
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>球拍 Pose 集成验收</title><style>
body{font:15px/1.6 system-ui;background:#f2f5f8;color:#182b40;max-width:1440px;margin:auto;padding:32px}h1{margin:0}a{color:#125cc2}header p{max-width:1050px}.cards,.videos{display:flex;gap:18px;flex-wrap:wrap}article{background:white;border-radius:12px;padding:20px;flex:1;min-width:220px}article strong{display:block;font-size:32px}article span{color:#526678}figure{flex:1;min-width:350px;margin:0}video{width:100%;background:#152538;border-radius:10px}table{width:100%;border-collapse:collapse;background:white;font-size:13px}th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #dce3ea}th{position:sticky;top:0;background:#e7edf5}section{margin:24px 0}.table{overflow:auto;max-height:700px}select{font:inherit;padding:8px;margin:12px 0}.note{border-left:4px solid #d98a0a;background:#fff8e8;padding:12px 18px}</style>
<header><h1>球拍 Pose 渐进集成验收</h1><p>保留 Reader → Inference → Analyzer → Swing Runtime；默认旧版。增强只使用同源帧新观测，超时回退，完整分析失败明确退出。以下是同一批本地素材的工程回归。</p></header>'''
    page += '<section class="cards">'+''.join(cards)+'</section>'
    page += f'<p>三轮实时：{rt_ok}；9 段完整分析：{full_ok}；影子事件语义一致：{compat_ok}；10 分钟资源测试：{summary["soak_passed"]}。</p>'
    page += '<p class="note">原模型保留。候选修复 78 个非有限 BN 统计项，并限制 FP16 解码的极端画外网格偏移。验收不代表独立泛化准确率；二维甜区接近不能证明真实触球或三维拍面。P95 从 Reader 读帧完成计至输出提交，非曝光到显示延迟。</p>'
    page += '<section class="videos">'+videos+'</section>'
    page += '<section><label>运行组 <select id="group"><option value="">全部</option>'+''.join(f'<option>{g}</option>' for g in groups)+'</select></label><p>final-single-s2 为最终单视角每两帧方案三轮；dual-comparison 的 stride3 为最终双视角三轮。optimized-single 与 paced-single 为此前每帧方案记录；更新后的每帧复测出现 16.8% 丢帧，因此未采用该档位。final-single 的实时 active 包含 Reader 修复前失败；旧长测 soak 中 RSS 增长 16.3%，并出现累计报告导致的降速，保留作诊断。历史运行不计入最终通过率。所有运行保留记录、配置和重放证据；部分重复视频副本按 derivative_retention.json 明确省略。</p><div class="table"><table><thead><tr><th>运行组</th><th>运行 / 产物</th><th>源帧</th><th>输出 FPS</th><th>P95 ms</th><th>丢帧</th><th>新观测选中</th><th>结果</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></section>'
    page += '<p><a href="acceptance-summary.json">完整汇总与阶段耗时 JSON</a> · <a href="soak-final-s2/soak.json">最终长测 JSON</a> · <a href="soak/soak.json">最初长测 JSON</a> · <a href="soak-after-history-fix/soak.json">中间版本长测 JSON</a></p><script>document.querySelector("#group").onchange=e=>document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!!e.target.value&&r.dataset.group!==e.target.value)</script></html>'
    (root/'comparison.html').write_text(page)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('groups','soak','source_compatibility')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
