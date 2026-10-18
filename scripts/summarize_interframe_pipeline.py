"""Audit source/semantic parity and render the measured three-profile report."""
import argparse
import html
import json
import tarfile
from pathlib import Path
from statistics import median


PROFILES = {'baseline':'同帧＋图像队列', 'shared':'同帧＋共享源帧', 'interframe':'跨帧＋共享源帧'}
FIELDS = ('source_time','ball','ball_observation','pose_observations','rackets',
          'racket_selection_source','dual_view_biomechanics')


def rows(path):
    path = Path(path)
    if path.is_file():
        content = path.read_text()
    else:
        # The existing acceptance runner losslessly archives repeat > 0.
        archive = path.parent.parent / (path.parent.name+'.tar.gz')
        with tarfile.open(archive,'r:gz') as handle:
            stream = handle.extractfile(path.parent.name+'/'+path.name)
            content = stream.read().decode('utf-8')
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def summarize(root):
    reports = json.loads((root/'matrix.json').read_text())
    groups = {}
    audits = []
    for report in reports:
        group = groups.setdefault((report['source_sha256'],report['repeat']),{})
        if report['profile'] in group:
            raise ValueError('Duplicate source/repeat/profile')
        group[report['profile']] = report
    for (source, repeat), group in groups.items():
        if set(group) != set(PROFILES):
            raise ValueError('Incomplete profile comparison')
        baseline = group['baseline']
        reference = rows(Path(baseline['session'])/'frames.jsonl')
        for profile, report in group.items():
            session = Path(report['session'])
            records = rows(session/'frames.jsonl')
            observations = rows(session/'output_racket_pose.jsonl')
            difference = {key:sum(a.get(key)!=b.get(key) for a,b in zip(reference,records)) for key in FIELDS}
            ids = [r['frame_id'] for r in records]
            expected_transport = 'queue' if profile=='baseline' else 'shared_memory'
            transport_ok = all(r['input_transport']['kind']==expected_transport and
                (expected_transport=='queue' or r['input_transport']['generation']==r['frame_id']) for r in observations)
            audit = {'profile':profile,'source_sha256':source,'repeat':repeat,
                'session':str(session), 'field_differences':difference,
                'source_order_complete':ids==list(range(250)) and [r['frame_id'] for r in observations]==ids,
                'transport_identity_valid':transport_ok,
                'events_equal_to_baseline':report['event_semantics']==baseline['event_semantics'],
                'numerical_failures':sum(r['status']!='observed' for r in observations)}
            audit['passed'] = bool(audit['source_order_complete'] and transport_ok and
                len(reference)==len(records)==len(observations)==250 and not any(difference.values()) and
                audit['events_equal_to_baseline'] and audit['numerical_failures']==0 and
                report['continuous_gate'] and report['exclusive_source_gate'])
            audits.append(audit)
    summary = []
    for profile, label in PROFILES.items():
        selected = [r for r in reports if r['profile']==profile]
        raw = [o for r in selected for o in rows(Path(r['session'])/'output_racket_pose.jsonl')]
        summary.append({'profile':profile,'label':label,'runs':len(selected),
            'frames':sum(r['frames'] for r in selected),
            'fps_median':median(r['output_fps'] for r in selected),
            'fps_min':min(r['output_fps'] for r in selected),
            'fps_max':max(r['output_fps'] for r in selected),
            'dispatch_p50_ms':median(r['inference_stage_ms']['racket_dispatch_ms']['p50'] for r in selected),
            'sidecar_queue_p50_ms':median(o['timing']['queue_wait_ms'] for o in raw),
            'stage_a_p50_ms':median(r['inference_stage_ms']['stage_a_ms']['p50'] for r in selected),
            'stage_b_p50_ms':median(r['inference_stage_ms']['stage_b_ms']['p50'] for r in selected),
            'nominal_input_p95_ms':median(r['nominal_input_to_output_ms']['p95'] for r in selected),
            'target_25fps_passed':all(r['output_fps']>=25 and r['sustained_realtime_gate'] for r in selected)})
    smoke_path = root/'smokes.json'
    smokes = json.loads(smoke_path.read_text()) if smoke_path.exists() else []
    result = {'schema':'tennis.interframe-assessment.v1', 'target_fps':25,
        'method':'One sequential run per profile per source; rotated order; buffered stride 1; startup excluded from output FPS.',
        'scope':'Dual body views, local Coach/events/reports, video encoding and HighGUI completion retained. External network and voice disabled for the main core-chain comparison.',
        'limits':'Three clips, one round; not nine-video three-round acceptance or ten-minute soak. Hardware contention attribution is an inference, not a device-level trace.',
        'default_schedule':'barrier','default_transport':'queue',
        'summary':summary,'audits':audits,'smokes':smokes,
        'parity_passed':bool(audits) and all(a['passed'] for a in audits),
        'production_realtime_accepted':False}
    (root/'assessment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result, reports


def render(root, result, reports):
    header = '<tr><th>视频</th><th>配置</th><th>输出 FPS</th><th>帧数／编码</th><th>延迟 P95¹</th><th>证据</th></tr>'
    cells = []
    for report in sorted(reports,key=lambda r:(r['source_sha256'],list(PROFILES).index(r['profile']))):
        session = Path(report['session'])
        link = html.escape(session.relative_to(root).as_posix(),quote=True)
        video = session.name.split('-s1-')[-1].rsplit('-r',1)[0]
        evidence = (f'<a href="{link}/report.html">事件报告</a>' if (session/'report.html').is_file()
                    else f'<a href="{link}.tar.gz">完整证据归档</a>')
        cells.append(f'<tr><td>{html.escape(video)}</td><td>{PROFILES[report["profile"]]}</td>'
            f'<td>{report["output_fps"]:.2f}</td><td>{report["frames"]} / {report["encoded_frames"]}</td>'
            f'<td>{report["nominal_input_to_output_ms"]["p95"]:.0f} ms</td>'
            f'<td><a href="{link}/measurement.json">测量</a> · {evidence}</td></tr>')
    bars = ''.join(f'<div class="metric"><b>{r["label"]}</b><div class="track"><i style="width:{r["fps_median"]/30*100:.2f}%"></i><span class="target"></span></div><strong>{r["fps_median"]:.2f} FPS</strong></div>' for r in result['summary'])
    stage_rows = ''.join(f'<tr><td>{r["label"]}</td><td>{r["dispatch_p50_ms"]:.3f}</td><td>{r["sidecar_queue_p50_ms"]:.3f}</td><td>{r["stage_a_p50_ms"]:.2f}</td><td>{r["stage_b_p50_ms"]:.2f}</td></tr>' for r in result['summary'])
    smokes = ''.join(f'<li>{html.escape(r["profile"])}：{r.get("frames",0)} 帧，连续完整性 {r.get("continuous_gate",False)}，退出码 {r["returncode"]}</li>' for r in result['smokes'])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>共享源帧与跨帧流水线实测</title>
<style>body{font:16px/1.7 system-ui,sans-serif;color:#172638;background:#f3f5f8;margin:0}main{max-width:1080px;margin:auto;padding:32px 24px}h1{font-size:30px}section{background:white;border:1px solid #dde3ea;border-radius:12px;padding:24px;margin:20px 0}.note{border-left:5px solid #d79422}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:12px 9px;border-bottom:1px solid #e4e8ef}th{background:#f7f9fb}a{color:#145fba}.scroll{overflow:auto}.metric{display:grid;grid-template-columns:190px 1fr 100px;gap:16px;align-items:center;margin:20px 0}.track{height:18px;background:#edf0f4;position:relative;border-radius:4px}.track i{display:block;height:100%;background:#6289b5;border-radius:4px}.target{position:absolute;left:83.333%;top:-5px;height:28px;border-left:2px dashed #ba7021}small{color:#536174}code{background:#eef2f7;padding:2px 5px}pre{white-space:pre-wrap;background:#eef2f7;padding:16px}@media(max-width:650px){.metric{grid-template-columns:1fr 80px}.metric b{grid-column:1 / -1}main{padding:16px}section{padding:16px}}</style><main>
<p>2026-10-18 · 连续逐帧 · 新版七点独占 · 默认配置保留</p><h1>两项方案已接通，25 FPS 尚未通过</h1>
<section class="note"><b>结论</b><p>共享源帧消除了大图序列化并降低派发成本；两阶段跨帧实现了有界重叠，但这轮输出速度仍约 20–21 FPS。没有稳定收益，未提升生产默认。源帧、人体、球、球拍和事件语义的逐帧对照：PARITY。</p></section>
<section><h2>输出吞吐率</h2><p>三段各一次，图中为三段 FPS 中位数；虚线为 25 FPS 目标。启动时间不计入输出 FPS。</p>BARS</section>
<section><h2>同源视频对照</h2><div class="scroll"><table>HEADERROWS</table></div><small>¹ 预热后本地名义输入节奏至输出提交的延迟，包含积压；不是相机曝光至屏幕的实测延迟。</small></section>
<section><h2>阶段耗时（ms）</h2><div class="scroll"><table><tr><th>配置</th><th>派发 P50</th><th>球拍队列 P50²</th><th>准备阶段 P50</th><th>聚合阶段 P50</th></tr>STAGES</table></div><small>除队列取全部记录中位数，其余为三个运行各自 P50 的中位数。² 包含传输与调度，不能解释为纯拷贝成本。原调度准备阶段记 0，聚合阶段包含同帧推理；跨帧两阶段会重叠，不能相加推算吞吐。</small><p>跨帧稳态仍重叠球检测、七点球拍及双人体调用；更换源帧编号没有减少模型工作量。共享加速资源竞争仍是主要解释，但本轮没有设备级执行轨迹证明具体单元的占用比例。</p></section>
<section><h2>完整性与状态</h2><p>三组各 750 帧，共 2250 帧；检查源帧顺序、PTS、球拍传输代际、人体观测、球选择、球拍输出、运动学结果和事件语义。原始记录、模型身份和输入哈希保留于 <a href="matrix.json">matrix.json</a> 和 <a href="assessment.json">assessment.json</a>。</p><ul>SMOKES</ul><p>两帧在途，一项准备结果待取；源槽在 sidecar 完成前不得回收，且检查代际。人体裁剪与所有时序状态保持顺序更新。正常 EOF 排空，错误使会话失败。</p></section>
<section><h2>使用与边界</h2><p>原控制台新增“启动推理调度”和“球拍图像传输”。选择新版七点独占、连续处理后可启用实验配置；旧版仍独立启动。</p><pre>--racket-pose-mode exclusive --processing-mode buffered
--racket-ipc shared_memory --inference-schedule interframe</pre><p>主对照保留本地 Coach、报告、事件片段、编码和显示；关闭了外部网络／语音以测核心链路。故障冒烟只验证旁路失败隔离，不代表真实语音并发性能通过。本轮为三段一次工程回归，未完成九段三轮与十分钟资源验收；没有自动推送或更新 tmp 标签。</p></section></main></html>'''
    replacements = {'PARITY':'通过' if result['parity_passed'] else '未通过', 'BARS':bars,
                    'HEADERROWS':header+''.join(cells), 'STAGES':stage_rows, 'SMOKES':smokes}
    for key,value in replacements.items():
        page = page.replace(key,value)
    (root/'index.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    root = Path(args.directory).resolve()
    result,reports = summarize(root)
    render(root,result,reports)
    print(json.dumps({'parity_passed':result['parity_passed'],'summary':result['summary']},ensure_ascii=False,indent=2))
    if not result['parity_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
