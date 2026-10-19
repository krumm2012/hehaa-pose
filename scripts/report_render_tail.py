"""Compare retained render-tail trials, including slow maxima and completeness."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    root=args.directory.resolve()
    phases={name:json.loads((root/name/'matrix.json').read_text())
            for name in ('before','after','after-snapshot')}
    micro={name:json.loads((root/('micro-'+name+'.json')).read_text()) for name in ('before','after')}
    snapshot_micro=json.loads((root/'snapshot-micro.json').read_text())
    parity={key:micro['before']['variants'][key]['output_sha256']==micro['after']['variants'][key]['output_sha256']
            for key in ('plain','coach','card')}
    if not all(parity.values()):
        raise ValueError('Fixed rendering pixels changed')
    rows=[]
    comparisons=[]
    for before,text,snapshot in zip(phases['before'],phases['after'],phases['after-snapshot']):
        if len({r['source_sha256'] for r in (before,text,snapshot)})!=1:
            raise ValueError('Source identity differs')
        clip=Path(before['session']).name.split('-s1-')[1].split('-r')[0]
        item={'clip':clip,'same_event_semantics':before['event_semantics']==text['event_semantics']==snapshot['event_semantics'],
              'measurements':{}}
        for name,r in [('before',before),('after',text),('after-snapshot',snapshot)]:
            g=r['render_groups']
            item['measurements'][name]={'render_p95_ms':g['all']['render_ms']['p95'],
                'render_max_ms':g['all']['render_ms']['max'],'card_p95_ms':g['card']['render_ms'].get('p95'),
                'card_frames':g['card']['frames'],'no_card_p95_ms':g['no_card']['render_ms'].get('p95'),
                'fps':r['output_fps'],'nominal_input_p95_ms':r['nominal_input_to_output_ms']['p95'],
                'continuous_complete':r['continuous_gate'],'session':r['session']}
        comparisons.append(item)
        m=item['measurements']
        rows.append('<tr><td>{}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.1f}%</td><td>{:.2f} → {:.2f}</td></tr>'.format(
            clip,m['before']['render_p95_ms'],m['after']['render_p95_ms'],m['after-snapshot']['render_p95_ms'],
            100*(1-m['after-snapshot']['render_p95_ms']/m['before']['render_p95_ms']),
            m['before']['render_max_ms'],m['after-snapshot']['render_max_ms']))
    summary={'schema':'tennis.render-tail.v1','date':'2026-10-19','fixed_pixels_equal':parity,
        'snapshot_micro':snapshot_micro,
        'phases':list(phases),'comparisons':comparisons,'model_mode':'640 ALL shadow; no active quality promotion',
        'scope':'Three clips, one sequential round per phase; 2250 frames. All abilities retained except external network/TTS bypasses.',
        'stable25_accepted':False,
        'clock':'Render is analysis-complete to render-complete; encoder/display submission measured separately. '
                'Thread CPU excludes library worker threads and does not alone prove GIL contention.'}
    (root/'assessment.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    details=[]
    for phase,runs in phases.items():
        for r in runs:
            label=Path(r['session']).name.split('-s1-')[1]
            g=r['render_groups']
            details.append('<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.2f}</td><td>{:.2f}</td>'
                           '<td>{:.3f}</td><td>{:.1f}</td><td>{}</td></tr>'.format(
                phase,label,g['card']['frames'],g['card']['render_ms']['p95'],g['no_card']['render_ms']['p95'],
                r['output_fps'],r['nominal_input_to_output_ms']['p95'],'250/250/250' if r['continuous_gate'] else '未通过'))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>渲染长尾诊断与两轮优化</title><style>body{font:16px/1.7 system-ui;background:#101824;color:#dfebf6;max-width:1120px;margin:auto;padding:28px}
h1,h2{line-height:1.4}.card{background:#1b2c3d;padding:20px;border-radius:12px;margin:20px 0}a{color:#81caff}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #3c5065}
.scroll{overflow-x:auto}code,pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>
<h1>渲染长尾诊断与两轮优化 · 2026-10-19</h1>
<div class="card"><b>渲染长尾可以优化，除了绘字，还有每5帧一次的快照重复序列化。</b>
<p>先减少文字处理中的整幅画面拷贝，再复用逐帧JSONL已编码的记录。保持所有源帧、文字内容、隐私遮挡、
报告、片段和编码输出，未通过关闭输出或降低刷新频率换取性能。</p></div>
<h2>三阶段对比：整段渲染P95（ms）</h2><div class="scroll"><table><tr><th>视频</th><th>优化前</th><th>文字优化后</th>
<th>再优化快照后</th><th>P95降低</th><th>最大单帧：前→后 ms</th></tr>ROWS</table></div>
<p>文字优化降低了三段P95；快照复用这一轮主要压低最大尖峰，未进一步改善P95，50.03的有卡片P95反而上升。
这些是单轮观察，不能宣称所有分位数稳定改善。慢轮完整保留。</p>
<h2>定位证据</h2><p>固定真实视频首帧、mock人体与同一卡片内容，无模型推理：普通画面P95约0.80ms，
教练文字约1.85ms，遥测卡片约6.44ms；卡片文字调用约10次。文字批量绘制、中文只转换墨迹覆盖区域后，
卡片P95约4.03ms，降低37%。三种固定输出像素哈希全部一致，保持绘制顺序、字号、位置和颜色。</p>
<p>但这不能解释26–37ms的单帧尖峰。前一轮慢帧常出现在5帧边界，例如240、160、245；
视图处理墙钟可达11–17ms，而当前线程CPU仅约1–3ms。该差值提示并发等待或库内部工作，单独不能证明GIL原因。</p>
<p>逐帧日志线程每5帧将最近200条嵌套记录重新json.dumps一次，真实快照约2.9MiB。
固定数据微基准P50约SNAP_FULLms；复用同一条JSONL编码后的组装约SNAP_REUSEms（仅序列化/组装，不含文件I/O）。
该重复编码与渲染同处Analyzer进程。第二轮只改变快照复用策略，再以整链路结果检验这一假设。</p>
<h2>改动与约束</h2><ul><li>字体：单次批量拥有一幅副本，中文仅在裁剪墨迹区域执行BGR/RGB转换。
保留混合英文/中文顺序、多行、负坐标、越界文字与并发缓存字体的行为。</li>
<li>快照：每条记录只编码一次，JSONL与最近帧快照复用相同片段；有界保存200条编码记录。
仍每5帧刷新、关闭时发布最终快照，保留原子临时文件替换和全部嵌套字段。</li>
<li>增加逐帧render_stage_ms和render_features：球拍叠加、裁剪、视图、HUD、遥测卡片；视图/HUD/卡片同时记录墙钟与线程CPU。
原render_ms不含后续编码和显示提交，输出端耗时继续单列。未把异步等待误算成模型耗时。</li></ul>
<h2>完整链路与边界</h2><div class="scroll"><table><tr><th>阶段</th><th>视频</th><th>卡片帧数</th><th>有卡片P95 ms</th>
<th>无卡片P95 ms</th><th>输出FPS</th><th>名义输入→输出P95 ms</th><th>处理/显示/编码</th></tr>DETAILS</table></div>
<p>三段视频每阶段各一轮，源视频哈希和Court02 ROI固定；640 ALL影子、buffered、stride1、barrier、共享源帧，
保留双人体、球、七点、异步显示、VideoToolbox编码、事件、本地Coach、证据与重放。前后事件语义见JSON。
卡片出现时机受异步事件发布影响，表中列出实际帧数。总2250帧。</p>
<p>当前新鲜基线本身约25.14–25.17FPS，与此前多轮的24.90中位数不同，因此不能把全部运行差异归因于代码。
本轮主要检验渲染余量；单轮三视频不替代九视频三轮、10分钟循环和故障验收，不更新640质量收据。
源视频按约25FPS节流，优化后的FPS可能不明显增加。固定启动缓冲无法消除持续计算能力不足。</p>
<h2>后续优先级</h2><p>优先看新阶段日志中仍然存在的峰值，再针对后台输出和模型资源争用做同负载对照。
无需重建渲染工作流，也无需降低画面内容。若考虑再减少视图副本，需要继续验证隐私遮挡、镜面坐标和源图不可变。</p>
<h2>证据与复现</h2><p><a href="assessment.json">汇总JSON</a> · <a href="before/matrix.json">优化前全记录</a> ·
<a href="after/matrix.json">文字优化全记录</a> · <a href="after-snapshot/matrix.json">快照优化全记录</a> ·
<a href="micro-before.json">固定渲染前</a> · <a href="micro-after.json">固定渲染后</a> ·
<a href="snapshot-micro.json">快照序列化微基准</a> · <a href="tests.log">回归日志</a></p>
<pre>venv_yolo26/bin/python scripts/benchmark_render_tail.py --output data/analysis_results/racket_pose_acceptance_20261018/render-tail-NEW
venv_yolo26/bin/python -m unittest -q test_render_text_batch test_frame_snapshot_hotpath test_realtime_swing_pipeline</pre>
<p>88项相关Python回归通过；覆盖文字像素一致性、快照字段/刷新频率/有界窗口/所有权、完整性和已有业务路径。
所有慢帧及阶段时间均保留在各matrix.json的slow_render_frames中；阶段存在并发，不能简单相加分位数。</p></html>'''
    page=page.replace('ROWS',''.join(rows)).replace('DETAILS',''.join(details))
    page=page.replace('SNAP_FULL',f"{snapshot_micro['full']['p50_ms']:.2f}").replace('SNAP_REUSE',f"{snapshot_micro['reuse']['p50_ms']:.2f}")
    (root/'index.html').write_text(page)
    print(json.dumps(comparisons,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
