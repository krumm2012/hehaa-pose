"""Separate first-25 latency from end-minus-start backlog growth using original clocks."""
import argparse
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.validate_racket_pose_pipeline import distribution


def window(rows):
    return {'frames':len(rows), 'nominal_input_to_output_ms':distribution([
        r['reader_schedule_lag_ms']+r['timing']['capture_to_output_submit_ms'] for r in rows]),
        'capture_to_inference_start_ms':distribution([r['timing']['capture_to_inference_start_ms'] for r in rows]),
        'inference_ms':distribution([r['timing']['inference_ms'] for r in rows]),
        'render_ms':distribution([r['timing']['render_ms'] for r in rows])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    base=ROOT/'data/analysis_results/racket_pose_acceptance_20261018'
    cases=[('release-compatible',base/'render-tail-20261019/after-snapshot'/('buffered-shadow-s1-'+clip+'-r0')/'frames.jsonl')
           for clip in ('48.43','49.43','50.03')]
    cases += [('latest-compatibility',base/'shadow-readiness-20261019/compatibility/buffered-shadow-s1-49.43-r0/frames.jsonl'),
              ('historical-slow',base/'640-tuning-20261019/font-and-preprocess/buffered-shadow-s1-49.43-r0/frames.jsonl')]
    records=[]
    for label,path in cases:
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert [r['frame_id'] for r in rows]==list(range(250))
        assert all(r['source_time']['source_frame_id']==r['frame_id'] and r['source_time']['basis']=='media_pts' for r in rows)
        windows={'first25':window(rows[:25]),'remaining225':window(rows[25:]),'last25':window(rows[-25:])}
        timings=[r['timing']['output_submitted_at_unix_ns'] for r in rows]
        head_wait=[r['timing']['capture_to_inference_start_ms'] for r in rows[:25]]
        peak=int(np.argmax(head_wait))
        settled=next((i for i in range(peak+1,21) if all(v<5 for v in head_wait[i:i+5])),None)
        first_latency=[r['reader_schedule_lag_ms']+r['timing']['capture_to_output_submit_ms'] for r in rows[:25]]
        last_latency=[r['reader_schedule_lag_ms']+r['timing']['capture_to_output_submit_ms'] for r in rows[-25:]]
        record={'label':label,'session':path.parent.name,'source':str(path.relative_to(ROOT)),
            'frames_complete':True,'source_pts_preserved':True,'windows':windows,
            'output_fps':249e9/(timings[-1]-timings[0]),
            'last25_minus_first25_latency_median_ms':float(np.median(last_latency)-np.median(first_latency)),
            'head_wait_settles_below5ms_for5frames_at':settled,
            'first12':[{'frame_id':r['frame_id'],'reader_schedule_lag_ms':r['reader_schedule_lag_ms'],
                'nominal_input_to_output_ms':r['reader_schedule_lag_ms']+r['timing']['capture_to_output_submit_ms'],
                'capture_to_inference_start_ms':r['timing']['capture_to_inference_start_ms'],
                'inference_ms':r['timing']['inference_ms'],'render_ms':r['timing']['render_ms']} for r in rows[:12]]}
        records.append(record)
    decoder=json.loads((args.output/'decoder-probe.json').read_text())
    report={'schema':'tennis.startup-backlog-assessment.v1','date':'2026-10-19',
        'clock_policy':'Original capture timestamps, source PTS and source schedule epoch unchanged; all 250 frames included in output FPS.',
        'conclusion':'Startup has a short first-frame decode/dispatch transient, not sustained first-25 backlog on the measured release-compatible runs.',
        'runtime_change_promoted':False,'decoder_probe_scope':decoder['scope'],'decoder_summary':decoder['summary'],'cases':records}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'assessment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    latest=next(r for r in records if r['label']=='latest-compatibility')
    first,last=latest['windows']['first25'],latest['windows']['remaining225']
    compare=[]
    for label,key in [('名义输入至输出P50','p50'),('名义输入至输出P95','p95'),('名义输入至输出最大值','max')]:
        compare.append('<tr><td>{}</td><td>{:.2f}</td><td>{:.2f}</td></tr>'.format(label,
            first['nominal_input_to_output_ms'][key],last['nominal_input_to_output_ms'][key]))
    trace=''.join('<tr><td>{}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>'.format(
        r['frame_id'],r['reader_schedule_lag_ms'],r['capture_to_inference_start_ms'],r['inference_ms'],r['nominal_input_to_output_ms'])
        for r in latest['first12'])
    cases_html=''.join('<tr><td>{}</td><td>{}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.3f}</td></tr>'.format(
        r['label'],html.escape(r['session']),r['windows']['first25']['nominal_input_to_output_ms']['p95'],
        r['windows']['remaining225']['nominal_input_to_output_ms']['p95'],
        r['last25_minus_first25_latency_median_ms'],r['output_fps']) for r in records)
    probes=''.join('<tr><td>{}</td><td>{}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.2f}</td></tr>'.format(
        label,decoder['summary'][key]['samples'],decoder['summary'][key]['median_first_read_ms'],
        decoder['summary'][key]['median_next_24_mean_ms'],decoder['summary'][key]['median_warmup_extra_ms'])
        for key,label in [('configured','当前FFmpeg自动线程'),('threads2','FFmpeg两线程'),('rewind_warmup','预读后回退到第0帧')])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>首25帧积压诊断</title><style>body{font:16px/1.75 system-ui;max-width:1120px;margin:auto;padding:28px;background:#101824;color:#e2ecf5}
h1,h2{line-height:1.4}.card{padding:20px;background:#1c2d3d;border-radius:12px}a{color:#8acfff}table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:10px;border-bottom:1px solid #3a5066}.scroll{overflow-x:auto}</style>
<h1>首25帧积压诊断 · 2026-10-19</h1><div class="card"><b>当前主要是开头几帧的短暂排队，不是整个首25帧持续积压。</b>
<p>报告原列“末/首25帧积压差”＝末25帧延迟中位数－首25帧延迟中位数。正值表示后段相对前段增长，
不是首25帧的延迟值。列名已改为“末25−首25延迟中位差”。</p></div>
<h2>最近同源shadow运行：49.43 / 250帧</h2><table><tr><th>指标 ms</th><th>首25帧</th><th>其余225帧</th></tr>COMPARE</table>
<p>首25帧P50仅比之后高约2ms；第8帧起，接收到推理启动的间隔连续五帧低于5ms。
这项间隔包含共享复制/IPC及排队，不是纯模型耗时。本轮完整FPS约25.18；没有排除首25帧来计算FPS。</p>
<h2>首12帧：使用原计时起点</h2><table><tr><th>源帧</th><th>Reader排程迟到 ms</th><th>接收→推理启动 ms</th>
<th>推理 ms</th><th>名义输入→输出 ms</th></tr>TRACE</table>
<p>首帧Reader迟到约30.8ms，与独立FFmpeg首读约27ms处在同一量级。随后绝对时间线追赶，第二帧只隔约11ms入队，
在推理前等待约29ms；等待依次消退，约第8帧恢复。此处未重置时间线、未跳帧、未改模型。
Source PTS、接收时间和本地名义输入排程各有含义，均不能作为物理曝光或屏幕呈现时间。</p>
<h2>两个低风险候选的实际排查</h2><table><tr><th>方案</th><th>样本</th><th>首读中位 ms</th>
<th>后24次读取均值的中位 ms</th><th>额外预热中位 ms</th></tr>PROBES</table>
<p>九段视频各两轮，每方案18次，共54组、1350次受比较读取；首25帧的逐帧像素哈希与PTS完全一致。
两线程只减少约1.2ms首读，后续读取更慢；预读后回退额外增加约27.5ms，下一次首读仍约27ms。
这是独立解码实验，含逐帧哈希核对间隔，不代表整链路FPS。两种候选都没有足够证据支持修改默认配置。</p>
<h2>为什么不增加启动缓冲</h2><p>固定启动缓冲可以平滑开头几帧的提交间隔，但只是把等待移到开始前；
不能把它算成吞吐提升，也不能修复长时间低于输入速度的积压。本轮首25帧P95约73ms已低于250ms门槛，
额外缓冲会增加固定延迟，收益有限。若后续需要更平滑的播放，可以单独测量实际播放节奏，
同时报告启动等待与原始端到端延迟；不重新定基准来美化结果。</p>
<h2>历史慢样本与当前实现分开看</h2><div class="scroll"><table><tr><th>范围</th><th>会话</th><th>首25 P95 ms</th>
<th>后225 P95 ms</th><th>末25−首25中位差 ms</th><th>完整FPS</th></tr>CASES</table></div>
<p>历史慢轮的首25帧中位已约285ms，末25帧约824ms，差约539ms；这是起步波动与后续持续积压并存。
单独增加首帧缓冲无法解决。当前发布兼容实现的数据更低，但运行负载不同，不把差异全部归因于某一项改动，
也不据此取代九视频三轮与长时间验收。</p>
<h2>结论与复现</h2><p>保留当前解码和排程默认，不发布无证据的启动优化，不改动已发布tag。
本轮优化报告口径与诊断透明度；实时推理、模型、原时间线和帧完整性保持原样。</p>
<p><a href="assessment.json">窗口测量与首帧轨迹JSON</a> · <a href="decoder-probe.json">九视频解码原始对照</a> ·
<a href="../640-tuning-20261019/index.html">原性能报告（列名已澄清）</a></p>
<p>探针：scripts/probe_decoder_startup.py；报告：scripts/assess_startup_backlog.py。
数据检查覆盖五个历史会话的0–249源帧与媒体PTS，对照中没有修改任何原记录。</p></html>'''
    page=page.replace('COMPARE',''.join(compare)).replace('TRACE',trace).replace('PROBES',probes).replace('CASES',cases_html)
    (args.output/'index.html').write_text(page)
    print(json.dumps({'latest_first25':first,'latest_remaining225':last,'decoder_summary':decoder['summary']},indent=2))


if __name__=='__main__':
    main()
