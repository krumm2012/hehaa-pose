"""Summarize repeated continuous 640 runs without dropping slow trials."""
import argparse
import html
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    runs = json.loads((root / 'font-and-preprocess/matrix.json').read_text())
    smoke = json.loads((root / 'control-smoke.json').read_text())
    summary = {
        'schema': 'tennis.640-tuning.v1',
        'date': '2026-10-19',
        'mode': 'shadow', 'imgsz': 640, 'compute_units': 'ALL',
        'model_sha256': runs[0]['sidecar']['model_sha256'],
        'clips': sorted({Path(r['session']).name.split('-s1-')[1].split('-r')[0] for r in runs}),
        'run_count': len(runs), 'frames': sum(r['frames'] for r in runs),
        'output_fps': {'median': statistics.median(r['output_fps'] for r in runs),
                       'min': min(r['output_fps'] for r in runs),
                       'max': max(r['output_fps'] for r in runs)},
        'continuous_complete': all(r['continuous_gate'] for r in runs),
        'strict_25fps_and_250ms_pass_count': sum(
            r['output_fps'] >= 25 and r['nominal_input_to_output_ms']['p95'] <= 250 for r in runs),
        'stable_25fps_accepted': False, 'quality_accepted': False,
        'default_model_changed': False,
        'scope': 'Three clips x three rounds; paced local input, Court02 ROI, barrier/shared memory, '
                 'dual body/ball/racket, encoding, async display, reports, events, local Coach; '
                 'external network/TTS disabled. Not nine clips x three rounds or a ten-minute soak.',
        'clock': 'FPS=(N-1)/(last-first output submission); startup excluded. P95 includes '
                 'local nominal-input backlog, not camera exposure or physical display latency.',
        'microbench': {
            'font_overlay': {'before_p50_ms': 2.4975, 'after_p50_ms': 1.0836,
                             'before_p95_ms': 3.5844, 'after_p95_ms': 1.3363,
                             'identical_output_sha256': '56d2a665ec66bfbb1b41067c08cd9ef7705bf22359211d533783704d77c7fc71'},
            'ball_preprocess': {'before_p50_ms': 1.3957, 'after_p50_ms': 0.9637,
                                'before_p95_ms': 2.1002, 'after_p95_ms': 1.4383,
                                'identical_output_sha256': '34a56028b41e4c5b83e3d4c2f9f84dd6cd12c225e3c28d9cc4c2c55dcca977d4'},
        },
        'control_smoke': {key: smoke[key] for key in (
            'session', 'startup_model', 'frames', 'output_fps', 'nominal_input_to_output_ms',
            'continuous_frames_complete', 'display_frames_complete', 'replay_semantic_match', 'output_sink')},
        'tests': {'python': 92, 'roi_node_suites': 3},
        'runs': [{key: r[key] for key in ('session', 'repeat', 'source_sha256', 'output_fps',
                  'nominal_input_to_output_ms', 'output_lag_growth_ms', 'continuous_gate',
                  'timing_ms', 'inference_stage_ms', 'output_stage_ms', 'sidecar')} for r in runs],
    }
    (root / 'assessment.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    rows = []
    for r in runs:
        label = Path(r['session']).name.split('-s1-')[1]
        passed = r['output_fps'] >= 25 and r['nominal_input_to_output_ms']['p95'] <= 250
        rows.append('<tr class="{}"><td>{}</td><td>{:.3f}</td><td>{:.1f}</td><td>{:+.1f}</td>'
                    '<td>{:.2f}</td><td>{:.2f}</td><td>250 / 250 / 250</td><td>{}</td></tr>'.format(
                        'pass' if passed else 'fail', html.escape(label), r['output_fps'],
                        r['nominal_input_to_output_ms']['p95'], r['output_lag_growth_ms'],
                        r['timing_ms']['inference_ms']['p50'], r['timing_ms']['render_ms']['p95'],
                        '通过本轮数值门槛' if passed else '未通过'))
    fps = summary['output_fps']
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>640逐帧优化复测</title>
<style>body{font:16px/1.7 system-ui;margin:auto;max-width:1150px;padding:28px;background:#101824;color:#e0e9f3}
h1,h2{line-height:1.3}a{color:#79caff}.card{padding:18px 24px;background:#1a2838;border-radius:12px;margin:18px 0}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #34465a}
.fail td:last-child{color:#ffbf78}.pass td:last-child{color:#90dfb2}img{max-width:100%;border:1px solid #43576e;border-radius:8px}
code{overflow-wrap:anywhere}.scroll{overflow-x:auto}</style>
<h1>640逐帧优化复测 · 2026-10-19</h1>
<div class="card"><b>可在控制台选择640影子预览；稳定25 FPS尚未验收。</b>
<p>三段视频各三轮，输出FPS中位数 MEDIAN，范围 MIN–MAX；严格25 FPS且P95≤250ms共 COUNT/9轮。
全部2250帧按顺序推理、显示和编码，源帧与PTS保留，证据重放通过。慢轮全部保留。</p></div>
<h2>为什么截图不能证明已经使用640</h2><p>“七点独占”选择球拍来源，不选择模型尺寸。
刚才旧会话仍加载960。控制台已增加“启动球拍模型”，选择“640 ALL：性能预览（影子）”后，
启动日志与“最近启动”显示640×640 / ALL。原始640包存在非有限输出，未覆盖、未用于本次预览；
使用单独的bn-repaired-640-v1有界FP16候选。源checkpoint与960修复基线相同。</p>
<p>该候选逐帧运行并渲染七点，事件、球选择和Coach仍采用旧链路；不等于新版独占的完整业务验收。
原始包异常的具体算子来源尚未查明，候选不宣称是坏包的数值等价替换。
默认模型配置未修改，640不能借用960质量收据。</p>
<h2>本轮完成的优化</h2><ul>
<li>字体缓存：共享有界16项字体缓存、复用字体管理器并在启动时预热；并发绘字加锁。中文叠字P50从2.50降至1.08ms。</li>
<li>检测预处理：先缩小BGR再转RGB，避免处理整幅2560×1440颜色通道。P50从1.40降至0.96ms。</li>
<li>两项微基准输出像素哈希前后一致，回归覆盖letterbox/stretch、背景不可变、多行与边缘文字。</li>
<li>显式OpenCV线程设置传递至球拍工作进程。此机OpenCV使用GCD，正整数1/2/4/8仍报告10线程，0报告1；
完整链路测试未发现稳定收益，因此未更改线程默认值。</li></ul>
<p>微基准改善已证实；整链路640原对照中位24.87 FPS，本轮24.90 FPS，波动与长尾仍在，不能把差异全部归因于这些改动。</p>
<h2>逐轮结果（完整编码）</h2><div class="scroll"><table><tr><th>视频 / 轮次</th><th>输出FPS</th>
<th>名义输入→输出P95 ms</th><th>末/首25帧积压差 ms</th><th>推理P50 ms</th><th>渲染P95 ms</th>
<th>处理/显示/编码</th><th>严格门槛</th></tr>ROWS</table></div>
<p>49.43-r0为23.46 FPS、P95 836ms，积压增加539ms。未取得系统级跟踪，不能确定该轮突降的外部原因。
源帧并未丢失；正是持续处理能力不足时等待队列积压的表现。</p>
<h2>剩余瓶颈与下一轮顺序</h2><p>典型整帧推理P50约38–39ms，双路人体阶段约36–37ms，
已占满40ms帧预算的大部分；七点模型独立耗时约10.56ms，并不是现在最慢的阶段。
渲染P95约21–30ms，异步片段与桌面显示并行时还有长尾。阶段存在并发，不能把这些P50直接相加。</p>
<ol><li>优先做球与双人体的资源争用分解：固定输入，比较串行、同帧并行与分组调度；保持全部源帧和原模型，不先增加线程。</li>
<li>跟踪渲染长尾：分开测双视图合成、中文叠字、事件刷新与片段编码；继续减少重复工作，并保留输出能力。</li>
<li>逐帧复核640镜面漏检：旧实验对960参考观测的保留为前景1896/2034、镜面544/733。这不是人工准确率，
在人工复核或受控导出改善前继续影子预览。</li>
<li>以上成立后完成九视频三轮、10分钟循环与故障验收。固定启动缓冲不能消除持续吞吐不足。</li></ol>
<h2>控制台实际启动验证</h2><p>本地49.43、Court02、640 ALL、逐帧影子、共享源帧，250帧全部完成、显示和重放，返回码0。
此轮未保存整段视频，独立列出：输出FPS SMOKE_FPS，名义输入至输出P95 SMOKE_P95 ms；同样未达到稳定25。</p>
<img src="control-panel-640.png" alt="控制台640影子选项与最近启动模型">
<h2>复现与证据</h2><pre>venv_yolo26/bin/python scripts/benchmark_640_optimized.py --repeats 3 --output data/analysis_results/racket_pose_acceptance_20261018/640-tuning-NEW
venv_yolo26/bin/python scripts/report_640_tuning.py data/analysis_results/racket_pose_acceptance_20261018/640-tuning-NEW</pre>
<p>报告生成另需将实际控制台运行的control-smoke.json放入目录；性能矩阵可独立运行。
本轮配置为Court02当前ROI、buffered、stride1、barrier、shared_memory、双人体、球、渲染、VideoToolbox编码、
报告、事件和本地Coach；外部网络与语音旁路未测试。三视频三轮不代表九视频三轮或10分钟压力测试。</p>
<p>FPS取首末输出提交间隔，启动/加载/预热不计入稳定FPS。P95包括本地名义输入排程积压；
不是摄像头曝光延迟，也不是物理屏幕显示延迟。输出视频配置25 FPS不代表实际处理25 FPS。</p>
<p><a href="assessment.json">汇总JSON及模型哈希</a> · <a href="font-and-preprocess/matrix.json">9轮原始测量</a> ·
<a href="control-smoke.json">控制台测量</a> · <a href="control-smoke.log">控制台日志</a> ·
<a href="../resolution-640-20261019/index.html">960/640前轮对照与质量差异</a></p>
<p>92项Python回归、3项ROI前端测试、JS语法与diff检查通过。重复轮产物经逐文件哈希验证后归档保存。</p></html>'''
    replacements = {'MEDIAN': f"{fps['median']:.2f}", 'MIN': f"{fps['min']:.2f}",
                    'MAX': f"{fps['max']:.2f}", 'COUNT': str(summary['strict_25fps_and_250ms_pass_count']),
                    'ROWS': ''.join(rows), 'SMOKE_FPS': f"{smoke['output_fps']:.2f}",
                    'SMOKE_P95': f"{smoke['nominal_input_to_output_ms']['p95']:.1f}"}
    for key, value in replacements.items():
        page = page.replace(key, value)
    (root / 'index.html').write_text(page)
    print(json.dumps({key: summary[key] for key in ('run_count', 'frames', 'output_fps',
          'strict_25fps_and_250ms_pass_count', 'continuous_complete')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
