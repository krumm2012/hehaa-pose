"""Summarize 640 numerical checks, isolated performance and shadow pipeline runs."""
import argparse
import html
import json
from pathlib import Path
import statistics

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    out = args.directory
    s = json.loads((out / 'isolated/summary.json').read_text())
    matrix = json.loads((out / 'pipeline/matrix.json').read_text())
    if len(s['clips']) != 9 or len(matrix) != 9:
        raise ValueError('Expected completed nine-video isolated and three-video/three-profile pipeline matrix')
    parity = json.loads((out / 'parity.json').read_text())
    original = json.loads((out / 'raw-smoke.json').read_text())
    candidate = json.loads((out / 'candidate.json').read_text())
    names = ['960-ANE', '640-ALL', '640-ANE']
    timings = {n: {'wall_ms': [], 'predict_ms': []} for n in names}
    for file in (out / 'isolated').glob('*_frames.jsonl'):
        for line in file.read_text().splitlines():
            row = json.loads(line)
            for n in names:
                v = row['profiles'][n]
                timings[n]['wall_ms'].append(v['wall_ms'])
                timings[n]['predict_ms'].append(v['observation']['timing']['predict_ms'])
    aggregate = {'schema': 'tennis.racket-resolution-assessment.v1', 'isolated': {}, 'pipeline': {},
                 'original_model_rejected': True, 'active_quality_accepted': False,
                 'scope': 'Nine isolated engineering videos; three encoded shadow pipeline videos, one round. No active-mode release or multi-round realtime acceptance.'}
    def stat(values):
        return {'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95))}
    for n in names:
        isolated = {k: stat(v) for k, v in timings[n].items()}
        isolated['frames'] = len(timings[n]['wall_ms'])
        isolated['nonfinite_candidates'] = sum(c['profiles'][n]['nonfinite_candidates'] for c in s['clips'])
        isolated['views'] = {v: {k: sum(c['profiles'][n]['views'][v][k] for c in s['clips']) for k in
                                ('reference_frames', 'retained_frames', 'qualified_selections')} for v in ('front', 'back')}
        runs = [r for r in matrix if r['resolution_profile'] == n]
        aggregate['isolated'][n] = isolated
        aggregate['pipeline'][n] = {'fps_median': statistics.median(r['output_fps'] for r in runs),
                                   'fps_each': [r['output_fps'] for r in runs],
                                   'startup_to_first_output_ms_median': statistics.median(r['startup_to_first_output_ms'] for r in runs),
                                   'complete_each': [r['continuous_gate'] for r in runs],
                                   'ge25_each': [r['output_fps'] >= 25 for r in runs]}
    aggregate['isolated_640_all_wall_reduction_percent'] = (1 - aggregate['isolated']['640-ALL']['wall_ms']['p50'] / aggregate['isolated']['960-ANE']['wall_ms']['p50']) * 100
    (out / 'assessment.json').write_text(json.dumps(aggregate, indent=2))
    snippet = {'racket_pose': {'mode': 'shadow', 'model_path': candidate['candidate_path'], 'backend': 'coreml',
                              'compute_units': 'ALL', 'imgsz': 640, 'confidence_threshold': .25,
                              'keypoint_threshold': .5, 'inference_stride': 1}}
    (out / 'racket-pose-640-shadow-snippet.yaml').write_text(yaml.safe_dump(snippet, sort_keys=False))
    def tr(cells):
        return '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in cells) + '</tr>'
    isolated_rows = ''
    for n in names:
        v = aggregate['isolated'][n]
        isolated_rows += tr([n, f'{v["predict_ms"]["p50"]:.2f}', f'{v["wall_ms"]["p50"]:.2f} / {v["wall_ms"]["p95"]:.2f}',
                             v['frames'], v['nonfinite_candidates'],
                             f'{v["views"]["front"]["retained_frames"]}/{v["views"]["front"]["reference_frames"]}',
                             f'{v["views"]["back"]["retained_frames"]}/{v["views"]["back"]["reference_frames"]}'])
    pipeline_rows = ''
    for r in matrix:
        report = Path(r['session']) / 'report.html'
        pipeline_rows += tr([r['resolution_profile'], r['session'].split('-s1-')[-1].removesuffix('-r0'),
                            f'{r["output_fps"]:.2f}', f'{r["startup_to_first_output_ms"]/1000:.2f}', f'{r["timing_ms"]["inference_ms"]["p50"]:.2f}',
                            f'{r["nominal_input_to_output_ms"]["p95"]:.1f}',
                            f'{r["output_lag_growth_ms"]:.1f}', f'{r["frames"]}/{r.get("encoded_frames")}',
                            r['continuous_gate']])
    max_error = max(r['point_error_p95_input_pixels'] for r in parity['rows'] if r['point_error_p95_input_pixels'] is not None)
    fps = '；'.join(f'{n}：{aggregate["pipeline"][n]["fps_median"]:.2f} FPS' for n in names)
    snippet_html = html.escape((out / 'racket-pose-640-shadow-snippet.yaml').read_text())
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>640球拍模型性能验证</title>
<style>body{{max-width:1200px;margin:36px auto;padding:0 24px;background:#f3f6fa;color:#203040;font:16px/1.8 system-ui}}section{{background:white;border:1px solid #dbe3eb;border-radius:12px;padding:24px;margin:22px 0}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dde6ee}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3f8;padding:18px}}.scroll{{overflow:auto}}code{{overflow-wrap:anywhere}}</style>
<h1>640×640 球拍模型 · 性能验证</h1><p>2026-10-19 · 全图输入，保留七点拓扑、置信度0.25、逐点阈值0.5，不跳帧。</p>
<section><h2>结论</h2><p>原始640包存在非有限输出，不能直接启用。基于既有修复权重另存的640候选完成数值和性能实验：独立球拍每帧耗时降低约 <b>{aggregate['isolated_640_all_wall_reduction_percent']:.1f}%</b>。三段影子链路输出中位数：{fps}。</p><p>影子性能结果不是七点增强质量验收。当前运行默认仍为960，640候选没有质量放行，也没有三轮完整实时验收。</p></section>
<section><h2>用户提供的原始640包</h2><p>输入RGB640×640，输出var_2295:[1,26,8400]。ALL后端抽查49.43.mp4的0、1、12、50、100、249帧，每帧出现1600个非有限候选、8000个非有限值。首帧predict {original['rows'][0]['predict_ms']:.2f} ms，后续帧约9.8–11.5 ms；速度不能抵消数值失败。</p><p>原包未修改，SHA-256：<code>{original['model_sha256']}</code>。尚未将此包的非有限值定位至具体内部算子，也不声称新候选与这个坏包数值等价。</p></section>
<section><h2>独立640候选与同像素对照</h2><p>从960基线使用的同一份修复checkpoint重新导出640，保持已学习参数，沿用有界七点坐标解码。新包：<code>{candidate['candidate_path']}</code>；哈希：<code>{candidate['candidate_sha256']}</code>。没有覆盖原包或借用960质量收据。</p><p>9段×250帧，每种配置都推理全部2250帧；策略执行顺序按帧轮换。对两个后端各9个同像素PyTorch样本进行导出对照，全部有限；各样本P95关键点误差最大为{max_error:.3f}个640输入像素。这是导出一致性，不是检测准确率。</p><div class="scroll"><table><tr><th>配置</th><th>predict P50 ms</th><th>每帧总耗时 P50/P95 ms</th><th>源帧数</th><th>非有限候选</th><th>前景960参考保留</th><th>镜面960参考保留</th></tr>{isolated_rows}</table></div><p>比较使用同源冻结手腕条件；保留指握把≤50原图像素、甜区≤75原图像素且两点合格。缺失参考的帧仍完成推理但不参加手腕比较。640对小尺寸镜面球拍的结果变化较大；这些数值是模型一致性，不能替代人工准确率。</p></section>
<section><h2>整链路影子性能</h2><p>Court02相同ROI、buffered逐帧、barrier、共享源帧IPC。保留球检测、双人体姿态、1080×720渲染、异步显示、VideoToolbox完整视频编码、事件报告、本地Coach及证据重放；测试不调用外部网络和语音。三段按配置轮换运行顺序。模型每帧执行并等待完成，旧链路继续决定事件。</p><div class="scroll"><table><tr><th>配置</th><th>视频</th><th>实际输出FPS</th><th>启动至首输出s</th><th>推理P50 ms</th><th>名义输入→输出P95 ms</th><th>首尾延迟变化ms</th><th>处理/编码帧数</th><th>完整性通过</th></tr>{pipeline_rows}</table></div><p>FPS来自输出提交时间戳，不能用编码标称25FPS代替。启动时间单列，不与稳定吞吐混算。名义输入时钟基于Reader启动后的输入FPS调度，不是摄像头曝光；异步显示返回也不等于物理屏幕呈现。只有一轮三段，未完成9段三轮、10分钟压力及完整增强工作流验收。</p></section>
<section><h2>配置支持与启用边界</h2><p>现已支持640/960显式输入、8400/18900候选解码、对应尺寸预热、模型包输入输出校验，以及验收收据与尺寸绑定。生产仍默认960；启用由mode决定，单写enabled:true不会启动新版。以下是合并到现有完整配置中的影子实验片段，不是独立完整配置文件：</p><pre>{snippet_html}</pre><p>当前候选无质量收据，active/exclusive会拒绝放行；不应伪造quality_gate_passed。仅在质量和实时验收完成后再决定正式切换。</p></section>
<section><h2>审计产物</h2><p><a href="assessment.json">总体指标</a> · <a href="isolated/summary.json">九段单模型明细</a> · <a href="pipeline/matrix.json">三段整链路明细</a> · <a href="parity.json">PyTorch导出对照</a> · <a href="raw-smoke.json">原640包异常输出</a> · <a href="candidate.json">独立候选来源</a> · <a href="racket-pose-640-shadow-snippet.yaml">影子配置片段</a></p><p>Core ML仍有ANE编译告警，完整日志保留。请求ALL/CPU_AND_NE不证明所有算子的实际设备放置；本轮未做设备级追踪。</p></section></html>'''
    (out / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
