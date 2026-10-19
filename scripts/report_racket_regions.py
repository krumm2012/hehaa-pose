"""Render measured crop comparisons and source-frame review panels."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from racket_pose_detector import draw_racket_pose


def statistics(v):
    return {name: float(np.percentile(v, q)) for name, q in (('p50', 50), ('p95', 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--video-dir', type=Path, default=Path('/Users/krum5539/Desktop/Camera/2026-09-30'))
    args = parser.parse_args()
    out = args.directory
    s = json.loads((out / 'summary.json').read_text())
    policies = ['full', 'union', 'split']
    timing = {p: {'wall_ms': [], 'predict_ms': [], 'preprocess_ms': []} for p in policies}
    cases = []
    csv_rows = []
    for clip in s['clips']:
        records = [json.loads(l) for l in (out / (Path(clip['clip']).stem + '_frames.jsonl')).read_text().splitlines()]
        selected_cases = set()
        for view in ('front', 'back'):
            lost = clip['policies']['union']['views'][view]['not_retained_frame_ids']
            added = clip['policies']['union']['views'][view]['additional_selection_frame_ids']
            selected_cases.update((fid, view) for fid in lost[:1] + added[:1])
        for r in records:
            for p in policies:
                for key in timing[p]:
                    timing[p][key].append(r['policies'][p][key])
                csv_rows.append([clip['clip'], r['frame_id'], p, r['has_body_reference'],
                                 *[r['policies'][p][k] for k in timing[p]]])
        cap = cv2.VideoCapture(str(args.video_dir / clip['clip']))
        try:
            for fid, view in sorted(selected_cases):
                row = records[fid]
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ok, frame = cap.read()
                if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != fid + 1:
                    raise ValueError('Review frame seek failed')
                candidates = [row['policies'][p]['selected'].get(view) for p in policies]
                boxes = [c['box'] for c in candidates if c]
                if not boxes:
                    continue
                x1, y1 = max(0, int(min(b[0] for b in boxes)) - 100), max(0, int(min(b[1] for b in boxes)) - 100)
                x2, y2 = min(frame.shape[1], int(max(b[2] for b in boxes)) + 100), min(frame.shape[0], int(max(b[3] for b in boxes)) + 100)
                panels = []
                for p, c in zip(policies, candidates):
                    image = draw_racket_pose(frame.copy(), {'status': 'observed', 'candidates': [c] if c else []})
                    if c:
                        a, b, right, bottom = map(round, c['box'])
                        cv2.rectangle(image, (a, b), (right, bottom), (0, 255, 0), 2)
                    image = image[y1:y2, x1:x2]
                    scale = min(390 / image.shape[1], 340 / image.shape[0])
                    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
                    panel = np.full((390, 400, 3), 24, np.uint8)
                    py, px = 45 + (340 - resized.shape[0]) // 2, (400 - resized.shape[1]) // 2
                    panel[py:py + resized.shape[0], px:px + resized.shape[1]] = resized
                    cv2.putText(panel, p + (' selected' if c else ' NO SELECTION'), (10, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (230, 230, 230), 1, cv2.LINE_AA)
                    panels.append(panel)
                name = f'{Path(clip["clip"]).stem}-{fid}-{view}.jpg'
                cv2.imwrite(str(out / name), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 88])
                cases.append({'clip': clip['clip'], 'frame_id': fid, 'view': view, 'image': name})
        finally:
            cap.release()
    aggregate = {'frames': sum(c['frames'] for c in s['clips']), 'videos': len(s['clips']), 'policies': {}}
    for p in policies:
        v = {k: statistics(vals) for k, vals in timing[p].items()}
        v['nonfinite_candidates'] = sum(c['policies'][p]['nonfinite_candidates'] for c in s['clips'])
        v['views'] = {}
        for view in ('front', 'back'):
            sums = {key: sum(c['policies'][p]['views'][view][key] for c in s['clips']) for key in
                    ('full_qualified_reference_frames', 'retained_reference_frames', 'qualified_selection_frames',
                     'full_reference_points', 'full_reference_points_outside_crop')}
            sums['additional_selection_frames'] = sum(len(c['policies'][p]['views'][view]['additional_selection_frame_ids']) for c in s['clips'])
            v['views'][view] = sums
        aggregate['policies'][p] = v
    (out / 'aggregate.json').write_text(json.dumps(aggregate, indent=2))
    with (out / 'timings.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip', 'frame_id', 'policy', 'has_body_reference', 'wall_ms', 'predict_ms', 'preprocess_ms'])
        writer.writerows(csv_rows)
    def table_row(values):
        return '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in values) + '</tr>'
    totals = ''
    per_clip = ''
    for p, label in zip(policies, ['全图', '扩大后的双视角并集', '扩大后的分视角（两次）']):
        v = aggregate['policies'][p]
        front, back = v['views']['front'], v['views']['back']
        totals += table_row([label, f'{v["wall_ms"]["p50"]:.2f} / {v["wall_ms"]["p95"]:.2f}',
                             f'{v["predict_ms"]["p50"]:.2f}',
                             f'{front["retained_reference_frames"]}/{front["full_qualified_reference_frames"]}',
                             f'{back["retained_reference_frames"]}/{back["full_qualified_reference_frames"]}',
                             f'{front["additional_selection_frames"]} / {back["additional_selection_frames"]}'])
    for c in s['clips']:
        for p in policies:
            v = c['policies'][p]
            per_clip += table_row([c['clip'], p, c['frames'], f'{v["wall_ms"]["p50"]:.2f}',
                                  str(v['views']['front']['not_retained_frame_ids']), str(v['views']['back']['not_retained_frame_ids'])])
    gallery = ''.join(f'<figure><figcaption>{html.escape(c["clip"])} · 源帧 {c["frame_id"]} · {c["view"]}，各栏为相同原图区域</figcaption><img loading="lazy" src="{c["image"]}"></figure>' for c in cases)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>球拍区域逐帧对照</title>
<style>body{{max-width:1200px;margin:36px auto;padding:0 24px;background:#f3f6f9;color:#203040;font:16px/1.8 system-ui}}section{{background:white;padding:22px;border:1px solid #dae2ea;border-radius:10px;margin:20px 0}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;border-bottom:1px solid #dbe4ec;padding:9px;overflow-wrap:anywhere}}.scroll{{overflow:auto}}img{{width:100%;height:auto}}figure{{margin:25px 0}}code{{overflow-wrap:anywhere}}</style>
<h1>球拍检测区域 · 逐帧工程对照</h1><p>{aggregate['videos']} 段本地视频，{aggregate['frames']} 个源帧；每个源帧都测试三种策略，未采样跳帧。</p>
<section><h2>结论：保留全图默认，裁剪方案作为实验候选</h2><p>扩大后的并集减少原图取样区域，但模型仍使用960×960固定输入，未证明推理加速；两路分别推理需要两次调用。裁剪还改变了部分原有观测，并新增部分选择，不能把新增数量当作准确率提升。尚不满足替换默认及实时25 FPS验收条件。</p></section>
<section><h2>区域设计</h2><p>冻结当前Court02机位配置；对front_view.default_roi和mirror_view.reflection_roi各向外增加{ s['margin_source_pixels'] }个原图像素，然后取并集或分别推理。区域不依赖当前人体结果，不新增人体等待屏障；球拍保持原图方向，不直接复用人体540×720缩放或镜像翻转后的画面。</p><p>全图[0,0,2560,1440]，2560×1440；并集[882,0,2329,1440]，1447×1440（原图面积56.5%）；分视角：正面[896,0,2176,1440]，1280×1440；镜面[882,0,2329,721]，1447×721。所有裁剪分别letterbox至960×960，框和七点还原到原图后再比较。</p></section>
<section><h2>总体测量</h2><div class="scroll"><table><tr><th>策略</th><th>每帧总耗时 P50/P95 ms</th><th>predict P50 ms</th><th>前景基线保留</th><th>镜面基线保留</th><th>新增前景/镜面选择</th></tr>{totals}</table></div><p>耗时包括该策略全部球拍调用、预后处理、坐标还原及跨区域NMS。轮换每帧执行顺序。这里没有并发运行球、人体、渲染、IPC和编码，因此不是整链路FPS，也不是25 FPS验收。</p><p>“保留”指同一份合格手腕条件下，相对全图观测的握把点距离≤50像素、甜区距离≤75像素，且两点均合格；这是模型一致性，非人工准确率。新增或不一致帧需要人工核实。历史人体记录未覆盖的帧仍完成三种球拍推理，但不进入手腕关联比较。</p></section>
<section><h2>逐视频差异</h2><div class="scroll"><table><tr><th>视频</th><th>策略</th><th>源帧数</th><th>耗时 P50 ms</th><th>前景基线未保留帧</th><th>镜面基线未保留帧</th></tr>{per_clip}</table></div></section>
<section><h2>差异帧人工复核</h2><p>每段选取并集方案首次不一致／新增选择的前景与镜面案例。三栏显示同一源像素区域。选中仅表示通过现有手腕关联条件，触球和真实目标身份仍需人工确认。</p>{gallery}</section>
<section><h2>产物与边界</h2><p>生产默认、原ROI、人体和球检测均未修改。实验代码提供独立区域规划、原图坐标还原和跨区域NMS。模型哈希：<code>{s['model_sha256']}</code>。每段输入、ROI和手腕参考哈希保存在明细中。原视频与历史结果保留。</p><p>九段仅测试一轮，按帧轮换策略顺序；没有独立人工真值，也没有设备级执行轨迹。退出阶段仍记录到MILCompilerForANE编译告警，进程返回0且所有正式结果有限；请求CPU_AND_NE不能作为实际每个算子都在ANE执行的证明。</p><p><a href="aggregate.json">总体JSON</a> · <a href="summary.json">逐视频明细</a> · <a href="timings.csv">逐帧耗时CSV</a> · <a href="execution-log.txt">运行日志</a></p></section></html>'''
    (out / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
