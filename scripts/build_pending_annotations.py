#!/usr/bin/env python3
"""Create a separate, unlabelled review queue from recorded swing candidates."""
import argparse
import hashlib
import html
import json
from pathlib import Path


def build_queue(source, event_path, fps):
    document = json.loads(event_path.read_text())
    events = document.get('events', [])
    return {
        'schema_version': 'tennis.pending-annotations.v1',
        'status': 'pending_independent_annotation',
        'source': str(source.resolve()),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'candidate_events': str(event_path.resolve()),
        'fps': fps, 'frame_index_base': 0,
        'instructions': [
            '先看整段视频，补记漏检挥拍；候选窗口并不代表真实事件边界。',
            '先独立标注，再查看模型结果。建议两名标注者独立完成并保留分歧。',
            '触球无法直接确认时留空，并填写遮挡、模糊或未见球等原因；不可把最近球帧当真值。',
            '记录前/背视角可见性及左右肩、髋、肘、腕的原始像素点，二维角度由这些点计算。',
            '先验证动作分类、触球时间误差、二维关节角度误差；二维峰值顺序不等于真实三维动力链。',
            '真实三维动力链仍需同步、标定、坐标定义和独立参考数据；本清单不能替代。',
        ],
        'items': [{
            'candidate_id': e.get('event_id', i+1),
            'candidate_window_frames': [e['start_frame'], e['end_frame']],
            'annotation': {
                'status': 'pending', 'annotator_id': None, 'stroke_type': None,
                'actual_start_frame': None, 'actual_end_frame': None,
                'contact_frame': None, 'contact_uncertainty_frames': None,
                'contact_visibility': None, 'front_visibility': None, 'back_visibility': None,
                'joint_observations': [], 'exclusion_reason': None, 'notes': None,
            },
        } for i, e in enumerate(events)],
        'missed_events': [],
        'independent_3d_reference': None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--fps', type=float, required=True)
    p.add_argument('--video-url', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    queue = build_queue(a.source, a.events, a.fps)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'pending_annotations.json').write_text(json.dumps(queue, ensure_ascii=False, indent=2))
    cards = []
    for i, item in enumerate(queue['items']):
        start, end = item['candidate_window_frames']
        fields = ''.join(f'<label>{label}<input data-i="{i}" data-key="{key}" {"type=number min=0 step=1" if key.endswith("frame") or key.endswith("frames") else ""}></label>' for key,label in [
            ('annotator_id','标注者'), ('stroke_type','动作类型（Forehand / Backhand / Two-Handed Backhand / Unknown）'),
            ('actual_start_frame','实际起始帧'), ('actual_end_frame','实际结束帧'), ('contact_frame','确认触球帧'),
            ('contact_uncertainty_frames','触球不确定范围 ±帧'), ('contact_visibility','触球可见性'),
            ('front_visibility','前视角可见性'), ('back_visibility','背视角可见性'),
            ('exclusion_reason','不能判定的原因'), ('notes','备注')])
        cards.append(f'<section><h2>候选 #{item["candidate_id"]} · 原始帧 {start}–{end}</h2><button onclick="video.currentTime={max(0,start-10)/a.fps}">定位到候选前 10 帧</button><div class="fields">{fields}</div></section>')
    encoded = json.dumps(queue, ensure_ascii=False).replace('<', '\\u003c')
    page = '''<!doctype html><meta charset="utf-8"><title>50.03 待标注清单</title>
<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1100px;margin:32px auto;padding:20px}video{width:100%;max-height:550px}section{background:#1f2937;padding:20px;margin:20px 0;border-radius:12px}.fields{display:grid;grid-template-columns:1fr 1fr;gap:12px}label{display:grid;gap:6px}input,button{padding:10px;border-radius:6px;border:1px solid #64748b}button{cursor:pointer}li{margin:8px 0}a{color:#93c5fd}</style>
<h1>50.03 · 独立待标注清单</h1><p>所有真值字段为空。以下是模型候选窗口，尚未经过人工确认。帧编号从 0 开始。页面输入仅保存在本页内存，离开前请导出；不会覆盖历史分析。</p>
<ul>INSTRUCTIONS</ul><video id="video" controls src="VIDEO"></video><p>视频时间只能用于定位；逐帧标注需使用原视频解码帧号，尤其在可变帧率素材中不要用播放时间推断帧号。</p>
CARDS
<p>原始关节点、漏检事件和第二位标注者结果请补入导出的 JSON。建议保留各人的独立文件后再仲裁。</p><button id="download">导出标注 JSON</button> <a href="pending_annotations.json">下载空白清单</a>
<script>const queue=QUEUE;document.querySelectorAll('input').forEach(el=>el.addEventListener('input',()=>{queue.items[Number(el.dataset.i)].annotation[el.dataset.key]=el.value===''?null:el.type==='number'?Number(el.value):el.value;}));document.getElementById('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(queue,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='50_03_annotations_draft.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};</script>'''
    page = page.replace('INSTRUCTIONS',''.join('<li>'+html.escape(x)+'</li>' for x in queue['instructions'])).replace('VIDEO',html.escape(a.video_url,quote=True)).replace('CARDS',''.join(cards)).replace('QUEUE',encoded)
    (a.output/'annotations.html').write_text(page)
    print(f'{len(queue["items"])} candidates; all truth fields pending; {a.output}')


if __name__ == '__main__':
    main()
