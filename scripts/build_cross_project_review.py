"""Build a separate Analyzer/Vision evidence index without changing either result."""
import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_project_evidence import associate_evidence, local_viewer_url, verify_viewer_metadata


def render(document, analyzer_url):
    escape = html.escape
    rows = []
    for event in document['events']:
        anchors = event['anchors']
        frames = ' / '.join(str(anchors[field]['Analyzer_frame_id']) for field in
                            ('start_frame', 'contact_frame', 'peak_frame', 'end_frame'))
        matched = ' / '.join(str(anchors[field]['Vision_frame_index']) if anchors[field]['Vision_frame_index'] is not None
                             else ('缺源帧记录' if anchors[field]['status'] == 'missing_Analyzer_frame_evidence' else '待映射')
                             for field in ('start_frame', 'contact_frame', 'peak_frame', 'end_frame'))
        verified_viewer = document.get('viewer_identity', {}).get('status') == 'verified'
        link = ('<a href="' + escape(event['viewer_url'], quote=True) + '" target="_blank" rel="noopener">打开 Vision 原结果</a>'
                if verified_viewer else 'Vision入口源身份未核验')
        rows.append('<tr>' + ''.join('<td>' + escape(str(value)) + '</td>' for value in
                    (event['event_id'], event['stroke_type'], frames, matched)) + '<td>' + link + '</td></tr>')
    status = ('同一输入视频，源帧关联通过' if document['source_frame_ids_verified'] else
              '源帧关联待验证：当前不提供自动同步帧号')
    report_link = ('<a href="' + escape(analyzer_url, quote=True) + '">打开 Analyzer 报告</a> · ') if analyzer_url else ''
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Analyzer × Vision 同源证据对照</title><style>body{background:#101923;color:#e6edf6;font:16px/1.7 system-ui;margin:28px}'
            'a{color:#89ddeb}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #526477;text-align:left}'
            'code{overflow-wrap:anywhere}main{max-width:1300px;margin:auto}</style><main><h1>Analyzer × Vision 同源证据对照</h1>'
            '<p>' + status + '</p><p>Analyzer：<code>' + escape(document['Analyzer']['source_sha256']) + '</code> · '
            + str(document['Analyzer']['recorded_frames']) + ' 个记录帧</p><p>Vision：<code>'
            + escape(document['Vision']['source_sha256']) + '</code> · ' + str(document['Vision']['frames']) + ' 帧</p>'
            '<p>同一文件及源帧身份仅用于定位证据。原片与派生片的映射、传感器曝光、人物身份、三维测量误差及技术评分仍需分别验证。</p>'
            '<p>' + report_link + '<a href="association.json">关联与阻断原因 JSON</a></p>'
            '<p>帧顺序：开始／触球候选／运动峰值／结束。Vision入口保留原结果，帧号供手动定位。</p>'
            '<table><thead><tr><th>事件</th><th>动作</th><th>Analyzer源帧</th><th>Vision数组帧</th><th>证据入口</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></main></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'vision-result', 'vision-source', 'reconstruction', 'viewer-url', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--analyzer-report-url')
    args = parser.parse_args()
    analyzer_url = local_viewer_url(args.analyzer_report_url) if args.analyzer_report_url else None
    document = associate_evidence(args.manifest, args.vision_result, args.vision_source, args.reconstruction, args.viewer_url)
    document['viewer_identity'] = verify_viewer_metadata(args.viewer_url, document['Vision']['source_sha256'], document['Vision']['frames'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'association.json').write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (out / 'report.html').write_text(render(document, analyzer_url), encoding='utf-8')
    print(json.dumps({'status': document['status'], 'events': len(document['events']), 'output': str(out),
                      'blockers': document['blockers'], 'accuracy_validated': False}))


if __name__ == '__main__':
    main()
