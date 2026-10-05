"""Build source-bound foot contact review; ankle hints never become sole labels."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def adapt_board(page, data):
    begin = page.index('const data=')
    end = page.index(';const $', begin)
    page = page[:begin] + 'const data=' + json.dumps(data, ensure_ascii=False).replace('<', '\\u003c') + page[end:]
    page = page.replace('独立关节点标注', '足部接地点辅助复核').replace('50.03 · ', '')
    page = page.replace('填写独立标注者编号', '填写复核者编号')
    page = page.replace('不展示模型点。', '橙色空心点仅为原始模型脚踝参考，不是足底接地点，也不能一键接受为接地标签。此页结果属于模型参考辅助复核，不作为独立准确率真值；全部复核只确认已填写项，不填补缺失。')
    page = page.replace('请按人物自身的左右标记；', '点击可见的鞋底与地面接触位置；脚离地请选择“腾空”，无法确认请选择“不可辨认”；请按人物自身的左右标记；')
    page = page.replace("names.forEach((n,i)", "names.splice(0,names.length,'left_contact','right_contact');cn.splice(0,cn.length,'左足底接地点','右足底接地点');names.forEach((n,i)")
    page = page.replace('<button id="unknown">', '<button id="airborne">腾空 / 未接地</button><button id="review-all">全部复核已填写项</button><button id="unknown">')
    page = page.replace("visible:true,x:Math.round(point.x*100)/100,y:Math.round(point.y*100)/100", "visible:true,x:Math.round(point.x*100)/100,y:Math.round(point.y*100)/100,contact_state:'ground_contact_visible',origin:'human_marked',reviewed:true")
    page = page.replace("reason:'not_identifiable'", "reason:'not_identifiable',contact_state:'unknown',origin:'human_review',reviewed:true")
    page = page.replace("$('unknown').onclick", "$('airborne').onclick=()=>{data.labels[key()]={visible:false,x:null,y:null,reason:'airborne',contact_state:'airborne',origin:'human_review',reviewed:true};draw()};$('review-all').onclick=()=>{if(!Object.keys(data.labels).length){$('state').textContent='请先标注接地点、腾空或不可辨认';return}for(const p of Object.values(data.labels)){p.reviewed=true;p.review_method='bulk_review'}$('confirm').checked=true;draw()};$('unknown').onclick", 1)
    page = page.replace("$('marks').replaceChildren();", """$('marks').replaceChildren();for(const side of ['left','right']){const hint=data.ankle_references[$('frame').value+':'+$('view').value+':'+side+'_ankle'];if(hint){const mark=document.createElementNS('http://www.w3.org/2000/svg','circle');mark.setAttribute('cx',hint.x);mark.setAttribute('cy',hint.y);mark.setAttribute('r',8);mark.setAttribute('fill','none');mark.setAttribute('stroke','#ffb020');mark.setAttribute('stroke-width',3);$('marks').append(mark)}}""")
    page = page.replace("'joint_labels_draft.json'", "'ground_contact_review.json'")
    page = page.replace("if (candidate.confirmed === true && (!candidate.annotator_id", "if (JSON.stringify(candidate.ankle_references) !== JSON.stringify(data.ankle_references) || candidate.journal_sha256 !== data.journal_sha256) throw Error('脚踝参考或证据来源已改变');\n  if (candidate.confirmed === true && (!candidate.annotator_id")
    page = page.replace("if (p.visible && (", "if (!['ground_contact_visible','airborne','unknown'].includes(p.contact_state) || p.visible !== (p.contact_state === 'ground_contact_visible')) throw Error('接地状态与坐标不一致');\n    if (p.visible && (")
    page = page.replace("'尚未标注'", "'尚未标注接地状态'")
    return page


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'journal', 'manifest', 'output', 'frames'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    source_hash = hashlib.sha256(Path(args.source).read_bytes()).hexdigest()
    manifest = json.loads(Path(args.manifest).read_text())
    binding = (manifest.get('session', {}).get('ground_calibration_application') or {}).get('input_binding', {})
    if binding.get('kind') != 'video_sha256' or binding.get('source_id') != source_hash:
        raise ValueError('Ground review requires the manifest actual input video hash to match')
    journal_path = Path(args.journal)
    journal_hash = hashlib.sha256(journal_path.read_bytes()).hexdigest()
    journal_entries = [x for x in manifest.get('artifacts', []) if x.get('role') == 'frame_journal']
    if len(journal_entries) != 1 or journal_entries[0].get('sha256') != journal_hash:
        raise ValueError('Frame journal does not match source evidence manifest')
    rows = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
    ids = [r.get('frame_id') for r in rows]
    if any(type(fid) is not int or fid < 0 for fid in ids) or len(ids) != len(set(ids)):
        raise ValueError('Invalid or duplicate journal source frame identity')
    requested = [int(x) for x in args.frames.split(',')]
    if not set(requested) <= set(ids):
        raise ValueError('Requested source frame missing from journal')
    subprocess.run([sys.executable, str(Path(__file__).with_name('build_joint_annotation_board.py')),
                    '--source', args.source, '--output', args.output, '--frames', args.frames], check=True)
    out = Path(args.output)
    data = json.loads((out/'blank_labels.json').read_text())
    data.update(schema='tennis.ground-contact-review.v1', annotation_mode='ankle_hint_assisted',
                independent_reference=False, accuracy_validated=False,
                requested_joints=['left_contact', 'right_contact'], ankle_references={},
                journal_sha256=journal_hash)
    bounds = {f['frame_id']: f for f in data['frames']}
    for row in rows:
        fid = row['frame_id']
        if fid not in bounds or row.get('pose_observation_coordinate_space') != 'original_source_pixels':
            continue
        for view in ('front', 'back'):
            for side in ('left', 'right'):
                from observation_policy import qualified_front_point
                point = (row.get('pose_observations', {}).get(view) or {}).get(side+'_ankle')
                value, reason = qualified_front_point(point, fid, minimum_score=.5)
                if reason or not isinstance(point, dict) or type(point.get('source_frame_id')) is not int or point['source_frame_id'] != fid:
                    continue
                x, y, score = value
                if 0 <= x < bounds[fid]['width'] and 0 <= y < bounds[fid]['height']:
                    data['ankle_references'][f'{fid}:{view}:{side}_ankle'] = {'x':x, 'y':y, 'score':score, 'source_frame_id':fid, 'meaning':'ankle_only_not_ground_contact'}
    (out/'blank_labels.json').unlink()
    (out/'ground_contact_review_template.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    (out/'index.html').write_text(adapt_board((out/'index.html').read_text(), data))
    print(f"{len(data['frames'])} frames; {len(data['ankle_references'])} ankle references; 0 inferred ground contacts")


if __name__ == '__main__':
    # Direct scripts use the repository's observation qualification policy.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
