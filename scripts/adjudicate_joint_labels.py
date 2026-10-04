"""Create a human-reference comparison board, then validate explicit decisions.

Create: --left A.json --right B.json --source original.mp4 --output NEW_DIR --tolerance-px N
Finalize: --left A.json --right B.json --plan PLAN.json --decisions DECISIONS.json --output NEW.json
Original inputs are never overwritten. Tolerance only groups differences for review.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_label_adjudication import build_adjudication, finalize_adjudication, document_digest


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, document):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def render_html(plan):
    data = json.dumps(plan, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>独立标注分歧仲裁</title>
<style>body{margin:24px;background:#111827;color:#e5e7eb;font:16px system-ui}button,input,select{padding:9px;margin:4px}button{cursor:pointer}svg{display:block;width:100%;max-height:70vh;background:#000;touch-action:none}#status{white-space:pre-wrap}.a{color:#fb7185}.b{color:#38bdf8}.decision{color:#fbbf24}table{border-collapse:collapse}td,th{padding:6px 12px;border:1px solid #475569}small{display:block;margin:12px 0;color:#cbd5e1}</style>
<h1>独立标注 · 分歧仲裁</h1><p>仅显示两位人工标注者的原稿。坐标不一致、可辨认性分歧和漏标均需裁决；容差只用于查看分歧大小，不自动合并坐标。</p>
<p><span class="a">红色 A</span> · <span class="b">蓝色 B</span> · <span class="decision">黄色 最终裁决</span>。按人物自身左右区分正面与镜面背影。</p>
<label>仲裁者编号<input id="annotator"></label><button id="prev">上一项 ←</button><select id="row"></select><button id="next">下一项 →</button><button id="missing">下一未裁决</button>
<table><thead><tr><th>原稿</th><th>可辨认</th><th>原图坐标</th></tr></thead><tbody id="comparison"></tbody></table>
<p id="details"></p><button id="left">接受 A</button><button id="right">接受 B</button><button id="unknown">最终不可辨认</button><button id="clear">撤销当前裁决</button><label>依据<input id="reason" placeholder="填写原帧复核依据"></label>
<small>重新点选：点击原帧中的关节位置。接受 A／B 时需要填写依据；修改后须重新确认。</small>
<svg id="canvas" xmlns="http://www.w3.org/2000/svg"><image id="photo" width="100%" height="100%"/><g id="marks"></g></svg>
<p id="status"></p><label><input type="checkbox" id="confirm">已逐项复核并完成全部裁决</label><button id="export">导出裁决文件</button><label>导入续标<input id="import" type="file" accept="application/json"></label>
<small>两份原稿和本计划保留不变。导出文件经 finalize 命令核验后才生成最终标签；人类一致或仲裁完成也不证明三维动力链与技术评分准确。</small>
<script>const plan=PLAN_DATA, planHash=PLAN_HASH;
const $=id=>document.getElementById(id), storageKey='tennis.joint-adjudication.v1:'+planHash;
const ready=plan.status!=='pending_reference_confirmation';
let data={schema:'tennis.joint-adjudication-decisions.v1',plan_sha256:planHash,annotator_id:null,confirmed:false,labels:{}};
const point=(p)=>!p?'未填写':p.visible?`(${p.x}, ${p.y})`:'—';
plan.rows.forEach(r=>$('row').add(new Option(r.key+' · '+r.status,r.key)));
function row(){return plan.rows.find(r=>r.key===$('row').value)}
function unresolved(){return plan.rows.filter(r=>r.review_required&&!data.labels[r.key])}
function validate(candidate){
 if(!candidate||candidate.schema!==data.schema||candidate.plan_sha256!==planHash||!candidate.labels||typeof candidate.labels!=='object'||Array.isArray(candidate.labels))throw Error('计划或裁决格式不匹配');
 for(const [key,d] of Object.entries(candidate.labels)){
  const r=plan.rows.find(r=>r.key===key);if(!r||!d||!['left','right','custom'].includes(d.choice)||typeof d.reason!=='string'||!d.reason.trim())throw Error('无效裁决');
  if(d.choice!=='custom'&&!r[d.choice])throw Error('不能接受未填写原稿');
  if(d.choice==='custom'){
   const p=d.point,f=plan.frames.find(f=>f.frame_id===r.frame_id);
   if(!p||typeof p.visible!=='boolean')throw Error('缺少可辨认性');
   if(p.visible&&(![p.x,p.y].every(v=>typeof v==='number'&&Number.isFinite(v))||p.x<0||p.y<0||p.x>=f.width||p.y>=f.height))throw Error('坐标越界');
   if(!p.visible&&(p.x!==null||p.y!==null))throw Error('不可辨认点必须为空坐标');
  }
 }
 return candidate;
}
function persist(){data.annotator_id=$('annotator').value.trim()||null;data.confirmed=$('confirm').checked;try{localStorage.setItem(storageKey,JSON.stringify(data))}catch(e){$('status').textContent+='\\n本地保存失败，请导出草稿'}}
function finalPoint(r){const d=data.labels[r.key];return d?d.choice==='custom'?d.point:r[d.choice]:r.review_required?null:r.left}
function draw(){
 const r=row(),f=plan.frames.find(f=>f.frame_id===r.frame_id);
 $('canvas').setAttribute('viewBox',`0 0 ${f.width} ${f.height}`);$('photo').setAttribute('href',`frame_${f.frame_id}.png`);$('marks').replaceChildren();
 for(const [p,color] of [[r.left,'#fb7185'],[r.right,'#38bdf8'],[finalPoint(r),'#fbbf24']])if(p&&p.visible){
  const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',8);c.setAttribute('fill','none');c.setAttribute('stroke',color);c.setAttribute('stroke-width',3);$('marks').append(c);
 }
 $('comparison').replaceChildren();for(const [name,p] of [['A',r.left],['B',r.right]]){
  const tr=document.createElement('tr');for(const t of [name,p?p.visible?'是':'否':'未填写',point(p)]){const td=document.createElement('td');td.textContent=t;tr.append(td)}$('comparison').append(tr);
 }
 $('details').textContent=r.status+(r.distance_px===null?'':` · 位置差 ${r.distance_px.toFixed(2)} px · 查看容差 ${plan.reporting_tolerance_px} px`);
 $('reason').value=data.labels[r.key]?.reason||'';
 $('left').disabled=!ready||!r.left;$('right').disabled=!ready||!r.right;$('unknown').disabled=!ready;$('clear').disabled=!ready;
 $('confirm').disabled=!ready||unresolved().length>0;if($('confirm').disabled)$('confirm').checked=false;
 $('status').textContent=ready?`需裁决 ${plan.summary.review_required_count} 项；剩余 ${unresolved().length} 项。已保存的人工裁决 ${Object.keys(data.labels).length} 项。`:'两份独立原稿尚未确认；请先完成独立标注，再重新建立仲裁计划。';
 persist();
}
function choose(choice,p){if(!ready)return;const reason=$('reason').value.trim();if(!reason){alert('请填写原帧复核依据');return}data.labels[row().key]={choice,reason,...(choice==='custom'?{point:p}:{})};$('confirm').checked=false;draw()}
$('left').onclick=()=>choose('left');$('right').onclick=()=>choose('right');$('unknown').onclick=()=>choose('custom',{visible:false,x:null,y:null,reason:'not_identifiable'});
$('clear').onclick=()=>{delete data.labels[row().key];$('confirm').checked=false;draw()};
$('canvas').addEventListener('click',e=>{if(!ready)return;const p=new DOMPoint(e.clientX,e.clientY).matrixTransform($('canvas').getScreenCTM().inverse()),f=plan.frames.find(f=>f.frame_id===row().frame_id);if(p.x>=0&&p.y>=0&&p.x<f.width&&p.y<f.height)choose('custom',{visible:true,x:Math.round(p.x*100)/100,y:Math.round(p.y*100)/100})});
function step(delta){$('row').selectedIndex=Math.max(0,Math.min(plan.rows.length-1,$('row').selectedIndex+delta));draw()}
$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('row').onchange=draw;
$('missing').onclick=()=>{const r=unresolved()[0];if(r){$('row').value=r.key;draw()}};
$('annotator').oninput=()=>{$('confirm').checked=false;persist()};$('confirm').onchange=persist;
$('reason').oninput=()=>{const d=data.labels[row().key];if(d){d.reason=$('reason').value;$('confirm').checked=false;persist()}};
$('export').onclick=()=>{persist();try{validate(data);if(data.confirmed&&(!data.annotator_id||unresolved().length||!ready))throw Error('确认前需仲裁者编号和完整裁决');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='joint_adjudication_decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}catch(e){alert(e.message)}};
$('import').onchange=async e=>{try{const candidate=validate(JSON.parse(await e.target.files[0].text()));data=candidate;$('annotator').value=data.annotator_id||'';$('confirm').checked=data.confirmed===true;draw()}catch(err){alert(err.message)}e.target.value=''};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'){e.preventDefault();step(-1)}if(e.key==='ArrowRight'){e.preventDefault();step(1)}});
try{const saved=localStorage.getItem(storageKey);if(saved){data=validate(JSON.parse(saved));$('annotator').value=data.annotator_id||'';$('confirm').checked=data.confirmed===true}}catch(e){$('status').textContent='草稿未恢复：'+e.message}
draw();</script></html>'''.replace('PLAN_HASH', json.dumps(document_digest(plan)), 1).replace('PLAN_DATA', data, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create', help='Compare two original references and create an original-frame board')
    finish = commands.add_parser('finalize', help='Validate decisions and produce confirmed independent labels')
    for command in (create, finish):
        command.add_argument('--left', required=True)
        command.add_argument('--right', required=True)
        command.add_argument('--output', required=True, help='New path only; existing files are never overwritten')
    create.add_argument('--source', required=True)
    create.add_argument('--tolerance-px', required=True, type=float, help='Review grouping only, not an accuracy criterion')
    finish.add_argument('--plan', required=True)
    finish.add_argument('--decisions', required=True)
    args = parser.parse_args()
    left, right = read(args.left), read(args.right)
    if args.command == 'finalize':
        result = finalize_adjudication(left, right, read(args.plan), read(args.decisions))
        write_new(args.output, result)
        print(json.dumps({'output': args.output, 'labels': len(result['labels']), 'accuracy_validated': False}))
        return
    plan = build_adjudication(left, right, args.tolerance_px)
    digest = hashlib.sha256()
    with Path(args.source).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    if digest.hexdigest() != plan['source_sha256']: raise ValueError('original source video hash mismatch')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    import cv2
    cap = cv2.VideoCapture(args.source)
    try:
        for frame in plan['frames']:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame['frame_id'])
            ok, pixels = cap.read()
            if not ok or pixels.shape[:2] != (frame['height'], frame['width']):
                raise ValueError('missing source frame or dimensions mismatch')
            if not cv2.imwrite(str(out / f"frame_{frame['frame_id']}.png"), pixels):
                raise OSError('source-frame image write failed')
    finally:
        cap.release()
    # Retain immutable copies; future confirmation is bound to these whole-document hashes.
    write_new(out / 'reference_left.json', left)
    write_new(out / 'reference_right.json', right)
    write_new(out / 'plan.json', plan)
    write_new(out / 'decisions_draft.json', {'schema': 'tennis.joint-adjudication-decisions.v1',
              'plan_sha256': document_digest(plan), 'annotator_id': None, 'confirmed': False, 'labels': {}})
    (out / 'index.html').write_text(render_html(plan), encoding='utf-8')
    print(json.dumps({'output': str(out), 'status': plan['status'], **plan['summary'], 'accuracy_validated': False}))


if __name__ == '__main__':
    main()
