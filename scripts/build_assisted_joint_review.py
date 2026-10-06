"""Generate model-assisted review separately from independent labels."""
import argparse,hashlib,json,math,subprocess,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observation_policy import qualified_front_point
from scripts.audit_assisted_joint_prelabels import audit_all_prelabels

def populate_suggestions(doc, pred, scale=1.0):
    names = doc["requested_joints"]
    doc.update(model_suggestions={}, unavailable={}, review_queue=[])
    for frame in doc["frames"]:
        fid = frame["frame_id"]
        scales = pred["samples"].get(str(fid), {})
        keys = [key for key in scales if float(key) == scale]
        if len(keys) > 1: raise ValueError("Ambiguous numeric scale keys")
        pose_by_view = scales[keys[0]] if keys else {}
        for view in ("front", "back"):
            for name in names:
                key = f"{fid}:{view}:{name}"
                point = pose_by_view.get(view, {}).get(name, {})
                value, reason = qualified_front_point(point, fid, minimum_score=.5)
                if point.get("source_frame_id") != fid: reason = "source_frame_mismatch"
                if value and not (0 <= value[0] < frame["width"] and 0 <= value[1] < frame["height"]): reason = "outside_source_frame"
                if reason or not value:
                    doc["unavailable"][key] = reason or "unavailable"
                    doc["review_queue"].append({"key":key,"priority":0,"reason":doc["unavailable"][key]})
                    continue
                suggestion = {"visible":True,"x":value[0],"y":value[1],"model_confidence":value[2],
                              "source_frame_id":fid,"origin":"model","reviewed":False}
                doc["model_suggestions"][key] = suggestion.copy()
                doc["labels"][key] = suggestion
                doc["review_queue"].append({"key":key,"priority":1 if value[2]<.8 else 2,
                                            "reason":"low_score" if value[2]<.8 else "pending_review"})
    doc["review_queue"].sort(key=lambda x: (x["priority"],int(x["key"].split(":")[0]),x["key"]))
    return doc


def main():
    p=argparse.ArgumentParser()
    for name in ('source','predictions','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--frames', default='27,65,110,130,180,191')
    p.add_argument('--torso-only', action='store_true')
    p.add_argument('--journal', help='Hash-bound journal for an audit of every prelabel')
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--require-complete-review', action='store_true')
    a=p.parse_args();out=Path(a.output)
    prediction_path=Path(a.predictions);pred=json.loads(prediction_path.read_text())
    digest=hashlib.sha256(Path(a.source).read_bytes()).hexdigest()
    if digest!=pred['source_sha256']:raise ValueError('source hash mismatch')
    subprocess.run([sys.executable,str(Path(__file__).with_name('build_joint_annotation_board.py')),
                    '--source',a.source,'--output',a.output,'--frames',a.frames],check=True)
    doc=json.loads((out/'blank_labels.json').read_text())
    doc.update(schema='tennis.assisted-joint-review.v1',annotation_mode='model_assisted',independent_reference=False,
               prediction_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest(),model_suggestions={},unavailable={})
    joints=('shoulder','hip') if a.torso_only else ('shoulder','hip','elbow','wrist','knee','ankle')
    names=[side+'_'+joint for side in ('left','right') for joint in joints]
    doc['requested_joints']=names
    populate_suggestions(doc, pred, a.scale)
    doc['require_complete_review'] = a.require_complete_review
    doc['prediction_scale'] = a.scale
    if a.journal:
        raw = Path(a.journal).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pred.get('frame_journal_sha256'):
            raise ValueError('Prediction journal hash mismatch')
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        if any(row.get('session_id') != pred.get('session_id') for row in rows):
            raise ValueError('Prediction session mismatch')
        audit = audit_all_prelabels(doc, rows)
        (out/'prelabel_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    page=(out/'index.html').read_text()
    # Apply the workflow after assisted schema/joint substitutions so its
    # identity and validation bind to the final board, never the blank schema.
    if a.torso_only:
        page=page.replace('data.frames.forEach', 'names.splice(4);cn.splice(4);data.frames.forEach', 1)
    page=page.replace('不展示模型点。', f'源帧 {doc["frames"][0]["frame_id"]}–{doc["frames"][-1]["frame_id"]}，共 {len(doc["frames"])} 帧。不展示模型点。')
    begin=page.index('const data=');end=page.index(';const $',begin)
    page=page[:begin]+'const data='+json.dumps(doc,ensure_ascii=False).replace('<','\\u003c')+page[end:]
    page=page.replace('独立关节点标注','模型预标注复核').replace('填写独立标注者编号','填写复核者编号')
    page=page.replace('不展示模型点。','橙色为模型建议，绿色为已复核，青色为当前选中点。模型低分或缺失点留空；点击图片可修改当前关节，或确认建议点。此页是模型辅助复核，不作为独立准确率真值。离开前请导出。')
    page=page.replace('<button id="unknown">','<button id="accept">确认当前建议</button><button id="unknown">')
    page=page.replace('<button id="accept">', '<button id="accept-all">全部复核（所有帧、双视角）</button><button id="accept">')
    page=page.replace('离开前请导出。', '全部复核仅确认已有建议，不填补缺失点，也不覆盖人工修改。离开前请导出。')
    old="if(p&&p.visible){const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',8);c.setAttribute('fill','#00ffff');$('marks').append(c)}"
    new="""for(const [labelKey,point] of Object.entries(data.labels)){if(!labelKey.startsWith($('frame').value+':'+$('view').value+':')||!point.visible)continue;const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',point.x);c.setAttribute('cy',point.y);c.setAttribute('r',labelKey===key()?10:6);c.setAttribute('fill',labelKey===key()?'#00ffff':point.reviewed?'#22ee88':'#ffb020');c.setAttribute('stroke','#000');c.setAttribute('stroke-width','2');$('marks').append(c)}"""
    assert old in page;page=page.replace(old,new)
    page=page.replace("(p?JSON.stringify(p):'尚未标注')", "(p?('坐标 '+p.x+', '+p.y+' · '+(p.reviewed?'已复核':'待复核')+' · 模型分数 '+(p.model_confidence??'无')):'未提供可靠模型点，请人工标注或标记不可辨认')")
    page=page.replace("'\\n已填项数：'+Object.keys(data.labels).length", "'\\n已复核：'+Object.values(data.labels).filter(p=>p.reviewed).length+' / 已填：'+Object.keys(data.labels).length")
    page=page.replace("visible:true,x:Math.round(point.x*100)/100,y:Math.round(point.y*100)/100", "visible:true,x:Math.round(point.x*100)/100,y:Math.round(point.y*100)/100,origin:'human_adjusted',reviewed:true")
    page=page.replace("reason:'not_identifiable'", "reason:'not_identifiable',origin:'human_review',reviewed:true")
    page=page.replace("$('unknown').onclick", "$('accept').onclick=()=>{const p=data.labels[key()];if(p){p.reviewed=true;p.origin='human_accepted_model';draw()}};$('unknown').onclick")
    page=page.replace("$('accept').onclick", "$('accept-all').onclick=()=>{const reviewedAt=new Date().toISOString();for(const p of Object.values(data.labels)){if(!p.reviewed&&p.origin==='model'){p.reviewed=true;p.origin='human_bulk_accepted_model';p.reviewed_at=reviewedAt}}$('confirm').checked=Object.values(data.labels).every(p=>p.reviewed);draw()};$('accept').onclick")
    page=page.replace("const blob=new Blob", "if(data.confirmed&&Object.values(data.labels).some(p=>!p.reviewed)){alert('仍有未复核的模型建议，请逐点复核或先取消复核勾选导出草稿');return}const blob=new Blob")
    page=page.replace("a.download='joint_labels_draft.json'", "a.download='assisted_joint_review.json'")
    if a.journal:
        page=page.replace('<h1>50.03 · 模型预标注复核</h1>', '<h1>50.03 · 模型预标注复核</h1><p>全部条目已自动审计。<a href="prelabel_audit.json">查看逐项审计</a>；异常时间间隔或跨帧跳变仅是人工复核提示。</p>')
    if a.require_complete_review:
        page=page.replace('已复核本次填写的点（未填写仍是缺失）','全部关节点已审核（含不可辨认项）')
        page=page.replace('全部复核（所有帧、双视角）', '确认当前帧本视角已显示建议')
        page=page.replace("for(const p of Object.values(data.labels)){if(!p.reviewed&&p.origin==='model')", "for(const [k,p] of Object.entries(data.labels)){if(k.startsWith($('frame').value+':'+$('view').value+':')&&!p.reviewed&&p.origin==='model')")
        page=page.replace("$('confirm').checked=Object.values(data.labels).every(p=>p.reviewed)", "$('confirm').checked=data.frames.length*2*names.length===Object.keys(data.labels).length&&Object.values(data.labels).every(p=>p.reviewed)")
        page=page.replace('全部复核仅确认已有建议，不填补缺失点，也不覆盖人工修改。', '需要全部审核：每项都须确认或标记不可辨认。可确认当前帧本视角已显示的建议；缺失点需人工处理。')
        page=page.replace('<button id="accept">', '<button id="next-review">下一待审核项</button><button id="accept">', 1)
        action="""const beforeAuditDraw=draw;draw=function(){beforeAuditDraw();const item=data.review_queue.find(x=>x.key===key());if(item?.audit_flags?.length)$('state').textContent+='\\n审核提示：'+item.audit_flags.map(flag=>({short_source_interval:'源时间间隔过短',large_temporal_step:'跨帧位移较大',left_right_image_order_changed:'左右次序变化，请核对人物自身左右'})[flag]||flag).join(', ')};$('next-review').onclick=()=>{for(const item of data.review_queue){if(data.labels[item.key]?.reviewed===true)continue;const [fid,view,joint]=item.key.split(':');$('frame').value=fid;$('view').value=view;$('joint').value=joint;draw();return}$('draft-state').textContent='全部条目已审核，请填写复核者并确认提交'};"""
        page=page.replace("$('accept').onclick", action+"$('accept').onclick",1)
        page=page.replace("const blob=new Blob", "if(data.confirmed&&(Object.keys(data.labels).length!==data.frames.length*2*names.length||Object.values(data.labels).some(p=>p.reviewed!==true))){alert('完整审核需要所有帧双视角每个关节都已审核；缺失项须补标或标记不可辨认');return}const blob=new Blob",1)
    (out/'index.html').write_text(page)
    (out/'model_prelabels.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(doc["labels"])} model suggestions; {len(doc["unavailable"])} missing; {out}')
if __name__=='__main__':main()
