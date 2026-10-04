"""Generate model-assisted review separately from independent labels."""
import argparse,hashlib,json,math,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    for name in ('source','predictions','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--frames', default='27,65,110,130,180,191')
    p.add_argument('--torso-only', action='store_true')
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
    for frame in doc['frames']:
        fid=frame['frame_id']
        for view in ('front','back'):
            pose=pred['samples'].get(str(fid),{}).get('1',{}).get(view,{})
            for name in names:
                key=f'{fid}:{view}:{name}';point=pose.get(name,{})
                good=(point.get('observed') is True and point.get('source_frame_id')==fid
                      and all(isinstance(point.get(k),(int,float)) and math.isfinite(point[k]) for k in ('x','y','confidence'))
                      and point['confidence']>=.5 and 0<=point['x']<frame['width'] and 0<=point['y']<frame['height'])
                if not good:
                    doc['unavailable'][key]='missing_low_confidence_or_not_current';continue
                suggestion={'visible':True,'x':point['x'],'y':point['y'],'model_confidence':point['confidence'],
                            'source_frame_id':fid,'origin':'model','reviewed':False}
                doc['model_suggestions'][key]=suggestion.copy();doc['labels'][key]=suggestion
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
    (out/'index.html').write_text(page)
    (out/'model_prelabels.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(doc["labels"])} model suggestions; {len(doc["unavailable"])} missing; {out}')
if __name__=='__main__':main()
