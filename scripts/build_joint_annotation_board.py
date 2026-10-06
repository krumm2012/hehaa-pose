"""Create a blind source-frame click annotation board; no model points are shown."""
import argparse,hashlib,json
from pathlib import Path
import cv2

def add_annotation_workflow(page):
    controls = '<button id="frame-prev">上一帧 ←</button><button id="frame-next">下一帧 →</button><button id="next-missing">下一未填项</button><label>导入续标<input type="file" id="draft-import" accept="application/json"></label><button id="draft-history">导出修订记录</button><button id="draft-retry">重试保存</button><p id="draft-state">编辑后自动保存本地草稿；导出文件可跨浏览器续标。</p>'
    page = page.replace('<svg id="canvas"', controls + '<svg id="canvas"', 1)
    workflow = Path(__file__).with_name('joint_annotation_workflow.js').read_text()
    return page.replace('</script>', workflow + '</script>', 1)

def add_bulk_review(page):
    page=page.replace('<button id="save">', '<button id="review-all">全部复核已填写点</button><button id="save">')
    page=page.replace("$('save').onclick", "$('review-all').onclick=()=>{if(!Object.keys(data.labels).length){$('state').textContent='尚未填写任何点，请先标注或标记不可辨认';return}const at=new Date().toISOString();for(const point of Object.values(data.labels)){point.reviewed=true;point.reviewed_at=at;point.review_method='bulk_manual_review'}$('confirm').checked=true;draw()};$('save').onclick", 1)
    page=page.replace("+'\\n已填项数：'+Object.keys(data.labels).length", "+'\\n已填项数：'+Object.keys(data.labels).length+' · 已复核：'+Object.values(data.labels).filter(p=>p.reviewed).length+' · 未填写：'+(data.frames.length*2*names.length-Object.keys(data.labels).length)")
    # Any subsequent edit requires confirmation again.
    page=page.replace(";draw()});", ";$('confirm').checked=false;draw()});")
    page=page.replace(";draw()};$('clear')", ";$('confirm').checked=false;draw()};$('clear')")
    page=page.replace("delete data.labels[key()];draw()", "delete data.labels[key()];$('confirm').checked=false;draw()")
    return page

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--frames', default='27,65,110,130,180,191', help='Comma-separated source frame IDs')
    p.add_argument('--torso-only', action='store_true')
    a=p.parse_args();out=Path(a.output)
    if (out/'index.html').exists(): raise FileExistsError('Annotation board exists; choose a new version directory')
    out.mkdir(parents=True,exist_ok=True)
    frame_ids=sorted(set(int(value) for value in a.frames.split(',')))
    if not frame_ids or min(frame_ids)<0:raise ValueError('invalid source frames')
    cap=cv2.VideoCapture(a.source);frames=[]
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,img=cap.read()
        if not ok:raise ValueError('missing frame '+str(fid))
        cv2.imwrite(str(out/f'frame_{fid}.png'),img)
        frames.append({'frame_id':fid,'width':img.shape[1],'height':img.shape[0],'file':f'frame_{fid}.png'})
    cap.release()
    doc={'schema':'tennis.independent-joint-labels.v1','source_sha256':hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),
         'annotator_id':None,'confirmed':False,'coordinate_space':'original_source_pixels','frame_index_base':0,
         'frames':frames,'labels':{},'requested_joints':
         [side+'_'+joint for joint in ('shoulder','hip') for side in ('left','right')]
         if a.torso_only else [side+'_'+joint for side in ('left','right') for joint in ('shoulder','hip','elbow','wrist','knee','ankle')]}
    (out/'blank_labels.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2))
    page='''<!doctype html><meta charset="utf-8"><title>独立关节点标注</title><style>body{font:16px system-ui;background:#111827;color:#eee;margin:24px}button,select,input{padding:10px;margin:6px}svg{width:100%;max-height:76vh;background:#000;touch-action:none}#state{white-space:pre-wrap}a{color:#67e8f9}</style>
<h1>50.03 · 独立关节点标注</h1><p>不展示模型点。请按人物自身的左右标记；正面与镜面背影分别标注。遮挡或位置无法确定时选“不可辨认”，不要猜测。坐标保存为原视频像素，帧号从0开始。</p>
<label>标注者<input id="annotator" placeholder="填写独立标注者编号"></label><select id="frame"></select><select id="view"><option value="front">正面人物</option><option value="back">镜面背影</option></select><select id="joint"></select>
<button id="unknown">不可辨认</button><button id="clear">撤销当前点</button><button id="save">导出草稿</button><label><input type="checkbox" id="confirm">已复核本次填写的点（未填写仍是缺失）</label>
<svg id="canvas" xmlns="http://www.w3.org/2000/svg"><image id="photo" width="100%" height="100%"/><g id="marks"></g></svg><p id="state"></p>
<script>const data=DATA;const $=id=>document.getElementById(id);const names=['left_shoulder','right_shoulder','left_hip','right_hip','left_elbow','right_elbow','left_wrist','right_wrist','left_knee','right_knee','left_ankle','right_ankle'];
const cn=['左肩','右肩','左髋','右髋','左肘','右肘','左腕','右腕','左膝','右膝','左踝','右踝'];
data.frames.forEach(f=>$('frame').add(new Option('源帧 '+f.frame_id,f.frame_id)));names.forEach((n,i)=>$('joint').add(new Option(cn[i],n)));
function key(){return [$('frame').value,$('view').value,$('joint').value].join(':')}
function draw(){const f=data.frames.find(f=>f.frame_id==$('frame').value);$('canvas').setAttribute('viewBox',`0 0 ${f.width} ${f.height}`);$('photo').setAttribute('href',f.file);$('marks').replaceChildren();const p=data.labels[key()];if(p&&p.visible){const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',8);c.setAttribute('fill','#00ffff');$('marks').append(c)}$('state').textContent=key()+' · '+(p?JSON.stringify(p):'尚未标注')+'\\n已填项数：'+Object.keys(data.labels).length;}
$('canvas').addEventListener('click',e=>{const point=new DOMPoint(e.clientX,e.clientY).matrixTransform($('canvas').getScreenCTM().inverse());const f=data.frames.find(f=>f.frame_id==$('frame').value);if(point.x<0||point.y<0||point.x>=f.width||point.y>=f.height)return;data.labels[key()]={visible:true,x:Math.round(point.x*100)/100,y:Math.round(point.y*100)/100};draw()});
$('unknown').onclick=()=>{data.labels[key()]={visible:false,x:null,y:null,reason:'not_identifiable'};draw()};$('clear').onclick=()=>{delete data.labels[key()];draw()};['frame','view','joint'].forEach(n=>$(n).onchange=draw);
$('save').onclick=()=>{data.annotator_id=$('annotator').value.trim()||null;data.confirmed=$('confirm').checked;if(data.confirmed&&!data.annotator_id){alert('复核结果需要标注者编号');return}const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='joint_labels_draft.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};draw();</script>'''
    if a.torso_only:
        page=add_bulk_review(page)
        page=page.replace("data.frames.forEach", "names.splice(4);cn.splice(4);data.frames.forEach", 1)
        page=page.replace('不展示模型点。', f'连续源帧 {frame_ids[0]}–{frame_ids[-1]}，共 {len(frame_ids)} 帧。逐帧标注双视角的左右肩、左右髋；不展示模型点。离开前请导出保存。')
    (out/'index.html').write_text(add_annotation_workflow(page.replace('DATA',json.dumps(doc,ensure_ascii=False).replace('<','\\u003c'))))
if __name__=='__main__':main()
