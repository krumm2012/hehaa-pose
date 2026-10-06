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
    page='''<!doctype html><meta charset="utf-8"><title>独立关节点标注 (9帧金标准)</title><style>
body{font:16px system-ui;background:#111827;color:#eee;margin:24px;max-width:1400px;margin:20px auto;padding:0 16px}
button,select,input{padding:9px 14px;margin:4px;border-radius:6px;border:1px solid #4b5563;background:#1f2937;color:#fff}
button{cursor:pointer}button:hover{background:#374151}.primary{background:#2563eb;font-weight:bold}.unknown{background:#dc2626}
svg{width:100%;height:min(72vh,750px);background:#000;touch-action:none;border-radius:8px;border:1px solid #374151;cursor:crosshair}
#state{white-space:pre-wrap;background:#1f2937;padding:12px;border-radius:8px;margin-top:10px;font-family:monospace}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0}
.guide-box{background:#1e293b;border-left:4px solid #38bdf8;padding:12px 16px;border-radius:4px;margin:12px 0}
a{color:#67e8f9}
</style>
<h1>🎾 Stage 3 · 独立躯干关节点 9 帧基准标注面板</h1>
<div class="guide-box">
<p><strong>盲测基准规范</strong>：本面板不显示任何模型预测点或辅助建议。请按选手自身左右标注；正面人物与镜面背影分别独立标注。</p>
<p>若关节点被肢体、球拍严重遮挡无法确定，请直接点击“不可辨认”，切勿猜测。自动聚焦模式支持 1:1 像素高精点击，点击后将自动引导至下一关节点。</p>
</div>
<div class="row">
<label>独立标注者编号: <input id="annotator" placeholder="输入姓名或工号 (如 annotator_01)"></label>
<select id="frame"></select>
<select id="view"><option value="front">正面人物 (Front View)</option><option value="back">镜面背影 (Mirror View)</option></select>
<select id="joint"></select>
<label><input type="checkbox" id="zoom" checked> 自动聚焦人物区域 (1:1 视窗)</label>
<label><input type="checkbox" id="autonext" checked> 标记后自动跳转下一项</label>
</div>
<div class="row">
<button id="unknown" class="unknown">❓ 标记当前点不可辨认</button>
<button id="clear">撤销当前点</button>
<button id="review-all" class="primary">全部复核已填项</button>
<button id="save" class="primary">💾 导出独立基准 JSON</button>
<label><input type="checkbox" id="confirm"> 已复核本次填写的点</label>
</div>
<svg id="canvas" xmlns="http://www.w3.org/2000/svg"><image id="photo" width="100%" height="100%"/><g id="marks"></g></svg>
<p id="state"></p>
<script>const data=DATA;const $=id=>document.getElementById(id);const names=['left_shoulder','right_shoulder','left_hip','right_hip','left_elbow','right_elbow','left_wrist','right_wrist','left_knee','right_knee','left_ankle','right_ankle'];
const cn=['左肩 (L-Shoulder)','右肩 (R-Shoulder)','左髋 (L-Hip)','右髋 (R-Hip)','左肘','右肘','左腕','右腕','左膝','右膝','左踝','右踝'];
data.frames.forEach(f=>$('frame').add(new Option('源帧 '+f.frame_id,f.frame_id)));names.forEach((n,i)=>$('joint').add(new Option(cn[i],n)));
function key(){return [$('frame').value,$('view').value,$('joint').value].join(':')}
function nextTarget(){
  let ji=$('joint').selectedIndex+1;
  if(ji<names.length){$('joint').selectedIndex=ji;draw();return;}
  $('joint').selectedIndex=0;
  if($('view').value==='front'){$('view').value='back';draw();return;}
  $('view').value='front';
  let fi=$('frame').selectedIndex+1;
  if(fi<data.frames.length){$('frame').selectedIndex=fi;draw();}
}
function draw(){
  const f=data.frames.find(f=>f.frame_id==$('frame').value);
  if($('zoom').checked){
    if($('view').value==='front'){$('canvas').setAttribute('viewBox','1150 150 850 850');}
    else {$('canvas').setAttribute('viewBox','1250 0 500 450');}
  } else {
    $('canvas').setAttribute('viewBox',`0 0 ${f.width} ${f.height}`);
  }
  $('photo').setAttribute('href',f.file);
  $('marks').replaceChildren();
  // Draw all joints in current frame & view
  names.forEach((n,i)=>{
    const k=[$('frame').value,$('view').value,n].join(':');
    const pt=data.labels[k];
    if(pt&&pt.visible){
      const isCur=(n===$('joint').value);
      const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
      c.setAttribute('cx',pt.x);c.setAttribute('cy',pt.y);c.setAttribute('r',isCur?7:5);
      c.setAttribute('fill',isCur?'#facc15':'#38bdf8');
      c.setAttribute('stroke','#fff');c.setAttribute('stroke-width','2');
      $('marks').append(c);
      const t=document.createElementNS('http://www.w3.org/2000/svg','text');
      t.setAttribute('x',pt.x+9);t.setAttribute('y',pt.y+4);
      t.setAttribute('fill',isCur?'#facc15':'#38bdf8');t.setAttribute('font-size','14');
      t.setAttribute('font-weight','bold');t.textContent=cn[i].split(' ')[0];
      $('marks').append(t);
    }
  });
  const p=data.labels[key()];
  const totalReq=data.frames.length*2*names.length;
  const filled=Object.keys(data.labels).length;
  $('state').textContent=`当前焦点项：${key()} (${$('view').selectedOptions[0].text} · ${$('joint').selectedOptions[0].text})\n`
    +`当前状态：${p?(p.visible?`已标记坐标 (${p.x}, ${p.y})`:'标记为不可辨认'):'尚未标注'}\n`
    +`整体进度：已填写 ${filled} / ${totalReq} 项 · 剩余 ${totalReq-filled} 项`;
}
$('canvas').addEventListener('click',e=>{
  const point=new DOMPoint(e.clientX,e.clientY).matrixTransform($('canvas').getScreenCTM().inverse());
  const f=data.frames.find(f=>f.frame_id==$('frame').value);
  if(point.x<0||point.y<0||point.x>=f.width||point.y>=f.height)return;
  data.labels[key()]={visible:true,x:Math.round(point.x*10)/10,y:Math.round(point.y*10)/10};
  $('confirm').checked=false;
  draw();
  if($('autonext').checked)setTimeout(nextTarget,150);
});
$('unknown').onclick=()=>{data.labels[key()]={visible:false,x:null,y:null,reason:'not_identifiable'};$('confirm').checked=false;draw();if($('autonext').checked)setTimeout(nextTarget,150);};
$('clear').onclick=()=>{delete data.labels[key()];$('confirm').checked=false;draw();};
['frame','view','joint','zoom'].forEach(n=>$(n).onchange=draw);
$('review-all').onclick=()=>{if(!Object.keys(data.labels).length){alert('尚未填写任何点，请先标注或标记不可辨认');return;}const at=new Date().toISOString();for(const point of Object.values(data.labels)){point.reviewed=true;point.reviewed_at=at;point.review_method='bulk_manual_review';}$('confirm').checked=true;draw();};
$('save').onclick=()=>{data.annotator_id=$('annotator').value.trim()||null;data.confirmed=$('confirm').checked;if(data.confirmed&&!data.annotator_id){alert('复核结果需要填写标注者编号');return;}const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='court02_independent_joint_labels.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
draw();</script>'''
    if a.torso_only:
        page=add_bulk_review(page)
        page=page.replace("data.frames.forEach", "names.splice(4);cn.splice(4);data.frames.forEach", 1)
        page=page.replace('不展示模型点。', f'覆盖关键切片 9 帧（静态准备/高速加速/遮挡随挥）。逐帧标注双视角左右肩、左右髋；不展示模型点。')
    (out/'index.html').write_text(add_annotation_workflow(page.replace('DATA',json.dumps(doc,ensure_ascii=False).replace('<','\\u003c'))))
if __name__=='__main__':main()
