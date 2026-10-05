"""Prepare empty held-out ground check points; never infer measurements."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frame', type=int, default=110)
    args = parser.parse_args()
    if args.frame < 0:
        raise ValueError('Negative source frame')
    source = Path(args.source)
    calibration_path = Path(args.calibration)
    calibration = json.loads(calibration_path.read_text())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if calibration['binding']['kind'] != 'video_sha256' or calibration['binding']['source_id'] != source_hash:
        raise ValueError('Calibration is not bound to this video')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(source))
    try:
        for _ in range(args.frame + 1):
            ok, image = cap.read()
            if not ok:
                raise ValueError('Source frame unavailable')
    finally:
        cap.release()
    if list(image.shape[1::-1]) != calibration['image_size']:
        raise ValueError('Source image size mismatch')
    if not cv2.imwrite(str(out / 'source.png'), image):
        raise OSError('Cannot write source frame')
    data = {
        'schema': 'tennis.independent-scale-check-draft.v1',
        'source_sha256': source_hash,
        'frame_id': args.frame,
        'frame_index_base': 0,
        'image_size': calibration['image_size'],
        'coordinate_space': 'original_source_pixels',
        'source_frame_png_sha256': hashlib.sha256((out / 'source.png').read_bytes()).hexdigest(),
        'calibration_id': calibration['calibration_id'],
        'calibration_sha256': hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        'binding': calibration['binding'],
        'axis_convention': 'A origin; AB width x; AD length y; metres',
        'measurement_status': 'awaiting_physical_measurements',
        'annotator_id': '',
        'measurement_evidence': '',
        'instrument': '',
        'measurement_uncertainty_m': None,
        'confirmed': False,
        'accuracy_validated': False,
        'check_points': [],
    }
    (out / 'blank_measurements.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    # Calibration corners are rejection references, never independent checks.
    context = {'data': data, 'corners': {view: item['points'] for view, item in calibration['views'].items()}}
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Court 02 · 独立尺度参照</title><style>body{font:17px/1.6 system-ui;background:#101827;color:#edf3fa;max-width:1200px;margin:24px auto;padding:0 16px}section{background:#1b2b40;padding:18px;margin:16px 0;border-radius:12px}input,select,button{font:inherit;padding:8px;margin:5px;max-width:95%}svg{width:100%;height:65vh;background:#050a12;touch-action:none}.hint{color:#fde68a}label{display:inline-block}#status{white-space:pre-wrap}button{cursor:pointer}</style>
<h1>Court 02 · 独立尺度参照</h1><p>先完成现场测量，再录入。此页不预填实测值，也不显示标定预测坐标。</p>
<section><h2>现场准备</h2><ol>
<li>保留机位、镜头缩放、画幅和分辨率；相机移动后需重新标定。</li>
<li>选择至少3个未参与四角拟合、分布在近端/中部/远端的贴地点，可增加到4–6个。避免仅在同一条线上取点。</li>
<li>用卷尺或贴地标尺测出每点相对A的两轴位置：沿AB为X，沿AD为Y，单位米。记录工具、测量不确定度和现场照片。</li>
<li>所测点必须能在当前源帧中明确对应，例如可识别的地砖交点。不能将投影结果、已用的四角或相机高度当成独立实测值。</li>
<li>若需要新铺标尺拍摄，保留新原片；本页绑定旧原片，不能将新片像素填入旧帧。应为新片单独生成检查页，并确认机位一致。</li>
<li>镜中点只有在实体对应明确时才填写；看不清的点留空。提交后计算独立重投影/米制误差，不自动调整原标定。</li>
</ol></section>
<section><p id="binding" class="hint"></p><label>标注者<input id="annotator"></label><label>测量工具<input id="instrument" placeholder="卷尺或标尺型号"></label><label>测量不确定度（米）<input id="uncertainty" type="number" min="0" step="0.001"></label><label>现场证据<input id="evidence" placeholder="照片文件名、测量日期及点位说明" size="65"></label></section>
<section><h2>点位录入</h2><label>点位<select id="point"></select></label><button id="add">增加点位</button><label>视角<select id="view"><option value="front">正面地面</option><option value="back">镜中地面</option></select></label>
<p>先在源图点击实测点，再录入现场测得的X/Y。坐标不是距离，请勿从图像估算实测值。</p>
<svg id="canvas" xmlns="http://www.w3.org/2000/svg"><image id="photo" href="source.png" width="100%" height="100%"/><g id="marks"></g></svg>
<label>实测X（米）<input id="world-x" type="number" step="0.001"></label><label>实测Y（米）<input id="world-y" type="number" step="0.001"></label><button id="clear">清空当前点</button><p id="current"></p></section>
<section><label><input id="confirmed" type="checkbox">点位来自现场实测，并已核对源图对应</label><button id="export">导出尺度检查JSON</button><p id="status" role="status" aria-live="polite">未录入实测点。可先导出准备模板；完成测量后再确认。</p></section>
<script>
const context=CONTEXT,data=context.data,$=id=>document.getElementById(id),points=[];
$('binding').textContent=`源帧${data.frame_id} · ${data.image_size.join('×')} · 标定 ${data.calibration_id.slice(0,12)} · 源视频 ${data.source_sha256.slice(0,12)}`;
$('canvas').setAttribute('viewBox',`0 0 ${data.image_size.join(' ')}`);
function current(){return points[Number($('point').value)]}
function add(){points.push({id:'check_'+(points.length+1),view:'front',image:null,world_m:[null,null]});$('point').add(new Option('检查点 '+points.length,String(points.length-1)));$('point').value=String(points.length-1);draw()}
function draw(){const p=current();$('view').value=p.view;$('world-x').value=p.world_m[0]??'';$('world-y').value=p.world_m[1]??'';$('current').textContent=p.image?'原图坐标：'+p.image.join(', '):'当前点尚未点击源图';$('marks').replaceChildren();for(const q of points){if(!q.image)continue;const c=document.createElementNS('http://www.w3.org/2000/svg','circle');for(const [k,v]of Object.entries({cx:q.image[0],cy:q.image[1],r:10,fill:q===p?'#facc15':'#5eead4'}))c.setAttribute(k,v);$('marks').append(c)}}
function changed(){ $('confirmed').checked=false;draw() }
$('point').onchange=draw;$('add').onclick=()=>{add();$('confirmed').checked=false};$('view').onchange=()=>{current().view=$('view').value;changed()};
$('canvas').onclick=e=>{const p=new DOMPoint(e.clientX,e.clientY).matrixTransform($('canvas').getScreenCTM().inverse());if(p.x<0||p.y<0||p.x>=data.image_size[0]||p.y>=data.image_size[1])return;current().image=[Math.round(p.x*100)/100,Math.round(p.y*100)/100];changed()};
for(const [id,i]of [['world-x',0],['world-y',1]])$(id).oninput=()=>{current().world_m[i]=$(id).value.trim()===''?null:Number($(id).value);$('confirmed').checked=false};
$('clear').onclick=()=>{current().image=null;current().world_m=[null,null];changed()};
$('export').onclick=()=>{const used=points.filter(p=>p.image||p.world_m.some(v=>v!==null)),complete=used.filter(p=>p.image&&p.world_m.every(v=>v!==null&&Number.isFinite(v)));data.annotator_id=$('annotator').value.trim();data.instrument=$('instrument').value.trim();data.measurement_evidence=$('evidence').value.trim();data.measurement_uncertainty_m=$('uncertainty').value.trim()===''?null:Number($('uncertainty').value);data.confirmed=$('confirmed').checked;data.check_points=used;data.measurement_status='draft';
if(data.confirmed){if(complete.length<3||complete.length!==used.length||!data.annotator_id||!data.instrument||!data.measurement_evidence||data.measurement_uncertainty_m===null||!Number.isFinite(data.measurement_uncertainty_m)||data.measurement_uncertainty_m<0){$('status').textContent='确认提交需要至少3个完整点、标注者、工具、证据和测量不确定度。';return}const seen=new Set();for(const p of used){if(context.corners[p.view].some(c=>Math.hypot(c[0]-p.image[0],c[1]-p.image[1])<1)){$('status').textContent='拟合四角不能作为独立检查点。';return}const key=p.view+':'+p.image.join(',');if(seen.has(key)){$('status').textContent='存在重复图像点。';return}seen.add(key)}data.measurement_status='operator_confirmed_pending_acceptance'}
const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='independent_scale_checks.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('status').textContent='已导出。实测结果接收核验前，准确性仍未验证。'};
for(let i=0;i<4;i++)add();$('point').value='0';draw();
</script></html>'''
    (out / 'index.html').write_text(page.replace('CONTEXT', json.dumps(context, ensure_ascii=False).replace('<', '\\u003c')))
    print(out)


if __name__ == '__main__':
    main()
