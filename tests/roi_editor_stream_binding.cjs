const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('local_control_panel.html','utf8');
const code = html.slice(html.indexOf('    function selectedStream()'),html.indexOf('    function escapeHtml')) + html.slice(html.indexOf('    const roiEditorState = {'),html.indexOf('    function parsePointString'));
const elements = new Map();
function $(id){if(!elements.has(id))elements.set(id,{value:'',hidden:false,style:{},checked:false});return elements.get(id);}
const streams = [
 {stream_id:'court01-main',label:'Court 01',source:'camera01',frame_size:[2560,1440],points:[[720,792],[1598,811],[1808,1430],[499,1421]],mirror_view:{polygon:[[.2,0],[.7,0],[.7,.3],[.2,.3]]}},
 {stream_id:'court02-main',label:'Court 02',source:'camera02',frame_size:[2560,1440],points:[[1170,336],[1952,358],[2186,1196],[962,1194]],mirror_view:{polygon:[[.4244,.2993],[.4267,0],[.8145,.0021],[.8029,.3199]]}}
];
const ctx=vm.createContext({$,state:{streams,videoName:'50.03.mp4'},CUSTOM_STREAM_ID:'custom',escapeHtml:v=>v,savePreset:()=>{},sortPointsTLTRBRBL:p=>p,drawRoiCanvas:()=>{}});
Object.assign($('preview-image'),{naturalWidth:1600,naturalHeight:900,clientWidth:1280,clientHeight:720,hidden:false,getBoundingClientRect:()=>({left:0,top:0})});
Object.assign($('roi-canvas'),{width:300,height:150});
$('preview-stage-container').getBoundingClientRect=()=>({left:0,top:0});
vm.runInContext(code,ctx);
$('stream_id').value='court01-main';vm.runInContext('updateStreamMeta()',ctx);
assert.equal($('roi_p1').value,'720, 792');
$('stream_id').value='local_video';$('mapped_stream_id').value='court02-main';
vm.runInContext('updateStreamMeta()',ctx);
assert.equal($('roi_p1').value,'1170, 336','local video must replace stale Court 01 sidebar coordinates');
assert.equal(vm.runInContext('selectedStream()?.stream_id',ctx),'court02-main','editor and save must resolve mapped camera');
assert.deepEqual(Array.from(vm.runInContext('getSourceDimensions()',ctx)),[2560,1440],'source dimensions must not come from resized preview');
vm.runInContext('roiEditorState.active=true; initRoiEditorFromStream(selectedStream()); syncCanvasGeometry()',ctx);
assert.deepEqual(Array.from(vm.runInContext('roiEditorState.courtPoints[0]',ctx)),[1170,336]);
assert.equal($('roi-canvas').width,2560);
assert.ok(Math.abs(vm.runInContext('previewToNorm(roiEditorState.mirrorPoints[0])[0]',ctx)-.4244)<.001);
$('mapped_stream_id').value='court01-main';vm.runInContext('updateStreamMeta()',ctx);
assert.deepEqual(Array.from(vm.runInContext('roiEditorState.courtPoints[0]',ctx)),[720,792]);
$('mapped_stream_id').value='';vm.runInContext('updateStreamMeta()',ctx);
assert.equal(vm.runInContext('selectedStream()',ctx),undefined);
assert.equal($('roi_p1').value,'','unbound video must not retain another camera coordinates');
assert.equal(vm.runInContext('roiEditorState.courtPoints.length',ctx),0);
console.log('PASS: local-video camera binding, sidebar, source size, canvas, mirror, camera switching, and unbinding.');
