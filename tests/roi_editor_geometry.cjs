// Exercise the page's actual coordinate functions and canvas initialization order.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('local_control_panel.html', 'utf8');
const code = html.slice(html.indexOf('    const roiEditorState = {'), html.indexOf('    function parsePointString'));
for (const initialWidth of [300, 2560, 1600]) {
  const source = {frame_size: [2560, 1440], points: [[1170,336],[1952,358],[2186,1196],[962,1194]], mirror_view: {polygon: [[.4244,.2993],[.4267,0],[.8145,.0021],[.8029,.3199]]}};
  const canvas = {width: initialWidth, height: initialWidth*9/16, style: {}};
  const img = {naturalWidth:1600,naturalHeight:900,clientWidth:1440,clientHeight:810,hidden:false,getBoundingClientRect:()=>({left:38,top:302})};
  const stage = {getBoundingClientRect:()=>({left:38,top:302})};
  const context = vm.createContext({selectedStream:()=>source, $:id=>({'preview-image':img,'roi-canvas':canvas,'preview-stage-container':stage})[id], sortPointsTLTRBRBL:p=>p, drawRoiCanvas:()=>{}});
  vm.runInContext(code+'\ninitRoiEditorFromStream(selectedStream()); syncCanvasGeometry();', context);
  const points = vm.runInContext('roiEditorState.courtPoints', context);
  for (let i=0;i<4;i++) {
    const actual = points[i][0]/canvas.width * img.clientWidth;
    const expected = source.points[i][0]/2560 * img.clientWidth;
    assert.ok(Math.abs(actual-expected)<1, `initial canvas ${initialWidth}: ROI corner ${i+1} shifted ${actual-expected}px`);
  }
  const mirror = vm.runInContext('roiEditorState.mirrorPoints', context);
  assert.ok(Math.abs(mirror[0][0]/canvas.width - source.mirror_view.polygon[0][0])<.001,'mirror shifted');
  const roundTrip = vm.runInContext('previewToSource(sourceToPreview([1170,336]))', context);
  assert.deepEqual(Array.from(roundTrip),[1170,336]);
  vm.runInContext('roiEditorState.courtPoints[0] = sourceToPreview([1180,346]); syncCanvasGeometry();',context);
  assert.deepEqual(Array.from(vm.runInContext('previewToSource(roiEditorState.courtPoints[0])',context)),[1180,346]);
}
console.log('PASS: ROI and mirror align on first entry, repeated entry and geometry refresh; edited source coordinates preserved.');
