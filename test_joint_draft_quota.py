import json
import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which('node'), 'Node required')
class DraftQuotaTests(unittest.TestCase):
    def test_2000_labels_save_under_quota_and_restore_every_change(self):
        workflow=Path('scripts/joint_annotation_workflow.js').read_text()
        harness='''
const names=['left_shoulder','right_shoulder','left_hip','right_hip'];
const data={schema:'tennis.assisted-joint-review.v1',source_sha256:'a'.repeat(64),
 prediction_sha256:'b'.repeat(64),annotation_mode:'model_assisted',independent_reference:false,
 coordinate_space:'original_source_pixels',frame_index_base:0,prediction_scale:1,
 frames:Array.from({length:250},(_,i)=>({frame_id:i,width:2560,height:1440,file:`frame_${i}.png`})),labels:{}};
for(const f of data.frames)for(const view of ['front','back'])for(const name of names){
 data.labels[`${f.frame_id}:${view}:${name}`]={visible:true,x:1200.123456789,y:700.123456789,
 model_confidence:.994140625,source_frame_id:f.frame_id,origin:'model',reviewed:false};}
data.model_suggestions=JSON.parse(JSON.stringify(data.labels));
const elements={};const $=id=>elements[id] ||= {value:'',checked:false,textContent:'',selectedIndex:0};
const document={addEventListener(){}};let failures=0,peakBytes=0;
const saved={};const localStorage={getItem(k){return saved[k]||null},setItem(k,v){
 const bytes=v.length*2;if(bytes>1024*1024){failures++;const e=new Error('exceeded the quota');e.name='QuotaExceededError';throw e}
 saved[k]=v;peakBytes=Math.max(peakBytes,bytes)}};
function draw(){};
'''
        checks='''
for(let i=0;i<24;i++){data.labels[`${i}:front:left_shoulder`].reviewed=true;draw()}
if(failures)throw Error('Saving 2000 labels exceeded the quota');
if(!$('draft-state').textContent.includes('已保存'))throw Error('Latest change not saved');
const payload=JSON.parse(localStorage.getItem(draftStorageKey));
const restored=typeof decodeAnnotationStorage==='function'?decodeAnnotationStorage(payload).current:payload.current;
validateAnnotationDraft(restored);
for(let i=0;i<24;i++)if(!restored.labels[`${i}:front:left_shoulder`].reviewed)throw Error('Lost confirmation');
if(Object.keys(restored.labels).length!==2000)throw Error('Lost labels');
// Legacy full-document drafts remain recoverable without changing suggestions.
const legacy={current:restored,history:[restored]};
const migrated=decodeAnnotationStorage(legacy);
if(!migrated.current.labels['23:front:left_shoulder'].reviewed)throw Error('Legacy migration lost edit');
// Deletion and human coordinates survive storage and backwards history replay.
delete data.labels['24:front:left_shoulder'];draw();
data.labels['25:front:left_shoulder']={visible:true,x:1,y:2,reviewed:true,origin:'human_adjusted'};draw();
const decoded=decodeAnnotationStorage(JSON.parse(localStorage.getItem(draftStorageKey)));
if(decoded.current.labels['24:front:left_shoulder'])throw Error('Deletion lost');
if(decoded.current.labels['25:front:left_shoulder'].x!==1)throw Error('Human edit lost');
if(decoded.history.length!==20)throw Error('History truncated unexpectedly');
// Even the final fully-reviewed 2000-point draft must remain saved under
// the simulated one-megabyte allowance; history may be reduced if needed.
data.labels['24:front:left_shoulder']={...data.model_suggestions['24:front:left_shoulder']};
for(const point of Object.values(data.labels))point.reviewed=true;draw();
if(!$('draft-state').textContent.includes('已保存'))throw Error('Full confirmation was not saved');
const full=decodeAnnotationStorage(JSON.parse(localStorage.getItem(draftStorageKey))).current;
if(Object.values(full.labels).some(point=>!point.reviewed))throw Error('Full confirmation lost');
// An entirely full origin reports a short actionable error, keeps in-memory
// edits, and never deletes unrelated drafts or claims successful persistence.
localStorage.setItem=()=>{const error=new Error('giant key');error.name='QuotaExceededError';throw error};
data.labels['26:front:left_shoulder'].x+=1;draw();
const revision=data.draft_revision;draw();
if(data.draft_revision!==revision)throw Error('Failed save duplicated revision on navigation');
if(!$('draft-state').textContent.includes('当前修改仍在本页'))throw Error('Unsafe failure message');
if(!data.labels['26:front:left_shoulder'].reviewed)throw Error('Quota failure lost in-memory edit');
console.log(JSON.stringify({peakBytes,failures}));
'''
        result=subprocess.run(['node','-e',harness+workflow+checks],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
