import json
import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which('node'), 'Node required for annotation workflow')
class AnnotationWorkflowTests(unittest.TestCase):
    def test_draft_restore_import_guards_and_history(self):
        workflow = Path('scripts/joint_annotation_workflow.js').read_text()
        data = {'schema':'tennis.independent-joint-labels.v1', 'source_sha256':'abc',
                'coordinate_space':'original_source_pixels','frame_index_base':0,
                'frames':[{'frame_id':1,'width':100,'height':100,'file':'frame_1.png'}], 'labels':{}}
        harness = '''
const data=DATA, names=['left_shoulder'];
const elements={}; const $=id=>elements[id] ||= {value:'',checked:false,textContent:'',selectedIndex:0};
const document={addEventListener(){}};
const saved={}; const localStorage={setItem(k,v){saved[k]=v},getItem(k){return saved[k]||null}};
function draw(){};
'''.replace('DATA', json.dumps(data))
        checks = '''
data.labels['1:front:left_shoulder']={visible:true,x:0,y:0}; draw();
data.labels['1:front:left_shoulder'].reviewed=true;
$('annotator').value='qa_test_only';
$('confirm').checked=true; draw();
if(!$('confirm').checked || !data.confirmed) throw Error('bulk review confirmation cleared');
data.labels['1:front:left_shoulder'].x=1; draw();
if($('confirm').checked || data.confirmed) throw Error('coordinate edit kept confirmation');
data.labels['1:front:left_shoulder'].x=0; draw();
const draft=decodeAnnotationStorage(JSON.parse(localStorage.getItem(draftStorageKey))).current;
if(draft.labels['1:front:left_shoulder'].x!==0) throw Error('zero lost');
data.labels={}; restoreAnnotationDraft(draft);
if(!data.labels['1:front:left_shoulder']) throw Error('restore lost label');
for(const change of [
 {schema:'tennis.assisted-joint-review.v1'}, {source_sha256:'other'},
 {confirmed:true,annotator_id:null},
 {labels:{'1:front:left_shoulder':{visible:true,x:100,y:0}}},
 {labels:{'1:front:left_shoulder':{visible:false,x:0,y:null}}},
 {labels:{'1:front:left_shoulder':{visible:true,x:5,y:5,origin:'human_accepted_model'}}},
 {labels:{'99:front:left_shoulder':{visible:true,x:0,y:0}}}
]) {let rejected=false;try{validateAnnotationDraft({...draft,...change})}catch(e){rejected=true}if(!rejected)throw Error('unsafe import accepted');}
if(!draftHistory.length) throw Error('history lost');
'''
        subprocess.run(['node','-e', harness+workflow+checks],check=True,capture_output=True,text=True)

    def test_assisted_complete_review_rejects_partial_confirmation(self):
        workflow=Path('scripts/joint_annotation_workflow.js').read_text()
        data={'schema':'tennis.assisted-joint-review.v1','source_sha256':'abc',
              'coordinate_space':'original_source_pixels','frame_index_base':0,
              'frames':[{'frame_id':1,'width':100,'height':100,'file':'frame_1.png'}],
              'labels':{},'model_suggestions':{},'prediction_scale':1.0,
              'annotation_mode':'model_assisted','independent_reference':False,
              'require_complete_review':True}
        harness="""
const data=DATA,names=['left_shoulder'];
const elements={};const $=id=>elements[id] ||= {value:'',checked:false,textContent:'',selectedIndex:0};
const document={addEventListener(){}};const localStorage={setItem(){},getItem(){return null}};
function draw(){};
""".replace('DATA',json.dumps(data))
        checks="""
const point={visible:true,x:20,y:20,reviewed:true};
const draft={...data,confirmed:true,annotator_id:'fixture',labels:{'1:front:left_shoulder':point}};
let failed=false;try{validateAnnotationDraft(draft)}catch(e){failed=true}if(!failed)throw Error('Partial confirmation accepted');
draft.labels['1:back:left_shoulder']=point;validateAnnotationDraft(draft);
draft.labels['1:back:left_shoulder']={...point,review_actor:'automatic'};
failed=false;try{validateAnnotationDraft(draft)}catch(e){failed=true}if(!failed)throw Error('Automatic review promoted to human confirmation');
draft.labels['1:back:left_shoulder']=point;
failed=false;try{validateAnnotationDraft({...draft,review_revision_id:'other'})}catch(e){failed=true}if(!failed)throw Error('Wrong revision accepted');
draft.labels['1:back:left_shoulder']={...point,reviewed:false};
failed=false;try{validateAnnotationDraft(draft)}catch(e){failed=true}if(!failed)throw Error('Unreviewed accepted');
"""
        subprocess.run(['node','-e',harness+workflow+checks],check=True,capture_output=True,text=True)
