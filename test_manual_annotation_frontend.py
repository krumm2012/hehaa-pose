"""Execute generated report JavaScript at the real annotation call sites."""
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

from realtime_swing_pipeline import RealtimeSwingOutputManager
from swing_report_builder import render_report_html


def rendered_pages():
    payload = {'paths': {}, 'events': [], 'summary': {}, 'timeline': {}}
    manager = RealtimeSwingOutputManager.__new__(RealtimeSwingOutputManager)
    manager.roi_metadata = {}
    manager.preview_path = None
    manager.output_html = Path('/tmp/frontend_contract_report.html')
    manager.output_json = Path('/tmp/events.json')
    return {'standalone': render_report_html(payload, '/tmp/frontend_contract_report.html'),
            'live': manager._render_live_html({'events': [], 'summary': {}})}


def function_source(page, name):
    start = re.search(r'(?:async )?function ' + name + r'\(', page)
    if not start:
        return ''
    after = page[start.start():]
    stop = re.search(r'\n    (?:(?:async )?function |document\.|annotationWorkspace\.|manualReviewFile\.)', after)
    return after[:stop.start()] if stop else after[:after.index('</script>')]


@unittest.skipUnless(shutil.which('node'), 'Generated JavaScript regressions require Node.js')
class ManualAnnotationFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = rendered_pages()

    def execute(self, surface, names, assertions):
        page = self.pages[surface]
        helpers = ['sourceFrameValue', 'sourceIdentityAttribute', 'annotationIdentity',
                   'annotationFrames', 'validateAnnotationFrames',
                   'validateAnnotationPayload', 'showAnnotationError']
        functions = '\n'.join(function_source(page, n) for n in dict.fromkeys(helpers + names))
        script = r'''
const assert = require('node:assert/strict');
const modelEventIds=[];
let downloads=0, writes=0, requests=0;
const nodes={};
const document={getElementById(id){return nodes[id] ||= {textContent:'previous',dataset:{}};},
 createElement(){return {click(){downloads++;}};}};
const URL={createObjectURL(){return 'test:';},revokeObjectURL(){}};
const Blob=function(){};
const annotationReadiness=document.getElementById('annotation-readiness');
const annotationStatus=document.getElementById('annotation-status');
const timelineReviewComplete={checked:false};
const localStorage={setItem(){writes++;}};
const annotationStorageKey='test';
let lastAnnotationInteraction=0;
function card(values={}){
 const fields={}; for(const key of ['start_frame','contact_frame','end_frame','actual_stroke_type','note','count_correct','valid_hit','needs_review']) fields[key]={value:values[key]??'109',checked:false,setCustomValidity(s){this.validationMessage=s;},setAttribute(){},removeAttribute(){}};
 return {dataset:{},fields,querySelector(selector){return fields[selector.match(/data-field="([^"]+)"/)[1]];},querySelectorAll(){return [];}};
}
const valid={schema_version:'swing_manual_annotations_v2',events:[{frames:{start:89,contact:110,end:137}}]};
''' + '\n(async()=>{\n' + functions + '\n' + assertions + '\n})().catch(e=>{console.error(e.stack);process.exitCode=1;});'
        result = subprocess.run(['node','-'],input=script,text=True,capture_output=True,timeout=8)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_typed_fraction_negative_and_unsafe_are_rejected_without_rounding(self):
        for surface in self.pages:
            with self.subTest(surface=surface):
                self.execute(surface,['integerField'],"""
for(const raw of ['109.6','-1','9007199254740993','1e2']){
 const c=card({contact_frame:raw});
 assert.throws(()=>integerField(c,'contact_frame'));
 assert.equal(c.fields.contact_frame.value,raw);
}
""")

    def test_valid_zero_and_integer_preserved_empty_draft_allowed(self):
        for surface in self.pages:
            with self.subTest(surface=surface):
                self.execute(surface,['integerField'],"""
assert.equal(integerField(card({contact_frame:'0'}),'contact_frame'),0);
assert.equal(integerField(card({contact_frame:'110'}),'contact_frame'),110);
assert.equal(integerField(card({contact_frame:''}),'contact_frame'),null);
""")

    def test_import_invalid_frames_rejected_before_card_mutation(self):
        for surface in self.pages:
            setter,apply = ('setField','applyAnnotationToCard') if surface=='standalone' else ('setAnnotationField','applyAnnotation')
            with self.subTest(surface=surface):
                self.execute(surface,[setter,apply],f"""
for(const bad of [true,'110',109.6,-1,9007199254740992]){{
 const c=card();const imported={{actual_stroke_type:'Serve',frames:{{start:89,contact:bad,end:137}}}};
 assert.throws(()=>{apply}(c,imported));
 assert.equal(c.fields.actual_stroke_type.value,'109');
 assert.equal(c.fields.contact_frame.value,'109');
}}
""")

    def test_import_explicit_missing_anchor_clears_previous_model_value(self):
        for surface in self.pages:
            setter,apply = ('setField','applyAnnotationToCard') if surface=='standalone' else ('setAnnotationField','applyAnnotation')
            with self.subTest(surface=surface):
                self.execute(surface,[setter,apply],f"""
const c=card();{apply}(c,{{frames:{{start:89,contact:null,end:137}},contact_frame:110}});
assert.equal(c.fields.contact_frame.value,'');
""")

    def test_invalid_export_does_not_download_previous_or_coerced_json(self):
        for surface in self.pages:
            with self.subTest(surface=surface):
                extra = "function refreshAnnotations(){return valid;}" if surface=='standalone' else "function saveAnnotations(){return valid;}"
                self.execute(surface,['downloadAnnotations'],extra+"""
valid.events[0].frames.contact=109.6;
document.getElementById('annotation-json').textContent=JSON.stringify(valid);
downloadAnnotations();assert.equal(downloads,0);
assert.match(annotationReadiness.textContent,/帧|整数/);
""")

    def test_valid_export_still_downloads(self):
        for surface in self.pages:
            with self.subTest(surface=surface):
                extra = "function refreshAnnotations(){return valid;}" if surface=='standalone' else "function saveAnnotations(){return valid;}"
                self.execute(surface,['downloadAnnotations'],extra+"""
document.getElementById('annotation-json').textContent=JSON.stringify(valid);
downloadAnnotations();assert.equal(downloads,1);
""")

    def test_invalid_evaluation_does_not_send_request(self):
        self.execute('live',['submitManualReview'],"""
const location={protocol:'http:'};
const evaluateImportedReview={},evaluateCurrentReview={};let importedReviewPayload=null;
function setReviewStatus(){} function setReviewStage(){} function renderReviewState(){}
async function manualReviewRequestHeaders(){return {};}
async function fetch(){requests++;return {ok:true,json:async()=>({})};}
valid.events[0].frames.contact=109.6;
await submitManualReview(valid);assert.equal(requests,0);
""")

    def test_invalid_draft_does_not_overwrite_previous_saved_review(self):
        self.execute('live',['saveAnnotations'],"""
function collectAnnotations(){return valid;}
function updateAnnotationReadiness(){}
valid.events[0].frames.contact=109.6;
assert.equal(saveAnnotations(),null);assert.equal(writes,0);
assert.match(annotationReadiness.textContent,/帧|整数/);
""")

    def test_invalid_refresh_clears_previous_export_preview(self):
        self.execute('standalone',['refreshAnnotations'],"""
function collectAnnotations(){return valid;}
function updateAnnotationReadiness(){}
valid.events[0].frames.contact=109.6;
assert.equal(refreshAnnotations(),null);
assert.match(document.getElementById('annotation-json').textContent,/校验/);
""")

    def test_empty_model_peak_is_not_exported_as_source_frame_zero(self):
        self.execute('live',['integerField','annotationFromCard'],"""
const c=card();c.dataset.peakFrame='';
assert.equal(annotationFromCard(c).frames.peak,null);
c.dataset.peakFrame='0';assert.equal(annotationFromCard(c).frames.peak,0);
""")

    def test_backend_rejects_unsafe_identity_and_invalid_frame_containers(self):
        from manual_review_workflow import _annotation_frames
        for frames in [[], True, 1, {'start':0,'contact':2**53,'end':2**53}]:
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                _annotation_frames({'frames':frames})

    def test_live_event_card_input_reaches_draft_validation(self):
        bindings = '\n'.join(match.group(0) for match in re.finditer(
            r"(?:document|annotationWorkspace)\.addEventListener\('(?:input|change)'[^;]+;",
            self.pages['live']))
        self.execute('live',['handleAnnotationEdit'],"""
const listeners=[];
document.addEventListener=(name,fn)=>listeners.push({root:'document',name,fn});
const annotationWorkspace={contains(){return false;},addEventListener(name,fn){listeners.push({root:'workspace',name,fn});}};
function saveAnnotations(){writes++;}
"""+bindings+"""
for(const row of listeners.filter(row=>row.root==='document'&&row.name==='input')){
 row.fn({target:{closest(){return {};}}});
}
assert.equal(writes,1,'Event cards are outside annotationWorkspace and must reach validation');
""")
