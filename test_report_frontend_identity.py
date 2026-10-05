"""Run both generated editors at their import, restore and export boundaries."""
import shutil
import subprocess
import unittest

from test_manual_annotation_frontend import function_source, rendered_pages


@unittest.skipUnless(shutil.which('node'), 'Generated editor tests require Node.js')
class ReportFrontendIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = rendered_pages()

    def execute(self, surface, names, assertions):
        helpers = ['sourceFrameValue', 'sourceIdentityAttribute', 'annotationIdentity',
                   'annotationFrames', 'validateAnnotationFrames', 'validateAnnotationPayload',
                   'prepareAnnotationImport', 'showAnnotationError']
        functions = '\n'.join(function_source(self.pages[surface], name)
                              for name in dict.fromkeys(helpers + names))
        script = r'''
const assert=require('node:assert/strict');
let added=[], cleared=0, persisted=0;
const nodes={};const modelEventIds=[1,2];
function card(id){
 const fields={};for(const key of ['start_frame','contact_frame','end_frame','actual_stroke_type','note','count_correct','valid_hit','needs_review'])fields[key]={value:'original',checked:false,setCustomValidity(){}};
 return {dataset:{sourceEventId:String(id),annotationId:`model-${id}`,peakFrame:'10'},fields,
  querySelector(s){return fields[s.match(/data-field="([^"]+)"/)[1]];},querySelectorAll(){return []}};
}
const cards=[card(1),card(2)];
const document={getElementById(id){return nodes[id]||={textContent:'previous',dataset:{}}},
 querySelector(s){return cards.find(c=>s.includes(`="${c.dataset.sourceEventId}"`))||null},
 querySelectorAll(){return cards;}};
const manualEvents={replaceChildren(){cleared++}},timelineReviewComplete={checked:false};
const annotationReadiness=document.getElementById('annotation-readiness');
const annotationStatus=document.getElementById('annotation-status');
const data={events:[{event_id:1,peak_frame:10},{event_id:2,peak_frame:20}]};
const localStorage={getItem(){return JSON.stringify(importPayload)}};
const annotationStorageKey='test';
function addMissedEvent(row){added.push(row);return {dataset:{annotationId:row.annotation_id}};}
function refreshAnnotations(){persisted++;}function saveAnnotations(){persisted++;}
let importPayload={schema_version:'swing_manual_annotations_v2',timeline_review_complete:true,
 events:[{annotation_id:'review-1',source_event_id:1,actual_stroke_type:'Serve',frames:{start:0,contact:10,end:20}}]};
''' + '\n(async()=>{\n' + functions + '\n' + assertions + '\n})().catch(e=>{console.error(e.stack);process.exitCode=1});'
        result=subprocess.run(['node','-'],input=script,text=True,capture_output=True,timeout=8)
        self.assertEqual(result.returncode,0,result.stderr)

    def importer(self, surface):
        return ('applyImportedAnnotations' if surface=='standalone' else 'applyImportedReviewToEditor')

    def names(self, surface, extra=None):
        return (['setField','applyAnnotationToCard'] if surface=='standalone'
                else ['setAnnotationField','applyAnnotation']) + [self.importer(surface)] + (extra or [])

    def test_bad_source_identity_cannot_mutate_or_link_to_existing_model_card(self):
        for surface in self.pages:
            for value in ('1', True, 1.9, -1, 2**53):
                with self.subTest(surface=surface,identity=value):
                    self.execute(surface,self.names(surface),f'''
importPayload.events[0].source_event_id={repr(value).replace('True','true')};
try{{{self.importer(surface)}(importPayload)}}catch(error){{assert.match(error.message,/事件|身份|ID|整数/)}}
assert.equal(cards[0].fields.actual_stroke_type.value,'original');
assert.equal(timelineReviewComplete.checked,false);assert.equal(cleared,0);assert.equal(added.length,0);
''')

    def test_duplicate_annotation_or_source_link_rejected_before_any_mutation(self):
        for surface in self.pages:
            for duplicate in ('annotation_id','source_event_id'):
                with self.subTest(surface=surface,duplicate=duplicate):
                    self.execute(surface,self.names(surface),f'''
importPayload.events.push({{...importPayload.events[0],annotation_id:'review-2',source_event_id:2}});
importPayload.events[1].{duplicate}=importPayload.events[0].{duplicate};
try{{{self.importer(surface)}(importPayload)}}catch(error){{assert.match(error.message,/重复/)}}
assert.equal(cards[0].fields.actual_stroke_type.value,'original');
assert.equal(cards[1].fields.actual_stroke_type.value,'original');assert.equal(cleared,0);assert.equal(persisted,0);
''')

    def test_explicit_null_source_link_does_not_inherit_legacy_event_id(self):
        for surface in self.pages:
            with self.subTest(surface=surface):
                self.execute(surface,self.names(surface),f'''
importPayload.events[0].source_event_id=null;importPayload.events[0].event_id=1;
{self.importer(surface)}(importPayload);
assert.equal(cards[0].fields.actual_stroke_type.value,'original');assert.equal(added.length,1);
assert.equal(added[0].source_event_id,null);
''')

    def test_missing_legacy_source_link_resolves_integer_and_preserves_annotation_id(self):
        for surface in self.pages:
            with self.subTest(surface=surface):
                self.execute(surface,self.names(surface),f'''
delete importPayload.events[0].source_event_id;importPayload.events[0].event_id=1;
{self.importer(surface)}(importPayload);
assert.equal(cards[0].fields.actual_stroke_type.value,'Serve');
assert.equal(cards[0].dataset.annotationId,'review-1');assert.equal(added.length,0);
''')

    def test_nonexistent_source_link_is_not_reclassified_as_manual_event(self):
        for surface in self.pages:
            with self.subTest(surface=surface):
                self.execute(surface,self.names(surface),f'''
importPayload.events[0].source_event_id=99;
try{{{self.importer(surface)}(importPayload)}}catch(error){{assert.match(error.message,/不存在|当前|来源/)}}
assert.equal(added.length,0);assert.equal(cleared,0);assert.equal(timelineReviewComplete.checked,false);
''')

    def test_import_identity_collision_with_unmodified_model_card_is_rejected(self):
        for surface in self.pages:
            with self.subTest(surface=surface):
                self.execute(surface,self.names(surface),f'''
importPayload.events[0].annotation_id='model-2';
try{{{self.importer(surface)}(importPayload)}}catch(error){{assert.match(error.message,/重复|冲突/)}}
assert.equal(cards[0].dataset.annotationId,'model-1');assert.equal(cleared,0);
''')

    def test_restore_uses_same_preflight_and_preserves_saved_review_identity(self):
        self.execute('live',['setAnnotationField','applyAnnotation','restoreAnnotations'],'''
restoreAnnotations();assert.equal(cards[0].dataset.annotationId,'review-1');
assert.equal(cards[0].fields.actual_stroke_type.value,'Serve');
''')

    def test_invalid_saved_review_does_not_restore_partial_values(self):
        self.execute('live',['setAnnotationField','applyAnnotation','restoreAnnotations'],'''
importPayload.events.push({...importPayload.events[0],annotation_id:'other',source_event_id:1});
restoreAnnotations();assert.equal(cards[0].fields.actual_stroke_type.value,'original');
assert.equal(timelineReviewComplete.checked,false);assert.equal(cleared,0);
''')

    def test_malformed_dom_source_link_is_not_coerced_on_export(self):
        for surface in self.pages:
            with self.subTest(surface=surface):
                self.execute(surface,['integerField','annotationFromCard'],'''
for(const text of ['1.9','true','01','1e0','9007199254740993']){
 cards[0].dataset.sourceEventId=text;
 assert.throws(()=>annotationFromCard(cards[0]),/身份|ID|整数/);
}
''')

    def test_invalid_shadowed_or_peak_frame_is_not_silently_dropped(self):
        for surface in self.pages:
            with self.subTest(surface=surface):
                self.execute(surface,self.names(surface),f'''
importPayload.events[0].frames.peak=10.5;
try{{{self.importer(surface)}(importPayload)}}catch(error){{assert.match(error.message,/帧|整数/)}}
assert.equal(cards[0].fields.actual_stroke_type.value,'original');assert.equal(cleared,0);
''')

    def test_explicit_missing_peak_is_preserved_through_import_and_export(self):
        for surface in self.pages:
            with self.subTest(surface=surface):
                self.execute(surface,self.names(surface,['integerField','annotationFromCard']),f'''
importPayload.events[0].frames.peak=null;importPayload.events[0].peak_frame=10;
{self.importer(surface)}(importPayload);
assert.equal(cards[0].dataset.peakFrame,'');assert.equal(annotationFromCard(cards[0]).frames.peak,null);
''')

    def test_valid_zero_source_and_frames_are_preserved(self):
        for surface in self.pages:
            with self.subTest(surface=surface):
                self.execute(surface,self.names(surface,['integerField','annotationFromCard']),f'''
cards[0].dataset.sourceEventId='0';modelEventIds.push(0);
importPayload.events[0].source_event_id=0;importPayload.events[0].frames={{start:0,contact:0,end:0,peak:0}};
{self.importer(surface)}(importPayload);
const exported=annotationFromCard(cards[0]);assert.equal(exported.source_event_id,0);
assert.deepEqual(exported.frames,{{start:0,contact:0,peak:0,end:0}});
''')

    def test_frontend_validation_does_not_mutate_original_import(self):
        for surface in self.pages:
            with self.subTest(surface=surface):
                self.execute(surface,self.names(surface),f'''
delete importPayload.events[0].annotation_id;const before=JSON.stringify(importPayload);
{self.importer(surface)}(importPayload);assert.equal(JSON.stringify(importPayload),before);
''')

    def test_missing_contact_or_peak_has_no_frame_zero_timeline_marker(self):
        self.execute('standalone',['percentForFrame','marker'],'''
const totalFrames=21;
assert.equal(marker('触球候选',null,'contact',1),'');
assert.equal(marker('动作峰值',undefined,'peak',1),'');
assert.match(marker('触球候选',0,'contact',1),/data-frame="0"/);
''')

    def test_missing_source_frame_does_not_seek_video_to_zero(self):
        self.execute('standalone',['seekFrame'],'''
const totalFrames=21,timeByFrame=new Map([[0,0],[10,0.4]]),sourceTimes=[[0,0],[10,0.4]];
const video={currentTime:77,play(){}},scrubber={value:'previous'},timelineStatus={textContent:''};
function updatePlaybackState(){}
seekFrame(null);assert.equal(video.currentTime,77);assert.equal(scrubber.value,'previous');
seekFrame('0');assert.ok(video.currentTime<0.001);
''')
