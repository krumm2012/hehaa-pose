import unittest
from copy import deepcopy
from swing_evaluation import evaluate_swing_events
from manual_review_policy import validate_review_fields


def case():
    return ({'events':[{'event_id':1,'stroke_type':'Unknown','start_frame':0,'contact_frame':5,'end_frame':10}]},
            {'schema_version':'swing_manual_annotations_v2','timeline_review_complete':True,
             'events':[{'annotation_id':'model-1','source_event_id':1,'actual_stroke_type':'Unclear',
                        'valid_hit':True,'needs_review':False,'frames':{'start':0,'contact':5,'end':10}}]})


class HumanCoachReviewTests(unittest.TestCase):
    def test_unclear_type_does_not_reject_valid_event(self):
        report = evaluate_swing_events(*case())
        self.assertEqual(report['summary']['true_positive_count'], 1)
        self.assertEqual(report['summary']['false_positive_count'], 0)
        self.assertEqual(report['summary']['stroke_type_comparable'], 0)
        self.assertIsNone(report['summary']['stroke_type_match_ratio'])

    def test_non_event_is_excluded(self):
        model, annotations=case(); annotations['events'][0]['valid_hit']=False
        self.assertEqual(evaluate_swing_events(model, annotations)['summary']['false_positive_count'], 1)

    def test_confirmed_contact_is_separate_from_candidate_alignment(self):
        model, annotations=case()
        report=evaluate_swing_events(model, annotations)
        self.assertEqual(report['summary']['contact_comparable'],1)
        self.assertEqual(report['summary']['confirmed_contact_comparable'],0)
        self.assertIsNone(report['summary']['confirmed_contact_within_tolerance_ratio'])
        annotations['events'][0]['contact_review']='confirmed'
        report=evaluate_swing_events(model, annotations)
        self.assertEqual(report['summary']['confirmed_contact_comparable'],1)
        self.assertEqual(report['summary']['confirmed_contact_within_tolerance_ratio'],1.)

    def test_score_confirmation_requires_evidence_not_defaults(self):
        _, annotations=case(); row=annotations['events'][0]
        row['practice_review']={'ratings':{'preparation':3},'confirmed':True}
        with self.assertRaises(ValueError): validate_review_fields(row)
        row['practice_review']={'ratings':{k:3 for k in ('preparation','positioning','contact','coordination','recovery')},'confirmed':True}
        with self.assertRaises(ValueError): validate_review_fields(row)
        row.update(contact_review='confirmed', coach_review={'verdict':'supported','reviewer':'coach-test','observed_fact':'可见','evidence':'F3–F8 正面'})
        validate_review_fields(row)

    def test_malformed_review_is_rejected(self):
        _, annotations=case()
        for updates in [{'contact_review':'yes'}, {'contact_review':None}, {'contact_review':'not_applicable'}, {'coach_review':None}, {'coach_review':{'reviewer':None}}, {'coach_review':[]}, {'coach_review':{'verdict':'supported'}}, {'practice_review':{'confirmed':'true'}}]:
            row=deepcopy(annotations['events'][0]); row.update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError): validate_review_fields(row)

class HumanCoachWorkflowTests(unittest.TestCase):
    def test_review_round_trip_and_original_events_unchanged(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from test_manual_review_workflow import ManualReviewWorkflowTests, _annotations
        from manual_review_workflow import process_manual_review, discover_session_paths, load_review_state
        with TemporaryDirectory() as directory:
            root=Path(directory); ManualReviewWorkflowTests()._write_session(root)
            paths=discover_session_paths(root); original=paths['events'].read_bytes()
            annotations=_annotations(False); row=annotations['events'][0]
            row.update(contact_review='confirmed', coach_review={'verdict':'supported','reviewer':'coach-test',
                       'observed_fact':'可见事实','evidence':'F5–F12 正面','suggestion':'一条练习建议'},
                       practice_review={'ratings':{k:4 for k in ('preparation','positioning','contact','coordination','recovery')},'confirmed':True})
            state=process_manual_review(paths,annotations)
            self.assertEqual(state['human_review_summary']['coach_score_confirmed_count'],1)
            self.assertEqual(state['annotations'], annotations)
            self.assertEqual(state,load_review_state(paths))
            manual=json.loads(paths['manual_events'].read_text())['events'][0]
            self.assertEqual(manual['human_practice_score']['score'],80.)
            self.assertEqual(manual['human_coach_review']['reviewer'],'coach-test')
            self.assertEqual(paths['events'].read_bytes(), original)
            rejected=deepcopy(annotations); rejected['events'][0]['coach_review']['reviewer']=''
            committed=paths['state'].read_bytes()
            with self.assertRaises(ValueError): process_manual_review(paths,rejected)
            self.assertEqual(paths['state'].read_bytes(), committed)

    def test_legacy_review_never_promotes_contact_or_coach_score(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from test_manual_review_workflow import ManualReviewWorkflowTests, _annotations
        from manual_review_workflow import process_manual_review, discover_session_paths
        with TemporaryDirectory() as directory:
            root=Path(directory); ManualReviewWorkflowTests()._write_session(root)
            state=process_manual_review(discover_session_paths(root),_annotations(False))
            self.assertEqual(state['human_review_summary']['contact_confirmed_count'],0)
            self.assertEqual(state['human_review_summary']['coach_score_confirmed_count'],0)
            comparison=state['comparisons'][0]
            self.assertEqual(comparison['human_contact_review'],'unreviewed')
            self.assertIsNone(comparison['human_practice_score']['score'])

class HumanCoachFrontendTests(unittest.TestCase):
    def test_dirty_review_and_revert_cannot_keep_stale_finalization(self):
        import subprocess
        from test_manual_annotation_frontend import function_source,rendered_pages
        page=rendered_pages()['live']
        names=['sourceFrameValue','annotationFrames','validateAnnotationFrames','reviewFingerprint','markReviewDirty']
        script="\n".join(function_source(page,n) for n in names)
        script="""
const assert=require('node:assert/strict');
let savedReviewFingerprint=null,lastPublishedReviewState={status:'finalized'};
let label='',restoreCalls=0;
const manualReviewState={dataset:{state:'finalized'}};
const manualReviewMetrics={hidden:false},coachComparisons={style:{opacity:'1'}};
function setReviewStatus(text,state){label=text;manualReviewState.dataset.state=state;}
function renderReviewState(){restoreCalls++;manualReviewState.dataset.state='finalized';manualReviewMetrics.hidden=false;}
"""+script+"""
const payload={timeline_review_complete:true,events:[{annotation_id:'model-1',frames:{start:0,contact:5,end:10}}]};
savedReviewFingerprint=reviewFingerprint(payload);
payload.events[0].note='changed';markReviewDirty(payload);
assert.equal(manualReviewState.dataset.state,'dirty');assert.equal(manualReviewMetrics.hidden,true);
payload.events[0].note='';markReviewDirty(payload);assert.equal(restoreCalls,1);assert.equal(manualReviewMetrics.hidden,false);
"""
        result=subprocess.run(['node','-'],input=script,text=True,capture_output=True,timeout=8)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_business_fields_round_trip_both_editors(self):
        import subprocess
        from test_manual_annotation_frontend import function_source,rendered_pages
        for surface,page in rendered_pages().items():
            functions='\n'.join(function_source(page,n) for n in ['humanReviewFromCard','applyHumanReview','validateHumanReview'])
            script="""
const assert=require('node:assert/strict');
const contact={value:'unreviewed'},confirmed={checked:false};
const coach=['reviewer','verdict','observed_fact','evidence','suggestion'].map(k=>({dataset:{coach:k},value:''}));
const ratings=['preparation','positioning','contact','coordination','recovery'].map(k=>({dataset:{rating:k},value:''}));
const card={querySelector(s){return s.includes('practice_confirmed')?confirmed:contact;},querySelectorAll(s){return s.includes('data-coach')?coach:ratings;}};
"""+functions+"""
const source={contact_review:'uncertain',coach_review:{reviewer:'coach-test',verdict:'insufficient',observed_fact:'遮挡',evidence:'F5 正面',suggestion:''},practice_review:{ratings:{preparation:4},confirmed:false}};
validateHumanReview(source);applyHumanReview(card,source);
const restored=humanReviewFromCard(card);validateHumanReview(restored);
assert.equal(restored.contact_review,'uncertain');assert.equal(restored.coach_review.reviewer,'coach-test');assert.equal(restored.practice_review.ratings.preparation,4);assert.equal(restored.practice_review.ratings.contact,null);
applyHumanReview(card,{});assert.equal(contact.value,'unreviewed');assert.equal(confirmed.checked,false);assert.equal(ratings[0].value,'');
"""
            result=subprocess.run(['node','-'],input=script,text=True,capture_output=True,timeout=8)
            self.assertEqual(result.returncode,0,(surface,result.stderr))

class HumanCoachUpgradeTests(unittest.TestCase):
    def test_upgrade_preserves_original_and_does_not_invent_approvals(self):
        import json
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from scripts.upgrade_human_coach_review import upgrade,inventory
        from test_manual_review_workflow import ManualReviewWorkflowTests,_annotations
        from manual_review_workflow import discover_session_paths,process_manual_review
        with TemporaryDirectory() as directory:
            source=Path(directory)/'source';source.mkdir()
            ManualReviewWorkflowTests()._write_session(source)
            process_manual_review(discover_session_paths(source),_annotations(False))
            before=inventory(source)
            report=upgrade(source)
            self.assertNotEqual(report.parent,source)
            self.assertEqual(inventory(source),before)
            state=json.loads(next(report.parent.glob('*_manual_review_state.json')).read_text())
            self.assertEqual(state['human_review_summary']['contact_confirmed_count'],0)
            self.assertIn('人工 Coach 审核 · V3',report.read_text())

    def test_excluded_event_keeps_human_opinion_in_comparison(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from test_manual_review_workflow import ManualReviewWorkflowTests,_annotations
        from manual_review_workflow import discover_session_paths,process_manual_review
        with TemporaryDirectory() as directory:
            root=Path(directory); ManualReviewWorkflowTests()._write_session(root)
            annotations=_annotations(False)
            annotations['events'][0].update(valid_hit=False,contact_review='not_applicable',coach_review={
                'verdict':'rejected','reviewer':'coach-test','observed_fact':'空挥','evidence':'F1–F16 正面'})
            state=process_manual_review(discover_session_paths(root),annotations)
            self.assertTrue(state['comparisons'][0]['excluded_event'])
            self.assertEqual(state['comparisons'][0]['human_coach_review']['verdict'],'rejected')
            self.assertEqual(state['evaluation']['summary']['false_positive_count'],1)
