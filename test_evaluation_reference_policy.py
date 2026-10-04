"""Evaluation data and both rendered reports must distinguish agreement from accuracy."""
import json
import shutil
import subprocess
import unittest

from swing_evaluation import evaluate_swing_events
from swing_report_builder import render_report_html
from test_manual_annotation_frontend import rendered_pages, function_source


def reference_case(complete=False, method='model_assisted_review'):
    model={'events':[{'event_id':1,'start_frame':10,'contact_frame':20,'end_frame':30,'stroke_type':'Forehand'}]}
    annotation={'schema_version':'swing_manual_annotations_v2','timeline_review_complete':complete,
                'source':{'reference_method':method,'independence_verified':True},
                'events':[{'annotation_id':'model-1','source_event_id':1,'actual_stroke_type':'Forehand',
                           'valid_hit':True,'needs_review':not complete,'frames':{'start':10,'contact':20,'end':30}}]}
    return model,annotation


class EvaluationReferencePolicyTests(unittest.TestCase):
    def test_unreviewed_model_defaults_are_not_accuracy(self):
        report=evaluate_swing_events(*reference_case())
        self.assertIsNone(report['summary']['stroke_type_accuracy'])
        self.assertIsNone(report['summary']['contact_accuracy'])
        self.assertEqual(report['summary']['stroke_type_match_ratio'],1.)
        self.assertEqual(report['summary']['contact_within_tolerance_ratio'],1.)
        self.assertFalse(report['reference_provenance']['comparison_review_complete'])

    def test_completed_assisted_review_still_is_not_independent_accuracy(self):
        report=evaluate_swing_events(*reference_case(True))
        self.assertTrue(report['summary']['metrics_finalized'])
        self.assertEqual(report['summary']['precision'],1.)
        self.assertEqual(report['summary']['stroke_type_match_ratio'],1.)
        self.assertIsNone(report['summary']['stroke_type_accuracy'])
        self.assertFalse(report['reference_provenance']['independence_verified'])
        self.assertFalse(report['reference_provenance']['accuracy_validated'])

    def test_imported_self_claim_does_not_approve_independence(self):
        report=evaluate_swing_events(*reference_case(True,'independent_annotation'))
        self.assertEqual(report['reference_provenance']['declared_method'],'independent_annotation')
        self.assertFalse(report['reference_provenance']['independence_verified'])
        self.assertIsNone(report['summary']['contact_accuracy'])

    def test_legacy_card_annotations_keep_counts_as_unverified_comparison(self):
        model,annotation=reference_case(True)
        annotation['schema_version']='swing_manual_annotations_v1'
        annotation['events'][0]['event_id']=1
        report=evaluate_swing_events(model,annotation)
        self.assertEqual(report['summary']['stroke_type_correct'],1)
        self.assertEqual(report['summary']['stroke_type_match_ratio'],1.)
        self.assertIsNone(report['summary']['stroke_type_accuracy'])
        self.assertFalse(report['reference_provenance']['comparison_review_complete'])

    def test_standalone_historical_provisional_data_does_not_show_accuracy_percentage(self):
        summary={'provisional':True,'metrics_finalized':False,'stroke_type_accuracy':1.,'contact_accuracy':1.,'precision':1.}
        page=render_report_html({'paths':{},'events':[],'evaluation':{'summary':summary}},'/tmp/report.html')
        section=page.split('class="evaluation-summary"')[1].split('</div>')[0]
        self.assertNotIn('100',section)
        self.assertNotIn('accuracy',section.lower())
        self.assertIn('待复核',section)

    def test_finalized_reference_percentages_are_labelled_as_agreement(self):
        report=evaluate_swing_events(*reference_case(True))
        page=render_report_html({'paths':{},'events':[],'evaluation':report},'/tmp/report.html')
        section=page.split('class="evaluation-summary"')[1].split('</div>')[0]
        self.assertIn('类型匹配比例',section)
        self.assertIn('100',section)
        self.assertIn('独立准确性未验证',section)

    @unittest.skipUnless(shutil.which('node'),'Generated JavaScript tests require Node.js')
    def test_live_renderer_withholds_all_provisional_percentages(self):
        page=rendered_pages()['live']
        funcs='\n'.join(function_source(page,name) for name in ['reviewPercent','reviewMetricRows','reviewReferenceNote','renderReviewState'])
        script=r'''
const assert=require('node:assert/strict');
function element(){return {children:[],textContent:'',dataset:{},append(...items){this.children.push(...items);},replaceChildren(){this.children=[];}};}
const document={createElement(){return element();}};
const manualReviewMetrics=element(),coachComparisons=element();
function setReviewStatus(){}function setReviewStage(){}
function flatten(node){return node.textContent+node.children.map(flatten).join(' ');}
'''+funcs+r'''
renderReviewState({status:'needs_review',evaluation:{summary:{provisional:true,metrics_finalized:false,precision:1,stroke_type_accuracy:1,contact_accuracy:1}}});
const text=flatten(manualReviewMetrics);assert.doesNotMatch(text,/100|准确率/);assert.match(text,/待复核/);
'''
        result=subprocess.run(['node','-'],input=script,text=True,capture_output=True,timeout=8)
        self.assertEqual(result.returncode,0,result.stderr)

    @unittest.skipUnless(shutil.which('node'),'Generated JavaScript tests require Node.js')
    def test_both_page_exports_declare_model_assisted_reference(self):
        for surface,page in rendered_pages().items():
            with self.subTest(surface=surface):
                script=r'''
const assert=require('node:assert/strict');
const document={querySelectorAll(){return [];}};const timelineReviewComplete={checked:false};
const data={paths:{},summary:{}};const eventJsonUrl='events.json';
function annotationFromCard(){}
'''+function_source(page,'collectAnnotations')+r'''
assert.equal(collectAnnotations().source.reference_method,'model_assisted_review');
assert.equal(collectAnnotations().source.model_predictions_visible,true);
'''
                result=subprocess.run(['node','-'],input=script,text=True,capture_output=True,timeout=8)
                self.assertEqual(result.returncode,0,result.stderr)
