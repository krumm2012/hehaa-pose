import unittest
from copy import deepcopy
from joint_annotation_evaluation import evaluate_joint_labels

class JointEvaluationTests(unittest.TestCase):
    def fixtures(self):
        labels={'schema':'tennis.independent-joint-labels.v1','source_sha256':'a'*64,
            'coordinate_space':'original_source_pixels','frame_index_base':0,'confirmed':True,'annotator_id':'test-only',
            'frames':[{'frame_id':1,'width':100,'height':100}],
            'labels':{'1:front:left_wrist':{'visible':True,'x':10,'y':10},
                      '1:front:right_wrist':{'visible':True,'x':20,'y':20},
                      '1:back:left_wrist':{'visible':False,'x':None,'y':None}}}
        pred={'schema':'tennis.pose-resolution-audit.v1','source_sha256':'a'*64,'scales':[1],
            'samples':{'1':{'1':{'front':{'left_wrist':{'x':13,'y':14,'confidence':.9,'observed':True,'source_frame_id':1}},
                                'back':{'left_wrist':{'x':50,'y':50,'confidence':.9,'observed':True,'source_frame_id':1}}}}}}
        return labels,pred

    def test_error_and_coverage_separate(self):
        a,b=self.fixtures();r=evaluate_joint_labels(a,b,4)
        front=next(x for x in r['groups'] if x['view']=='front' and x['joint']=='all')
        self.assertEqual(front['mean_error_px'],5)
        self.assertEqual(front['qualified_output_rate'],.5)
        self.assertEqual(front['incorrect_among_qualified_rate'],1)
        self.assertEqual(front['within_tolerance_among_visible_truth_rate'],0)
        self.assertEqual(front['no_qualified_output_count'],1)
        back=next(x for x in r['groups'] if x['view']=='back' and x['joint']=='all')
        self.assertIsNone(back['mean_error_px']);self.assertEqual(back['model_outputs_on_unidentifiable_truth'],1)

    def test_requested_joint_subset_controls_missing_label_denominator(self):
        a,b=self.fixtures();a['requested_joints']=['left_wrist','right_wrist']
        r=evaluate_joint_labels(a,b,4)
        self.assertEqual(r['planned_labels'],4);self.assertEqual(r['unlabelled_count'],1)

    def test_cached_points_cannot_improve_accuracy(self):
        a,b=self.fixtures();b['samples']['1']['1']['front']['left_wrist']['observed']=False
        r=evaluate_joint_labels(a,b,5)
        self.assertTrue(all(g['mean_error_px'] is None for g in r['groups']))

    def test_renaming_assisted_schema_cannot_make_independent_truth(self):
        a,b=self.fixtures()
        for change in ({'annotation_mode':'model_assisted'}, {'independent_reference':False},
                       {'model_suggestions':{'1:front:left_wrist':{'x':10,'y':10}}}):
            with self.assertRaises(ValueError):evaluate_joint_labels({**a,**change},b,5)
        a['labels']['1:front:left_wrist']['origin']='model_suggestion'
        with self.assertRaises(ValueError):evaluate_joint_labels(a,b,5)

    def test_unconfirmed_draft_cannot_produce_accuracy(self):
        a,b=self.fixtures();a['confirmed']=False
        self.assertEqual(evaluate_joint_labels(a,b,5)['groups'],[])

    def test_incompatible_source_or_coordinates_fail(self):
        a,b=self.fixtures()
        for key,value in [('source_sha256','b'*64),('coordinate_space','crop_pixels')]:
            altered=deepcopy(a);altered[key]=value
            with self.assertRaises(ValueError):evaluate_joint_labels(altered,b,5)
        a['labels']['1:front:left_wrist']['x']=float('nan')
        with self.assertRaises(ValueError):evaluate_joint_labels(a,b,5)

    def test_lower_output_coverage_cannot_improve_overall_success(self):
        a,b=self.fixtures();before=evaluate_joint_labels(a,b,5)
        b['samples']['1']['1']['front']={};after=evaluate_joint_labels(a,b,5)
        group=lambda result:next(x for x in result['groups'] if x['view']=='front' and x['joint']=='all')
        self.assertEqual(group(before)['within_tolerance_among_visible_truth_rate'],.5)
        self.assertEqual(group(after)['within_tolerance_among_visible_truth_rate'],0)
