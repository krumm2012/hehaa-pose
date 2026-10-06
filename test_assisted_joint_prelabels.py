import unittest
from scripts.build_assisted_joint_review import populate_suggestions
from scripts.audit_assisted_joint_prelabels import audit_all_prelabels


class PrelabelTests(unittest.TestCase):
    def draft(self):
        return {'source_sha256':'a'*64,'frames':[{'frame_id':21,'width':2560,'height':1440}],
                'requested_joints':['right_hip'],'labels':{}}

    def point(self, **changes):
        return {'x':1200,'y':700,'confidence':.9,'observed':True,
                'source_frame_id':21,'confidence_source':'model',**changes}

    def pred(self, point, key='1.0'):
        return {'samples':{'21':{key:{'front':{'right_hip':point}}}}}

    def test_integer_and_decimal_scale_keys_preserve_raw_suggestions(self):
        for key in ('1','1.0'):
            doc=populate_suggestions(self.draft(),self.pred(self.point(),key))
            p=doc['labels']['21:front:right_hip']
            self.assertEqual(p['model_confidence'],.9)
            self.assertFalse(p['reviewed'])
            self.assertEqual(len(doc['review_queue']),2)
            self.assertIn('21:back:right_hip',doc['unavailable'])

    def test_stale_mirror_recovered_nonfinite_and_outside_points_stay_empty(self):
        for change in [{'source_frame_id':20},{'recovered_from_mirror':True},
                       {'x':float('nan')},{'confidence':True},{'x':2560}, {'confidence':.49}]:
            doc=populate_suggestions(self.draft(),self.pred(self.point(**change)))
            self.assertFalse(doc['labels'])
            self.assertEqual(len(doc['unavailable']),2)

    def test_ambiguous_scale_rejected(self):
        pred=self.pred(self.point());pred['samples']['21']['1']=pred['samples']['21']['1.0']
        with self.assertRaises(ValueError):populate_suggestions(self.draft(),pred)

    def test_full_audit_flags_time_without_confirming_human_review(self):
        point=self.point();doc=populate_suggestions(self.draft(),self.pred(point))
        rows=[{'frame_id':fid,'pose_observations':{'front':{'right_hip':self.point(source_frame_id=fid)}},
               'source_time':{'source_frame_id':fid,'timestamp_seconds':t,'quality':'reported','basis':'media_pts'}}
              for fid,t in [(20,.8),(21,.802)]]
        audit=audit_all_prelabels(doc,rows)
        self.assertEqual(audit['automatically_audited_count'],2)
        self.assertIn('short_source_interval',audit['flag_counts'])
        self.assertEqual(audit['human_confirmed_count'],0)
        self.assertFalse(doc['labels']['21:front:right_hip']['reviewed'])
        rows[1]['pose_observations']['front']['right_hip']['x']+=1
        with self.assertRaises(ValueError):audit_all_prelabels(doc,rows)

    def test_assisted_comparison_requires_all_declared_labels(self):
        from scripts.compare_assisted_joint_review import compare
        review={'schema':'tennis.assisted-joint-review.v1','source_sha256':'a'*64,
                'independent_reference':False,'confirmed':True,'annotator_id':'fixture',
                'coordinate_space':'original_source_pixels','frame_index_base':0,
                'frames':[{'frame_id':21,'width':2560,'height':1440}],
                'requested_joints':['right_hip'],'require_complete_review':True,
                'labels':{'21:front:right_hip':{'visible':True,'x':1200,'y':700,'reviewed':True}}}
        pred=self.pred(self.point());pred.update(source_sha256='a'*64,scales=[1.0])
        with self.assertRaises(ValueError):compare(review,pred)
        review['labels']['21:back:right_hip']={'visible':False,'x':None,'y':None,'reviewed':True}
        result=compare(review,pred)
        self.assertIsNone(result['independent_accuracy'])

    def test_left_right_flip_is_a_review_flag_not_automatic_correction(self):
        draft=self.draft();draft['requested_joints']=['left_hip','right_hip']
        def pose(fid, swapped):
            return {side+'_hip':self.point(source_frame_id=fid,x=x)
                    for side,x in [('left',1300 if swapped else 1200),('right',1200 if swapped else 1300)]}
        pred={'samples':{'21':{'1.0':{'front':pose(21,True)}}}}
        doc=populate_suggestions(draft,pred)
        rows=[{'frame_id':fid,'pose_observations':{'front':pose(fid,flip)},
               'source_time':{'source_frame_id':fid,'timestamp_seconds':t,'quality':'reported','basis':'media_pts'}}
              for fid,flip,t in [(20,False,.8),(21,True,.84)]]
        audit=audit_all_prelabels(doc,rows)
        self.assertEqual(audit['flag_counts']['left_right_image_order_changed'],2)
        self.assertEqual(doc['labels']['21:front:left_hip']['x'],1300)
        self.assertFalse(doc['labels']['21:front:left_hip']['reviewed'])


if __name__=='__main__':unittest.main()
