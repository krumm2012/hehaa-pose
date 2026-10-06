import copy
import unittest
from assisted_joint_optimization import optimize_remaining


class OptimizationTests(unittest.TestCase):
    def setup_review(self):
        frames=[{'frame_id':i,'width':2560,'height':1440} for i in (0,1,2)]
        labels={f'{i}:front:left_hip':{'visible':True,'x':100+i,'y':200,'reviewed':False} for i in (0,1,2)}
        labels.update({f'{i}:back:left_hip':{'visible':True,'x':100+i,'y':200,'reviewed':False} for i in (0,1,2)})
        review={'frames':frames,'requested_joints':['left_hip'],'labels':labels,
                'review_queue':[{'key':key,'priority':2} for key in labels]}
        rows=[{'frame_id':i,'source_time':{'basis':'media_pts','quality':'reported','source_frame_id':i,'timestamp_seconds':i*.04},
               'pose_observations':{view:{'left_hip':{'x':100+i,'y':200,'confidence':.99,'source_frame_id':i,'observed':True}}
                                    for view in ('front','back')}} for i in (0,1,2)]
        return review,rows,{'samples':[]}

    def test_human_unknown_and_coordinates_preserved_verbatim(self):
        review,rows,audit=self.setup_review()
        human={'visible':False,'x':None,'y':None,'reviewed':True,'origin':'human_review','custom_note':'遮挡'}
        review['labels']['0:front:left_hip']=human
        saved=copy.deepcopy(review)
        result,report=optimize_remaining(review,rows,audit)
        self.assertEqual(result['labels']['0:front:left_hip'],human)
        self.assertEqual(review,saved)
        self.assertFalse(result['labels']['1:front:left_hip']['visible'])
        self.assertIsNone(result['labels']['1:front:left_hip']['x'])
        self.assertEqual(report['counts']['human_preserved'],1)
        self.assertFalse(result['confirmed'])

    def test_stale_low_score_and_identity_flags_abstain(self):
        for kind in ('stale','low','identity'):
            review,rows,audit=self.setup_review()
            if kind=='stale':rows[1]['pose_observations']['front']['left_hip']['observed']=False
            if kind=='low':rows[1]['pose_observations']['front']['left_hip']['confidence']=.8
            if kind=='identity':audit={'samples':[{'key':'1:front:left_hip','flags':['left_right_image_order_changed']}]}
            result,_=optimize_remaining(review,rows,audit)
            self.assertFalse(result['labels']['1:front:left_hip']['visible'])
            self.assertEqual(result['labels']['1:front:left_hip']['review_actor'],'automatic')

    def test_adjustments_bounded_and_not_human_confirmation(self):
        review,rows,audit=self.setup_review();rows[1]['pose_observations']['front']['left_hip']['x']+=5
        result,_=optimize_remaining(review,rows,audit);p=result['labels']['1:front:left_hip']
        self.assertLessEqual(abs(p['x']-106),2)
        self.assertEqual(p['review_actor'],'automatic')
        self.assertFalse(p['measurement_eligible'])
        self.assertFalse(result['human_review_complete'])
        self.assertFalse(result['independent_reference'])


if __name__=='__main__':unittest.main()
