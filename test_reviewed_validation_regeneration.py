import copy,unittest
from scripts.regenerate_reviewed_validation import mask_unknown_joints,summarize_review


class ReviewedValidationTests(unittest.TestCase):
    def test_mask_removes_unknown_without_promoting_smoothed_coordinates(self):
        rows=[{'frame_id':0,'source_time':{'timestamp_seconds':0},'kinematic_views':{
            'front':{'left_hip':{'x':10,'y':20,'observed':True}},
            'back':{'left_hip':{'x':30,'y':40,'observed':True}}}}]
        before=copy.deepcopy(rows)
        review={'requested_joints':['left_hip'],'labels':{
            '0:front:left_hip':{'visible':True,'x':11,'y':21,'review_actor':'automatic'},
            '0:back:left_hip':{'visible':False,'x':None,'y':None}}}
        masked=mask_unknown_joints(rows,review)
        self.assertEqual(rows,before)
        self.assertEqual(masked[0]['source_time'],rows[0]['source_time'])
        self.assertEqual(masked[0]['kinematic_views']['front'],rows[0]['kinematic_views']['front'])
        self.assertNotIn('left_hip',masked[0]['kinematic_views']['back'])

    def test_displacement_reports_actors_separately_not_independent_error(self):
        review={'prediction_scale':1.0,'labels':{
            '0:front:left_hip':{'visible':True,'x':13,'y':24,'reviewed':True},
            '0:back:left_hip':{'visible':False,'x':None,'y':None,'review_actor':'automatic','abstention_reasons':['uncertain']}}}
        pred={'samples':{'0':{'1.0':{'front':{'left_hip':{'x':10,'y':20}}}}}}
        result=summarize_review(review,pred)
        self.assertEqual(result['counts'],{'human':1,'human_visible':1,'automatic':1,'automatic_unknown':1})
        self.assertEqual(result['displacement_from_original_px']['human']['max'],5)
        self.assertIsNone(result['independent_accuracy'])
        self.assertEqual(result['unknown_reasons'],{'uncertain':1})


if __name__=='__main__':unittest.main()
