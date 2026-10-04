import unittest
from dataclasses import replace
from test_practice_scoring import evidence
from practice_scoring import score_event, DIMENSIONS
from dual_pose_estimator import measurement_points
from dual_view_manager import DualViewCropInfo
from dual_view_biomechanics import Keypoint, DualViewBiomechanicsEngine
import test_osd_evidence_gate as fixtures

class ProxyScalePolicyTests(unittest.TestCase):
    def test_input_flags_cannot_certify_automatic_rubric(self):
        event=evidence()
        for m in event['biomechanics']['metrics'].values():
            m.update(confidence=1,accuracy_validated=True,coach_eligible=True)
        result=score_event(event)
        self.assertIsNone(result['score'])
        self.assertTrue(all(r['reason']=='automatic_rubric_not_independently_validated' for r in result['calibration']['excluded_metrics']))
        event['practice_review']={'ratings':{k:4 for k in DIMENSIONS},'confirmed':True}
        self.assertEqual(score_event(event)['score'],80)

    def test_hip_and_foot_qualification_invariant_under_coordinate_scaling(self):
        for slope in (.25,1):
            results=[]
            for factor in (.5,1,2):
                qualify,rows,features,ext=fixtures.OsdEvidenceGateTests().fixture()
                for i,row in enumerate(rows):
                    for name,p in row['pose_observations']['front'].items():
                        if 'hip' in name: p['y']=40-i*slope
                        p['x']*=factor;p['y']*=factor
                qualify(ext,{'contact_frame':12,'contact_status':'candidate'},rows,features,100*factor)
                results.append([(ext[k]['measurement_evidence']['display_eligible'],ext[k].get(field)) for k,field in [('leg_drive','drive_ratio'),('stance','image_foot_line_angle_deg')]])
            self.assertEqual(results[0],results[1]);self.assertEqual(results[1],results[2])

    def test_independent_roi_resize_does_not_change_geometry(self):
        points={'left_shoulder':Keypoint(10,20,.9),'right_shoulder':Keypoint(110,40,.9)}
        engine=DualViewBiomechanicsEngine()
        outputs=[]
        for size in [(200,200),(400,100)]:
            info=DualViewCropInfo((0,0,200,200),(200,200),size)
            scaled={k:replace(p,x=p.x*info.scale_x,y=p.y*info.scale_y) for k,p in points.items()}
            result=engine.calculate_dual_biomechanics(points,measurement_points(scaled,info))
            outputs.append((result.robust_shoulder_turn_deg,result.scapular_retraction_ratio))
        self.assertEqual(outputs[0],outputs[1])
