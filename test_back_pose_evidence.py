import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from dual_view_renderer import DualViewRenderer
from dual_view_biomechanics import Keypoint, DualViewBiomechanicsEngine
from dual_pose_estimator import DualPoseEstimator

class BackPoseEvidenceTests(unittest.TestCase):
    def test_low_score_wrist_has_no_bone(self):
        points={'right_elbow':Keypoint(10,10,.99),'right_wrist':Keypoint(80,20,.254)}
        with patch('dual_view_renderer.cv2.line') as line:
            DualViewRenderer().draw_skeleton(np.zeros((100,100,3),np.uint8),points,True)
        line.assert_not_called()

    def test_cached_and_recovered_bones_are_dashed(self):
        for recovered in (False,True):
            points={'right_elbow':Keypoint(10,10,.99),'right_wrist':Keypoint(80,20,.9,observed=False,recovered_from_mirror=recovered)}
            with patch('dual_view_renderer.cv2.line') as line:
                DualViewRenderer().draw_skeleton(np.zeros((100,100,3),np.uint8),points,True)
            self.assertGreater(line.call_count,1)
            self.assertTrue(all(call.args[4]==1 for call in line.call_args_list))

    def test_healing_requires_fresh_noncollapsed_anchors(self):
        engine=DualViewBiomechanicsEngine()
        shoulders={'left_shoulder':Keypoint(10,10,.9),'right_shoulder':Keypoint(50,10,.9)}
        back={**shoulders,'right_wrist':Keypoint(70,40,.9)}
        self.assertIn('right_wrist',engine.heal_occluded_pose(shoulders,back)[1])
        self.assertEqual(engine.heal_occluded_pose({},back)[1],[])
        for change in ({'observed':False},{'conf':.2},{'x':49}):
            front={**shoulders,'left_shoulder':replace(shoulders['left_shoulder'],**change)}
            self.assertEqual(engine.heal_occluded_pose(front,back)[1],[])
        back['right_wrist']=replace(back['right_wrist'],observed=False)
        self.assertEqual(engine.heal_occluded_pose(shoulders,back)[1],[])

    def test_mirror_region_and_identity_continuity(self):
        estimator=DualPoseEstimator.__new__(DualPoseEstimator)
        estimator._back_missing_count=0
        def person(x,y):
            return {'left_shoulder':(x,y,.9),'right_shoulder':(x+40,y,.9),
                    'left_hip':(x,y+60,.9),'right_hip':(x+40,y+60,.9)}
        allowed=lambda x,y:x<300
        self.assertEqual(estimator._select_back_candidate([person(400,0),person(50,80)],None,allowed),1)
        self.assertEqual(estimator._select_back_candidate([person(200,5),person(55,82)],None,allowed),1)
        self.assertIsNone(estimator._select_back_candidate([person(200,5)],None,allowed))
        estimator._back_missing_count=3
        self.assertEqual(estimator._select_back_candidate([person(200,5)],None,allowed),0)

    def test_renderer_projection_preserves_observation_provenance(self):
        from types import SimpleNamespace
        from dual_view_manager import DualViewCropInfo
        info=DualViewCropInfo((0,0,100,100),(100,100),(100,100))
        img=np.zeros((100,100,3),np.uint8)
        point=Keypoint(30,40,.8,observed=False,source_frame_id=12,confidence_source='model')
        frame=SimpleNamespace(front_frame=img,back_frame=img,front_info=info,back_info=info)
        pose=SimpleNamespace(fused_pose_orig={'right_wrist':point},back_pose_orig={'right_wrist':point})
        renderer=DualViewRenderer(show_hud=False,mask_backview_eyes=False)
        seen=[]
        def capture(image,points,**kwargs):
            seen.append(points['right_wrist']);return image
        with patch.object(renderer,'draw_skeleton',side_effect=capture):
            renderer.render_dual_frame(frame,pose)
        for projected in seen:
            self.assertFalse(projected.observed)
            self.assertEqual(projected.source_frame_id,12)
            self.assertEqual(projected.conf,.8)
            self.assertEqual(projected.confidence_source,'model')

    def test_partial_foreground_person_inside_mirror_polygon_is_rejected(self):
        estimator=DualPoseEstimator.__new__(DualPoseEstimator)
        estimator._back_missing_count=0
        cropped={'left_shoulder':(100,100,.99),'right_shoulder':(140,100,.99),
                 'left_eye':(110,80,.99),'right_eye':(120,80,.99)}
        self.assertIsNone(estimator._select_back_candidate([cropped],None,lambda x,y:True))

    def test_held_pose_keeps_original_coordinates_when_crop_moves(self):
        from dual_view_manager import DualViewCropInfo
        estimator=DualPoseEstimator.__new__(DualPoseEstimator)
        old=DualViewCropInfo((100,100,200,200),(100,100),(200,200),True)
        new=DualViewCropInfo((120,80,220,180),(100,100),(100,100),True)
        point=Keypoint(30,60,.9,source_frame_id=12,confidence_source='model')
        estimator._last_valid_back_pose={'right_wrist':point}
        estimator._last_valid_back_crop=old
        held=estimator._cached_pose('back',new)['right_wrist']
        self.assertEqual(old.map_to_original(point.x,point.y),new.map_to_original(held.x,held.y))
        self.assertFalse(held.observed);self.assertEqual(held.source_frame_id,12)

    def test_roi_tracking_does_not_treat_held_pose_as_new_measurement(self):
        from dual_view_manager import DualViewManager
        manager=DualViewManager()
        points={str(i):Keypoint(i*20,i*20,.9,observed=False) for i in range(5)}
        self.assertIsNone(manager.update_player_from_keypoints(points))
        self.assertEqual(manager._tracking_lost_counter,1)

    def test_back_identity_survives_uniform_scaling_and_side_on_shoulders(self):
        for scale in (1., .5, .25):
            estimator=DualPoseEstimator.__new__(DualPoseEstimator)
            estimator._back_missing_count=0
            for width in (30, 11.54, 1):
                pose={name:(x*scale,y*scale,.99) for name,x,y in (
                    ('left_shoulder',100,100),('right_shoulder',100+width,100),
                    ('left_hip',100,160),('right_hip',115,160))}
                self.assertEqual(estimator._select_back_candidate([pose],None,lambda x,y:True),0)

    def test_degenerate_torso_is_not_an_identity_anchor(self):
        estimator=DualPoseEstimator.__new__(DualPoseEstimator)
        estimator._back_missing_count=0
        pose={name:(100,100,.99) for name in ('left_shoulder','right_shoulder','left_hip','right_hip')}
        self.assertIsNone(estimator._select_back_candidate([pose],None,lambda x,y:True))
