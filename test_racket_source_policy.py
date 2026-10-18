import copy
from collections import deque
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

from racket_pose_detector import pose_config
from local_control_panel import ControlSettings
from main_pipe import build_argument_parser
from yolo26n_unified_detector import YOLO26nUnifiedDetector


class ExclusiveSourceTests(unittest.TestCase):
    def test_startup_exclusive_normalizes_to_active_with_fixed_source(self):
        config=pose_config({'mode':'exclusive'})
        self.assertEqual(config['mode'],'active')
        self.assertEqual(config['selection_policy'],'exclusive')
        self.assertEqual(config['selection_policy_version'],'exclusive_grip_views_v1')
        self.assertEqual(build_argument_parser().parse_args(['--racket-pose-mode','exclusive']).racket_pose_mode,'exclusive')
        self.assertEqual(ControlSettings.from_payload({'stream_id':'local_video','racket_pose_mode':'exclusive'}).racket_pose_mode,'exclusive')

    def test_only_current_finite_pose_candidates_survive_including_mirror(self):
        from racket_source_policy import current_pose_candidates
        candidates=[{'source':'racket_pose_model','source_frame_id':7,'box':[1,2,30,40],'confidence':.8,'observed':True},
                    {'source':'racket_pose_model','source_frame_id':7,'box':[50,2,70,40],'confidence':.7,'observed':True},
                    {'source':'legacy','source_frame_id':7,'box':[1,2,30,40],'confidence':.99,'observed':True}]
        observation={'status':'observed','frame_id':7,'diagnostics':{'nonfinite_candidates':0},'candidates':candidates}
        frozen=copy.deepcopy(observation)
        result=current_pose_candidates(observation,7)
        self.assertEqual(len(result),2)
        result[0]['box'][0]=999
        self.assertEqual(observation,frozen)
        self.assertEqual(current_pose_candidates(observation,8),[])
        self.assertEqual(current_pose_candidates({**observation,'status':'pending_or_skipped'},7),[])
        self.assertEqual(current_pose_candidates({**observation,'diagnostics':{'nonfinite_candidates':1}},7),[])
        self.assertEqual(current_pose_candidates({**observation,'candidates':[]},7),[])

    def test_external_racket_source_skips_legacy_selection_and_defers_ball_support(self):
        detector=YOLO26nUnifiedDetector.__new__(YOLO26nUnifiedDetector)
        detector.config={};detector._coreml_input_names={'image'};detector.detection_times=deque()
        detector._preprocess=Mock(return_value=object());detector.model=Mock();detector.model.predict.return_value={}
        detector.last_parse_diagnostics={'ball':{},'racket':{}};detector.last_ball_diagnostics={}
        ball={'position':(5,5),'confidence':.8};old={'box':[0,0,9,9],'confidence':.99}
        detector._parse_predictions=Mock(return_value=([ball],[old]));detector.roi_manager=None
        detector._filter_static_balls=Mock(return_value=[ball]);detector._select_primary_racket=Mock()
        with patch('yolo26n_unified_detector.recover_overlay_detections',return_value={'balls':[],'rackets':[],'diagnostics':{}}):
            balls,rackets,_=detector.detect_unified(np.zeros((12,12,3),np.uint8),racket_source='external')
        self.assertEqual(balls,[ball]);self.assertEqual(rackets,[])
        detector._select_primary_racket.assert_not_called();detector._filter_static_balls.assert_not_called()
        fresh={'source':'racket_pose_model','confidence':.4,'box':[1,1,8,8]}
        detector.finalize_external_ball_candidates(balls,[fresh],confidence_threshold=.25)
        detector._filter_static_balls.assert_called_once_with([ball],[fresh])
        self.assertEqual(detector.last_ball_diagnostics['racket_support_source'],'racket_pose_model')

    def test_exclusive_cannot_start_sampling_realtime_worker(self):
        from racket_pose_runtime import RacketPoseRuntime
        with patch('racket_pose_runtime.mp.get_context') as context:
            with self.assertRaisesRegex(ValueError,'continuous'):
                RacketPoseRuntime({'mode':'exclusive'},'realtime')
            context.assert_not_called()
        with self.assertRaises(ValueError):
            ControlSettings.from_payload({'stream_id':'local_video','racket_pose_mode':'exclusive','processing_mode':'realtime'})

    def test_resolver_cannot_leak_legacy_or_stale_observed_output(self):
        from racket_source_policy import validate_exclusive_output
        with self.assertRaises(RuntimeError):
            validate_exclusive_output([{'source':'legacy','observed':True,'source_frame_id':7}],7)
        with self.assertRaises(RuntimeError):
            validate_exclusive_output([{'source':'racket_pose_model','observed':True,'source_frame_id':6}],7)
        validate_exclusive_output([{'source':'racket_pose_model','observed':False,'source_frame_id':6}],7)
        validate_exclusive_output([],7)

    def test_exclusive_keeps_same_frame_grip_gate_for_both_views(self):
        from racket_source_policy import associated_pose_candidates
        def candidate(x,score=.9):
            return {'source':'racket_pose_model','source_frame_id':7,'box':[x-10,0,x+10,30],
                'confidence':.8,'observed':True,'keypoints':{'grip_center':{
                    'x':x,'y':10,'confidence':score,'observed':True,'source_frame_id':7}}}
        observation={'status':'observed','frame_id':7,'candidates':[candidate(10),candidate(80,.1)]}
        wrist=lambda x: {'x':x,'y':10,'confidence':.9,'observed':True,'source_frame_id':7}
        views={'front':[wrist(10)],'back':[wrist(80)]}
        config=pose_config({'mode':'exclusive'})
        result=associated_pose_candidates(observation,7,views,(100,100),config)
        self.assertEqual(len(result),1);self.assertEqual(result[0]['association_view'],'front')
        observation['candidates'][1]=candidate(80)
        result=associated_pose_candidates(observation,7,views,(100,100),config)
        self.assertEqual([r['association_view'] for r in result],['front','back'])
        views['front'][0]['observed']=False
        self.assertEqual([r['association_view'] for r in associated_pose_candidates(observation,7,views,(100,100),config)],['back'])

    def test_mirror_and_temporal_recovery_preserve_new_source_but_are_not_raw(self):
        from racket_resolution import RacketResolver
        resolver=RacketResolver()
        point=lambda x,y: SimpleNamespace(x=x,y=y,conf=.95,observed=True,source_frame_id=7)
        front={'right_wrist':point(1800,900)};back={'right_wrist':point(850,100)}
        candidate={'source':'racket_pose_model','source_frame_id':7,'box':[800,50,900,150],'confidence':.8}
        with patch('racket_resolution.map_mirror_racket_to_front',return_value=[1700,800,1800,900]):
            result=resolver.resolve([candidate],front_pose=front,back_pose=back,frame_idx=7)
        self.assertEqual(result.status,'recovered')
        self.assertEqual(result.rackets[0]['source'],'racket_pose_model')
        self.assertFalse(result.rackets[0]['observed'])
        self.assertFalse(result.rackets[0]['measurement_eligible'])
        later=resolver.resolve([],front_pose=front,back_pose={},frame_idx=8)
        self.assertEqual(later.status,'decayed')
        self.assertEqual(later.rackets[0]['source'],'racket_pose_model')
        self.assertFalse(later.rackets[0]['observed'])

    def test_back_association_never_becomes_a_direct_front_observation(self):
        from racket_resolution import RacketResolver
        wrist=SimpleNamespace(x=1800,y=900,conf=.95,observed=True,source_frame_id=7)
        candidate={'source':'racket_pose_model','source_frame_id':7,'association_view':'back',
                   'box':[1770,870,1830,930],'confidence':.8}
        result=RacketResolver().resolve([candidate],front_pose={'right_wrist':wrist},frame_idx=7)
        self.assertEqual(result.status,'missing')

if __name__=='__main__':unittest.main()
