"""Physical invariants and abstention at production module boundaries."""
import unittest
from dual_view_biomechanics import DualViewBiomechanicsEngine
from swing_motion_features import extract_motion_features
from swing_event_classifier import classify_swing_event
from swing_biomechanics import aggregate_event_biomechanics


class ObservationEligibilityTests(unittest.TestCase):
    def test_mismatched_source_frame_points_are_not_current_measurements(self):
        from observation_policy import measurement_pose
        frame={'frame_id':5,'pose_observations':{'front':{'right_wrist':{
            'x':10,'y':20,'confidence':.9,'observed':True,'source_frame_id':4}}}}
        self.assertEqual(measurement_pose(frame),{})
        from test_osd_evidence_gate import OsdEvidenceGateTests
        qualify,rows,features,ext=OsdEvidenceGateTests().fixture()
        for row in rows:
            for point in row['pose_observations']['front'].values(): point['source_frame_id']=row['frame_id']-1
            row['rackets'][0]['source_frame_id']=row['frame_id']-1
        qualify(ext,{'contact_frame':12,'contact_status':'candidate'},rows,features,100)
        self.assertIsNone(ext['stance']['image_foot_line_angle_deg'])
        self.assertIsNone(ext['brush_angle']['drop_depth_ratio'])
        for row in rows: row['racket']=[0,0,20,20];row['rackets'][0]['box']=[0,0,20,20]
        self.assertTrue(all(f['racket_measurement_point'] is None for f in extract_motion_features(rows)))

    def test_box_resize_has_no_translation(self):
        frames = [{'frame_id': i, 'timestamp': i*.04, 'racket': box}
                  for i, box in enumerate([[80,80,120,120], [60,60,140,140]])]
        features = extract_motion_features(frames)
        self.assertEqual(features[-1]['racket_center'], (100,100))
        self.assertEqual(features[-1]['racket_speed'], 0)

    def test_missing_pose_is_not_a_measurement(self):
        result = DualViewBiomechanicsEngine().calculate_dual_biomechanics({}, {})
        self.assertIsNone(result.robust_shoulder_turn_deg)
        self.assertIsNone(result.takeback_depth_ratio)
        self.assertIsNone(result.scapular_retraction_ratio)

    def test_unknown_class_is_not_forehand(self):
        result = classify_swing_event([{'frame_id': i} for i in range(3)])
        self.assertEqual(result['stroke_type'], 'Unknown')

    def test_weak_votes_do_not_become_high_confidence(self):
        frames = [{'frame_id': i, 'timestamp': i*.04, 'dual_view_biomechanics': {
            'shot_classification': {'stroke_type': label, 'confidence': .01}}}
            for i, label in enumerate(['Forehand','Forehand','Backhand'])]
        result = classify_swing_event(extract_motion_features(frames))
        self.assertEqual(result['stroke_type'], 'Unknown')

    def test_remote_rebound_is_not_contact(self):
        result = classify_swing_event([{'frame_id':i, 'ball':[1000,y],
            'ball_racket_distance':1000} for i,y in enumerate([0,30,0])])
        self.assertFalse(result['is_valid_contact'])

    def test_no_ball_detection_is_unknown_not_shadow(self):
        result = classify_swing_event([{'frame_id': i} for i in range(3)])
        self.assertFalse(result['is_shadow_swing'])
        self.assertEqual(result['contact_status'], 'unknown')

    def test_legacy_healed_pose_cannot_change_measurements(self):
        pose = {'right_wrist':[100,100], 'right_elbow':[90,100], 'right_shoulder':[80,100]}
        frame = {'frame_id':0, 'timestamp':0, 'pose':pose,
                 'healed_pose':{**pose,'right_wrist':[90,110]}}
        self.assertEqual(extract_motion_features([frame])[0]['arm_extension_deg'],180)

    def test_cached_and_mirror_points_cannot_supply_joint_angle(self):
        from observation_policy import measurement_pose
        for extra in ({'observed': False}, {'recovered_from_mirror': True}, {'confidence_source': 'unavailable'}):
            point = {'x':100, 'y':100, 'confidence':.9, 'observed':True, **extra}
            frame = {'pose': {'right_wrist':[100,100]}, 'pose_observations': {'front': {'right_wrist':point}},
                     'metrics': {'arm_extension':150}, 'frame_id':0}
            self.assertEqual(measurement_pose(frame), {})
            self.assertIsNone(extract_motion_features([frame])[0]['arm_extension_deg'])

    def test_raw_decoder_retains_score_with_one_inference(self):
        from unittest.mock import Mock
        import numpy as np
        from pose_estimator_yolo26 import PoseEstimatorYOLO26
        estimator = PoseEstimatorYOLO26.__new__(PoseEstimatorYOLO26)
        estimator._preprocess_frame = Mock(return_value=None)
        estimator._apply_temporal_smoothing = Mock(side_effect=AssertionError('raw observations must not be smoothed'))
        estimator.original_size = estimator.input_size = (640,640)
        estimator.confidence_threshold = .25
        estimator.keypoint_confidence = .3
        estimator.keypoint_names = [str(i) for i in range(17)]
        estimator.roi_manager = None
        detection = np.array([0,0,100,100,.9,0] + [20,30,.63]*17).reshape(1,1,57)
        estimator.model = Mock()
        estimator.model.predict.return_value = {'output':detection}
        points = estimator.get_keypoints_with_confidence(np.zeros((640,640,3)))
        self.assertEqual(points[0]['0'], (20,30,.63))
        estimator.model.predict.assert_called_once()
        estimator._apply_temporal_smoothing.assert_not_called()

    def test_uncertain_events_do_not_get_automatic_scores_or_technique_corrections(self):
        from test_practice_scoring import evidence
        from practice_scoring import score_event
        from local_realtime_coach import LocalRealtimeCoach
        for overrides in ({'stroke_type':'Unknown'}, {'contact_status':'unknown'}, {'stroke_type':'Two-Handed Backhand'}):
            event = {**evidence(), **overrides}
            self.assertIsNone(score_event(event)['score'])
            self.assertTrue(all(a['category'] == 'review' for a in LocalRealtimeCoach().advise_all(event)))

    def test_cached_racket_box_is_not_a_detection(self):
        frame = {'frame_id':0, 'racket':[80,80,120,120],
                 'rackets':[{'box':[80,80,120,120], 'observed':False, 'confidence':.9}]}
        self.assertIsNone(extract_motion_features([frame])[0]['racket_center'])
