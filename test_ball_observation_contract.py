import unittest

from ball_observation_contract import stamp_ball_detections, selected_ball_observation, measurement_ball
from frame_processor import FrameProcessor
from swing_motion_features import extract_motion_features


class BallObservationContractTests(unittest.TestCase):
    def test_output_timing_keeps_clock_meaning_and_unknown_intervals(self):
        from analysis_data_contracts import stamp_render_output
        frame={'timing':{'captured_at_unix_ns':1_000_000,
                       'analysis_completed_at_unix_ns':5_000_000}}
        stamp_render_output(frame,7_000_000,10_000_000)
        self.assertEqual(frame['timing']['render_ms'],2)
        self.assertEqual(frame['timing']['capture_to_output_submit_ms'],9)
        self.assertIsNone(frame['timing']['capture_to_inference_start_ms'])
        self.assertIn('not_display_photons',frame['timing']['output_clock_semantics'])

    def detection(self):
        return {'position':[10,20],'confidence':.21,'model_confidence':.83,'source':'model_detection'}

    def test_raw_score_and_selection_score_remain_separate_through_frame_record(self):
        detection=stamp_ball_detections([self.detection()],7)[0]
        processor=FrameProcessor.__new__(FrameProcessor);processor.fps=25
        frame=processor.build_frame_record(7,'Forehand',(10,20),[],[],{},ball_detection=detection)
        observation=frame['ball_observation']
        self.assertEqual(frame['ball'],[10,20])
        self.assertEqual(observation['model_confidence'],.83)
        self.assertEqual(observation['selection_score'],.21)
        self.assertEqual(observation['source_frame_id'],7)
        self.assertEqual(measurement_ball(frame),([10,20],'fresh_model_observation'))

    def test_overlay_prediction_unknown_and_stale_cannot_supply_contact(self):
        for change in ({'source':'overlay_ball_outline','confidence':.99},
                       {'source':'prediction'}, {'source':None}, {'observed':False},
                       {'source_frame_id':6}, {'model_confidence':None}):
            detection=stamp_ball_detections([{**self.detection(),**change}],7)[0]
            observation=selected_ball_observation(detection,7)
            frame={'frame_id':7,'timestamp':.28,'ball':[10,20],'ball_observation':observation,
                   'racket':[0,0,20,40],'pose':{'right_wrist':[10,20]}}
            feature=extract_motion_features([frame])[0]
            self.assertIsNone(feature['ball'])
            self.assertEqual(feature['contact_score'],0)
            self.assertFalse(observation['accuracy_validated'])

    def test_legacy_xy_remains_explicitly_unverified(self):
        self.assertEqual(measurement_ball({'frame_id':7,'ball':[10,20]}),([10,20],'legacy_source_unverified'))

    def test_serialized_eligibility_cannot_bypass_geometry_or_score_validation(self):
        observation=selected_ball_observation(stamp_ball_detections([self.detection()],7)[0],7)
        for change in ({'position':[float('nan'),20]}, {'model_confidence':None},
                       {'model_confidence':2}, {'coordinate_space':'crop_pixels'}):
            self.assertIsNone(measurement_ball({'frame_id':7,'ball_observation':{**observation,**change}})[0])

    def test_contact_does_not_use_interpolated_racket(self):
        frames=[{'frame_id':i,'timestamp':i*.04,'racket':box,'ball':[10,20]} for i,box in enumerate(([0,0,20,40],None,[0,0,20,40]))]
        feature=extract_motion_features(frames)[1]
        self.assertEqual(feature['racket_center_source'],'interpolated')
        self.assertIsNone(feature['ball_racket_distance'])
        self.assertEqual(feature['contact_score'],0)
