import unittest
from realtime_swing_runtime import build_impact_telemetry_card

class OsdEvidenceGateTests(unittest.TestCase):
    def test_realtime_committed_tail_remains_contact_measurement_context(self):
        from unittest.mock import patch
        from realtime_swing_pipeline import RealtimeSwingEventEngine
        _, rows, _, _ = self.fixture()
        engine = RealtimeSwingEventEngine(fps=25, min_event_frames=4, min_event_gap=0)
        engine._frames.extend(rows)
        engine._latest_frame_id = 15
        engine._events.append({'event_id':1,'start_frame':0,'end_frame':5,'peak_frame':1,'stroke_type':'Forehand'})
        candidate = {'start_frame':10,'end_frame':15,'contact_frame':12,'peak_frame':12,
                     'stroke_type':'Forehand','contact_status':'candidate'}
        with patch('swing_event_analyzer.segment_swing_events', return_value={'events':[candidate],'frame_trace':[]}) as segment:
            emitted = engine.flush()
        self.assertEqual(segment.call_args.args[0][0]['frame_id'], 6)
        evidence = emitted[0]['extended_biomechanics']['brush_angle']['measurement_evidence']
        self.assertEqual(evidence['window_frames'], list(range(13)))

    def test_unqualified_numbers_do_not_appear_as_osd_measurements(self):
        event={'contact_status':'unknown','extended_biomechanics':{
            'brush_angle':{'low_to_high_angle_deg':18.1,'drop_depth_ratio':1.26,'confidence':0},
            'stance':{'image_foot_line_angle_deg':86.8,'confidence':0},
            'leg_drive':{'drive_ratio':.03,'confidence':0},
            'kinematic_sequence':{'cross_validation':{'status':'single_view'},'sequence_quality':'UNRESOLVED_AT_FRAME_RATE'}}}
        card=build_impact_telemetry_card(event)
        self.assertIsNone(card['brush_angle_deg'])
        self.assertIsNone(card['foot_line_angle_deg'])
        self.assertIsNone(card['leg_drive_ratio'])
        self.assertIn('先后难辨',card['kinematic_sequence_text'])

    def test_coincident_chord_keeps_qualified_zero_rise(self):
        from osd_evidence import display_value
        qualify,rows,features,ext=self.fixture()
        for i,f in enumerate(features): f['racket_measurement_point']=[i*10,i*2]
        qualify(ext,{'contact_frame':12,'contact_status':'candidate'},rows,features,100)
        self.assertIsNone(display_value(ext['brush_angle'],'low_to_high_angle_deg'))
        self.assertEqual(display_value(ext['brush_angle'],'drop_depth_ratio'),0)
        rows[5]['rackets']=[];rows[6]['rackets']=[];rows[7]['rackets']=[]
        qualify(ext,{'contact_frame':12,'contact_status':'candidate'},rows,features,100)
        self.assertIsNone(display_value(ext['brush_angle'],'drop_depth_ratio'))

    def test_event_start_does_not_clip_contact_time_window(self):
        from swing_biomechanics import aggregate_event_biomechanics
        from swing_motion_features import extract_motion_features
        _, rows, _, _ = self.fixture()
        result=aggregate_event_biomechanics(
            {'start_frame':10,'contact_frame':12,'end_frame':15,'contact_status':'candidate'},
            rows,extract_motion_features(rows))
        window=result['extended_biomechanics']['brush_angle']['measurement_evidence']['window_frames']
        self.assertEqual(window,list(range(13)))

    def fixture(self):
        from osd_evidence import qualify_extended_observations
        rows=[];features=[]
        for i in range(16):
            pts={name:{'x':x,'y':y-i,'confidence':.9,'observed':True,'confidence_source':'model'} for name,x,y in [
                ('left_ankle',0,100),('right_ankle',100,100),('left_hip',0,40),('right_hip',100,40)]}
            rows.append({'frame_id':i,'source_time':{'schema_version':'tennis.source-time.v1',
                'source_kind':'video_file','source_frame_id':i,'timestamp_seconds':i*.04,'basis':'media_pts','quality':'reported'},
                         'pose_observations':{'front':pts},'rackets':[{'confidence':.9,'observed':True}]})
            features.append({'frame_id':i,'racket_measurement_point':[i*10,100-i*2]})
        ext={'brush_angle':{},'stance':{},'leg_drive':{}}
        return qualify_extended_observations,rows,features,ext

    def test_qualified_measurements_retain_sources_without_claiming_accuracy(self):
        qualify,rows,features,ext=self.fixture()
        qualify(ext,{'contact_frame':12,'contact_status':'candidate'},rows,features,100)
        self.assertTrue(ext['brush_angle']['measurement_evidence']['display_eligible'])
        self.assertFalse(ext['brush_angle']['measurement_evidence']['accuracy_validated'])
        self.assertEqual(ext['brush_angle']['measurement_evidence']['path_endpoint_frames'],[0,12])
        self.assertEqual(ext['stance']['image_foot_line_angle_deg'],0)
        self.assertEqual(ext['leg_drive']['drive_px'],12)

    def test_sparse_racket_track_cannot_generate_chord_even_with_strong_scores(self):
        qualify,rows,features,ext=self.fixture()
        for i in (1,2,3,4,5,6): rows[i]['rackets']=[]
        qualify(ext,{'contact_frame':12,'contact_status':'candidate'},rows,features,100)
        self.assertIsNone(ext['brush_angle']['low_to_high_angle_deg'])
        self.assertIn('low_observation_coverage',ext['brush_angle']['measurement_evidence']['reasons'])

    def test_small_hip_displacement_is_not_reported_as_effort_percentage(self):
        qualify,rows,features,ext=self.fixture()
        for i,row in enumerate(rows):
            for p in row['pose_observations']['front'].values(): p['y'] += i*.95
        qualify(ext,{'contact_frame':12,'contact_status':'candidate'},rows,features,100)
        self.assertIsNone(ext['leg_drive']['drive_ratio'])
        self.assertIn('below_motion_resolution_guard',ext['leg_drive']['measurement_evidence']['reasons'])

    def test_cached_ankles_cannot_qualify_osd_angle(self):
        qualify,rows,features,ext=self.fixture()
        for row in rows: row['pose_observations']['front']['left_ankle']['observed']=False
        qualify(ext,{'contact_frame':12,'contact_status':'candidate'},rows,features,100)
        self.assertIsNone(ext['stance']['image_foot_line_angle_deg'])

    def test_partial_sequence_missing_racket_is_not_full_chain(self):
        from osd_evidence import sequence_osd_label
        seq={'cross_validation':{'status':'single_view'}, 'latency_hip_to_shoulder_ms':0,
             'peak_time_uncertainty_ms':42,'racket_peak_frame':None}
        self.assertEqual(sequence_osd_label(seq),'髋肩先后难辨·缺拍峰')
