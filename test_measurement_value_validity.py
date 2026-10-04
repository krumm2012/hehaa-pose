"""Invalid raw observations must not become finite, high-quality metrics."""
from copy import deepcopy
import json
import unittest

from observation_policy import measurement_pose
from swing_motion_features import extract_motion_features
from swing_biomechanics import aggregate_event_biomechanics
from dual_view_biomechanics import DualViewBiomechanicsEngine, Keypoint


def row(fid, bad=float('nan'), observed=False):
    pose = {'left_shoulder':[0, 0], 'right_shoulder':[100, 0],
            'left_hip':[0, 100], 'right_hip':[100, 100],
            'right_elbow':[bad, 50], 'right_wrist':[130, 90],
            'left_knee':[bad, 160], 'left_ankle':[0, 220],
            'right_knee':[bad, 160], 'right_ankle':[100, 220]}
    r={'frame_id':fid,'timestamp':fid*.04,'pose':pose,'has_pose':True,
       'source_time':{'schema_version':'tennis.source-time.v1','source_kind':'video_file',
          'source_frame_id':fid,'timestamp_seconds':fid*.04,'basis':'media_pts','quality':'reported'}}
    if observed:
        r['pose_observations']={'front':{k:{'x':v[0],'y':v[1],'confidence':.99,
            'observed':True,'source_frame_id':fid} for k,v in pose.items()}}
    return r


def event():
    return {'event_id':1,'stroke_type':'Forehand','start_frame':0,'peak_frame':2,
        'contact_frame':2,'end_frame':4,'quality_flags':{'pose_frame_ratio':1}}


class MeasurementValueValidityTests(unittest.TestCase):
    def test_fresh_high_score_nonfinite_joint_is_not_a_measurement(self):
        r=row(0,observed=True);original=deepcopy(r)
        self.assertNotIn('right_elbow',measurement_pose(r))
        self.assertEqual(r['pose_observations']['front']['right_elbow']['confidence'],.99)
        self.assertEqual(r['frame_id'],original['frame_id'])

    def test_invalid_joint_cannot_be_clamped_into_zero_arm_angle(self):
        for observed in (False,True):
            with self.subTest(observed=observed):
                rows=[row(i,observed=observed) for i in range(5)]
                features=extract_motion_features(rows)
                self.assertTrue(all(f['arm_extension_deg'] is None for f in features))
                metric=aggregate_event_biomechanics(event(),rows,features)['metrics']['arm_extension']
                self.assertIsNone(metric['value'])
                self.assertEqual(metric['confidence'],0)

    def test_invalid_knee_cannot_be_clamped_into_180_degree_flexion(self):
        rows=[row(i) for i in range(5)]
        features=extract_motion_features(rows)
        metric=aggregate_event_biomechanics(event(),rows,features)['metrics']['preparation_knee_flexion']
        self.assertIsNone(metric['value'])
        self.assertEqual(metric['confidence'],0)

    def test_nonfinite_torso_cannot_supply_displacement(self):
        rows=[row(i) for i in range(5)]
        for r in rows:
            r['pose']['right_shoulder']=[float('inf'),0]
        features=extract_motion_features(rows)
        metrics=aggregate_event_biomechanics(event(),rows,features)['metrics']
        self.assertIsNone(metrics['weight_transfer']['value'])
        self.assertIsNone(metrics['balance_drift']['value'])

    def test_nonfinite_legacy_derived_metric_is_not_an_observed_angle(self):
        rows=[{'frame_id':0,'timestamp':0,'metrics':{'swing_motion':{'arm_ext':float('inf')}}}]
        self.assertIsNone(extract_motion_features(rows)[0]['arm_extension_deg'])

    def test_invalid_model_score_does_not_authorize_a_joint(self):
        for bad in (float('nan'),float('inf'),1.01,-1,True,'0.99'):
            with self.subTest(score=bad):
                r=row(0,observed=True)
                r['pose_observations']['front']['right_wrist']['confidence']=bad
                self.assertNotIn('right_wrist',measurement_pose(r))

    def test_collector_does_not_export_fake_joint_angles(self):
        from swing_coach_data_collector import _body_metrics
        rows=[row(i) for i in range(5)];features=extract_motion_features(rows)
        metrics=_body_metrics({r['frame_id']:r for r in rows},
                              {f['frame_id']:f for f in features},0,2,2,4)
        self.assertIsNone(metrics['pose_angles_at_contact']['right_elbow_deg'])
        self.assertIsNone(metrics['pose_angles_at_contact']['left_knee_deg'])

    def test_nonfinite_or_reversed_racket_box_is_not_a_candidate_point(self):
        for box in ([0,0,float('inf'),50],[float('nan'),0,10,10],[20,0,10,10],[0,0,0,10]):
            with self.subTest(box=box):
                r=row(0);r['racket']=box
                self.assertIsNone(extract_motion_features([r])[0]['racket_measurement_point'])

    def test_invalid_primary_racket_score_cannot_borrow_legacy_box(self):
        r=row(0);r['racket']=[0,0,10,10]
        r['rackets']=[{'box':r['racket'],'observed':True,'confidence':float('inf'),'source_frame_id':0}]
        self.assertIsNone(extract_motion_features([r])[0]['racket_measurement_point'])

    def test_dual_view_cannot_classify_a_nonfinite_shoulder(self):
        pose=row(0)['pose'];pose['right_shoulder']=[float('nan'),0]
        result=DualViewBiomechanicsEngine().calculate_dual_biomechanics(pose,{})
        self.assertEqual(result.shot_classification.shot_type,'Unknown')
        self.assertEqual(result.shot_classification.confidence,0)

    def test_dual_view_parse_retains_observation_metadata(self):
        raw={'right_wrist':Keypoint(10,20,.99,observed=False,
              recovered_from_mirror=True,source_frame_id=4,confidence_source='model')}
        parsed=DualViewBiomechanicsEngine.parse_pose_dict(raw)['right_wrist']
        self.assertFalse(parsed.observed)
        self.assertTrue(parsed.recovered_from_mirror)
        self.assertEqual(parsed.source_frame_id,4)

    def test_zero_pose_quality_is_not_replaced_by_present_pose(self):
        rows=[row(i,bad=110) for i in range(5)]
        e=event();e['quality_flags']['pose_frame_ratio']=0
        result=aggregate_event_biomechanics(e,rows,extract_motion_features(rows))
        self.assertEqual(result['quality']['pose_frame_ratio'],0)
        self.assertLess(result['metrics']['arm_extension']['confidence'],.92)

    def test_invalid_joint_reason_reaches_metric_contract(self):
        rows=[row(i,observed=True) for i in range(5)]
        metric=aggregate_event_biomechanics(event(),rows,extract_motion_features(rows))['metrics']['arm_extension']
        self.assertIn('invalid_point_coordinates',metric['contract']['missing_reasons'])

    def test_invalid_pose_container_abstains_without_legacy_pose_borrow(self):
        for declared in ('malformed', {'front':'malformed'}):
            r=row(0,bad=110);r['pose_observations']=declared
            self.assertEqual(measurement_pose(r),{})

    def test_large_finite_vectors_are_normalized_before_angle_product(self):
        r=row(0);r['pose'].update(right_shoulder=[1e200,0],right_elbow=[0,0],right_wrist=[0,1e200])
        self.assertEqual(extract_motion_features([r])[0]['arm_extension_deg'],90)

    def test_nonfinite_inputs_produce_strict_json_measurement_output(self):
        rows=[row(i,observed=True) for i in range(5)]
        json.dumps(aggregate_event_biomechanics(event(),rows,extract_motion_features(rows)),allow_nan=False)

    def test_invalid_score_cannot_qualify_osd_measurements(self):
        from test_osd_evidence_gate import OsdEvidenceGateTests
        qualify,rows,features,ext=OsdEvidenceGateTests().fixture()
        for r in rows:
            for p in r['pose_observations']['front'].values():p['confidence']=2
            r['rackets'][0]['confidence']=float('inf')
        qualify(ext,{'contact_frame':12,'contact_status':'candidate'},rows,features,100)
        self.assertIsNone(ext['stance']['image_foot_line_angle_deg'])
        self.assertIsNone(ext['brush_angle']['drop_depth_ratio'])

    def test_invalid_score_cannot_qualify_kinematic_peaks(self):
        from test_kinematic_cross_validation import frames
        from kinematic_sequence import analyze_kinematic_sequence
        rows=frames()
        for r in rows:
            for pose in r['kinematic_views'].values():
                for p in pose.values():p['confidence']=2
        result=analyze_kinematic_sequence(rows,20,25)
        self.assertIsNone(result['latency_hip_to_shoulder_ms'])
        self.assertEqual(result['cross_validation']['status'],'unavailable')

    def test_malformed_front_container_does_not_crash_event_aggregation(self):
        for declared in ('malformed', {'front':'malformed'}, {'front':[]}):
            with self.subTest(declared=declared):
                rows=[row(i,observed=True) for i in range(5)]
                for r in rows:r['pose_observations']=declared
                result=aggregate_event_biomechanics(event(),rows,extract_motion_features(rows))
                metric=result['metrics']['arm_extension']
                self.assertIsNone(metric['value'])
                self.assertIn('invalid_front_pose_observations',metric['contract']['missing_reasons'])

    def test_invalid_racket_scores_and_dimensions_cannot_supply_peak_samples(self):
        from kinematic_sequence import _racket_evidence
        times=[i*.04 for i in range(12)]
        for bad in (2,True,'0.99','reversed_box'):
            with self.subTest(invalid=bad):
                rows=[]
                for i in range(12):
                    x=i*10
                    box=[x+20,0,x,20] if bad=='reversed_box' else [x,0,x+20,20]
                    rows.append({'frame_id':i,'rackets':[{'box':box,'confidence':.99 if bad=='reversed_box' else bad,'observed':True}]})
                self.assertEqual(_racket_evidence(rows,times,.04)['status'],'low_coverage')

    def test_malformed_racket_records_abstain_in_features_osd_and_peaks(self):
        from kinematic_sequence import _racket_evidence
        for detections in ('malformed', {'box':[0,0,20,20]}, ['malformed']):
            with self.subTest(detections=detections):
                rows=[row(i,observed=True) for i in range(5)]
                for r in rows:r['rackets']=detections
                features=extract_motion_features(rows)
                self.assertTrue(all(f['racket_measurement_point'] is None for f in features))
                bio=aggregate_event_biomechanics(event(),rows,features)
                self.assertIsNone(bio['metrics']['brush_angle']['value'])
                self.assertEqual(_racket_evidence(rows,[i*.04 for i in range(5)],.04)['status'],'low_coverage')

    def test_build_tracks_the_raw_observation_gate_source(self):
        from analysis_provenance import analysis_build_info
        build=analysis_build_info()
        self.assertIn('observation_policy.py',build['code_sha256'])
        self.assertIn('dual_view_biomechanics.py',build['code_sha256'])



if __name__=='__main__':unittest.main()
