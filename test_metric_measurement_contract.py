import unittest
from swing_motion_features import extract_motion_features
from swing_biomechanics import _calculate_extended_tier_biomechanics


def frame(i, x, t, width=20):
    return {'frame_id':i, 'timestamp':t, 'source_time':{'schema_version':'tennis.source-time.v1',
            'source_kind':'video_file','source_frame_id':i,'timestamp_seconds':t,'basis':'media_pts','quality':'reported'},
            'pose':{'left_shoulder':[0,0], 'right_shoulder':[width,0]},
            'racket':[x-10,0,x+10,20]}


class MetricMeasurementContractTests(unittest.TestCase):
    def test_uncalibrated_speed_never_becomes_180_kmh(self):
        features = extract_motion_features([frame(0,0,0),frame(1,100,.04)])
        self.assertIsNone(features[-1]['racket_head_speed_kmh'])
        self.assertAlmostEqual(features[-1]['racket_speed_px_s'],2500)

    def test_turning_shoulders_cannot_change_image_speed(self):
        values = [extract_motion_features([frame(0,0,0,w),frame(1,10,.04,w)])[-1].get('racket_speed_px_s') for w in (20,200)]
        self.assertEqual(values,[250,250])

    def test_missing_contact_speed_is_not_event_peak_times_point88(self):
        metrics = _calculate_extended_tier_biomechanics([{'frame_id':0,'racket_head_speed_kmh':100}],0,10,12,100)
        self.assertIsNone(metrics['racket_head_speed']['contact_kmh'])

    def test_wrist_cannot_substitute_racket_drop(self):
        features=[{'frame_id':i,'racket_center':[i*10,100],'wrist':[i*10,200-i*20]} for i in range(3)]
        metric=_calculate_extended_tier_biomechanics(features,0,2,2,100)['brush_angle']
        self.assertEqual(metric['drop_depth_px'],0)

    def test_invalid_source_clock_never_falls_back_to_40ms(self):
        for second in (frame(1,10,0), frame(1,10,-.04)):
            self.assertIsNone(extract_motion_features([frame(0,0,0),second])[-1]['racket_speed_px_s'])
        second=frame(1,10,.04)
        second['source_time']['quality']='unavailable'
        self.assertIsNone(extract_motion_features([frame(0,0,0),second])[-1]['racket_speed_px_s'])

    def test_source_time_not_legacy_timestamp_controls_speed(self):
        frames=[frame(0,0,0),frame(1,10,.1)]
        frames[1]['timestamp']=999
        self.assertEqual(extract_motion_features(frames)[1]['racket_speed_px_s'],100)

    def test_interpolated_and_nonconsecutive_boxes_are_not_speed_measurements(self):
        frames=[frame(0,0,0),frame(1,10,.04),frame(2,20,.08)]
        frames[1]['racket']=None
        features=extract_motion_features(frames)
        self.assertEqual(features[1]['racket_center_source'],'interpolated')
        self.assertTrue(all(f['racket_speed_px_s'] is None for f in features))
        self.assertIsNone(extract_motion_features([frames[0],frames[2]])[-1]['racket_speed_px_s'])

    def test_shoulder_change_uses_same_view_line_not_width_proxy(self):
        from swing_biomechanics import _shoulder_turn_change_metric
        rows={i:{'shoulder_line_angle_deg':0, 'shoulder_turn_deg':i*20} for i in range(5)}
        self.assertEqual(_shoulder_turn_change_metric(rows,0,4,1)['value'],0)

    def test_missing_hips_at_contact_are_not_replaced_by_last_pose(self):
        metrics=_calculate_extended_tier_biomechanics([
            {'frame_id':0,'hip_vertical_pos':100}, {'frame_id':1,'hip_vertical_pos':90}],0,2,3,100)
        self.assertIsNone(metrics['leg_drive']['drive_ratio'])
