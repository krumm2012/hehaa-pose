import unittest
from copy import deepcopy
from baseline_observations import baseline_profiles, normalized_view_trends

def frames(factor=1, moving=False):
    rows=[]
    for i in range(40):
        points={name:{'x':x*factor,'y':(y+(i*2 if moving else 0))*factor,
                     'confidence':.9,'observed':True,'confidence_source':'model'}
                for name,x,y in [('left_shoulder',10,10),('right_shoulder',50,10),('left_hip',15,80),('right_hip',45,80)]}
        rows.append({'frame_id':i,'source_time':{'schema_version':'tennis.source-time.v1',
            'source_kind':'video_file','source_frame_id':i,'timestamp_seconds':i*.04,'basis':'media_pts','quality':'reported'},
                     'pose_observations':{'front':points,'back':deepcopy(points)}})
    return rows

class BaselineTests(unittest.TestCase):
    def test_stable_and_scaled_baselines(self):
        a,b=frames(),frames(2)
        pa,pb=baseline_profiles(a,25),baseline_profiles(b,25)
        self.assertTrue(pa['views']['front']['valid'])
        self.assertEqual(pb['views']['front']['scale_px'],2*pa['views']['front']['scale_px'])
        self.assertEqual(normalized_view_trends(a,pa,25,39),normalized_view_trends(b,pb,25,39))

    def test_moving_baseline_is_unknown_not_zero_noise(self):
        p=baseline_profiles(frames(moving=True),25)['views']['front']
        self.assertFalse(p['valid']);self.assertIn('baseline_motion',p['reasons'])
        self.assertIsNone(p['hip_position_scaled_mad_torso_ratio'])

    def test_causal_window_and_independent_views(self):
        rows=frames();before=baseline_profiles(rows,25)
        for row in rows[25:]:
            for point in row['pose_observations']['front'].values():point['y']+=1000
        self.assertEqual(before,baseline_profiles(rows,25))
        for row in rows:
            row['pose_observations']['back']['left_hip']['observed']=False
        p=baseline_profiles(rows,25)
        self.assertTrue(p['views']['front']['valid']);self.assertFalse(p['views']['back']['valid'])

    def test_missing_clock_and_short_baseline_abstain(self):
        rows=frames();rows[10]['source_time']['quality']='unavailable'
        self.assertFalse(baseline_profiles(rows,25)['views']['front']['valid'])
        self.assertFalse(baseline_profiles(frames(),2)['views']['front']['valid'])
