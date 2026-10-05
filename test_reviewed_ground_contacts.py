import copy
import math
import unittest
from scripts.audit_reviewed_ground_contacts import analyze, angle_to_lane
from test_ground_reference import calibration


def fixture(ids=(0,1)):
    c=calibration()
    labels={'schema':'tennis.ground-contact-review.v1','coordinate_space':'original_source_pixels',
            'confirmed':True,'independent_reference':False,'source_sha256':'a'*64,
            'frames':[{'frame_id':f,'width':600,'height':540} for f in ids],'labels':{}}
    journal=[]
    for fid in ids:
        for view in ('front','back'):
            for side,xy in [('left',(60+fid*5,120)),('right',(160+fid*5,220))]:
                if view=='back':xy=(450-fid*5,400 if side=='left' else 300)
                labels['labels'][f'{fid}:{view}:{side}_contact']={'reviewed':True,'visible':True,'x':xy[0],'y':xy[1],'contact_state':'ground_contact_visible'}
        journal.append({'frame_id':fid,'source_time':{'source_frame_id':fid,'source_kind':'video_file','basis':'media_pts','quality':'reported','timestamp_seconds':fid*.04}})
    return labels,c,journal


class ReviewedGroundTests(unittest.TestCase):
    def test_known_world_pair_and_unsigned_angle(self):
        l,c,j=fixture();r=analyze(l,c,j);p=r['foot_pairs'][0]
        self.assertAlmostEqual(p['distance_m'],math.sqrt(2))
        self.assertAlmostEqual(p['line_angle_to_lane_deg'],45)
        self.assertEqual(angle_to_lane([0,0],[1,1]),angle_to_lane([1,1],[0,0]))
        self.assertIsNone(angle_to_lane([0,0],[0,0]))
        self.assertFalse(r['accuracy_validated'])

    def test_unknown_not_mapped_or_bridged(self):
        l,c,j=fixture((0,1,2));l['labels']['1:front:left_contact'].update(visible=False,x=None,y=None,contact_state='unknown')
        r=analyze(l,c,j)
        self.assertFalse(any(p['view']=='front' and p['side']=='left' for p in r['consecutive_candidate_displacements']))
        self.assertFalse(any(p['frame_id']==1 and p['view']=='front' for p in r['foot_pairs']))

    def test_missing_frame_no_displacement_bridge(self):
        l,c,j=fixture((0,2));self.assertEqual(analyze(l,c,j)['consecutive_candidate_displacements'],[])

    def test_unconfirmed_mirror_never_fused(self):
        l,c,j=fixture();c['correspondence_confirmed']=False;r=analyze(l,c,j)
        self.assertTrue(all(not p['fusion_eligible'] for p in r['cross_view_diagnostics']))
        self.assertTrue(all(p['diagnostic_only'] for p in r['foot_pairs'] if p['view']=='back'))

    def test_perturbation_is_not_accuracy_or_error_interval(self):
        l,c,j=fixture();r=analyze(l,c,j);p=r['foot_pairs'][0]
        lo,hi=p['sensitivity']['6']['distance_sample_range_m']
        self.assertLess(lo,p['distance_m']);self.assertGreater(hi,p['distance_m'])
        small=[s for s in r['consecutive_candidate_displacements'] if s['view']=='front']
        self.assertTrue(all(not s['larger_than_6px_sample_envelope'] for s in small))
        self.assertFalse(r['coaching_eligible'])

    def test_bad_identity_unreviewed_and_nonfinite_rejected(self):
        l,c,j=fixture();bad=copy.deepcopy(l);bad['source_sha256']='b'*64
        with self.assertRaises(ValueError):analyze(bad,c,j)
        bad=copy.deepcopy(l);bad['labels']['0:front:left_contact']['reviewed']=False
        with self.assertRaises(ValueError):analyze(bad,c,j)
        bad=copy.deepcopy(l);bad['labels']['0:front:left_contact']['x']=float('nan')
        with self.assertRaises(ValueError):analyze(bad,c,j)
        bad=copy.deepcopy(l);bad['frames'][0]['width']=1200
        with self.assertRaises(ValueError):analyze(bad,c,j)

    def test_outside_ground_patch_not_extrapolated(self):
        l,c,j=fixture();l['labels']['0:front:left_contact']['x']=500
        p=next(p for p in analyze(l,c,j)['points'] if p['key']=='0:front:left_contact')
        self.assertIsNone(p['projected_xy_m']);self.assertEqual(p['reason'],'outside_calibrated_ground_patch')

    def test_invalid_clock_no_motion_measurement(self):
        for change in ({'quality':'unavailable'},{'basis':'wall_clock'},{'source_frame_id':9},{'timestamp_seconds':-.5}):
            l,c,j=fixture();j[1]['source_time'].update(change)
            self.assertEqual(analyze(l,c,j)['consecutive_candidate_displacements'],[])


if __name__=='__main__':unittest.main()
