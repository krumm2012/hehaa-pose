import copy
import unittest
import numpy as np
from scripts.audit_pose_contact_consistency import audit,qualified_pose,robust_ratio_outlier,shoe_edge_evidence,source_dt


def fixture():
    rows=[];images={};labels={'frames':[{'frame_id':i} for i in range(12)],'labels':{}}
    for fid in range(12):
        views={}
        for view,scale in [('front',1.),('back',.7)]:
            points={}
            for part,y in [('shoulder',40),('elbow',60),('wrist',80),('hip',90),('knee',130),('ankle',170)]:
                for side,x in [('left',80),('right',120)]:
                    points[f'{side}_{part}']={'x':x*scale,'y':y*scale,'observed':True,'confidence':.99,'source_frame_id':fid,'confidence_source':'model'}
            views[view]=points
        rows.append({'frame_id':fid,'pose_observations':views,'pose_observation_coordinate_space':'original_source_pixels','source_time':{'source_frame_id':fid,'source_kind':'video_file','basis':'media_pts','quality':'reported','timestamp_seconds':fid*.04}})
        images[fid]=np.zeros((220,220,3),np.uint8)
    return rows,labels,images


class PoseContactConsistencyTests(unittest.TestCase):
    def test_constant_ratios_not_flagged(self):
        r,l,im=fixture();self.assertEqual(audit(r,l,im)['flags'],[])

    def test_ratio_outlier_needs_local_support(self):
        self.assertIsNotNone(robust_ratio_outlier(3,[1]*6))
        self.assertIsNone(robust_ratio_outlier(3,[1]*4))
        self.assertIsNone(robust_ratio_outlier(1,[1]*8))

    def test_stale_mirror_recovered_and_missing_identity_rejected(self):
        r,_,_=fixture();p=r[0]['pose_observations']['back'];p['left_ankle']['source_frame_id']=4;p['left_knee']['recovered_from_mirror']=True;p['left_wrist'].pop('source_frame_id')
        q,bad=qualified_pose(r[0],'back')
        self.assertNotIn('left_ankle',q);self.assertIn('left_knee',bad);self.assertIn('left_wrist',bad)

    def test_jump_detected_but_invalid_clock_abstains(self):
        r,l,im=fixture();r[6]['pose_observations']['back']['left_wrist']['x']+=60
        result=audit(r,l,im);self.assertTrue(any(p['frame_id']==6 and p['reason']=='joint_relative_jump' for p in result['flags']))
        r[6]['source_time']['quality']='unavailable'
        result=audit(r,l,im);self.assertFalse(any(p['frame_id']==6 and p['reason']=='joint_relative_jump' for p in result['flags']))

    def test_swapped_labels_flagged_no_mutation(self):
        r,l,im=fixture();p=r[6]['pose_observations']['back'];p['left_ankle']['x'],p['right_ankle']['x']=p['right_ankle']['x'],p['left_ankle']['x'];before=copy.deepcopy(r)
        self.assertTrue(any(p['reason']=='possible_left_right_swap' for p in audit(r,l,im)['flags']));self.assertEqual(r,before)

    def test_edges_cannot_confirm_shoe_contact(self):
        image=np.zeros((200,200,3),np.uint8);image[:,100:]=255
        r=shoe_edge_evidence(image,[100,110],[100,100],[100,50])
        self.assertFalse(r['weak_edge_support']);self.assertFalse(r['ground_contact_verified']);self.assertFalse(r['semantic_shoe_mask_available'])
        r=shoe_edge_evidence(np.zeros_like(image),[100,110],[100,100],[100,50]);self.assertTrue(r['weak_edge_support'])

    def test_missing_ankle_abstains_foot_check(self):
        r,l,im=fixture();r[0]['pose_observations']['front']['left_ankle']['confidence']=.1;l['labels']['0:front:left_contact']={'visible':True,'x':80,'y':175}
        result=audit(r,l,im);self.assertEqual(result['shoe_boundary_checks'][0]['status'],'unavailable')

    def test_time_gaps_and_short_intervals_not_normalized_into_jumps(self):
        r,l,im=fixture();self.assertIsNone(source_dt(r[0],r[2]));r[6]['source_time']['timestamp_seconds']=r[5]['source_time']['timestamp_seconds']+.002
        r[6]['pose_observations']['back']['left_wrist']['x']+=60
        a=audit(r,l,im);self.assertTrue(any(p['frame_id']==6 for p in a['clock_rejections']));self.assertFalse(any(p['frame_id']==6 and p['reason']=='joint_relative_jump' for p in a['flags']))

if __name__=='__main__':unittest.main()
