import copy
import unittest
from joint_racket_2d_analysis import analyze_frames, associate_rackets, missing_runs, summarize, qualified_point, BODY_POINTS, RACKET_POINTS
from scripts.serve_joint_racket_2d_review import validate_contact_review
from scripts.serve_joint_racket_2d_review import review_progress


def point(x=100,y=100,fid=0):
    return {'x':x,'y':y,'confidence':.9,'observed':True,'source_frame_id':fid}


def row(fid,t=None):
    f={'frame_id':fid,'pose_observation_coordinate_space':'original_source_pixels',
       'source_time':{'schema_version':'tennis.source-time.v1','source_kind':'video_file',
         'source_frame_id':fid,'basis':'media_pts','quality':'reported','timestamp_seconds':fid*.04 if t is None else t},
       'pose_observations':{},'racket_pose':{'frame_id':fid,'coordinate_space':'original_source_pixels','status':'observed','candidates':[]}}
    for v,y in [('front',300),('back',100)]:
        f['pose_observations'][v]={n:point(100+(j%2)*80,y+j*3,fid) for j,n in enumerate(BODY_POINTS)}
        f['pose_observations'][v]['right_wrist']=point(150+fid,y,fid)
        c={'source':'racket_pose_model','source_frame_id':fid,'observed':True,'confidence':.9,
           'keypoints':{n:point(150+fid+j*4,y+j*3,fid) for j,n in enumerate(RACKET_POINTS)}}
        c['keypoints']['grip_center']=point(151+fid,y,fid);f['racket_pose']['candidates'].append(c)
    return f


class JointRacketAnalysisTests(unittest.TestCase):
    def test_sources_and_recovered_points_do_not_become_observed(self):
        for change in ({'source_frame_id':1},{'observed':False},{'recovered_from_mirror':True},{'confidence':float('nan')},{'confidence':.49}):
            p=point();p.update(change);self.assertIsNone(qualified_point(p,0)[0])

    def test_separate_views_select_distinct_grips(self):
        frames,_=analyze_frames([row(0)],(1000,600))
        self.assertEqual(frames[0]['associations']['front']['candidate_index'],0)
        self.assertEqual(frames[0]['associations']['back']['candidate_index'],1)
        self.assertFalse(frames[0]['associations']['front']['person_assignment_verified'])

    def test_one_contour_cannot_be_assigned_to_both_views(self):
        r=row(0);r['racket_pose']['candidates']=r['racket_pose']['candidates'][:1]
        r['pose_observations']['back']=copy.deepcopy(r['pose_observations']['front'])
        frames,_=analyze_frames([r],(1000,600))
        self.assertTrue(all(a['reason']=='same_candidate_competes_for_both_views' for a in frames[0]['associations'].values()))

    def test_ambiguous_close_candidates_abstain(self):
        r=row(0);r['racket_pose']['candidates'].append(copy.deepcopy(r['racket_pose']['candidates'][0]))
        frames,_=analyze_frames([r],(1000,600))
        self.assertEqual(frames[0]['associations']['front']['reason'],'ambiguous_grip_assignment')

    def test_speed_uses_source_dt_and_keeps_missing_null(self):
        rows=[row(i) for i in range(6)];frames,_=analyze_frames(rows,(1000,600))
        self.assertIsNone(frames[0]['metrics']['front']['sweet_spot_speed_px_s'])
        self.assertAlmostEqual(frames[1]['metrics']['front']['sweet_spot_speed_px_s'],25)
        rows[2]['racket_pose']['candidates']=[];frames,_=analyze_frames(rows,(1000,600))
        self.assertIsNone(frames[2]['metrics']['front']['sweet_spot_speed_px_s'])
        self.assertIsNone(frames[3]['metrics']['front']['sweet_spot_speed_px_s'])
        self.assertAlmostEqual(frames[4]['metrics']['front']['sweet_spot_speed_px_s'],25)

    def test_short_pts_and_missing_source_frames_cannot_produce_speed(self):
        frames,_=analyze_frames([row(0,0),row(1,.04),row(2,.041),row(3,.12),row(5,.2)],(1000,600))
        self.assertIsNone(frames[2]['metrics']['front']['sweet_spot_speed_px_s'])
        self.assertIsNone(frames[4]['metrics']['front']['sweet_spot_speed_px_s'])

    def test_exact_three_gap_boundary_and_no_bridge_over_missing_journal(self):
        rows=[row(i) for i in range(9)]
        for r in rows[1:4]+rows[5:7]:r['racket_pose']['candidates']=[]
        frames,_=analyze_frames(rows,(1000,600));gaps=missing_runs(frames,'front.racket.sweet_spot')
        self.assertEqual([(g['start_frame'],g['end_frame'],g['count']) for g in gaps],[(1,3,3)])
        self.assertAlmostEqual(gaps[0]['pts_span_ms'],80)
        frames=[f for f in frames if f['frame_id']!=2]
        self.assertEqual(missing_runs(frames,'front.racket.sweet_spot'),[])

    def test_gap_overlap_and_leading_trailing_source_missing(self):
        rows=[row(i) for i in range(1,6)]
        for r in rows[1:4]:r['racket_pose']['candidates']=[]
        frames,_=analyze_frames(rows,(1000,600));s=summarize(frames,[{'event_id':1,'start_frame':1,'end_frame':5,'contact_frame':3}],7)
        g=next(g for g in s['observation_gaps_ge_3'] if g['track']=='front.racket.sweet_spot')
        self.assertEqual(g['priority'],'contact_window');self.assertEqual(g['near_contact_event_ids'],[1])
        self.assertEqual([g['start_frame'] for g in s['journal_gaps']],[0,6])

    def test_duplicate_frames_rejected(self):
        with self.assertRaises(ValueError):analyze_frames([row(0),row(0)],(1000,600))


class ContactReviewTests(unittest.TestCase):
    def test_saved_review_restores_human_decisions_and_session_progress(self):
        import tempfile
        import json
        import hashlib
        from pathlib import Path
        from scripts.serve_joint_racket_2d_review import saved_reviews
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'human_reviews';folder.mkdir()
            body=json.dumps({'review':self.review}).encode()
            (folder/(hashlib.sha256(body).hexdigest()+'.json')).write_bytes(body)
            latest,invalid=saved_reviews(directory,self.dataset)
            self.assertEqual(invalid,0)
            self.assertEqual(latest['s']['events']['1']['anchor_frame'],2)
            progress=review_progress(directory,self.dataset)
            self.assertEqual(progress['saved_status_counts']['confirmed'],1)
            self.assertTrue(progress['sessions'][0]['complete'])

    def test_progress_excludes_unsigned_or_tampered_receipts(self):
        import tempfile
        import json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            dataset = {'sessions': [{'id': 's', 'events': [{'event_id': 1}]}]}
            self.assertEqual(review_progress(directory, dataset)['reviewed_events'], 0)
            folder = Path(directory)/'human_reviews'; folder.mkdir()
            (folder/'invalid.json').write_text(json.dumps({'review': {}}))
            progress = review_progress(directory, dataset)
            self.assertEqual(progress['reviewed_events'], 0)
            self.assertEqual(progress['invalid_receipts_ignored'], 1)

    def setUp(self):
        self.dataset={'sessions':[{'id':'s','hashes':{'source_video':'abc'},'events':[{'event_id':1}],
                                  'frames':[{'frame_id':i,'source_seconds':i*.04} for i in range(5)]}]}
        self.review={'schema':'tennis.joint-racket-contact-review.v1','session_id':'s','hashes':{'source_video':'abc'},
                     'annotation_mode':'model_assisted_review','independent_reference':False,'reviewer':'test only',
                     'events':{'1':{'status':'confirmed','start_frame':1,'end_frame':3,'anchor_frame':2,
                                    'interval_pts':[.04,.12],'view':'front','note':'Synthetic test, not real labels'}}}

    def test_receipt_separates_human_anchor_from_model(self):
        r=validate_contact_review(self.review,self.dataset)
        self.assertEqual(r['events']['1']['anchor_source_seconds'],.08)
        self.assertFalse(r['model_events_modified']);self.assertFalse(r['accuracy_validated'])
        self.assertNotIn('anchor_source_seconds',self.review['events']['1'])

    def test_mismatched_identity_forged_pts_and_unexplained_confirmation_rejected(self):
        for change in ('hash','pts','anchor','note'):
            review=copy.deepcopy(self.review)
            if change=='hash':review['hashes']['source_video']='wrong'
            elif change=='pts':review['events']['1']['interval_pts']=[.01,.12]
            elif change=='anchor':review['events']['1']['anchor_frame']=4
            else:review['events']['1']['note']=''
            with self.assertRaises(ValueError):validate_contact_review(review,self.dataset)

    def test_interval_is_not_confirmed_and_unknown_cannot_keep_anchor(self):
        r=self.review['events']['1'];r.update(status='interval',anchor_frame=None)
        self.assertIsNone(validate_contact_review(self.review,self.dataset)['events']['1']['anchor_source_seconds'])
        r['status']='unknown'
        with self.assertRaises(ValueError):validate_contact_review(self.review,self.dataset)


if __name__=='__main__':unittest.main()
