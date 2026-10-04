"""Manual reference identities and recovery labels must retain source meaning."""
import unittest

from manual_review_workflow import derive_manual_coach_events, validate_manual_annotations
from motion_time_contract import candidate_timeline
from swing_event_segmenter import summarize_manual_event_range, _phase_counts
from swing_motion_features import extract_motion_features
from test_manual_review_workflow import _frame, _event_document, _annotations
from swing_coach_data_collector import build_coach_dataset


def recorded_frames(missing=()):
    rows=[_frame(i) for i in range(20) if i not in missing]
    for row in rows:
        row['source_time']={'schema_version':'tennis.source-time.v1','source_kind':'video_file',
            'source_frame_id':row['frame_id'],'timestamp_seconds':row['frame_id']*.041,
            'basis':'media_pts','quality':'reported'}
    return rows


def phase_features(ids, times):
    return candidate_timeline([{'frame_id':i,'timestamp':999.,'contact_score':0,
        'source_time':{'schema_version':'tennis.source-time.v1','source_kind':'video_file',
            'source_frame_id':i,'timestamp_seconds':t,'basis':'media_pts','quality':'reported'}}
        for i,t in zip(ids,times)])[0]


class ManualAnchorPhaseContractTests(unittest.TestCase):
    def test_missing_contact_is_preserved_instead_of_borrowing_a_neighbor(self):
        result=summarize_manual_event_range(extract_motion_features(recorded_frames((9,))),1,9,16)
        self.assertEqual(result['contact_frame'],9)
        self.assertEqual(result['manual_anchor_evidence']['anchors']['contact_frame']['status'],'missing')
        self.assertEqual(result['frame_phases'],{})
        self.assertEqual(result['quality_flags']['ball_contact_window_basis'],'unavailable')
        self.assertIn('contact_anchor_observation_missing',result['quality_flags']['ball_contact_window_reasons'])

    def test_missing_start_and_end_are_preserved_as_manual_boundaries(self):
        result=summarize_manual_event_range(extract_motion_features(recorded_frames((1,16))),1,9,16)
        self.assertEqual((result['start_frame'],result['end_frame']),(1,16))
        self.assertEqual(result['manual_anchor_evidence']['observed_frame_count'],14)

    def test_real_manual_recompute_does_not_manufacture_contact_time(self):
        result=derive_manual_coach_events(_event_document(),_annotations(False),recorded_frames((9,)))
        event=result['events'][0]
        self.assertEqual(event['contact_frame'],9)
        self.assertIsNone(event['phase_timing']['duration_seconds'])
        self.assertIn('missing_event_anchor_record',event['phase_timing']['reasons'])
        self.assertEqual(event['review_provenance']['evidence_range'],[1,9,16])

    def test_missing_anchor_reason_is_visible_in_the_real_timing_renderer(self):
        from swing_report_builder import _build_event_source_timing_html
        event=derive_manual_coach_events(_event_document(),_annotations(False),recorded_frames((9,)))['events'][0]
        report=_build_event_source_timing_html(event['phase_timing'])
        self.assertIn('缺少事件锚点源帧',report)
        self.assertIn('源时间不可核验',report)

    def test_noninteger_manual_identities_are_not_coerced_into_other_frames(self):
        for bad in (True,9.9,'9'):
            with self.subTest(contact=bad):
                annotations=_annotations(False);annotations['events'][0]['frames']['contact']=bad
                with self.assertRaises(ValueError):
                    validate_manual_annotations(_event_document(),annotations,recorded_frames())

    def test_collector_preserves_the_missing_manual_reference(self):
        rows=recorded_frames((9,));features=extract_motion_features(rows)
        manual=derive_manual_coach_events(_event_document(),_annotations(False),rows)
        collected=build_coach_dataset({'frames':rows},{**manual,'features':features})['events'][0]
        self.assertEqual(collected['frames']['contact'],9)
        self.assertEqual(collected['frames']['contact_confidence'],0)
        self.assertIsNone(collected['timing']['duration_seconds'])

    def test_collector_does_not_reselect_a_declared_event_contact(self):
        from swing_coach_data_collector import _event_frames
        markers=_event_frames({'start_frame':0,'contact_frame':9,'peak_frame':0,'end_frame':9},
            [{'frame_id':0,'contact_score':.99},{'frame_id':9,'contact_score':.5}],[])
        self.assertEqual(markers['contact'],9)
        self.assertEqual(markers['contact_confidence'],.5)

    def test_duplicate_contact_observations_are_disclosed_as_ambiguous(self):
        rows=recorded_frames();rows.insert(10,dict(rows[9]))
        result=summarize_manual_event_range(extract_motion_features(rows),1,9,16)
        self.assertEqual(result['contact_frame'],9)
        self.assertEqual(result['manual_anchor_evidence']['anchors']['contact_frame']['status'],'ambiguous')
        self.assertEqual(result['frame_phases'],{})

    def test_dense_short_quiet_run_does_not_become_recovery(self):
        features=phase_features(list(range(13)),[i*.01 for i in range(13)])
        _,phases=_phase_counts(features,[10,0,0,0,0,10,10,0,0,0,0,0,0],0,12,0)
        self.assertEqual(phases[1],'follow_through')

    def test_quiet_observations_supporting_the_source_interval_allow_recovery(self):
        features=phase_features(list(range(13)),[i*.01 for i in range(13)])
        _,phases=_phase_counts(features,[10]+[0]*12,0,12,0)
        self.assertEqual(phases[1],'ready')

    def test_recovery_does_not_bridge_missing_source_identities(self):
        ids=[0,1,3,4,5,6,7]
        features=phase_features(ids,[i*.04 for i in ids])
        _,phases=_phase_counts(features,[10]+[0]*6,0,6,0)
        self.assertEqual(phases[1],'follow_through')


if __name__=='__main__':unittest.main()
