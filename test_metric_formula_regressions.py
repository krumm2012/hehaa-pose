import math
import unittest
from swing_biomechanics import _shoulder_turn_change_metric
from kinematic_sequence import _view_evidence
import test_osd_evidence_gate as osd_tests

class FormulaRegressions(unittest.TestCase):
    def test_circular_baseline(self):
        result=_shoulder_turn_change_metric({i:{'shoulder_line_angle_deg':a} for i,a in enumerate([179,-179]*4)},0,7,1)
        self.assertLessEqual(result['value'],2)

    def test_missing_time_cannot_claim_full_window(self):
        qualify,rows,features,ext=osd_tests.OsdEvidenceGateTests().fixture()
        for row in rows[:8]: row['source_time']['quality']='unavailable'
        qualify(ext,{'contact_frame':12,'contact_status':'candidate'},rows,features,100)
        self.assertFalse(ext['brush_angle']['measurement_evidence']['display_eligible'])
        self.assertLess(ext['brush_angle']['measurement_evidence']['coverage'],1)

    def test_filter_does_not_join_occluded_segments(self):
        rows=[]
        for i,a in enumerate([0,0,0,4,None,0,0,4,4,4]):
            pose={}
            if a is not None:
                for segment in ('hip','shoulder'):
                    pose['left_'+segment]={'x':0,'y':0,'confidence':.9,'observed':True}
                    pose['right_'+segment]={'x':100*math.cos(math.radians(a)),'y':100*math.sin(math.radians(a)),'confidence':.9,'observed':True}
            rows.append({'frame_id':i,'kinematic_views':{'front':pose}})
        result=_view_evidence(rows,[i*.04 for i in range(10)],'front',.04)
        self.assertIsNone(result['segments']['hip']['peak'])

class PeakIntervalTests(unittest.TestCase):
    def test_broad_racket_peak_overlaps_shoulder(self):
        from kinematic_sequence import _pair_interval
        self.assertFalse(_pair_interval([.2,.2],[.18,.34],.04)['resolved'])
        self.assertTrue(_pair_interval([.2,.2],[.30,.32],.04)['resolved'])
        self.assertEqual(_pair_interval([.3,.32],[.1,.12],.04)['order'],'reverse')


class ContractDeliveryTests(unittest.TestCase):
    def test_json_report_and_osd_share_contract_and_missing_reason(self):
        from metric_contracts import attach_metric_contracts, DEFINITIONS
        from analysis_metric_delivery import event_analysis_metrics
        from realtime_swing_runtime import build_impact_telemetry_card
        bio={'metrics':{'brush_angle':{'value':None,'unit':'deg','measurement_evidence':{'reasons':['incomplete_source_time']}}},
             'extended_biomechanics':{'brush_angle':{'low_to_high_angle_deg':None,'measurement_evidence':{'reasons':['incomplete_source_time']}}}}
        attach_metric_contracts(bio)
        event={'biomechanics':bio}
        canonical=bio['metrics']['brush_angle']['contract']
        self.assertEqual(canonical['missing_reasons'],['incomplete_source_time'])
        self.assertEqual(build_impact_telemetry_card(event)['metric_contracts']['brush_angle'],canonical)
        bio['metrics']['brush_angle']['value']=12
        bio['metrics']['brush_angle']['measurement_evidence']={'reasons':[]}
        attach_metric_contracts(bio)
        self.assertEqual(event_analysis_metrics(event)[0]['contract'],bio['metrics']['brush_angle']['contract'])
        from analysis_metric_delivery import LABELS
        self.assertFalse(set(LABELS)-set(DEFINITIONS))

    def test_broad_racket_peak_propagates_through_sequence(self):
        from unittest.mock import patch
        from test_kinematic_cross_validation import frames
        from kinematic_sequence import analyze_kinematic_sequence
        original=analyze_kinematic_sequence(frames(),20,25)
        shoulder=original['peak_time_ranges_seconds']['shoulder'][0]
        racket={'status':'usable','peak':{'frame_id':24,'time':shoulder+.10,
                'time_range':[shoulder,shoulder+.16],'speed':200}}
        with patch('kinematic_sequence._racket_evidence',return_value=racket):
            result=analyze_kinematic_sequence(frames(),20,25)
        self.assertFalse(result['pair_timing']['shoulder_to_racket']['resolved'])
        self.assertEqual(result['sequence_quality'],'UNRESOLVED_AT_FRAME_RATE')

if __name__=='__main__': unittest.main()
