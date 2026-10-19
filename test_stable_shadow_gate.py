import copy
import unittest
from scripts.validate_stable_shadow import strict_gate


class StableShadowGateTests(unittest.TestCase):
    def good(self):
        return dict(returncode=0, continuous_gate=True, frames=250, unique_source_frames=250,
            new_same_frame=250, encoded_frames=250, display_frames_complete=True,
            source_time_valid=True, replay_semantic_match=True, output_fps=25.1,
            nominal_input_to_output_ms={'p95':100}, output_lag_growth_ms=0,
            clip_evidence={'passed':True}, sidecar={'metrics':dict(submitted=250, matched=250,
                skipped=0, replaced_pending=0, late=0, fallback=0)})

    def test_good_and_missing_evidence(self):
        good=self.good()
        self.assertTrue(strict_gate(good))
        for key in good:
            case=copy.deepcopy(good); del case[key]
            self.assertFalse(strict_gate(case),key)

    def test_slow_or_incomplete_runs_cannot_pass(self):
        for key,value in [('output_fps',24.99),('frames',249),('encoded_frames',249),
                ('new_same_frame',249),('clip_evidence',{'passed':False}),
                ('nominal_input_to_output_ms',{'p95':250.01}),('output_lag_growth_ms',40.1)]:
            case=self.good(); case[key]=value
            self.assertFalse(strict_gate(case),key)
        for key in ('late','skipped','fallback','replaced_pending'):
            case=self.good(); case['sidecar']['metrics'][key]=1
            self.assertFalse(strict_gate(case),key)


if __name__=='__main__': unittest.main()
