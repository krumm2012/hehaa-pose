import copy
import json
from pathlib import Path
import unittest

from scripts.verify_stable_shadow_release import verify_validation


class StableShadowReleaseTests(unittest.TestCase):
    def fixture(self):
        model=json.loads(Path('docs/releases/640-shadow-preview-20261019.json').read_text())['model']['sha256']
        runs=[]
        for repeat in range(3):
            for clip in range(9):
                runs.append(dict(returncode=0,continuous_gate=True,frames=250,unique_source_frames=250,
                    new_same_frame=250,encoded_frames=250,display_frames_complete=True,source_time_valid=True,
                    replay_semantic_match=True,output_fps=25.1,nominal_input_to_output_ms={'p95':100},
                    output_lag_growth_ms=0,clip_evidence={'passed':True},source_sha256='clip'+str(clip),
                    processing='buffered',repeat=repeat,build_id='frozen',source_commit='abc',sidecar={'model_sha256':model,
                    'config':dict(imgsz=640,backend='coreml',compute_units='ALL',inference_stride=1,mode='shadow'),
                    'metrics':dict(submitted=250,matched=250,late=0,skipped=0,replaced_pending=0,fallback=0)}))
        sidecar=copy.deepcopy(runs[0]['sidecar'])
        sidecar['metrics'].update(submitted=15250,matched=15250)
        identity=dict(build_id='frozen',source_commit='abc')
        return dict(version='test',source_commit='abc',build_id='frozen',active_or_exclusive_allowed=False,
            regression=dict(**identity,runs=runs,source_files_unchanged=True),
            soak=dict(**identity,passed=True,sidecar=sidecar,expected_frames=15250,
                pts_monotonic=True,stable_output_seconds=480,stable_output_fps=25.1,
                stable_nominal_p95_ms=100,rss_growth_fraction=.05,returncode=0,
                source_frames_complete=True,source_time_valid=True,display_frames_complete=True,
                replay_semantic_match=True,frames=15250,unique_source_frames=15250,
                encoded_frames=15250,new_same_frame=15250,clip_evidence={'passed':True}),
            compatibility={**identity,'passed':True},fault_workflow={**identity,'passed':True},
            regression_tests={**identity,'passed':True})

    def test_exact_scope_and_all_sections_required(self):
        fixture=self.fixture()
        verify_validation(fixture,'test','abc')
        for key in ('soak','compatibility','fault_workflow','regression_tests'):
            case=copy.deepcopy(fixture);del case[key]
            with self.subTest(key=key),self.assertRaises(ValueError):
                verify_validation(case,'test','abc')
        with self.assertRaises(ValueError):verify_validation(fixture,'test','other-commit')

    def test_historical_sampled_soak_and_wrong_run_model_are_rejected(self):
        fixture=self.fixture()
        fixture['soak']['sidecar']['model_sha256']='old-960-model'
        with self.assertRaisesRegex(ValueError,'historical'):verify_validation(fixture,'test','abc')
        fixture=self.fixture()
        fixture['regression']['runs'][0]['sidecar']['config']['inference_stride']=2
        with self.assertRaisesRegex(ValueError,'sampled'):verify_validation(fixture,'test','abc')

    def test_single_slow_or_reused_clip_run_blocks_packaging(self):
        for modify in ('slow','duplicate','incomplete-clip'):
            fixture=self.fixture()
            if modify=='slow':fixture['regression']['runs'][0]['output_fps']=24.9
            if modify=='duplicate':fixture['regression']['runs'][0]['source_sha256']='clip1'
            if modify=='incomplete-clip':fixture['regression']['runs'][0]['clip_evidence']['passed']=False
            with self.subTest(modify=modify),self.assertRaisesRegex(ValueError,'27-run'):
                verify_validation(fixture,'test','abc')

    def test_mixed_build_or_repeated_same_round_cannot_pass(self):
        for modify in ('run-build','section-build','run-commit','duplicate-round'):
            fixture=self.fixture()
            if modify=='run-build':fixture['regression']['runs'][0]['build_id']='other'
            if modify=='section-build':fixture['soak']['build_id']='other'
            if modify=='run-commit':fixture['regression']['runs'][0]['source_commit']='other'
            if modify=='duplicate-round':fixture['regression']['runs'][9]['repeat']=0
            with self.subTest(modify=modify),self.assertRaises(ValueError):
                verify_validation(fixture,'test','abc')

    def test_soak_pass_flag_cannot_hide_incomplete_or_slow_measurements(self):
        for key,value in [('stable_output_fps',24.9),('stable_nominal_p95_ms',251),
                          ('rss_growth_fraction',.11),('new_same_frame',15249),
                          ('source_frames_complete',False),('encoded_frames',15249)]:
            fixture=self.fixture();fixture['soak'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                verify_validation(fixture,'test','abc')


if __name__=='__main__':unittest.main()
