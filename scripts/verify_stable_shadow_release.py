"""Validate a stable SHADOW contract; never enables active/exclusive inference."""
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.verify_shadow_release import DEFAULT_CONTRACT, verify as verify_preview
from scripts.validate_stable_shadow import strict_gate


def verify(workspace,contract_path):
    c=json.loads(Path(contract_path).read_text())
    base=json.loads(DEFAULT_CONTRACT.read_text())
    if (c.get('schema')!='tennis.stable-shadow-release-contract.v1' or c.get('channel')!='stable'
        or c.get('default_mode')!='legacy' or c.get('quality_gate_passed') is not False
        or c.get('active_or_exclusive_allowed') is not False
        or c.get('not_an_active_acceptance_receipt') is not True
        or c.get('stable_25fps_accepted') is not True
        or c.get('model')!=base['model'] or c.get('startup')!=base['startup']):
        raise ValueError('Invalid stable shadow scope or model/startup contract')
    result=verify_preview(workspace)
    return {**result,'version':c['version'],'scope':'Stable shadow model identity; business remains legacy'}


def verify_validation(validation,version,commit):
    if validation.get('version')!=version or validation.get('source_commit')!=commit:
        raise ValueError('Validation does not bind this release commit')
    regression=validation.get('regression',{})
    runs=regression.get('runs',[])
    clips={r.get('source_sha256') for r in runs}
    if (len(runs)!=27 or len(clips)!=9 or None in clips
        or any(sum(r.get('source_sha256')==clip for r in runs)!=3 for clip in clips)
        or any({r.get('repeat') for r in runs if r.get('source_sha256')==clip}!={0,1,2} for clip in clips)
        or not all(strict_gate(r) for r in runs)
        or regression.get('source_files_unchanged') is not True):
        raise ValueError('Strict 27-run acceptance missing')
    expected=json.loads(DEFAULT_CONTRACT.read_text())['model']['sha256']
    build=validation.get('build_id')
    sections=[regression,*runs]+[validation.get(k,{}) for k in
                               ('soak','compatibility','fault_workflow','regression_tests')]
    if (not build or any(s.get('build_id')!=build or s.get('source_commit')!=commit for s in sections)):
        raise ValueError('Validation phases use different or missing frozen builds')
    for run in runs:
        sidecar=run.get('sidecar',{})
        config=sidecar.get('config',{})
        if (sidecar.get('model_sha256')!=expected or config.get('imgsz')!=640
            or config.get('backend')!='coreml' or config.get('compute_units')!='ALL'
            or config.get('inference_stride')!=1 or config.get('mode')!='shadow'
            or run.get('processing')!='buffered'):
            raise ValueError('Run uses another model or sampled profile')
    soak=validation.get('soak',{})
    if (soak.get('sidecar',{}).get('model_sha256')!=expected
        or soak.get('expected_frames',0)<15000 or soak.get('pts_monotonic') is not True
        or soak.get('stable_output_seconds',0)<480):
        raise ValueError('Continuous 640 soak missing; historical sampled soak is inapplicable')
    count=soak['expected_frames']
    metrics=soak.get('sidecar',{}).get('metrics',{})
    config=soak.get('sidecar',{}).get('config',{})
    def finite(value):
        return type(value) in (float,int) and math.isfinite(value)
    if (soak.get('returncode')!=0
        or any(soak.get(k) is not True for k in ('source_frames_complete','source_time_valid',
                   'display_frames_complete','replay_semantic_match'))
        or any(soak.get(k)!=count for k in ('frames','unique_source_frames','encoded_frames','new_same_frame'))
        or metrics.get('submitted')!=count or metrics.get('matched')!=count
        or any(metrics.get(k)!=0 for k in ('late','skipped','replaced_pending','fallback'))
        or soak.get('clip_evidence',{}).get('passed') is not True
        or not finite(soak.get('stable_output_fps')) or soak['stable_output_fps']<25
        or not finite(soak.get('stable_nominal_p95_ms')) or not 0<=soak['stable_nominal_p95_ms']<=250
        or not finite(soak.get('rss_growth_fraction')) or soak['rss_growth_fraction']>.10
        or config.get('imgsz')!=640 or config.get('backend')!='coreml'
        or config.get('compute_units')!='ALL' or config.get('inference_stride')!=1
        or config.get('mode')!='shadow' or soak.get('sidecar',{}).get('disabled_reason')):
        raise ValueError('Continuous soak measurements incomplete or outside thresholds')
    for key in ('soak','compatibility','fault_workflow','regression_tests'):
        if validation.get(key,{}).get('passed') is not True:
            raise ValueError('Required acceptance missing: '+key)
    if validation.get('active_or_exclusive_allowed') is not False:
        raise ValueError('Shadow receipt cannot enable active inference')
