"""Run one continuous encoded 640-shadow session using references to original media.

The FFmpeg concat demuxer keeps source pixels and PTS, without copying the videos.
This is a local-file stability test, not camera/network transport certification.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stress_racket_pose_pipeline import sample_tree
from scripts.validate_racket_pose_pipeline import measure
from scripts.validate_stable_shadow import clip_evidence
from scripts.validate_stable_shadow import verify_frozen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--frozen-inputs',required=True,type=Path)
    args=parser.parse_args()
    frozen=verify_frozen(args.frozen_inputs)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path('/Users/krum5539/Desktop/Camera/2026-09-30/49.43.mp4')
    (out/source.name).symlink_to(source)
    concat=out/'source.ffconcat'
    concat.write_text('ffconcat version 1.0\n'+'file 49.43.mp4\n'*61)
    expected=250*61
    config=yaml.safe_load(args.template.read_text())
    config['video_input_path']=str(concat)
    config['video_processing']['max_frames']=0
    (out/'runtime.yaml').write_text(yaml.safe_dump(config))
    fixture=args.template.parent/'dual_fixture.yaml'
    (out/'dual_fixture.yaml').write_bytes(fixture.read_bytes())
    command=[str(ROOT/'venv_yolo26/bin/python'),'-u',str(ROOT/'main_pipe.py'),
        '--config',str(out/'runtime.yaml'),'--input',str(concat),'--output',str(out/'output.mp4'),
        '--processing-mode','buffered','--racket-pose-mode','shadow','--no-dual-view',
        '--session-id',out.name,'--algo2-dual-view','--algo2-config',str(out/'dual_fixture.yaml'),
        '--realtime-swing-events','--realtime-coach','--realtime-swing-json',str(out/'events.json'),
        '--realtime-swing-html',str(out/'report.html'),'--realtime-frame-jsonl',str(out/'frames.jsonl'),
        '--realtime-frame-snapshot-json',str(out/'frames_latest.json'),
        '--evidence-manifest',str(out/'evidence_manifest.json')]
    samples=[];start=time.monotonic()
    with (out/'run.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            sample={'elapsed_s':time.monotonic()-start,**sample_tree(process.pid)}
            samples.append(sample)
            with (out/'resources.jsonl').open('a') as resource_log:
                resource_log.write(json.dumps(sample)+'\n')
            if sample['elapsed_s']>900:
                process.terminate();process.wait(timeout=20)
                raise TimeoutError('Continuous session exceeded 900 seconds')
            time.sleep(1)
    verify_frozen(args.frozen_inputs)
    report={'build_id':frozen['build_id'],'source_commit':frozen['source_commit'],
        'scope':'Unbroken local-file pipeline, 61 concatenated references to original 49.43; not live transport acceptance',
        'returncode':process.returncode,'expected_frames':expected,'wall_seconds':time.monotonic()-start,
        'memory_semantics':'Process-tree summed RSS may double-count shared pages; median 120–240s vs final120s',
        'resource_samples':samples,'passed':False}
    if process.returncode==0:
        report.update(measure(out))
        rows=[json.loads(s) for s in (out/'frames.jsonl').read_text().splitlines()]
        sidecar=json.loads((out/'output_racket_pose.summary.json').read_text())
        m=sidecar['metrics']
        report['sidecar']=sidecar
        report['clip_evidence']=clip_evidence(out)
        report['source_frames_complete']=[r['frame_id'] for r in rows]==list(range(expected))
        pts=[r['source_time'].get('timestamp_seconds') for r in rows]
        report['pts_monotonic']=all(isinstance(t,(int,float)) for t in pts) and all(b>a for a,b in zip(pts,pts[1:]))
        report['source_time_example']=rows[0]['source_time']
        first=rows[0]['timing']['output_submitted_at_unix_ns']
        stable=[r for r in rows if r['timing']['output_submitted_at_unix_ns']>=first+120e9]
        elapsed=(stable[-1]['timing']['output_submitted_at_unix_ns']-stable[0]['timing']['output_submitted_at_unix_ns'])/1e9
        report['stable_output_fps']=(len(stable)-1)/elapsed
        lags=[r.get('reader_schedule_lag_ms',0)+r['timing']['capture_to_output_submit_ms'] for r in stable]
        report['stable_nominal_p95_ms']=float(np.percentile(lags,95))
        report['stable_output_seconds']=elapsed
        baseline=[s['rss_kib'] for s in samples if 120<=s['elapsed_s']<240]
        ending=[s['rss_kib'] for s in samples if s['elapsed_s']>=samples[-1]['elapsed_s']-120]
        growth=float(np.median(ending)/np.median(baseline)-1) if baseline and ending else None
        report['rss_growth_fraction']=growth
        report['queue_depth_observation']='macOS may not expose qsize; all source/worker/display queues remain construction-bounded; per-frame available depths retained'
        report['passed']=bool(report['source_frames_complete'] and report['source_time_valid'] and report['pts_monotonic']
            and report['display_frames_complete'] and report['encoded_frames']==expected
            and report['new_same_frame']==expected and report['replay_semantic_match']
            and not sidecar['disabled_reason'] and m['submitted']==expected and m['matched']==expected
            and all(m[k]==0 for k in ('late','skipped','replaced_pending','fallback'))
            and report['clip_evidence']['passed'] and growth is not None and growth<=.10
            and report['stable_output_fps']>=25 and report['stable_nominal_p95_ms']<=250
            and report['stable_output_seconds']>=480)
    (out/'soak.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('sidecar','resource_samples','event_semantics')},ensure_ascii=False),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
