"""Run a new CoreML pixel session using frozen geometry, with process-tree telemetry."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from session_evidence_bundle import verify_evidence_manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--python',default='venv_yolo26/bin/python')
    parser.add_argument('--max-frames',type=int,default=250)
    args=parser.parse_args();out=Path(args.output).resolve()
    if out.exists(): raise FileExistsError('Choose a new session directory')
    manifest=Path(args.manifest).resolve()
    verification=verify_evidence_manifest(str(manifest))
    if not verification['replayable']: raise ValueError('Frozen input verification failed')
    doc=json.loads(manifest.read_text());roi=doc['capture']['roi'];out.mkdir(parents=True)
    config=yaml.safe_load(Path('configs/yolo26_tennis_config.yaml').read_text())
    source=next(a for a in doc['artifacts'] if a['role']=='source_video')
    source_path=(manifest.parent/Path(source['path'])).resolve()
    fixture={'streams':[{'stream_id':'acceptance-frozen','stream_source':str(source_path),'default':True,
                        'roi_enabled':True,'frame_size':roi['frame_size'],'roi_points':roi['points'],
                        'mirror_view':roi['mirror_view'],'front_view':roi['front_view']}]}
    (out/'roi_fixture.yaml').write_text(yaml.safe_dump(fixture,allow_unicode=True))
    config['roi_settings'].update(enabled=True,auto_load_config=True,roi_config_path=str(out/'roi_fixture.yaml'),target_stream_id='acceptance-frozen')
    config['target_stream_id']='acceptance-frozen';config['save_video']=True
    config.setdefault('pipeline_perf',{}).update(dual_view_enabled=False,hdmi_output_enabled=False)
    config.setdefault('realtime_coach_tts',{})['enabled']=False
    config.setdefault('deepseek_coach',{})['enabled']=False
    (out/'runtime.yaml').write_text(yaml.safe_dump(config,allow_unicode=True))
    (out/'dual_fixture.yaml').write_text(yaml.safe_dump({'mirror_view':roi['mirror_view'],'front_view':roi['front_view'],'unified_detection':config['unified_detection']},allow_unicode=True))
    command=[args.python,'main_pipe.py','--config',str(out/'runtime.yaml'),'--input',str(source_path),
             '--algo2-config',str(out/'dual_fixture.yaml'),'--algo2-dual-view','--stream-id','acceptance-frozen',
             '--max-frames',str(args.max_frames),'--output',str(out/'pixels_osd.mp4'),'--output-fps','25',
             '--inference-workers','2','--session-id',out.name,'--session-output-root',str(out.parent),
             '--realtime-swing-json',str(out/'events.json'),'--realtime-swing-html',str(out/'report.html'),
             '--realtime-swing-event-log',str(out/'events.jsonl'),'--realtime-frame-output',
             '--realtime-frame-jsonl',str(out/'frames.jsonl'),'--realtime-frame-snapshot-json',str(out/'frames_latest.json'),
             '--evidence-manifest',str(out/'evidence_manifest.json')]
    samples=[];started=time.monotonic()
    with (out/'run.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            result=subprocess.run(['ps','-axo','pid=,ppid=,%cpu=,rss='],capture_output=True,text=True,check=True)
            processes={}
            for line in result.stdout.splitlines():
                fields=line.split()
                if len(fields)==4:
                    pid,parent,cpu,rss=fields
                    processes[int(pid)]={'pid':int(pid),'parent':int(parent),'cpu_percent':float(cpu),'rss_kib':int(rss)}
            descendants={process.pid}
            while True:
                children={pid for pid,p in processes.items() if p['parent'] in descendants}
                if children <= descendants: break
                descendants |= children
            current=[processes[p] for p in sorted(descendants) if p in processes]
            samples.append({'elapsed_s':round(time.monotonic()-started,3),'processes':current,
                            'summed_rss_kib':sum(p['rss_kib'] for p in current),
                            'summed_cpu_percent':sum(p['cpu_percent'] for p in current)})
            time.sleep(.5)
    run={'schema':'tennis.pixel-acceptance-run.v1','command':command,'returncode':process.returncode,
         'wall_seconds':round(time.monotonic()-started,3),'samples':samples,
         'peak_summed_rss_kib':max((s['summed_rss_kib'] for s in samples),default=None),
         'resource_semantics':'ps process-tree RSS sum can double-count shared memory; CPU is ps-reported average, not instantaneous utilization',
         'display_semantics':'OpenCV window submission measured; no sensor-to-photon/exposure verification',
         'frozen_manifest_verification':verification}
    if (out/'frames.jsonl').exists():
        frames=[json.loads(line) for line in (out/'frames.jsonl').read_text().splitlines() if line.strip()]
        run['frame_count']=len(frames)
        for field,label in [('captured_at_unix_ns','reader_receipt_fps'),('output_submitted_at_unix_ns','output_submission_fps')]:
            times=[r.get('timing',{}).get(field) for r in frames]
            valid=[t for t in times if type(t) is int]
            run[label]=(len(valid)-1)*1e9/(valid[-1]-valid[0]) if len(valid)>1 and valid[-1]>valid[0] else None
    (out/'run.json').write_text(json.dumps(run,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in run.items() if k not in ('samples','command','frozen_manifest_verification')},ensure_ascii=False))
    if process.returncode: raise SystemExit(process.returncode)


if __name__=='__main__':main()
