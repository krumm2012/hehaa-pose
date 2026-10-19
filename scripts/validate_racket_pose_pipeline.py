"""Frozen local-video A/B runs; reports measured throughput, not encoded FPS."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import shutil
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import yaml
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from session_evidence_bundle import verify_evidence_manifest, replay_evidence_manifest


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def distribution(values):
    return {'p50':float(np.percentile(values,50)), 'p95':float(np.percentile(values,95)), 'max':float(max(values))} if values else {}


def measure(session):
    rows=[json.loads(l) for l in (session/'frames.jsonl').read_text().splitlines() if l.strip()]
    fields=('inference_ms','analysis_ms','capture_to_analysis_ms','render_ms','output_submit_ms','capture_to_output_submit_ms','capture_to_inference_start_ms','inference_to_analysis_start_ms')
    report={'frames':len(rows),'unique_source_frames':len({r['frame_id'] for r in rows}),
            'timing_ms':{k:distribution([r['timing'][k] for r in rows if isinstance(r.get('timing',{}).get(k),(float,int))]) for k in fields}}
    times=[r['timing']['output_submitted_at_unix_ns'] for r in rows]
    report['first_output_unix_ns']=times[0] if times else None
    report['output_fps']=(len(times)-1)*1e9/(times[-1]-times[0]) if len(times)>1 else 0
    report['drop_rate']=max(0,1-len(rows)/250)
    report['new_selected']=sum(r.get('racket_selection_source')=='racket_pose_model' and bool(r.get('rackets')) and r['rackets'][0].get('observed') is True and r['rackets'][0].get('source_frame_id')==r['frame_id'] for r in rows)
    report['new_same_frame']=sum((r.get('racket_pose') or {}).get('status')=='observed' and r['racket_pose'].get('frame_id')==r['frame_id'] for r in rows)
    report['racket_selection_sources']=dict(Counter(r.get('racket_selection_source','not_recorded') for r in rows))
    report['ball_racket_support_sources']=dict(Counter(r.get('detection_diagnostics',{}).get('racket_support_source','legacy') for r in rows))
    report['legacy_candidate_frames']=sum(bool(r.get('legacy_rackets')) for r in rows)
    ev=json.loads((session/'events.json').read_text())
    report['event_semantics']=[{k:e.get(k) for k in ('event_id','start_frame','contact_frame','peak_frame','end_frame','stroke_type')} for e in ev.get('events',[])]
    report['source_time_valid']=all(r['source_time']['source_frame_id']==r['frame_id'] and r['source_time']['basis']=='media_pts' for r in rows)
    report['replayable']=verify_evidence_manifest(str(session/'evidence_manifest.json'))['replayable']
    report['replay_semantic_match']=replay_evidence_manifest(str(session/'evidence_manifest.json'))['semantic_match']
    report['output_sink']='encoded_video' if (session/'output.mp4').exists() else 'render_only'
    if report['output_sink']=='encoded_video' and shutil.which('ffprobe'):
        result=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames','-of','json',str(session/'output.mp4')],
            capture_output=True,text=True,check=True,timeout=max(30,len(rows)/50))
        report['encoded_frames']=int(json.loads(result.stdout)['streams'][0]['nb_read_frames'])
    stage_names={k for row in rows for k in row.get('inference_stage_ms',{})}
    report['inference_stage_ms']={k:distribution([row['inference_stage_ms'][k] for row in rows if k in row.get('inference_stage_ms',{})]) for k in stage_names}
    output_stage_names={k for row in rows for k in row.get('output_stage_ms',{})}
    report['output_stage_ms']={k:distribution([row['output_stage_ms'][k] for row in rows if k in row.get('output_stage_ms',{})]) for k in output_stage_names}
    report['reader_schedule_lag_ms']=distribution([r['reader_schedule_lag_ms'] for r in rows if isinstance(r.get('reader_schedule_lag_ms'),(int,float))])
    lags=[r.get('reader_schedule_lag_ms',0)+r['timing'].get('capture_to_output_submit_ms',0) for r in rows]
    report['nominal_input_to_output_ms']=distribution(lags)
    report['nominal_input_clock_scope']='Local input nominal-FPS schedule after startup readiness; not camera exposure or VFR media PTS'
    report['output_lag_growth_ms']=float(np.median(lags[-25:])-np.median(lags[:25])) if lags else None
    report['continuous_frames_complete']=bool([r['frame_id'] for r in rows]==list(range(250)) and report['source_time_valid'])
    display_path=session/'output_display.jsonl'
    if display_path.exists():
        shown=[json.loads(l) for l in display_path.read_text().splitlines() if l.strip()]
        report['display_frames_complete']=[r['frame_id'] for r in shown]==[r['frame_id'] for r in rows]
        report['display_completion_ms']=distribution([(r['display_returned_at_unix_ns']-r['submitted_at_unix_ns'])/1e6 for r in shown])
        frame_map={r['frame_id']:r for r in rows}
        display_lags=[frame_map[r['frame_id']].get('reader_schedule_lag_ms',0)+(r['display_returned_at_unix_ns']-frame_map[r['frame_id']]['timing']['captured_at_unix_ns'])/1e6 for r in shown if r['frame_id'] in frame_map]
        report['nominal_input_to_display_return_ms']=distribution(display_lags)
        report['display_clock_scope']='HighGUI return; not physical screen presentation'
    report['realtime_gate']=bool(report['output_fps']>=24 and report['drop_rate']<=.05 and report['timing_ms']['capture_to_output_submit_ms'].get('p95',1e9)<=250 and report['source_time_valid'] and report['replayable'])
    return report


def compact_repeat(session, report):
    """Verify a lossless archive before removing only THIS run's generated copies."""
    archive=session.parent/(session.name+'.tar.gz')
    if archive.exists():raise FileExistsError(archive)
    hashes={p.relative_to(session).as_posix():digest(p) for p in session.rglob('*') if p.is_file()}
    with tarfile.open(archive,'w:gz',compresslevel=1) as tf:
        tf.add(session,arcname=session.name)
    seen=set()
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            if not member.isfile():continue
            rel=Path(member.name).relative_to(session.name).as_posix()
            h=hashlib.sha256()
            stream=tf.extractfile(member)
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                h.update(chunk)
            if h.hexdigest()!=hashes[rel]:raise ValueError('Archive verification failed: '+rel)
            seen.add(rel)
    if seen!=set(hashes):raise ValueError('Archive is incomplete')
    for p in session.iterdir():
        if p.name=='measurement.json':continue
        if p.is_dir():shutil.rmtree(p)
        else:p.unlink()
    report['verified_archive']=str(archive)
    (session/'measurement.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))


def run_case(args, video, mode, processing, stride, repeat):
    # CLI experiments must select their geometry deliberately. Older programmatic
    # callers retain the explicitly recorded disabled fixture for reproducibility.
    roi_policy=getattr(args,'roi_policy','disabled')
    roi_stream_id=getattr(args,'roi_stream_id',None)
    if roi_policy not in ('project','disabled'):
        raise ValueError('Unsupported regression ROI policy')
    if roi_policy=='project' and not roi_stream_id:
        raise ValueError('本地回归使用项目 ROI 需显式指定 --roi-stream-id，避免套用默认机位')
    session=Path(args.output).resolve()/f'{processing}-{mode}-s{stride}-{video.stem}-r{repeat}'
    if session.exists():
        if (session/'measurement.json').is_file():
            cached=json.loads((session/'measurement.json').read_text())
            if cached.get('roi_policy','disabled')!=roi_policy or cached.get('roi_stream_id')!=roi_stream_id:
                raise ValueError('ROI policy differs from cached run; use a new output directory')
            requested_size = getattr(args, 'imgsz', None)
            cached_size = (cached.get('sidecar', {}).get('config') or {}).get('imgsz', 960)
            if requested_size is not None and cached_size != requested_size:
                raise ValueError('Racket resolution differs from cached run; use a new output directory')
            return cached
        raise FileExistsError(session)
    session.mkdir(parents=True)
    config=yaml.safe_load((ROOT/'configs/yolo26_tennis_config.yaml').read_text())
    config['video_input_path']=str(video);config['save_video']=args.save_video
    config['video_processing']['max_frames']=0
    if roi_policy=='disabled':
        config['roi_settings'].update(enabled=False,auto_load_config=False)
    else:
        from roi_stream_config import resolve_roi_stream_profile
        config['roi_settings']['target_stream_id']=roi_stream_id
        profile=resolve_roi_stream_profile(config,str(video),(2560,1440),target_stream_id=roi_stream_id)
        if not profile.enabled or profile.stream_id!=roi_stream_id:
            raise ValueError('Requested project ROI is unavailable: '+profile.reason)
    config['pipeline_perf'].update(dual_view_enabled=False,hdmi_output_enabled=False,collect_frame_results=False)
    config['realtime_swing']['deepseek']['enabled']=False
    config['realtime_swing'].setdefault('coach_tts',{})['enabled']=False
    if args.fault_sidecars:
        config['realtime_swing']['deepseek'].update(enabled=True,base_url='http://127.0.0.1:9',api_key_env='TENNIS_ACCEPTANCE_DUMMY_KEY',timeout_seconds=.2)
        config['realtime_swing']['coach_tts'].update(enabled=True,playback=False,startup_timeout_seconds=.2,request_timeout_seconds=.2)
    config['racket_pose'].update(mode=mode,inference_stride=stride,acceptance_run=not args.production_check)
    config['pipeline_perf']['inference_schedule']=getattr(args,'inference_schedule','barrier')
    if getattr(args,'pose_resize_policy',None):
        config['pipeline_perf']['pose_resize_policy']=args.pose_resize_policy
    config['racket_pose']['ipc_transport']=getattr(args,'racket_ipc','queue')
    if getattr(args,'pose_compute_units',None):config['pose_compute_units']=args.pose_compute_units
    if getattr(args,'display_mode',None):config['pipeline_perf']['async_display_enabled']=args.display_mode=='async'
    if getattr(args,'opencv_threads',None) is not None:config['pipeline_perf']['opencv_threads']=args.opencv_threads
    if getattr(args,'detector_compute_units',None):config['unified_detection']['compute_units']=args.detector_compute_units
    if args.model:config['racket_pose']['model_path']=str(Path(args.model).resolve())
    if args.acceptance:config['racket_pose']['acceptance_path']=str(Path(args.acceptance).resolve())
    if args.backend:config['racket_pose']['backend']=args.backend
    if args.compute_units:config['racket_pose']['compute_units']=args.compute_units
    if getattr(args, 'imgsz', None) is not None:config['racket_pose']['imgsz']=args.imgsz
    # Geometry fixture is for THESE source videos only; never edits camera settings.
    dv=yaml.safe_load((ROOT/'configs/dual_view_config.yaml').read_text())
    if roi_policy=='disabled':dv['front_view']['default_roi']=[.50,.18,.82,.65]
    (session/'dual_fixture.yaml').write_text(yaml.safe_dump(dv))
    (session/'runtime.yaml').write_text(yaml.safe_dump(config))
    command=[str(ROOT/'venv_yolo26/bin/python'),str(ROOT/'main_pipe.py'),'--config',str(session/'runtime.yaml'),
             '--input',str(video),'--output',str(session/'output.mp4'),'--processing-mode',processing,
             '--racket-pose-mode',mode,'--no-dual-view','--session-id',session.name,
             '--realtime-swing-events','--realtime-coach','--realtime-swing-json',str(session/'events.json'),
             '--realtime-swing-html',str(session/'report.html'),'--realtime-frame-jsonl',str(session/'frames.jsonl'),
             '--realtime-frame-snapshot-json',str(session/'frames_latest.json'),
             '--evidence-manifest',str(session/'evidence_manifest.json')]
    if args.dual:command+=['--algo2-dual-view','--algo2-config',str(session/'dual_fixture.yaml')]
    if not args.save_video:command.append('--no-save-video')
    started=time.monotonic();started_unix_ns=time.time_ns()
    env=os.environ.copy()
    if args.fault_sidecars:
        env['TENNIS_ACCEPTANCE_DUMMY_KEY']='local-fault-test'
        env['TENNIS_QWEN3_TTS_PYTHON']='/__tennis_acceptance_missing_worker__'
    with (session/'run.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        samples=[]
        while process.poll() is None:
            ps=subprocess.run(['ps','-axo','pid=,ppid=,rss=,%cpu='],capture_output=True,text=True).stdout
            rows={}
            for line in ps.splitlines():
                v=line.split()
                if len(v)==4:rows[int(v[0])]=(int(v[1]),int(v[2]),float(v[3]))
            ids={process.pid}
            while True:
                children={p for p,(parent,_,_) in rows.items() if parent in ids}
                if children<=ids:break
                ids|=children
            samples.append({'elapsed_s':time.monotonic()-started,'summed_rss_kib':sum(rows[p][1] for p in ids if p in rows),'summed_cpu_percent':sum(rows[p][2] for p in ids if p in rows)})
            if time.monotonic()-started>180:
                process.terminate();process.wait(timeout=10);raise TimeoutError(session)
            time.sleep(.5)
    report={'session':str(session),'mode':mode,'processing':processing,'stride':stride,'repeat':repeat,'source_sha256':digest(video),
            'pose_resize_policy':config['pipeline_perf'].get('pose_resize_policy','legacy'),
            'roi_policy':roi_policy,'roi_stream_id':roi_stream_id,
            'returncode':process.returncode,'wall_seconds':time.monotonic()-started,'resource_samples':samples,
            'resource_semantics':'Process-tree RSS sums may double count shared pages; ps CPU is averaged.'}
    if process.returncode==0:report.update(measure(session))
    if process.returncode==0:
        report['effective_roi']=json.loads((session/'evidence_manifest.json').read_text()).get('capture',{}).get('roi')
    if report.get('first_output_unix_ns'):
        report['startup_to_first_output_ms']=(report['first_output_unix_ns']-started_unix_ns)/1e6
    if args.fault_sidecars and process.returncode==0:
        events=json.loads((session/'events.json').read_text()).get('events',[])
        report['sidecar_fault_statuses']=[{k:e.get(k) for k in ('deepseek_advice','coach_tts')} for e in events]
    summary_path=session/'output_racket_pose.summary.json'
    if summary_path.exists():
        report['sidecar']=json.loads(summary_path.read_text())
        if report['sidecar']['disabled_reason']:
            report['realtime_gate']=False
    if processing=='buffered' and process.returncode==0:
        sidecar=report.get('sidecar',{})
        metrics=sidecar.get('metrics',{})
        report['continuous_model_complete']=mode=='legacy' or (report.get('new_same_frame')==250 and not sidecar.get('disabled_reason') and all(metrics.get(k)==0 for k in ('skipped','replaced_pending','fallback')))
        report['continuous_gate']=bool(report.get('continuous_frames_complete') and report.get('continuous_model_complete') and report.get('display_frames_complete',True) and report.get('encoded_frames',250)==250 and report.get('replay_semantic_match'))
        report['sustained_realtime_gate']=bool(report['continuous_gate'] and report.get('output_fps',0)>=24 and report.get('nominal_input_to_output_ms',{}).get('p95',1e9)<=250 and report.get('output_lag_growth_ms',1e9)<=40)
    if mode=='exclusive' and process.returncode==0:
        report['exclusive_source_gate']=bool(report.get('legacy_candidate_frames')==0
            and set(report.get('racket_selection_sources',{})) <= {'racket_pose_model','racket_pose_model_derived','missing'}
            and report.get('ball_racket_support_sources')=={'racket_pose_model':report.get('frames')})
    if report.get('replay_semantic_match') is False:report['realtime_gate']=False
    if mode=='active' and processing=='realtime' and report.get('new_selected',0)==0:
        report['realtime_gate']=False
        report['active_gate_reason']='No fresh new racket was selected; fallback alone cannot pass enhanced mode'
    if getattr(args,'collect_clip_evidence',False) and process.returncode==0:
        from scripts.validate_stable_shadow import clip_evidence
        report['clip_evidence']=clip_evidence(session)
    (session/'measurement.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if repeat>0 and process.returncode==0:compact_repeat(session,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('resource_samples','event_semantics','timing_ms')},ensure_ascii=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--model');parser.add_argument('--acceptance');parser.add_argument('--backend');parser.add_argument('--compute-units',choices=('ANE','ALL','CPU'));parser.add_argument('--fault-sidecars',action='store_true');parser.add_argument('--production-check',action='store_true');parser.add_argument('--videos',default='/Users/krum5539/Desktop/Camera/2026-09-30');parser.add_argument('--mode',choices=('legacy','exclusive','shadow','active'),default='legacy');parser.add_argument('--processing',choices=('realtime','buffered','offline'),default='realtime');parser.add_argument('--stride',type=int,default=1);parser.add_argument('--repeats',type=int,default=1);parser.add_argument('--smoke',action='store_true');parser.add_argument('--dual',action='store_true');parser.add_argument('--save-video',action='store_true');parser.add_argument('--pose-compute-units',choices=('ALL','ANE','CPU'));parser.add_argument('--display-mode',choices=('sync','async'));parser.add_argument('--opencv-threads',type=int);parser.add_argument('--detector-compute-units',choices=('ALL','ANE','CPU','GPU'));parser.add_argument('--inference-schedule',choices=('barrier','interframe'),default='barrier');parser.add_argument('--racket-ipc',choices=('queue','shared_memory'),default='queue');parser.add_argument('--roi-policy',choices=('project','disabled'),default='project');parser.add_argument('--roi-stream-id');parser.add_argument('--imgsz',type=int,choices=(640,960));args=parser.parse_args()
    os.chdir(ROOT);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    videos=sorted(Path(args.videos).glob('*.mp4'))
    if args.smoke:videos=[p for p in videos if p.name in ('48.43.mp4','49.23.mp4','50.03.mp4')]
    reports=[]
    for repeat in range(args.repeats):
        for video in videos:
            reports.append(run_case(args,video,args.mode,args.processing,args.stride,repeat))
            (out/f'{args.processing}-{args.mode}-s{args.stride}-summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
