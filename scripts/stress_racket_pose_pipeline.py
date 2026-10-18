"""Loop ORIGINAL local videos over loopback MJPEG for a bounded resource soak.

This tests live resource stability, NOT sensor time, independent accuracy or
source-frame correspondence through the MJPEG encoding.
"""
import argparse
import math
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]


class VideoServer(ThreadingHTTPServer):
    daemon_threads=True


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        if self.path!='/video':self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame');self.end_headers()
        cap=None
        try:
            while not self.server.stop_event.is_set():
                for video in self.server.videos:
                    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS) or 25;deadline=time.monotonic()
                    while not self.server.stop_event.is_set():
                        ok,frame=cap.read()
                        if not ok:break
                        ok,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
                        if not ok:raise RuntimeError('MJPEG encode failed')
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg.tobytes()+b'\r\n');self.wfile.flush()
                        deadline+=1/fps
                        self.server.stop_event.wait(max(0,deadline-time.monotonic()))
                    cap.release();cap=None
        except (BrokenPipeError,ConnectionResetError):pass
        finally:
            if cap is not None:cap.release()


def sample_tree(pid):
    output=subprocess.run(['ps','-axo','pid=,ppid=,rss=,%cpu='],capture_output=True,text=True).stdout
    rows={}
    for line in output.splitlines():
        v=line.split()
        if len(v)==4:rows[int(v[0])]=(int(v[1]),int(v[2]),float(v[3]))
    ids={pid}
    while True:
        children={p for p,(parent,_,_) in rows.items() if parent in ids}
        if children<=ids:break
        ids|=children
    return {'pids':sorted(ids),'rss_kib':sum(rows[p][1] for p in ids if p in rows),'cpu_percent':sum(rows[p][2] for p in ids if p in rows)}


def frame_metrics(rows,warmup_seconds=120):
    stable=[]
    if rows:
        first=rows[0]['timing']['output_submitted_at_unix_ns']
        stable=[r for r in rows if r['timing']['output_submitted_at_unix_ns']>=first+warmup_seconds*1e9]
    elapsed=(stable[-1]['timing']['output_submitted_at_unix_ns']-stable[0]['timing']['output_submitted_at_unix_ns'])/1e9 if len(stable)>1 else 0
    fps=(len(stable)-1)/elapsed if elapsed>0 else 0
    values=[r['timing'].get('capture_to_output_submit_ms') for r in stable]
    valid=[v for v in values if isinstance(v,(int,float)) and math.isfinite(v) and v>=0]
    p95=float(np.percentile(valid,95)) if valid else None
    coverage=len(valid)/len(stable) if stable else 0
    drop=1-len(stable)/(stable[-1]['frame_id']-stable[0]['frame_id']+1) if stable else None
    return {'frames':len(rows),'stable_output_fps':fps,'stable_p95_ms':p95,'reader_frame_drop_rate':drop,
            'timing_coverage':coverage,'invalid_timing_frames':len(stable)-len(valid),
            'realtime_passed':bool(fps>=24 and p95 is not None and p95<=250 and drop is not None and drop<=.05 and coverage>=.999)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--seconds',type=int,default=600);parser.add_argument('--videos',default='/Users/krum5539/Desktop/Camera/2026-09-30');parser.add_argument('--mode',choices=('legacy','shadow','active'),default='active');parser.add_argument('--stride',type=int,choices=(1,2,3),default=1);args=parser.parse_args()
    os.chdir(ROOT);out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    server=VideoServer(('127.0.0.1',0),Handler);server.stop_event=threading.Event();server.videos=sorted(Path(args.videos).glob('*.mp4'))
    if not server.videos:raise ValueError('No local videos')
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    config=yaml.safe_load((ROOT/'configs/yolo26_tennis_config.yaml').read_text());config['roi_settings'].update(enabled=False,auto_load_config=False)
    config['pipeline_perf'].update(dual_view_enabled=False,collect_frame_results=False,hdmi_output_enabled=False)
    config['realtime_swing']['deepseek']['enabled']=False;config['realtime_swing'].setdefault('coach_tts',{})['enabled']=False
    config['video_processing']['max_frames']=0
    config['racket_pose']['acceptance_run']=True
    config['racket_pose']['inference_stride']=args.stride
    (out/'runtime.yaml').write_text(yaml.safe_dump(config))
    command=[str(ROOT/'venv_yolo26/bin/python'),'-u',str(ROOT/'main_pipe.py'),'--config',str(out/'runtime.yaml'),
             '--input',f'http://127.0.0.1:{server.server_port}/video','--output',str(out/'output.mp4'),
             '--processing-mode','realtime','--racket-pose-mode',args.mode,'--no-save-video','--no-dual-view',
             '--realtime-swing-events','--realtime-coach','--realtime-swing-json',str(out/'events.json'),
             '--realtime-swing-html',str(out/'report.html'),'--realtime-frame-jsonl',str(out/'frames.jsonl')]
    samples=[];started=time.monotonic();stopped_by_test=False
    try:
        with (out/'run.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                elapsed=time.monotonic()-started;samples.append({'elapsed_s':elapsed,**sample_tree(process.pid)})
                with (out/'resources.jsonl').open('a') as resource_log:
                    resource_log.write(json.dumps(samples[-1])+'\n')
                if int(elapsed)%30==0:print('SOAK',round(elapsed),'seconds','RSS MiB',round(samples[-1]['rss_kib']/1024),flush=True)
                if elapsed>=args.seconds:
                    process.send_signal(signal.SIGINT);stopped_by_test=True;break
                time.sleep(1)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
    finally:
        server.stop_event.set();server.shutdown();server.server_close();thread.join(timeout=2)
    (out/'lifecycle.json').write_text(json.dumps({'seconds_requested':args.seconds,'wall_seconds':time.monotonic()-started,
        'returncode':process.returncode,'stopped_by_test':stopped_by_test},indent=2))
    base=[s['rss_kib'] for s in samples if 120<=s['elapsed_s']<240]
    end=[s['rss_kib'] for s in samples if s['elapsed_s']>=args.seconds-120]
    growth=float(np.median(end)/np.median(base)-1) if base and end else None
    summary_path=out/'output_racket_pose.summary.json'
    sidecar=json.loads(summary_path.read_text()) if summary_path.exists() else None
    sidecar_ok = args.mode == 'legacy' or bool(sidecar and not sidecar['disabled_reason'] and sidecar['metrics']['matched'] > 0)
    rows=[json.loads(line) for line in (out/'frames.jsonl').read_text().splitlines()] if (out/'frames.jsonl').exists() else []
    metrics=frame_metrics(rows);realtime_ok=metrics['realtime_passed']
    report={'scope':'10 minute loopback MJPEG resource soak; source media time/independent accuracy unverified',
            'seconds_requested':args.seconds,'wall_seconds':time.monotonic()-started,'returncode':process.returncode,
            'stopped_by_test':stopped_by_test,'rss_growth_fraction':growth,'sidecar':sidecar,
            **metrics,'source_identity_scope':'Reader frame IDs count known grab/replaced drops; MJPEG source exposure and pre-Reader network drops are unverified',
            'resource_semantics':'Process-tree summed RSS can double-count shared pages; relative plateau comparison after 120s warmup',
            'passed':bool(stopped_by_test and process.returncode==0 and growth is not None and growth<=.1 and sidecar_ok and realtime_ok),
            'samples':samples}
    (out/'soak.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','sidecar')},ensure_ascii=False),flush=True)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
