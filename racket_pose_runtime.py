"""Bounded, killable racket sidecar; no model runtime in Reader or Analyzer."""
from __future__ import annotations

import json
import multiprocessing as mp
import queue
import time
from pathlib import Path

from racket_pose_detector import pose_config, verify_pipeline_acceptance, model_digest


def accepted_runtime_config(config,processing_mode):
    """Use the highest cadence that actually passed for this input/view profile."""
    config=pose_config({**config,'processing_mode':processing_mode})
    if processing_mode in ('buffered','offline'):
        config.update(configured_stride=config['inference_stride'],inference_stride=1,
                      schedule_source='continuous_every_source_frame')
        return config
    if (processing_mode!='realtime' or config['mode']=='legacy' or config.get('acceptance_run',False)
            or not config.get('use_accepted_schedule',True) or not config.get('runtime_acceptance_path')):
        return config
    try:
        report=json.loads(Path(config['runtime_acceptance_path']).read_text())
        profile=(report.get('profiles') or {}).get(config.get('pipeline_profile','single')) or {}
        proposed=pose_config({**config,'inference_stride':profile.get('inference_stride',config['inference_stride'])})
        verify_pipeline_acceptance(proposed,model_digest(config['model_path']))
        proposed['configured_stride']=config['inference_stride'];proposed['schedule_source']='accepted_profile'
        return proposed
    except (OSError,ValueError,TypeError):
        # Active worker reports the failed receipt and quarantines itself; shadow
        # remains available for explicit experiments with an unvalidated profile.
        return config


def _worker(config, incoming, outgoing, journal_path):
    import signal
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    from frame_transport import FrameTransportReader
    frames=FrameTransportReader()
    journal=None
    try:
        loading_started=time.perf_counter()
        from racket_pose_detector import RacketPoseDetector
        import numpy as np
        if config.get('opencv_threads') is not None:
            import cv2
            cv2.setNumThreads(max(0, int(config['opencv_threads'])))
        detector=RacketPoseDetector(config)
        load_ms=(time.perf_counter()-loading_started)*1000
        warming_started=time.perf_counter()
        size = config['imgsz']
        warmup=detector.detect(np.zeros((size,size,3),dtype=np.uint8),-1,None)
        warmup_ms=(time.perf_counter()-warming_started)*1000
        if config['mode']=='active' and warmup['status']=='numerical_failure':
            raise ValueError('Racket model failed finite-output warmup')
        if journal_path:
            Path(journal_path).parent.mkdir(parents=True,exist_ok=True)
            journal=open(journal_path,'x',encoding='utf-8')
        outgoing.put({'kind':'ready','model_sha256':detector.digest,'startup_timing_ms':{'load_ms':load_ms,'warmup_ms':warmup_ms}})
        while True:
            task=incoming.get()
            if task is None:break
            started_unix_ns=time.time_ns()
            with frames.read(task) as frame:
                result=detector.detect(frame,task['frame_id'],task['source_time'],task.get('prepared'))
            result['input_transport']={'kind':'shared_memory' if task.get('shared_frame') else 'queue',
                                       'frame_id':task['frame_id'],
                                       'generation':(task.get('shared_frame') or {}).get('generation')}
            submitted=task.get('submitted_at_unix_ns')
            delay=(started_unix_ns-submitted)/1e6 if submitted is not None else None
            result['timing']['queue_wait_ms']=delay if delay is not None and delay>=0 else None
            result['timing']['queue_clock']='shared_wall_clock; includes IPC transfer; not exposure time'
            result['timing']['queue_clock_valid']=delay is not None and delay>=0
            result['timing']['completed_at_unix_ns']=time.time_ns()
            if journal:
                journal.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');journal.flush()
            if config['mode']=='active' and result['status']=='numerical_failure':
                raise ValueError('Nonfinite model output; active sidecar quarantined')
            try:outgoing.put({'kind':'result','result':result},timeout=.05)
            except queue.Full:pass  # Complete observations remain in the sidecar journal.
    except BaseException as exc:
        try:outgoing.put({'kind':'error','error':type(exc).__name__+': '+str(exc)},timeout=.1)
        except Exception:pass
    finally:
        if journal:journal.close()
        frames.close()


class RacketPoseRuntime:
    def __init__(self,config,processing_mode,journal_path=None,worker_target=None):
        startup_started=time.perf_counter()
        self.config=accepted_runtime_config(config,processing_mode);self.processing_mode=processing_mode
        from racket_source_policy import is_exclusive
        if is_exclusive(self.config) and processing_mode not in ('buffered','offline'):
            raise ValueError('Exclusive racket source requires continuous buffered/offline processing')
        self.startup_timing_ms={}
        self.full_analysis = processing_mode in ('offline', 'buffered') and self.config['mode'] in ('active', 'shadow')
        if self.config['ipc_transport']=='shared_memory' and not self.full_analysis:
            raise ValueError('Shared source IPC requires continuous processing and retained source slots')
        self.disabled_reason=None;self.results={};self.last_progress=time.monotonic();self.submitted=False
        self.outstanding = {}
        self.shared_leases = {}
        self.metrics={'submitted':0,'replaced_pending':0,'skipped':0,'matched':0,'late':0,'fallback':0}
        self.journal_path=journal_path;self.model_sha256=None
        ctx=mp.get_context('spawn');self.incoming=ctx.Queue(maxsize=1);self.outgoing=ctx.Queue(maxsize=8)
        self.process=ctx.Process(target=worker_target or _worker,args=(self.config,self.incoming,self.outgoing,journal_path),name='tennis-racket-pose')
        self.process.start()
        deadline=time.monotonic()+self.config['startup_timeout_seconds']
        while time.monotonic()<deadline:
            try:message=self.outgoing.get(timeout=.05)
            except queue.Empty:
                if not self.process.is_alive():break
                continue
            if message.get('kind')=='ready':
                self.model_sha256=message.get('model_sha256')
                self.startup_timing_ms={**message.get('startup_timing_ms',{}),'spawn_to_ready_ms':(time.perf_counter()-startup_started)*1000}
                return
            if message.get('kind')=='error':self._disable(message['error']);return
        self._disable('Racket pose worker failed to become ready')

    def _disable(self,reason):
        self.disabled_reason=reason
        self.results.clear();self.outstanding.clear()
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=1)
        getattr(self,'shared_leases',{}).clear()
        if self.full_analysis:raise RuntimeError(reason)

    def _pump(self):
        if self.disabled_reason:return
        while True:
            try:message=self.outgoing.get_nowait()
            except queue.Empty:break
            self.last_progress=time.monotonic()
            if message.get('kind')=='error':self._disable(message['error']);return
            if message.get('kind')=='result':
                result=message['result'];self.results[result['frame_id']]=result
                self.outstanding.pop(result['frame_id'], None)
                self.shared_leases={name:fid for name,fid in getattr(self,'shared_leases',{}).items() if fid!=result['frame_id']}
                while len(self.results)>16:self.results.pop(min(self.results));self.metrics['late']+=1
        if not self.process.is_alive():self._disable('Racket worker exited: '+str(self.process.exitcode))
        elif self.outstanding and time.monotonic()-self.last_progress>self.config['timeout_seconds']:
            self._disable('Racket worker inference timed out')

    def submit(self,frame,frame_id,source_time,shared_frame=None):
        self._pump()
        if self.disabled_reason:return False
        if self.processing_mode=='realtime' and frame_id%self.config['inference_stride']:
            self.metrics['skipped']+=1;return False
        task={'frame_id':frame_id,'source_time':source_time,'submitted_ns':time.monotonic_ns(),
              'submitted_at_unix_ns':time.time_ns()}
        if self.config['ipc_transport']=='shared_memory':
            if not shared_frame or shared_frame['generation']!=frame_id:
                raise ValueError('Shared source IPC requires a matching leased descriptor')
            if shared_frame['name'] in self.shared_leases:
                raise ValueError('Shared source slot is already leased')
            task['shared_frame']=dict(shared_frame)
            self.shared_leases[shared_frame['name']]=frame_id
        else:
            from racket_pose_detector import letterbox
            started=time.perf_counter()
            image, transform = letterbox(frame, self.config['imgsz'])
            task.update(frame=image,prepared={'transform':transform,'preprocess_ms':(time.perf_counter()-started)*1000})
        task['submitted_at_unix_ns']=time.time_ns()
        was_idle = not self.outstanding
        if self.full_analysis:
            try:self.incoming.put(task,timeout=self.config['timeout_seconds'])
            except queue.Full:self._disable('Racket input timed out');return False
        else:
            try:self.incoming.put_nowait(task)
            except queue.Full:
                try:
                    replaced = self.incoming.get(timeout=.001)
                    self.outstanding.pop(replaced['frame_id'], None)
                    self.metrics['replaced_pending']+=1
                except queue.Empty:pass
                try:self.incoming.put_nowait(task)
                except queue.Full:self.metrics['skipped']+=1;return False
        self.metrics['submitted']+=1;self.submitted=True
        if was_idle: self.last_progress = time.monotonic()
        self.outstanding[frame_id] = time.monotonic()
        # Lost notifications are journaled; completed work must not grow a parent cache forever.
        while len(self.outstanding) > 16:
            self.outstanding.pop(min(self.outstanding))
        return True

    def resolve(self,frame_id,submitted=True):
        wait=(self.config['timeout_seconds'] if self.full_analysis else
              self.config['join_ms']/1000 if self.config['mode']=='active' else 0)
        deadline=time.monotonic()+wait if submitted else time.monotonic()
        while True:
            self._pump()
            for old in [k for k in self.results if k<frame_id]:self.results.pop(old);self.metrics['late']+=1
            if frame_id in self.results:
                result=self.results.pop(frame_id);self.metrics['matched']+=1;return result
            if self.disabled_reason or time.monotonic()>=deadline:break
            time.sleep(.001)
        if self.full_analysis and submitted and not self.disabled_reason:
            self._disable('Racket frame result timed out')
        self.metrics['fallback']+=1
        return {'frame_id':frame_id,'status':'disabled' if self.disabled_reason else 'pending_or_skipped',
                'reason':self.disabled_reason,'candidates':[]}

    def close(self):
        if not self.disabled_reason:
            deadline=time.monotonic()+min(1.0,self.config['timeout_seconds'])
            while self.submitted and time.monotonic()<deadline:
                self._pump()
                try:self.incoming.put_nowait(None);break
                except queue.Full:time.sleep(.01)
        self.process.join(timeout=.5)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=1)
        if self.journal_path:
            summary={'config':self.config,'model_sha256':self.model_sha256,'metrics':self.metrics,
                     'disabled_reason':self.disabled_reason,'processing_mode':self.processing_mode,'startup_timing_ms':self.startup_timing_ms}
            Path(self.journal_path).with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        for q in (self.incoming,self.outgoing):q.cancel_join_thread();q.close()
