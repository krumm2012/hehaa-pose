import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from racket_pose_detector import decode_candidates, letterbox, pose_config, select_pose_candidate, verify_acceptance, verify_pipeline_acceptance
from racket_pose_runtime import RacketPoseRuntime


def fake_worker(config, incoming, outgoing, journal):
    outgoing.put({'kind':'ready','model_sha256':'fake'})
    while True:
        task=incoming.get()
        if task is None:return
        if config.get('test_crash'):os._exit(9)
        time.sleep(config.get('test_delay',0))
        outgoing.put({'kind':'result','result':{'frame_id':task['frame_id'],'status':'observed',
                     'pixel':int(task['frame'][0,0,0]),'source_time':task['source_time'],'candidates':[]}})


def unordered_worker(config, incoming, outgoing, journal):
    outgoing.put({'kind':'ready','model_sha256':'fake'})
    first=incoming.get();second=incoming.get()
    for frame_id in (second['frame_id'],first['frame_id']):
        outgoing.put({'kind':'result','result':{'frame_id':frame_id,'status':'observed','candidates':[]}})
    while incoming.get() is not None:pass


class RacketContractTests(unittest.TestCase):
    def config(self,**kwargs):return pose_config({'mode':'shadow',**kwargs})

    def test_letterbox_and_inverse_points(self):
        image,transform=letterbox(np.zeros((1440,2560,3),np.uint8))
        self.assertEqual(image.shape,(960,960,3));self.assertEqual(transform['top'],210)
        raw=np.zeros((1,26,18900),np.float32);raw[0,:5,0]=[480,480,120,120,.9]
        raw[0,5:,0]=np.tile([480,480,.9],7)
        rows,diag=decode_candidates(raw,transform,self.config())
        self.assertEqual(len(rows),1);self.assertEqual(diag['nonfinite_candidates'],0)
        self.assertAlmostEqual(rows[0]['keypoints']['grip_center']['x'],1280)
        self.assertAlmostEqual(rows[0]['keypoints']['grip_center']['y'],720)

    def test_low_score_nan_still_disqualifies_measurement(self):
        raw=np.zeros((1,26,18900),np.float32);raw[0,0,100]=np.nan
        raw[0,:5,0]=[480,480,120,120,.9];raw[0,5:,0]=np.tile([480,480,.9],7)
        _,t=letterbox(np.zeros((960,960,3),np.uint8))
        rows,diag=decode_candidates(raw,t,self.config())
        self.assertEqual(diag['nonfinite_candidates'],1)
        self.assertFalse(rows[0]['observed']);self.assertFalse(rows[0]['measurement_eligible'])

    def test_nms_keeps_high_score_candidate_and_low_kp_not_observed(self):
        a=np.zeros((1,26,18900),np.float32)
        for i in (0,1):
            a[0,:5,i]=[480,480,120,120,.9-i*.1];a[0,5:,i]=np.tile([480,480,.2],7)
        _,t=letterbox(np.zeros((960,960,3),np.uint8));rows,_=decode_candidates(a,t,self.config())
        self.assertEqual(len(rows),1);self.assertFalse(rows[0]['keypoints']['grip_center']['observed'])

    def test_active_rejects_late_or_recovered_wrists(self):
        candidate={'keypoints':{'grip_center':{'x':100,'y':100,'confidence':.9,'observed':True,'source_frame_id':7}},'box':[80,80,120,120],'source_frame_id':7}
        row={'frame_id':7,'status':'observed','candidates':[candidate]}
        self.assertIsNone(select_pose_candidate(row,8,[(100,100)],(1000,1000),self.config()))
        self.assertIsNone(select_pose_candidate(row,7,[{'x':100,'y':100,'observed':False,'source_frame_id':7}],(1000,1000),self.config()))
        self.assertIsNotNone(select_pose_candidate(row,7,[(100,100)],(1000,1000),self.config()))

    def test_acceptance_binds_digest_and_backend(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'accept.json';p.write_text(json.dumps({'model_sha256':'abc','numerical_valid':True,'quality_gate_passed':True,'backend':'coreml','compute_units':'ANE'}))
            c=self.config(acceptance_path=str(p));verify_acceptance(c,'abc')
            with self.assertRaises(ValueError):verify_acceptance(c,'different')
            c['compute_units']='ALL'
            with self.assertRaises(ValueError):verify_acceptance(c,'abc')

    def test_realtime_gate_binds_pipeline_profile_and_three_rounds(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'runtime.json';c=self.config(runtime_acceptance_path=str(p),source_dimensions=[2560,1440],source_fps=25.07)
            profile={k:c[k] for k in ('inference_stride','imgsz','confidence_threshold','keypoint_threshold','iou_threshold')}
            profile.update(passed=True,runs=27,source_dimensions=[2560,1440],max_source_fps=25.1)
            p.write_text(json.dumps({'model_sha256':'abc','backend':'coreml','compute_units':'ANE','profiles':{'single':profile}}))
            verify_pipeline_acceptance(c,'abc')
            for change in ({'pipeline_profile':'dual'},{'inference_stride':2},{'source_dimensions':[1920,1080]},{'source_fps':60}):
                with self.assertRaises(ValueError):verify_pipeline_acceptance({**c,**change},'abc')
            profile['runs']=3
            p.write_text(json.dumps({'model_sha256':'abc','backend':'coreml','compute_units':'ANE','profiles':{'single':profile}}))
            with self.assertRaises(ValueError):verify_pipeline_acceptance(c,'abc')

    def test_validated_schedule_is_scoped_and_experiments_keep_requested_stride(self):
        from racket_pose_detector import model_digest
        from racket_pose_runtime import accepted_runtime_config
        with tempfile.TemporaryDirectory() as d:
            model=Path(d)/'model';model.write_bytes(b'model identity')
            p=Path(d)/'pipeline.json'
            c=self.config(model_path=str(model),runtime_acceptance_path=str(p),source_dimensions=[2560,1440],source_fps=25.07)
            profile={k:c[k] for k in ('imgsz','confidence_threshold','keypoint_threshold','iou_threshold')}
            profile.update(passed=True,runs=27,inference_stride=2,source_dimensions=[2560,1440],max_source_fps=25.1)
            p.write_text(json.dumps({'model_sha256':model_digest(model),'backend':'coreml','compute_units':'ANE','profiles':{'single':profile}}))
            self.assertEqual(accepted_runtime_config(c,'realtime')['inference_stride'],2)
            self.assertEqual(c['inference_stride'],1)
            self.assertEqual(accepted_runtime_config({**c,'acceptance_run':True},'realtime')['inference_stride'],1)
            self.assertEqual(accepted_runtime_config({**c,'source_fps':60},'realtime')['inference_stride'],1)
            self.assertEqual(accepted_runtime_config(c,'offline')['inference_stride'],1)
            self.assertEqual(accepted_runtime_config(c,'buffered')['inference_stride'],1)

    def test_legacy_bypass_ignores_unused_new_model_configuration(self):
        c=pose_config({'mode':'legacy','backend':'unavailable','compute_units':'GPU','imgsz':640})
        self.assertEqual(c['mode'],'legacy')

    def test_gpu_and_invalid_config_rejected(self):
        for c in ({'compute_units':'GPU'},{'inference_stride':0},{'imgsz':640},{'join_ms':float('nan')}):
            with self.assertRaises(ValueError):pose_config({**c,'mode':'shadow'})

    def test_seven_point_order_and_out_of_frame_measurements(self):
        from racket_pose_detector import KEYPOINT_NAMES
        raw=np.zeros((1,26,18900),np.float32)
        raw[0,:5,0]=[480,480,500,500,.9]
        points=np.asarray([[100+i*50,200+i*30,.9] for i in range(7)],np.float32)
        points[3,0]=-100
        raw[0,5:,0]=points.reshape(-1)
        _,t=letterbox(np.zeros((960,960,3),np.uint8))
        rows,_=decode_candidates(raw,t,self.config())
        for i,name in enumerate(KEYPOINT_NAMES):self.assertEqual(rows[0]['keypoints'][name]['x'],points[i,0])
        self.assertFalse(rows[0]['keypoints']['sweet_spot']['observed'])
        self.assertFalse(rows[0]['measurement_eligible'])


class RacketWorkerTests(unittest.TestCase):
    def runtime(self,processing='offline',**config):
        return RacketPoseRuntime({'mode':'shadow','startup_timeout_seconds':5,'timeout_seconds':.5,**config},processing,worker_target=fake_worker)

    def test_buffered_waits_for_every_frame_despite_sampling_config(self):
        worker=self.runtime('buffered',mode='active',inference_stride=3,test_delay=.01)
        try:
            for fid in range(7):
                self.assertTrue(worker.submit(np.zeros((8,8,3),np.uint8),fid,{'source_frame_id':fid}))
                self.assertEqual(worker.resolve(fid)['frame_id'],fid)
            self.assertEqual(worker.metrics['matched'],7)
            self.assertEqual(worker.metrics['skipped']+worker.metrics['replaced_pending']+worker.metrics['fallback'],0)
        finally:worker.close()

    def test_buffered_timeout_is_failure_not_successful_fallback(self):
        worker=self.runtime('buffered',mode='active',test_delay=2,timeout_seconds=.1)
        try:
            worker.submit(np.zeros((8,8,3),np.uint8),0,None)
            with self.assertRaises(RuntimeError):worker.resolve(0)
        finally:worker.close()

    def test_offline_same_frame_owned_copy_and_time(self):
        worker=self.runtime(mode='active')
        try:
            frame=np.full((8,8,3),42,np.uint8);worker.submit(frame,4,{'timestamp_seconds':.16});frame[:]=99
            r=worker.resolve(4);self.assertEqual(r['pixel'],42);self.assertEqual(r['source_time']['timestamp_seconds'],.16)
        finally:worker.close()
        self.assertFalse(worker.process.is_alive())

    def test_shadow_does_not_wait_and_late_is_not_current(self):
        worker=self.runtime('realtime',test_delay=.08)
        try:
            worker.submit(np.zeros((8,8,3),np.uint8),1,None);t=time.monotonic();r=worker.resolve(1)
            self.assertLess(time.monotonic()-t,.03);self.assertEqual(r['status'],'pending_or_skipped')
            time.sleep(.15);r=worker.resolve(2,False);self.assertNotEqual(r['status'],'observed');self.assertGreater(worker.metrics['late'],0)
        finally:worker.close()

    def test_native_crash_realtime_degrades_and_offline_active_fails(self):
        worker=self.runtime('realtime',test_crash=True)
        try:
            worker.submit(np.zeros((8,8,3),np.uint8),1,None);time.sleep(.1);r=worker.resolve(1)
            self.assertEqual(r['status'],'disabled');self.assertIn('exited',r['reason'])
        finally:worker.close()
        worker=self.runtime('offline',mode='active',test_crash=True)
        try:
            worker.submit(np.zeros((8,8,3),np.uint8),1,None)
            with self.assertRaises(RuntimeError):worker.resolve(1)
        finally:worker.close()

    def test_hung_worker_timeout_terminates(self):
        worker=self.runtime('offline',mode='active',test_delay=2,timeout_seconds=.1)
        try:
            worker.submit(np.zeros((8,8,3),np.uint8),1,None)
            with self.assertRaises(RuntimeError):worker.resolve(1)
        finally:worker.close()
        self.assertFalse(worker.process.is_alive())

    def test_realtime_queue_and_cache_remain_bounded(self):
        worker=self.runtime('realtime',test_delay=.02)
        try:
            for i in range(60):worker.submit(np.zeros((8,8,3),np.uint8),i,None);worker.resolve(i,False)
            self.assertEqual(worker.incoming._maxsize,1);self.assertLessEqual(len(worker.results),16)
        finally:worker.close()

    def test_out_of_order_results_keep_their_source_frame(self):
        worker=RacketPoseRuntime({'mode':'active','timeout_seconds':.5},'offline',worker_target=unordered_worker)
        try:
            worker.submit(np.zeros((8,8,3),np.uint8),7,None)
            worker.submit(np.zeros((8,8,3),np.uint8),8,None)
            self.assertEqual(worker.resolve(7)['frame_id'],7)
            self.assertEqual(worker.resolve(8)['frame_id'],8)
            self.assertNotEqual(worker.resolve(9,False)['status'],'observed')
        finally:worker.close()

    def test_missing_model_realtime_records_failure_without_raising(self):
        worker=RacketPoseRuntime({'mode':'shadow','model_path':'/__tennis_missing_model__','startup_timeout_seconds':3},'realtime')
        try:
            self.assertIn('FileNotFoundError',worker.disabled_reason)
            self.assertFalse(worker.submit(np.zeros((8,8,3),np.uint8),1,None))
        finally:worker.close()

    def test_ipc_latency_uses_shared_wall_time_not_process_local_monotonic(self):
        import queue
        from unittest.mock import patch
        from racket_pose_runtime import _worker
        class Detector:
            digest='fake'
            def __init__(self,config):pass
            def detect(self,*args):return {'frame_id':args[1],'status':'observed','timing':{}}
        incoming=queue.Queue();outgoing=queue.Queue()
        incoming.put({'frame':np.zeros((8,8,3),np.uint8),'frame_id':7,'source_time':None,
            'submitted_ns':10**30,'submitted_at_unix_ns':time.time_ns()-20_000_000})
        incoming.put(None)
        with patch('racket_pose_detector.RacketPoseDetector',Detector),patch('signal.signal'):
            _worker(pose_config({'mode':'shadow'}),incoming,outgoing,None)
        outgoing.get_nowait()
        result=outgoing.get_nowait()['result']
        self.assertGreaterEqual(result['timing']['queue_wait_ms'],15)
        self.assertLess(result['timing']['queue_wait_ms'],500)

    def test_exit_quarantines_even_a_cached_current_result(self):
        import queue
        from unittest.mock import Mock
        worker=RacketPoseRuntime.__new__(RacketPoseRuntime)
        worker.config=pose_config({'mode':'active','join_ms':0})
        worker.processing_mode='realtime';worker.full_analysis=False;worker.disabled_reason=None
        worker.results={7:{'frame_id':7,'status':'observed','candidates':[]}}
        worker.outstanding={};worker.metrics={'fallback':0,'matched':0,'late':0}
        worker.outgoing=Mock();worker.outgoing.get_nowait.side_effect=queue.Empty
        worker.process=Mock();worker.process.is_alive.return_value=False;worker.process.exitcode=9
        self.assertEqual(worker.resolve(7)['status'],'disabled')
        self.assertEqual(worker.results,{})

    def test_realtime_model_is_quarantined_before_loading_without_pipeline_receipt(self):
        from racket_pose_detector import model_digest
        with tempfile.TemporaryDirectory() as d:
            model=Path(d)/'model';model.write_text('A model that must never be loaded')
            acceptance=Path(d)/'numeric.json'
            acceptance.write_text(json.dumps({'model_sha256':model_digest(model),'numerical_valid':True,
                'quality_gate_passed':True,'backend':'coreml','compute_units':'ANE'}))
            worker=RacketPoseRuntime({'mode':'active','model_path':str(model),'acceptance_path':str(acceptance),
                                     'startup_timeout_seconds':3},'realtime')
            try:
                self.assertIn('pipeline acceptance',worker.disabled_reason)
                self.assertFalse(worker.submit(np.zeros((8,8,3),np.uint8),1,None))
            finally:worker.close()


if __name__=='__main__':unittest.main()
