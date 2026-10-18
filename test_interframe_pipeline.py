import queue
import struct
import threading
import time
import unittest
from collections import deque
from multiprocessing import shared_memory
from unittest.mock import Mock

import numpy as np

from frame_transport import FrameTransportReader
from interframe_pipeline import InterframePipeline
from racket_pose_runtime import RacketPoseRuntime
from yolo26n_unified_detector import YOLO26nUnifiedDetector


def transport_worker(config, incoming, outgoing, journal):
    reader=FrameTransportReader()
    outgoing.put({'kind':'ready','model_sha256':'fake'})
    try:
        while True:
            task=incoming.get()
            if task is None:break
            if config.get('test_crash'):
                import os
                os._exit(9)
            with reader.read(task) as frame:
                time.sleep(config.get('test_delay',0))
                pixel=int(frame[0,0,0])
            outgoing.put({'kind':'result','result':{'frame_id':task['frame_id'],
                'source_time':task['source_time'],'pixel':pixel,'status':'observed','candidates':[]}})
    except Exception as exc:
        outgoing.put({'kind':'error','error':str(exc)})
    finally:
        reader.close()


class SharedFrameTests(unittest.TestCase):
    def setUp(self):
        self.shape=(16,24,3);self.size=int(np.prod(self.shape))
        self.shm=shared_memory.SharedMemory(create=True,size=self.size+8)
        self.image=np.ndarray(self.shape,np.uint8,buffer=self.shm.buf)
        self.image[:]=42
        self.write_generation(7)
        self.descriptor={'name':self.shm.name,'slot':0,'shape':self.shape,'dtype':'uint8','generation':7}

    def tearDown(self):
        self.shm.close();self.shm.unlink()

    def write_generation(self,generation):
        struct.pack_into('<q',self.shm.buf,self.size,generation)

    def task(self):
        return {'frame_id':7,'shared_frame':self.descriptor}

    def test_readonly_borrow_does_not_copy_or_unlink_owner(self):
        reader=FrameTransportReader()
        try:
            with reader.read(self.task()) as view:
                self.assertTrue(np.shares_memory(view,np.ndarray(self.shape,np.uint8,buffer=reader.handles[self.shm.name].buf)))
                with self.assertRaises(ValueError):view[0,0,0]=5
                self.assertEqual(int(view[0,0,0]),42)
        finally:reader.close()
        self.assertEqual(self.image[0,0,0],42)

    def test_overwrite_before_or_during_read_and_replay_are_rejected(self):
        reader=FrameTransportReader()
        try:
            self.write_generation(8)
            with self.assertRaisesRegex(ValueError,'overwritten'):
                with reader.read(self.task()):pass
            self.write_generation(7)
            with self.assertRaisesRegex(ValueError,'overwritten'):
                with reader.read(self.task()):self.write_generation(8)
            self.write_generation(7)
            with reader.read(self.task()):pass
            with self.assertRaisesRegex(ValueError,'Replayed'):
                with reader.read(self.task()):pass
        finally:reader.close()

    def test_spawned_worker_matches_identity_and_safely_reuses_slot(self):
        worker=RacketPoseRuntime({'mode':'active','ipc_transport':'shared_memory',
            'startup_timeout_seconds':5,'timeout_seconds':1},'buffered',worker_target=transport_worker)
        try:
            self.assertTrue(worker.submit(self.image,7,{'pts':.28},self.descriptor))
            with self.assertRaisesRegex(ValueError,'already leased'):
                worker.submit(self.image,8,None,{**self.descriptor,'generation':8})
            result=worker.resolve(7)
            self.assertEqual((result['pixel'],result['frame_id'],result['source_time']),(42,7,{'pts':.28}))
            self.assertFalse(worker.shared_leases)
            self.image[:]=99;self.write_generation(8)
            worker.submit(self.image,8,{'pts':.32},{**self.descriptor,'generation':8})
            self.assertEqual(worker.resolve(8)['pixel'],99)
        finally:worker.close()
        self.assertFalse(worker.process.is_alive())
        self.assertEqual(self.image[0,0,0],99)

    def test_invalid_shared_generation_fails_continuous_session(self):
        worker=RacketPoseRuntime({'mode':'active','ipc_transport':'shared_memory',
            'startup_timeout_seconds':5,'timeout_seconds':1},'offline',worker_target=transport_worker)
        try:
            self.write_generation(8)
            worker.submit(self.image,7,None,self.descriptor)
            with self.assertRaisesRegex(RuntimeError,'overwritten'):worker.resolve(7)
        finally:worker.close()

    def test_shared_worker_crash_and_hang_fail_without_reclaiming_parent_memory(self):
        for options in ({'test_crash':True},{'test_delay':2}):
            worker=RacketPoseRuntime({'mode':'active','ipc_transport':'shared_memory',
                'startup_timeout_seconds':5,'timeout_seconds':.1,**options},'buffered',worker_target=transport_worker)
            try:
                worker.submit(self.image,7,None,self.descriptor)
                with self.assertRaises(RuntimeError):worker.resolve(7)
                self.assertFalse(worker.process.is_alive())
                self.assertFalse(worker.shared_leases)
            finally:worker.close()
            self.assertEqual(int(self.image[0,0,0]),42)

    def test_unleased_realtime_shared_input_is_rejected_before_spawn(self):
        with self.assertRaisesRegex(ValueError,'retained source slots'):
            RacketPoseRuntime({'mode':'shadow','ipc_transport':'shared_memory'},'realtime')


class StartupControlTests(unittest.TestCase):
    def test_startup_controls_are_explicit_and_invalid_combinations_rejected(self):
        from main_pipe import build_argument_parser
        from local_control_panel import ControlSettings
        args=build_argument_parser().parse_args(['--inference-schedule','interframe','--racket-ipc','shared_memory'])
        self.assertEqual((args.inference_schedule,args.racket_ipc),('interframe','shared_memory'))
        defaults=ControlSettings.from_payload({'stream_id':'local_video'})
        self.assertEqual((defaults.inference_schedule,defaults.racket_ipc),('barrier','queue'))
        valid=ControlSettings.from_payload({'stream_id':'local_video','racket_pose_mode':'exclusive',
            'inference_schedule':'interframe','racket_ipc':'shared_memory'})
        self.assertEqual(valid.inference_schedule,'interframe')
        for payload in ({'inference_schedule':'interframe'},
                        {'processing_mode':'realtime','racket_pose_mode':'shadow','racket_ipc':'shared_memory'}):
            with self.assertRaises(ValueError):ControlSettings.from_payload({'stream_id':'local_video',**payload})


class InterframeTests(unittest.TestCase):
    def make(self,prepare,count=4):
        inputs=iter(range(count));stop=threading.Event()
        return InterframePipeline(lambda cancelled:next(inputs,None),prepare,stop,close_timeout=1)

    def test_next_preparation_overlaps_aggregation_but_third_frame_cannot_run_ahead(self):
        prepared=[];second=threading.Event()
        def prepare(frame):
            prepared.append(frame)
            if frame==1:second.set()
            return frame
        pipeline=self.make(prepare)
        iterator=iter(pipeline)
        try:
            self.assertEqual(next(iterator),0)
            self.assertTrue(second.wait(1))
            self.assertEqual(prepared,[0,1])
            self.assertEqual(list(iterator),[1,2,3])
        finally:iterator.close();pipeline.close()

    def test_empty_and_short_eof_drain_in_order(self):
        for count in (0,1,2,19):
            p=self.make(lambda n:{'frame':n},count)
            try:self.assertEqual([r['frame'] for r in p],list(range(count)))
            finally:p.close()

    def test_preparation_failure_propagates_instead_of_successful_eof(self):
        def prepare(n):
            if n==1:raise ValueError('model failed')
            return n
        p=self.make(prepare)
        try:
            with self.assertRaisesRegex(RuntimeError,'preparation failed'):list(p)
        finally:p.close()

    def test_cancel_when_pending_queue_is_full_terminates(self):
        second=threading.Event()
        def prepare(n):
            if n==1:second.set()
            return n
        p=self.make(prepare)
        self.assertTrue(second.wait(1));p.close()
        self.assertFalse(p.thread.is_alive())

    def test_candidate_snapshot_cannot_be_overwritten_by_next_frame(self):
        detector=YOLO26nUnifiedDetector.__new__(YOLO26nUnifiedDetector)
        detector.model=object();detector.detection_times=deque()
        detector.last_parse_diagnostics={};detector.last_racket_diagnostics={}
        detector.last_ball_diagnostics={};detector.static_ball_filter=object();detector.ball_track_selector=object()
        raw=detector.candidate_inference_view()
        self.assertIs(raw.model,detector.model)
        self.assertIsNone(raw.ball_track_selector)
        def infer(frame,**kwargs):
            raw.last_parse_diagnostics={'value':[int(frame)]}
            raw.last_racket_diagnostics={'source':int(frame)}
            raw.original_height=10;raw.original_width=20;raw.tracking_frame_width=20;raw.tracking_frame_height=10
            return [{'position':[int(frame),0]}],[],0
        raw.detect_unified=infer
        first=raw.external_candidate_packet(7,7,(20,10))
        raw.external_candidate_packet(8,8,(20,10))
        self.assertEqual(detector.adopt_external_packet(first,7)[0]['position'],[7,0])
        self.assertEqual(detector.last_parse_diagnostics,{'value':[7]})
        self.assertEqual(raw.last_parse_diagnostics,{'value':[8]})
        with self.assertRaises(ValueError):detector.adopt_external_packet(first,8)

if __name__=='__main__':unittest.main()
