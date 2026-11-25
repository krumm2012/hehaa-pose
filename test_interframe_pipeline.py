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
        for mode in ('active', 'shadow'):
            for options in ({'test_crash':True},{'test_delay':2}):
                with self.subTest(mode=mode, fault=options):
                    worker=RacketPoseRuntime({'mode':mode,'ipc_transport':'shared_memory',
                        'startup_timeout_seconds':5,'timeout_seconds':.1,**options},'buffered',worker_target=transport_worker)
                    try:
                        worker.submit(self.image,7,None,self.descriptor)
                        with self.assertRaises(RuntimeError):worker.resolve(7)
                        self.assertFalse(worker.process.is_alive())
                        self.assertFalse(worker.shared_leases)
                    finally:worker.close()
                    self.assertEqual(int(self.image[0,0,0]),42)

    def test_continuous_shadow_uses_every_frame_and_releases_shared_slots(self):
        worker=RacketPoseRuntime({'mode':'shadow','imgsz':640,'ipc_transport':'shared_memory',
            'inference_stride':3,'startup_timeout_seconds':5,'timeout_seconds':1},
            'buffered',worker_target=transport_worker)
        try:
            for fid in range(7,14):
                self.write_generation(fid)
                worker.submit(self.image,fid,{'source_frame_id':fid},dict(self.descriptor,generation=fid))
                result=worker.resolve(fid)
                self.assertEqual(result['frame_id'],fid)
                self.assertEqual(result['source_time']['source_frame_id'],fid)
                self.assertFalse(worker.shared_leases)
            self.assertEqual(worker.metrics['matched'],7)
            self.assertEqual(sum(worker.metrics[k] for k in ('skipped','replaced_pending','fallback','late')),0)
        finally:worker.close()
        self.assertFalse(worker.process.is_alive())

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

    def test_three_stage_preprocess_preserves_order_and_propagates_errors(self):
        inputs = iter(range(6))
        stop = threading.Event()
        pipe = InterframePipeline(
            lambda cancelled: next(inputs, None),
            lambda item: {'frame': item['frame'], 'prep': item['prep']},
            stop,
            close_timeout=1,
            max_in_flight=3,
            preprocess=lambda n: {'frame': n, 'prep': n * 10},
        )
        try:
            self.assertEqual(list(pipe), [{'frame': i, 'prep': i * 10} for i in range(6)])
        finally:
            pipe.close()

        inputs_err = iter(range(4))
        stop_err = threading.Event()
        pipe_err = InterframePipeline(
            lambda cancelled: next(inputs_err, None),
            lambda item: item,
            stop_err,
            close_timeout=1,
            max_in_flight=3,
            preprocess=lambda n: (_ for _ in ()).throw(ValueError('prep failed')) if n == 1 else n,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, 'preparation failed'):
                list(pipe_err)
        finally:
            pipe_err.close()

    def test_detector_prepare_input_packet_matches_preprocess_and_reuses_letterbox(self):
        det = YOLO26nUnifiedDetector.__new__(YOLO26nUnifiedDetector)
        det.input_width = 960
        det.input_height = 960
        det.preprocess_mode = 'letterbox'
        det.model = Mock()
        det.model.predict.return_value = {}
        det._coreml_input_names = set()
        det.detection_times = deque(maxlen=10)
        det.last_parse_diagnostics = {}
        det.last_ball_diagnostics = {}
        det.last_racket_diagnostics = {}
        det.roi_manager = None
        det.config = {}
        det.racket_conf_threshold = 0.5
        det.ball_conf_threshold = 0.5
        det._parse_predictions = lambda preds: ([], [])
        det._filter_static_balls = lambda b, r: b
        det._select_primary_racket = lambda r, b: r

        frame = np.full((1440, 2560, 3), 64, dtype=np.uint8)
        pkt1 = det.prepare_input_packet(frame)
        self.assertEqual(pkt1[0].size, (960, 960))
        self.assertEqual(pkt1[1:], (960 / 2560, (0, 210), 1440, 2560))
        geom_first = det._letterbox_geom
        pkt2 = det.prepare_input_packet(frame)
        self.assertEqual(det._letterbox_geom, geom_first)
        self.assertEqual(np.asarray(pkt1[0])[0, 0].tolist(), [114, 114, 114])
        self.assertEqual(np.asarray(pkt2[0])[300, 100].tolist(), [64, 64, 64])

        det.detect_unified(frame, prepared_input=pkt2)
        self.assertEqual(det._preprocess_gain, pkt2[1])
        self.assertEqual(det._preprocess_pad, pkt2[2])


if __name__=='__main__':unittest.main()
