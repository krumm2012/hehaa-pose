"""Inference worker process for MultiprocessPipeline."""

from __future__ import annotations

import os
import queue
import time
import traceback
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np


def run_inference_process(pipeline, ignore_child_interrupts_fn=None):
    """进程 2: AI 推理核心 (平行调度)"""
    (ignore_child_interrupts_fn() if ignore_child_interrupts_fn else None)
    print("🚀 [Inference] 加载 Core ML 并行架构...")
    from concurrent.futures import ThreadPoolExecutor
    from performance_metrics import timed_call
    shms = []
    executor = None
    dual_pose_estimator = None
    racket_pose_runtime = None
    interframe = None
    try:
        from detection_frame_context import DetectionFrameContext
        from pose_estimator_yolo26 import PoseEstimatorYOLO26
        from roi_manager import ROIManager
        from yolo26n_unified_detector import YOLO26nUnifiedDetector

        roi_manager = None
        if pipeline.roi_profile.enabled:
            candidate = ROIManager(pipeline.config)
            if candidate.set_roi_points(pipeline.roi_profile.points):
                # 自动将镜面多边形注册为排除区域，阻止镜中虚像被锁定追踪
                if pipeline.roi_profile.has_mirror_view and pipeline.roi_profile.mirror_polygon:
                    mirror_px = pipeline.roi_profile.get_mirror_polygon_pixels((pipeline.width, pipeline.height))
                    if mirror_px:
                        candidate.add_exclusion_polygon(mirror_px, label="mirror_zone")
                for ex_poly in pipeline.roi_profile.exclusion_polygons:
                    candidate.add_exclusion_polygon(ex_poly, label="custom_exclusion")
                roi_manager = candidate

        if (
            roi_manager is not None
            and pipeline.algo2_dual_view
            and pipeline.config.get("roi_settings", {}).get("preserve_upper_airspace", True)
        ):
            roi_manager.configure_ball_airspace((pipeline.width, pipeline.height))

        detector = YOLO26nUnifiedDetector(
            pipeline.config['unified_detection']['model_path'],
            pipeline.config['unified_detection'],
            roi_manager=roi_manager,
        )
        pose_estimator = None
        dual_view_mgr = None
        dual_pose_estimator = None

        if pipeline.algo2_dual_view:
            from dual_view_manager import DualViewManager
            from dual_pose_estimator import DualPoseEstimator

            dual_view_mgr = DualViewManager(
                config_path=pipeline.algo2_config_path,
                stream_profile=pipeline.roi_profile,
            )
            dual_pose_estimator = DualPoseEstimator(
                model_path=pipeline.config.get('yolo_pose_model_path'),
                dominant_hand=pipeline.swing_analysis_options.get('dominant_hand', 'right'),
                backend='auto',
                compute_units=pipeline.config.get('pose_compute_units', 'ALL'),
            )
            print(f"🚀 [Inference] 双视角姿态引擎就绪 | 后端: {dual_pose_estimator.backend} | 请求计算单元: {dual_pose_estimator.compute_units}")
        else:
            pose_estimator = PoseEstimatorYOLO26(
                pipeline.config['yolo_pose_model_path'],
                pipeline.config,
                roi_manager=roi_manager,
            )

        shms = [shared_memory.SharedMemory(name=name) for name in pipeline.shm_names]
        shared_frames = [
            np.ndarray((pipeline.height, pipeline.width, 3), dtype=np.uint8, buffer=s.buf)
            for s in shms
        ]
        perf_cfg = pipeline.config.get('pipeline_perf', {})
        unified_workers = (
            pipeline.config.get('unified_detection', {}).get('max_workers')
            or pipeline.config.get('performance_optimization', {}).get('max_workers')
            or 2
        )
        inference_workers = max(1, int(perf_cfg.get('inference_workers', unified_workers)))
        executor = ThreadPoolExecutor(max_workers=inference_workers)
        print(f"⚙️ [Inference] 线程池并发数: {inference_workers}")
        print(f"⚙️ [Inference] 启动调度: {pipeline.inference_schedule} | 球拍 IPC: {pipeline.racket_pose_config['ipc_transport']}")
        if pipeline.algo2_dual_view:
            print("⚙️ [Inference] 人体ROI缩放: " + perf_cfg.get('pose_resize_policy', 'legacy'))
        if pipeline.inference_schedule=='interframe':
            print('⚙️ [Inference] 两帧在途：候选/球拍准备与同帧人体/时序聚合重叠')
        else:
            print("⚙️ [Inference] 单帧并行任务数: 2（目标检测 + 姿态估计）")
        if inference_workers > 2 and pipeline.inference_schedule=='barrier':
            print(
                f"ℹ️ [Inference] 当前执行图每帧最多使用 2 个 worker；"
                f"其余 {inference_workers - 2} 个用于后续多帧并发扩展"
            )
        if pipeline.drop_stale_frames:
            print("⚡ [Inference] 直播新鲜度模式：Reader只保留最新待推理帧")
        if pipeline.racket_pose_config['mode'] != 'legacy':
            from racket_pose_runtime import RacketPoseRuntime
            racket_pose_runtime = RacketPoseRuntime(
                {**pipeline.racket_pose_config,'opencv_threads':perf_cfg.get('opencv_threads'),
                 'pipeline_profile':'dual' if pipeline.algo2_dual_view else 'single',
                 'source_dimensions':[pipeline.width,pipeline.height],'source_fps':pipeline.fps},
                pipeline.processing_mode, pipeline.racket_pose_journal,
            )
            if racket_pose_runtime.disabled_reason:
                print('⚠️ [RacketPose] 旁路停用，保留旧链路: ' + racket_pose_runtime.disabled_reason)
        if pipeline.lossless_realtime:
            from pipeline_startup import warm_coreml_wrappers
            wrappers = ([detector, dual_pose_estimator.model_front, dual_pose_estimator.model_back]
                        if dual_pose_estimator is not None else [detector, pose_estimator])
            warming = warm_coreml_wrappers(wrappers)
            print(f'🔥 [Inference] 源帧开始前完成 Core ML 内核预热: {warming} ms')
        pipeline.inf_ready.set()

        from racket_temporal_tracker import RacketTemporalTracker
        recovery_config = pipeline.config.get("unified_detection", {})
        racket_tracker = (RacketTemporalTracker(recovery_config)
                          if recovery_config.get("racket_temporal_recovery_enabled", False) else None)
        from racket_resolution import RacketResolver, RacketResolutionConfig
        racket_resolver = RacketResolver(RacketResolutionConfig.from_dict(pipeline.config))

        def read_next(cancelled=None):
            while not pipeline.stop_event.is_set() and not (cancelled and cancelled.is_set()):
                try:
                    return pipeline.q_inference.get(timeout=.1)
                except queue.Empty:
                    if pipeline.rd_done.is_set():
                        return None
            return None

        def source_descriptor(task):
            return {'name':pipeline.shm_names[task['slot']], 'slot':task['slot'],
                    'shape':[pipeline.height,pipeline.width,3], 'dtype':'uint8', 'generation':task['idx']}

        def serial_tasks():
            while not pipeline.stop_event.is_set():
                task=read_next()
                if task is None:
                    return
                yield task

        use_interframe = pipeline.inference_schedule == 'interframe'
        if use_interframe:
            from interframe_pipeline import InterframePipeline
            raw_detector = detector.candidate_inference_view()
            if pipeline.inference_schedule != 'interframe':
                print('⚡ [Inference] 离线多帧流水线并发已开启：相邻帧 检测/7点 与 双路人体姿态 交错重叠执行')

            def prepare(task):
                started = time.perf_counter()
                started_unix_ns = time.time_ns()
                frame = shared_frames[task['slot']]
                submitted = (
                    racket_pose_runtime.submit(frame, task['idx'], task.get('source_time'), source_descriptor(task))
                    if racket_pose_runtime is not None else False
                )
                dispatch_ms = (time.perf_counter() - started) * 1000
                ctx = (
                    DetectionFrameContext.build(task['idx'], frame, roi_manager, pipeline.config)
                    if roi_manager is not None and not pipeline.algo2_dual_view else None
                )
                det_frame = frame if pipeline.algo2_dual_view else (ctx.detection_frame if ctx else frame)
                det_offset = (0, 0) if pipeline.algo2_dual_view else (ctx.roi_offset if ctx else (0, 0))
                packet, detection_ms = timed_call(
                    raw_detector.external_candidate_packet,
                    det_frame,
                    task['idx'],
                    (pipeline.width, pipeline.height),
                    det_offset,
                    retain_racket_candidates=racket_tracker is not None,
                    racket_source='external' if pipeline.exclusive_racket_source else 'legacy',
                )
                join = time.perf_counter()
                observation = (
                    racket_pose_runtime.resolve(task['idx'], submitted)
                    if racket_pose_runtime is not None else None
                )
                return {
                    **task,
                    '_stage_a': {
                        'packet': packet,
                        'observation': observation,
                        'started_at_unix_ns': started_unix_ns,
                        'racket_dispatch_ms': dispatch_ms,
                        'legacy_detection_ms': detection_ms,
                        'racket_join_ms': (time.perf_counter() - join) * 1000,
                        'stage_a_ms': (time.perf_counter() - started) * 1000,
                    },
                }

            interframe = InterframePipeline(
                read_next,
                prepare,
                pipeline.stop_event,
                close_timeout=pipeline.racket_pose_config['timeout_seconds'] + 2,
            )
        tasks=interframe if interframe is not None else serial_tasks()
        for task in tasks:
            slot = task['slot']
            frame_ptr = shared_frames[slot]
            stage_a=task.get('_stage_a')
            stage_b_started=time.perf_counter()
            inference_started_at_unix_ns = stage_a['started_at_unix_ns'] if stage_a else time.time_ns()
            dispatch_started = time.perf_counter()
            frame_context = None
            detection_frame = frame_ptr
            pose_detection_frame = frame_ptr
            pose_submitted = (
                racket_pose_runtime.submit(frame_ptr, task['idx'], task.get('source_time'),source_descriptor(task))
                if racket_pose_runtime is not None and not stage_a else False
            )
            racket_pose_observation = stage_a['observation'] if stage_a else None
            racket_dispatch_ms = stage_a['racket_dispatch_ms'] if stage_a else (time.perf_counter() - dispatch_started) * 1000
            if roi_manager is not None:
                frame_context = DetectionFrameContext.build(
                    task['idx'],
                    frame_ptr,
                    roi_manager,
                    pipeline.config,
                )
                detection_frame = frame_context.detection_frame
                pose_detection_frame = frame_context.pose_detection_frame

            if pipeline.algo2_dual_view and dual_view_mgr is not None and dual_pose_estimator is not None:
                if stage_a:
                    from concurrent.futures import Future
                    f1 = Future()
                    candidates, adopted_rackets = detector.adopt_external_packet(
                        stage_a['packet'],
                        task['idx'],
                        return_rackets=True,
                        retain_racket_candidates=racket_tracker is not None,
                    )
                    f1.set_result(((candidates, adopted_rackets, 0.), stage_a['legacy_detection_ms']))
                else:
                    # Submit 960 detector before split_frame so crop/resize/flip overlaps with detector inference.
                    f1 = executor.submit(
                        timed_call,
                        detector.detect_unified,
                        frame_ptr,
                        (0, 0),
                        (pipeline.width, pipeline.height),
                        retain_racket_candidates=racket_tracker is not None,
                        racket_source='external' if pipeline.exclusive_racket_source else 'legacy',
                    )
                dual_frame = dual_view_mgr.split_frame(
                    frame_ptr, frame_id=task['idx'],
                    for_inference=(dual_pose_estimator.supports_source_pixels and
                        perf_cfg.get('pose_resize_policy', 'legacy') != 'legacy'),
                    source_views=(('front',) if perf_cfg.get('pose_resize_policy') == 'direct_front'
                                  else ('front', 'back')))
                f2 = executor.submit(timed_call, dual_pose_estimator.estimate_dual_pose, dual_frame)

                (ball, raw_rackets, _), legacy_detection_ms = f1.result()
                from ball_observation_contract import stamp_ball_detections
                pose_res, human_pose_ms = f2.result()
                join_started = time.perf_counter()
                legacy_rackets = list(raw_rackets or [])
                if racket_pose_runtime is not None and not stage_a:
                    racket_pose_observation = racket_pose_runtime.resolve(task['idx'], pose_submitted)
                    if pipeline.racket_pose_config['mode'] == 'active' and not pipeline.exclusive_racket_source:
                        from racket_pose_detector import select_pose_candidate
                        wrists = [
                            {'x': kp.x, 'y': kp.y, 'observed': kp.observed,
                             'source_frame_id': kp.source_frame_id, 'confidence': kp.conf}
                            for name, kp in pose_res.front_pose_orig.items() if 'wrist' in name
                        ]
                        selected_pose = select_pose_candidate(
                            racket_pose_observation, task['idx'], wrists,
                            (pipeline.width, pipeline.height), {**pipeline.racket_pose_config,'association_view':'front'},
                        )
                        if selected_pose is not None:
                            raw_rackets = [selected_pose] + list(raw_rackets or [])
                racket_join_ms = stage_a['racket_join_ms'] if stage_a else (time.perf_counter() - join_started) * 1000
                association_started = time.perf_counter()
                if pipeline.exclusive_racket_source:
                    from racket_source_policy import associated_pose_candidates
                    wrists_by_view = {
                        view: [{'x':kp.x,'y':kp.y,'observed':kp.observed,
                                'source_frame_id':kp.source_frame_id,'confidence':kp.conf}
                               for name,kp in points.items() if 'wrist' in name]
                        for view,points in (('front',pose_res.front_pose_orig),('back',pose_res.back_pose_orig))
                    }
                    raw_rackets = associated_pose_candidates(racket_pose_observation, task['idx'],
                        wrists_by_view, (pipeline.width,pipeline.height), pipeline.racket_pose_config)
                    ball = detector.finalize_external_ball_candidates(
                        ball, raw_rackets, confidence_threshold=pipeline.racket_pose_config['confidence_threshold'])
                stamp_ball_detections(ball, task['idx'])
                # 双重保障：排除镜面反射球与ROI范围外球，防止镜中虚像被锁定追踪
                if ball:
                    valid_balls = []
                    for b in ball:
                        b_pos = b.get("position")
                        if not b_pos:
                            continue
                        bx, by = float(b_pos[0]), float(b_pos[1])
                        if dual_view_mgr is not None and dual_view_mgr.is_point_in_mirror(bx, by, pipeline.width, pipeline.height):
                            continue
                        if roi_manager is not None and not roi_manager.is_ball_in_roi((bx, by)):
                            continue
                        valid_balls.append(b)
                    ball = valid_balls
                ball_diagnostics = detector.get_last_ball_diagnostics()
                racket_diagnostics = detector.get_last_racket_diagnostics()
                # Both futures already finished: associate existing observations without inference.
                if ball and ball[0].get("position") is not None:
                    target_pose = pose_res.fused_pose_local if (pose_res and pose_res.fused_pose_local) else (pose_res.front_pose_local if pose_res else {})
                    observed_front = {k: v for k, v in target_pose.items() if getattr(v, "observed", True)}
                    position = dual_frame.front_info.map_from_original(*ball[0]["position"])
                    pose_res.biomechanics.shot_classification = dual_pose_estimator.biomech_engine.classify_shot(
                        observed_front, ball_pos=position, allow_recovered=True
                    )
                current_player_bbox = None
                if dual_view_mgr is not None and hasattr(dual_view_mgr, "update_player_from_keypoints"):
                    dual_view_mgr.update_player_from_keypoints(pose_res.front_pose_orig or pose_res.fused_pose_orig)
                    current_player_bbox = dual_view_mgr.tracked_player_bbox

                # A~D. 调用 RacketResolver 完成镜中候选关联、手腕解剖门控、正面遮挡自愈与时序平滑
                racket_res = racket_resolver.resolve(
                    raw_rackets=raw_rackets,
                    pose_res=pose_res,
                    frame_idx=task['idx'],
                    source_time=task.get('source_time'),
                    frame_size=(pipeline.width, pipeline.height),
                    dual_view_mgr=dual_view_mgr,
                    racket_tracker=racket_tracker,
                    detector=detector,
                    base_diagnostics=racket_diagnostics,
                    ground_calibration=pipeline.ground_reference.calibration if pipeline.ground_reference else None,
                )
                racket = racket_res.rackets
                if pipeline.exclusive_racket_source:
                    from racket_source_policy import validate_exclusive_output
                    validate_exclusive_output(racket, task['idx'])
                racket_box = racket_res.racket_box
                back_racket_box = racket_res.back_racket_box
                is_racket_recovered = racket_res.is_racket_recovered
                racket_diagnostics = racket_res.diagnostics

                front_pose_dict = {
                    k: (int(round(kp.x)), int(round(kp.y)))
                    for k, kp in pose_res.front_pose_orig.items()
                    if kp.conf > 0.25
                }
                healed_pose_dict = {
                    k: (int(round(kp.x)), int(round(kp.y)))
                    for k, kp in pose_res.fused_pose_orig.items()
                    if kp.conf > 0.25
                }
                pose = [front_pose_dict] if front_pose_dict else []

                b = pose_res.biomechanics
                dual_view_biomech = {
                    "shoulder_turn": {
                        "shoulder_turn_deg": b.robust_shoulder_turn_deg,
                        "confidence": None,
                        "confidence_source": "unvalidated_proxy",
                    },
                    "takeback_depth": {
                        "takeback_depth_ratio": b.takeback_depth_ratio,
                        "confidence": None,
                        "confidence_source": "unvalidated_proxy",
                    },
                    "scapular_retraction": {
                        "scapular_retraction_ratio": b.scapular_retraction_ratio,
                        "confidence": None,
                        "confidence_source": "unvalidated_proxy",
                    },
                    "shot_classification": {
                        "stroke_type": b.shot_classification.shot_type,
                        "is_two_handed": b.shot_classification.is_two_handed,
                        "confidence": b.shot_classification.confidence,
                    },
                    "contact_distance_gate": {
                        "is_valid_contact": b.shot_classification.is_valid_contact,
                    },
                }
                inference_completed_at_unix_ns = time.time_ns()

                pipeline.q_analyzer.put({
                    'id': task['idx'],
                    'slot': slot,
                    'ball': ball,
                    'racket': racket,
                    'racket_box': racket_box,
                    'back_racket_box': back_racket_box,
                    'is_racket_recovered': is_racket_recovered,
                    'pose': pose,
                    'healed_pose': healed_pose_dict,
                    'dual_pose_res': pose_res,
                    'dual_view_biomechanics': dual_view_biomech,
                    'player_bbox': current_player_bbox,
                    'front_crop_bbox': dual_frame.front_info.bbox_orig if dual_frame is not None else None,
                    'pose_input_geometry': {
                        name: {
                            'source_crop': list(info.bbox_orig),
                            'crop_size': list(info.crop_size),
                            'pixels_size': list(image.shape[1::-1]),
                            'coordinate_size': list(info.view_size),
                            'model_input_size': list(model.input_size) if hasattr(model, 'input_size') else None,
                            'source_pixels': raw,
                        }
                        for name, info, image, model, raw in (
                            ('front', dual_frame.front_info, dual_frame.front_frame,
                             dual_pose_estimator.model_front, dual_frame.front_source_pixels),
                            ('back', dual_frame.back_info, dual_frame.back_frame,
                             dual_pose_estimator.model_back, dual_frame.back_source_pixels),
                        )
                    },
                    'ball_diagnostics': ball_diagnostics,
                    'racket_diagnostics': racket_diagnostics,
                    'racket_pose': racket_pose_observation,
                    'legacy_rackets': legacy_rackets,
                    'inference_schedule': pipeline.inference_schedule,
                    'inference_stage_ms': {'stage_a_ms':stage_a['stage_a_ms'] if stage_a else 0.,
                        'stage_b_ms':(time.perf_counter()-stage_b_started)*1000,
                        'legacy_detection_ms': legacy_detection_ms,
                        'human_pose_ms': human_pose_ms, 'racket_dispatch_ms': racket_dispatch_ms,
                        'racket_join_ms': racket_join_ms,
                        'association_ms': (time.perf_counter() - association_started) * 1000},
                    'reader_schedule_lag_ms': task.get('reader_schedule_lag_ms'),
                    'captured_at': task.get('captured_at'),
                    'captured_at_unix_ns': task.get('captured_at_unix_ns'),
                    'source_time': task.get('source_time'),
                    'inference_started_at_unix_ns': inference_started_at_unix_ns,
                    'inference_completed_at_unix_ns': inference_completed_at_unix_ns,
                })
            else:
                # 使用 ThreadPool 同时驱动检测和姿态模型。
                if stage_a:
                    from concurrent.futures import Future
                    f1 = Future()
                    candidates, adopted_rackets = detector.adopt_external_packet(
                        stage_a['packet'],
                        task['idx'],
                        return_rackets=True,
                        retain_racket_candidates=racket_tracker is not None,
                    )
                    f1.set_result(((candidates, adopted_rackets, 0.), stage_a['legacy_detection_ms']))
                else:
                    f1 = executor.submit(
                        timed_call,
                        detector.detect_unified,
                        detection_frame,
                        frame_context.roi_offset if frame_context is not None else (0, 0),
                        (pipeline.width, pipeline.height),
                        racket_source='external' if pipeline.exclusive_racket_source else 'legacy',
                    )
                f2 = executor.submit(timed_call, pose_estimator.get_keypoints, pose_detection_frame)

                (ball, racket, _), legacy_detection_ms = f1.result()
                from ball_observation_contract import stamp_ball_detections
                pose, human_pose_ms = f2.result()
                if frame_context is not None:
                    pose, ball, racket = frame_context.adjust_detections(
                        pose,
                        ball,
                        racket,
                        object_coordinates_are_full_frame=True,
                    )
                legacy_rackets = list(racket or [])
                join_started = time.perf_counter()
                if racket_pose_runtime is not None and not stage_a:
                    racket_pose_observation = racket_pose_runtime.resolve(task['idx'], pose_submitted)
                    if pipeline.racket_pose_config['mode'] == 'active' and not pipeline.exclusive_racket_source:
                        from racket_pose_detector import select_pose_candidate
                        wrists = [value for person in (pose or []) for name, value in person.items() if 'wrist' in name]
                        selected_pose = select_pose_candidate(
                            racket_pose_observation, task['idx'], wrists,
                            (pipeline.width, pipeline.height), pipeline.racket_pose_config,
                        )
                        if selected_pose is not None:
                            racket = [selected_pose] + list(racket or [])
                if pipeline.exclusive_racket_source:
                    from racket_source_policy import current_pose_candidates
                    from racket_pose_detector import select_pose_candidate
                    fresh = current_pose_candidates(racket_pose_observation, task['idx'])
                    wrists = [value for person in (pose or []) for name, value in person.items() if 'wrist' in name]
                    selected_pose = select_pose_candidate(
                        {**racket_pose_observation, 'candidates': fresh}, task['idx'], wrists,
                        (pipeline.width, pipeline.height), pipeline.racket_pose_config)
                    racket = [selected_pose] if selected_pose is not None else []
                    ball = detector.finalize_external_ball_candidates(
                        ball, racket, confidence_threshold=pipeline.racket_pose_config['confidence_threshold'])
                stamp_ball_detections(ball, task['idx'])
                ball_diagnostics = detector.get_last_ball_diagnostics()
                racket_diagnostics = detector.get_last_racket_diagnostics()
                if pipeline.exclusive_racket_source:
                    from racket_source_policy import validate_exclusive_output
                    validate_exclusive_output(racket, task['idx'])
                inference_completed_at_unix_ns = time.time_ns()

                pipeline.q_analyzer.put({
                    'id': task['idx'],
                    'slot': slot,
                    'ball': ball,
                    'racket': racket,
                    'racket_box': racket[0]['box'] if racket and racket[0].get('box') else None,
                    'pose': pose,
                    'ball_diagnostics': ball_diagnostics,
                    'racket_diagnostics': racket_diagnostics,
                    'racket_pose': racket_pose_observation,
                    'legacy_rackets': legacy_rackets,
                    'inference_schedule': pipeline.inference_schedule,
                    'inference_stage_ms': {'stage_a_ms':stage_a['stage_a_ms'] if stage_a else 0.,
                        'stage_b_ms':(time.perf_counter()-stage_b_started)*1000,
                        'legacy_detection_ms': legacy_detection_ms,
                        'human_pose_ms': human_pose_ms, 'racket_dispatch_ms': racket_dispatch_ms,
                        'racket_join_ms': stage_a['racket_join_ms'] if stage_a else (time.perf_counter() - join_started) * 1000},
                    'reader_schedule_lag_ms': task.get('reader_schedule_lag_ms'),
                    'captured_at': task.get('captured_at'),
                    'captured_at_unix_ns': task.get('captured_at_unix_ns'),
                    'source_time': task.get('source_time'),
                    'inference_started_at_unix_ns': inference_started_at_unix_ns,
                    'inference_completed_at_unix_ns': inference_completed_at_unix_ns,
                })
    except Exception as exc:
        print(f"❌ [Inference] 初始化或推理失败: {exc}")
        pipeline.stop_event.set()
        pipeline.inf_ready.set()
        raise
    finally:
        preparation_error = None
        if interframe is not None:
            try:
                interframe.close()
            except Exception as exc:
                preparation_error = exc
        if racket_pose_runtime is not None:
            racket_pose_runtime.close()
        if executor is not None:
            executor.shutdown()
        if dual_pose_estimator is not None and hasattr(dual_pose_estimator, "close"):
            try:
                dual_pose_estimator.close()
            except Exception:
                pass
        # Do not invalidate a borrowed ndarray in a stuck native preparation
        # call. The failed process exits; its owner unlinks after supervision.
        if interframe is None or not interframe.thread.is_alive():
            for s in shms:
                s.close()
        pipeline.inf_done.set()
        print("✅ [Inference] 退出")
        if preparation_error is not None:
            raise preparation_error

