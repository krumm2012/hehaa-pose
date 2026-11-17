"""Analyzer and rendering worker process for MultiprocessPipeline."""

from __future__ import annotations

import json
import os
import queue
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from analysis_data_contracts import document_contract, stamp_frame_record
from video_overlay_primitives import draw_ball_outline


def run_analyzer_process(pipeline, ignore_child_interrupts_fn=None):
    """进程 3: 业务核心 + 全视觉渲染"""
    (ignore_child_interrupts_fn() if ignore_child_interrupts_fn else None)
    print("🚀 [Analyzer] 初始化高清渲染引擎...")
    from frame_processor import FrameProcessor
    from performance_metrics import FpsTracker, timed_call
    from pose_renderer import draw_pose_keypoints
    from video_writer_backend import create_video_writer

    perf_cfg = pipeline.config.get('pipeline_perf', {})
    dual_view_mgr = None
    dual_view_renderer = None
    sbs_width = pipeline.width
    sbs_height = pipeline.height
    ball_trail = []
    from joint_racket_2d_live import LiveJointRacket2D
    joint_2d_live = LiveJointRacket2D()

    if pipeline.algo2_dual_view:
        from dual_view_manager import DualViewManager
        from dual_view_renderer import DualViewRenderer

        dual_view_mgr = DualViewManager(
            config_path=pipeline.algo2_config_path,
            stream_profile=pipeline.roi_profile,
        )
        dual_view_renderer = DualViewRenderer(show_hud=True, show_skeleton=True)
        fw = dual_view_mgr.front_target_size[0] if dual_view_mgr.front_target_size else 540
        bw = dual_view_mgr.mirror_target_size[0] if dual_view_mgr.mirror_target_size else 540
        fh = dual_view_mgr.front_target_size[1] if dual_view_mgr.front_target_size else 720
        sbs_width = fw + bw
        sbs_height = fh
        print(f"🚀 [Analyzer] 算法 2.0 虚拟双机位 Side-by-Side 实时渲染引擎就绪: {sbs_width}x{sbs_height}")

    # 视频录制
    save_video = pipeline.config.get('save_video', True)
    out_writer = None
    out_file = None
    if save_video:
        out_file = create_output_directory(pipeline.config['video_output_path'])
        writer_w = sbs_width if pipeline.algo2_dual_view else pipeline.width
        writer_h = sbs_height if pipeline.algo2_dual_view else pipeline.height
        out_writer = create_video_writer(
            output_path=out_file,
            width=writer_w,
            height=writer_h,
            fps=pipeline.output_fps,
            backend=perf_cfg.get('video_encoder_backend', 'auto'),
            bitrate=perf_cfg.get('video_encoder_bitrate', '12M'),
        )
        print(
            f"🎬 [Recorder] 录制中: {out_file}"
            f" | 编码器: {out_writer.backend_name}"
        )
        if pipeline.output_fps != pipeline.fps:
            print(f"🎬 [Recorder] 输出帧率: {pipeline.output_fps:.2f} FPS")

    shms = [shared_memory.SharedMemory(name=name) for name in pipeline.shm_names]
    shared_frames = [np.ndarray((pipeline.height, pipeline.width, 3), dtype=np.uint8, buffer=s.buf) for s in shms]

    last_ball_pos = None
    last_frame_id = -1
    count = 0
    fps_tracker = FpsTracker()
    # 性能开关（默认兼容原行为）
    dual_view_enabled = bool(perf_cfg.get('dual_view_enabled', True))
    dual_view_scale = float(perf_cfg.get('dual_view_scale', 0.5))
    hdmi_output_enabled = bool(perf_cfg.get('hdmi_output_enabled', False))
    display_origin = perf_cfg.get('display_origin') or [0, 0]
    processed_display = None
    if hdmi_output_enabled:
        from processed_video_display import ProcessedVideoDisplay

        processed_display = ProcessedVideoDisplay(
            cv2_module=cv2,
            origin=(int(display_origin[0]), int(display_origin[1])),
            fullscreen=True,
        )
        print(
            '📺 [HDMI] 处理后画面全屏输出已开启'
            f' | origin: ({int(display_origin[0])}, {int(display_origin[1])})'
            ' | ESC/q 退出'
        )
    queued_display = None
    if perf_cfg.get('async_display_enabled', False) and (hdmi_output_enabled or pipeline.algo2_dual_view or dual_view_enabled):
        from queued_video_display import QueuedVideoDisplay
        if hdmi_output_enabled or pipeline.algo2_dual_view:
            display_shape = (sbs_height, sbs_width, 3)
        else:
            display_shape = (int(pipeline.height*dual_view_scale), int(pipeline.width*dual_view_scale)*2, 3)
        queued_display = QueuedVideoDisplay(display_shape, pipeline.stop_event,
            origin=tuple(display_origin), fullscreen=hdmi_output_enabled,
            window_name='Tennis AI - HDMI Output' if hdmi_output_enabled else ('Tennis AI Analyzer - Algo 2.0 Dual View (ESC to Quit)' if pipeline.algo2_dual_view else 'Tennis AI Analyer - Dual Comparison (ESC to Quit)'),
            journal_path=str(Path(pipeline.config['video_output_path']).with_suffix(''))+'_display.jsonl',
            preserve_frames=True)
    analysis_stride = max(1, int(perf_cfg.get('analysis_stride', 1)))
    draw_metrics_text = bool(perf_cfg.get('draw_metrics_text', True))
    collect_frame_results = bool(perf_cfg.get('collect_frame_results', True))
    json_dump_indent = perf_cfg.get('json_dump_indent', 4)
    frame_results_path = None
    diagnostics_path = None
    frame_results_fp = None
    diagnostics_records = []
    first_frame_record = True
    output_base = out_file if out_file else create_output_directory(
        pipeline.config['video_output_path']
    )
    if collect_frame_results:
        frame_results_path = os.path.splitext(output_base)[0] + '.json'
        diagnostics_path = os.path.splitext(output_base)[0] + '_diagnostics.json'
        frame_results_fp = open(frame_results_path, 'w', encoding='utf-8')
        frame_results_fp.write('{\n')
        frame_contract = document_contract("frames", pipeline.session_metadata)
        frame_results_fp.write(
            f'  "schema_version": {json.dumps(frame_contract["schema_version"])},\n'
        )
        frame_results_fp.write(
            f'  "document_type": {json.dumps(frame_contract["document_type"])},\n'
        )
        frame_results_fp.write(
            f'  "session": {json.dumps(pipeline.session_metadata, ensure_ascii=False)},\n'
        )
        frame_results_fp.write('  "video_info": {\n')
        frame_results_fp.write(f'    "path": {json.dumps(pipeline.video_path, ensure_ascii=False)},\n')
        frame_results_fp.write(f'    "fps": {json.dumps(pipeline.fps)},\n')
        frame_results_fp.write(f'    "resolution": {json.dumps([pipeline.width, pipeline.height])}\n')
        frame_results_fp.write('  },\n')
        frame_results_fp.write('  "frames": [\n')
    frame_processor = FrameProcessor(
        config=pipeline.config,
        frame_dimensions=(pipeline.height, pipeline.width),
        fps=pipeline.fps,
        analysis_stride=analysis_stride,
    )
    output_stem = os.path.splitext(output_base)[0]
    realtime_engine = None
    realtime_output = None
    frame_journal = None
    realtime_coach = None
    deepseek_sidecar = None
    coach_tts_sidecar = None
    realtime_runtime = None
    if pipeline.realtime_frame_output:
        from realtime_swing_pipeline import RealtimeFrameJournal

        realtime_frame_jsonl = (
            pipeline.realtime_frame_jsonl or f'{output_stem}_frames.jsonl'
        )
        realtime_frame_snapshot_json = (
            pipeline.realtime_frame_snapshot_json
            or f'{output_stem}_frames_latest.json'
        )
        frame_journal = RealtimeFrameJournal(
            jsonl_path=realtime_frame_jsonl,
            snapshot_path=realtime_frame_snapshot_json,
            snapshot_size=pipeline.realtime_frame_snapshot_size,
            flush_interval=pipeline.realtime_frame_flush_interval,
            session_metadata=pipeline.session_metadata,
        )
        print(
            f'📝 [Frame-Live] 实时逐帧输出已开启'
            f' | JSONL: {realtime_frame_jsonl}'
            f' | 快照: {realtime_frame_snapshot_json}'
        )
    if pipeline.realtime_coach:
        from local_realtime_coach import LocalRealtimeCoach

        realtime_coach = LocalRealtimeCoach(
            max_chars=pipeline.realtime_coach_max_chars,
            max_suggestions=pipeline.realtime_coach_max_suggestions,
            min_confidence=pipeline.realtime_coach_min_confidence,
            thresholds=pipeline.realtime_coach_biomechanics,
        )
    if pipeline.deepseek_coach_options['enabled']:
        from deepseek_realtime_coach import DeepSeekCoachSidecar

        api_key_env = pipeline.deepseek_coach_options['api_key_env']
        deepseek_sidecar = DeepSeekCoachSidecar(
            api_key=os.environ.get(api_key_env, ''),
            model=pipeline.deepseek_coach_options['model'],
            base_url=pipeline.deepseek_coach_options['base_url'],
            timeout_seconds=pipeline.deepseek_coach_options['timeout_seconds'],
            workers=pipeline.deepseek_coach_options['workers'],
            max_pending=pipeline.deepseek_coach_options['max_pending'],
            max_chars=pipeline.deepseek_coach_options['max_chars'],
        )
        key_status = '已配置' if os.environ.get(api_key_env) else '未配置'
        print(
            f'🧠 [DeepSeek] 旁路Coach已开启'
            f' | 模型: {pipeline.deepseek_coach_options["model"]}'
            f' | 超时: {pipeline.deepseek_coach_options["timeout_seconds"]:.1f}s'
            f' | {api_key_env}: {key_status}'
        )
    if pipeline.coach_tts_options['enabled']:
        from qwen3_tts_sidecar import CoachTtsSidecar

        coach_tts_dir = f'{output_stem}_coach_audio'
        coach_tts_sidecar = CoachTtsSidecar(
            output_dir=coach_tts_dir,
            model=pipeline.coach_tts_options['model'],
            voice=pipeline.coach_tts_options['voice'],
            language=pipeline.coach_tts_options['language'],
            playback=pipeline.coach_tts_options['playback'],
            max_pending=pipeline.coach_tts_options['max_pending'],
            concise_speech=pipeline.coach_tts_options['concise_speech'],
            warmup=pipeline.coach_tts_options['warmup'],
            startup_timeout_seconds=pipeline.coach_tts_options['startup_timeout_seconds'],
            request_timeout_seconds=pipeline.coach_tts_options['request_timeout_seconds'],
            streaming_interval_seconds=(
                pipeline.coach_tts_options['streaming_interval_seconds']
            ),
            streaming_prebuffer_chunks=(
                pipeline.coach_tts_options['streaming_prebuffer_chunks']
            ),
            audio_path_prefix=Path(coach_tts_dir).name,
        )
        print(
            f'🔊 [Qwen3-TTS] 本地语音播报已开启'
            f' | 模型: {pipeline.coach_tts_options["model"]}'
            f' | 声音: {pipeline.coach_tts_options["voice"]}'
            f' | 语态: {"极简短口令" if pipeline.coach_tts_options["concise_speech"] else "完整句式"}'
            f' | 预热: {"后台自动预热" if pipeline.coach_tts_options["warmup"] else "懒加载"}'
            f' | 扬声器: {"开启" if pipeline.coach_tts_options["playback"] else "关闭"}'
        )
    if pipeline.realtime_swing_events:
        from realtime_swing_pipeline import (
            RealtimeSwingEventEngine,
            RealtimeSwingOutputManager,
        )

        realtime_json = pipeline.realtime_swing_json or f'{output_stem}_swing_events.json'
        realtime_html = pipeline.realtime_swing_html or f'{output_stem}_swing_report.html'
        realtime_clips_dir = pipeline.realtime_swing_clips_dir or f'{output_stem}_swing_clips'
        effective_window_frames = (
            pipeline.realtime_window_frames
            if pipeline.realtime_window_frames is not None
            else max(32, int(round(REFERENCE_HZ * 8.0)))
        )
        effective_settle_frames = (
            pipeline.realtime_settle_frames
            if pipeline.realtime_settle_frames is not None
            else max(0, int(round(REFERENCE_HZ * 0.6)))
        )
        realtime_engine = RealtimeSwingEventEngine(
            fps=pipeline.fps,
            analysis_interval_frames=pipeline.realtime_analysis_interval,
            settle_frames=effective_settle_frames,
            window_frames=effective_window_frames,
            coach=realtime_coach,
            session_metadata=pipeline.session_metadata,
            **pipeline.swing_analysis_options,
        )
        effective_output_size = (
            (sbs_width, sbs_height) if pipeline.algo2_dual_view else (pipeline.width, pipeline.height)
        )
        realtime_output = RealtimeSwingOutputManager(
            output_json=realtime_json,
            output_html=realtime_html,
            clips_dir=realtime_clips_dir,
            fps=pipeline.output_fps,
            frame_size=effective_output_size,
            buffer_frames=(
                effective_window_frames
                + effective_settle_frames
                + pipeline.realtime_clip_padding_frames * 2
            ),
            clip_workers=pipeline.realtime_clip_workers,
            video_backend=perf_cfg.get('video_encoder_backend', 'auto'),
            video_bitrate=perf_cfg.get('video_encoder_bitrate', '12M'),
            clip_padding_frames=pipeline.realtime_clip_padding_frames,
            clip_max_width=pipeline.realtime_clip_max_width,
            jpeg_quality=pipeline.realtime_clip_jpeg_quality,
            preview_path=(
                f"{os.path.splitext(realtime_html)[0]}_roi_preview.jpg"
                if pipeline.roi_profile.enabled
                else None
            ),
            roi_metadata=pipeline.roi_profile.as_metadata(),
            preview_interval_frames=max(
                1,
                int(
                    pipeline.config.get("roi_settings", {}).get(
                        "preview_interval_frames",
                        round(pipeline.output_fps),
                    )
                ),
            ),
            event_log_path=pipeline.realtime_swing_event_log,
            session_metadata=pipeline.session_metadata,
            is_dual_view=pipeline.algo2_dual_view,
        )
        print(
            f'⚡ [Swing-Live] 实时事件分析已开启'
            f' | JSON: {realtime_json}'
            f' | HTML: {realtime_html}'
            f' | 异步片段线程: {pipeline.realtime_clip_workers}'
            f' | 本地Coach: {"开启" if realtime_coach is not None else "关闭"}'
            f' | DeepSeek旁路: {"开启" if deepseek_sidecar is not None else "关闭"}'
            f' | Qwen3-TTS: {"开启" if coach_tts_sidecar is not None else "关闭"}'
        )
        if pipeline.realtime_open_report:
            import webbrowser

            webbrowser.open(Path(realtime_html).resolve().as_uri())

    if realtime_engine is not None or frame_journal is not None:
        from realtime_swing_runtime import RealtimeSwingRuntime

        realtime_runtime = RealtimeSwingRuntime(
            engine=realtime_engine,
            output=realtime_output,
            frame_journal=frame_journal,
            deepseek_sidecar=deepseek_sidecar,
            coach_tts_sidecar=coach_tts_sidecar,
            stop_event=pipeline.stop_event,
            queue_size=max(64, int(round(pipeline.fps * 12))),
        )

    if dual_view_renderer is not None:
        dual_view_renderer.font_mgr.warmup_fonts()
    pipeline.analyzer_ready.set()
    while not pipeline.stop_event.is_set():
        try:
            data = pipeline.q_analyzer.get(timeout=2.0)
        except queue.Empty:
            if pipeline.inf_done.is_set():
                break
            continue
        except Exception as exc:
            print(f"❌ [Analyzer] 获取分析任务失败: {exc}")
            pipeline.stop_event.set()
            raise RuntimeError("Analyzer queue read failed") from exc

        fid, slot = data['id'], data['slot']
        if count == 0:
            fps_tracker.start()
        # 严格对齐图像
        canvas = shared_frames[slot].copy()

        ball_pos = data['ball'][0]['position'] if data['ball'] else None
        racket_list = data['racket']
        poses = data['pose']

        # --- 1. 深度分析计算 ---
        analysis_started_at_unix_ns = time.time_ns()
        frame_analysis = frame_processor.process(
            frame_id=fid,
            poses=poses,
            racket_detections=racket_list,
            ball_position=(ball_pos[0], ball_pos[1]) if ball_pos else None,
        )
        swing_label = frame_analysis["swing_type"]
        detailed_data = frame_analysis["phase_metrics"]
        norm_ball_pos = frame_analysis["ball_position"]

        healed_pose = data.get('healed_pose')
        dv_biomech = data.get('dual_view_biomechanics')
        racket_box = data.get('racket_box')
        back_racket_box = data.get('back_racket_box')
        is_racket_recovered = data.get('is_racket_recovered', False)
        pose_res = data.get('dual_pose_res')

        # --- 保存每帧数据 ---
        frame_record = None
        if (
            collect_frame_results
            or realtime_engine is not None
            or frame_journal is not None
        ):
            frame_record = frame_processor.build_frame_record(
                frame_id=fid,
                swing_type=swing_label,
                ball_position=norm_ball_pos,
                racket_detections=racket_list,
                poses=poses,
                phase_metrics=detailed_data,
                healed_pose=healed_pose,
                dual_view_biomechanics=dv_biomech,
                racket=racket_box,
                source_time=data.get("source_time"),
                ball_detection=data['ball'][0] if data['ball'] else None,
            )
            if pose_res is not None:
                from kinematic_sequence import serialize_kinematic_views
                frame_record["kinematic_views"] = serialize_kinematic_views(pose_res)
                frame_record["pose_observations"] = serialize_kinematic_views(pose_res, all_joints=True)
                frame_record["pose_input_geometry"] = data.get('pose_input_geometry')
                frame_record['pose_observation_coordinate_space'] = 'original_source_pixels'
            if pipeline.ground_reference is not None:
                frame_record['ground_reference'] = pipeline.ground_reference.observe(frame_record)
                frame_record['ground_calibration'] = pipeline.ground_reference.calibration
            frame_record["detection_diagnostics"] = data.get("ball_diagnostics") or {}
            frame_record["racket_detection_diagnostics"] = (
                data.get("racket_diagnostics") or {}
            )
            frame_record['reader_schedule_lag_ms'] = data.get('reader_schedule_lag_ms')
            frame_record['inference_stage_ms'] = data.get('inference_stage_ms', {})
            frame_record['inference_schedule'] = data.get('inference_schedule','barrier')
            if data.get('racket_pose') is not None:
                frame_record['racket_pose'] = data['racket_pose']
                frame_record['racket_selection_policy'] = pipeline.racket_pose_config.get('selection_policy','mixed')
                frame_record['racket_selection_policy_version'] = pipeline.racket_pose_config.get('selection_policy_version')
                frame_record['legacy_rackets'] = data.get('legacy_rackets', [])
                frame_record['racket_selection_source'] = (
                    racket_list[0].get('source', 'legacy') if racket_list else 'missing'
                )
                if (racket_list and frame_record['racket_selection_source']=='racket_pose_model'
                        and (racket_list[0].get('observed') is not True
                             or racket_list[0].get('source_frame_id')!=fid)):
                    frame_record['racket_selection_source']='racket_pose_model_derived'
            analysis_completed_at_unix_ns = time.time_ns()
            stamp_frame_record(
                frame_record,
                session=pipeline.session_metadata,
                captured_at_unix_ns=(
                    data.get("captured_at_unix_ns")
                    or analysis_started_at_unix_ns
                ),
                inference_started_at_unix_ns=(
                    data.get("inference_started_at_unix_ns")
                    or analysis_started_at_unix_ns
                ),
                inference_completed_at_unix_ns=(
                    data.get("inference_completed_at_unix_ns")
                    or analysis_started_at_unix_ns
                ),
                analysis_started_at_unix_ns=analysis_started_at_unix_ns,
                analysis_completed_at_unix_ns=analysis_completed_at_unix_ns,
            )
        # --- 2. 视觉渲染流程 ---
        render_stages = {}
        render_features = {}
        overlay_started = time.perf_counter_ns()
        if data.get('racket_pose') and pipeline.racket_pose_config.get('render_skeleton', True):
            from racket_pose_detector import draw_racket_pose
            draw_racket_pose(canvas, data['racket_pose'])
        joint_diagnostic = None
        if (frame_record is not None and data.get('racket_pose') is not None
                and pipeline.racket_pose_config.get('render_joint_2d', True)):
            joint_diagnostic = joint_2d_live.update(frame_record, canvas.shape[1::-1])
            frame_record['joint_racket_2d_live'] = joint_diagnostic
            joint_2d_live.draw_source(canvas)
        render_stages['racket_overlay_wall_ms'] = (time.perf_counter_ns() - overlay_started) / 1e6
        if pipeline.algo2_dual_view and dual_view_mgr is not None and dual_view_renderer is not None and pose_res is not None:
            if norm_ball_pos:
                ball_trail.append(norm_ball_pos)
            if len(ball_trail) > 8:
                ball_trail.pop(0)

            disp = realtime_runtime.get_active_display_state(fid) if realtime_runtime is not None else {}
            ev_label = disp.get("event_label") or swing_label or "READY STANCE"
            ev_coach = disp.get("coaching_text") or ""
            telemetry_card = disp.get("telemetry_card")
            render_features = {'coach': bool(ev_coach), 'telemetry_card': telemetry_card is not None}

            split_started = time.perf_counter_ns()
            dual_frame = dual_view_mgr.split_frame(
                canvas,
                player_bbox=data.get('player_bbox'),
                frame_id=fid,
                front_crop_bbox=data.get('front_crop_bbox'),
            )
            render_stages['split_wall_ms'] = (time.perf_counter_ns() - split_started) / 1e6
            rendered_canvas = dual_view_renderer.render_dual_frame(
                dual_frame,
                pose_res,
                event_label=ev_label,
                coaching_text=ev_coach,
                ball_trail=list(ball_trail),
                racket_box=racket_box,
                back_racket_box=back_racket_box,
                is_racket_recovered=is_racket_recovered,
                telemetry_card=telemetry_card,
                ground_reference=frame_record.get('ground_reference') if frame_record else None,
                ground_geometry=pipeline.ground_reference.overlay_geometry if pipeline.ground_reference else None,
                timing_ms=render_stages,
            )
        else:
            # 背景半透明面板
            mask = canvas.copy()
            cv2.rectangle(mask, (20, 20), (420, 620), (0, 0, 0), -1)
            cv2.addWeighted(mask, 0.75, canvas, 0.25, 0, canvas)

            # A. 核心文本显示 (English Only)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, f"Action: {swing_label}", (40, 65), font, 0.8, (0, 255, 255), 2)
            cv2.putText(canvas, f"Frame: {fid:04d}", (40, 100), font, 0.5, (200, 200, 200), 1)

            # B. 绘制详细肢体指标
            if draw_metrics_text:
                y_ptr = 140
                for phase, metrics in detailed_data.items():
                    for k, v in metrics.items():
                        color = (0, 255, 0) if "angle" in k or "ext" in k else (220, 220, 220)
                        cv2.putText(canvas, f"{k}: {v}", (40, y_ptr), font, 0.5, color, 1)
                        y_ptr += 26
                        if y_ptr > 600:
                            break

            # C. 绘制视觉元素 (骨架、球、球拍)
            canvas = draw_pose_keypoints(canvas, poses)
            if norm_ball_pos:
                draw_ball_outline(canvas, norm_ball_pos)

            for ra in racket_list:
                x1, y1, x2, y2 = [int(round(v)) for v in ra['box']]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 128, 0), 2)
            rendered_canvas = canvas

        if joint_diagnostic is not None:
            joint_2d_live.draw_status(rendered_canvas, joint_diagnostic)
            render_features['joint_racket_2d_live'] = True
        # --- 3. 提交并释放 ---
        render_completed_at_unix_ns = time.time_ns()
        output_stages = {'encoder_write_ms': 0.0, 'display_submit_ms': 0.0, 'display_wait_ms': 0.0}
        if out_writer:
            _, output_stages['encoder_write_ms'] = timed_call(out_writer.write, rendered_canvas)
        display_started = time.perf_counter()

        # HDMI 单路输出、双视角全屏输出与双窗口对比显示
        if processed_display is not None:
            if (queued_display.show(rendered_canvas, fid) if queued_display else processed_display.show(rendered_canvas)):
                pipeline.stop_event.set()
        elif pipeline.algo2_dual_view:
            if queued_display:
                key = 27 if queued_display.show(rendered_canvas, fid) else -1
            else:
                cv2.imshow("Tennis AI Analyzer - Algo 2.0 Dual View (ESC to Quit)", rendered_canvas)
                key, output_stages['display_wait_ms'] = timed_call(cv2.waitKey, 1)
                key &= 0xFF
            if key == 27 or key == ord('q'):
                pipeline.stop_event.set()
        elif dual_view_enabled:
            font = cv2.FONT_HERSHEY_SIMPLEX
            h, w = canvas.shape[:2]
            scaled_size = (int(w * dual_view_scale), int(h * dual_view_scale))
            orig_small = cv2.resize(shared_frames[slot], scaled_size)
            proc_small = cv2.resize(canvas, scaled_size)

            dual_view = np.hstack((orig_small, proc_small))
            cv2.putText(dual_view, "ORIGINAL", (20, 30), font, 1.0, (255, 255, 255), 2)
            cv2.putText(
                dual_view,
                "PROCESSED",
                (int(w * dual_view_scale) + 20, 30),
                font,
                1.0,
                (255, 255, 255),
                2
            )

            if queued_display:
                key = 27 if queued_display.show(dual_view, fid) else -1
            else:
                cv2.imshow("Tennis AI Analyer - Dual Comparison (ESC to Quit)", dual_view)
                key, output_stages['display_wait_ms'] = timed_call(cv2.waitKey, 1)
                key &= 0xFF
            if key == 27:
                pipeline.stop_event.set()

        output_stages['display_submit_ms'] = (time.perf_counter() - display_started) * 1000
        if frame_record is not None:
            frame_record['render_stage_ms'] = render_stages
            frame_record['render_features'] = render_features
            frame_record['output_stage_ms'] = output_stages
            frame_record['display_submission_scope'] = 'bounded_queue_preserve_all' if queued_display else 'synchronous_highgui_return'
            from analysis_data_contracts import stamp_render_output
            stamp_render_output(frame_record, render_completed_at_unix_ns, time.time_ns())
            queue_depths = {}
            for name, stage_queue in (('inference', pipeline.q_inference), ('analyzer', pipeline.q_analyzer)):
                try: queue_depths[name] = stage_queue.qsize()
                except (NotImplementedError, AttributeError): queue_depths[name] = None
            frame_record['pipeline_queue_depth'] = {'values':queue_depths,
                'null_reason':'queue depth unavailable on backend; stage waiting times are recorded separately'}
        if realtime_runtime is not None and frame_record is not None:
            realtime_runtime.submit_frame(frame_record)
            realtime_runtime.record_rendered_frame(fid, rendered_canvas)
        if collect_frame_results and frame_record is not None:
            if frame_results_fp is not None:
                if not first_frame_record:
                    frame_results_fp.write(',\n')
                frame_results_fp.write(json.dumps(frame_record, ensure_ascii=False))
                first_frame_record = False
            diagnostics_records.append({
                "frame_id": fid,
                "timestamp": round(fid / pipeline.fps, 3) if pipeline.fps else 0.0,
                "ball_selected": [norm_ball_pos[0], norm_ball_pos[1]] if norm_ball_pos else None,
                "ball_diagnostics": data.get("ball_diagnostics") or {},
            })

        pipeline.q_free.put(slot) # 释放回池子

        snapshot = fps_tracker.tick()
        count = snapshot.frame_count
        if count % 25 == 0:
            print(
                f"📊 [Sync-Analyzer] Processing Frame {fid}"
                f" | FPS: {snapshot.cumulative_fps:.2f}"
                f" | 25F: {snapshot.window_25_fps:.2f}"
                f" | 100F: {snapshot.window_100_fps:.2f}"
            )

    if realtime_runtime is not None:
        try:
            realtime_runtime.close()
        except Exception as exc:
            print(f"❌ [Swing-Live] 实时数据运行时关闭失败: {exc}")
            pipeline.stop_event.set()
            raise
        if realtime_engine is not None:
            print(
                f"✅ [Swing-Live] 实时分析结束"
                f" | events: {realtime_runtime.snapshot()['summary']['swing_event_count']}"
            )
        if frame_journal is not None:
            print("✅ [Frame-Live] 实时逐帧输出结束")

    if queued_display:
        queued_display.close()
    if out_writer:
        out_writer.release()
    if processed_display is not None:
        processed_display.close()

    if collect_frame_results and frame_results_fp is not None:
        frame_results_fp.write('\n  ],\n')
        frame_results_fp.write(
            f'  "summary": {json.dumps({"total_frames": count, "session_id": pipeline.session_id}, ensure_ascii=False, indent=json_dump_indent)}\n'
        )
        frame_results_fp.write('}\n')
        frame_results_fp.close()
        print(f"📊 [Analyzer] 数据已保存至: {frame_results_path}")
        if pipeline.analyze_swings:
            # Keep event segmentation outside the frame loop so pipeline throughput and
            # live rendering remain independent from the offline event-analysis cost.
            from swing_event_analyzer import (
                analyze_frame_records,
                default_output_paths,
                load_frame_records,
                write_analysis_outputs,
            )

            swing_paths = default_output_paths(frame_results_path)
            swing_output_json = pipeline.swing_output_json or swing_paths['json']
            swing_events_csv = pipeline.swing_events_csv or swing_paths['events_csv']
            swing_frames_csv = pipeline.swing_frames_csv or swing_paths['frames_csv']
            print('🏌️ [Swing] 开始事件级挥拍分段分析...')
            swing_analysis = analyze_frame_records(
                load_frame_records(frame_results_path),
                session_metadata=pipeline.session_metadata,
                **pipeline.swing_analysis_options,
            )
            write_analysis_outputs(
                swing_analysis,
                swing_output_json,
                swing_events_csv,
                swing_frames_csv,
            )
            swing_summary = swing_analysis['summary']
            print(
                '🏌️ [Swing] 事件分析完成: '
                f"{swing_summary['swing_event_count']} events "
                f"{swing_summary['swing_event_type_counts']}"
            )
            print(f'🏌️ [Swing] 事件 JSON: {swing_output_json}')
        if diagnostics_path:
            tuning_suggestions = []
            total_diag_frames = len(diagnostics_records)
            decisions = {}
            rejection_counts = {
                "static_hard_mask": 0,
                "low_conf_unsupported": 0,
                "upper_mirror_unsupported": 0,
                "track_became_static": 0,
            }
            continuity_disabled = 0
            selected_none = 0
            for rec in diagnostics_records:
                diag = rec.get("ball_diagnostics") or {}
                decision = diag.get("final_decision", "unknown")
                decisions[decision] = decisions.get(decision, 0) + 1
                rej = diag.get("rejections") or {}
                for k in rejection_counts:
                    rejection_counts[k] += int(rej.get(k, 0) or 0)
                if not bool(diag.get("continuity_enabled", True)):
                    continuity_disabled += 1
                if rec.get("ball_selected") is None:
                    selected_none += 1

            no_ball_ratio = (selected_none / total_diag_frames) if total_diag_frames else 0.0
            cfg_u = pipeline.config.get("unified_detection", {})
            if rejection_counts["upper_mirror_unsupported"] > 0:
                cur = float(cfg_u.get("ball_play_area_min_y_ratio_hard", 0.35))
                if no_ball_ratio <= 0.40:
                    nxt = min(0.50, round(cur + 0.03, 3))
                    tuning_suggestions.append({
                        "priority": "high",
                        "issue": "Upper mirror interference detected",
                        "recommendation": f"Increase ball_play_area_min_y_ratio_hard from {cur} to {nxt}",
                        "parameter": "unified_detection.ball_play_area_min_y_ratio_hard",
                    })
                else:
                    nxt = max(0.20, round(cur - 0.02, 3))
                    tuning_suggestions.append({
                        "priority": "medium",
                        "issue": "Upper mirror interference exists but no-ball ratio is already high",
                        "recommendation": f"Keep mirror gate conservative for now; if recall is poor, try lowering ball_play_area_min_y_ratio_hard from {cur} to {nxt}",
                        "parameter": "unified_detection.ball_play_area_min_y_ratio_hard",
                    })
            if rejection_counts["low_conf_unsupported"] > max(10, total_diag_frames * 0.08):
                cur = float(cfg_u.get("ball_min_selected_confidence", 0.05))
                nxt = max(0.03, round(cur - 0.01, 3))
                tuning_suggestions.append({
                    "priority": "medium",
                    "issue": "Many low-confidence unsupported rejections",
                    "recommendation": f"Try lowering ball_min_selected_confidence from {cur} to {nxt} to reduce no-ball gaps",
                    "parameter": "unified_detection.ball_min_selected_confidence",
                })
            if rejection_counts["static_hard_mask"] > max(20, total_diag_frames * 0.15) and no_ball_ratio > 0.20:
                cur = int(cfg_u.get("static_ball_hard_mask_min_seen_frames", 4))
                nxt = min(8, cur + 1)
                tuning_suggestions.append({
                    "priority": "medium",
                    "issue": "Static hard mask may be too aggressive in this clip",
                    "recommendation": f"Increase static_ball_hard_mask_min_seen_frames from {cur} to {nxt}",
                    "parameter": "unified_detection.static_ball_hard_mask_min_seen_frames",
                })
            if continuity_disabled > max(10, total_diag_frames * 0.08):
                cur = float(cfg_u.get("ball_max_motion_for_continuity_px", 140.0))
                nxt = min(220.0, round(cur + 20.0, 1))
                tuning_suggestions.append({
                    "priority": "low",
                    "issue": "Continuity frequently disabled by motion spike gate",
                    "recommendation": f"Consider raising ball_max_motion_for_continuity_px from {cur} to {nxt} if true tracks are fragmented",
                    "parameter": "unified_detection.ball_max_motion_for_continuity_px",
                })
            if no_ball_ratio > 0.35:
                tuning_suggestions.append({
                    "priority": "high",
                    "issue": "No-ball frames ratio is high",
                    "recommendation": "Primary recommendation: relax hard gates first (min_selected_confidence, continuity motion gate, hard-mask maturity) before tightening mirror gates",
                    "parameter": "composite",
                })

            diagnostics_payload = {
                **document_contract("frames", pipeline.session_metadata),
                "video_info": {
                    "path": pipeline.video_path,
                    "fps": pipeline.fps,
                    "resolution": [pipeline.width, pipeline.height],
                },
                "config_snapshot": {
                    "ball_min_selected_confidence": cfg_u.get("ball_min_selected_confidence"),
                    "ball_play_area_min_y_ratio_hard": cfg_u.get("ball_play_area_min_y_ratio_hard"),
                    "ball_max_motion_for_continuity_px": cfg_u.get("ball_max_motion_for_continuity_px"),
                    "static_ball_hard_mask_min_seen_frames": cfg_u.get("static_ball_hard_mask_min_seen_frames"),
                    "static_ball_hard_mask_radius_px": cfg_u.get("static_ball_hard_mask_radius_px"),
                },
                "summary": {
                    "total_frames": total_diag_frames,
                    "no_ball_frames": selected_none,
                    "no_ball_ratio": round(no_ball_ratio, 4),
                    "final_decision_counts": decisions,
                    "rejection_counts": rejection_counts,
                    "continuity_disabled_frames": continuity_disabled,
                },
                "tuning_suggestions": tuning_suggestions,
                "frames": diagnostics_records,
            }
            with open(diagnostics_path, "w", encoding="utf-8") as fp:
                json.dump(diagnostics_payload, fp, ensure_ascii=False, indent=json_dump_indent)
            print(f"🩺 [Analyzer] 诊断数据已保存至: {diagnostics_path}")

    cv2.destroyAllWindows()
    for s in shms: s.close()
    if pipeline.drop_stale_frames:
        print(f"⚡ [Analyzer] 已丢弃旧帧: {pipeline.dropped_stale_frames.value}")
    if out_writer is not None:
        print("✅ [Analyzer] Exit and saved video")
    else:
        print("✅ [Analyzer] Exit without video output")
    pipeline.stop_event.set()

