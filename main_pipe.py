#!/usr/bin/env python3
"""
网球分析多进程流水线 (Multiprocessing Pipeline) - 满负载完全同步版
特性: 严格索引锁定、全量视觉元素绘制、动态时间轴校准、多进程解耦
"""

import cv2
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import yaml
import os
import json
import argparse
import signal
import traceback
import uuid
from pathlib import Path
from typing import Optional

from analysis_data_contracts import (
    build_session_metadata,
    document_contract,
    normalize_session_id,
    stamp_frame_record,
)
from reader_runtime import DeadlinePacer, SourceFrameClock
from roi_stream_config import resolve_roi_stream_profile, sanitize_stream_source
from video_overlay_primitives import draw_ball_outline


def create_output_directory(output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_path


def ignore_child_interrupts():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class PipelineProcessError(RuntimeError):
    """Raised when any pipeline child exits unexpectedly."""


class MultiprocessPipeline:
    def __init__(
        self,
        config_path,
        input_path=None,
        output_path=None,
        max_frames=None,
        live_mode=False,
        drop_stale_frames=False,
        no_dual_view=False,
        hdmi_output=False,
        display_origin=None,
        no_metrics_text=False,
        no_frame_results=False,
        no_save_video=False,
        output_fps=None,
        inference_workers=None,
        analyze_swings=False,
        swing_output_json=None,
        swing_events_csv=None,
        swing_frames_csv=None,
        dominant_hand='right',
        min_peak_energy=9.0,
        active_energy=5.5,
        min_event_frames=8,
        max_internal_gap=3,
        min_event_gap=18,
        realtime_swing_events=False,
        realtime_swing_json=None,
        realtime_swing_html=None,
        realtime_swing_clips_dir=None,
        realtime_swing_event_log=None,
        realtime_analysis_interval=None,
        realtime_settle_frames=None,
        realtime_window_frames=None,
        realtime_clip_workers=None,
        realtime_frame_output=False,
        realtime_frame_jsonl=None,
        realtime_frame_snapshot_json=None,
        realtime_frame_snapshot_size=None,
        realtime_frame_flush_interval=None,
        realtime_coach=False,
        realtime_coach_max_chars=None,
        realtime_coach_max_suggestions=None,
        realtime_coach_min_confidence=None,
        deepseek_coach_options=None,
        realtime_open_report=False,
        session_id=None,
        session_output_root=None,
        evidence_manifest=None,
    ):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.session_started_at_unix_ns = time.time_ns()
        self.session_id = normalize_session_id(session_id, prefix="tennis")
        self.session_output_dir = None
        if session_output_root:
            self.session_output_dir = (
                Path(session_output_root).expanduser() / self.session_id
            ).resolve()
            self.session_output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_manifest = evidence_manifest

        # 命令行参数覆盖配置
        if input_path:
            self.config['video_input_path'] = input_path
        if output_path:
            self.config['video_output_path'] = output_path
        elif self.session_output_dir is not None:
            configured_output = Path(
                self.config.get("video_output_path") or "output_video.mp4"
            )
            self.config["video_output_path"] = str(
                self.session_output_dir / configured_output.name
            )
        if no_save_video:
            self.config['save_video'] = False

        self.video_path = self.config['video_input_path']
        configured_max_frames = self.config.get('video_processing', {}).get('max_frames', 0)
        self.max_frames = max(0, int(configured_max_frames if max_frames is None else max_frames))
        perf_cfg = self.config.setdefault('pipeline_perf', {})
        self.live_mode = bool(live_mode or perf_cfg.get('live_mode', False))
        self.drop_stale_frames = bool(drop_stale_frames or perf_cfg.get('drop_stale_frames', False))
        if self.live_mode:
            self.drop_stale_frames = True
            perf_cfg['dual_view_enabled'] = False
            perf_cfg['draw_metrics_text'] = False
            perf_cfg['collect_frame_results'] = False
        if no_dual_view:
            perf_cfg['dual_view_enabled'] = False
        if hdmi_output:
            perf_cfg['hdmi_output_enabled'] = True
            perf_cfg['dual_view_enabled'] = False
        if display_origin is not None:
            perf_cfg['display_origin'] = [
                int(display_origin[0]),
                int(display_origin[1]),
            ]
        if no_metrics_text:
            perf_cfg['draw_metrics_text'] = False
        if no_frame_results:
            perf_cfg['collect_frame_results'] = False
        if inference_workers is not None:
            perf_cfg['inference_workers'] = max(1, int(inference_workers))
        self.analyze_swings = bool(analyze_swings)
        self.swing_output_json = swing_output_json
        self.swing_events_csv = swing_events_csv
        self.swing_frames_csv = swing_frames_csv
        self.swing_analysis_options = {
            'dominant_hand': dominant_hand,
            'min_peak_energy': float(min_peak_energy),
            'active_energy': float(active_energy),
            'min_event_frames': max(1, int(min_event_frames)),
            'max_internal_gap': max(0, int(max_internal_gap)),
            'min_event_gap': max(0, int(min_event_gap)),
        }
        realtime_cfg = self.config.setdefault('realtime_swing', {})
        self.realtime_swing_events = bool(
            realtime_swing_events or realtime_cfg.get('enabled', False)
        )
        self.realtime_swing_json = realtime_swing_json
        self.realtime_swing_html = realtime_swing_html
        self.realtime_swing_clips_dir = realtime_swing_clips_dir
        self.realtime_swing_event_log = realtime_swing_event_log
        self.realtime_analysis_interval = max(
            1,
            int(
                realtime_analysis_interval
                if realtime_analysis_interval is not None
                else realtime_cfg.get('analysis_interval_frames', 5)
            ),
        )
        configured_settle = (
            realtime_settle_frames
            if realtime_settle_frames is not None
            else realtime_cfg.get('settle_frames')
        )
        configured_window = (
            realtime_window_frames
            if realtime_window_frames is not None
            else realtime_cfg.get('window_frames')
        )
        self.realtime_settle_frames = (
            None if configured_settle is None else max(0, int(configured_settle))
        )
        self.realtime_window_frames = (
            None if configured_window is None else max(32, int(configured_window))
        )
        self.realtime_clip_workers = max(
            1,
            int(
                realtime_clip_workers
                if realtime_clip_workers is not None
                else realtime_cfg.get('clip_workers', 1)
            ),
        )
        self.realtime_frame_output = bool(
            realtime_frame_output
            or realtime_frame_jsonl
            or realtime_frame_snapshot_json
            or realtime_cfg.get('frame_output_enabled', False)
        )
        if self.evidence_manifest:
            # A replayable bundle must retain both the inference journal and
            # the event snapshot, regardless of live-mode output defaults.
            self.realtime_frame_output = True
            self.realtime_swing_events = True
        self.realtime_frame_jsonl = realtime_frame_jsonl
        self.realtime_frame_snapshot_json = realtime_frame_snapshot_json
        self.realtime_frame_snapshot_size = max(
            1,
            int(
                realtime_frame_snapshot_size
                if realtime_frame_snapshot_size is not None
                else realtime_cfg.get('frame_snapshot_size', 200)
            ),
        )
        self.realtime_frame_flush_interval = max(
            1,
            int(
                realtime_frame_flush_interval
                if realtime_frame_flush_interval is not None
                else realtime_cfg.get('frame_flush_interval', 5)
            ),
        )
        self.realtime_coach = bool(
            realtime_coach or realtime_cfg.get('coach_enabled', False)
        )
        self.realtime_coach_max_chars = min(
            15,
            max(
                1,
                int(
                    realtime_coach_max_chars
                    if realtime_coach_max_chars is not None
                    else realtime_cfg.get('coach_max_chars', 15)
                ),
            ),
        )
        self.realtime_coach_max_suggestions = min(
            3,
            max(
                1,
                int(
                    realtime_coach_max_suggestions
                    if realtime_coach_max_suggestions is not None
                    else realtime_cfg.get('coach_max_suggestions', 3)
                ),
            ),
        )
        self.realtime_coach_min_confidence = max(
            0.0,
            min(
                1.0,
                float(
                    realtime_coach_min_confidence
                    if realtime_coach_min_confidence is not None
                    else realtime_cfg.get('coach_min_confidence', 0.45)
                ),
            ),
        )
        self.realtime_coach_biomechanics = dict(
            realtime_cfg.get('coach_biomechanics') or {}
        )
        deepseek_cfg = realtime_cfg.get('deepseek') or {}
        deepseek_overrides = deepseek_coach_options or {}

        def deepseek_option(name, default):
            override = deepseek_overrides.get(name)
            if override is not None:
                return override
            return deepseek_cfg.get(name, default)

        self.deepseek_coach_options = {
            'enabled': bool(
                deepseek_overrides.get('enabled', False)
                or deepseek_cfg.get('enabled', False)
            ),
            'model': str(deepseek_option('model', 'deepseek-v4-flash')),
            'base_url': str(
                deepseek_option('base_url', 'https://api.deepseek.com')
            ),
            'api_key_env': str(
                deepseek_option('api_key_env', 'DEEPSEEK_API_KEY')
            ),
            'timeout_seconds': max(
                0.2,
                float(deepseek_option('timeout_seconds', 3.0)),
            ),
            'workers': max(1, int(deepseek_option('workers', 2))),
            'max_chars': min(
                15,
                max(1, int(deepseek_option('max_chars', 15))),
            ),
        }
        if self.deepseek_coach_options['enabled']:
            self.realtime_coach = True
        if self.realtime_coach:
            self.realtime_swing_events = True
        self.realtime_clip_padding_frames = max(
            0,
            int(realtime_cfg.get('clip_padding_frames', 8)),
        )
        self.realtime_clip_max_width = max(
            160,
            int(realtime_cfg.get('clip_max_width', 1280)),
        )
        self.realtime_clip_jpeg_quality = max(
            40,
            min(100, int(realtime_cfg.get('clip_jpeg_quality', 85))),
        )
        self.realtime_open_report = bool(
            realtime_open_report or realtime_cfg.get('open_report', False)
        )
        if self.analyze_swings and not bool(perf_cfg.get('collect_frame_results', True)):
            raise ValueError(
                '--analyze-swings requires frame JSON output; remove --live-mode/--no-frame-results '
                'or run swing_event_analyzer.py after recording.'
            )

        self.is_stream_source = isinstance(self.video_path, str) and self.video_path.startswith(
            ('http://', 'https://', 'rtsp://', 'rtmp://', 'tcp://', 'udp://')
        )
        self.capture_open_timeout_ms = max(0, int(perf_cfg.get('reader_open_timeout_ms', 5000)))
        self.capture_read_timeout_ms = max(0, int(perf_cfg.get('reader_read_timeout_ms', 2000)))
        self.live_reconnect = bool(perf_cfg.get('live_reconnect', True))
        self.live_reconnect_delay = max(0.0, float(perf_cfg.get('live_reconnect_delay', 0.5)))
        self.live_slot_wait = max(0.0, float(perf_cfg.get('live_slot_wait_ms', 2.0)) / 1000.0)

        cap = self._open_capture()
        if not cap.isOpened():
            raise RuntimeError(
                f"无法打开视频输入: {sanitize_stream_source(self.video_path)}"
            )
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        if self.width <= 0 or self.height <= 0:
            raise RuntimeError(
                f"无法读取视频尺寸: {sanitize_stream_source(self.video_path)}"
            )

        self.roi_profile = resolve_roi_stream_profile(
            self.config,
            self.video_path,
            (self.width, self.height),
        )
        if self.roi_profile.enabled:
            print(
                f"🎯 [ROI] 已匹配 {self.roi_profile.label}"
                f" | {self.roi_profile.source}"
                f" | {list(self.roi_profile.points)}"
            )
        elif self.config.get("roi_settings", {}).get("enabled", False):
            print(
                f"⚠️ [ROI] 未启用裁剪: {self.roi_profile.reason}"
                f" | {self.roi_profile.source}"
            )
        self.session_metadata = build_session_metadata(
            session_id=self.session_id,
            source=self.video_path,
            stream_id=self.roi_profile.stream_id,
            stream_label=self.roi_profile.label,
            started_at_unix_ns=self.session_started_at_unix_ns,
            config=self.config,
            models={
                "detector": str(
                    (self.config.get("unified_detection") or {}).get("model_path")
                    or ""
                ),
                "pose": str(self.config.get("yolo_pose_model_path") or ""),
            },
        )
        print(
            f"🗂️ [Session] {self.session_id}"
            + (
                f" | {self.session_output_dir}"
                if self.session_output_dir is not None
                else ""
            )
        )

        configured_output_fps = self.config.get('video_processing', {}).get('output_fps', self.fps)
        self.output_fps = float(configured_output_fps if output_fps is None else output_fps)
        if self.output_fps <= 0:
            self.output_fps = self.fps

        # --- 共享内存池 ---
        normal_slots = max(3, int(perf_cfg.get('reader_buffer_slots', 12)))
        live_slots = max(3, int(perf_cfg.get('live_reader_buffer_slots', 3)))
        self.shm_num = live_slots if self.live_mode else normal_slots
        run_token = f"{os.getpid():x}{uuid.uuid4().hex[:4]}"
        self.shm_names = [f"tns3_{run_token}_{i}" for i in range(self.shm_num)]
        self.frame_size = self.height * self.width * 3

        # 同步原语
        self.q_free = mp.Queue(maxsize=self.shm_num)
        for i in range(self.shm_num): self.q_free.put(i)

        inference_queue_size = 1 if self.drop_stale_frames else self.shm_num
        self.q_inference = mp.Queue(maxsize=inference_queue_size)
        self.q_analyzer = mp.Queue(maxsize=self.shm_num)
        self.q_failures = mp.Queue()

        self.stop_event = mp.Event()
        self.inf_ready = mp.Event()
        self.rd_done = mp.Event()
        self.inf_done = mp.Event()
        self.dropped_stale_frames = mp.Value('i', 0)

    def _open_capture(self):
        if not self.is_stream_source:
            return cv2.VideoCapture(self.video_path)

        params = []
        if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC') and self.capture_open_timeout_ms:
            params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.capture_open_timeout_ms])
        if hasattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC') and self.capture_read_timeout_ms:
            params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.capture_read_timeout_ms])
        try:
            return cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG, params)
        except (TypeError, cv2.error):
            return cv2.VideoCapture(self.video_path)

    def reader_process(self):
        """进程 1: 稳定解码流"""
        ignore_child_interrupts()
        print("🚀 [Reader] 等待 AI 载入...")
        while not self.inf_ready.wait(timeout=0.5):
            if self.stop_event.is_set():
                self.rd_done.set()
                return
        if self.stop_event.is_set():
            self.rd_done.set()
            return

        cap = self._open_capture()
        shms = [shared_memory.SharedMemory(name=name) for name in self.shm_names]
        shared_frames = [
            np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=s.buf)
            for s in shms
        ]

        perf_cfg = self.config.get('pipeline_perf', {})
        if self.live_mode:
            limit_reader_fps = bool(perf_cfg.get('live_limit_reader_fps', True))
        else:
            limit_reader_fps = bool(perf_cfg.get('limit_reader_fps', True))
        max_lag_intervals = (
            float(perf_cfg.get('live_reader_max_lag_frames', 1.0))
            if self.live_mode
            else None
        )
        pacer = DeadlinePacer(
            self.fps if limit_reader_fps else 0.0,
            max_lag_intervals=max_lag_intervals,
        )
        frame_clock = SourceFrameClock()
        if self.live_mode:
            print(
                f"⚡ [Reader] 直播缓冲槽: {self.shm_num} | "
                f"推理邮箱: 1 | 时间线节流: {'开启' if limit_reader_fps else '关闭'}"
            )
        elif limit_reader_fps:
            print(f"⏱️ [Reader] 绝对时间线节流: {self.fps:.2f} FPS")

        try:
            while not self.stop_event.is_set():
                if self.max_frames and frame_clock.processed_count >= self.max_frames:
                    print(f"✅ [Reader] 达到最大帧数 {self.max_frames}，结束实时输入")
                    break

                slot = None
                if self.drop_stale_frames:
                    try:
                        stale_task = self.q_inference.get_nowait()
                        slot = stale_task['slot']
                        with self.dropped_stale_frames.get_lock():
                            self.dropped_stale_frames.value += 1
                    except queue.Empty:
                        pass

                if slot is None:
                    try:
                        timeout = self.live_slot_wait if self.drop_stale_frames else 2.0
                        slot = self.q_free.get(timeout=timeout)
                    except queue.Empty:
                        if self.drop_stale_frames:
                            if cap.grab():
                                frame_clock.dropped()
                                with self.dropped_stale_frames.get_lock():
                                    self.dropped_stale_frames.value += 1
                            continue
                        print("⚠️ [Reader] 延迟积压，等待消费...")
                        continue
                    except Exception as exc:
                        print(f"❌ [Reader] 获取空闲缓冲失败: {exc}")
                        self.stop_event.set()
                        raise RuntimeError("Reader buffer acquisition failed") from exc

                ret, frame = cap.read()
                captured_at = time.perf_counter()
                captured_at_unix_ns = time.time_ns()
                if not ret:
                    self.q_free.put(slot)
                    if self.is_stream_source and self.live_reconnect and not self.stop_event.is_set():
                        print("⚠️ [Reader] 码流中断，准备重连...")
                        cap.release()
                        if self.stop_event.wait(self.live_reconnect_delay):
                            break
                        cap = self._open_capture()
                        if not cap.isOpened():
                            print("⚠️ [Reader] 重连失败，继续重试")
                        continue
                    break

                source_frame_id = frame_clock.accepted()
                shared_frames[slot][:] = frame[:]
                self.q_inference.put({
                    'idx': source_frame_id,
                    'slot': slot,
                    'captured_at': captured_at,
                    'captured_at_unix_ns': captured_at_unix_ns,
                })

                delay = pacer.next_delay()
                if delay > 0 and self.stop_event.wait(delay):
                    break
        finally:
            cap.release()
            for s in shms:
                s.close()
            self.rd_done.set()

        print(
            f"✅ [Reader] 结束，共解析 {frame_clock.processed_count} 帧"
            f" | 源时间线 {frame_clock.source_count} 帧"
        )

    def inference_process(self):
        """进程 2: AI 推理核心 (平行调度)"""
        ignore_child_interrupts()
        print("🚀 [Inference] 加载 Core ML 并行架构...")
        from concurrent.futures import ThreadPoolExecutor
        shms = []
        executor = None
        try:
            from detection_frame_context import DetectionFrameContext
            from pose_estimator_yolo26 import PoseEstimatorYOLO26
            from roi_manager import ROIManager
            from yolo26n_unified_detector import YOLO26nUnifiedDetector

            roi_manager = None
            if self.roi_profile.enabled:
                candidate = ROIManager(self.config)
                if candidate.set_roi_points(self.roi_profile.points):
                    roi_manager = candidate

            detector = YOLO26nUnifiedDetector(
                self.config['unified_detection']['model_path'],
                self.config['unified_detection'],
                roi_manager=roi_manager,
            )
            pose_estimator = PoseEstimatorYOLO26(
                self.config['yolo_pose_model_path'],
                self.config,
                roi_manager=roi_manager,
            )

            shms = [shared_memory.SharedMemory(name=name) for name in self.shm_names]
            shared_frames = [
                np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=s.buf)
                for s in shms
            ]
            perf_cfg = self.config.get('pipeline_perf', {})
            unified_workers = (
                self.config.get('unified_detection', {}).get('max_workers')
                or self.config.get('performance_optimization', {}).get('max_workers')
                or 2
            )
            inference_workers = max(1, int(perf_cfg.get('inference_workers', unified_workers)))
            executor = ThreadPoolExecutor(max_workers=inference_workers)
            print(f"⚙️ [Inference] 线程池并发数: {inference_workers}")
            print("⚙️ [Inference] 单帧并行任务数: 2（目标检测 + 姿态估计）")
            if inference_workers > 2:
                print(
                    f"ℹ️ [Inference] 当前执行图每帧最多使用 2 个 worker；"
                    f"其余 {inference_workers - 2} 个用于后续多帧并发扩展"
                )
            if self.drop_stale_frames:
                print("⚡ [Inference] 直播新鲜度模式：Reader只保留最新待推理帧")
            self.inf_ready.set()

            while not self.stop_event.is_set():
                try:
                    task = self.q_inference.get(timeout=1.0)
                except queue.Empty:
                    if self.rd_done.is_set():
                        break
                    continue
                except Exception as exc:
                    print(f"❌ [Inference] 获取推理任务失败: {exc}")
                    self.stop_event.set()
                    raise RuntimeError("Inference queue read failed") from exc

                slot = task['slot']
                frame_ptr = shared_frames[slot]
                inference_started_at_unix_ns = time.time_ns()
                frame_context = None
                detection_frame = frame_ptr
                pose_detection_frame = frame_ptr
                if roi_manager is not None:
                    frame_context = DetectionFrameContext.build(
                        task['idx'],
                        frame_ptr,
                        roi_manager,
                        self.config,
                    )
                    detection_frame = frame_context.detection_frame
                    pose_detection_frame = frame_context.pose_detection_frame

                # 使用 ThreadPool 同时驱动检测和姿态模型。
                f1 = executor.submit(
                    detector.detect_unified,
                    detection_frame,
                    frame_context.roi_offset if frame_context is not None else (0, 0),
                    (self.width, self.height),
                )
                f2 = executor.submit(pose_estimator.get_keypoints, pose_detection_frame)

                ball, racket, _ = f1.result()
                ball_diagnostics = detector.get_last_ball_diagnostics()
                racket_diagnostics = detector.get_last_racket_diagnostics()
                pose = f2.result()
                if frame_context is not None:
                    pose, ball, racket = frame_context.adjust_detections(
                        pose,
                        ball,
                        racket,
                        object_coordinates_are_full_frame=True,
                    )
                inference_completed_at_unix_ns = time.time_ns()

                self.q_analyzer.put({
                    'id': task['idx'],
                    'slot': slot,
                    'ball': ball,
                    'racket': racket,
                    'pose': pose,
                    'ball_diagnostics': ball_diagnostics,
                    'racket_diagnostics': racket_diagnostics,
                    'captured_at': task.get('captured_at'),
                    'captured_at_unix_ns': task.get('captured_at_unix_ns'),
                    'inference_started_at_unix_ns': inference_started_at_unix_ns,
                    'inference_completed_at_unix_ns': inference_completed_at_unix_ns,
                })
        except Exception as exc:
            print(f"❌ [Inference] 初始化或推理失败: {exc}")
            self.stop_event.set()
            self.inf_ready.set()
            raise
        finally:
            if executor is not None:
                executor.shutdown()
            for s in shms:
                s.close()
            self.inf_done.set()
            print("✅ [Inference] 退出")

    def analyzer_process(self):
        """进程 3: 业务核心 + 全视觉渲染"""
        ignore_child_interrupts()
        print("🚀 [Analyzer] 初始化高清渲染引擎...")
        from frame_processor import FrameProcessor
        from performance_metrics import FpsTracker
        from pose_renderer import draw_pose_keypoints
        from video_writer_backend import create_video_writer

        perf_cfg = self.config.get('pipeline_perf', {})
        # 视频录制
        save_video = self.config.get('save_video', True)
        out_writer = None
        out_file = None
        if save_video:
            out_file = create_output_directory(self.config['video_output_path'])
            out_writer = create_video_writer(
                output_path=out_file,
                width=self.width,
                height=self.height,
                fps=self.output_fps,
                backend=perf_cfg.get('video_encoder_backend', 'auto'),
                bitrate=perf_cfg.get('video_encoder_bitrate', '12M'),
            )
            print(
                f"🎬 [Recorder] 录制中: {out_file}"
                f" | 编码器: {out_writer.backend_name}"
            )
            if self.output_fps != self.fps:
                print(f"🎬 [Recorder] 输出帧率: {self.output_fps:.2f} FPS")

        shms = [shared_memory.SharedMemory(name=name) for name in self.shm_names]
        shared_frames = [np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=s.buf) for s in shms]

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
            self.config['video_output_path']
        )
        if collect_frame_results:
            frame_results_path = os.path.splitext(output_base)[0] + '.json'
            diagnostics_path = os.path.splitext(output_base)[0] + '_diagnostics.json'
            frame_results_fp = open(frame_results_path, 'w', encoding='utf-8')
            frame_results_fp.write('{\n')
            frame_contract = document_contract("frames", self.session_metadata)
            frame_results_fp.write(
                f'  "schema_version": {json.dumps(frame_contract["schema_version"])},\n'
            )
            frame_results_fp.write(
                f'  "document_type": {json.dumps(frame_contract["document_type"])},\n'
            )
            frame_results_fp.write(
                f'  "session": {json.dumps(self.session_metadata, ensure_ascii=False)},\n'
            )
            frame_results_fp.write('  "video_info": {\n')
            frame_results_fp.write(f'    "path": {json.dumps(self.video_path, ensure_ascii=False)},\n')
            frame_results_fp.write(f'    "fps": {json.dumps(self.fps)},\n')
            frame_results_fp.write(f'    "resolution": {json.dumps([self.width, self.height])}\n')
            frame_results_fp.write('  },\n')
            frame_results_fp.write('  "frames": [\n')
        frame_processor = FrameProcessor(
            config=self.config,
            frame_dimensions=(self.height, self.width),
            fps=self.fps,
            analysis_stride=analysis_stride,
        )
        output_stem = os.path.splitext(output_base)[0]
        realtime_engine = None
        realtime_output = None
        frame_journal = None
        realtime_coach = None
        deepseek_sidecar = None
        realtime_runtime = None
        if self.realtime_frame_output:
            from realtime_swing_pipeline import RealtimeFrameJournal

            realtime_frame_jsonl = (
                self.realtime_frame_jsonl or f'{output_stem}_frames.jsonl'
            )
            realtime_frame_snapshot_json = (
                self.realtime_frame_snapshot_json
                or f'{output_stem}_frames_latest.json'
            )
            frame_journal = RealtimeFrameJournal(
                jsonl_path=realtime_frame_jsonl,
                snapshot_path=realtime_frame_snapshot_json,
                snapshot_size=self.realtime_frame_snapshot_size,
                flush_interval=self.realtime_frame_flush_interval,
                session_metadata=self.session_metadata,
            )
            print(
                f'📝 [Frame-Live] 实时逐帧输出已开启'
                f' | JSONL: {realtime_frame_jsonl}'
                f' | 快照: {realtime_frame_snapshot_json}'
            )
        if self.realtime_coach:
            from local_realtime_coach import LocalRealtimeCoach

            realtime_coach = LocalRealtimeCoach(
                max_chars=self.realtime_coach_max_chars,
                max_suggestions=self.realtime_coach_max_suggestions,
                min_confidence=self.realtime_coach_min_confidence,
                thresholds=self.realtime_coach_biomechanics,
            )
        if self.deepseek_coach_options['enabled']:
            from deepseek_realtime_coach import DeepSeekCoachSidecar

            api_key_env = self.deepseek_coach_options['api_key_env']
            deepseek_sidecar = DeepSeekCoachSidecar(
                api_key=os.environ.get(api_key_env, ''),
                model=self.deepseek_coach_options['model'],
                base_url=self.deepseek_coach_options['base_url'],
                timeout_seconds=self.deepseek_coach_options['timeout_seconds'],
                workers=self.deepseek_coach_options['workers'],
                max_chars=self.deepseek_coach_options['max_chars'],
            )
            key_status = '已配置' if os.environ.get(api_key_env) else '未配置'
            print(
                f'🧠 [DeepSeek] 旁路Coach已开启'
                f' | 模型: {self.deepseek_coach_options["model"]}'
                f' | 超时: {self.deepseek_coach_options["timeout_seconds"]:.1f}s'
                f' | {api_key_env}: {key_status}'
            )
        if self.realtime_swing_events:
            from realtime_swing_pipeline import (
                RealtimeSwingEventEngine,
                RealtimeSwingOutputManager,
            )

            realtime_json = self.realtime_swing_json or f'{output_stem}_swing_events.json'
            realtime_html = self.realtime_swing_html or f'{output_stem}_swing_report.html'
            realtime_clips_dir = self.realtime_swing_clips_dir or f'{output_stem}_swing_clips'
            effective_window_frames = (
                self.realtime_window_frames
                if self.realtime_window_frames is not None
                else max(32, int(round(self.fps * 8.0)))
            )
            effective_settle_frames = (
                self.realtime_settle_frames
                if self.realtime_settle_frames is not None
                else max(0, int(round(self.fps * 0.6)))
            )
            realtime_engine = RealtimeSwingEventEngine(
                fps=self.fps,
                analysis_interval_frames=self.realtime_analysis_interval,
                settle_frames=effective_settle_frames,
                window_frames=effective_window_frames,
                coach=realtime_coach,
                session_metadata=self.session_metadata,
                **self.swing_analysis_options,
            )
            realtime_output = RealtimeSwingOutputManager(
                output_json=realtime_json,
                output_html=realtime_html,
                clips_dir=realtime_clips_dir,
                fps=self.output_fps,
                frame_size=(self.width, self.height),
                buffer_frames=(
                    effective_window_frames
                    + effective_settle_frames
                    + self.realtime_clip_padding_frames * 2
                ),
                clip_workers=self.realtime_clip_workers,
                video_backend=perf_cfg.get('video_encoder_backend', 'auto'),
                video_bitrate=perf_cfg.get('video_encoder_bitrate', '12M'),
                clip_padding_frames=self.realtime_clip_padding_frames,
                clip_max_width=self.realtime_clip_max_width,
                jpeg_quality=self.realtime_clip_jpeg_quality,
                preview_path=(
                    f"{os.path.splitext(realtime_html)[0]}_roi_preview.jpg"
                    if self.roi_profile.enabled
                    else None
                ),
                roi_metadata=self.roi_profile.as_metadata(),
                preview_interval_frames=max(
                    1,
                    int(
                        self.config.get("roi_settings", {}).get(
                            "preview_interval_frames",
                            round(self.output_fps),
                        )
                    ),
                ),
                event_log_path=self.realtime_swing_event_log,
                session_metadata=self.session_metadata,
            )
            print(
                f'⚡ [Swing-Live] 实时事件分析已开启'
                f' | JSON: {realtime_json}'
                f' | HTML: {realtime_html}'
                f' | 异步片段线程: {self.realtime_clip_workers}'
                f' | 本地Coach: {"开启" if realtime_coach is not None else "关闭"}'
                f' | DeepSeek旁路: {"开启" if deepseek_sidecar is not None else "关闭"}'
            )
            if self.realtime_open_report:
                from pathlib import Path
                import webbrowser

                webbrowser.open(Path(realtime_html).resolve().as_uri())

        if realtime_engine is not None or frame_journal is not None:
            from realtime_swing_runtime import RealtimeSwingRuntime

            realtime_runtime = RealtimeSwingRuntime(
                engine=realtime_engine,
                output=realtime_output,
                frame_journal=frame_journal,
                deepseek_sidecar=deepseek_sidecar,
                stop_event=self.stop_event,
                queue_size=max(64, int(round(self.fps * 12))),
            )

        while not self.stop_event.is_set():
            try:
                data = self.q_analyzer.get(timeout=2.0)
            except queue.Empty:
                if self.inf_done.is_set():
                    break
                continue
            except Exception as exc:
                print(f"❌ [Analyzer] 获取分析任务失败: {exc}")
                self.stop_event.set()
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
                )
                frame_record["detection_diagnostics"] = data.get("ball_diagnostics") or {}
                frame_record["racket_detection_diagnostics"] = (
                    data.get("racket_diagnostics") or {}
                )
                analysis_completed_at_unix_ns = time.time_ns()
                stamp_frame_record(
                    frame_record,
                    session=self.session_metadata,
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
            if collect_frame_results and frame_record is not None:
                if frame_results_fp is not None:
                    if not first_frame_record:
                        frame_results_fp.write(',\n')
                    frame_results_fp.write(json.dumps(frame_record, ensure_ascii=False))
                    first_frame_record = False
                diagnostics_records.append({
                    "frame_id": fid,
                    "timestamp": round(fid / self.fps, 3) if self.fps else 0.0,
                    "ball_selected": [norm_ball_pos[0], norm_ball_pos[1]] if norm_ball_pos else None,
                    "ball_diagnostics": data.get("ball_diagnostics") or {},
                })

            # --- 2. 视觉渲染流程 (原生 OpenCV 绘制，性能极大提升) ---
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
                        # k 已经是英文 (如 shoulder_turn)，颜色区分保持
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
                x1, y1, x2, y2 = ra['box']
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 128, 0), 2)

            # --- 3. 提交并释放 ---
            if realtime_runtime is not None and frame_record is not None:
                realtime_runtime.submit_frame(frame_record)
                # Swing/Coach data is latency-sensitive; enqueue it before the
                # 1440p rendered-frame copy used only for previews and clips.
                realtime_runtime.record_rendered_frame(fid, canvas)
            if out_writer: out_writer.write(canvas)

            # HDMI 单路输出与双窗口对比显示互斥。
            if processed_display is not None:
                if processed_display.show(canvas):
                    self.stop_event.set()
            elif dual_view_enabled:
                h, w = canvas.shape[:2]
                # 缩放原始图和处理图
                scaled_size = (int(w * dual_view_scale), int(h * dual_view_scale))
                orig_small = cv2.resize(shared_frames[slot], scaled_size)
                proc_small = cv2.resize(canvas, scaled_size)

                # 水平堆叠
                dual_view = np.hstack((orig_small, proc_small))

                # 为双视角增加文字标识
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

                cv2.imshow("Tennis AI Analyer - Dual Comparison (ESC to Quit)", dual_view)
                key = cv2.waitKey(1) & 0xFF
                if key == 27: # ESC 键退出
                    self.stop_event.set()

            self.q_free.put(slot) # 释放回池子

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
                self.stop_event.set()
                raise
            if realtime_engine is not None:
                print(
                    f"✅ [Swing-Live] 实时分析结束"
                    f" | events: {realtime_runtime.snapshot()['summary']['swing_event_count']}"
                )
            if frame_journal is not None:
                print("✅ [Frame-Live] 实时逐帧输出结束")

        if out_writer:
            out_writer.release()
        if processed_display is not None:
            processed_display.close()

        if collect_frame_results and frame_results_fp is not None:
            frame_results_fp.write('\n  ],\n')
            frame_results_fp.write(
                f'  "summary": {json.dumps({"total_frames": count, "session_id": self.session_id}, ensure_ascii=False, indent=json_dump_indent)}\n'
            )
            frame_results_fp.write('}\n')
            frame_results_fp.close()
            print(f"📊 [Analyzer] 数据已保存至: {frame_results_path}")
            if self.analyze_swings:
                # Keep event segmentation outside the frame loop so pipeline throughput and
                # live rendering remain independent from the offline event-analysis cost.
                from swing_event_analyzer import (
                    analyze_frame_records,
                    default_output_paths,
                    load_frame_records,
                    write_analysis_outputs,
                )

                swing_paths = default_output_paths(frame_results_path)
                swing_output_json = self.swing_output_json or swing_paths['json']
                swing_events_csv = self.swing_events_csv or swing_paths['events_csv']
                swing_frames_csv = self.swing_frames_csv or swing_paths['frames_csv']
                print('🏌️ [Swing] 开始事件级挥拍分段分析...')
                swing_analysis = analyze_frame_records(
                    load_frame_records(frame_results_path),
                    session_metadata=self.session_metadata,
                    **self.swing_analysis_options,
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
                cfg_u = self.config.get("unified_detection", {})
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
                    **document_contract("frames", self.session_metadata),
                    "video_info": {
                        "path": self.video_path,
                        "fps": self.fps,
                        "resolution": [self.width, self.height],
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
        if self.drop_stale_frames:
            print(f"⚡ [Analyzer] 已丢弃旧帧: {self.dropped_stale_frames.value}")
        if out_writer is not None:
            print("✅ [Analyzer] Exit and saved video")
        else:
            print("✅ [Analyzer] Exit without video output")
        self.stop_event.set()

    def _process_entry(self, role):
        ignore_child_interrupts()
        target = {
            "Reader": self.reader_process,
            "Inference": self.inference_process,
            "Analyzer": self.analyzer_process,
        }[role]
        try:
            target()
        except BaseException as exc:
            report = {
                "role": role,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            try:
                self.q_failures.put(report)
            finally:
                self.stop_event.set()
            print(
                f"❌ [{role}] 子进程异常退出"
                f" | {report['error_type']}: {report['message']}"
            )
            raise

    @staticmethod
    def _join_or_terminate(processes, timeout=5.0):
        deadline = time.monotonic() + max(0.0, float(timeout))
        for process in processes.values():
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in processes.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    def _write_evidence_manifest(self, status: str) -> Optional[str]:
        if not self.evidence_manifest:
            return None
        from session_evidence_bundle import write_session_evidence_manifest

        output_base = str(self.config.get('video_output_path') or 'output_video.mp4')
        output_stem = os.path.splitext(output_base)[0]
        frame_jsonl = self.realtime_frame_jsonl or f'{output_stem}_frames.jsonl'
        frame_snapshot = (
            self.realtime_frame_snapshot_json
            or f'{output_stem}_frames_latest.json'
        )
        event_json = self.realtime_swing_json or f'{output_stem}_swing_events.json'
        event_log = self.realtime_swing_event_log or str(
            Path(event_json).with_suffix('.jsonl')
        )
        event_html = self.realtime_swing_html or f'{output_stem}_swing_report.html'
        clips_dir = Path(
            self.realtime_swing_clips_dir or f'{output_stem}_swing_clips'
        )
        artifacts = [
            {
                'role': 'frame_journal',
                'path': frame_jsonl,
                'required_for_replay': True,
            },
            {
                'role': 'event_snapshot',
                'path': event_json,
                'required_for_replay': True,
            },
            {'role': 'frame_snapshot', 'path': frame_snapshot},
            {'role': 'event_log', 'path': event_log},
            {'role': 'report', 'path': event_html},
        ]
        if bool(self.config.get('save_video', True)):
            artifacts.append({'role': 'annotated_video', 'path': output_base})
        if not self.is_stream_source and Path(self.video_path).expanduser().is_file():
            artifacts.append({'role': 'source_video', 'path': self.video_path})
        if clips_dir.is_dir():
            artifacts.extend(
                {'role': 'swing_clip', 'path': str(path)}
                for path in sorted(clips_dir.glob('*.mp4'))
            )
        effective_settle = (
            self.realtime_settle_frames
            if self.realtime_settle_frames is not None
            else max(0, int(round(self.fps * 0.6)))
        )
        effective_window = (
            self.realtime_window_frames
            if self.realtime_window_frames is not None
            else max(32, int(round(self.fps * 8.0)))
        )
        replay = {
            'engine': 'RealtimeSwingEventEngine',
            'fps': self.fps,
            'analysis_interval_frames': self.realtime_analysis_interval,
            'settle_frames': effective_settle,
            'window_frames': effective_window,
            'swing_options': dict(self.swing_analysis_options),
            'coach': {
                'enabled': self.realtime_coach,
                'max_chars': self.realtime_coach_max_chars,
                'max_suggestions': self.realtime_coach_max_suggestions,
                'min_confidence': self.realtime_coach_min_confidence,
                'thresholds': dict(self.realtime_coach_biomechanics),
            },
        }
        capture = {
            'source_kind': 'stream' if self.is_stream_source else 'video_file',
            'source': self.session_metadata.get('source'),
            'fps': self.fps,
            'output_fps': self.output_fps,
            'resolution': [self.width, self.height],
            'drop_stale_frames': self.drop_stale_frames,
            'dropped_stale_frames': int(self.dropped_stale_frames.value),
            'max_frames': self.max_frames,
            'roi': self.roi_profile.as_metadata(),
        }
        path = write_session_evidence_manifest(
            self.evidence_manifest,
            session=self.session_metadata,
            status=status,
            capture=capture,
            replay=replay,
            artifacts=artifacts,
        )
        print(f'📦 [Evidence] 可重放证据清单: {path}')
        return path

    def run(self):
        objs = []
        try:
            for name in self.shm_names:
                objs.append(
                    shared_memory.SharedMemory(
                        name=name,
                        create=True,
                        size=self.frame_size,
                    )
                )
        except Exception:
            for shm in objs:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            raise

        processes = {
            role: mp.Process(
                target=self._process_entry,
                args=(role,),
                name=f"tennis-{role.lower()}",
            )
            for role in ("Reader", "Inference", "Analyzer")
        }
        for process in processes.values():
            process.start()
        failure = None
        interrupted = False
        try:
            while any(process.is_alive() for process in processes.values()):
                try:
                    failure = self.q_failures.get(timeout=0.1)
                except queue.Empty:
                    failure = None
                if failure is not None:
                    self.stop_event.set()
                    break
                for role, process in processes.items():
                    if process.exitcode not in (None, 0):
                        failure = {
                            "role": role,
                            "error_type": "ProcessExit",
                            "message": f"exit code {process.exitcode}",
                            "traceback": "",
                        }
                        self.stop_event.set()
                        break
                if failure is not None:
                    break
        except KeyboardInterrupt:
            interrupted = True
            print("\n🛑 收到停止请求，正在安全关闭Pipeline...")
            self.stop_event.set()
        finally:
            self._join_or_terminate(processes)
            if failure is None and not interrupted:
                for role, process in processes.items():
                    if process.exitcode not in (0, None):
                        failure = {
                            "role": role,
                            "error_type": "ProcessExit",
                            "message": f"exit code {process.exitcode}",
                            "traceback": "",
                        }
                        break
            for shm in objs:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass

        evidence_error = None
        try:
            self._write_evidence_manifest(
                'failed' if failure is not None else ('stopped' if interrupted else 'completed')
            )
        except Exception as exc:
            evidence_error = exc
            print(f'❌ [Evidence] 证据清单生成失败: {exc}')

        if failure is not None:
            detail = failure.get("traceback") or failure.get("message") or ""
            raise PipelineProcessError(
                f"{failure.get('role', 'Pipeline')} failed: "
                f"{failure.get('error_type', 'Error')}: "
                f"{failure.get('message', '')}\n{detail}"
            )
        if evidence_error is not None:
            raise RuntimeError("Evidence manifest generation failed") from evidence_error
        print("🏁 任务流执行完毕")


def build_argument_parser():
    parser = argparse.ArgumentParser(description='网球分析多进程流水线')
    parser.add_argument('--config', '-c', default='configs/yolo26_tennis_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--input', '-i', default=None, help='输入视频路径，覆盖配置文件')
    parser.add_argument('--output', '-o', default=None, help='输出视频路径，覆盖配置文件')
    parser.add_argument(
        '--session-id',
        default=None,
        help='本次分析会话ID；默认自动生成并写入所有JSON/事件记录',
    )
    parser.add_argument(
        '--session-output-root',
        default=None,
        help='会话输出根目录；指定后默认产物写入 <root>/<session-id>/，显式输出路径仍保持不变',
    )
    parser.add_argument(
        '--evidence-manifest',
        default=None,
        help='可重放证据包清单；写入FrameRecord/Event哈希与重放参数',
    )
    parser.add_argument('--max-frames', type=int, default=None,
                        help='最大处理帧数；直播流建议设置以便安全结束。默认使用 video_processing.max_frames')
    parser.add_argument('--live-mode', action='store_true',
                        help='直播优化：关闭双视图、详细指标文本和逐帧 JSON，并优先处理最新帧')
    parser.add_argument('--drop-stale-frames', action='store_true',
                        help='处理跟不上输入时丢弃排队旧帧，降低直播延迟')
    parser.add_argument('--no-dual-view', action='store_true', help='关闭原始/处理结果双窗口显示')
    parser.add_argument(
        '--hdmi-output',
        action='store_true',
        help='将处理后单路画面无边框全屏输出；自动关闭 dual-view',
    )
    parser.add_argument(
        '--display-origin',
        type=int,
        nargs=2,
        metavar=('X', 'Y'),
        default=None,
        help='扩展桌面模式下目标显示器左上角坐标；镜像模式使用默认 0 0',
    )
    parser.add_argument('--no-metrics-text', action='store_true', help='关闭逐帧详细指标文字绘制')
    parser.add_argument('--no-frame-results', action='store_true', help='关闭逐帧 JSON 与诊断 JSON 写入')
    parser.add_argument('--no-save-video', action='store_true', help='禁用处理后视频录制')
    parser.add_argument('--output-fps', type=float, default=None, help='覆盖输出视频帧率')
    parser.add_argument(
        '--inference-workers',
        type=int,
        default=None,
        help='推理线程池 worker 数；当前执行图每帧最多并行 2 个任务',
    )
    parser.add_argument('--analyze-swings', action='store_true',
                        help='逐帧 JSON 写完后自动生成 Swing 事件 JSON/CSV；不可与 --live-mode 或 --no-frame-results 同用')
    parser.add_argument('--swing-output-json',
                        help='自动 Swing 分析的事件 JSON 输出路径；默认与逐帧 JSON 同目录')
    parser.add_argument('--swing-events-csv',
                        help='自动 Swing 分析的事件摘要 CSV 输出路径')
    parser.add_argument('--swing-frames-csv',
                        help='自动 Swing 分析的逐帧审计 CSV 输出路径')
    parser.add_argument('--dominant-hand', choices=['right', 'left'], default='right',
                        help='Swing 分析的持拍手，默认 right')
    parser.add_argument('--min-peak-energy', type=float, default=9.0,
                        help='事件峰值最小运动能量，默认 9.0')
    parser.add_argument('--active-energy', type=float, default=5.5,
                        help='进入挥拍事件的最小运动能量，默认 5.5')
    parser.add_argument('--min-event-frames', type=int, default=8,
                        help='有效挥拍事件的最小帧数，默认 8')
    parser.add_argument('--max-internal-gap', type=int, default=3,
                        help='同一挥拍内允许的最大非活跃间隔帧数，默认 3')
    parser.add_argument('--min-event-gap', type=int, default=18,
                        help='相邻挥拍事件的最小间隔帧数，默认 18')
    parser.add_argument('--realtime-swing-events', action='store_true',
                        help='对直播码流或按时间线播放的视频滚动识别完整挥拍，并更新事件 JSON、HTML 与异步事件片段')
    parser.add_argument('--realtime-swing-json',
                        help='实时 Swing 事件 JSON；默认 <output_stem>_swing_events.json')
    parser.add_argument('--realtime-swing-html',
                        help='实时 Swing HTML 页面；默认 <output_stem>_swing_report.html')
    parser.add_argument('--realtime-swing-clips-dir',
                        help='实时 Swing 独立片段目录；默认 <output_stem>_swing_clips')
    parser.add_argument(
        '--realtime-swing-event-log',
        help='追加式实时事件JSONL；默认与实时事件JSON同名并使用 .jsonl 后缀',
    )
    parser.add_argument('--realtime-analysis-interval', type=int, default=None,
                        help='每隔多少个已处理帧运行一次滚动事件分析，默认 5')
    parser.add_argument('--realtime-settle-frames', type=int, default=None,
                        help='挥拍结束后等待多少源帧再发布，默认约 0.6 秒')
    parser.add_argument('--realtime-window-frames', type=int, default=None,
                        help='实时事件分析滚动窗口帧数，默认约 8 秒')
    parser.add_argument('--realtime-clip-workers', type=int, default=None,
                        help='异步 Swing 片段编码线程数，默认 1')
    parser.add_argument('--realtime-frame-output', action='store_true',
                        help='异步保存已完成推理的逐帧 JSONL 与最近帧 JSON 快照')
    parser.add_argument('--realtime-frame-jsonl',
                        help='实时逐帧 JSONL；默认 <output_stem>_frames.jsonl')
    parser.add_argument('--realtime-frame-snapshot-json',
                        help='最近帧原子快照 JSON；默认 <output_stem>_frames_latest.json')
    parser.add_argument('--realtime-frame-snapshot-size', type=int, default=None,
                        help='最近帧 JSON 保留的记录数，默认 200')
    parser.add_argument('--realtime-frame-flush-interval', type=int, default=None,
                        help='每多少个已处理帧刷新 JSONL 与最近帧快照，默认 5')
    parser.add_argument('--realtime-coach', action='store_true',
                        help='为每个确认挥拍生成1-3条不超过15字的本地实时动作纠错')
    parser.add_argument('--realtime-coach-max-chars', type=int, default=None,
                        help='本地实时指导最大字数，范围 1-15，默认 15')
    parser.add_argument('--realtime-coach-max-suggestions', type=int, choices=[1, 2, 3], default=None,
                        help='每次挥拍输出建议数上限，范围 1-3，默认 3')
    parser.add_argument('--realtime-coach-min-confidence', type=float, default=None,
                        help='采用生物力学指标的最低置信度，范围 0-1，默认 0.45')
    parser.add_argument('--deepseek-coach', action='store_true',
                        help='异步调用 DeepSeek V4 Flash 生成旁路指导；本地建议不等待')
    parser.add_argument('--deepseek-model',
                        help='DeepSeek 模型名，默认 deepseek-v4-flash')
    parser.add_argument('--deepseek-base-url',
                        help='OpenAI兼容地址，默认 https://api.deepseek.com')
    parser.add_argument('--deepseek-api-key-env',
                        help='保存API密钥的环境变量名，默认 DEEPSEEK_API_KEY')
    parser.add_argument('--deepseek-timeout-seconds', type=float, default=None,
                        help='单次DeepSeek请求超时秒数，默认 3.0')
    parser.add_argument('--deepseek-workers', type=int, default=None,
                        help='DeepSeek旁路并发请求数，默认 2')
    parser.add_argument('--deepseek-coach-max-chars', type=int, default=None,
                        help='DeepSeek建议最大字数，范围 1-15，默认 15')
    parser.add_argument('--realtime-open-report', action='store_true',
                        help='启动实时 Swing 输出时在系统浏览器打开 HTML 页面')
    return parser


def main_cli(argv=None):
    args = build_argument_parser().parse_args(argv)

    # 如果没有指定输入且配置文件里也没有，给个默认值
    if not args.input:
        # 尝试从配置加载看有没有
        with open(args.config, 'r') as f:
            tmp_cfg = yaml.safe_load(f)
            if 'video_input_path' not in tmp_cfg:
                args.input = "data/16.10.mp4"

    MultiprocessPipeline(
        args.config,
        input_path=args.input,
        output_path=args.output,
        max_frames=args.max_frames,
        live_mode=args.live_mode,
        drop_stale_frames=args.drop_stale_frames,
        no_dual_view=args.no_dual_view,
        hdmi_output=args.hdmi_output,
        display_origin=args.display_origin,
        no_metrics_text=args.no_metrics_text,
        no_frame_results=args.no_frame_results,
        no_save_video=args.no_save_video,
        output_fps=args.output_fps,
        inference_workers=args.inference_workers,
        analyze_swings=args.analyze_swings,
        swing_output_json=args.swing_output_json,
        swing_events_csv=args.swing_events_csv,
        swing_frames_csv=args.swing_frames_csv,
        dominant_hand=args.dominant_hand,
        min_peak_energy=args.min_peak_energy,
        active_energy=args.active_energy,
        min_event_frames=args.min_event_frames,
        max_internal_gap=args.max_internal_gap,
        min_event_gap=args.min_event_gap,
        realtime_swing_events=args.realtime_swing_events,
        realtime_swing_json=args.realtime_swing_json,
        realtime_swing_html=args.realtime_swing_html,
        realtime_swing_clips_dir=args.realtime_swing_clips_dir,
        realtime_swing_event_log=args.realtime_swing_event_log,
        realtime_analysis_interval=args.realtime_analysis_interval,
        realtime_settle_frames=args.realtime_settle_frames,
        realtime_window_frames=args.realtime_window_frames,
        realtime_clip_workers=args.realtime_clip_workers,
        realtime_frame_output=args.realtime_frame_output,
        realtime_frame_jsonl=args.realtime_frame_jsonl,
        realtime_frame_snapshot_json=args.realtime_frame_snapshot_json,
        realtime_frame_snapshot_size=args.realtime_frame_snapshot_size,
        realtime_frame_flush_interval=args.realtime_frame_flush_interval,
        realtime_coach=args.realtime_coach,
        realtime_coach_max_chars=args.realtime_coach_max_chars,
        realtime_coach_max_suggestions=args.realtime_coach_max_suggestions,
        realtime_coach_min_confidence=args.realtime_coach_min_confidence,
        deepseek_coach_options={
            'enabled': args.deepseek_coach,
            'model': args.deepseek_model,
            'base_url': args.deepseek_base_url,
            'api_key_env': args.deepseek_api_key_env,
            'timeout_seconds': args.deepseek_timeout_seconds,
            'workers': args.deepseek_workers,
            'max_chars': args.deepseek_coach_max_chars,
        },
        realtime_open_report=args.realtime_open_report,
        session_id=args.session_id,
        session_output_root=args.session_output_root,
        evidence_manifest=args.evidence_manifest,
    ).run()


if __name__ == "__main__":
    main_cli()
