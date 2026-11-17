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
from motion_time_contract import REFERENCE_HZ, POLICY_VERSION as MOTION_TIME_POLICY

from analysis_data_contracts import (
    SOURCE_TIME_SCHEMA_VERSION,
    build_session_metadata,
    document_contract,
    normalize_session_id,
    stamp_frame_record,
)
from reader_runtime import DeadlinePacer, SourceFrameClock, SourceMediaClock
from roi_stream_config import resolve_roi_stream_profile, sanitize_stream_source
from video_overlay_primitives import draw_ball_outline

from pipeline import create_output_directory
from pipeline.cli import build_argument_parser, main_cli as _pipeline_main_cli
from pipeline.evidence import write_evidence_manifest
from pipeline.inference_worker import run_inference_process
from pipeline.analyzer_worker import run_analyzer_process


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
        refractory_frames=None,
        min_wrist_sweep=120.0,
        min_arm_extension_range=65.0,
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
        coach_tts_options=None,
        realtime_open_report=False,
        session_id=None,
        session_output_root=None,
        evidence_manifest=None,
        algo2_dual_view=False,
        algo2_config=None,
        stream_id=None,
        racket_pose_mode=None,
        processing_mode=None,
        inference_schedule=None,
        racket_ipc=None,
    ):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.stream_id = (
            stream_id
            or self.config.get("stream_id")
            or self.config.get("target_stream_id")
            or (self.config.get("roi_settings") or {}).get("target_stream_id")
        )
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
        if perf_cfg.get('pose_resize_policy', 'legacy') not in ('direct_source', 'direct_front', 'legacy'):
            raise ValueError('pose_resize_policy must be direct_source, direct_front or legacy')
        self.live_mode = bool(live_mode or perf_cfg.get('live_mode', False))
        self.drop_stale_frames = bool(drop_stale_frames or perf_cfg.get('drop_stale_frames', False))
        requested_processing = processing_mode or perf_cfg.get('processing_mode', 'auto')
        if requested_processing not in ('auto', 'realtime', 'buffered', 'offline'):
            raise ValueError('processing_mode must be auto, realtime, buffered or offline')
        self.lossless_realtime = requested_processing == 'buffered'
        if self.lossless_realtime:
            self.live_mode = True
            self.drop_stale_frames = False
        elif requested_processing == 'realtime':
            self.live_mode = True
            self.drop_stale_frames = True
        elif requested_processing == 'offline':
            self.live_mode = False
            self.drop_stale_frames = False
            perf_cfg['limit_reader_fps'] = False
        self.processing_mode = 'buffered' if self.lossless_realtime else ('realtime' if self.live_mode else 'offline')
        from racket_pose_detector import pose_config
        racket_cfg = dict(self.config.get('racket_pose') or {})
        if racket_pose_mode is not None:
            racket_cfg['mode'] = racket_pose_mode
            racket_cfg['selection_policy'] = 'exclusive' if racket_pose_mode=='exclusive' else 'mixed'
        if racket_ipc is not None:
            racket_cfg['ipc_transport'] = racket_ipc
        self.racket_pose_config = pose_config(racket_cfg)
        from racket_source_policy import is_exclusive
        self.exclusive_racket_source = is_exclusive(self.racket_pose_config)
        if self.exclusive_racket_source and self.processing_mode=='realtime':
            raise ValueError('新版七点独占方案要求 buffered 或 offline 逐帧处理')
        self.inference_schedule = inference_schedule or perf_cfg.get('inference_schedule', 'barrier')
        if self.inference_schedule not in ('barrier','interframe'):
            raise ValueError('Unsupported inference schedule')
        if self.inference_schedule=='interframe' and (not self.exclusive_racket_source or self.processing_mode=='realtime'):
            raise ValueError('Interframe scheduling requires exclusive buffered/offline processing')
        if (self.racket_pose_config['mode']!='legacy' and self.racket_pose_config['ipc_transport']=='shared_memory'
                and self.processing_mode=='realtime'):
            raise ValueError('Shared source IPC requires buffered/offline processing')
        perf_cfg['inference_schedule'] = self.inference_schedule
        print('🎾 [Startup] 球拍方案: '+('新版七点独占，禁止旧拍框回退' if self.exclusive_racket_source
              else ('旧版' if self.racket_pose_config['mode']=='legacy' else '兼容诊断 '+self.racket_pose_config['mode'])))
        self.config['racket_pose'] = dict(self.racket_pose_config)
        if requested_processing != 'auto':
            perf_cfg['processing_mode'] = requested_processing
        self.racket_pose_journal = None

        self.algo2_dual_view = bool(
            algo2_dual_view
            or self.config.get('algo2_dual_view', {}).get('enabled', False)
            or perf_cfg.get('algo2_dual_view', False)
        )
        self.algo2_config_path = (
            algo2_config
            or self.config.get('algo2_dual_view', {}).get('config_path')
            or "configs/dual_view_config.yaml"
        )
        if self.algo2_dual_view:
            perf_cfg['dual_view_enabled'] = False
            realtime_swing_events = True
            realtime_coach = True

        if self.live_mode:
            self.drop_stale_frames = not self.lossless_realtime
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
            'refractory_frames': int(refractory_frames) if refractory_frames is not None else None,
            'min_wrist_sweep': float(min_wrist_sweep) if min_wrist_sweep is not None else 0.0,
            'min_arm_extension_range': float(min_arm_extension_range) if min_arm_extension_range is not None else 0.0,
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
            'max_pending': max(0, int(deepseek_option('max_pending', 2))),
            'max_chars': min(
                15,
                max(1, int(deepseek_option('max_chars', 15))),
            ),
        }
        coach_tts_cfg = realtime_cfg.get('coach_tts') or {}
        coach_tts_overrides = coach_tts_options or {}

        def coach_tts_option(name, default):
            override = coach_tts_overrides.get(name)
            if override is not None:
                return override
            return coach_tts_cfg.get(name, default)

        self.coach_tts_options = {
            'enabled': bool(
                coach_tts_overrides.get('enabled', False)
                or coach_tts_cfg.get('enabled', False)
            ),
            'model': str(
                coach_tts_option(
                    'model',
                    'mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
                )
            ),
            'voice': str(coach_tts_option('voice', 'Vivian')),
            'language': str(coach_tts_option('language', 'Chinese')),
            'playback': bool(coach_tts_option('playback', True)),
            'max_pending': max(1, int(coach_tts_option('max_pending', 2))),
            'concise_speech': bool(coach_tts_option('concise_speech', True)),
            'warmup': bool(coach_tts_option('warmup', True)),
            'startup_timeout_seconds': max(0.1, float(coach_tts_option('startup_timeout_seconds', 120))),
            'request_timeout_seconds': max(0.1, float(coach_tts_option('request_timeout_seconds', 30))),
            'streaming_interval_seconds': max(
                0.08,
                float(coach_tts_option('streaming_interval_seconds', 0.32)),
            ),
            'streaming_prebuffer_chunks': max(
                1,
                int(coach_tts_option('streaming_prebuffer_chunks', 2)),
            ),
        }
        if (
            self.deepseek_coach_options['enabled']
            or self.coach_tts_options['enabled']
        ):
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
            target_stream_id=self.stream_id,
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
        racket_model_metadata = {}
        if self.racket_pose_config['mode'] != 'legacy':
            from racket_pose_detector import model_digest
            try:
                racket_model_digest = model_digest(self.racket_pose_config['model_path'])
            except OSError:
                # Optional-model failure is handled by the isolated runtime, not Reader setup.
                racket_model_digest = None
            racket_model_metadata = {'racket_pose': {
                'path': self.racket_pose_config['model_path'],
                'sha256': racket_model_digest,
                'backend': self.racket_pose_config['backend'],
                'compute_units': self.racket_pose_config['compute_units'],
                'imgsz': self.racket_pose_config['imgsz'],
                'mode': self.racket_pose_config['mode'],
                'selection_policy': self.racket_pose_config.get('selection_policy','mixed'),
                'selection_policy_version': self.racket_pose_config.get('selection_policy_version'),
            }}
            self.racket_pose_journal = str(Path(self.config['video_output_path']).with_suffix('')) + '_racket_pose.jsonl'
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
                **racket_model_metadata,
            },
        )
        self.ground_reference = None
        ground = self.config.get('ground_reference') or {}
        if ground.get('calibration') is not None:
            from ground_reference import GroundReference
            from ground_calibration_store import source_binding
            binding = source_binding(self.video_path, ground.get('context_stream_id') or self.roi_profile.stream_id)
            self.ground_reference = GroundReference(ground['calibration'], binding, (self.width, self.height),
                                                    application=ground.get('application'))
            self.session_metadata['ground_calibration'] = self.ground_reference.calibration
            if ground.get('application'):
                self.session_metadata['ground_calibration_application'] = ground['application']
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
        self.analyzer_ready = mp.Event()
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

        if getattr(self, 'lossless_realtime', False):
            while not self.analyzer_ready.wait(timeout=.25):
                if self.stop_event.is_set():self.rd_done.set();return
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
            if self.live_mode and not getattr(self, 'lossless_realtime', False)
            else None
        )
        pacer = DeadlinePacer(
            self.fps if limit_reader_fps else 0.0,
            max_lag_intervals=max_lag_intervals,
        )
        frame_clock = SourceFrameClock()
        media_clock = SourceMediaClock(self.fps, "stream" if self.is_stream_source else "video_file")
        if self.live_mode:
            print(
                f"⚡ [Reader] 直播缓冲槽: {self.shm_num} | "
                f"推理邮箱: {1 if self.drop_stale_frames else self.shm_num} | "
                f"时间线节流: {'开启' if limit_reader_fps else '关闭'} | "
                f"{'允许丢旧帧' if self.drop_stale_frames else '连续处理，满槽等待'}"
            )
        elif limit_reader_fps:
            print(f"⏱️ [Reader] 绝对时间线节流: {self.fps:.2f} FPS")

        source_schedule_start = time.perf_counter()
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
                                # Discarded frames still consume source time. Without
                                # pacing, one transient stall fast-forwards local video.
                                delay = pacer.next_delay()
                                if delay > 0 and self.stop_event.wait(delay):
                                    break
                            elif not self.is_stream_source:
                                break
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
                source_time = media_clock.observe(
                    source_frame_id,
                    cap.get(cv2.CAP_PROP_POS_MSEC) if not self.is_stream_source else None,
                )
                shared_frames[slot][:] = frame[:]
                import struct
                struct.pack_into('<q', shms[slot].buf, self.frame_size, source_frame_id)
                self.q_inference.put({
                    'idx': source_frame_id,
                    'slot': slot,
                    'captured_at': captured_at,
                    'captured_at_unix_ns': captured_at_unix_ns,
                    'reader_schedule_lag_ms': max(0., (captured_at-source_schedule_start-source_frame_id/self.fps)*1000) if self.fps else None,
                    'source_time': source_time,
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
        return run_inference_process(self, ignore_child_interrupts_fn=ignore_child_interrupts)

    def analyzer_process(self):
        return run_analyzer_process(self, ignore_child_interrupts_fn=ignore_child_interrupts)

    def _process_entry(self, role):
        ignore_child_interrupts()
        threads = getattr(self, 'config', {}).get('pipeline_perf', {}).get('opencv_threads')
        if threads is not None:
            cv2.setNumThreads(max(0, int(threads)))
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
        return write_evidence_manifest(self, status)

    def run(self):
        objs = []
        try:
            for name in self.shm_names:
                objs.append(
                    shared_memory.SharedMemory(
                        name=name,
                        create=True,
                        size=self.frame_size + 8,
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


def main_cli(argv=None):
    return _pipeline_main_cli(argv=argv, pipeline_cls=MultiprocessPipeline)


if __name__ == "__main__":
    main_cli()
