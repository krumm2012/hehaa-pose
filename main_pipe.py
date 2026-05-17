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

# 导入分析组件
from yolo26n_unified_detector import YOLO26nUnifiedDetector
from pose_estimator_yolo26 import PoseEstimatorYOLO26
from frame_processor import FrameProcessor
from main import put_chinese_text, create_output_directory
from roi_manager import ROIManager
from speed_analyzer import SpeedAnalyzer
from hit_zone_analyzer import HitZoneAnalyzer

class MultiprocessPipeline:
    def __init__(self, config_path, input_path=None, output_path=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 命令行参数覆盖配置
        if input_path:
            self.config['video_input_path'] = input_path
        if output_path:
            self.config['video_output_path'] = output_path

        self.video_path = self.config['video_input_path']

        cap = cv2.VideoCapture(self.video_path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        # --- 共享内存池 ---
        self.shm_num = 12
        self.shm_names = [f"tennis_shm_v2_{i}" for i in range(self.shm_num)]
        self.frame_size = self.height * self.width * 3

        # 同步原语
        self.q_free = mp.Queue(maxsize=self.shm_num)
        for i in range(self.shm_num): self.q_free.put(i)

        self.q_inference = mp.Queue(maxsize=self.shm_num)
        self.q_analyzer = mp.Queue(maxsize=self.shm_num)

        self.stop_event = mp.Event()
        self.inf_ready = mp.Event()
        self.rd_done = mp.Event()

    def reader_process(self):
        """进程 1: 稳定解码流"""
        print("🚀 [Reader] 等待 AI 载入...")
        self.inf_ready.wait()

        cap = cv2.VideoCapture(self.video_path)
        shms = [shared_memory.SharedMemory(name=name) for name in self.shm_names]
        shared_frames = [np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=s.buf) for s in shms]

        # 计算理论等待时间以维持原片节奏（可关闭以追求最高吞吐）
        perf_cfg = self.config.get('pipeline_perf', {})
        limit_reader_fps = bool(perf_cfg.get('limit_reader_fps', True))
        frame_interval = 1.0 / self.fps if self.fps > 0 else 0.0

        frame_id = 0
        while not self.stop_event.is_set():
            t_start = time.time()
            try:
                slot = self.q_free.get(timeout=2.0)
            except queue.Empty:
                print("⚠️ [Reader] 延迟积压，等待消费...")
                continue
            except Exception as exc:
                print(f"❌ [Reader] 获取空闲缓冲失败: {exc}")
                self.stop_event.set()
                break

            ret, frame = cap.read()
            if not ret:
                self.q_free.put(slot)
                break

            shared_frames[slot][:] = frame[:]
            self.q_inference.put({'idx': frame_id, 'slot': slot})
            frame_id += 1

            # 动态休眠以维持输出 FPS 稳定（吞吐模式下关闭）
            if limit_reader_fps:
                wait = frame_interval - (time.time() - t_start)
                if wait > 0:
                    time.sleep(wait)

        cap.release()
        for s in shms: s.close()
        self.rd_done.set()
        print(f"✅ [Reader] 结束，共解析 {frame_id} 帧")

    def inference_process(self):
        """进程 2: AI 推理核心 (平行调度)"""
        print("🚀 [Inference] 加载 Core ML 并行架构...")
        from concurrent.futures import ThreadPoolExecutor
        detector = YOLO26nUnifiedDetector(self.config['unified_detection']['model_path'], self.config['unified_detection'])
        pose_estimator = PoseEstimatorYOLO26(self.config['yolo_pose_model_path'], self.config)

        shms = [shared_memory.SharedMemory(name=name) for name in self.shm_names]
        shared_frames = [np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=s.buf) for s in shms]
        perf_cfg = self.config.get('pipeline_perf', {})
        unified_workers = (
            self.config.get('unified_detection', {}).get('max_workers')
            or self.config.get('performance_optimization', {}).get('max_workers')
            or 2
        )
        inference_workers = max(1, int(perf_cfg.get('inference_workers', unified_workers)))
        executor = ThreadPoolExecutor(max_workers=inference_workers)
        print(f"⚙️ [Inference] 线程池并发数: {inference_workers}")
        self.inf_ready.set()

        while not self.stop_event.is_set():
            try:
                task = self.q_inference.get(timeout=1.0)
            except queue.Empty:
                if self.rd_done.is_set(): break
                continue
            except Exception as exc:
                print(f"❌ [Inference] 获取推理任务失败: {exc}")
                self.stop_event.set()
                break

            slot = task['slot']
            frame_ptr = shared_frames[slot]

            # 使用 ThreadPool 同时驱动 ANE 和 GPU
            f1 = executor.submit(detector.detect_unified, frame_ptr)
            f2 = executor.submit(pose_estimator.get_keypoints, frame_ptr)

            ball, racket, _ = f1.result()
            ball_diagnostics = detector.get_last_ball_diagnostics()
            pose = f2.result()

            self.q_analyzer.put({
                'id': task['idx'],
                'slot': slot,
                'ball': ball,
                'racket': racket,
                'pose': pose,
                'ball_diagnostics': ball_diagnostics,
            })

        for s in shms: s.close()
        executor.shutdown()
        print("✅ [Inference] 退出")

    def analyzer_process(self):
        """进程 3: 业务核心 + 全视觉渲染"""
        print("🚀 [Analyzer] 初始化高清渲染引擎...")

        # 组件初始化
        hit_cfg = self.config.get('hit_zone_analysis', {})
        hit_analyzer = HitZoneAnalyzer(sweet_spot_ratio=hit_cfg.get('sweet_spot_ratio', 0.3))
        # 视频录制
        save_video = self.config.get('save_video', True)
        out_writer = None
        out_file = None
        if save_video:
            out_file = create_output_directory(self.config['video_output_path'])
            out_writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'avc1'), self.fps, (self.width, self.height))
            print(f"🎬 [Recorder] 录制中: {out_file}")

        shms = [shared_memory.SharedMemory(name=name) for name in self.shm_names]
        shared_frames = [np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=s.buf) for s in shms]

        last_ball_pos = None
        last_frame_id = -1
        t_start = time.time()
        count = 0
        # 性能开关（默认兼容原行为）
        perf_cfg = self.config.get('pipeline_perf', {})
        dual_view_enabled = bool(perf_cfg.get('dual_view_enabled', True))
        dual_view_scale = float(perf_cfg.get('dual_view_scale', 0.5))
        analysis_stride = max(1, int(perf_cfg.get('analysis_stride', 1)))
        draw_metrics_text = bool(perf_cfg.get('draw_metrics_text', True))
        collect_frame_results = bool(perf_cfg.get('collect_frame_results', True))
        json_dump_indent = perf_cfg.get('json_dump_indent', 4)
        frame_results_path = None
        diagnostics_path = None
        frame_results_fp = None
        diagnostics_records = []
        first_frame_record = True
        if collect_frame_results:
            output_base = out_file if out_file else create_output_directory(self.config['video_output_path'])
            frame_results_path = os.path.splitext(output_base)[0] + '.json'
            diagnostics_path = os.path.splitext(output_base)[0] + '_diagnostics.json'
            frame_results_fp = open(frame_results_path, 'w', encoding='utf-8')
            frame_results_fp.write('{\n')
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

        while not self.stop_event.is_set():
            try:
                data = self.q_analyzer.get(timeout=2.0)
            except queue.Empty:
                if self.rd_done.is_set(): break
                continue
            except Exception as exc:
                print(f"❌ [Analyzer] 获取分析任务失败: {exc}")
                self.stop_event.set()
                break

            fid, slot = data['id'], data['slot']
            # 严格对齐图像
            canvas = shared_frames[slot].copy()

            ball_pos = data['ball'][0]['position'] if data['ball'] else None
            racket_list = data['racket']
            poses = data['pose']

            # --- 1. 深度分析计算 ---
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
            if collect_frame_results:
                frame_record = frame_processor.build_frame_record(
                    frame_id=fid,
                    swing_type=swing_label,
                    ball_position=norm_ball_pos,
                    racket_detections=racket_list,
                    poses=poses,
                    phase_metrics=detailed_data,
                )
                frame_record["detection_diagnostics"] = data.get("ball_diagnostics") or {}
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
            canvas = PoseEstimatorYOLO26.draw_keypoints_static(canvas, poses)
            if norm_ball_pos:
                cv2.circle(canvas, (int(norm_ball_pos[0]), int(norm_ball_pos[1])), 10, (0, 255, 255), -1)
                cv2.circle(canvas, (int(norm_ball_pos[0]), int(norm_ball_pos[1])), 12, (255, 255, 255), 2)

            for ra in racket_list:
                x1, y1, x2, y2 = ra['box']
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 128, 0), 2)

            # --- 3. 提交并释放 ---
            if out_writer: out_writer.write(canvas)

            # --- 新增: 实时双窗口对比显示 ---
            if dual_view_enabled:
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

            count += 1
            if count % 25 == 0:
                fps = count / (time.time() - t_start)
                print(f"📊 [Sync-Analyzer] Processing Frame {fid} | FPS: {fps:.2f}")

        if out_writer:
            out_writer.release()

        if collect_frame_results and frame_results_fp is not None:
            frame_results_fp.write('\n  ],\n')
            frame_results_fp.write(f'  "summary": {json.dumps({"total_frames": count}, ensure_ascii=False, indent=json_dump_indent)}\n')
            frame_results_fp.write('}\n')
            frame_results_fp.close()
            print(f"📊 [Analyzer] 数据已保存至: {frame_results_path}")
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
        print("✅ [Analyzer] Exit and saved video")
        self.stop_event.set()

    def run(self):
        objs = []
        for name in self.shm_names:
            try: objs.append(shared_memory.SharedMemory(name=name, create=True, size=self.frame_size))
            except: objs.append(shared_memory.SharedMemory(name=name))

        ps = [mp.Process(target=self.reader_process), mp.Process(target=self.inference_process), mp.Process(target=self.analyzer_process)]
        for p in ps: p.start()
        try:
            for p in ps: p.join()
        except KeyboardInterrupt:
            self.stop_event.set()
            for p in ps: p.terminate()
        for s in objs:
            s.close()
            try: s.unlink()
            except: pass
        print("🏁 任务流执行完毕")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='网球分析多进程流水线')
    parser.add_argument('--config', '-c', default='configs/yolo26_tennis_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--input', '-i', default=None, help='输入视频路径，覆盖配置文件')
    parser.add_argument('--output', '-o', default=None, help='输出视频路径，覆盖配置文件')

    args = parser.parse_args()

    # 如果没有指定输入且配置文件里也没有，给个默认值
    if not args.input:
        # 尝试从配置加载看有没有
        with open(args.config, 'r') as f:
            tmp_cfg = yaml.safe_load(f)
            if 'video_input_path' not in tmp_cfg:
                args.input = "data/16.10.mp4"

    MultiprocessPipeline(args.config, input_path=args.input, output_path=args.output).run()
