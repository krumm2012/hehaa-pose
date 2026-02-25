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
import yaml
import os
import json
import argparse

# 导入分析组件
from yolo26n_unified_detector import YOLO26nUnifiedDetector
from pose_estimator_yolo26 import PoseEstimatorYOLO26
from full_swing_analyzer import FullSwingAnalyzer
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
        
        # 计算理论等待时间以维持原片节奏
        frame_interval = 1.0 / self.fps
        
        frame_id = 0
        while not self.stop_event.is_set():
            t_start = time.time()
            try:
                slot = self.q_free.get(timeout=2.0)
            except:
                print("⚠️ [Reader] 延迟积压，等待消费...")
                continue
                
            ret, frame = cap.read()
            if not ret:
                self.q_free.put(slot)
                break
            
            shared_frames[slot][:] = frame[:]
            self.q_inference.put({'idx': frame_id, 'slot': slot})
            frame_id += 1
            
            # 动态休眠以维持输出 FPS 稳定
            wait = frame_interval - (time.time() - t_start)
            if wait > 0: time.sleep(wait)
            
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
        executor = ThreadPoolExecutor(max_workers=2)
        self.inf_ready.set()
        
        while not self.stop_event.is_set():
            try:
                task = self.q_inference.get(timeout=1.0)
            except:
                if self.rd_done.is_set(): break
                continue
            
            slot = task['slot']
            frame_ptr = shared_frames[slot]
            
            # 使用 ThreadPool 同时驱动 ANE 和 GPU
            f1 = executor.submit(detector.detect_unified, frame_ptr)
            f2 = executor.submit(pose_estimator.get_keypoints, frame_ptr)
            
            ball, racket, _ = f1.result()
            pose = f2.result()
            
            self.q_analyzer.put({
                'id': task['idx'], 
                'slot': slot, 
                'ball': ball, 
                'racket': racket, 
                'pose': pose
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
        swing_analyzer = FullSwingAnalyzer(self.config)
        
        # 视频录制
        save_video = self.config.get('save_video', True)
        out_writer = None
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
        frame_results = []
        
        while not self.stop_event.is_set():
            try:
                data = self.q_analyzer.get(timeout=2.0)
            except:
                if self.rd_done.is_set(): break
                continue
            
            fid, slot = data['id'], data['slot']
            # 严格对齐图像
            canvas = shared_frames[slot].copy()
            
            ball_pos = data['ball'][0]['position'] if data['ball'] else None
            racket_list = data['racket']
            poses = data['pose']
            
            # --- 1. 深度分析计算 ---
            swing_label = "Scanning..."
            detailed_data = {}
            if poses:
                swing_label = PoseEstimatorYOLO26.classify_swing_static(poses[0], self.config)
                detailed_data = swing_analyzer.analyze_swing_components(poses, racket_list, ball_pos, (self.height, self.width))

            # --- 保存每帧数据 ---
            frame_results.append({
                'frame_id': fid,
                'timestamp': round(fid / self.fps, 3),
                'swing_type': swing_label,
                'ball': ball_pos,
                'rackets': racket_list,
                'pose': poses[0] if poses else None,
                'metrics': detailed_data
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
            y_ptr = 140
            for phase, metrics in detailed_data.items():
                for k, v in metrics.items():
                    # k 已经是英文 (如 shoulder_turn)，颜色区分保持
                    color = (0, 255, 0) if "angle" in k or "ext" in k else (220, 220, 220)
                    cv2.putText(canvas, f"{k}: {v}", (40, y_ptr), font, 0.5, color, 1)
                    y_ptr += 26
                    if y_ptr > 600: break

            # C. 绘制视觉元素 (骨架、球、球拍)
            canvas = PoseEstimatorYOLO26.draw_keypoints_static(canvas, poses)
            if ball_pos:
                cv2.circle(canvas, (int(ball_pos[0]), int(ball_pos[1])), 10, (0, 255, 255), -1)
                cv2.circle(canvas, (int(ball_pos[0]), int(ball_pos[1])), 12, (255, 255, 255), 2)
            
            for ra in racket_list:
                x1, y1, x2, y2 = ra['box']
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 128, 0), 2)

            # --- 3. 提交并释放 ---
            if out_writer: out_writer.write(canvas)
            
            # --- 新增: 实时双窗口对比显示 ---
            h, w = canvas.shape[:2]
            display_scale = 0.5  # 缩放至 50% 以便在屏幕显示
            
            # 缩放原始图和处理图
            orig_small = cv2.resize(shared_frames[slot], (int(w * display_scale), int(h * display_scale)))
            proc_small = cv2.resize(canvas, (int(w * display_scale), int(h * display_scale)))
            
            # 水平堆叠
            dual_view = np.hstack((orig_small, proc_small))
            
            # 为双视角增加文字标识
            cv2.putText(dual_view, "ORIGINAL", (20, 30), font, 1.0, (255, 255, 255), 2)
            cv2.putText(dual_view, "PROCESSED", (int(w * display_scale) + 20, 30), font, 1.0, (255, 255, 255), 2)
            
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
            # 保存 JSON 数据
            json_file = out_file.rsplit('.', 1)[0] + '.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'video_info': {
                        'path': self.video_path,
                        'fps': self.fps,
                        'resolution': [self.width, self.height],
                        'total_frames': count
                    },
                    'frames': frame_results
                }, f, indent=4, ensure_ascii=False)
            print(f"📊 [Analyzer] 数据已保存至: {json_file}")

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
