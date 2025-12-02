# main_optimized.py - 性能优化版本
import cv2
import yaml
from pose_estimator import PoseEstimator
from ball_tracker import BallTracker
from racket_detector import RacketDetector
from full_swing_analyzer import FullSwingAnalyzer
from head_replacement_processor import HeadReplacementProcessor
from roi_manager import ROIManager
from enhanced_motion_capture import EnhancedMotionCapture
import os
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

def load_config(config_path="configs/default_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_output_directory(output_path):
    """确保输出目录存在"""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_path

def put_chinese_text_cached(img, text, position, font_size=24, color=(255,255,255), font_cache=None):
    """带缓存的中文文本渲染，避免重复加载字体"""
    if font_cache is None:
        font_cache = {}
    
    # 使用字体大小作为缓存键
    if isinstance(font_size, float):
        if font_size <= 0.5:
            font_size_int = 18
        elif font_size <= 0.7:
            font_size_int = 22
        elif font_size <= 1.0:
            font_size_int = 26
        else:
            font_size_int = 32
    else:
        font_size_int = int(font_size)
    
    if font_size_int not in font_cache:
        font_path = "/System/Library/Fonts/STHeiti Light.ttc"
        font_cache[font_size_int] = ImageFont.truetype(font_path, font_size_int)
    
    font = font_cache[font_size_int]
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class OptimizedDetectionPipeline:
    """优化的检测流水线，支持并行处理和帧跳跃"""
    
    def __init__(self, config, roi_manager):
        self.config = config
        self.roi_manager = roi_manager
        self.font_cache = {}
        
        # 性能优化配置
        self.perf_config = config.get('performance_optimization', {})
        self.enable_parallel = self.perf_config.get('enable_parallel_detection', True)
        self.skip_frames = self.perf_config.get('skip_frames', 0)  # 每N帧跳过检测
        self.max_workers = self.perf_config.get('max_workers', min(4, mp.cpu_count()))
        self.enable_gpu_acceleration = self.perf_config.get('enable_gpu_acceleration', True)
        
        # 检测模块初始化
        print(f"🚀 优化模式启动 - 并行: {self.enable_parallel}, 跳帧: {self.skip_frames}, 工作线程: {self.max_workers}")
        
        self.pose_module = PoseEstimator(config['yolo_pose_model_path'], config, roi_manager)
        self.ball_module = BallTracker(config.get('tracknet_model_path', None), config, roi_manager)
        self.racket_module = RacketDetector(config['racket_yolo_model_path'], config, roi_manager)
        self.swing_analyzer = FullSwingAnalyzer(config)
        self.head_processor = HeadReplacementProcessor(config)
        
        # 结果缓存
        self.last_results = {
            'pose': None,
            'ball': None,
            'racket': None,
            'frame_num': -1
        }
        
        # 线程池
        if self.enable_parallel:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
    def detect_parallel(self, frame, frame_num):
        """并行检测处理"""
        if not self.enable_parallel:
            return self.detect_sequential(frame, frame_num)
        
        # 检查是否需要跳帧
        if self.skip_frames > 0 and frame_num % (self.skip_frames + 1) != 0:
            return self.last_results.copy()
        
        # 提交并行任务
        futures = {
            'pose': self.executor.submit(self.pose_module.get_keypoints, frame),
            'ball': self.executor.submit(self.ball_module.predict_ball, frame),
            'racket': self.executor.submit(self.racket_module.detect_rackets, frame)
        }
        
        # 收集结果
        results = {}
        for key, future in futures.items():
            try:
                results[key] = future.result(timeout=1.0)  # 1秒超时
            except Exception as e:
                print(f"⚠️ {key}检测超时或失败: {e}")
                results[key] = self.last_results.get(key)
        
        results['frame_num'] = frame_num
        self.last_results = results.copy()
        return results
    
    def detect_sequential(self, frame, frame_num):
        """顺序检测处理（回退方案）"""
        # 检查是否需要跳帧
        if self.skip_frames > 0 and frame_num % (self.skip_frames + 1) != 0:
            return self.last_results.copy()
        
        results = {
            'pose': self.pose_module.get_keypoints(frame),
            'ball': self.ball_module.predict_ball(frame),
            'racket': self.racket_module.detect_rackets(frame),
            'frame_num': frame_num
        }
        
        self.last_results = results.copy()
        return results
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

class OptimizedRenderer:
    """优化的渲染器，减少重复计算"""
    
    def __init__(self, config):
        self.config = config
        self.display_opts = config.get('display_options', {})
        self.roi_settings = config.get('roi_settings', {})
        self.font_cache = {}
        
        # 预计算信息面板模板
        self.panel_template = None
        self.panel_alpha = 0.7
        
    def create_info_panel_template(self, frame_width, frame_height):
        """创建信息面板模板（一次性计算）"""
        if self.panel_template is not None:
            return self.panel_template
        
        panel = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        panel_height = 400
        cv2.rectangle(panel, (30, 20), (330, panel_height), (0, 0, 0), -1)
        self.panel_template = panel
        return panel
    
    def render_optimized(self, frame, results, roi_manager, frame_num):
        """优化的渲染流程"""
        display_frame = frame.copy()
        
        # 1. 头像替换（如果启用）
        if self.config.get('face_replacement', {}).get('enabled', False):
            display_frame = results.get('head_replacement', display_frame)
        
        # 2. 绘制检测结果
        pose_results = results.get('pose')
        ball_positions = results.get('ball')
        racket_results = results.get('racket')
        
        # 姿态关键点
        if pose_results and self.display_opts.get('show_pose_keypoints', True):
            # 直接在display_frame上绘制，避免额外复制
            from pose_estimator import PoseEstimator
            PoseEstimator.draw_keypoints_static(display_frame, pose_results)
        
        # 球位置
        ball_position = ball_positions[0] if ball_positions else None
        if ball_position and self.display_opts.get('show_ball_position', True):
            cv2.circle(display_frame, (int(ball_position[0]), int(ball_position[1])), 
                      5, (0, 255, 0), -1)
        
        # 球拍检测
        if racket_results and self.display_opts.get('show_racket_state', True):
            for racket in racket_results:
                if isinstance(racket, dict) and 'box' in racket:
                    box = racket['box']
                    cv2.rectangle(display_frame, (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), (255, 0, 0), 2)
        
        # 3. ROI渲染
        if roi_manager.is_roi_set and self.roi_settings.get('visualization', {}).get('show_roi_boundary', True):
            display_frame = roi_manager.draw_roi(display_frame, 
                                               self.roi_settings.get('visualization', {}).get('show_roi_fill', True))
            
            # 高亮检测结果
            if self.roi_settings.get('visualization', {}).get('highlight_detections', True):
                if ball_positions:
                    display_frame = roi_manager.highlight_roi_detections(display_frame, ball_positions, "ball")
                if racket_results:
                    display_frame = roi_manager.highlight_roi_detections(display_frame, racket_results, "racket")
        
        # 4. 轻量级信息显示
        if self.display_opts.get('show_fps', False) or roi_manager.is_roi_set:
            self.render_minimal_info(display_frame, results, roi_manager, frame_num)
        
        return display_frame
    
    def render_minimal_info(self, display_frame, results, roi_manager, frame_num):
        """渲染最小化的信息显示"""
        frame_height = display_frame.shape[0]
        
        # ROI状态信息
        if roi_manager.is_roi_set:
            roi_info = f"ROI: Active ({len(roi_manager.roi_points)} points)"
            cv2.putText(display_frame, roi_info, (10, frame_height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 检测统计
            pose_count = len(results.get('pose', []))
            ball_count = len(results.get('ball', []))
            racket_count = len(results.get('racket', []))
            detection_info = f"P:{pose_count} B:{ball_count} R:{racket_count}"
            cv2.putText(display_frame, detection_info, (10, frame_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main_optimized(config_path="configs/roi_enabled_config.yaml", input_path: str = None, output_path: str = None):
    """优化版本的主处理函数"""
    print("🚀 启动优化版本视频处理")
    print("=" * 50)
    
    # 加载配置
    config = load_config(config_path)
    if input_path:
        config['video_input_path'] = input_path
    if output_path:
        config['video_output_path'] = output_path
    video_path = config['video_input_path']
    
    # 添加性能优化配置（如果不存在）
    if 'performance_optimization' not in config:
        config['performance_optimization'] = {
            'enable_parallel_detection': True,
            'skip_frames': 0,  # 0表示不跳帧，1表示每隔1帧处理一次
            'max_workers': min(4, mp.cpu_count()),
            'enable_gpu_acceleration': True,
            'buffer_size': 30,  # 视频缓冲帧数
            'minimal_ui': True   # 简化UI以提高性能
        }
    
    perf_config = config['performance_optimization']
    print(f"📊 性能设置:")
    print(f"   - 并行检测: {perf_config.get('enable_parallel_detection', True)}")
    print(f"   - 跳帧设置: {perf_config.get('skip_frames', 0)}")
    print(f"   - 工作线程: {perf_config.get('max_workers', 4)}")
    print(f"   - 简化UI: {perf_config.get('minimal_ui', True)}")
    
    # 视频初始化
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 视频信息: {frame_width}x{frame_height}, {fps}FPS, {total_frames}帧")
    
    # 输出视频设置
    output_path = create_output_directory(config['video_output_path'])
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'),
                          fps, (frame_width, frame_height))
    
    # ROI初始化
    print("🎯 初始化ROI管理器...")
    roi_manager = ROIManager(config)
    roi_settings = config.get('roi_settings', {})
    
    if roi_settings.get('enabled', False):
        roi_config_path = roi_settings.get('roi_config_path', 'configs/roi_config.yaml')
        if os.path.exists(roi_config_path):
            if roi_manager.load_roi_config(roi_config_path):
                print("✅ ROI配置加载成功")
    
    # 初始化优化的处理流水线
    print("🔧 初始化优化处理流水线...")
    detection_pipeline = OptimizedDetectionPipeline(config, roi_manager)
    renderer = OptimizedRenderer(config)
    
    # 性能统计
    frame_num = 0
    start_time = time.time()
    detection_times = []
    render_times = []
    total_detection_time = 0
    total_render_time = 0
    
    print("🎬 开始优化视频处理...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测阶段计时
            detection_start = time.time()
            if perf_config.get('enable_parallel_detection', True):
                results = detection_pipeline.detect_parallel(frame, frame_num)
            else:
                results = detection_pipeline.detect_sequential(frame, frame_num)
            detection_time = time.time() - detection_start
            total_detection_time += detection_time
            
            # 渲染阶段计时
            render_start = time.time()
            display_frame = renderer.render_optimized(frame, results, roi_manager, frame_num)
            render_time = time.time() - render_start
            total_render_time += render_time
            
            # 写入视频
            if out:
                out.write(display_frame)
            
            # 显示窗口（可选）
            if not perf_config.get('headless', False):
                cv2.imshow("Tennis Analysis - Optimized", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 性能监控
            if frame_num % 30 == 0:  # 每30帧显示一次
                elapsed_time = time.time() - start_time
                current_fps = (frame_num + 1) / elapsed_time if elapsed_time > 0 else 0
                progress = (frame_num + 1) / total_frames * 100 if total_frames > 0 else 0
                
                avg_detection_time = total_detection_time / (frame_num + 1) * 1000
                avg_render_time = total_render_time / (frame_num + 1) * 1000
                
                print(f"📊 帧 {frame_num+1}/{total_frames} ({progress:.1f}%) - "
                      f"FPS: {current_fps:.1f} - "
                      f"检测: {avg_detection_time:.1f}ms - "
                      f"渲染: {avg_render_time:.1f}ms")
            
            frame_num += 1
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断处理")
    
    finally:
        # 清理资源
        detection_pipeline.cleanup()
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # 最终性能报告
        total_time = time.time() - start_time
        avg_fps = frame_num / total_time if total_time > 0 else 0
        
        print("\n🏁 优化处理完成!")
        print(f"📊 性能报告:")
        print(f"   - 处理帧数: {frame_num}")
        print(f"   - 总时间: {total_time:.2f}秒")
        print(f"   - 平均FPS: {avg_fps:.2f}")
        if frame_num > 0:
            print(f"   - 平均检测时间: {total_detection_time/frame_num*1000:.1f}ms/帧")
            print(f"   - 平均渲染时间: {total_render_time/frame_num*1000:.1f}ms/帧")
        else:
            print("   - 平均检测时间: N/A (未处理帧)")
            print("   - 平均渲染时间: N/A (未处理帧)")
        
        # 性能改进建议
        if avg_fps < 10:
            print("💡 性能优化建议:")
            print("   - 增加 skip_frames 参数（跳帧处理）")
            print("   - 启用 minimal_ui 模式")
            print("   - 考虑降低视频分辨率")
        elif avg_fps < 20:
            print("💡 可以考虑启用更多并行优化")

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description='优化版网球分析系统')
    parser.add_argument('--config', '-c', default='configs/roi_enabled_config.yaml', help='配置文件路径')
    parser.add_argument('--input', '-i', default=None, help='输入视频路径或URL，覆盖配置文件')
    parser.add_argument('--output', '-o', default=None, help='输出视频完整路径，覆盖配置文件')
    parser.add_argument('--output_dir', default=None, help='输出目录（与输入同名文件）')
    args = parser.parse_args()

    final_output = args.output
    if not final_output and args.output_dir:
        in_base = os.path.basename(args.input) if args.input else None
        if not in_base:
            in_base = os.path.basename(load_config(args.config)['video_input_path'])
        if not in_base:
            in_base = 'output_video.mp4'
        elif not os.path.splitext(in_base)[1]:
            in_base = f"{in_base}.mp4"
        final_output = os.path.join(args.output_dir, in_base)

    main_optimized(args.config, input_path=args.input, output_path=final_output)
