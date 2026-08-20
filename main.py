# main.py
import cv2
import yaml
from pose_estimator import PoseEstimator
from ball_tracker import BallTracker
from racket_detector import RacketDetector # <-- IMPORT
from frame_processor import FrameProcessor
from head_replacement_processor import HeadReplacementProcessor  # <-- NEW IMPORT
from roi_manager import ROIManager  # <-- ROI IMPORT
from enhanced_motion_capture import EnhancedMotionCapture  # <-- MOTION CAPTURE IMPORT
from async_detector import AsyncDetector  # <-- ASYNC DETECTOR
from speed_analyzer import SpeedAnalyzer  # <-- SPEED ANALYZER
from hit_zone_analyzer import HitZoneAnalyzer  # <-- HIT ZONE ANALYZER
from highlight_service import HighlightService
from detection_frame_context import DetectionFrameContext
from yolo26n_unified_detector import YOLO26nUnifiedDetector, BallDetectionWrapper, RacketDetectionWrapper  # <-- UNIFIED DETECTOR
from video_io_pipeline import VideoIOPipeline  # <-- I/O PIPELINE
import os
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache

def load_config(config_path="configs/default_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_output_directory(output_path):
    """确保输出目录存在"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_path

@lru_cache(maxsize=8)
def _load_chinese_font(font_size_int: int):
    font_path = "/System/Library/Fonts/STHeiti Light.ttc"
    try:
        return ImageFont.truetype(font_path, font_size_int)
    except Exception:
        # 回退到默认字体，确保无字体环境下不崩溃
        return ImageFont.load_default()

def put_chinese_text(img, text, position, font_size=24, color=(255,255,255)):
    # font_size 允许传 float，但自动映射为合适的字号
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
    font = _load_chinese_font(font_size_int)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def main(config_path="configs/default_config.yaml", input_path: str = None, output_path: str = None, original_name: str = None, output_dir: str = None):
    print("正在加载配置...")
    config = load_config(config_path)
    # 覆盖输入/输出路径（如通过命令行传入）
    if input_path:
        config['video_input_path'] = input_path
    if output_path:
        config['video_output_path'] = output_path
    # 如果指定了输出目录，更新高光文件输出目录
    if output_dir:
        if 'highlights' not in config:
            config['highlights'] = {}
        config['highlights']['output_dir'] = output_dir
        print(f"高光文件输出目录设置为: {output_dir}")

    video_path = config['video_input_path']
    # 如果提供了原始文件名，在日志中显示，否则显示实际路径
    display_name = original_name if original_name else video_path
    print(f"配置加载完成，视频路径: {video_path}")
    if original_name:
        print(f"处理文件: {original_name}")

    # 获取显示选项配置
    display_opts = config.get('display_options', {})

    # ========== 新增：手动画监控区域功能 ==========
    if config.get('enable_manual_boundary', False):
        print("手动画定监控区域模式已启用。即将弹出窗口，请用鼠标拖拽画出监控区域（矩形），松开左键确定。")
        cap_tmp = cv2.VideoCapture(video_path)
        ret, frame = cap_tmp.read()
        cap_tmp.release()
        if not ret:
            raise RuntimeError("无法读取视频第一帧，无法手动画定监控区域！")
        drawing = False
        ix, iy = -1, -1
        rect = [config.get('boundary_x1', 0), config.get('boundary_y1', 0), config.get('boundary_x2', 0), config.get('boundary_y2', 0)]
        temp_frame = frame.copy()
        def draw_rectangle(event, x, y, flags, param):
            nonlocal drawing, ix, iy, rect, temp_frame
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                img2 = frame.copy()
                cv2.rectangle(img2, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('请用鼠标画监控区域，松开左键确定', img2)
                temp_frame = img2
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                rect = [min(ix, x), min(iy, y), max(ix, x), max(iy, y)]
                img2 = frame.copy()
                cv2.rectangle(img2, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
                cv2.imshow('请用鼠标画监控区域，松开左键确定', img2)
                temp_frame = img2
        cv2.namedWindow('请用鼠标画监控区域，松开左键确定')
        cv2.setMouseCallback('请用鼠标画监控区域，松开左键确定', draw_rectangle)
        cv2.imshow('请用鼠标画监控区域，松开左键确定', frame)
        print("请用鼠标在弹出窗口画出监控区域（矩形），松开左键确定。画完后按任意键继续。")
        cv2.waitKey(0)
        cv2.destroyWindow('请用鼠标画监控区域，松开左键确定')
        print(f"你选择的监控区域为: {rect}")
        config['boundary_x1'], config['boundary_y1'], config['boundary_x2'], config['boundary_y2'] = rect
    # ========== 手动画监控区域功能结束 ==========

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 获取总帧数，处理潜在的 cv2.CAP_PROP_FRAME_COUNT 问题
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:  # 某些视频格式可能无法准确报告 total_frames
        print("警告: 无法获取视频总帧数，或总帧数为0。进度百分比可能不准确。")
        # 如果 fps 可用，基于较长的时长进行估计
        total_frames = fps * 3600 if fps > 0 else -1  # 默认为非常大的数字或 -1（如果 fps 也有问题）

    print(f"视频信息 - 宽度: {frame_width}, 高度: {frame_height}, FPS: {fps}, 总帧数 (估计): {total_frames if total_frames > 0 else 'N/A'}")

    # 确保输出目录存在
    save_video_enabled = config.get('save_video', True)
    out = None

    if save_video_enabled:
        output_path = create_output_directory(config['video_output_path'])
        if original_name:
            print(f"创建输出视频: {output_path} (来源: {original_name})")
        else:
            print(f"创建输出视频: {output_path}")

        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*'avc1'),
                              fps if fps > 0 else 25,  # 如果原始fps为0，提供默认值
                              (frame_width, frame_height))
    else:
        print("💡 [配置] 已禁用视频保存，处理完成后将不会产生输出视频。")

    # 精彩瞬间配置
    highlights_cfg = dict(config.get('highlights', {}))
    highlights_cfg.setdefault('ball_radius_px', config.get('ball_radius_px', 10))

    # Headless/最小化UI与最大帧数快速验证
    perf_opts = config.get('performance_optimization', {})
    headless = bool(perf_opts.get('headless', False))
    minimal_ui = bool(perf_opts.get('minimal_ui', False))
    vp_opts = config.get('video_processing', {})
    max_frames_limit = int(vp_opts.get('max_frames', 0)) if vp_opts.get('max_frames', 0) else 0

    # 🎯 初始化ROI管理器
    print("🎯 初始化ROI管理器...")
    roi_manager = ROIManager(config)
    print("✅ ROI管理器创建完成")

    print(f"ROI设置状态: {config.get('roi_settings', {})}")

    # 🔍 调试：详细检查ROI配置
    roi_settings = config.get('roi_settings', {})
    print(f"🔍 [调试] 详细ROI配置:")
    print(f"   - enabled: {roi_settings.get('enabled', 'NOT_FOUND')}")
    print(f"   - interactive_selection: {roi_settings.get('interactive_selection', 'NOT_FOUND')}")
    print(f"   - auto_load_config: {roi_settings.get('auto_load_config', 'NOT_FOUND')}")

    # 检查是否需要ROI交互式选择
    if roi_settings.get('enabled', False):
        if roi_settings.get('auto_load_config', True):
            roi_config_path = roi_settings.get('roi_config_path', 'configs/roi_config.yaml')
            if os.path.exists(roi_config_path):
                if roi_manager.load_roi_config(roi_config_path):
                    print("✅ ROI配置已从文件加载")
                else:
                    print("⚠️ ROI配置文件存在但未启用或无效，将启动交互式选择")

        # 如果没有ROI或需要交互式选择
        if not roi_manager.is_roi_set and roi_settings.get('interactive_selection', True):
            print("🎯 启动ROI交互式选择...")
            # 读取第一帧用于ROI选择
            cap_tmp = cv2.VideoCapture(video_path)
            ret, first_frame = cap_tmp.read()
            cap_tmp.release()

            if ret:
                selected_points = roi_manager.interactive_roi_selection(first_frame, "选择网球场兴趣区域")
                if selected_points and len(selected_points) == 4:
                    print(f"✅ ROI选择完成: {selected_points}")
                    # 保存ROI配置
                    roi_manager.save_roi_config(roi_settings.get('roi_config_path', 'configs/roi_config.yaml'))
                else:
                    print("⚠️ ROI选择取消或无效")
            else:
                print("❌ 无法读取视频第一帧进行ROI选择")
    else:
        print("🎯 ROI功能未启用")

    # 显示最终ROI状态
    print(f"🎯 ROI最终状态: {'已设置' if roi_manager.is_roi_set else '未设置'}")
    if roi_manager.is_roi_set:
        roi_stats = roi_manager.get_roi_stats()
        print(f"   ROI面积: {roi_stats.get('roi_area', 0):.0f} 像素²")
        print(f"   ROI点数: {len(roi_manager.roi_points)}")

    # 🎬 初始化增强动作捕捉
    motion_capture = None
    if config.get('roi_motion_capture', {}).get('enabled', False):
        print("初始化增强动作捕捉模块...")
        motion_capture = EnhancedMotionCapture(config)

    # 初始化组件
    print("🤖 初始化姿势估计模块...")
    pose_module = PoseEstimator(config['yolo_pose_model_path'], config, roi_manager)
    print("✅ 姿势估计模块初始化完成")

    # 🎯 检查是否启用统一检测
    unified_config = config.get('unified_detection', {})
    use_unified_detection = unified_config.get('enabled', False)

    if use_unified_detection:
        print("� 初始化 YOLO26n 统一检测器...")
        unified_detector = YOLO26nUnifiedDetector(
            unified_config.get('model_path', 'yolo26n.mlpackage'),
            unified_config,
            roi_manager
        )
        # 创建兼容包装器
        ball_module = BallDetectionWrapper(unified_detector)
        racket_module = RacketDetectionWrapper(unified_detector)
        print("✅ 统一检测器初始化完成")
    else:
        print("�🎾 初始化球追踪模块...")
        ball_module = BallTracker(config.get('tracknet_model_path', None), config, roi_manager)
        print("✅ 球追踪模块初始化完成")

        print("🏓 初始化球拍检测模块...")
        racket_module = RacketDetector(config['racket_yolo_model_path'], config, roi_manager)
        print("✅ 球拍检测模块初始化完成")

    analysis_stride = max(1, int(config.get('pipeline_perf', {}).get('analysis_stride', 1)))

    # 🚀 初始化异步检测器（如果未使用统一检测）
    async_config = config.get('async_detection', {})
    if async_config.get('enabled', True) and not use_unified_detection:
        print("🚀 初始化异步检测器...")
        async_detector = AsyncDetector(max_workers=async_config.get('max_workers', 3))
        print("✅ 异步检测器初始化完成")
        use_async = True
    else:
        use_async = False
        if use_unified_detection:
            print("ℹ️  使用统一检测，异步检测已禁用")
        else:
            print("⚠️ 异步检测已禁用，使用顺序检测")

    # ⚡ 初始化速度分析器
    speed_config = config.get('speed_analysis', {})
    if speed_config.get('enabled', True):
        print("⚡ 初始化速度分析器...")
        speed_analyzer = SpeedAnalyzer(
            fps=fps,
            pixel_to_meter=speed_config.get('pixel_to_meter', 0.01)
        )
        print("✅ 速度分析器初始化完成")
        use_speed_analysis = True
    else:
        use_speed_analysis = False
        speed_analyzer = None
        print("⚠️ 速度分析已禁用")

    # 🎯 初始化击球点分析器
    hit_zone_config = config.get('hit_zone_analysis', {})
    if hit_zone_config.get('enabled', True):
        print("🎯 初始化击球点分析器...")
        hit_zone_analyzer = HitZoneAnalyzer(
            sweet_spot_ratio=hit_zone_config.get('sweet_spot_ratio', 0.3)
        )
        print("✅ 击球点分析器初始化完成")
        use_hit_zone_analysis = True
    else:
        use_hit_zone_analysis = False
        hit_zone_analyzer = None
        print("⚠️ 击球点分析已禁用")

    print("🏸 初始化完整挥拍分析模块...")
    frame_processor = FrameProcessor(
        config=config,
        frame_dimensions=(frame_height, frame_width),
        fps=fps,
        analysis_stride=analysis_stride,
        speed_analyzer=speed_analyzer,
        hit_zone_analyzer=hit_zone_analyzer,
        enable_speed=use_speed_analysis,
        enable_hit_zone=use_hit_zone_analysis,
    )
    print(f"✅ 完整挥拍分析模块初始化完成 (analysis_stride={analysis_stride})")
    highlight_service = HighlightService(highlights_cfg=highlights_cfg, fps=fps)

    print("🎬 开始视频处理循环...")

    # 初始化头像替换处理器
    print("初始化头像替换模块...")
    head_processor = HeadReplacementProcessor(config)
    status = head_processor.get_status()
    print(f"头像替换状态: {status}")
    if status['enabled']:
        print(f"检测方法: {status['current_detection_method']}")
        # 从配置中获取混合模式
        blend_mode = config.get('face_replacement', {}).get('blend_mode', 'alpha')
        print(f"混合模式: {blend_mode}")
        # 检查Judy头像是否加载
        judy_loaded = config.get('face_replacement', {}).get('judy_head_image_path', '') != ''
        print(f"Judy头像路径已配置: {judy_loaded}")

    # 进度和时间跟踪
    frame_num = 0
    start_time = time.time()
    batch_start_time = start_time

    print("开始处理视频...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频处理完成!")
            break

        # 精彩瞬间上下文缓存与延迟写出
        highlight_service.add_recent_frame(frame_num, frame)
        highlight_service.process_pending_clips(frame_num, frame)

        # 显示进度
        if total_frames > 0 and frame_num % (fps if fps > 0 else 30) == 0:  # 大约每秒打印一次
            elapsed_time = time.time() - start_time
            batch_time = time.time() - batch_start_time
            batch_start_time = time.time()

            frames_processed = frame_num + 1
            progress = frames_processed / total_frames * 100 if total_frames > 0 else 0

            # 估计剩余时间
            if frame_num > 0:
                time_per_frame = elapsed_time / frames_processed
                frames_remaining = total_frames - frames_processed
                estimated_time_remaining = frames_remaining * time_per_frame if total_frames > 0 else "未知"

                print(f"处理帧 {frames_processed}/{total_frames} ({progress:.1f}%) - "
                      f"批处理时间: {batch_time:.2f}秒, "
                      f"估计剩余时间: {estimated_time_remaining if isinstance(estimated_time_remaining, str) else f'{estimated_time_remaining:.1f}秒'}")

        elif frame_num % 100 == 0:  # 如果total_frames未知，则每100帧显示一次
            print(f"处理帧 {frame_num}...")

        # 🎯 ROI预处理与检测帧上下文
        display_frame = frame.copy()
        det_ctx = DetectionFrameContext.build(
            frame_num=frame_num,
            frame=frame,
            roi_manager=roi_manager,
            config=config,
        )
        detection_frame = det_ctx.detection_frame
        pose_detection_frame = det_ctx.pose_detection_frame
        roi_offset = det_ctx.roi_offset

        # 🚀 异步并行检测（姿态、球、球拍）
        if use_async:
            detection_results = async_detector.detect_async(
                frame=pose_detection_frame,
                roi_frame=detection_frame,
                roi_offset=roi_offset,
                pose_estimator=pose_module,
                ball_tracker=ball_module,
                racket_detector=racket_module,
                frame_count=frame_num
            )
            pose_results = detection_results['keypoints']
            ball_positions = detection_results['balls']
            racket_detections = detection_results['rackets']

            # 显示性能信息
            if frame_num % 100 == 0:
                timing = detection_results['timing']
                print(f"⚡ [帧{frame_num}] 异步检测: {timing['total']*1000:.1f}ms "
                      f"(姿态:{timing['pose']*1000:.1f}ms, "
                      f"球:{timing['ball']*1000:.1f}ms, "
                      f"球拍:{timing['racket']*1000:.1f}ms, "
                      f"效率:{timing['parallel_efficiency']:.2f}x)")
        else:
            # 顺序检测（原方法）
            pose_results = pose_module.get_keypoints(pose_detection_frame)
            if frame_num % 30 == 0:
                print(f"🎾 [帧{frame_num}] 开始球检测，检测区域: {detection_frame.shape}")
            ball_positions = ball_module.predict_ball(detection_frame)
            racket_detections = racket_module.detect_rackets(detection_frame)

        # ROI坐标回填 + ROI内球过滤
        pose_results, ball_positions, racket_detections = det_ctx.adjust_detections(
            pose_results=pose_results,
            ball_positions=ball_positions,
            racket_detections=racket_detections,
        )

        if pose_results and display_opts.get('show_pose_keypoints', True):
            display_frame = pose_module.draw_keypoints(display_frame, pose_results)

        # 🎾 处理球追踪结果
        # 使用仅含 ROI 内候选的结果进行轨迹追踪与选球
        active_balls = ball_module.advanced_ball_processing(ball_positions or [], frame_num)
        ball_position = active_balls[0] if active_balls else (ball_positions[0] if ball_positions else None)

        if ball_position:
            if display_opts.get('show_ball_position', True):
                cv2.circle(display_frame, (int(ball_position[0]), int(ball_position[1])),
                          5, (0, 255, 0), -1)

            if display_opts.get('show_ball_trajectory', True) and config.get('ball_tracking_enabled', True):
                # 绘制球的轨迹（仅在追踪启用时）
                ball_module.draw_trajectory(display_frame)

        frame_analysis = frame_processor.process(
            frame_id=frame_num,
            poses=pose_results if pose_results else [],
            racket_detections=racket_detections if racket_detections else [],
            ball_position=(ball_position[0], ball_position[1]) if ball_position else None,
        )
        swing_type = frame_analysis["swing_type"]
        phase_est = frame_analysis["phase_metrics"]

        # 🏃‍♂️ 速度分析（由 FrameProcessor 统一计算）
        ball_speed_kmh = frame_analysis["speed_kmh"]
        if ball_speed_kmh is not None and ball_speed_kmh > 0 and ball_position:
            if display_opts.get('show_ball_speed', True):
                cv2.putText(display_frame, f"{ball_speed_kmh:.1f} km/h",
                            (int(ball_position[0]) + 15, int(ball_position[1]) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 🎯 击球点分析（由 FrameProcessor 统一计算）
        hit_analysis = frame_analysis["hit_analysis"]
        if hit_analysis and ball_position and display_opts.get('show_hit_quality', True):
            quality_text = f"Q:{hit_analysis['quality']*100:.0f}%"
            color = (0, 255, 0) if hit_analysis['quality'] > 0.7 else (0, 165, 255)
            cv2.putText(display_frame, quality_text,
                        (int(ball_position[0]) + 15, int(ball_position[1]) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ⭐ 精彩瞬间检测与触发（由独立 service 管理）
        highlight_service.process_frame(
            frame_num=frame_num,
            display_frame=display_frame,
            ball_position=(ball_position[0], ball_position[1]) if ball_position else None,
            racket_detections=racket_detections if racket_detections else [],
            ball_module=ball_module,
            pose_results=pose_results if pose_results else [],
            frame_processor=frame_processor,
        )

        if racket_detections and display_opts.get('show_racket_state', True):
            # 绘制球拍状态
            for racket in racket_detections:
                if isinstance(racket, dict) and 'box' in racket:
                    box = racket['box']
                    cv2.rectangle(display_frame, (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])), (255, 0, 0), 2)

        # 🎬 ROI增强动作捕捉
        if motion_capture and roi_manager.is_roi_set:
            # 分析ROI内的动作
            motion_analysis = motion_capture.analyze_roi_motion(
                pose_results if pose_results else [],
                ball_positions if ball_positions else [],
                racket_detections if racket_detections else [],
                frame_num
            )

            # 绘制动作分析结果
            if roi_settings.get('visualization', {}).get('show_roi_stats', True):
                display_frame = motion_capture.draw_motion_analysis(display_frame, motion_analysis)

        # 🎯 绘制ROI
        if roi_manager.is_roi_set and roi_settings.get('visualization', {}).get('show_roi_boundary', True):
            display_frame = roi_manager.draw_roi(display_frame,
                                               roi_settings.get('visualization', {}).get('show_roi_fill', True))

            # 高亮ROI内的检测结果
            if roi_settings.get('visualization', {}).get('highlight_detections', True):
                if ball_positions:
                    display_frame = roi_manager.highlight_roi_detections(display_frame, ball_positions, "ball")
                if racket_detections:
                    display_frame = roi_manager.highlight_roi_detections(display_frame, racket_detections, "racket")

        # 显示挥拍类型
        if pose_results and display_opts.get('show_swing_type', True):
            if swing_type != "No Pose" and swing_type != "Incomplete Pose":
                display_frame = put_chinese_text(display_frame, f"Swing Type: {swing_type}", (10, 30), 0.7, (255, 255, 255))

        # 显示边界框
        if config['use_boundary'] and display_opts.get('show_boundary', False):
            cv2.rectangle(display_frame,
                         (config['boundary_x1'], config['boundary_y1']),
                         (config['boundary_x2'], config['boundary_y2']),
                         (0, 255, 0), 2)

        # 显示静态球
        if display_opts.get('show_static_balls', False) and config.get('ball_tracking_enabled', True):
            ball_module.draw_static_balls(display_frame)

        # 显示FPS
        if display_opts.get('show_fps', False):
            current_fps = (frame_num + 1) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
            display_frame = put_chinese_text(display_frame, f"FPS: {current_fps:.1f}", (10, 60), 0.7, (0, 255, 0))

        # 🎯 显示ROI状态信息
        if roi_manager.is_roi_set:
            roi_info = f"ROI: Active ({len(roi_manager.roi_points)} points)"
            cv2.putText(display_frame, roi_info, (10, frame_height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 显示检测统计
            detection_info = f"Poses:{len(pose_results)} Balls:{len(ball_positions) if ball_positions else 0} Rackets:{len(racket_detections) if racket_detections else 0}"
            cv2.putText(display_frame, detection_info, (10, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示帧号
        if display_opts.get('show_frame_number', False):
            display_frame = put_chinese_text(display_frame, f"Frame: {frame_num}", (10, 90), 0.7, (255, 255, 255))

        # 创建半透明信息面板（headless/minimal_ui 下跳过，降低渲染开销）
        if (not headless and not minimal_ui and
                any([display_opts.get(opt, True) for opt in ['show_swing_type', 'show_ball_position', 'show_racket_state']])):
            info_panel = display_frame.copy()
            panel_height = 600  # 增加面板高度以显示更多指标
            panel_width = 400   # 增加宽度防止截断
            cv2.rectangle(info_panel, (30, 20), (panel_width, panel_height), (0, 0, 0), -1)
            alpha = 0.7
            display_frame = cv2.addWeighted(info_panel, alpha, display_frame, 1 - alpha, 0)

            # 文本显示设置
            text_x_offset = 50
            text_y_offset = 50
            line_height = 25
            current_y = text_y_offset

            # 显示基本挥拍类型
            if display_opts.get('show_swing_type', True):
                display_frame = put_chinese_text(display_frame, f"Swing: {swing_type}", (text_x_offset, current_y), 0.7, (255, 255, 255))
                current_y += line_height

            # 显示球位置信息
            if display_opts.get('show_ball_position', True):
                ball_text = "Ball: Detected" if ball_position else "Ball: Not Detected"
                ball_color = (0, 255, 0) if ball_position else (0, 0, 255)
                display_frame = put_chinese_text(display_frame, ball_text, (text_x_offset, current_y), 0.7, ball_color)
                current_y += line_height

            # 添加球拍状态信息
            if display_opts.get('show_racket_state', True):
                if racket_detections:
                    best_racket = max(racket_detections, key=lambda x: x.get('confidence', 0))
                    conf = best_racket.get('confidence', 0)
                    racket_text = f"Racket: Detected ({conf:.2f})"
                    racket_color = (0, 255, 0)
                else:
                    racket_text = "Racket: Not Detected"
                    racket_color = (0, 0, 255)
                display_frame = put_chinese_text(display_frame, racket_text, (text_x_offset, current_y), 0.7, racket_color)
                current_y += line_height

            # 显示完整挥拍分析指标
            if display_opts.get('show_swing_type', True) and pose_results:
                display_frame = put_chinese_text(display_frame, "--- Full Swing Analysis ---", (text_x_offset, current_y), 0.6, (0, 200, 200))
                current_y += line_height

                # 显示挥拍阶段估计（复用当前帧分析结果）
                if isinstance(phase_est, dict):
                    for category, cat_metrics in phase_est.items():
                        if isinstance(cat_metrics, dict) and cat_metrics:
                            display_frame = put_chinese_text(display_frame, f"[{category.upper()}]", (text_x_offset, current_y), 0.5, (200, 200, 0))
                            current_y += (line_height - 7)

                            for key, value in cat_metrics.items():
                                if current_y > panel_height - 10: break
                                display_text = f"  {key.replace('_', ' ').title()}: {value}"
                                display_frame = put_chinese_text(display_frame, display_text, (text_x_offset + 10, current_y), 0.45, (220, 220, 220))
                                current_y += (line_height - 7)

                            if current_y > panel_height - 10: break

            # 添加帧号（面板底部）
            display_frame = put_chinese_text(display_frame, f"Frame: {frame_num}", (text_x_offset, panel_height - 10), 0.6, (255, 255, 255))

        if not headless:
            cv2.imshow("Tennis Analysis", display_frame)
        if out:
            out.write(display_frame)

        if not headless:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User pressed 'q', exiting program")
                break

        # 可配置的最大帧数限制，便于快速验证
        if max_frames_limit > 0 and frame_num >= max_frames_limit:
            print(f"达到最大验证帧数 {max_frames_limit}，提前结束运行以便快速验证")
            break
        frame_num += 1

    # 计算总处理时间
    total_time = time.time() - start_time
    frames_processed = frame_num
    avg_fps = frames_processed / total_time if total_time > 0 else 0

    print(f"处理完成! 共处理 {frames_processed} 帧")
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"平均处理速度: {avg_fps:.2f} FPS")

    # 显示头像替换统计信息
    if head_processor.enabled:
        final_status = head_processor.get_status()
        stats = final_status.get('stats', {})
        print(f"\n头像替换统计:")
        print(f"  处理的帧数: {stats.get('frames_processed', 0)}")
        print(f"  检测到的人脸: {stats.get('faces_detected', 0)}")
        print(f"  替换的人脸: {stats.get('faces_replaced', 0)}")
        if stats.get('frames_processed', 0) > 0:
            avg_faces = stats.get('faces_detected', 0) / stats.get('frames_processed', 1)
            print(f"  平均每帧人脸数: {avg_faces:.2f}")
            detection_rate = (stats.get('faces_detected', 0) / stats.get('frames_processed', 1)) * 100
            print(f"  检测成功率: {detection_rate:.1f}%")
        print(f"  检测方法使用: {stats.get('detection_method_usage', {})}")
        print(f"  错误数量: {stats.get('error_count', 0)}")

    print("释放资源...")
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("程序结束")

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='网球分析系统')
    parser.add_argument('--config', '-c', default='configs/default_config.yaml',
                        help='配置文件路径 (默认: configs/default_config.yaml)')
    parser.add_argument('--input', '-i', default=None, help='输入视频路径或URL，覆盖配置文件')
    parser.add_argument('--output', '-o', default=None, help='输出视频完整路径，覆盖配置文件')
    parser.add_argument('--output_dir', default=None, help='输出目录（与输入同名文件）')
    parser.add_argument('--original_name', default=None, help='原始文件名（用于显示和日志）')

    args = parser.parse_args()

    # 处理 output_dir 生成完整输出文件路径
    final_output = args.output
    if not final_output and args.output_dir:
        # 如果有原始文件名，使用原始文件名作为输出文件名基础
        if args.original_name:
            in_base = args.original_name
        else:
            in_base = os.path.basename(args.input) if args.input else None
            if not in_base:
                in_base = os.path.basename(load_config(args.config)['video_input_path'])
        # 确保扩展名
        if not in_base:
            in_base = 'output_video.mp4'
        elif not os.path.splitext(in_base)[1]:
            in_base = f"{in_base}.mp4"
        final_output = os.path.join(args.output_dir, in_base)

    main(args.config, input_path=args.input, output_path=final_output, original_name=args.original_name, output_dir=args.output_dir)
