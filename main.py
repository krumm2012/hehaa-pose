# main.py
import cv2
import yaml
from pose_estimator import PoseEstimator
from ball_tracker import BallTracker
from racket_detector import RacketDetector # <-- IMPORT
from full_swing_analyzer import FullSwingAnalyzer # <-- IMPORT NEW ANALYZER
from head_replacement_processor import HeadReplacementProcessor  # <-- NEW IMPORT
from roi_manager import ROIManager  # <-- ROI IMPORT
from enhanced_motion_capture import EnhancedMotionCapture  # <-- MOTION CAPTURE IMPORT
import os
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import json
from collections import deque

def load_config(config_path="configs/default_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_output_directory(output_path):
    """确保输出目录存在"""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_path

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
    font_path = "/System/Library/Fonts/STHeiti Light.ttc"
    font = ImageFont.truetype(font_path, font_size_int)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def save_highlight_frame(frame: np.ndarray, out_dir: str, frame_num: int, tag: str = "hit") -> str:
    """保存精彩瞬间帧到指定目录（兼容保留）"""
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    filename = f"highlight_{frame_num:06d}_{tag}.jpg"
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, frame)
    return path

def save_highlight_clip(frames_map: dict, out_dir: str, center_frame_num: int, fps: int, tag: str = "hit") -> str:
    """保存以 center 为中心，按帧号排序的一段视频"""
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    ordered_nums = sorted(frames_map.keys())
    if not ordered_nums:
        return ""
    h, w = frames_map[ordered_nums[0]].shape[:2]
    filename = f"highlight_{center_frame_num:06d}_{tag}.mp4"
    path = os.path.join(out_dir, filename)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'avc1'),
                             fps if fps > 0 else 25, (w, h))
    for fn in ordered_nums:
        writer.write(frames_map[fn])
    writer.release()
    return path

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
    output_path = create_output_directory(config['video_output_path'])
    if original_name:
        print(f"创建输出视频: {output_path} (来源: {original_name})")
    else:
        print(f"创建输出视频: {output_path}")
    
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'avc1'),
                          fps if fps > 0 else 25,  # 如果原始fps为0，提供默认值
                          (frame_width, frame_height))

    # 精彩瞬间配置
    highlights_cfg = config.get('highlights', {})
    highlight_enabled = highlights_cfg.get('enabled', True)
    highlight_dir = highlights_cfg.get('output_dir', 'data/highlights')
    hit_distance_factor = float(highlights_cfg.get('hit_distance_factor', 1.2))
    racket_radius_factor = float(highlights_cfg.get('racket_radius_factor', 1.0))
    distance_scale = float(highlights_cfg.get('distance_scale', 1.0))
    cooldown_frames = int(highlights_cfg.get('cooldown_frames', 12))
    max_highlights = int(highlights_cfg.get('max_highlights', 50))
    pre_frames = int(highlights_cfg.get('pre_frames', 5))
    post_frames = int(highlights_cfg.get('post_frames', 15))
    save_center_image = bool(highlights_cfg.get('save_center_image', True))
    annotate_center_image = bool(highlights_cfg.get('annotate_center_image', True))
    racket_selection_mode = str(highlights_cfg.get('racket_selection', 'nearest')).lower()
    # 高级抑制条件
    min_inside_frames = int(highlights_cfg.get('min_inside_frames', 1))
    min_relative_threshold_ratio = float(highlights_cfg.get('min_relative_threshold_ratio', 1.2))
    min_exit_increase_px = float(highlights_cfg.get('min_exit_increase_px', 20.0))
    min_speed_px_per_frame = float(highlights_cfg.get('min_speed_px_per_frame', 8.0))
    min_enter_decrease_px = float(highlights_cfg.get('min_enter_decrease_px', 20.0))

    # Headless/最小化UI与最大帧数快速验证
    perf_opts = config.get('performance_optimization', {})
    headless = bool(perf_opts.get('headless', False))
    minimal_ui = bool(perf_opts.get('minimal_ui', False))
    vp_opts = config.get('video_processing', {})
    max_frames_limit = int(vp_opts.get('max_frames', 0)) if vp_opts.get('max_frames', 0) else 0

    last_hit_frame = -10**9
    highlight_count = 0

    # 击球判定的时序变量（用于寻找局部最小距离）
    prev2_dist = None
    prev_dist = None
    prev_effective_racket_radius = None
    prev_threshold = None
    prev_inside_flag = False
    prev_display_snapshot = None
    prev_frame_num_snapshot = None
    consecutive_inside_count = 0
    max_consecutive_inside_allowed = int(highlights_cfg.get('max_inside_frames', 2))

    # 高光视频上下文：缓存最近帧，及等待收集未来帧的任务
    recent_frames = deque(maxlen=60)
    pending_clips = []  # 每项: {start, end, center, frames:{frame_num: frame}}

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
    
    print("🎾 初始化球追踪模块...")
    ball_module = BallTracker(config.get('tracknet_model_path', None), config, roi_manager)
    print("✅ 球追踪模块初始化完成")
    
    print("🏓 初始化球拍检测模块...")
    racket_module = RacketDetector(config['racket_yolo_model_path'], config, roi_manager)
    print("✅ 球拍检测模块初始化完成")
    
    print("🏸 初始化完整挥拍分析模块...")
    swing_analyzer = FullSwingAnalyzer(config)
    print("✅ 完整挥拍分析模块初始化完成")
    
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

        # 缓存最近帧
        recent_frames.append((frame_num, frame.copy()))

        # 处理待生成短视频的任务（收集未来帧）
        if pending_clips:
            still_pending = []
            for task in pending_clips:
                if frame_num <= task['end'] and ret:
                    task['frames'][frame_num] = frame.copy()
                    still_pending.append(task)
                else:
                    # 任务完成，保存短视频
                    clip_path = save_highlight_clip(task['frames'], highlight_dir, task['center'], fps, tag="hit")
                    print(f"🎞️ 精彩瞬间短视频已保存: {clip_path}")
            pending_clips = still_pending

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
            
        # 🎯 第一步：ROI预处理 - 在所有检测之前进行ROI区域提取
        display_frame = frame.copy()
        roi_cropped_frame = None
        roi_offset = (0, 0)  # ROI区域在原图中的偏移量
        
        if roi_manager.is_roi_set:
            # 提取ROI区域用于检测，这样可以大幅减少检测计算量
            roi_mask = roi_manager.get_roi_mask(frame.shape[:2])
            if roi_mask is not None:
                # 找到ROI的边界矩形
                roi_bbox = roi_manager.get_roi_bounding_box()
                if roi_bbox:
                    x1, y1, x2, y2 = roi_bbox

                    # 在ROI外接矩形基础上扩张边距，避免裁剪导致的边界效应
                    margin = int(config.get('roi_settings', {}).get('crop_margin', 12))  # 默认12，可设8~16
                    x1_expanded = max(0, x1 - margin)
                    y1_expanded = max(0, y1 - margin)
                    x2_expanded = min(frame.shape[1], x2 + margin)
                    y2_expanded = min(frame.shape[0], y2 + margin)

                    roi_offset = (x1_expanded, y1_expanded)
                    roi_cropped_frame = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
                    if frame_num % 30 == 0:  # 每30帧显示一次ROI信息
                        print(f"🎯 [帧{frame_num}] ROI区域提取(含边距{margin}px): {roi_cropped_frame.shape} at offset {roi_offset}")
                else:
                    roi_cropped_frame = frame
            else:
                roi_cropped_frame = frame
        else:
            roi_cropped_frame = frame

        # 确定用于检测的帧（ROI裁剪帧或完整帧）
        detection_frame = roi_cropped_frame if roi_cropped_frame is not None else frame

        # **头像替换处理** - 已关闭以提高性能
        # display_frame = head_processor.process_frame(display_frame)

        # 🤖 处理姿势估计 - 根据配置决定使用完整帧还是ROI区域
        pose_detection_frame = frame  # 默认使用完整帧进行姿态检测
        pose_use_roi = config.get('pose_estimation_debug', {}).get('use_roi_detection', False)
        
        if pose_use_roi and detection_frame is not None:
            pose_detection_frame = detection_frame
            if frame_num % 30 == 0:  # 每30帧显示一次检测信息
                print(f"🤖 [帧{frame_num}] 开始姿态检测（ROI模式），检测区域: {detection_frame.shape}")
        else:
            if frame_num % 30 == 0:  # 每30帧显示一次检测信息
                print(f"🤖 [帧{frame_num}] 开始姿态检测（全帧模式），检测区域: {pose_detection_frame.shape}")
        
        pose_results = pose_module.get_keypoints(pose_detection_frame)
        
        # 如果使用了ROI模式进行姿态检测，需要将检测结果坐标转换回原图坐标系
        if pose_use_roi and roi_offset != (0, 0) and pose_results:
            pose_results = roi_manager.adjust_detection_coordinates(pose_results, roi_offset, "pose")
            
        if pose_results and display_opts.get('show_pose_keypoints', True):
            display_frame = pose_module.draw_keypoints(display_frame, pose_results)
            
        # 🎾 处理球追踪 - 在ROI区域内检测
        if frame_num % 30 == 0:  # 每30帧显示一次检测信息
            print(f"🎾 [帧{frame_num}] 开始球检测，检测区域: {detection_frame.shape}")
        ball_positions = ball_module.predict_ball(detection_frame)
        
        # 坐标转换
        if roi_offset != (0, 0) and ball_positions:
            ball_positions = roi_manager.adjust_detection_coordinates(ball_positions, roi_offset, "ball")

        # 先在主流程中过滤 ROI 外的球，避免静止球（ROI外）参与轨迹与抢占
        if roi_manager.is_roi_set:
            roi_cfg = config.get('roi_settings', {})
            if roi_cfg.get('filter_balls_outside_roi', True):
                ball_positions = roi_manager.filter_detections_by_roi(ball_positions or [], "ball")

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
        
        # 🏓 处理球拍检测 - 在ROI区域内检测
        if frame_num % 30 == 0:  # 每30帧显示一次检测信息
            print(f"🏓 [帧{frame_num}] 开始球拍检测，检测区域: {detection_frame.shape}")
        racket_results = racket_module.detect_rackets(detection_frame)
        
        # 坐标转换
        if roi_offset != (0, 0) and racket_results:
            racket_results = roi_manager.adjust_detection_coordinates(racket_results, roi_offset, "racket")

        # ⭐ 精彩瞬间（改进）：使用前后帧距离变化寻找局部最小距离帧
        if (highlight_enabled and ball_position and racket_results 
                and highlight_count < max_highlights):
            bx, by = float(ball_position[0]), float(ball_position[1])

            # 选择一个代表性的球拍：优先离球最近
            selected_racket = None
            if racket_selection_mode == 'nearest' and ball_position:
                bx_tmp, by_tmp = float(ball_position[0]), float(ball_position[1])
                best_dist = float('inf')
                for racket in racket_results:
                    if not isinstance(racket, dict) or 'box' not in racket:
                        continue
                    x1, y1, x2, y2 = racket['box']
                    cx_tmp = (x1 + x2) / 2.0
                    cy_tmp = (y1 + y2) / 2.0
                    d_tmp = ((bx_tmp - cx_tmp) ** 2 + (by_tmp - cy_tmp) ** 2) ** 0.5
                    if d_tmp < best_dist:
                        best_dist = d_tmp
                        selected_racket = racket
            else:
                # 回退：取最大面积
                max_area = -1
                for racket in racket_results:
                    if not isinstance(racket, dict) or 'box' not in racket:
                        continue
                    x1, y1, x2, y2 = racket['box']
                    area = max(1, (x2 - x1) * (y2 - y1))
                    if area > max_area:
                        max_area = area
                        selected_racket = racket

            if selected_racket:
                x1, y1, x2, y2 = selected_racket['box']
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                rw = max(1.0, (x2 - x1))
                rh = max(1.0, (y2 - y1))
                effective_racket_radius = max(rw, rh) / 2.0

                ball_r = float(config.get('ball_radius_px', 10))
                impact_threshold = (effective_racket_radius * racket_radius_factor + ball_r * hit_distance_factor) * distance_scale

                dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
                inside_now = dist <= impact_threshold
                if frame_num % 5 == 0:
                    print(f"[HL] frame={frame_num} dist={dist:.2f} thr={impact_threshold:.2f} inside={inside_now}")

                # 记录当前帧，若进入阈值区域则保存快照，候选为局部最小距离帧
                if inside_now:
                    consecutive_inside_count = min(max_consecutive_inside_allowed, consecutive_inside_count + 1)
                    # 更新候选：当距离更小或还没有候选时
                    if prev_display_snapshot is None or (prev_dist is not None and dist < prev_dist):
                        prev_display_snapshot = display_frame.copy()
                        prev_frame_num_snapshot = frame_num
                else:
                    consecutive_inside_count = 0

                # 寻找局部最小：前一帧在阈值内，当前帧距离开始回升或离开阈值
                if (
                    prev_dist is not None and prev2_dist is not None and
                    prev_dist <= ((prev_effective_racket_radius or effective_racket_radius) * racket_radius_factor + ball_r * hit_distance_factor) * distance_scale and
                    prev_dist <= prev2_dist and  # 下降到前一帧
                    dist >= prev_dist and        # 当前回升
                    prev_display_snapshot is not None and
                    (frame_num - last_hit_frame) >= cooldown_frames
                ):
                    # 额外抑制条件：需要在阈值内停留至少若干帧、局部最小足够小、离开阶段距离有明显增加、球速足够高
                    passed = True
                    if consecutive_inside_count < min_inside_frames:
                        passed = False
                    if not (prev_threshold is not None and prev_dist <= prev_threshold * min_relative_threshold_ratio):
                        passed = False
                    if (dist - prev_dist) < min_exit_increase_px:
                        passed = False
                    # 3b) 进入阶段距离下降幅度
                    if (prev2_dist - prev_dist) < min_enter_decrease_px:
                        passed = False
                    recent_speed = 0.0
                    try:
                        if len(ball_module.tracked_balls_history) >= 2:
                            p1 = np.array(ball_module.tracked_balls_history[-1]['coords'], dtype=float)
                            p0 = np.array(ball_module.tracked_balls_history[-2]['coords'], dtype=float)
                            recent_speed = float(np.linalg.norm(p1 - p0))
                    except Exception:
                        recent_speed = 0.0
                    if recent_speed < min_speed_px_per_frame:
                        passed = False

                    if not passed:
                        prev_display_snapshot = None
                        prev_frame_num_snapshot = None
                    else:
                        # 在回升点触发保存，使用之前记录的最佳快照
                        # 生成以中心帧为核心的前后短视频
                        # 1) 收集过去帧（可配置 pre_frames）
                        clip_frames = {}
                        for fn, fr in list(recent_frames):
                            if prev_frame_num_snapshot - pre_frames <= fn <= prev_frame_num_snapshot:
                                clip_frames[fn] = fr.copy()
                        # 2) 创建未来帧收集任务（可配置：前pre_frames，后post_frames）
                        pending_clips.append({
                            'start': prev_frame_num_snapshot + 1,
                            'end': prev_frame_num_snapshot + post_frames,
                            'center': prev_frame_num_snapshot,
                            'frames': clip_frames
                        })
                        # 3) 保存中心帧图片（带可选注释）
                        if save_center_image and prev_display_snapshot is not None:
                            snapshot = prev_display_snapshot.copy()
                            if annotate_center_image:
                                cv2.circle(snapshot, (int(bx), int(by)), 8, (0, 0, 255), -1)
                                cv2.circle(snapshot, (int(cx), int(cy)), 8, (255, 0, 0), -1)
                                cv2.putText(snapshot, f"HIT dist={prev_dist:.1f}", (int(bx)+10, int(by)-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                            img_path = save_highlight_frame(snapshot, highlight_dir, prev_frame_num_snapshot, tag="hit")
                            print(f"📸 精彩瞬间中心帧已保存: {img_path}")

                        # 4) 生成分析JSON与阶段占位截图（与最终mp4同名基名）
                        try:
                            base_name = f"highlight_{prev_frame_num_snapshot:06d}_hit"
                            base_path = os.path.join(highlight_dir, base_name)

                            # 分析当前帧的指标（使用最新的 pose/racket/ball）
                            try:
                                analysis = swing_analyzer.analyze_swing_components(
                                    pose_results if pose_results else [],
                                    racket_results if racket_results else [],
                                    (ball_position[0], ball_position[1]) if ball_position else None,
                                    (frame_height, frame_width)
                                )
                            except Exception as _e:
                                print(f"分析组件计算失败: {_e}")
                                analysis = {}

                            analysis_payload = {
                                "frame_center": int(prev_frame_num_snapshot),
                                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "analysis": analysis
                            }

                            # 写入 .analysis.json
                            try:
                                with open(f"{base_path}.analysis.json", "w", encoding="utf-8") as f:
                                    json.dump(analysis_payload, f, ensure_ascii=False, indent=2)
                                print(f"📝 分析JSON已保存: {base_path}.analysis.json")
                            except Exception as _e:
                                print(f"保存分析JSON失败: {_e}")

                            # 生成阶段占位截图（先用同一张snapshot占位，后续可替换为阶段化帧）
                            try:
                                if prev_display_snapshot is not None:
                                    phase_tags = [
                                        ("prep", "准备"),
                                        ("turn", "转身"),
                                        ("drop", "降拍"),
                                        ("swing", "挥拍"),
                                        ("foot", "步伐")
                                    ]
                                    for tag, zh in phase_tags:
                                        out_img = prev_display_snapshot.copy()
                                        # 轻微标注角标，便于区分（不影响主流程）
                                        try:
                                            cv2.putText(out_img, zh, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                                        except Exception:
                                            pass
                                        cv2.imwrite(f"{base_path}_{tag}.jpg", out_img)
                                    print(f"🖼️ 阶段占位截图已保存: {base_path}_[prep|turn|drop|swing|foot].jpg")
                            except Exception as _e:
                                print(f"保存阶段占位截图失败: {_e}")
                        except Exception as e_inner:
                            print(f"生成分析侧车文件失败: {e_inner}")
                        last_hit_frame = prev_frame_num_snapshot
                        highlight_count += 1
                        print(f"⭐ 精彩瞬间-击球(最小距离): 计划保存短视频（前5后15），中心帧 {prev_frame_num_snapshot} (total={highlight_count})")
                        # 重置候选
                        prev_display_snapshot = None
                        prev_frame_num_snapshot = None

                # 更新时序状态
                prev2_dist = prev_dist
                prev_dist = dist
                prev_effective_racket_radius = effective_racket_radius
                prev_threshold = impact_threshold
                prev_inside_flag = inside_now
            
        if racket_results and display_opts.get('show_racket_state', True):
            # 绘制球拍状态
            for racket in racket_results:
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
                racket_results if racket_results else [],
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
                if racket_results:
                    display_frame = roi_manager.highlight_roi_detections(display_frame, racket_results, "racket")
        
        # 显示挥拍类型
        swing_type = "No Pose"  # 默认值
        if pose_results and display_opts.get('show_swing_type', True):
            swing_type = pose_module.classify_swing(pose_results)
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
            detection_info = f"Poses:{len(pose_results)} Balls:{len(ball_positions) if ball_positions else 0} Rackets:{len(racket_results) if racket_results else 0}"
            cv2.putText(display_frame, detection_info, (10, frame_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示帧号
        if display_opts.get('show_frame_number', False):
            display_frame = put_chinese_text(display_frame, f"Frame: {frame_num}", (10, 90), 0.7, (255, 255, 255))

        # 创建半透明信息面板
        if any([display_opts.get(opt, True) for opt in ['show_swing_type', 'show_ball_position', 'show_racket_state']]):
            info_panel = display_frame.copy()
            panel_height = 400
            cv2.rectangle(info_panel, (30, 20), (330, panel_height), (0, 0, 0), -1)
            alpha = 0.7
            display_frame = cv2.addWeighted(info_panel, alpha, display_frame, 1 - alpha, 0)
            
            # 文本显示设置
            text_x_offset = 50
            text_y_offset = 50
            line_height = 25
            current_y = text_y_offset

            # 显示基本挥拍类型
            if display_opts.get('show_swing_type', True):
                display_frame = put_chinese_text(display_frame, f"Swing Type: {swing_type}", (text_x_offset, current_y), 0.7, (255, 255, 255))
                current_y += line_height
            
            # 显示球位置信息
            if display_opts.get('show_ball_position', True):
                if ball_position:
                    display_frame = put_chinese_text(display_frame, f"Ball Position: ({int(ball_position[0])}, {int(ball_position[1])})", (text_x_offset, current_y), 0.7, (255, 255, 255))
                else:
                    display_frame = put_chinese_text(display_frame, "Ball: Not Detected", (text_x_offset, current_y), 0.7, (255, 255, 255))
                current_y += line_height

            # 添加球拍状态信息
            if display_opts.get('show_racket_state', True) and racket_results:
                for racket in racket_results:
                    racket_state = racket.get('state', 'Unknown')
                    display_frame = put_chinese_text(display_frame, f"Racket State: {racket_state}", (text_x_offset, current_y), 0.7, (255, 255, 255))
                    current_y += line_height

            # 显示完整挥拍分析指标
            if display_opts.get('show_swing_type', True) and pose_results:
                display_frame = put_chinese_text(display_frame, "--- Full Swing Analysis ---", (text_x_offset, current_y), 0.6, (0, 200, 200))
                current_y += line_height
                
                # 显示挥拍阶段估计
                phase_est = swing_analyzer.analyze_swing_components(pose_results, racket_results, ball_position, (frame_height, frame_width))
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

        # 添加帧号
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