# main.py
import cv2
import yaml
from pose_estimator import PoseEstimator
from ball_tracker import BallTracker
from racket_detector import RacketDetector # <-- IMPORT
from full_swing_analyzer import FullSwingAnalyzer # <-- IMPORT NEW ANALYZER
import os
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

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

def main():
    print("开始加载配置...")
    config = load_config()
    video_path = config['video_input_path']
    print(f"配置加载完成，视频路径: {video_path}")
    
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
    print(f"创建输出视频: {output_path}")
    
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps if fps > 0 else 25,  # 如果原始fps为0，提供默认值
                          (frame_width, frame_height))

    # 初始化组件
    print("初始化姿势估计模块...")
    pose_module = PoseEstimator(config['yolo_pose_model_path'], config)
    print("初始化球追踪模块...")
    ball_module = BallTracker(config.get('tracknet_model_path', None), config)
    print("初始化球拍检测模块...")
    racket_module = RacketDetector(config['racket_yolo_model_path'], config)
    print("初始化完整挥拍分析模块...") # <-- NEW
    swing_analyzer = FullSwingAnalyzer(config) # <-- NEW

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
            
        display_frame = frame.copy()

        # 处理姿势估计
        pose_results = pose_module.get_keypoints(display_frame)
        if pose_results and display_opts.get('show_pose_keypoints', True):
            display_frame = pose_module.draw_keypoints(display_frame, pose_results)
            
        # 处理球追踪
        ball_positions = ball_module.predict_ball(display_frame)
        ball_position = ball_positions[0] if ball_positions else None
        if ball_position:
            if display_opts.get('show_ball_position', True):
                cv2.circle(display_frame, (int(ball_position[0]), int(ball_position[1])), 
                          5, (0, 255, 0), -1)
            
            if display_opts.get('show_ball_trajectory', True):
                # 绘制球的轨迹
                ball_module.draw_trajectory(display_frame)
        
        # 处理球拍检测
        racket_results = racket_module.detect_rackets(display_frame)
        if racket_results and display_opts.get('show_racket_state', True):
            # 绘制球拍状态
            for racket in racket_results:
                box = racket['box']
                cv2.rectangle(display_frame, (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), (255, 0, 0), 2)
        
        # 显示挥拍类型
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
        if display_opts.get('show_static_balls', False):
            ball_module.draw_static_balls(display_frame)
        
        # 显示FPS
        if display_opts.get('show_fps', False):
            current_fps = (frame_num + 1) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
            display_frame = put_chinese_text(display_frame, f"FPS: {current_fps:.1f}", (10, 60), 0.7, (0, 255, 0))
        
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

        cv2.imshow("Tennis Analysis", display_frame)
        if out:
            out.write(display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User pressed 'q', exiting program")
            break
        frame_num += 1

    # 计算总处理时间
    total_time = time.time() - start_time
    frames_processed = frame_num
    avg_fps = frames_processed / total_time if total_time > 0 else 0
    
    print(f"处理完成! 共处理 {frames_processed} 帧")
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"平均处理速度: {avg_fps:.2f} FPS")

    print("释放资源...")
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("程序结束")

if __name__ == "__main__":
    main()