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
            
        display_frame = frame.copy()  # 在副本上绘制
        frame_dimensions = (frame_height, frame_width) # 传递给分析器 <-- NEW

        # 1. 姿势估计
        person_keypoints_list = pose_module.get_keypoints(frame)
        
        # 计算并显示关键关节角度
        if person_keypoints_list and len(person_keypoints_list) > 0:
            keypoints = person_keypoints_list[0]
            
            # 计算手肘角度
            if all(keypoints.get(kp) for kp in ["right_shoulder", "right_elbow", "right_wrist"]):
                right_elbow_angle = pose_module.calculate_angle(
                    keypoints["right_shoulder"], 
                    keypoints["right_elbow"], 
                    keypoints["right_wrist"]
                )
                # 在右手肘位置显示角度
                if keypoints["right_elbow"]:
                    elbow_pos = keypoints["right_elbow"]
                    display_frame = put_chinese_text(display_frame, f"{right_elbow_angle:.0f}°", (elbow_pos[0] + 10, elbow_pos[1]), 0.5, (255, 140, 0))
            
            # 计算左手肘角度
            if all(keypoints.get(kp) for kp in ["left_shoulder", "left_elbow", "left_wrist"]):
                left_elbow_angle = pose_module.calculate_angle(
                    keypoints["left_shoulder"], 
                    keypoints["left_elbow"], 
                    keypoints["left_wrist"]
                )
                # 在左手肘位置显示角度
                if keypoints["left_elbow"]:
                    elbow_pos = keypoints["left_elbow"]
                    display_frame = put_chinese_text(display_frame, f"{left_elbow_angle:.0f}°", (elbow_pos[0] + 10, elbow_pos[1]), 0.5, (135, 206, 235))
            
            # 计算躯干角度（肩部连线与垂直线的角度）
            if all(keypoints.get(kp) for kp in ["right_shoulder", "left_shoulder"]):
                rs = np.array(keypoints["right_shoulder"])
                ls = np.array(keypoints["left_shoulder"])
                shoulder_vector = rs - ls
                vertical_vector = np.array([0, 1])
                dot_product = np.dot(shoulder_vector, vertical_vector)
                norm_product = np.linalg.norm(shoulder_vector) * np.linalg.norm(vertical_vector)
                if norm_product > 0:
                    torso_angle = np.degrees(np.arccos(dot_product / norm_product))
                    # 在肩部中心位置显示躯干角度
                    shoulder_center = ((rs[0] + ls[0]) // 2, (rs[1] + ls[1]) // 2)
                    display_frame = put_chinese_text(display_frame, f"Torso: {torso_angle:.0f}°", (shoulder_center[0], shoulder_center[1] - 20), 0.5, (75, 0, 130))
        
        # 2. 绘制姿势关键点和骨架
        display_frame = pose_module.draw_keypoints(display_frame, person_keypoints_list)

        # 3. 挥拍分类
        swing_type = "No Swing"  # 默认值
        if person_keypoints_list:  # 如果检测到人
            # a>使用第一个检测到的人作为球员
            swing_type = pose_module.classify_swing(person_keypoints_list)

        # 4. 球拍检测和关联 <-- NEW
        raw_racket_detections = racket_module.detect_rackets(frame)
        associated_rackets_info = racket_module.associate_racket_to_player(raw_racket_detections, person_keypoints_list, frame_num)

        # 5. 球检测 (使用 TrackNetV2 模型或模拟)
        raw_ball_detections = ball_module.predict_ball(frame)
        
        # 可选：从检测到的坐标生成热图
        # heatmap = ball_module.generate_heatmap_from_coords(raw_ball_detections, frame.shape)
        # 可选：处理热图找到球的位置
        # raw_ball_detections = ball_module.process_heatmap(heatmap, frame.shape)

        # 6. 高级球处理：过滤静态球并追踪主要移动球
        active_ball_this_frame_list = ball_module.advanced_ball_processing(raw_ball_detections, frame_num)

        # 7. 确定球拍状态基于球的位置 <-- NEW
        player_id_for_state = 0  # 使用第一个检测到的人作为主要球员
        # 判断球拍状态（是否接近球、碰撞、跟随）
        if player_id_for_state in associated_rackets_info and active_ball_this_frame_list:
            racket_module.determine_racket_state(player_id_for_state, active_ball_this_frame_list[0], frame_height)

        # 8. 执行完整的挥拍分析 <-- NEW
        full_analysis_metrics = {}
        if person_keypoints_list:  # 只有在检测到人时才进行分析
            full_analysis_metrics = swing_analyzer.analyze_swing_components(
                person_keypoints_list,
                associated_rackets_info,  # 传递关联的球拍信息
                active_ball_this_frame_list[0] if active_ball_this_frame_list else None,
                frame_dimensions
            )

        # 每30帧应用一次轨迹平滑和离群值移除
        if frame_num % 30 == 0:
            ball_module.remove_outliers(threshold=3.0)
            ball_module.interpolate_trajectory()

        # 9. 绘制可视化
        # 绘制关联的球拍和状态
        display_frame = racket_module.draw_associated_rackets(display_frame)

        # 绘制活动的球（如果有）
        display_frame = ball_module.draw_ball(display_frame, active_ball_this_frame_list)
        # 绘制基于内部历史的轨迹
        display_frame = ball_module.draw_trajectory(display_frame)
        # 绘制确认的静态球
        display_frame = ball_module.draw_static_balls(display_frame)

        # 10. 添加文本信息
        # 创建半透明信息面板 - 扩大面板高度以容纳更多信息
        info_panel = display_frame.copy()
        panel_height = 400  # 增大高度以显示更多分析数据
        cv2.rectangle(info_panel, (30, 20), (330, panel_height), (0, 0, 0), -1)
        alpha = 0.7  # 透明度
        display_frame = cv2.addWeighted(info_panel, alpha, display_frame, 1 - alpha, 0)
        
        # 文本显示设置
        text_x_offset = 50
        text_y_offset = 50
        line_height = 25

        # 显示基本挥拍类型
        display_frame = put_chinese_text(display_frame, f"Swing Type: {swing_type}", (text_x_offset, text_y_offset), 0.7, (255, 255, 255))
        current_y = text_y_offset + line_height
        
        # 显示球位置信息
        if active_ball_this_frame_list:  # 检查列表是否非空
            ball_pos = active_ball_this_frame_list[0]
            display_frame = put_chinese_text(display_frame, f"Ball Position: ({int(ball_pos[0])}, {int(ball_pos[1])})", (text_x_offset, current_y), 0.7, (255, 255, 255))
        else:
            display_frame = put_chinese_text(display_frame, "Ball: Not Detected", (text_x_offset, current_y), 0.7, (255, 255, 255))
        current_y += line_height

        # 添加球拍状态信息
        if player_id_for_state in racket_module.player_rackets:
            racket_state = racket_module.player_rackets[player_id_for_state]['state']
            display_frame = put_chinese_text(display_frame, f"Racket State: {racket_state}", (text_x_offset, current_y), 0.7, (255, 255, 255))
        current_y += line_height

        # 显示完整挥拍分析指标 <-- NEW
        display_frame = put_chinese_text(display_frame, "--- Full Swing Analysis ---", (text_x_offset, current_y), 0.6, (0, 200, 200))
        current_y += line_height
        
        # 显示挥拍阶段估计
        phase_est = full_analysis_metrics.get("phase_estimation", "Unknown")
        display_frame = put_chinese_text(display_frame, f"Estimated Phase: {phase_est}", (text_x_offset, current_y), 0.5, (0, 200, 200))
        current_y += (line_height - 5)

        # 显示其他分析指标
        for category, cat_metrics in full_analysis_metrics.items():
            if category == "phase_estimation": continue  # 已经显示过了
            if isinstance(cat_metrics, dict) and cat_metrics:
                display_frame = put_chinese_text(display_frame, f"[{category.upper()}]", (text_x_offset, current_y), 0.5, (200, 200, 0))
                current_y += (line_height - 5)
                
                for key, value in cat_metrics.items():
                    if current_y > panel_height - 10: break  # 避免绘制超出面板
                    display_text = f"  {key.replace('_', ' ').title()}: {value}"
                    display_frame = put_chinese_text(display_frame, display_text, (text_x_offset + 10, current_y), 0.45, (220, 220, 220))
                    current_y += (line_height - 7)
                    
                if current_y > panel_height - 10: break  # 避免绘制超出面板

        # 添加帧号
        display_frame = put_chinese_text(display_frame, f"Frame: {frame_num}", (text_x_offset, panel_height - 10), 0.6, (255, 255, 255))

        # 显示FPS
        current_fps = (frame_num + 1) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        display_frame = put_chinese_text(display_frame, f"FPS: {current_fps:.1f}", (frame_width - 120, 30), 0.7, (0, 255, 0))

        # 添加虚线边框效果(类似图片中的效果)
        h, w = display_frame.shape[:2]
        # 左上角点
        cv2.circle(display_frame, (30, 30), 5, (0, 0, 255), -1)
        # 右上角点
        cv2.circle(display_frame, (w-30, 30), 5, (0, 0, 255), -1)
        # 左下角点
        cv2.circle(display_frame, (30, h-30), 5, (0, 0, 255), -1)
        # 右下角点
        cv2.circle(display_frame, (w-30, h-30), 5, (0, 0, 255), -1)
        # 上中点
        cv2.circle(display_frame, (w//2, 30), 5, (0, 0, 255), -1)
        # 下中点
        cv2.circle(display_frame, (w//2, h-30), 5, (0, 0, 255), -1)
        # 左中点
        cv2.circle(display_frame, (30, h//2), 5, (0, 0, 255), -1)
        # 右中点
        cv2.circle(display_frame, (w-30, h//2), 5, (0, 0, 255), -1)
        
        # 绘制虚线边框
        # 上边线
        for x in range(30, w-30, 10):
            cv2.line(display_frame, (x, 30), (x+5, 30), (0, 0, 255), 1)
        # 下边线
        for x in range(30, w-30, 10):
            cv2.line(display_frame, (x, h-30), (x+5, h-30), (0, 0, 255), 1)
        # 左边线
        for y in range(30, h-30, 10):
            cv2.line(display_frame, (30, y), (30, y+5), (0, 0, 255), 1)
        # 右边线
        for y in range(30, h-30, 10):
            cv2.line(display_frame, (w-30, y), (w-30, y+5), (0, 0, 255), 1)

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