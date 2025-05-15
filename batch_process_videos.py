# batch_process_videos.py
import cv2
import yaml
import os
import numpy as np
import time
import copy
from pose_estimator import PoseEstimator
from ball_tracker import BallTracker
from racket_detector import RacketDetector

def load_config(config_path="configs/default_config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_output_directory(output_path):
    """确保输出目录存在"""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_path

def process_video(video_path, config):
    """处理单个视频文件并保存到同级的out目录"""
    # 创建输出目录（与输入视频同级的out目录）
    input_dir = os.path.dirname(video_path)
    out_dir = os.path.join(input_dir, "out")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # 构建输出视频路径
    video_filename = os.path.basename(video_path)
    output_path = os.path.join(out_dir, video_filename)
    
    print(f"处理视频: {video_path}")
    print(f"输出到: {output_path}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return False
    
    # 获取视频信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print("警告: 无法获取视频总帧数，或总帧数为0。进度百分比可能不准确。")
        total_frames = fps * 3600 if fps > 0 else -1
    
    print(f"视频信息 - 宽度: {frame_width}, 高度: {frame_height}, FPS: {fps}, 总帧数: {total_frames if total_frames > 0 else 'N/A'}")
    
    # 创建视频写入对象
    out = cv2.VideoWriter(output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps if fps > 0 else 25,
                        (frame_width, frame_height))
    
    # 初始化组件
    pose_module = PoseEstimator(config['yolo_pose_model_path'], config)
    ball_module = BallTracker(config.get('tracknet_model_path', None), config)
    racket_module = RacketDetector(config['racket_yolo_model_path'], config)
    
    # 进度和时间跟踪
    frame_num = 0
    start_time = time.time()
    batch_start_time = start_time
    
    print(f"开始处理视频 {video_filename}...")
    
    # 逐帧处理视频
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"视频 {video_filename} 处理完成!")
            break
        
        # 显示进度
        if total_frames > 0 and frame_num % (fps if fps > 0 else 30) == 0:
            elapsed_time = time.time() - start_time
            batch_time = time.time() - batch_start_time
            batch_start_time = time.time()
            
            frames_processed = frame_num + 1
            progress = frames_processed / total_frames * 100
            
            # 估计剩余时间
            if frame_num > 0:
                time_per_frame = elapsed_time / frames_processed
                frames_remaining = total_frames - frames_processed
                estimated_time_remaining = frames_remaining * time_per_frame
                
                print(f"处理帧 {frames_processed}/{total_frames} ({progress:.1f}%) - "
                      f"批处理时间: {batch_time:.2f}秒, "
                      f"估计剩余时间: {estimated_time_remaining:.1f}秒")
        
        display_frame = frame.copy()  # 在副本上绘制
        
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
                    cv2.putText(display_frame, f"{right_elbow_angle:.0f}°", 
                                (elbow_pos[0] + 10, elbow_pos[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)
            
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
                    cv2.putText(display_frame, f"{left_elbow_angle:.0f}°", 
                                (elbow_pos[0] + 10, elbow_pos[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (135, 206, 235), 2)
            
            # 计算躯干角度
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
                    cv2.putText(display_frame, f"躯干: {torso_angle:.0f}°", 
                                (shoulder_center[0], shoulder_center[1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (75, 0, 130), 2)
        
        # 2. 绘制姿势关键点和骨架
        display_frame = pose_module.draw_keypoints(display_frame, person_keypoints_list)
        
        # 3. 挥拍分类
        swing_type = "No Swing"  # 默认值
        if person_keypoints_list:  # 如果检测到人
            swing_type = pose_module.classify_swing(person_keypoints_list)
        
        # 4. 球拍检测和关联
        raw_racket_detections = racket_module.detect_rackets(frame)
        associated_rackets_info = racket_module.associate_racket_to_player(raw_racket_detections, person_keypoints_list, frame_num)
        
        # 5. 球检测 (使用 TrackNetV2 模型或模拟)
        raw_ball_detections = ball_module.predict_ball(frame)
        
        # 6. 高级球处理：过滤静态球并追踪主要移动球
        active_ball_this_frame_list = ball_module.advanced_ball_processing(raw_ball_detections, frame_num)
        
        # 7. 确定球拍状态基于球的位置
        player_id_for_state = 0  # 使用第一个检测到的人作为主要球员
        if player_id_for_state in associated_rackets_info and active_ball_this_frame_list:
            racket_module.determine_racket_state(player_id_for_state, active_ball_this_frame_list[0], frame_height)
        
        # 每30帧应用一次轨迹平滑和离群值移除
        if frame_num % 30 == 0:
            ball_module.remove_outliers(threshold=3.0)
            ball_module.interpolate_trajectory()
        
        # 8. 绘制可视化
        # 绘制关联的球拍和状态
        display_frame = racket_module.draw_associated_rackets(display_frame)
        
        # 绘制活动的球（如果有）
        display_frame = ball_module.draw_ball(display_frame, active_ball_this_frame_list)
        # 绘制基于内部历史的轨迹
        display_frame = ball_module.draw_trajectory(display_frame)
        # 绘制确认的静态球
        display_frame = ball_module.draw_static_balls(display_frame)
        
        # 9. 添加文本信息
        # 创建半透明信息面板
        info_panel = display_frame.copy()
        cv2.rectangle(info_panel, (30, 20), (300, 150), (0, 0, 0), -1)  # 增加高度以容纳更多信息
        alpha = 0.7  # 透明度
        display_frame = cv2.addWeighted(info_panel, alpha, display_frame, 1 - alpha, 0)
        
        # 在面板上添加挥拍类型信息
        cv2.putText(display_frame, f"挥拍类型: {swing_type}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 在面板上添加球位置信息
        if active_ball_this_frame_list:  # 检查列表是否非空
            ball_pos = active_ball_this_frame_list[0]
            cv2.putText(display_frame, f"球位置: ({int(ball_pos[0])}, {int(ball_pos[1])})", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(display_frame, "球: 未检测到", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 添加球拍状态信息
        if player_id_for_state in racket_module.player_rackets:
            racket_state = racket_module.player_rackets[player_id_for_state]['state']
            cv2.putText(display_frame, f"球拍状态: {racket_state}", (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 添加帧号和视频名
        cv2.putText(display_frame, f"帧号: {frame_num}", (50, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, os.path.basename(video_path), (frame_width - 300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 添加虚线边框效果
        h, w = display_frame.shape[:2]
        # 四个角和中点
        cv2.circle(display_frame, (30, 30), 5, (0, 0, 255), -1)  # 左上角点
        cv2.circle(display_frame, (w-30, 30), 5, (0, 0, 255), -1)  # 右上角点
        cv2.circle(display_frame, (30, h-30), 5, (0, 0, 255), -1)  # 左下角点
        cv2.circle(display_frame, (w-30, h-30), 5, (0, 0, 255), -1)  # 右下角点
        cv2.circle(display_frame, (w//2, 30), 5, (0, 0, 255), -1)  # 上中点
        cv2.circle(display_frame, (w//2, h-30), 5, (0, 0, 255), -1)  # 下中点
        cv2.circle(display_frame, (30, h//2), 5, (0, 0, 255), -1)  # 左中点
        cv2.circle(display_frame, (w-30, h//2), 5, (0, 0, 255), -1)  # 右中点
        
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
        
        # 写入输出视频
        out.write(display_frame)
        frame_num += 1
    
    # 计算总处理时间
    total_time = time.time() - start_time
    frames_processed = frame_num
    avg_fps = frames_processed / total_time if total_time > 0 else 0
    
    print(f"视频 {video_filename} 处理完成!")
    print(f"总帧数: {frames_processed}, 处理时间: {total_time:.2f} 秒, 平均帧率: {avg_fps:.2f} FPS")
    
    # 释放资源
    cap.release()
    out.release()
    
    return True

def find_video_files(directory):
    """递归查找目录及其子目录中的所有视频文件"""
    video_files = []
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    return video_files

def main():
    print("开始批量处理视频...")
    
    # 加载配置
    config = load_config()
    # 复制一份避免修改原始配置
    processing_config = copy.deepcopy(config)
    
    # 要处理的视频目录
    player_video_dir = "player-video"
    if not os.path.exists(player_video_dir):
        print(f"错误: 目录 {player_video_dir} 不存在")
        return
    
    # 查找所有视频文件
    video_files = find_video_files(player_video_dir)
    if not video_files:
        print(f"未找到视频文件在 {player_video_dir} 目录下")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频文件
    for i, video_path in enumerate(video_files):
        print(f"\n[{i+1}/{len(video_files)}] 处理视频: {video_path}")
        try:
            process_video(video_path, processing_config)
        except Exception as e:
            print(f"处理视频 {video_path} 时发生错误: {str(e)}")
    
    print("\n所有视频处理完成!")

if __name__ == "__main__":
    main()