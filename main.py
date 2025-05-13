# main.py
import cv2
import yaml
from pose_estimator import PoseEstimator
from ball_tracker import BallTracker
import os
import numpy as np
import time

def load_config(config_path="configs/default_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_output_directory(output_path):
    """确保输出目录存在"""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_path

def main():
    print("开始加载配置...")
    config = load_config()
    video_path = config['video_input_path']
    print(f"配置加载完成，视频路径: {video_path}")
    
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

        # 1. 姿势估计
        person_keypoints_list = pose_module.get_keypoints(frame)
        display_frame = pose_module.draw_keypoints(display_frame, person_keypoints_list)

        # 2. 挥拍分类
        swing_type = "No Swing"  # 默认值
        if person_keypoints_list:  # 如果检测到人
            # a>使用第一个检测到的人作为球员
            swing_type = pose_module.classify_swing(person_keypoints_list)

        # 3. 球检测 (使用 TrackNetV2 模型或模拟)
        raw_ball_detections = ball_module.predict_ball(frame)
        
        # 可选：从检测到的坐标生成热图
        # heatmap = ball_module.generate_heatmap_from_coords(raw_ball_detections, frame.shape)
        # 可选：处理热图找到球的位置
        # raw_ball_detections = ball_module.process_heatmap(heatmap, frame.shape)

        # 4. 高级球处理：过滤静态球并追踪主要移动球
        active_ball_this_frame_list = ball_module.advanced_ball_processing(raw_ball_detections, frame_num)

        # 每30帧应用一次轨迹平滑和离群值移除
        if frame_num % 30 == 0:
            ball_module.remove_outliers(threshold=3.0)
            ball_module.interpolate_trajectory()

        # 5. 注释
        # 绘制活动的球（如果有）
        display_frame = ball_module.draw_ball(display_frame, active_ball_this_frame_list)
        # 绘制基于内部历史的轨迹
        display_frame = ball_module.draw_trajectory(display_frame)
        # 绘制确认的静态球
        display_frame = ball_module.draw_static_balls(display_frame)

        # 添加文本信息
        cv2.putText(display_frame, f"挥拍类型: {swing_type}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        if active_ball_this_frame_list:  # 检查列表是否非空
            ball_pos = active_ball_this_frame_list[0]
            cv2.putText(display_frame, f"球位置: ({int(ball_pos[0])}, {int(ball_pos[1])})", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(display_frame, "球: 未检测到", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # 添加帧号
        cv2.putText(display_frame, f"帧号: {frame_num}", (50, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("网球分析", display_frame)
        if out:
            out.write(display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户按下'q'键，程序退出")
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