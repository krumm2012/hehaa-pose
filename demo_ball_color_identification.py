#!/usr/bin/env python3
"""
🎾 球颜色标识演示脚本
演示静止球(蓝色)和运动球(红色)的颜色标识功能
"""

import cv2
import yaml
import numpy as np
from ball_tracker import BallTracker

# 全局变量用于存储鼠标坐标
mouse_x, mouse_y = 0, 0
mouse_clicked = False
show_mouse_coords = True  # 控制是否显示鼠标坐标

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数 - 跟踪鼠标位置和点击"""
    global mouse_x, mouse_y, mouse_clicked
    mouse_x, mouse_y = x, y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True
        print(f"🖱️  鼠标点击坐标: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"📍 右键点击坐标: ({x}, {y}) - 可用于设置屏蔽区域")

def draw_mouse_coordinates(frame, x, y, clicked=False):
    """在帧上绘制鼠标坐标信息"""
    height, width = frame.shape[:2]
    
    # 坐标显示栏位置 - 顶部中央
    bar_width = 300
    bar_height = 60
    bar_x = (width - bar_width) // 2
    bar_y = 10
    
    # 绘制半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                  (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # 绘制边框
    border_color = (0, 255, 0) if clicked else (255, 255, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                  border_color, 2)
    
    # 显示坐标文本
    coord_text = f"Mouse: ({x}, {y})"
    click_text = "Left Click Detected!" if clicked else "Move mouse to see coordinates"
    
    # 主坐标文本 - 大字体
    cv2.putText(frame, coord_text, (bar_x + 10, bar_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 提示文本 - 小字体
    cv2.putText(frame, click_text, (bar_x + 10, bar_y + 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # 在鼠标位置绘制十字准线
    if 0 <= x < width and 0 <= y < height:
        # 水平线
        cv2.line(frame, (max(0, x-20), y), (min(width-1, x+20), y), (0, 255, 255), 1)
        # 垂直线
        cv2.line(frame, (x, max(0, y-20)), (x, min(height-1, y+20)), (0, 255, 255), 1)
        
        # 鼠标位置点
        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        
        # 在鼠标旁边显示坐标（当鼠标不在顶部时）
        if y > 80:  # 避免与顶部坐标栏重叠
            coord_bg_x = max(5, min(width - 120, x + 15))
            coord_bg_y = max(20, y - 5)
            
            # 小坐标标签背景
            cv2.rectangle(frame, (coord_bg_x, coord_bg_y - 15), 
                         (coord_bg_x + 80, coord_bg_y + 5), (0, 0, 0), -1)
            cv2.rectangle(frame, (coord_bg_x, coord_bg_y - 15), 
                         (coord_bg_x + 80, coord_bg_y + 5), (0, 255, 255), 1)
            
            # 小坐标文本
            cv2.putText(frame, f"({x},{y})", (coord_bg_x + 3, coord_bg_y - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    return frame

def load_config(config_path="configs/noise_filtered_ball_config.yaml"):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        print(f"✅ 配置加载成功: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return None

def draw_legend(frame):
    """绘制颜色标识图例"""
    height, width = frame.shape[:2]
    
    # 图例位置
    legend_x = 20
    legend_y = height - 200
    legend_width = 300
    legend_height = 160
    
    # 绘制半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), 
                  (50, 50, 50), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # 绘制边框
    cv2.rectangle(frame, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), 
                  (255, 255, 255), 2)
    
    # 图例标题
    cv2.putText(frame, "Ball Color Identification", (legend_x + 10, legend_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 运动球图例 - 红色
    cv2.circle(frame, (legend_x + 30, legend_y + 60), 12, (0, 0, 255), -1)
    cv2.circle(frame, (legend_x + 30, legend_y + 60), 15, (0, 0, 255), 2)
    cv2.putText(frame, "Moving Ball (Red)", (legend_x + 60, legend_y + 67), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 静止球图例 - 蓝色
    cv2.circle(frame, (legend_x + 30, legend_y + 95), 12, (255, 100, 0), -1)
    cv2.circle(frame, (legend_x + 30, legend_y + 95), 15, (255, 100, 0), 2)
    cv2.putText(frame, "Static Ball (Blue)", (legend_x + 60, legend_y + 102), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    
    # 轨迹图例 - 绿色
    cv2.line(frame, (legend_x + 20, legend_y + 125), (legend_x + 40, legend_y + 125), (0, 255, 0), 4)
    cv2.putText(frame, "Ball Trajectory (Green)", (legend_x + 60, legend_y + 132), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

def demo_ball_color_identification():
    """演示球颜色标识功能"""
    print("🚀 启动球颜色标识演示...")
    print("\n🎨 颜色标识系统:")
    print("🔴 红色: 运动球 - 正在移动的网球")
    print("🔵 蓝色: 静止球 - 保持静止的网球")  
    print("🟢 绿色: 轨迹线 - 球的运动路径")
    print("🟡 黄色: 边界框 - 检测区域边界")
    
    # 加载配置
    config = load_config()
    if not config:
        return
    
    # 显示配置信息
    print(f"\n🔧 球检测配置:")
    print(f"静止球阈值: {config['static_ball_movement_threshold_px']}px")
    print(f"静止帧数要求: {config['static_ball_frames_threshold']}帧")
    print(f"球尺寸范围: {config['min_ball_radius']}-{config['max_ball_radius']}px")
    print(f"边界检查: {'启用' if config['use_boundary'] else '禁用'}")
    print(f"静止球显示: {'启用' if config['draw_static_balls'] else '禁用'}")
    
    # 初始化球检测模块
    print(f"\n📹 初始化球检测模块...")
    ball_tracker = BallTracker(config.get('tracknet_model_path', None), config)
    
    # 打开视频
    video_path = config['video_input_path']
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {frame_width}x{frame_height}, {fps}FPS, {total_frames}帧")
    
    # 设置输出视频路径
    output_video_path = config.get('video_output_path', 'data/ball_color_identification_output.mp4')
    print(f"🎬 输出视频路径: {output_video_path}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    if not video_writer.isOpened():
        print(f"❌ 无法创建输出视频文件: {output_video_path}")
        cap.release()
        return
    
    print(f"\n🎯 开始处理视频并生成带标识的输出...")
    print(f"控制说明: 空格键暂停/继续, 'q'键退出, 's'键保存截图, 'o'键切换显示, 'm'键切换鼠标坐标")
    print(f"🖱️  鼠标功能: 移动显示坐标, 左键记录位置, 右键标记屏蔽区域")
    
    frame_count = 0
    paused = False
    static_count = 0
    moving_count = 0
    show_display = True  # 控制是否显示实时窗口
    global mouse_clicked, show_mouse_coords
    
    # 创建窗口并设置鼠标回调
    window_name = 'Ball Color Identification Demo'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print(f"\n🏁 视频处理完成，共处理 {frame_count} 帧")
                break
            
            frame_count += 1
        
        # 执行球检测
        detected_balls = ball_tracker.predict_ball(frame)
        
        # 应用高级处理获取运动球
        moving_balls = ball_tracker.advanced_ball_processing(detected_balls, frame_count)
        
        # 绘制显示帧
        display_frame = frame.copy()
        
        # 绘制轨迹（绿色）
        display_frame = ball_tracker.draw_trajectory(display_frame)
        
        # 绘制静止球（蓝色）
        display_frame = ball_tracker.draw_static_balls(display_frame)
        
        # 绘制运动球（红色）
        for ball in moving_balls:
            # 处理不同的数据格式
            if isinstance(ball, dict):
                center = (int(ball['x']), int(ball['y']))
                radius = int(ball.get('radius', 10))
            else:
                # 假设是 (x, y, radius) 元组格式
                center = (int(ball[0]), int(ball[1]))
                radius = int(ball[2]) if len(ball) > 2 else 10
            
            # 绘制实心圆
            cv2.circle(display_frame, center, radius, (0, 0, 255), -1)
            # 绘制边框
            cv2.circle(display_frame, center, radius + 3, (0, 0, 255), 2)
            # 添加标签
            cv2.putText(display_frame, "MOVING", (center[0] - 30, center[1] - radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 绘制边界框（如果启用）
        if config.get('draw_boundary', False):
            boundary_color = (0, 255, 255)  # 黄色边界
            cv2.rectangle(display_frame, 
                         (config['boundary_x1'], config['boundary_y1']), 
                         (config['boundary_x2'], config['boundary_y2']), 
                         boundary_color, 2)
        
        # 计算统计信息
        current_static = 0
        for ball_id, static_info in ball_tracker.static_ball_candidates.items():
            if static_info['frames_still'] >= config['static_ball_frames_threshold']:
                current_static += 1
        
        current_moving = len(moving_balls)
        
        # 更新计数
        static_count = max(static_count, current_static)
        if current_moving > 0:
            moving_count += 1
        
        # 绘制图例
        display_frame = draw_legend(display_frame)
        
        # 显示统计信息
        info_y = 30
        cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", 
                   (frame_width - 250, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Static Balls: {current_static}", 
                   (frame_width - 250, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        cv2.putText(display_frame, f"Moving Balls: {current_moving}", 
                   (frame_width - 250, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Raw Detections: {len(detected_balls)}", 
                   (frame_width - 250, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 🖱️ 绘制鼠标坐标显示
        if show_mouse_coords:
            display_frame = draw_mouse_coordinates(display_frame, mouse_x, mouse_y, mouse_clicked)
        
        # 重置点击状态
        if mouse_clicked:
            mouse_clicked = False
        
        # 添加输出状态指示
        output_status = f"Recording: {output_video_path}"
        cv2.putText(display_frame, output_status, (20, frame_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 状态指示 - 调整位置避免与鼠标坐标重叠
        status_text = "PAUSED" if paused else "PROCESSING"
        status_color = (0, 255, 255) if paused else (0, 255, 0)
        cv2.putText(display_frame, status_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # 写入视频文件
        video_writer.write(display_frame)
        
        # 显示窗口（可选）
        if show_display:
            cv2.imshow(window_name, display_frame)
            
            # 处理按键
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            
            if key == ord('q'):
                print("\n🛑 用户退出处理")
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'⏸️  暂停' if paused else '▶️  继续'}")
            elif key == ord('s'):
                screenshot_name = f"ball_color_demo_frame_{frame_count:04d}.jpg"
                cv2.imwrite(screenshot_name, display_frame)
                print(f"📸 截图已保存: {screenshot_name}")
            elif key == ord('o'):
                show_display = not show_display
                if not show_display:
                    cv2.destroyAllWindows()
                print(f"{'👁️  显示窗口: 开启' if show_display else '👁️  显示窗口: 关闭'}")
            elif key == ord('m'):
                show_mouse_coords = not show_mouse_coords
                print(f"{'🖱️  鼠标坐标显示: 开启' if show_mouse_coords else '🖱️  鼠标坐标显示: 关闭'}")
        else:
            # 非显示模式下的简单按键检测
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 用户退出处理")
                break
            elif key == ord('o'):
                show_display = True
                print("👁️  显示窗口: 开启")
        
        # 显示进度
        if frame_count % 30 == 0 and not paused:
            progress = (frame_count / total_frames) * 100
            print(f"📈 处理进度: {progress:.1f}% ({frame_count}/{total_frames}) - 正在写入视频...")
    
    # 清理资源
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    # 检查输出文件
    import os
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
        print(f"\n✅ 输出视频已保存: {output_video_path}")
        print(f"📁 文件大小: {file_size:.1f} MB")
    else:
        print(f"\n❌ 输出视频保存失败: {output_video_path}")
    
    # 显示最终统计
    print(f"\n📊 处理统计结果:")
    print(f"最大静止球数量: {static_count}")
    print(f"检测到运动球的帧数: {moving_count}")
    print(f"静止球识别率: {static_count/max(1, frame_count)*100:.1f}%")
    print(f"运动球识别率: {moving_count/max(1, frame_count)*100:.1f}%")
    
    if len(ball_tracker.tracked_balls_history) > 0:
        print(f"总轨迹长度: {len(ball_tracker.tracked_balls_history)} 点")
        
        # 计算轨迹总距离
        total_distance = 0
        if len(ball_tracker.tracked_balls_history) > 1:
            for i in range(1, len(ball_tracker.tracked_balls_history)):
                p1 = np.array(ball_tracker.tracked_balls_history[i-1]['coords'])
                p2 = np.array(ball_tracker.tracked_balls_history[i]['coords'])
                total_distance += np.linalg.norm(p2 - p1)
        print(f"轨迹总距离: {total_distance:.1f}px")
    
    print(f"\n🎉 球颜色标识处理完成!")
    print(f"🔴 红色圆圈表示运动球")
    print(f"🔵 蓝色圆圈表示静止球") 
    print(f"🟢 绿色线条表示球的运动轨迹")
    print(f"🟡 黄色边框表示检测边界")

def main():
    """主函数"""
    try:
        demo_ball_color_identification()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断演示")
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 