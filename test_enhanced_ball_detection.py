#!/usr/bin/env python3
"""
🎾 增强球检测测试脚本
测试改进的静止球过滤和运动球追踪功能
"""

import cv2
import time
import yaml
import numpy as np
from ball_tracker import BallTracker

def load_config(config_path="configs/enhanced_ball_detection_config.yaml"):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        print(f"✅ 配置加载成功: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return None

def analyze_detection_results(ball_module, frame_results):
    """分析检测结果统计"""
    total_detections = sum(len(results['raw_detections']) for results in frame_results)
    total_filtered = sum(len(results['filtered_balls']) for results in frame_results)
    total_final = sum(1 if results['final_ball'] else 0 for results in frame_results)
    
    print(f"\n📊 检测结果统计分析:")
    print(f"总原始检测数: {total_detections}")
    print(f"过滤后球数: {total_filtered}")
    print(f"最终追踪球数: {total_final}")
    print(f"静止球过滤率: {((total_detections - total_filtered) / max(1, total_detections) * 100):.1f}%")
    print(f"最终追踪成功率: {(total_final / len(frame_results) * 100):.1f}%")
    
    # 分析球的移动历史
    if hasattr(ball_module, 'ball_movement_history'):
        static_balls = 0
        moving_balls = 0
        
        for ball_key, history in ball_module.ball_movement_history.items():
            if history['static_count'] > history['moving_count']:
                static_balls += 1
            else:
                moving_balls += 1
        
        print(f"识别的静止球数量: {static_balls}")
        print(f"识别的运动球数量: {moving_balls}")

def draw_color_legend(frame):
    """绘制颜色标识图例"""
    height, width = frame.shape[:2]
    
    # 图例背景
    legend_x = width - 280
    legend_y = 60
    legend_width = 270
    legend_height = 120
    
    # 绘制半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), 
                  (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # 图例标题
    cv2.putText(frame, "Ball Color Legend:", (legend_x + 10, legend_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 运动球图例 - 红色
    cv2.circle(frame, (legend_x + 30, legend_y + 50), 8, (0, 0, 255), -1)
    cv2.putText(frame, "Moving Ball", (legend_x + 50, legend_y + 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 静止球图例 - 蓝色
    cv2.circle(frame, (legend_x + 30, legend_y + 75), 8, (255, 100, 0), -1)
    cv2.putText(frame, "Static Ball", (legend_x + 50, legend_y + 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
    
    # 轨迹图例 - 绿色
    cv2.line(frame, (legend_x + 20, legend_y + 100), (legend_x + 40, legend_y + 100), (0, 255, 0), 3)
    cv2.putText(frame, "Ball Trajectory", (legend_x + 50, legend_y + 105), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def test_enhanced_ball_detection():
    """测试增强球检测功能"""
    print("🚀 启动增强球检测测试...")
    print("\n🎨 颜色标识说明:")
    print("🔴 红色圆圈: 运动球 (MOVING)")
    print("🔵 蓝色圆圈: 静止球 (STATIC)")  
    print("🟢 绿色线条: 球的运动轨迹")
    print("🟡 黄色矩形: 监控边界区域")
    
    # 加载配置
    config = load_config()
    if not config:
        return
    
    # 显示关键参数
    print(f"\n🔧 关键检测参数:")
    print(f"静止球移动阈值: {config['static_ball_movement_threshold_px']}px")
    print(f"静止球帧数阈值: {config['static_ball_frames_threshold']}帧")
    print(f"最小球移动距离: {config['min_ball_movement']}px")
    print(f"球尺寸范围: {config['min_ball_radius']}-{config['max_ball_radius']}px")
    print(f"边界区域: ({config['boundary_x1']},{config['boundary_y1']}) - ({config['boundary_x2']},{config['boundary_y2']})")
    
    # 初始化球检测模块
    print(f"\n📹 初始化球检测模块...")
    ball_module = BallTracker(config.get('tracknet_model_path', None), config)
    
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
    
    # 测试配置
    test_frames = 150  # 测试帧数
    display_interval = 10  # 显示间隔
    
    print(f"\n🎯 开始测试 {test_frames} 帧...")
    
    frame_results = []
    start_time = time.time()
    
    for frame_num in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"视频结束于第 {frame_num} 帧")
            break
        
        # 执行球检测
        raw_detections = ball_module.predict_ball(frame)
        
        # 应用高级处理
        final_balls = ball_module.advanced_ball_processing(raw_detections, frame_num)
        
        # 记录结果
        result = {
            'frame_num': frame_num,
            'raw_detections': raw_detections,
            'filtered_balls': ball_module.filter_static_candidates(raw_detections, frame_num),
            'final_ball': final_balls[0] if final_balls else None
        }
        frame_results.append(result)
        
        # 定期显示进度和统计
        if frame_num % display_interval == 0:
            elapsed = time.time() - start_time
            fps_current = (frame_num + 1) / elapsed if elapsed > 0 else 0
            
            print(f"帧 {frame_num:3d}: 原始={len(raw_detections)}, "
                  f"过滤后={len(result['filtered_balls'])}, "
                  f"最终={'✅' if result['final_ball'] else '❌'}, "
                  f"处理速度={fps_current:.2f}FPS")
            
            # 显示检测结果（带可视化）
            if raw_detections or final_balls:
                display_frame = frame.copy()
                
                # 绘制原始检测（蓝色）
                for detection in raw_detections:
                    cv2.circle(display_frame, (int(detection[0]), int(detection[1])), 8, (255, 0, 0), 2)
                    cv2.putText(display_frame, "RAW", (int(detection[0])-15, int(detection[1])-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # 绘制最终球（红色）
                if final_balls:
                    final_ball = final_balls[0]
                    cv2.circle(display_frame, (int(final_ball[0]), int(final_ball[1])), 10, (0, 0, 255), -1)
                    cv2.putText(display_frame, "FINAL", (int(final_ball[0])-20, int(final_ball[1])+25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 绘制轨迹
                display_frame = ball_module.draw_trajectory(display_frame)
                
                # 绘制静态球（灰色）
                display_frame = ball_module.draw_static_balls(display_frame)
                
                # 显示帧信息
                info_text = f"Frame {frame_num}: Raw={len(raw_detections)}, Final={'YES' if final_balls else 'NO'}"
                cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 绘制颜色图例
                display_frame = draw_color_legend(display_frame)
                
                # 显示图像（按 'q' 退出，按空格暂停）
                cv2.imshow('Enhanced Ball Detection Test', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户退出测试")
                    break
                elif key == ord(' '):
                    print("暂停 - 按任意键继续...")
                    cv2.waitKey(0)
    
    # 测试完成
    total_time = time.time() - start_time
    avg_fps = len(frame_results) / total_time if total_time > 0 else 0
    
    print(f"\n✅ 测试完成!")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"平均处理速度: {avg_fps:.2f}FPS")
    
    # 分析结果
    analyze_detection_results(ball_module, frame_results)
    
    # 显示轨迹统计
    if len(ball_module.tracked_balls_history) > 0:
        print(f"\n🎯 轨迹分析:")
        print(f"总轨迹点数: {len(ball_module.tracked_balls_history)}")
        
        if len(ball_module.tracked_balls_history) > 1:
            total_distance = 0
            for i in range(1, len(ball_module.tracked_balls_history)):
                p1 = np.array(ball_module.tracked_balls_history[i-1]['coords'])
                p2 = np.array(ball_module.tracked_balls_history[i]['coords'])
                total_distance += np.linalg.norm(p2 - p1)
            
            avg_speed = ball_module._calculate_ball_speed()
            print(f"总轨迹长度: {total_distance:.1f}px")
            print(f"平均球速: {avg_speed:.1f}px/frame")
    
    # 清理
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n🎉 测试完成! 请查看上方的统计结果。")

def main():
    """主函数"""
    try:
        test_enhanced_ball_detection()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 