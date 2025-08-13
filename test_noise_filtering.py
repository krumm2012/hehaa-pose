#!/usr/bin/env python3
"""
🧹 噪点过滤测试脚本
测试改进的噪点过滤功能，对比过滤前后的检测效果
"""

import cv2
import time
import yaml
import numpy as np
from ball_tracker import BallTracker

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

def create_comparison_display(original_frame, filtered_detections, raw_detections):
    """创建对比显示，展示过滤前后的差异"""
    # 创建并排显示
    display_height = original_frame.shape[0]
    display_width = original_frame.shape[1] * 2
    comparison_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
    # 左侧：原始检测结果
    left_frame = original_frame.copy()
    for detection in raw_detections:
        cv2.circle(left_frame, (int(detection[0]), int(detection[1])), 12, (0, 255, 255), 2)  # 黄色 - 原始检测
        cv2.putText(left_frame, "RAW", (int(detection[0])-15, int(detection[1])-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # 右侧：过滤后结果
    right_frame = original_frame.copy()
    for detection in filtered_detections:
        cv2.circle(right_frame, (int(detection[0]), int(detection[1])), 12, (0, 0, 255), 2)  # 红色 - 过滤后
        cv2.putText(right_frame, "FILTERED", (int(detection[0])-25, int(detection[1])-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # 添加标题
    cv2.putText(left_frame, f"Raw Detections: {len(raw_detections)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(right_frame, f"Filtered: {len(filtered_detections)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 合并到对比帧
    comparison_frame[:, :original_frame.shape[1]] = left_frame
    comparison_frame[:, original_frame.shape[1]:] = right_frame
    
    # 添加分割线
    cv2.line(comparison_frame, (original_frame.shape[1], 0), 
             (original_frame.shape[1], display_height), (255, 255, 255), 2)
    
    return comparison_frame

def analyze_noise_filtering_results(results):
    """分析噪点过滤结果"""
    total_raw = sum(len(r['raw_detections']) for r in results)
    total_filtered = sum(len(r['filtered_detections']) for r in results)
    total_noise_removed = total_raw - total_filtered
    
    print(f"\n📊 噪点过滤效果分析:")
    print(f"总原始检测数: {total_raw}")
    print(f"过滤后检测数: {total_filtered}")
    print(f"移除噪点数量: {total_noise_removed}")
    print(f"噪点过滤率: {(total_noise_removed / max(1, total_raw) * 100):.1f}%")
    
    # 分析过滤效果的稳定性
    filter_rates = []
    for r in results:
        raw_count = len(r['raw_detections'])
        filtered_count = len(r['filtered_detections'])
        if raw_count > 0:
            filter_rate = (raw_count - filtered_count) / raw_count
            filter_rates.append(filter_rate)
    
    if filter_rates:
        avg_filter_rate = np.mean(filter_rates)
        std_filter_rate = np.std(filter_rates)
        print(f"平均过滤率: {avg_filter_rate:.1%}")
        print(f"过滤率标准差: {std_filter_rate:.1%}")
        print(f"过滤稳定性: {'稳定' if std_filter_rate < 0.2 else '不稳定'}")

def test_noise_filtering():
    """测试噪点过滤功能"""
    print("🧹 启动噪点过滤测试...")
    print("\n🎨 显示说明:")
    print("🟡 黄色圆圈: 原始检测结果 (包含噪点)")
    print("🔴 红色圆圈: 过滤后结果 (去除噪点)")
    print("左侧: 原始检测 | 右侧: 噪点过滤")
    
    # 加载配置
    config = load_config()
    if not config:
        return
    
    # 显示关键过滤参数
    print(f"\n🔧 噪点过滤参数:")
    print(f"最小球半径: {config['min_ball_radius']}px")
    print(f"最大球半径: {config['max_ball_radius']}px")
    print(f"最小面积: {config['min_ball_area']}px²")
    print(f"最大面积: {config['max_ball_area']}px²")
    print(f"质量阈值: {config['noise_filter_quality_threshold']}")
    print(f"圆度阈值: {config['noise_filter_circularity_threshold']}")
    print(f"最小边距: {config['noise_filter_min_edge_distance']}px")
    
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
    test_frames = 100  # 测试帧数
    display_interval = 5  # 显示间隔
    
    print(f"\n🎯 开始噪点过滤测试 {test_frames} 帧...")
    
    frame_results = []
    start_time = time.time()
    
    for frame_num in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"视频结束于第 {frame_num} 帧")
            break
        
        # 执行原始球检测 (不应用高级过滤)
        raw_detections = ball_module._detect_with_hsv(frame)
        
        # 应用边界检查
        boundary_filtered = ball_module._apply_boundary_check(raw_detections)
        
        # 记录结果
        result = {
            'frame_num': frame_num,
            'raw_detections': raw_detections,
            'filtered_detections': boundary_filtered
        }
        frame_results.append(result)
        
        # 定期显示进度和对比效果
        if frame_num % display_interval == 0:
            elapsed = time.time() - start_time
            fps_current = (frame_num + 1) / elapsed if elapsed > 0 else 0
            
            noise_removed = len(raw_detections) - len(boundary_filtered)
            
            print(f"帧 {frame_num:3d}: 原始={len(raw_detections)}, "
                  f"过滤后={len(boundary_filtered)}, "
                  f"去噪={noise_removed}, "
                  f"处理速度={fps_current:.2f}FPS")
            
            # 创建对比显示
            if raw_detections or boundary_filtered:
                comparison_display = create_comparison_display(frame, boundary_filtered, raw_detections)
                
                # 添加帧信息
                info_text = f"Frame {frame_num}: Raw={len(raw_detections)}, Filtered={len(boundary_filtered)}, Noise Removed={noise_removed}"
                cv2.putText(comparison_display, info_text, (10, comparison_display.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 显示图像
                cv2.imshow('Noise Filtering Comparison', comparison_display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户退出测试")
                    break
                elif key == ord(' '):
                    print("暂停 - 按任意键继续...")
                    cv2.waitKey(0)
                elif key == ord('s'):
                    # 保存对比图像
                    save_path = f"debug_noise_filter_frame_{frame_num}.jpg"
                    cv2.imwrite(save_path, comparison_display)
                    print(f"保存对比图像: {save_path}")
    
    # 测试完成
    total_time = time.time() - start_time
    avg_fps = len(frame_results) / total_time if total_time > 0 else 0
    
    print(f"\n✅ 噪点过滤测试完成!")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"平均处理速度: {avg_fps:.2f}FPS")
    
    # 分析噪点过滤效果
    analyze_noise_filtering_results(frame_results)
    
    # 显示噪点过滤的详细统计
    frame_with_noise = sum(1 for r in frame_results if len(r['raw_detections']) > len(r['filtered_detections']))
    frame_no_noise = len(frame_results) - frame_with_noise
    
    print(f"\n🎯 检测到噪点的帧数: {frame_with_noise}/{len(frame_results)} ({frame_with_noise/len(frame_results)*100:.1f}%)")
    print(f"无噪点的帧数: {frame_no_noise}/{len(frame_results)} ({frame_no_noise/len(frame_results)*100:.1f}%)")
    
    # 清理
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n🎉 噪点过滤测试完成! 按 's' 可保存对比图像。")

def main():
    """主函数"""
    try:
        test_noise_filtering()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 