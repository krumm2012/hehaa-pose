#!/usr/bin/env python3
"""
球检测详细日志测试脚本
用于调试球识别不到的问题
"""

import cv2
import yaml
import time
from ball_tracker import BallTracker

def load_config(config_path="configs/balanced_config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_ball_detection_with_logs(config_path="configs/balanced_config.yaml", max_frames=50):
    """测试球检测并输出详细日志"""
    print(f"🔍 球检测详细日志测试开始...")
    print(f"📄 配置文件: {config_path}")
    print(f"🎯 测试帧数: {max_frames}")
    print("="*80)
    
    # 加载配置
    config = load_config(config_path)
    
    # 确保调试日志启用
    if 'ball_detection_debug' not in config:
        config['ball_detection_debug'] = {}
    config['ball_detection_debug']['enabled'] = True
    
    # 显示配置信息
    debug_config = config['ball_detection_debug']
    print("🔧 调试配置:")
    print(f"   HSV检测日志: {debug_config.get('log_hsv_detection', False)}")
    print(f"   边界检查日志: {debug_config.get('log_boundary_check', False)}")
    print(f"   尺寸过滤日志: {debug_config.get('log_size_filtering', False)}")
    print(f"   静态过滤日志: {debug_config.get('log_static_filtering', False)}")
    print(f"   模拟检测日志: {debug_config.get('log_simulation', False)}")
    print(f"   保存调试帧: {debug_config.get('save_debug_frames', False)}")
    print("="*80)
    
    # 初始化球检测器
    ball_tracker = BallTracker(config.get('tracknet_model_path', None), config)
    
    # 打开视频
    video_path = config['video_input_path']
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 视频信息:")
    print(f"   总帧数: {total_frames}")
    print(f"   帧率: {fps:.2f} FPS")
    print(f"   时长: {total_frames/fps:.2f} 秒")
    print("="*80)
    
    # 统计信息
    detection_stats = {
        'total_frames': 0,
        'frames_with_balls': 0,
        'total_balls_detected': 0,
        'hsv_success_frames': 0,
        'simulation_frames': 0,
        'boundary_filtered': 0
    }
    
    start_time = time.time()
    
    try:
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            detection_stats['total_frames'] += 1
            
            print(f"\n🎯 处理帧 {frame_count}")
            print("-" * 60)
            
            # 球检测
            detected_balls = ball_tracker.predict_ball(frame)
            
            # 更新统计
            if detected_balls:
                detection_stats['frames_with_balls'] += 1
                detection_stats['total_balls_detected'] += len(detected_balls)
            
            # 简要总结
            print(f"📊 帧 {frame_count} 总结: 检测到 {len(detected_balls)} 个球")
            if detected_balls:
                for i, ball in enumerate(detected_balls):
                    print(f"   球 {i+1}: 位置 ({ball[0]}, {ball[1]})")
            
            # 每10帧显示一次进度
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                progress = frame_count / max_frames * 100
                print(f"\n⏱️  进度: {progress:.1f}% ({frame_count}/{max_frames}), 耗时: {elapsed:.2f}s")
    
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
    
    finally:
        cap.release()
        
        # 显示最终统计
        elapsed_time = time.time() - start_time
        print("\n" + "="*80)
        print("📊 球检测详细统计报告")
        print("="*80)
        print(f"测试时长: {elapsed_time:.2f} 秒")
        print(f"处理速度: {detection_stats['total_frames']/elapsed_time:.2f} FPS")
        print()
        print(f"总处理帧数: {detection_stats['total_frames']}")
        print(f"有球检测帧数: {detection_stats['frames_with_balls']}")
        print(f"球检测成功率: {detection_stats['frames_with_balls']/detection_stats['total_frames']*100:.1f}%")
        print(f"总球检测数量: {detection_stats['total_balls_detected']}")
        print(f"平均每帧球数: {detection_stats['total_balls_detected']/detection_stats['total_frames']:.2f}")
        print()
        
        # 诊断建议
        print("🔍 诊断建议:")
        if detection_stats['frames_with_balls'] == 0:
            print("❌ 完全没有检测到球，可能原因:")
            print("   1. HSV颜色范围不适合当前视频")
            print("   2. 球的尺寸超出设定范围")
            print("   3. 边界设置过于严格")
            print("   4. 视频中没有黄绿色网球")
        elif detection_stats['frames_with_balls'] < detection_stats['total_frames'] * 0.3:
            print("⚠️  球检测率较低，建议:")
            print("   1. 调整HSV颜色范围")
            print("   2. 放宽球尺寸限制")
            print("   3. 检查边界设置")
        else:
            print("✅ 球检测率正常")
        
        print(f"\n💡 要启用调试帧保存，请在配置文件中设置:")
        print(f"   ball_detection_debug:")
        print(f"     save_debug_frames: true")
        print(f"     debug_frame_interval: 5")

def main():
    """主函数"""
    import sys
    
    config_path = "configs/balanced_config.yaml"
    max_frames = 30
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if len(sys.argv) > 2:
        max_frames = int(sys.argv[2])
    
    test_ball_detection_with_logs(config_path, max_frames)

if __name__ == "__main__":
    main() 