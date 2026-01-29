#!/usr/bin/env python3
"""
分析姿态检测性能异常
找出 41.26ms 峰值的原因
"""

import cv2
import yaml
import time
import numpy as np
import coremltools as ct
from pose_estimator import PoseEstimator
from roi_manager import ROIManager


def analyze_pose_detection_spikes(config_path, video_path, num_frames=200):
    """
    详细分析姿态检测性能，找出峰值原因
    """
    print("=" * 80)
    print("🔍 姿态检测性能异常分析")
    print("=" * 80)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    # 初始化
    roi_manager = ROIManager(config)
    pose_module = PoseEstimator(config['yolo_pose_model_path'], config, roi_manager)
    
    print(f"\n📹 分析 {num_frames} 帧...")
    
    # 记录每帧的详细信息
    frame_times = []
    frame_details = []
    
    for frame_num in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 测试姿态检测
        start = time.time()
        pose_results = pose_module.get_keypoints(frame)
        elapsed = (time.time() - start) * 1000
        
        frame_times.append(elapsed)
        frame_details.append({
            'frame': frame_num,
            'time': elapsed,
            'num_persons': len(pose_results) if pose_results else 0,
            'has_keypoints': bool(pose_results)
        })
        
        if (frame_num + 1) % 50 == 0:
            print(f"   处理帧 {frame_num + 1}/{num_frames}")
    
    cap.release()
    
    # 统计分析
    times = np.array(frame_times)
    avg = np.mean(times)
    std = np.std(times)
    min_t = np.min(times)
    max_t = np.max(times)
    median = np.median(times)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)
    
    print("\n" + "=" * 80)
    print("📊 统计结果")
    print("=" * 80)
    print(f"   平均值: {avg:.2f}ms")
    print(f"   中位数: {median:.2f}ms")
    print(f"   标准差: {std:.2f}ms")
    print(f"   最小值: {min_t:.2f}ms")
    print(f"   最大值: {max_t:.2f}ms")
    print(f"   95分位: {p95:.2f}ms")
    print(f"   99分位: {p99:.2f}ms")
    
    # 找出异常帧
    threshold = avg + 2 * std  # 2倍标准差
    outliers = [(i, t) for i, t in enumerate(frame_times) if t > threshold]
    
    print("\n" + "=" * 80)
    print(f"🔥 异常帧分析 (>{threshold:.2f}ms)")
    print("=" * 80)
    print(f"   异常帧数量: {len(outliers)}/{num_frames} ({len(outliers)/num_frames*100:.1f}%)")
    
    if outliers:
        print(f"\n   前10个异常帧:")
        for i, (frame_idx, frame_time) in enumerate(outliers[:10]):
            detail = frame_details[frame_idx]
            print(f"      [{i+1}] 帧{frame_idx}: {frame_time:.2f}ms "
                  f"(人数:{detail['num_persons']})")
    
    # 分析峰值原因
    print("\n" + "=" * 80)
    print("💡 峰值原因分析")
    print("=" * 80)
    
    # 1. 首帧效应
    first_10 = times[:10]
    rest = times[10:]
    print(f"\n1. 首帧效应:")
    print(f"   前10帧平均: {np.mean(first_10):.2f}ms")
    print(f"   其余帧平均: {np.mean(rest):.2f}ms")
    if np.mean(first_10) > np.mean(rest) * 1.5:
        print(f"   ⚠️  首帧有明显预热效应")
    
    # 2. 检测人数影响
    with_person = [d['time'] for d in frame_details if d['num_persons'] > 0]
    without_person = [d['time'] for d in frame_details if d['num_persons'] == 0]
    
    print(f"\n2. 检测人数影响:")
    if with_person:
        print(f"   有人帧平均: {np.mean(with_person):.2f}ms ({len(with_person)}帧)")
    if without_person:
        print(f"   无人帧平均: {np.mean(without_person):.2f}ms ({len(without_person)}帧)")
    
    # 3. 时间分布
    print(f"\n3. 时间分布:")
    bins = [0, 12, 15, 20, 30, 100]
    labels = ['<12ms', '12-15ms', '15-20ms', '20-30ms', '>30ms']
    hist, _ = np.histogram(times, bins=bins)
    for label, count in zip(labels, hist):
        percentage = count / len(times) * 100
        print(f"   {label:<10} {count:>4}帧 ({percentage:>5.1f}%)")
    
    # 4. Core ML 特性
    print(f"\n4. Core ML 特性:")
    print(f"   ⚠️  Core ML 模型首次推理通常较慢（JIT编译、优化）")
    print(f"   ⚠️  可能的原因:")
    print(f"      - 模型预热不足")
    print(f"      - 内存分配")
    print(f"      - 硬件调度")
    print(f"      - 系统负载波动")
    
    # 建议
    print("\n" + "=" * 80)
    print("🎯 优化建议")
    print("=" * 80)
    
    if np.mean(first_10) > avg * 1.5:
        print(f"\n1. 添加预热步骤:")
        print(f"   ```python")
        print(f"   # 初始化后预热模型")
        print(f"   dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)")
        print(f"   for _ in range(5):")
        print(f"       pose_module.get_keypoints(dummy_frame)")
        print(f"   ```")
    
    if max_t > avg * 3:
        print(f"\n2. 降低检测频率:")
        print(f"   - 当前峰值: {max_t:.2f}ms")
        print(f"   - 每2帧检测可避免峰值影响")
        print(f"   - 预期平均: {avg/2:.2f}ms")
    
    if std > avg * 0.3:
        print(f"\n3. 稳定性优化:")
        print(f"   - 标准差较大 ({std:.2f}ms)")
        print(f"   - 考虑固定 CPU 频率")
        print(f"   - 关闭后台任务")
    
    print(f"\n4. 计算单元优化:")
    print(f"   - 当前: ALL (自动选择)")
    print(f"   - 可尝试: ANE (更稳定)")
    
    return frame_times, frame_details


if __name__ == "__main__":
    config_path = "configs/yolo26_tennis_config.yaml"
    video_path = "data/16.10.mp4"
    
    times, details = analyze_pose_detection_spikes(config_path, video_path, num_frames=200)
    
    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)
