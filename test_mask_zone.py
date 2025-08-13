#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚫 屏蔽区域功能测试脚本
测试在指定区域(661,391)的20px矩形框内屏蔽球检测功能
"""

import cv2
import yaml
import numpy as np
from ball_tracker import BallTracker
import os

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_test_frame():
    """创建测试帧，包含多个球，其中一个在屏蔽区域内"""
    # 创建一个1280x720的测试帧 (常见的HD分辨率)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 填充背景色（绿色，模拟网球场）
    frame[:] = (34, 139, 34)  # 森林绿
    
    # 测试球的位置
    test_balls = [
        (500, 300, "正常球1"),           # 在正常区域的球
        (800, 500, "正常球2"),           # 在正常区域的球
        (661+10, 391+10, "屏蔽区域内的球"),  # 在屏蔽区域内的球 (661,391) 20x20区域的中心
        (1000, 200, "正常球3"),          # 在正常区域的球
        (670, 400, "接近屏蔽区域的球"),    # 接近但不在屏蔽区域内的球
    ]
    
    # 绘制测试球 (网球颜色 - 荧光黄绿色)
    for x, y, name in test_balls:
        # 使用HSV颜色空间创建符合网球检测的颜色
        # 网球颜色：HSV范围大约是 (25, 100, 200) 
        tennis_ball_color = (47, 255, 173)  # BGR格式的荧光黄绿色，对应HSV约(30, 175, 255)
        
        # 绘制球的外圈和填充
        cv2.circle(frame, (x, y), 12, tennis_ball_color, -1)  # 网球色填充
        cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)     # 白色边框
        
        # 添加一些纹理使其更像真实网球
        cv2.circle(frame, (x, y), 8, (30, 220, 150), -1)      # 内部稍暗的绿色
        cv2.circle(frame, (x, y), 4, tennis_ball_color, -1)   # 中心亮色
        
        # 添加球的标签
        cv2.putText(frame, name, (x-50, y-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame, test_balls

def test_mask_zone_detection():
    """测试屏蔽区域检测功能"""
    print("🚫 开始测试屏蔽区域功能...")
    
    # 加载配置
    config_path = "configs/mask_zone_config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件 {config_path} 不存在")
        return
    
    config = load_config(config_path)
    
    # 创建球检测器
    ball_tracker = BallTracker(model_path="", config=config)
    
    # 创建测试帧
    test_frame, test_balls = create_test_frame()
    
    print(f"\n📊 测试配置:")
    print(f"  - 屏蔽区域启用: {config['use_mask_zones']}")
    print(f"  - 屏蔽区域数量: {len(config['mask_zones'])}")
    for i, zone in enumerate(config['mask_zones']):
        print(f"  - 屏蔽区域{i+1}: ({zone['x']}, {zone['y']}) {zone['width']}x{zone['height']}")
    
    print(f"\n🎾 测试球位置:")
    for i, (x, y, name) in enumerate(test_balls):
        print(f"  - 球{i+1}: ({x}, {y}) - {name}")
    
    # 执行球检测
    print(f"\n🔍 执行球检测...")
    detected_balls = ball_tracker.predict_ball(test_frame)
    
    # 分析结果
    print(f"\n📊 检测结果:")
    print(f"  - 原始球数量: {len(test_balls)}")
    print(f"  - 检测到球数量: {len(detected_balls)}")
    print(f"  - 被屏蔽的球数量: {len(test_balls) - len(detected_balls)}")
    
    if detected_balls:
        print(f"  - 检测到的球坐标:")
        for i, (x, y) in enumerate(detected_balls):
            print(f"    球{i+1}: ({x}, {y})")
    
    # 验证屏蔽区域功能
    mask_zone = config['mask_zones'][0]
    zone_x1, zone_y1 = mask_zone['x'], mask_zone['y']
    zone_x2, zone_y2 = zone_x1 + mask_zone['width'], zone_y1 + mask_zone['height']
    
    masked_balls = []
    unmasked_balls = []
    
    for x, y, name in test_balls:
        if zone_x1 <= x <= zone_x2 and zone_y1 <= y <= zone_y2:
            masked_balls.append((x, y, name))
        else:
            unmasked_balls.append((x, y, name))
    
    print(f"\n✅ 预期结果分析:")
    print(f"  - 应该被屏蔽的球: {len(masked_balls)}")
    for x, y, name in masked_balls:
        print(f"    {name}: ({x}, {y})")
    
    print(f"  - 应该保留的球: {len(unmasked_balls)}")
    for x, y, name in unmasked_balls:
        print(f"    {name}: ({x}, {y})")
    
    # 创建结果可视化
    result_frame = test_frame.copy()
    
    # 绘制屏蔽区域和轨迹
    result_frame = ball_tracker.draw_trajectory(result_frame)
    
    # 绘制检测结果
    for i, (x, y) in enumerate(detected_balls):
        cv2.circle(result_frame, (x, y), 15, (0, 0, 255), 3)  # 红色圆圈表示检测到的球
        cv2.putText(result_frame, f"DETECTED", (x-30, y+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # 添加测试信息
    info_y = 50
    cv2.putText(result_frame, f"Mask Zone Test Results", (10, info_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result_frame, f"Original balls: {len(test_balls)}, Detected: {len(detected_balls)}, Masked: {len(test_balls) - len(detected_balls)}", 
               (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 保存结果图像
    output_path = "mask_zone_test_result.jpg"
    cv2.imwrite(output_path, result_frame)
    print(f"\n💾 结果图像已保存: {output_path}")
    
    # 验证功能是否正常工作
    expected_detected = len(unmasked_balls)
    actual_detected = len(detected_balls)
    
    print(f"\n🎯 功能验证:")
    if actual_detected == expected_detected:
        print(f"✅ 屏蔽区域功能工作正常！")
        print(f"   预期检测到 {expected_detected} 个球，实际检测到 {actual_detected} 个球")
    else:
        print(f"❌ 屏蔽区域功能可能存在问题")
        print(f"   预期检测到 {expected_detected} 个球，实际检测到 {actual_detected} 个球")
    
    return result_frame

def main():
    """主函数"""
    print("🎾 屏蔽区域功能测试")
    print("=" * 50)
    
    try:
        result_frame = test_mask_zone_detection()
        
        print(f"\n🎉 测试完成！")
        print(f"📝 查看 mask_zone_test_result.jpg 文件可以看到可视化结果")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 