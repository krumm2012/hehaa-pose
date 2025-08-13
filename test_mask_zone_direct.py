#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚫 屏蔽区域功能直接测试脚本
直接测试屏蔽区域过滤功能，不依赖HSV检测
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

def test_mask_zone_filtering():
    """直接测试屏蔽区域过滤逻辑"""
    print("🚫 开始直接测试屏蔽区域过滤功能...")
    
    # 加载配置
    config_path = "configs/mask_zone_config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件 {config_path} 不存在")
        return
    
    config = load_config(config_path)
    
    # 创建球检测器
    ball_tracker = BallTracker(model_path="", config=config)
    
    # 测试球的位置 - 直接使用坐标列表
    test_balls = [
        (500, 300, "正常球1"),           # 在正常区域的球
        (800, 500, "正常球2"),           # 在正常区域的球
        (671, 401, "第一屏蔽区域内的球"),  # 在第一屏蔽区域内的球 (661,391) 20x20区域内
        (1000, 200, "正常球3"),          # 在正常区域的球
        (670, 400, "接近第一屏蔽区域的球"), # 接近但不在第一屏蔽区域内的球
        (661, 391, "第一屏蔽区域边界的球"), # 在第一屏蔽区域边界上的球
        (681, 411, "第一屏蔽区域边界2的球"), # 在第一屏蔽区域另一边界上的球
        (1235, 374, "第二屏蔽区域内的球"),  # 在第二屏蔽区域内的球 (1225,364) 20x20区域内
        (1225, 364, "第二屏蔽区域边界的球"), # 在第二屏蔽区域边界上的球
        (1245, 384, "第二屏蔽区域边界2的球"), # 在第二屏蔽区域另一边界上的球
        (1200, 350, "接近第二屏蔽区域的球"), # 接近但不在第二屏蔽区域内的球
    ]
    
    print(f"\n📊 测试配置:")
    print(f"  - 屏蔽区域启用: {config['use_mask_zones']}")
    print(f"  - 屏蔽区域数量: {len(config['mask_zones'])}")
    for i, zone in enumerate(config['mask_zones']):
        print(f"  - 屏蔽区域{i+1}: ({zone['x']}, {zone['y']}) {zone['width']}x{zone['height']}")
        print(f"    范围: x[{zone['x']}-{zone['x']+zone['width']}], y[{zone['y']}-{zone['y']+zone['height']}]")
    
    print(f"\n🎾 测试球位置:")
    for i, (x, y, name) in enumerate(test_balls):
        print(f"  - 球{i+1}: ({x}, {y}) - {name}")
    
    # 将测试球转换为坐标列表
    input_balls = [(x, y) for x, y, name in test_balls]
    
    # 直接测试屏蔽区域过滤功能
    print(f"\n🔍 执行屏蔽区域过滤...")
    filtered_balls = ball_tracker._apply_mask_zones_check(input_balls)
    
    # 分析结果
    print(f"\n📊 过滤结果:")
    print(f"  - 输入球数量: {len(input_balls)}")
    print(f"  - 过滤后球数量: {len(filtered_balls)}")
    print(f"  - 被屏蔽的球数量: {len(input_balls) - len(filtered_balls)}")
    
    if filtered_balls:
        print(f"  - 通过过滤的球坐标:")
        for i, (x, y) in enumerate(filtered_balls):
            print(f"    球{i+1}: ({x}, {y})")
    
    # 手动验证屏蔽区域功能
    print(f"\n🔍 手动验证屏蔽区域:")
    for i, mask_zone in enumerate(config['mask_zones']):
        zone_x1, zone_y1 = mask_zone['x'], mask_zone['y']
        zone_x2, zone_y2 = zone_x1 + mask_zone['width'], zone_y1 + mask_zone['height']
        print(f"  - 屏蔽区域{i+1}: ({zone_x1}, {zone_y1}) 到 ({zone_x2}, {zone_y2}) - {mask_zone.get('name', 'unnamed')}")
    
    manually_masked_balls = []
    manually_unmasked_balls = []
    
    for x, y, name in test_balls:
        is_in_any_mask = False
        mask_zone_name = ""
        
        # 检查球是否在任何屏蔽区域内
        for i, mask_zone in enumerate(config['mask_zones']):
            zone_x1, zone_y1 = mask_zone['x'], mask_zone['y']
            zone_x2, zone_y2 = zone_x1 + mask_zone['width'], zone_y1 + mask_zone['height']
            
            if zone_x1 <= x <= zone_x2 and zone_y1 <= y <= zone_y2:
                is_in_any_mask = True
                mask_zone_name = mask_zone.get('name', f'屏蔽区域{i+1}')
                break
        
        if is_in_any_mask:
            manually_masked_balls.append((x, y, name))
            print(f"  - 🚫 {name}: ({x}, {y}) - 在{mask_zone_name}内")
        else:
            manually_unmasked_balls.append((x, y, name))
            print(f"  - ✅ {name}: ({x}, {y}) - 在所有屏蔽区域外")
    
    print(f"\n✅ 预期结果分析:")
    print(f"  - 应该被屏蔽的球: {len(manually_masked_balls)}")
    for x, y, name in manually_masked_balls:
        print(f"    {name}: ({x}, {y})")
    
    print(f"  - 应该保留的球: {len(manually_unmasked_balls)}")
    for x, y, name in manually_unmasked_balls:
        print(f"    {name}: ({x}, {y})")
    
    # 验证功能是否正常工作
    expected_detected = len(manually_unmasked_balls)
    actual_detected = len(filtered_balls)
    
    print(f"\n🎯 功能验证:")
    if actual_detected == expected_detected:
        print(f"✅ 屏蔽区域功能工作正常！")
        print(f"   预期通过过滤 {expected_detected} 个球，实际通过过滤 {actual_detected} 个球")
        
        # 进一步验证具体位置是否正确
        filtered_coords = set(filtered_balls)
        expected_coords = set((x, y) for x, y, name in manually_unmasked_balls)
        
        if filtered_coords == expected_coords:
            print(f"✅ 过滤结果坐标完全正确！")
        else:
            print(f"⚠️  过滤结果数量正确，但坐标可能有差异")
            print(f"    实际过滤结果: {filtered_coords}")
            print(f"    预期过滤结果: {expected_coords}")
            
    else:
        print(f"❌ 屏蔽区域功能存在问题")
        print(f"   预期通过过滤 {expected_detected} 个球，实际通过过滤 {actual_detected} 个球")
    
    # 创建可视化
    create_visualization(test_balls, filtered_balls, config)
    
    return len(manually_masked_balls) > 0 and actual_detected == expected_detected

def create_visualization(test_balls, filtered_balls, config):
    """创建可视化图像"""
    # 创建测试帧
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (34, 139, 34)  # 绿色背景
    
    # 绘制所有测试球（灰色表示原始位置）
    for x, y, name in test_balls:
        cv2.circle(frame, (x, y), 15, (128, 128, 128), 2)  # 灰色圆圈
        cv2.putText(frame, name, (x-60, y-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # 绘制通过过滤的球（绿色）
    for x, y in filtered_balls:
        cv2.circle(frame, (x, y), 12, (0, 255, 0), -1)  # 绿色实心圆
        cv2.circle(frame, (x, y), 15, (0, 255, 0), 2)   # 绿色边框
        cv2.putText(frame, "PASS", (x-15, y+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # 绘制屏蔽区域
    if config['use_mask_zones']:
        for i, zone in enumerate(config['mask_zones']):
            x1 = zone['x']
            y1 = zone['y']
            x2 = x1 + zone['width']
            y2 = y1 + zone['height']
            
            zone_name = zone.get('name', f'MASK_{i+1}')
            
            # 绘制屏蔽区域
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 红色边框
            
            # 半透明填充
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # 标签
            cv2.putText(frame, f"MASKED: {zone_name}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 十字标记
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(frame, (x1, y2), (x2, y1), (0, 0, 255), 2)
    
    # 添加信息
    info_y = 50
    cv2.putText(frame, f"Mask Zone Direct Test Results", (10, info_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Original: {len(test_balls)}, Filtered: {len(filtered_balls)}, Masked: {len(test_balls) - len(filtered_balls)}", 
               (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 保存
    output_path = "mask_zone_direct_test_result.jpg"
    cv2.imwrite(output_path, frame)
    print(f"\n💾 可视化结果已保存: {output_path}")

def main():
    """主函数"""
    print("🎾 屏蔽区域功能直接测试")
    print("=" * 50)
    
    try:
        success = test_mask_zone_filtering()
        
        if success:
            print(f"\n🎉 测试成功！屏蔽区域功能正常工作")
        else:
            print(f"\n⚠️  测试完成，但可能存在问题")
            
        print(f"📝 查看 mask_zone_direct_test_result.jpg 文件可以看到可视化结果")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 