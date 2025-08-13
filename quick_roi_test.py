# quick_roi_test.py
"""
快速ROI可视化测试 - 直接显示ROI效果
"""

import cv2
import yaml
import numpy as np
from roi_manager import ROIManager

def quick_roi_visual_test():
    print("🎯 快速ROI可视化测试")
    print("=" * 40)
    
    # 加载配置
    with open("configs/roi_enabled_config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化ROI管理器
    roi_manager = ROIManager(config)
    
    # 加载ROI配置
    if roi_manager.load_roi_config("configs/roi_config.yaml"):
        print("✅ ROI配置加载成功")
    else:
        print("❌ ROI配置加载失败")
        return
    
    # 读取视频第一帧
    video_path = config.get('video_input_path', 'data/input_video.mp4')
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ 无法读取视频帧")
        return
    
    print(f"✅ 视频帧读取成功: {frame.shape}")
    
    # 创建测试帧
    test_frame = frame.copy()
    
    # 绘制ROI
    test_frame = roi_manager.draw_roi(test_frame, show_fill=True)
    
    # 添加测试点来演示ROI内外效果
    test_points = [
        (100, 100, "外部1"),
        (1000, 500, "内部1"), 
        (1500, 700, "内部2"),
        (2400, 100, "外部2"),
        (500, 1300, "内部3")
    ]
    
    for x, y, label in test_points:
        is_inside = roi_manager.is_point_in_roi((x, y))
        color = (0, 255, 0) if is_inside else (0, 0, 255)  # 绿色内部，红色外部
        
        # 绘制测试点
        cv2.circle(test_frame, (x, y), 15, color, -1)
        cv2.circle(test_frame, (x, y), 20, (255, 255, 255), 2)
        
        # 添加标签
        cv2.putText(test_frame, f"{label}", (x+25, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(test_frame, "IN" if is_inside else "OUT", (x+25, y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 添加说明文字
    cv2.putText(test_frame, "ROI Test - Green: Inside, Red: Outside", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(test_frame, "Yellow Border: ROI Boundary", 
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(test_frame, "Press any key to close", 
               (10, test_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 显示ROI统计信息
    stats = roi_manager.get_roi_stats()
    info_text = f"ROI Area: {stats.get('roi_area', 0):.0f} pixels²"
    cv2.putText(test_frame, info_text, (10, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 显示测试结果
    cv2.namedWindow("ROI Visual Test", cv2.WINDOW_NORMAL)
    cv2.imshow("ROI Visual Test", test_frame)
    
    print("🎨 ROI可视化测试显示中...")
    print("   - 黄色边界: ROI区域")
    print("   - 绿色点: ROI内部")
    print("   - 红色点: ROI外部")
    print("   - 按任意键关闭窗口")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存测试图片
    output_path = "roi_visual_test_result.jpg"
    cv2.imwrite(output_path, test_frame)
    print(f"💾 测试结果已保存: {output_path}")
    
    # 统计测试结果
    inside_count = sum(1 for x, y, _ in test_points if roi_manager.is_point_in_roi((x, y)))
    outside_count = len(test_points) - inside_count
    
    print(f"\n📊 测试统计:")
    print(f"   总测试点: {len(test_points)}")
    print(f"   ROI内部: {inside_count}")
    print(f"   ROI外部: {outside_count}")
    print(f"   ROI面积: {stats.get('roi_area', 0):.0f} 像素²")
    
    print("\n✅ ROI可视化测试完成!")

if __name__ == "__main__":
    quick_roi_visual_test()
