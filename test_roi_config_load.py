# test_roi_config_load.py
"""
测试ROI配置文件加载功能
"""

import yaml
from roi_manager import ROIManager

def test_roi_config_loading():
    print("🧪 测试ROI配置文件加载")
    print("=" * 40)
    
    # 创建基础配置
    config = {
        'roi_settings': {
            'enabled': True,
            'visualization': {
                'show_roi_boundary': True,
                'show_roi_fill': True,
                'show_roi_points': True,
                'highlight_detections': True,
                'show_roi_stats': True
            }
        }
    }
    
    # 初始化ROI管理器
    roi_manager = ROIManager(config)
    
    # 测试加载roi_config.yaml
    print("📁 测试加载 configs/roi_config.yaml...")
    success = roi_manager.load_roi_config("configs/roi_config.yaml")
    
    if success:
        print("✅ ROI配置加载成功！")
        print(f"🎯 ROI设置状态: {roi_manager.is_roi_set}")
        print(f"🔢 ROI点数: {len(roi_manager.roi_points)}")
        print(f"📍 ROI坐标: {roi_manager.roi_points}")
        
        # 获取统计信息
        stats = roi_manager.get_roi_stats()
        print(f"📊 ROI面积: {stats.get('roi_area', 0):.0f} 像素²")
        print(f"📏 ROI周长: {stats.get('roi_perimeter', 0):.0f} 像素")
        
        # 测试点在ROI内判断
        test_points = [
            (1000, 500),    # 可能在ROI内
            (100, 100),     # 可能在ROI外
            (1500, 700),    # 中心附近
        ]
        
        print(f"\n🎯 测试点在ROI内判断:")
        for point in test_points:
            is_inside = roi_manager.is_point_in_roi(point)
            print(f"  点 {point}: {'✅ 在ROI内' if is_inside else '❌ 在ROI外'}")
            
    else:
        print("❌ ROI配置加载失败")
        
        # 检查配置文件内容
        try:
            with open("configs/roi_config.yaml", 'r', encoding='utf-8') as f:
                config_content = yaml.safe_load(f)
            print(f"📄 配置文件内容: {config_content}")
        except Exception as e:
            print(f"❌ 读取配置文件失败: {e}")
    
    print("\n" + "=" * 40)
    print("🧪 测试完成")

if __name__ == "__main__":
    test_roi_config_loading()
