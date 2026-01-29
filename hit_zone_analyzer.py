#!/usr/bin/env python3
"""
hit_zone_analyzer.py
击球点分析模块 - 分析击球位置和质量
"""

import numpy as np
from typing import Tuple, Dict, Optional


class HitZoneAnalyzer:
    """击球点分析器 - 判断击球位置质量"""
    
    def __init__(self, sweet_spot_ratio: float = 0.3):
        """
        初始化击球点分析器
        
        Args:
            sweet_spot_ratio: 甜区比例（相对于球拍中心）
        """
        self.sweet_spot_ratio = sweet_spot_ratio
        
        # 统计数据
        self.total_hits = 0
        self.sweet_spot_hits = 0
        self.center_hits = 0
        self.edge_hits = 0
        self.miss_hits = 0
    
    def analyze_hit_zone(
        self,
        ball_pos: Tuple[float, float],
        racket_bbox: Tuple[float, float, float, float]
    ) -> Dict[str, any]:
        """
        分析击球点位置
        
        Args:
            ball_pos: 球位置 (x, y)
            racket_bbox: 球拍边界框 (x1, y1, x2, y2)
            
        Returns:
            {
                'zone': 'sweet_spot' | 'center' | 'edge' | 'miss',
                'quality': 0.0-1.0,
                'offset_x': float,  # 相对中心的归一化偏移 (-1 到 1)
                'offset_y': float,
                'distance': float,  # 距离中心的归一化距离
                'description': str
            }
        """
        # 计算球拍中心
        racket_center_x = (racket_bbox[0] + racket_bbox[2]) / 2
        racket_center_y = (racket_bbox[1] + racket_bbox[3]) / 2
        
        # 计算球拍尺寸
        racket_width = racket_bbox[2] - racket_bbox[0]
        racket_height = racket_bbox[3] - racket_bbox[1]
        
        # 避免除零
        if racket_width == 0 or racket_height == 0:
            return self._create_miss_result()
        
        # 计算归一化偏移 (-1 到 1)
        offset_x = (ball_pos[0] - racket_center_x) / (racket_width / 2)
        offset_y = (ball_pos[1] - racket_center_y) / (racket_height / 2)
        
        # 计算距离中心的归一化距离
        distance = np.sqrt(offset_x**2 + offset_y**2)
        
        # 判断区域和质量
        if distance < self.sweet_spot_ratio:
            zone = 'sweet_spot'
            quality = 1.0
            description = "甜区击球 - 完美！"
            self.sweet_spot_hits += 1
        elif distance < 0.5:
            zone = 'center'
            quality = 0.8 - (distance - self.sweet_spot_ratio) * 0.6
            description = "中心区击球 - 良好"
            self.center_hits += 1
        elif distance < 0.8:
            zone = 'edge'
            quality = 0.5 - (distance - 0.5) * 0.8
            description = "边缘击球 - 一般"
            self.edge_hits += 1
        else:
            zone = 'miss'
            quality = max(0.0, 0.2 - (distance - 0.8) * 0.5)
            description = "偏离击球 - 需改进"
            self.miss_hits += 1
        
        self.total_hits += 1
        
        return {
            'zone': zone,
            'quality': quality,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'distance': distance,
            'description': description,
            'racket_center': (racket_center_x, racket_center_y),
            'racket_size': (racket_width, racket_height)
        }
    
    def _create_miss_result(self) -> Dict:
        """创建未击中结果"""
        self.miss_hits += 1
        self.total_hits += 1
        return {
            'zone': 'miss',
            'quality': 0.0,
            'offset_x': 0.0,
            'offset_y': 0.0,
            'distance': float('inf'),
            'description': "未检测到击球",
            'racket_center': (0, 0),
            'racket_size': (0, 0)
        }
    
    def is_hit(
        self,
        ball_pos: Tuple[float, float],
        racket_bbox: Tuple[float, float, float, float],
        threshold: float = 1.0
    ) -> bool:
        """
        判断是否击中球
        
        Args:
            ball_pos: 球位置
            racket_bbox: 球拍边界框
            threshold: 距离阈值（归一化距离）
            
        Returns:
            是否击中
        """
        result = self.analyze_hit_zone(ball_pos, racket_bbox)
        # 撤销统计（因为这只是检查）
        self.total_hits -= 1
        if result['zone'] == 'sweet_spot':
            self.sweet_spot_hits -= 1
        elif result['zone'] == 'center':
            self.center_hits -= 1
        elif result['zone'] == 'edge':
            self.edge_hits -= 1
        elif result['zone'] == 'miss':
            self.miss_hits -= 1
        
        return result['distance'] < threshold
    
    def get_stats(self) -> Dict[str, any]:
        """获取统计数据"""
        if self.total_hits == 0:
            return {
                'total_hits': 0,
                'sweet_spot_rate': 0.0,
                'center_rate': 0.0,
                'edge_rate': 0.0,
                'miss_rate': 0.0
            }
        
        return {
            'total_hits': self.total_hits,
            'sweet_spot_hits': self.sweet_spot_hits,
            'center_hits': self.center_hits,
            'edge_hits': self.edge_hits,
            'miss_hits': self.miss_hits,
            'sweet_spot_rate': self.sweet_spot_hits / self.total_hits,
            'center_rate': self.center_hits / self.total_hits,
            'edge_rate': self.edge_hits / self.total_hits,
            'miss_rate': self.miss_hits / self.total_hits
        }
    
    def reset_stats(self):
        """重置统计"""
        self.total_hits = 0
        self.sweet_spot_hits = 0
        self.center_hits = 0
        self.edge_hits = 0
        self.miss_hits = 0


# 测试代码
if __name__ == "__main__":
    print("🧪 测试击球点分析器\n")
    
    # 创建分析器
    analyzer = HitZoneAnalyzer(sweet_spot_ratio=0.3)
    
    # 模拟球拍
    racket_bbox = (100, 100, 200, 300)  # x1, y1, x2, y2
    racket_center = (150, 200)
    
    print(f"📊 球拍信息:")
    print(f"   边界框: {racket_bbox}")
    print(f"   中心: {racket_center}")
    print(f"   尺寸: 100x200\n")
    
    # 测试不同位置的击球
    test_positions = [
        ((150, 200), "正中心"),
        ((160, 210), "轻微偏离"),
        ((175, 225), "中心区"),
        ((190, 270), "边缘"),
        ((210, 320), "偏离")
    ]
    
    print("📊 击球点分析:")
    for ball_pos, desc in test_positions:
        result = analyzer.analyze_hit_zone(ball_pos, racket_bbox)
        print(f"\n   {desc}: {ball_pos}")
        print(f"   区域: {result['zone']}")
        print(f"   质量: {result['quality']:.2f}")
        print(f"   偏移: X={result['offset_x']:.2f}, Y={result['offset_y']:.2f}")
        print(f"   距离: {result['distance']:.2f}")
        print(f"   评价: {result['description']}")
    
    # 统计
    print(f"\n📈 统计数据:")
    stats = analyzer.get_stats()
    print(f"   总击球数: {stats['total_hits']}")
    print(f"   甜区率: {stats['sweet_spot_rate']*100:.1f}%")
    print(f"   中心率: {stats['center_rate']*100:.1f}%")
    print(f"   边缘率: {stats['edge_rate']*100:.1f}%")
    print(f"   偏离率: {stats['miss_rate']*100:.1f}%")
    
    print("\n✅ 测试完成")
