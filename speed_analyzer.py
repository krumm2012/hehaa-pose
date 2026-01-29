#!/usr/bin/env python3
"""
speed_analyzer.py
速度分析模块 - 计算球速和挥拍速度
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque


class SpeedAnalyzer:
    """速度分析器 - 计算球速和挥拍速度"""
    
    def __init__(self, fps: float = 25.0, pixel_to_meter: float = 0.01):
        """
        初始化速度分析器
        
        Args:
            fps: 视频帧率
            pixel_to_meter: 像素到米的转换比例（需要根据实际场景校准）
        """
        self.fps = fps
        self.pixel_to_meter = pixel_to_meter
        self.time_delta = 1.0 / fps
        
        # 速度历史记录（用于平滑）
        self.ball_speed_history = deque(maxlen=5)
        self.racket_speed_history = deque(maxlen=5)
        
        # 统计数据
        self.max_ball_speed = 0.0
        self.max_racket_speed = 0.0
    
    def calculate_ball_speed(
        self,
        prev_pos: Optional[Tuple[float, float]],
        curr_pos: Tuple[float, float],
        smooth: bool = True
    ) -> float:
        """
        计算球速
        
        Args:
            prev_pos: 上一帧位置 (x, y)，None 表示第一帧
            curr_pos: 当前帧位置 (x, y)
            smooth: 是否使用平滑
            
        Returns:
            速度 (km/h)
        """
        if prev_pos is None:
            return 0.0
        
        # 计算像素距离
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        pixel_distance = np.sqrt(dx**2 + dy**2)
        
        # 转换为实际距离（米）
        meter_distance = pixel_distance * self.pixel_to_meter
        
        # 计算速度（米/秒）
        speed_ms = meter_distance / self.time_delta
        
        # 转换为 km/h
        speed_kmh = speed_ms * 3.6
        
        # 平滑处理
        if smooth:
            self.ball_speed_history.append(speed_kmh)
            speed_kmh = np.mean(self.ball_speed_history)
        
        # 更新最大速度
        if speed_kmh > self.max_ball_speed:
            self.max_ball_speed = speed_kmh
        
        return speed_kmh
    
    def calculate_swing_speed(
        self,
        racket_trajectory: List[Tuple[float, float]],
        window_size: int = 5
    ) -> Dict[str, float]:
        """
        计算挥拍速度
        
        Args:
            racket_trajectory: 球拍轨迹列表 [(x, y), ...]
            window_size: 计算窗口大小
            
        Returns:
            {
                'current': 当前速度,
                'average': 平均速度,
                'max': 最大速度
            } (km/h)
        """
        if len(racket_trajectory) < 2:
            return {'current': 0.0, 'average': 0.0, 'max': 0.0}
        
        # 计算最近几帧的速度
        speeds = []
        for i in range(max(1, len(racket_trajectory) - window_size), len(racket_trajectory)):
            prev = racket_trajectory[i-1]
            curr = racket_trajectory[i]
            
            # 计算像素距离
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            pixel_distance = np.sqrt(dx**2 + dy**2)
            
            # 转换为速度
            meter_distance = pixel_distance * self.pixel_to_meter
            speed_ms = meter_distance / self.time_delta
            speed_kmh = speed_ms * 3.6
            
            speeds.append(speed_kmh)
        
        if not speeds:
            return {'current': 0.0, 'average': 0.0, 'max': 0.0}
        
        current_speed = speeds[-1]
        avg_speed = np.mean(speeds)
        max_speed = max(speeds)
        
        # 更新全局最大速度
        if max_speed > self.max_racket_speed:
            self.max_racket_speed = max_speed
        
        return {
            'current': current_speed,
            'average': avg_speed,
            'max': max_speed
        }
    
    def calculate_swing_phase_speeds(
        self,
        racket_trajectory: List[Tuple[float, float]],
        swing_phases: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> Dict[str, float]:
        """
        计算挥拍各阶段的速度
        
        Args:
            racket_trajectory: 球拍轨迹
            swing_phases: 挥拍阶段 {'preparation': (start, end), 'swing': (start, end), ...}
            
        Returns:
            各阶段速度 (km/h)
        """
        if swing_phases is None or len(racket_trajectory) < 2:
            return {}
        
        phase_speeds = {}
        
        for phase_name, (start_idx, end_idx) in swing_phases.items():
            if start_idx >= len(racket_trajectory) or end_idx > len(racket_trajectory):
                phase_speeds[phase_name] = 0.0
                continue
            
            # 计算该阶段的平均速度
            phase_trajectory = racket_trajectory[start_idx:end_idx]
            if len(phase_trajectory) < 2:
                phase_speeds[phase_name] = 0.0
                continue
            
            speeds = []
            for i in range(1, len(phase_trajectory)):
                prev = phase_trajectory[i-1]
                curr = phase_trajectory[i]
                
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                pixel_distance = np.sqrt(dx**2 + dy**2)
                
                meter_distance = pixel_distance * self.pixel_to_meter
                speed_ms = meter_distance / self.time_delta
                speed_kmh = speed_ms * 3.6
                
                speeds.append(speed_kmh)
            
            phase_speeds[phase_name] = np.mean(speeds) if speeds else 0.0
        
        return phase_speeds
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计数据"""
        return {
            'max_ball_speed': self.max_ball_speed,
            'max_racket_speed': self.max_racket_speed
        }
    
    def reset_stats(self):
        """重置统计"""
        self.max_ball_speed = 0.0
        self.max_racket_speed = 0.0
        self.ball_speed_history.clear()
        self.racket_speed_history.clear()


# 测试代码
if __name__ == "__main__":
    print("🧪 测试速度分析器\n")
    
    # 创建分析器
    analyzer = SpeedAnalyzer(fps=25.0, pixel_to_meter=0.01)
    
    # 模拟球轨迹（从左到右移动）
    print("📊 球速测试:")
    ball_positions = [
        (100, 200),
        (150, 210),
        (200, 220),
        (250, 230),
        (300, 240)
    ]
    
    prev_pos = None
    for i, pos in enumerate(ball_positions):
        speed = analyzer.calculate_ball_speed(prev_pos, pos)
        print(f"   帧{i}: 位置={pos}, 速度={speed:.2f} km/h")
        prev_pos = pos
    
    # 模拟挥拍轨迹（加速过程）
    print("\n📊 挥拍速度测试:")
    racket_trajectory = [
        (200, 300),
        (210, 310),
        (225, 325),
        (245, 345),
        (270, 370),
        (300, 400)
    ]
    
    for i in range(2, len(racket_trajectory) + 1):
        speeds = analyzer.calculate_swing_speed(racket_trajectory[:i])
        print(f"   帧{i-1}: 当前={speeds['current']:.2f}, "
              f"平均={speeds['average']:.2f}, "
              f"最大={speeds['max']:.2f} km/h")
    
    # 统计
    print(f"\n📈 统计数据:")
    stats = analyzer.get_stats()
    print(f"   最大球速: {stats['max_ball_speed']:.2f} km/h")
    print(f"   最大挥拍速度: {stats['max_racket_speed']:.2f} km/h")
    
    print("\n✅ 测试完成")
