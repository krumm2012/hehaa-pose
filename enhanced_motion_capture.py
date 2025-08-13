# enhanced_motion_capture.py
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from collections import deque
import time

class EnhancedMotionCapture:
    """
    增强的动作捕捉系统 - 专门针对ROI内的精确动作分析
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # ROI动作捕捉配置
        self.roi_motion_config = config.get('roi_motion_capture', {})
        
        # 动作历史记录
        self.motion_history = {
            'pose': deque(maxlen=self.roi_motion_config.get('pose_history_length', 30)),
            'ball': deque(maxlen=self.roi_motion_config.get('ball_history_length', 50)),
            'racket': deque(maxlen=self.roi_motion_config.get('racket_history_length', 30))
        }
        
        # 动作分析参数
        self.motion_thresholds = {
            'significant_movement': self.roi_motion_config.get('significant_movement_threshold', 15),
            'rapid_movement': self.roi_motion_config.get('rapid_movement_threshold', 30),
            'pose_change_threshold': self.roi_motion_config.get('pose_change_threshold', 20),
            'swing_velocity_threshold': self.roi_motion_config.get('swing_velocity_threshold', 25)
        }
        
        # 颜色定义
        self.colors = {
            'roi_pose': (255, 100, 255),      # 紫色 - ROI内姿态
            'roi_ball': (0, 255, 0),          # 绿色 - ROI内球
            'roi_racket': (255, 165, 0),      # 橙色 - ROI内球拍
            'motion_trail': (100, 200, 255),  # 浅蓝色 - 动作轨迹
            'velocity_vector': (0, 255, 255), # 青色 - 速度向量
            'high_activity': (0, 0, 255),     # 红色 - 高活动区域
            'medium_activity': (0, 255, 255), # 黄色 - 中等活动
            'low_activity': (0, 255, 0)       # 绿色 - 低活动
        }
        
        # 当前帧信息
        self.current_frame_num = 0
        self.last_analysis_time = time.time()
        
        # 动作统计
        self.motion_stats = {
            'total_pose_detections': 0,
            'total_ball_detections': 0,
            'total_racket_detections': 0,
            'significant_movements': 0,
            'swing_events': 0,
            'roi_activity_level': 'low'
        }
        
        # 日志配置
        self.logger = logging.getLogger('EnhancedMotionCapture')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('🎬 [动作捕捉] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def analyze_roi_motion(self, pose_data: List[Dict], ball_positions: List[Tuple], 
                          racket_data: List[Dict], frame_num: int) -> Dict[str, Any]:
        """
        分析ROI内的动作模式
        
        Args:
            pose_data: ROI内的姿态数据
            ball_positions: ROI内的球位置
            racket_data: ROI内的球拍数据
            frame_num: 当前帧号
            
        Returns:
            动作分析结果字典
        """
        self.current_frame_num = frame_num
        current_time = time.time()
        
        # 更新动作历史
        self._update_motion_history(pose_data, ball_positions, racket_data, frame_num)
        
        # 分析各类动作
        pose_analysis = self._analyze_pose_motion(pose_data)
        ball_analysis = self._analyze_ball_motion(ball_positions)
        racket_analysis = self._analyze_racket_motion(racket_data)
        
        # 综合动作分析
        comprehensive_analysis = self._comprehensive_motion_analysis()
        
        # 更新统计信息
        self._update_motion_stats(pose_analysis, ball_analysis, racket_analysis)
        
        # 检测动作事件
        motion_events = self._detect_motion_events(pose_analysis, ball_analysis, racket_analysis)
        
        analysis_result = {
            'frame_num': frame_num,
            'timestamp': current_time,
            'pose_analysis': pose_analysis,
            'ball_analysis': ball_analysis,
            'racket_analysis': racket_analysis,
            'comprehensive_analysis': comprehensive_analysis,
            'motion_events': motion_events,
            'activity_level': self.motion_stats['roi_activity_level'],
            'stats': self.motion_stats.copy()
        }
        
        self.last_analysis_time = current_time
        return analysis_result
    
    def _update_motion_history(self, pose_data: List[Dict], ball_positions: List[Tuple], 
                              racket_data: List[Dict], frame_num: int):
        """更新动作历史记录"""
        timestamp = time.time()
        
        # 更新姿态历史
        if pose_data:
            pose_entry = {
                'frame': frame_num,
                'timestamp': timestamp,
                'keypoints': pose_data[0] if pose_data else {},
                'person_count': len(pose_data)
            }
            self.motion_history['pose'].append(pose_entry)
        
        # 更新球历史
        if ball_positions:
            ball_entry = {
                'frame': frame_num,
                'timestamp': timestamp,
                'positions': ball_positions,
                'ball_count': len(ball_positions)
            }
            self.motion_history['ball'].append(ball_entry)
        
        # 更新球拍历史
        if racket_data:
            racket_entry = {
                'frame': frame_num,
                'timestamp': timestamp,
                'rackets': racket_data,
                'racket_count': len(racket_data)
            }
            self.motion_history['racket'].append(racket_entry)
    
    def _analyze_pose_motion(self, current_pose_data: List[Dict]) -> Dict[str, Any]:
        """分析姿态动作"""
        if not self.motion_history['pose'] or not current_pose_data:
            return {'status': 'no_data', 'movement_magnitude': 0, 'movement_direction': None}
        
        # 获取最近的姿态数据
        recent_poses = list(self.motion_history['pose'])[-10:]  # 最近10帧
        
        if len(recent_poses) < 2:
            return {'status': 'insufficient_data', 'movement_magnitude': 0}
        
        # 计算关键点运动
        current_keypoints = current_pose_data[0] if current_pose_data else {}
        prev_keypoints = recent_poses[-2]['keypoints'] if len(recent_poses) >= 2 else {}
        
        movement_vectors = []
        significant_movements = []
        
        # 关键关节点列表
        key_joints = ['left_wrist', 'right_wrist', 'left_elbow', 'right_elbow', 
                     'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        
        for joint in key_joints:
            current_pos = current_keypoints.get(joint)
            prev_pos = prev_keypoints.get(joint)
            
            if current_pos and prev_pos:
                # 计算运动向量
                dx = current_pos[0] - prev_pos[0]
                dy = current_pos[1] - prev_pos[1]
                magnitude = np.sqrt(dx**2 + dy**2)
                
                movement_vectors.append({
                    'joint': joint,
                    'vector': (dx, dy),
                    'magnitude': magnitude,
                    'direction': np.arctan2(dy, dx) * 180 / np.pi
                })
                
                # 检测显著运动
                if magnitude > self.motion_thresholds['significant_movement']:
                    significant_movements.append({
                        'joint': joint,
                        'magnitude': magnitude,
                        'is_rapid': magnitude > self.motion_thresholds['rapid_movement']
                    })
        
        # 计算整体运动水平
        total_movement = sum(mv['magnitude'] for mv in movement_vectors)
        avg_movement = total_movement / len(movement_vectors) if movement_vectors else 0
        
        # 检测挥拍动作
        swing_detected = self._detect_swing_motion(movement_vectors)
        
        return {
            'status': 'analyzed',
            'movement_vectors': movement_vectors,
            'significant_movements': significant_movements,
            'total_movement': total_movement,
            'average_movement': avg_movement,
            'swing_detected': swing_detected,
            'activity_level': self._classify_activity_level(avg_movement)
        }
    
    def _analyze_ball_motion(self, current_ball_positions: List[Tuple]) -> Dict[str, Any]:
        """分析球运动"""
        if not self.motion_history['ball'] or not current_ball_positions:
            return {'status': 'no_data', 'velocity': 0, 'trajectory': []}
        
        recent_balls = list(self.motion_history['ball'])[-20:]  # 最近20帧
        
        if len(recent_balls) < 2:
            return {'status': 'insufficient_data', 'velocity': 0}
        
        # 计算球的速度和轨迹
        trajectories = []
        velocities = []
        
        for i in range(1, len(recent_balls)):
            prev_frame = recent_balls[i-1]
            curr_frame = recent_balls[i]
            
            # 计算时间差
            dt = curr_frame['timestamp'] - prev_frame['timestamp']
            if dt == 0:
                continue
            
            # 对每个球计算运动
            for curr_pos in curr_frame['positions']:
                # 找到最近的前一帧球位置
                min_dist = float('inf')
                closest_prev_pos = None
                
                for prev_pos in prev_frame['positions']:
                    dist = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_prev_pos = prev_pos
                
                if closest_prev_pos and min_dist < 100:  # 合理的最大移动距离
                    # 计算速度
                    dx = curr_pos[0] - closest_prev_pos[0]
                    dy = curr_pos[1] - closest_prev_pos[1]
                    velocity = np.sqrt(dx**2 + dy**2) / dt
                    
                    velocities.append(velocity)
                    trajectories.append({
                        'from': closest_prev_pos,
                        'to': curr_pos,
                        'velocity': velocity,
                        'direction': np.arctan2(dy, dx) * 180 / np.pi,
                        'frame': curr_frame['frame']
                    })
        
        # 计算平均速度和运动模式
        avg_velocity = np.mean(velocities) if velocities else 0
        max_velocity = np.max(velocities) if velocities else 0
        
        # 检测球的运动模式
        motion_pattern = self._analyze_ball_pattern(trajectories)
        
        return {
            'status': 'analyzed',
            'trajectories': trajectories,
            'velocities': velocities,
            'average_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'motion_pattern': motion_pattern,
            'ball_count': len(current_ball_positions)
        }
    
    def _analyze_racket_motion(self, current_racket_data: List[Dict]) -> Dict[str, Any]:
        """分析球拍运动"""
        if not self.motion_history['racket'] or not current_racket_data:
            return {'status': 'no_data', 'swing_speed': 0, 'swing_direction': None}
        
        recent_rackets = list(self.motion_history['racket'])[-15:]  # 最近15帧
        
        if len(recent_rackets) < 2:
            return {'status': 'insufficient_data', 'swing_speed': 0}
        
        # 分析球拍运动
        racket_movements = []
        
        for i in range(1, len(recent_rackets)):
            prev_frame = recent_rackets[i-1]
            curr_frame = recent_rackets[i]
            
            dt = curr_frame['timestamp'] - prev_frame['timestamp']
            if dt == 0:
                continue
            
            # 对每个球拍计算运动
            for curr_racket in curr_frame['rackets']:
                curr_box = curr_racket['box']
                curr_center = ((curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2)
                
                # 找到对应的前一帧球拍
                min_dist = float('inf')
                closest_prev_center = None
                
                for prev_racket in prev_frame['rackets']:
                    prev_box = prev_racket['box']
                    prev_center = ((prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2)
                    
                    dist = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_prev_center = prev_center
                
                if closest_prev_center and min_dist < 150:  # 合理的最大移动距离
                    # 计算球拍运动
                    dx = curr_center[0] - closest_prev_center[0]
                    dy = curr_center[1] - closest_prev_center[1]
                    speed = np.sqrt(dx**2 + dy**2) / dt
                    
                    racket_movements.append({
                        'from': closest_prev_center,
                        'to': curr_center,
                        'speed': speed,
                        'direction': np.arctan2(dy, dx) * 180 / np.pi,
                        'frame': curr_frame['frame'],
                        'confidence': curr_racket.get('confidence', 0)
                    })
        
        # 分析挥拍模式
        avg_speed = np.mean([rm['speed'] for rm in racket_movements]) if racket_movements else 0
        max_speed = np.max([rm['speed'] for rm in racket_movements]) if racket_movements else 0
        
        # 检测挥拍事件
        swing_events = [rm for rm in racket_movements if rm['speed'] > self.motion_thresholds['swing_velocity_threshold']]
        
        return {
            'status': 'analyzed',
            'movements': racket_movements,
            'average_speed': avg_speed,
            'max_speed': max_speed,
            'swing_events': swing_events,
            'racket_count': len(current_racket_data)
        }
    
    def _comprehensive_motion_analysis(self) -> Dict[str, Any]:
        """综合动作分析"""
        # 分析整体活动水平
        pose_activity = len(self.motion_history['pose']) > 0
        ball_activity = len(self.motion_history['ball']) > 0
        racket_activity = len(self.motion_history['racket']) > 0
        
        overall_activity = 'high' if (pose_activity and ball_activity and racket_activity) else \
                          'medium' if (pose_activity and (ball_activity or racket_activity)) else \
                          'low'
        
        # 计算ROI利用率
        recent_frames = max(
            len(self.motion_history['pose']),
            len(self.motion_history['ball']),
            len(self.motion_history['racket'])
        )
        
        roi_utilization = min(100.0, (recent_frames / 30) * 100)  # 基于30帧的利用率
        
        # 检测协调动作（如球拍击球）
        coordination_events = self._detect_coordination_events()
        
        return {
            'overall_activity': overall_activity,
            'roi_utilization': roi_utilization,
            'coordination_events': coordination_events,
            'motion_complexity': self._calculate_motion_complexity()
        }
    
    def _detect_swing_motion(self, movement_vectors: List[Dict]) -> Dict[str, Any]:
        """检测挥拍动作"""
        if not movement_vectors:
            return {'detected': False}
        
        # 查找手腕和肘部的显著运动
        wrist_movements = [mv for mv in movement_vectors if 'wrist' in mv['joint']]
        elbow_movements = [mv for mv in movement_vectors if 'elbow' in mv['joint']]
        
        # 检测挥拍模式：手腕快速运动 + 肘部配合运动
        swing_detected = False
        swing_arm = None
        swing_magnitude = 0
        
        for wrist_mv in wrist_movements:
            if wrist_mv['magnitude'] > self.motion_thresholds['swing_velocity_threshold']:
                # 检查对应的肘部运动
                arm_side = 'left' if 'left' in wrist_mv['joint'] else 'right'
                corresponding_elbow = next((em for em in elbow_movements if arm_side in em['joint']), None)
                
                if corresponding_elbow and corresponding_elbow['magnitude'] > self.motion_thresholds['significant_movement']:
                    swing_detected = True
                    swing_arm = arm_side
                    swing_magnitude = wrist_mv['magnitude']
                    break
        
        return {
            'detected': swing_detected,
            'arm': swing_arm,
            'magnitude': swing_magnitude,
            'wrist_movements': wrist_movements,
            'elbow_movements': elbow_movements
        }
    
    def _analyze_ball_pattern(self, trajectories: List[Dict]) -> Dict[str, Any]:
        """分析球的运动模式"""
        if not trajectories:
            return {'pattern': 'stationary'}
        
        # 分析速度变化
        velocities = [t['velocity'] for t in trajectories]
        velocity_changes = []
        
        for i in range(1, len(velocities)):
            velocity_changes.append(abs(velocities[i] - velocities[i-1]))
        
        # 分析方向变化
        directions = [t['direction'] for t in trajectories]
        direction_changes = []
        
        for i in range(1, len(directions)):
            angle_diff = abs(directions[i] - directions[i-1])
            # 处理角度跨越180度的情况
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            direction_changes.append(angle_diff)
        
        # 判断运动模式
        avg_velocity_change = np.mean(velocity_changes) if velocity_changes else 0
        avg_direction_change = np.mean(direction_changes) if direction_changes else 0
        
        if avg_velocity_change > 50 and avg_direction_change > 30:
            pattern = 'erratic'  # 不规则运动
        elif avg_direction_change > 45:
            pattern = 'bouncing'  # 弹跳
        elif avg_velocity_change < 10 and avg_direction_change < 15:
            pattern = 'linear'  # 直线运动
        else:
            pattern = 'curved'  # 曲线运动
        
        return {
            'pattern': pattern,
            'avg_velocity_change': avg_velocity_change,
            'avg_direction_change': avg_direction_change,
            'trajectory_length': len(trajectories)
        }
    
    def _detect_coordination_events(self) -> List[Dict]:
        """检测协调动作事件（如击球）"""
        events = []
        
        # 检查是否有球拍和球的接近事件
        if self.motion_history['ball'] and self.motion_history['racket']:
            recent_balls = list(self.motion_history['ball'])[-5:]
            recent_rackets = list(self.motion_history['racket'])[-5:]
            
            for ball_frame in recent_balls:
                for racket_frame in recent_rackets:
                    if abs(ball_frame['frame'] - racket_frame['frame']) <= 2:  # 2帧内
                        # 检查球和球拍的距离
                        for ball_pos in ball_frame['positions']:
                            for racket in racket_frame['rackets']:
                                racket_box = racket['box']
                                racket_center = ((racket_box[0] + racket_box[2]) / 2, (racket_box[1] + racket_box[3]) / 2)
                                
                                distance = np.sqrt((ball_pos[0] - racket_center[0])**2 + (ball_pos[1] - racket_center[1])**2)
                                
                                if distance < 80:  # 接触距离阈值
                                    events.append({
                                        'type': 'ball_racket_contact',
                                        'frame': ball_frame['frame'],
                                        'ball_position': ball_pos,
                                        'racket_center': racket_center,
                                        'distance': distance,
                                        'confidence': racket.get('confidence', 0)
                                    })
        
        return events
    
    def _calculate_motion_complexity(self) -> float:
        """计算动作复杂度"""
        complexity_score = 0.0
        
        # 基于不同类型检测的数量
        pose_variety = len(self.motion_history['pose'])
        ball_variety = len(self.motion_history['ball'])
        racket_variety = len(self.motion_history['racket'])
        
        # 标准化得分
        complexity_score = min(1.0, (pose_variety + ball_variety + racket_variety) / 90.0)
        
        return complexity_score
    
    def _classify_activity_level(self, avg_movement: float) -> str:
        """分类活动水平"""
        if avg_movement > 20:
            return 'high'
        elif avg_movement > 8:
            return 'medium'
        else:
            return 'low'
    
    def _detect_motion_events(self, pose_analysis: Dict, ball_analysis: Dict, racket_analysis: Dict) -> List[Dict]:
        """检测动作事件"""
        events = []
        
        # 检测挥拍事件
        if pose_analysis.get('swing_detected', {}).get('detected', False):
            events.append({
                'type': 'swing_detected',
                'source': 'pose',
                'details': pose_analysis['swing_detected'],
                'frame': self.current_frame_num
            })
        
        # 检测快速球运动
        if ball_analysis.get('max_velocity', 0) > 100:  # 高速球运动
            events.append({
                'type': 'high_speed_ball',
                'source': 'ball',
                'velocity': ball_analysis['max_velocity'],
                'frame': self.current_frame_num
            })
        
        # 检测球拍快速移动
        if racket_analysis.get('swing_events'):
            for swing_event in racket_analysis['swing_events']:
                events.append({
                    'type': 'racket_swing',
                    'source': 'racket',
                    'details': swing_event,
                    'frame': self.current_frame_num
                })
        
        return events
    
    def _update_motion_stats(self, pose_analysis: Dict, ball_analysis: Dict, racket_analysis: Dict):
        """更新运动统计信息"""
        # 更新检测计数
        if pose_analysis.get('status') == 'analyzed':
            self.motion_stats['total_pose_detections'] += 1
        
        if ball_analysis.get('status') == 'analyzed':
            self.motion_stats['total_ball_detections'] += 1
        
        if racket_analysis.get('status') == 'analyzed':
            self.motion_stats['total_racket_detections'] += 1
        
        # 更新显著运动计数
        if pose_analysis.get('significant_movements'):
            self.motion_stats['significant_movements'] += len(pose_analysis['significant_movements'])
        
        # 更新挥拍事件计数
        if pose_analysis.get('swing_detected', {}).get('detected', False):
            self.motion_stats['swing_events'] += 1
        
        # 更新活动水平
        activity_levels = [
            pose_analysis.get('activity_level', 'low'),
            ball_analysis.get('motion_pattern', {}).get('pattern', 'low'),
            racket_analysis.get('status', 'low')
        ]
        
        if 'high' in activity_levels:
            self.motion_stats['roi_activity_level'] = 'high'
        elif 'medium' in activity_levels:
            self.motion_stats['roi_activity_level'] = 'medium'
        else:
            self.motion_stats['roi_activity_level'] = 'low'
    
    def draw_motion_analysis(self, frame: np.ndarray, analysis_result: Dict[str, Any]) -> np.ndarray:
        """在帧上绘制动作分析结果"""
        result_frame = frame.copy()
        
        # 绘制动作轨迹
        self._draw_motion_trails(result_frame, analysis_result)
        
        # 绘制速度向量
        self._draw_velocity_vectors(result_frame, analysis_result)
        
        # 绘制活动热力图
        self._draw_activity_heatmap(result_frame, analysis_result)
        
        # 绘制动作事件
        self._draw_motion_events(result_frame, analysis_result)
        
        # 绘制统计信息
        self._draw_motion_stats(result_frame, analysis_result)
        
        return result_frame
    
    def _draw_motion_trails(self, frame: np.ndarray, analysis_result: Dict[str, Any]):
        """绘制动作轨迹"""
        # 绘制球轨迹
        ball_analysis = analysis_result.get('ball_analysis', {})
        trajectories = ball_analysis.get('trajectories', [])
        
        for traj in trajectories[-10:]:  # 最近10个轨迹点
            cv2.line(frame, 
                    (int(traj['from'][0]), int(traj['from'][1])),
                    (int(traj['to'][0]), int(traj['to'][1])),
                    self.colors['motion_trail'], 2)
    
    def _draw_velocity_vectors(self, frame: np.ndarray, analysis_result: Dict[str, Any]):
        """绘制速度向量"""
        # 绘制姿态运动向量
        pose_analysis = analysis_result.get('pose_analysis', {})
        movement_vectors = pose_analysis.get('movement_vectors', [])
        
        for mv in movement_vectors:
            if mv['magnitude'] > self.motion_thresholds['significant_movement']:
                # 根据关节位置绘制速度向量（需要当前关节位置）
                pass  # 实际实现需要当前关节坐标
    
    def _draw_activity_heatmap(self, frame: np.ndarray, analysis_result: Dict[str, Any]):
        """绘制活动热力图"""
        activity_level = analysis_result.get('activity_level', 'low')
        
        # 在右上角显示活动水平
        color = self.colors['high_activity'] if activity_level == 'high' else \
                self.colors['medium_activity'] if activity_level == 'medium' else \
                self.colors['low_activity']
        
        cv2.putText(frame, f"ROI Activity: {activity_level.upper()}", 
                   (frame.shape[1] - 300, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _draw_motion_events(self, frame: np.ndarray, analysis_result: Dict[str, Any]):
        """绘制动作事件"""
        events = analysis_result.get('motion_events', [])
        
        y_offset = 60
        for event in events:
            event_text = f"{event['type'].upper()}"
            cv2.putText(frame, event_text, (frame.shape[1] - 300, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y_offset += 25
    
    def _draw_motion_stats(self, frame: np.ndarray, analysis_result: Dict[str, Any]):
        """绘制运动统计信息"""
        stats = analysis_result.get('stats', {})
        
        stats_text = [
            f"Poses: {stats.get('total_pose_detections', 0)}",
            f"Balls: {stats.get('total_ball_detections', 0)}",
            f"Rackets: {stats.get('total_racket_detections', 0)}",
            f"Swings: {stats.get('swing_events', 0)}"
        ]
        
        # 在左下角显示统计信息
        y_start = frame.shape[0] - 120
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (10, y_start + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['roi_pose'], 2)
