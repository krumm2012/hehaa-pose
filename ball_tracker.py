# ball_tracker.py
import cv2
import numpy as np
import os
import queue
import scipy.signal as signal
from PIL import Image, ImageDraw

class BallTracker:
    def __init__(self, model_path, config):
        self.config = config
        self.model_path = model_path
        
        # 轨迹追踪状态
        self.tracked_balls_history = []  # 存储球的轨迹坐标 {'coords': (x,y), 'frame': frame_num}
        self.last_tracked_ball_coords = None
        self.frames_ball_lost = 0
        
        # 球的坐标队列 - 用于绘制轨迹
        self.ball_queue = queue.deque()
        for i in range(8):  # 存储8帧的历史轨迹
            self.ball_queue.appendleft(None)
        
        # 静态球过滤状态
        self.static_ball_candidates = {}
        self.next_ball_id = 0
        
        # 预处理相关参数
        self.process_width = 640
        self.process_height = 360
        
        # 配置参数
        self.config.setdefault('ball_trajectory_max_len', 50)
        self.config.setdefault('max_lost_frames_for_track', 5)  # 多少帧未检测到球视为丢失
        self.config.setdefault('max_ball_match_distance_px', 75)  # 关联检测到的球的最大距离
        self.config.setdefault('static_ball_movement_threshold_px', 5)
        self.config.setdefault('static_ball_frames_threshold', 8)  # 减少静态球判定阈值，更快识别静态球
        self.config.setdefault('ball_confidence_threshold', 127)  # 球检测热图的阈值
        self.config.setdefault('ball_detection_threshold', 30)    # HSV颜色阈值
        
        # 球尺寸过滤参数
        self.config.setdefault('min_ball_radius', 3)   # 最小球半径（像素）
        self.config.setdefault('max_ball_radius', 20)  # 最大球半径（像素）
        self.config.setdefault('min_ball_movement', 8) # 最小球移动距离（像素/帧）
        
        # 边界检查参数
        self.config.setdefault('use_boundary', False)
        self.config.setdefault('boundary_x1', 0)
        self.config.setdefault('boundary_y1', 0)
        self.config.setdefault('boundary_x2', 9999)
        self.config.setdefault('boundary_y2', 9999)
        self.config.setdefault('draw_boundary', False)
        
        # 初始化模拟参数
        self.sim_ball_pos = None
        self.sim_ball_vel = None
        self.sim_static_balls = []
        self.config.setdefault('sim_ball_detection_noise', 3) 
        self.config.setdefault('sim_num_static_balls', 1)
        
        # 记录前一帧检测到的球
        self.prev_ball_positions = []
        
        message = "使用HSV颜色分割和形状检测跟踪网球，已启用静止球过滤和尺寸过滤"
        if self.config['use_boundary']:
            message += "，已启用边界检查"
        print(message)
    
    def predict_ball(self, frame):
        """
        检测球的位置
        使用HSV颜色空间和Hough变换检测球的位置
        """
        # 1. 使用颜色检测方式识别网球
        detected_balls = self._detect_with_hsv(frame)
        
        # 2. 如果颜色检测失败，使用模拟模式
        if not detected_balls:
            detected_balls = self._simulate_ball_detection(frame)
            
        # 应用边界检查
        if self.config['use_boundary']:
            detected_balls = self._apply_boundary_check(detected_balls)
            
        return detected_balls
    
    def _apply_boundary_check(self, detected_balls):
        """检查球是否在指定的边界内"""
        x1 = self.config['boundary_x1']
        y1 = self.config['boundary_y1']
        x2 = self.config['boundary_x2']
        y2 = self.config['boundary_y2']
        
        filtered_balls = []
        for ball in detected_balls:
            x, y = ball
            if x1 <= x <= x2 and y1 <= y <= y2:
                filtered_balls.append(ball)
        
        return filtered_balls
    
    def _detect_with_hsv(self, frame):
        """使用HSV颜色空间检测网球"""
        # 预处理 - 调整大小以加快处理速度
        resized_frame = cv2.resize(frame, (self.process_width, self.process_height))
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
        
        # 网球的颜色范围 (荧光黄绿色)
        lower_ball = np.array([25, 50, 50])
        upper_ball = np.array([65, 255, 255])
        
        # 创建一个掩膜
        mask = cv2.inRange(hsv, lower_ball, upper_ball)
        
        # 对掩膜进行形态学操作以去除噪点
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # 使用Hough变换查找球（圆形）
        circles = cv2.HoughCircles(
            mask, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=10, 
            param1=50, 
            param2=8, 
            minRadius=self.config['min_ball_radius'], 
            maxRadius=self.config['max_ball_radius']
        )
        
        detected_balls = []
        valid_balls_with_radius = []
        
        if circles is not None:
            # 将圆的坐标调整为原始图像大小
            scale_x = frame.shape[1] / self.process_width
            scale_y = frame.shape[0] / self.process_height
            
            for circle in circles[0]:
                x = int(circle[0] * scale_x)
                y = int(circle[1] * scale_y)
                radius = int(circle[2] * scale_x)  # 假设x和y缩放比例相近
                
                # 存储候选球和它们的半径，用于后处理
                valid_balls_with_radius.append((x, y, radius))
        
        # 后处理：过滤尺寸不合理的球
        for x, y, radius in valid_balls_with_radius:
            # 检查球是否在合理半径范围内
            if (self.config['min_ball_radius'] <= radius <= self.config['max_ball_radius']):
                detected_balls.append((x, y))
        
        return detected_balls
    
    def _simulate_ball_detection(self, frame):
        """模拟球的检测（当其他方法失败时的备选）"""
        h, w = frame.shape[:2]
        detections = []
        
        # 模拟移动的球
        if self.sim_ball_pos is None:  # 初始化移动球
            self.sim_ball_pos = np.array([w / 2, h / 2], dtype=float)
            self.sim_ball_vel = np.array([np.random.uniform(-10, 10), np.random.uniform(-7, 7)], dtype=float)
        else:
            self.sim_ball_pos += self.sim_ball_vel
            
            # 边界反弹
            if not (0 < self.sim_ball_pos[0] < w - 1):
                self.sim_ball_vel[0] *= -0.85  # 减速并反向
                self.sim_ball_pos[0] = np.clip(self.sim_ball_pos[0], 0, w - 1)
            if not (0 < self.sim_ball_pos[1] < h - 1):
                self.sim_ball_vel[1] *= -0.85  # 减速并反向
                self.sim_ball_pos[1] = np.clip(self.sim_ball_pos[1], 0, h - 1)
            
            # 随机小幅度改变速度
            if np.random.random() < 0.03:  # 3%概率改变方向
                self.sim_ball_vel[0] += np.random.normal(0, 2)
                self.sim_ball_vel[1] += np.random.normal(0, 2)
                self.sim_ball_vel[0] = np.clip(self.sim_ball_vel[0], -15, 15)
                self.sim_ball_vel[1] = np.clip(self.sim_ball_vel[1], -10, 10)
        
        if np.random.random() > 0.05 and self.sim_ball_pos is not None:  # 95%的几率检测到球
            detected_x = self.sim_ball_pos[0] + np.random.normal(0, self.config['sim_ball_detection_noise'])
            detected_y = self.sim_ball_pos[1] + np.random.normal(0, self.config['sim_ball_detection_noise'])
            detections.append((int(detected_x), int(detected_y)))
        
        return detections
    
    def filter_static_candidates(self, detected_balls, frame_num):
        """过滤掉静止的球"""
        filtered_balls = []
        
        # 如果这是首帧，只存储位置
        if not self.prev_ball_positions:
            self.prev_ball_positions = detected_balls
            return detected_balls
        
        # 对每个检测到的球，计算与前一帧球的距离
        for ball in detected_balls:
            is_moving = True
            
            for prev_ball in self.prev_ball_positions:
                dist = np.sqrt((ball[0] - prev_ball[0])**2 + (ball[1] - prev_ball[1])**2)
                
                # 如果球几乎没有移动（静止），则标记为静止球
                if dist < self.config['min_ball_movement']:
                    is_moving = False
                    break
            
            if is_moving:
                filtered_balls.append(ball)
        
        # 更新前一帧球的位置
        self.prev_ball_positions = detected_balls
        
        return filtered_balls
    
    def advanced_ball_processing(self, all_detected_balls_current_frame, frame_num):
        """
        高级球处理：过滤静态球并追踪主要移动球，形成一致的轨迹
        返回：包含当前帧中单个活动球(x,y)坐标的列表，若无则为空列表
        更新：self.tracked_balls_history, self.static_ball_candidates, self.last_tracked_ball_coords
        """
        # 首先过滤静止和尺寸不合理的球
        filtered_balls = self.filter_static_candidates(all_detected_balls_current_frame, frame_num)
        
        # --- 第1部分: 静态球过滤 ---
        current_potential_moving_balls = []
        updated_static_candidates = {}
        unmatched_current_detections = list(filtered_balls)
        
        # 将当前检测与现有的静态候选进行匹配
        active_static_keys_this_frame = set()
        for ball_id, static_info in self.static_ball_candidates.items():
            best_match_dist = float('inf')
            best_match_idx = -1
            for i, current_ball_coords in enumerate(unmatched_current_detections):
                dist = np.linalg.norm(np.array(current_ball_coords) - np.array(static_info['coords']))
                # 较宽的初始匹配，然后是严格的运动阈值
                if dist < self.config['max_ball_match_distance_px'] / 2 and dist < best_match_dist:
                    best_match_dist = dist
                    best_match_idx = i
            
            if best_match_idx != -1:
                active_static_keys_this_frame.add(ball_id)
                matched_ball_coords = unmatched_current_detections.pop(best_match_idx)
                if best_match_dist < self.config['static_ball_movement_threshold_px']:
                    updated_static_candidates[ball_id] = {
                        'coords': matched_ball_coords,
                        'frames_still': static_info['frames_still'] + 1,
                        'last_seen_frame': frame_num
                    }
                else:  # 之前是静态的，但现在移动显著
                    current_potential_moving_balls.append(matched_ball_coords)
                    # 不再是静态候选
        
        # 新检测到的作为新的静态候选或潜在移动球
        for new_ball_coords in unmatched_current_detections:
            # 假设新检测到的初始为潜在移动，除非它们迅速变为静态
            current_potential_moving_balls.append(new_ball_coords)
            # 同时作为新的静态候选，将被后续确认或过滤
            new_id = self.next_ball_id
            updated_static_candidates[new_id] = {'coords': new_ball_coords, 'frames_still': 0, 'last_seen_frame': frame_num}
            active_static_keys_this_frame.add(new_id)
            self.next_ball_id += 1
        
        # 清除旧的静态候选
        final_static_candidates = {}
        for ball_id, info in updated_static_candidates.items():
            if frame_num - info['last_seen_frame'] < self.config['static_ball_frames_threshold'] * 3:  # 保留长一点时间
                final_static_candidates[ball_id] = info
        self.static_ball_candidates = final_static_candidates
        
        # 从`current_potential_moving_balls`中过滤掉确认为静态的球
        final_truly_moving_balls = []
        for mb_coords in current_potential_moving_balls:
            is_confirmed_static = False
            for static_id, static_info in self.static_ball_candidates.items():
                if static_info['frames_still'] >= self.config['static_ball_frames_threshold']:
                    dist = np.linalg.norm(np.array(mb_coords) - np.array(static_info['coords']))
                    if dist < self.config['static_ball_movement_threshold_px']:
                        is_confirmed_static = True
                        break
            if not is_confirmed_static:
                final_truly_moving_balls.append(mb_coords)
        
        # --- 第2部分: 从`final_truly_moving_balls`中追踪球 ---
        chosen_ball_for_trajectory = None
        
        if final_truly_moving_balls:
            self.frames_ball_lost = 0  # 重置丢失计数器
            
            if len(final_truly_moving_balls) == 1:
                chosen_ball_for_trajectory = final_truly_moving_balls[0]
            else:  # 多个潜在移动球，尝试匹配到最后已知位置
                if self.last_tracked_ball_coords:
                    min_dist = float('inf')
                    best_candidate = None
                    for ball_candidate_coords in final_truly_moving_balls:
                        dist = np.linalg.norm(np.array(ball_candidate_coords) - np.array(self.last_tracked_ball_coords))
                        if dist < min_dist and dist < self.config['max_ball_match_distance_px']:
                            min_dist = dist
                            best_candidate = ball_candidate_coords
                    chosen_ball_for_trajectory = best_candidate
                
                if not chosen_ball_for_trajectory:  # 仍然没有选择球
                    # 选择第一个球
                    chosen_ball_for_trajectory = final_truly_moving_balls[0]
        else:  # 本帧未检测到真正移动的球
            self.frames_ball_lost += 1
        
        # 更新轨迹历史
        if chosen_ball_for_trajectory:
            self.tracked_balls_history.append({'coords': chosen_ball_for_trajectory, 'frame': frame_num})
            self.last_tracked_ball_coords = chosen_ball_for_trajectory
            
            # 更新球队列
            self.ball_queue.appendleft(chosen_ball_for_trajectory)
            self.ball_queue.pop()
            
            # 修剪历史记录
            if len(self.tracked_balls_history) > self.config['ball_trajectory_max_len']:
                self.tracked_balls_history.pop(0)
        elif self.frames_ball_lost > self.config['max_lost_frames_for_track']:
            self.last_tracked_ball_coords = None  # 宣布球丢失
            
            # 更新球队列
            self.ball_queue.appendleft(None)
            self.ball_queue.pop()
        else:
            # 球暂时丢失但未超过阈值，使用None更新队列
            self.ball_queue.appendleft(None)
            self.ball_queue.pop()
        
        # 返回当前帧中识别为在场上的单个球
        return [chosen_ball_for_trajectory] if chosen_ball_for_trajectory else []
    
    def draw_ball(self, frame, active_ball_coords_list):
        """在帧上绘制活动球（列表应该只有0个或1个元素）"""
        if active_ball_coords_list:  # advanced_ball_processing的输出
            ball_coords = active_ball_coords_list[0]
            cv2.circle(frame, tuple(map(int, ball_coords)), 7, (0, 0, 255), -1)  # 亮红色代表活动球
        return frame
    
    def draw_trajectory(self, frame):
        """在帧上绘制球轨迹"""
        # 从PIL格式开始处理，以便于绘制
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 绘制前8帧的轨迹点（从队列）
        for i, coords in enumerate(self.ball_queue):
            if coords is not None:
                draw_x, draw_y = coords
                # 根据距离当前帧的远近调整点的大小
                size = 2 + (8 - i) * 0.5
                bbox = (draw_x - size, draw_y - size, draw_x + size, draw_y + size)
                # 绘制黄色椭圆
                draw.ellipse(bbox, outline='yellow')
        
        # 将PIL图像转回OpenCV格式
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 绘制连接线（从轨迹历史）
        if len(self.tracked_balls_history) > 1:
            for i in range(1, min(len(self.tracked_balls_history), 30)):  # 限制显示最近30帧
                pt1 = tuple(map(int, self.tracked_balls_history[i-1]['coords']))
                pt2 = tuple(map(int, self.tracked_balls_history[i]['coords']))
                # 根据轨迹点的新旧调整线条颜色
                alpha = max(0.3, 1.0 - i * 0.03)  # 较新的点更明亮
                color = (int(0 * alpha), int(255 * alpha), int(0 * alpha))  # 绿色，透明度渐变
                cv2.line(frame, pt1, pt2, color, 2)
        
        # 绘制监控区域边界
        if self.config['use_boundary'] and self.config['draw_boundary']:
            x1 = self.config['boundary_x1']
            y1 = self.config['boundary_y1']
            x2 = self.config['boundary_x2']
            y2 = self.config['boundary_y2']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 黄色矩形边界
        
        return frame
    
    def draw_static_balls(self, frame):
        """在帧上绘制确认的静态球"""
        # 我们已经过滤了静态球，但可以保留这个函数用于调试
        # 默认情况下现在不绘制静态球
        if self.config.get('draw_static_balls', False):
            for ball_id, static_info in self.static_ball_candidates.items():
                if static_info['frames_still'] >= self.config['static_ball_frames_threshold']:
                    cv2.circle(frame, tuple(map(int, static_info['coords'])), 5, (128, 128, 128), -1)  # 灰色表示静态球
        return frame
    
    def interpolate_trajectory(self):
        """插值轨迹中的缺失点"""
        if not self.tracked_balls_history:
            return
        
        coords = [entry['coords'] for entry in self.tracked_balls_history]
        frames = [entry['frame'] for entry in self.tracked_balls_history]
        
        # 提取x和y坐标
        x_coords = np.array([c[0] for c in coords])
        y_coords = np.array([c[1] for c in coords])
        
        # 平滑轨迹
        if len(x_coords) > 3:  # 需要至少3个点才能进行平滑
            try:
                window_size = min(7, len(x_coords) - 2)  # 窗口大小为7或更小
                if window_size % 2 == 0:  # 确保窗口大小为奇数
                    window_size -= 1
                if window_size >= 3:  # 确保至少有3个点
                    x_smoothed = signal.savgol_filter(x_coords, window_size, 2)
                    y_smoothed = signal.savgol_filter(y_coords, window_size, 2)
                    
                    # 更新轨迹
                    for i in range(len(self.tracked_balls_history)):
                        self.tracked_balls_history[i]['coords'] = (x_smoothed[i], y_smoothed[i])
            except Exception as e:
                print(f"轨迹平滑处理出错: {e}")
    
    def remove_outliers(self, threshold=3.0):
        """移除轨迹中的离群值"""
        if len(self.tracked_balls_history) < 3:
            return
        
        # 计算连续点之间的距离
        dists = []
        coords = [entry['coords'] for entry in self.tracked_balls_history]
        
        for i in range(1, len(coords)):
            p1 = np.array(coords[i-1])
            p2 = np.array(coords[i])
            dist = np.linalg.norm(p2 - p1)
            dists.append(dist)
        
        # 计算距离的均值和标准差
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        
        # 标记离群值
        outliers = []
        for i in range(len(dists)):
            if dists[i] > mean_dist + threshold * std_dist:
                # i+1是离群点的索引（因为dists从索引1的点开始）
                outliers.append(i + 1)
        
        # 替换离群值（简单地用前一个点替代）
        for idx in outliers:
            if 0 < idx < len(self.tracked_balls_history):
                # 使用前一个点的坐标
                self.tracked_balls_history[idx]['coords'] = self.tracked_balls_history[idx-1]['coords']