# ball_tracker.py
import cv2
import numpy as np
import os
import queue
import scipy.signal as signal
from PIL import Image, ImageDraw
import logging

class BallTracker:
    def __init__(self, model_path, config, roi_manager=None):
        self.config = config
        self.model_path = model_path
        self.roi_manager = roi_manager  # ROI管理器，用于区域过滤
        
        # 🔍 初始化球检测调试日志
        self.debug_config = config.get('ball_detection_debug', {})
        self.debug_enabled = self.debug_config.get('enabled', False)
        self.frame_counter = 0
        
        if self.debug_enabled:
            # 设置专用的球检测日志器
            self.logger = logging.getLogger('BallDetection')
            if not self.logger.handlers:
                # 同时输出到终端与文件
                handler_console = logging.StreamHandler()
                handler_file = logging.FileHandler('ball_detection.log', mode='w', encoding='utf-8')
                formatter = logging.Formatter('🎾 [球检测-帧%(frame_num)s] %(message)s')
                handler_console.setFormatter(formatter)
                handler_file.setFormatter(formatter)
                self.logger.addHandler(handler_console)
                self.logger.addHandler(handler_file)
                self.logger.setLevel(logging.INFO)
                # 避免同时沿用根日志器导致重复输出
                self.logger.propagate = False
            self._log_config_info()
        else:
            self.logger = None
        
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
        
        # 球尺寸过滤参数 - 从配置文件读取，提供默认值
        self.config.setdefault('min_ball_radius', 3)   # 最小球半径（像素）
        self.config.setdefault('max_ball_radius', 45)  # 最大球半径（像素）- 允许更大的球
        self.config.setdefault('min_ball_movement', 8) # 最小球移动距离（像素/帧）
        
        # 🎯 噪点过滤参数 - 优化后的默认值，更宽松
        self.config.setdefault('noise_filter_quality_threshold', 0.25)      # 质量阈值（降低以减少误识别）
        self.config.setdefault('noise_filter_circularity_threshold', 0.58)  # 圆度阈值（降低以减少误识别）
        self.config.setdefault('noise_filter_min_edge_distance', 5)         # 最小边缘距离（降低以减少误识别）
        
        # 边界检查参数
        self.config.setdefault('use_boundary', False)
        self.config.setdefault('boundary_x1', 0)
        self.config.setdefault('boundary_y1', 0)
        self.config.setdefault('boundary_x2', 9999)
        self.config.setdefault('boundary_y2', 9999)
        self.config.setdefault('draw_boundary', False)
        
        # 🚫 屏蔽区域配置参数
        self.config.setdefault('use_mask_zones', False)
        self.config.setdefault('mask_zones', [])  # 屏蔽区域列表，格式: [{'x': x, 'y': y, 'width': w, 'height': h, 'name': 'zone_name'}]
        self.config.setdefault('draw_mask_zones', False)  # 是否绘制屏蔽区域

        # 模块化开关与回退策略
        self.config.setdefault('enable_hough_detection', True)
        self.config.setdefault('enable_contour_fallback', True)
        self.config.setdefault('allow_simulation_fallback', False)
        
        # 可选：仅当显式启用use_mask_zones且未提供mask_zones时，添加默认屏蔽区域
        if self.config.get('use_mask_zones', False) and not self.config.get('mask_zones'):
            default_mask_zone = {
                'x': 661,
                'y': 391,
                'width': 20,
                'height': 20,
                'name': 'default_mask_zone'
            }
            self.config['mask_zones'] = [default_mask_zone]
        
        # 初始化模拟参数
        self.sim_ball_pos = None
        self.sim_ball_vel = None
        self.sim_static_balls = []
        self.config.setdefault('sim_ball_detection_noise', 3) 
        self.config.setdefault('sim_num_static_balls', 1)
        
        # 记录前一帧检测到的球
        self.prev_ball_positions = []
        
        # 🔍 显示当前使用的所有球检测参数
        print("\n🎾 === 球检测配置参数 ===")
        print(f"球半径范围: {self.config['min_ball_radius']}-{self.config['max_ball_radius']}px")
        print(f"质量阈值: {self.config['noise_filter_quality_threshold']}")
        print(f"圆度阈值: {self.config['noise_filter_circularity_threshold']}")
        print(f"边缘距离阈值: {self.config['noise_filter_min_edge_distance']}px")
        print(f"最小移动距离: {self.config['min_ball_movement']}px")
        print(f"静态球判定阈值: {self.config['static_ball_movement_threshold_px']}px")
        print(f"静态球帧数阈值: {self.config['static_ball_frames_threshold']}")
        
        message = "使用HSV颜色分割和形状检测跟踪网球，已启用静止球过滤和尺寸过滤"
        if self.config['use_boundary']:
            message += "，已启用边界检查"
            print(f"边界范围: ({self.config['boundary_x1']},{self.config['boundary_y1']})-({self.config['boundary_x2']},{self.config['boundary_y2']})")
        else:
            print("边界检查: 已关闭")
            
        if self.config.get('use_mask_zones', False):
            mask_count = len(self.config.get('mask_zones', []))
            message += f"，已启用屏蔽区域({mask_count}个区域)"
            print(f"屏蔽区域: {mask_count}个区域")
        else:
            print("屏蔽区域: 已关闭")
        print("========================\n")
        print(message)
    
    def _log_config_info(self):
        """记录球检测配置信息"""
        if not self.debug_enabled:
            return
        
        self.logger.info("=== 球检测配置信息 ===", extra={'frame_num': 'CONFIG'})
        self.logger.info(f"HSV检测: {self.debug_config.get('log_hsv_detection', False)}", extra={'frame_num': 'CONFIG'})
        self.logger.info(f"边界检查: {self.debug_config.get('log_boundary_check', False)}", extra={'frame_num': 'CONFIG'})
        self.logger.info(f"屏蔽区域: {self.debug_config.get('log_mask_zones', False)}", extra={'frame_num': 'CONFIG'})
        self.logger.info(f"尺寸过滤: {self.debug_config.get('log_size_filtering', False)}", extra={'frame_num': 'CONFIG'})
        self.logger.info(f"静态过滤: {self.debug_config.get('log_static_filtering', False)}", extra={'frame_num': 'CONFIG'})
        self.logger.info(f"球尺寸范围: {self.config['min_ball_radius']}-{self.config['max_ball_radius']}px", extra={'frame_num': 'CONFIG'})
        self.logger.info(f"边界范围: ({self.config['boundary_x1']},{self.config['boundary_y1']})-({self.config['boundary_x2']},{self.config['boundary_y2']})", extra={'frame_num': 'CONFIG'})
        if self.config.get('use_mask_zones', False):
            mask_zones = self.config.get('mask_zones', [])
            self.logger.info(f"屏蔽区域数量: {len(mask_zones)}", extra={'frame_num': 'CONFIG'})
            for i, zone in enumerate(mask_zones):
                self.logger.info(f"屏蔽区域{i+1}: ({zone['x']},{zone['y']}) {zone['width']}x{zone['height']}", extra={'frame_num': 'CONFIG'})
        self.logger.info("==================", extra={'frame_num': 'CONFIG'})
    
    def _debug_log(self, message, log_type='general'):
        """统一的调试日志输出"""
        if not self.debug_enabled or not self.logger:
            return
        
        # 检查特定类型的日志是否启用
        log_enabled = {
            'hsv': self.debug_config.get('log_hsv_detection', False),
            'boundary': self.debug_config.get('log_boundary_check', False),
            'mask': self.debug_config.get('log_mask_zones', False),
            'size': self.debug_config.get('log_size_filtering', False),
            'static': self.debug_config.get('log_static_filtering', False),
            'advanced': self.debug_config.get('log_advanced_processing', False),
            'simulation': self.debug_config.get('log_simulation', False),
            'roi': self.debug_config.get('log_roi_filtering', True),  # ROI过滤日志
            'general': True
        }
        
        if log_enabled.get(log_type, True):
            self.logger.info(message, extra={'frame_num': self.frame_counter})
    
    def predict_ball(self, frame):
        """
        检测球的位置
        使用HSV颜色空间和Hough变换检测球的位置
        """
        self.frame_counter += 1
        
        # 🔍 开始日志记录
        self._debug_log(f"开始球检测 (帧尺寸: {frame.shape[1]}x{frame.shape[0]})")
        
        # 1. 使用颜色检测方式识别网球
        detected_balls = self._detect_with_hsv(frame)
        self._debug_log(f"HSV检测结果: {len(detected_balls)} 个球", 'hsv')
        
        # 2. 如果颜色检测失败，根据配置决定是否使用模拟模式
        if not detected_balls and self.config.get('allow_simulation_fallback', False):
            self._debug_log("HSV检测失败，启用模拟检测(已允许)", 'simulation')
            detected_balls = self._simulate_ball_detection(frame)
            self._debug_log(f"模拟检测结果: {len(detected_balls)} 个球", 'simulation')
            
        # 应用边界检查
        if self.config['use_boundary']:
            original_count = len(detected_balls)
            detected_balls = self._apply_boundary_check(detected_balls)
            filtered_count = len(detected_balls)
            self._debug_log(f"边界过滤: {original_count} → {filtered_count} 个球", 'boundary')
        
        # 🚫 应用屏蔽区域检查
        if self.config.get('use_mask_zones', False):
            original_count = len(detected_balls)
            detected_balls = self._apply_mask_zones_check(detected_balls)
            filtered_count = len(detected_balls)
            self._debug_log(f"屏蔽区域过滤: {original_count} → {filtered_count} 个球", 'mask')
        
        # 🎯 应用ROI过滤 - 修复坐标系统问题
        if self.roi_manager and self.roi_manager.is_roi_set:
            original_count = len(detected_balls)
            
            # 修复：将ROI裁剪帧坐标转换为原始帧坐标
            # 注意：这里需要知道ROI的偏移量，但ball_tracker没有这个信息
            # 临时解决方案：跳过ROI过滤，因为ROI裁剪已经确保了检测区域
            self._debug_log(f"ROI过滤: 跳过（已在ROI裁剪帧上检测）", 'roi')
            
            # 如果需要严格的ROI过滤，需要从外部传入ROI偏移量
            # detected_balls = self.roi_manager.filter_detections_by_roi(detected_balls, "ball")
            # filtered_count = len(detected_balls)
            # self._debug_log(f"ROI过滤: {original_count} → {filtered_count} 个球", 'roi')
            
        self._debug_log(f"最终检测结果: {len(detected_balls)} 个球")
        
        # 🔍 保存调试帧（如果启用）
        if (self.debug_config.get('save_debug_frames', False) and 
            self.frame_counter % self.debug_config.get('debug_frame_interval', 10) == 0):
            self._save_debug_frame(frame, detected_balls)
            
        return detected_balls
    
    def _apply_boundary_check(self, detected_balls):
        """检查球是否在指定的边界内"""
        x1 = self.config['boundary_x1']
        y1 = self.config['boundary_y1']
        x2 = self.config['boundary_x2']
        y2 = self.config['boundary_y2']
        
        filtered_balls = []
        for i, ball in enumerate(detected_balls):
            x, y = ball
            in_boundary = x1 <= x <= x2 and y1 <= y <= y2
            
            if in_boundary:
                filtered_balls.append(ball)
                self._debug_log(f"球 {i+1}: ({x},{y}) ✅ 在边界内", 'boundary')
            else:
                self._debug_log(f"球 {i+1}: ({x},{y}) ❌ 超出边界 [边界: ({x1},{y1})-({x2},{y2})]", 'boundary')
        
        return filtered_balls
    
    def _apply_mask_zones_check(self, detected_balls):
        """检查球是否在屏蔽区域内，如果在则过滤掉。
        注意：屏蔽区坐标以“原始全帧”为基准。如果当前检测在ROI裁剪帧上进行，
        则需要将屏蔽区坐标转换到ROI局部坐标系（减去ROI偏移量）。"""
        if not self.config.get('mask_zones'):
            return detected_balls

        # 计算ROI偏移（若启用ROI裁剪）以便把全局屏蔽坐标映射到ROI局部坐标
        offset_x, offset_y = 0, 0
        if self.roi_manager and self.roi_manager.is_roi_set:
            roi_bbox = self.roi_manager.get_roi_bounding_box()
            if roi_bbox:
                x1, y1, x2, y2 = roi_bbox
                margin = int(self.config.get('roi_settings', {}).get('crop_margin', 12))
                offset_x = max(0, x1 - margin)
                offset_y = max(0, y1 - margin)
                # 日志说明实际使用的ROI偏移
                self._debug_log(f"屏蔽区域：应用ROI偏移 ({offset_x},{offset_y}) 以匹配裁剪帧坐标", 'mask')

        filtered_balls = []

        for i, ball in enumerate(detected_balls):
            x, y = ball  # 此坐标相对于当前检测帧（可能是ROI裁剪帧）
            is_in_mask_zone = False
            zone_name = ""

            # 检查球是否在任何屏蔽区域内（将全局屏蔽区转换为局部坐标再比较）
            for zone in self.config['mask_zones']:
                global_x1 = zone['x']
                global_y1 = zone['y']
                width = zone['width']
                height = zone['height']

                # 转换至ROI局部坐标
                zone_x1_local = global_x1 - offset_x
                zone_y1_local = global_y1 - offset_y
                zone_x2_local = zone_x1_local + width
                zone_y2_local = zone_y1_local + height

                if zone_x1_local <= x <= zone_x2_local and zone_y1_local <= y <= zone_y2_local:
                    is_in_mask_zone = True
                    zone_name = zone.get('name', f'zone_{global_x1}_{global_y1}')
                    break

            if not is_in_mask_zone:
                filtered_balls.append(ball)
                self._debug_log(f"球 {i+1}: ({x},{y}) ✅ 不在屏蔽区域内", 'mask')
            else:
                self._debug_log(f"球 {i+1}: ({x},{y}) ❌ 在屏蔽区域 '{zone_name}' 内（已按ROI偏移换算），已过滤", 'mask')

        return filtered_balls
    
    def _detect_with_hsv(self, frame):
        """
        优化的HSV颜色空间网球检测 - 增强噪点过滤
        """
        # 预处理 - 调整大小以加快处理速度
        resized_frame = cv2.resize(frame, (self.process_width, self.process_height))
        self._debug_log(f"图像预处理: {frame.shape[1]}x{frame.shape[0]} → {self.process_width}x{self.process_height}", 'hsv')
        
        # 🔧 增强预处理 - 噪点过滤
        # 1. 高斯滤波去除噪点
        resized_frame = cv2.GaussianBlur(resized_frame, (3, 3), 0.5)
        
        # 2. 中值滤波进一步去除椒盐噪点
        resized_frame = cv2.medianBlur(resized_frame, 3)
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
        
        # 从配置读取HSV范围（优先），否则使用合理默认值
        lower_h = int(self.config.get('hsv_lower_hue', 20))
        upper_h = int(self.config.get('hsv_upper_hue', 70))
        lower_s = int(self.config.get('hsv_lower_sat', 50))
        upper_s = int(self.config.get('hsv_upper_sat', 255))
        lower_v = int(self.config.get('hsv_lower_val', 50))
        upper_v = int(self.config.get('hsv_upper_val', 255))

        lower_ball = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
        upper_ball = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)

        self._debug_log(f"HSV颜色范围: [{lower_h},{lower_s},{lower_v}] - [{upper_h},{upper_s},{upper_v}]", 'hsv')

        # 单一配置范围掩膜
        mask = cv2.inRange(hsv, lower_ball, upper_ball)
        
        mask_pixels = np.sum(mask > 0)
        self._debug_log(f"颜色匹配像素数: {mask_pixels}", 'hsv')
        
        # 🧹 增强的形态学操作 - 更强的噪点过滤
        kernel_tiny = np.ones((1, 1), np.uint8)     # 微小噪点
        kernel_small = np.ones((2, 2), np.uint8)    # 小噪点
        kernel_medium = np.ones((3, 3), np.uint8)   # 中等噪点
        kernel_large = np.ones((4, 4), np.uint8)    # 较大结构
        
        # 阶段1: 去除微小噪点 (开运算)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_tiny, iterations=1)
        
        # 阶段2: 填充球内部的小空洞 (闭运算)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # 阶段3: 去除中等噪点 (再次开运算)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 阶段4: 区域面积过滤 - 去除过小的连通区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 将面积阈值从原始帧坐标系尺度换算到缩放掩膜坐标系
        scale_factor = self.process_width / frame.shape[1]
        min_area_cfg = float(self.config.get('min_ball_area', 100))
        max_area_cfg = float(self.config.get('max_ball_area', 2000))
        min_area = int(max(1, min_area_cfg * (scale_factor ** 2)))
        max_area = int(max_area_cfg * (scale_factor ** 2))
        self._debug_log(f"面积阈值(缩放后): min={min_area} max={max_area}", 'hsv')
        
        # 创建清理后的掩膜
        clean_mask = np.zeros_like(mask)
        valid_contours = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                cv2.drawContours(clean_mask, [contour], -1, 255, -1)
                valid_contours += 1
            else:
                self._debug_log(f"区域面积过滤: 面积{area:.1f}px 超出范围[{min_area}-{max_area}]", 'hsv')
        
        mask = clean_mask
        self._debug_log(f"面积过滤后有效区域数: {valid_contours}", 'hsv')
        
        # 阶段5: 最后的膨胀以确保球的完整性
        mask = cv2.dilate(mask, kernel_medium, iterations=1)
        
        cleaned_pixels = np.sum(mask > 0)
        self._debug_log(f"形态学清理后像素数: {cleaned_pixels} (减少了{mask_pixels - cleaned_pixels})", 'hsv')
        
        # 使用可配置的Hough变换参数（半径与距离在缩放后坐标系下计算）
        hough_dp = self.config.get('hough_dp', 1)
        # 半径在配置中以原始帧像素为单位，这里换算到缩放掩膜坐标系
        min_radius_cfg = int(self.config.get('min_ball_radius', 3))
        max_radius_cfg = int(self.config.get('max_ball_radius', 45))
        min_radius_resized = max(3, int(min_radius_cfg * scale_factor))
        max_radius_resized = max(min_radius_resized + 1, int(max_radius_cfg * scale_factor))
        # 圆心最小距离同样在缩放坐标系
        default_min_dist = max(12, min_radius_resized * 2)
        hough_min_dist = int(self.config.get('hough_min_dist', default_min_dist))
        hough_param1 = self.config.get('hough_param1', 35)
        hough_param2 = self.config.get('hough_param2', 6)

        self._debug_log(
            f"Hough参数(缩放后): minR={min_radius_resized}, maxR={max_radius_resized}, minDist={hough_min_dist}",
            'hsv'
        )

        circles = None
        if self.config.get('enable_hough_detection', True):
            circles = cv2.HoughCircles(
                mask,
                cv2.HOUGH_GRADIENT,
                dp=hough_dp,
                minDist=hough_min_dist,
                param1=hough_param1,
                param2=hough_param2,
                minRadius=min_radius_resized,
                maxRadius=max_radius_resized
            )
        
        detected_balls = []
        valid_balls_with_radius = []
        
        if circles is not None:
            self._debug_log(f"Hough圆检测找到: {len(circles[0])} 个圆", 'hsv')
            # 将圆的坐标调整为原始图像大小
            scale_x = frame.shape[1] / self.process_width
            scale_y = frame.shape[0] / self.process_height
            
            for i, circle in enumerate(circles[0]):
                x = int(circle[0] * scale_x)
                y = int(circle[1] * scale_y)
                radius = int(circle[2] * scale_x)  # 假设x和y缩放比例相近
                
                self._debug_log(f"圆 {i+1}: 位置({x},{y}), 半径{radius}px", 'hsv')
                
                # 验证圆的质量 - 检查原始图像中的颜色分布
                quality_score = self._evaluate_ball_quality(frame, x, y, radius)
                self._debug_log(f"圆 {i+1}: 质量分数 {quality_score:.2f}", 'hsv')
                
                # 存储候选球和它们的信息
                valid_balls_with_radius.append((x, y, radius, quality_score))
        else:
            self._debug_log("Hough圆检测未找到任何圆", 'hsv')
            # 轮廓回退（可配置）
            if self.config.get('enable_contour_fallback', True):
                try:
                    contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    min_area_cfg = int(self.config.get('min_ball_area', 0))
                    if contours2:
                        # 选最大面积且面积>=阈值的轮廓
                        contours2 = [c for c in contours2 if cv2.contourArea(c) >= min_area_cfg]
                        if contours2:
                            largest = max(contours2, key=cv2.contourArea)
                            (cx, cy), r = cv2.minEnclosingCircle(largest)
                            x, y, radius = int(cx), int(cy), int(r)
                            quality_score = self._evaluate_ball_quality(frame, x, y, max(radius, 4))
                            self._debug_log(
                                f"轮廓回退: 位置({x},{y}), 半径{radius}px, 面积{cv2.contourArea(largest):.1f}, 质量{quality_score:.2f}",
                                'hsv'
                            )
                            valid_balls_with_radius.append((x, y, radius, quality_score))
                except Exception as e:
                    self._debug_log(f"轮廓回退失败: {e}", 'hsv')
        
        # 🔍 增强的后处理：多层过滤系统
        # 按半径从大到小排序，优先保留“大球”
        if valid_balls_with_radius:
            valid_balls_with_radius.sort(key=lambda t: t[2], reverse=True)
        for i, (x, y, radius, quality) in enumerate(valid_balls_with_radius):
            
            # 过滤器1: 严格的尺寸检查
            min_radius = max(self.config['min_ball_radius'], 4)  # 至少4像素半径
            max_radius = self.config['max_ball_radius']
            size_valid = (min_radius <= radius <= max_radius)
            
            # 过滤器2: 提高质量阈值
            quality_threshold = self.config.get('noise_filter_quality_threshold', 0.4)  # 提高质量阈值
            quality_valid = quality >= quality_threshold
            
            # 过滤器3: 检查球的圆度（排除拉长的形状）
            circularity_score = self._check_ball_circularity(frame, x, y, radius)
            # 使用配置中的圆度阈值；前面已通过 setdefault 保证该键存在
            circularity_threshold = self.config['noise_filter_circularity_threshold']
            circularity_valid = circularity_score >= circularity_threshold
            
            # 过滤器4: 检查周围环境（避免边缘和角落的噪点）
            edge_distance = min(x, y, frame.shape[1] - x, frame.shape[0] - y)
            min_edge_distance = self.config.get('noise_filter_min_edge_distance', 15)
            edge_valid = edge_distance >= min_edge_distance
            
            # 综合判断
            all_valid = size_valid and quality_valid and edge_valid and circularity_valid
            
            if all_valid:
                detected_balls.append((x, y))
                self._debug_log(f"球 {i+1}: ({x},{y}) ✅ 半径{radius}px, 质量{quality:.2f}, 圆度{circularity_score:.2f}, 边距{edge_distance}px 通过过滤", 'size')
            else:
                reason = []
                if not size_valid:
                    reason.append(f"半径{radius}px超出范围[{min_radius}-{max_radius}]")
                if not quality_valid:
                    reason.append(f"质量{quality:.2f}低于{quality_threshold}")
                if not circularity_valid:
                    reason.append(f"圆度{circularity_score:.2f}低于{circularity_threshold}")
                if not edge_valid:
                    reason.append(f"边距{edge_distance}px小于{min_edge_distance}")
                self._debug_log(f"球 {i+1}: ({x},{y}) ❌ {', '.join(reason)}", 'size')
        
        self._debug_log(f"噪点过滤完成: {len(valid_balls_with_radius)} → {len(detected_balls)} 个球", 'hsv')
        # 只返回最大球（如果配置偏好）
        prefer_largest = self.config.get('ball_detection_strategy', {}).get('prefer_largest_ball', True)
        if prefer_largest and detected_balls:
            return [detected_balls[0]]
        return detected_balls
    
    def _evaluate_ball_quality(self, frame, x, y, radius):
        """
        评估检测到的球的质量
        基于颜色一致性、形状和周围环境
        """
        try:
            # 确保圆在图像边界内
            h, w = frame.shape[:2]
            if x - radius < 0 or x + radius >= w or y - radius < 0 or y + radius >= h:
                return 0.1  # 边界外的球质量很低
            
            # 提取球的区域
            ball_region = frame[max(0, y-radius):min(h, y+radius), 
                              max(0, x-radius):min(w, x+radius)]
            
            if ball_region.size == 0:
                return 0.1
            
            # 转换到HSV空间
            hsv_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)
            
            # 计算网球颜色像素的比例
            lower_ball = np.array([20, 40, 40])
            upper_ball = np.array([70, 255, 255])
            mask = cv2.inRange(hsv_region, lower_ball, upper_ball)
            
            total_pixels = ball_region.shape[0] * ball_region.shape[1]
            ball_pixels = np.sum(mask > 0)
            color_ratio = ball_pixels / total_pixels if total_pixels > 0 else 0
            
            # 计算亮度标准差（网球通常有相对均匀的亮度）
            gray_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2GRAY)
            brightness_std = np.std(gray_region)
            
            # 计算圆度（检查是否接近圆形）
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            circularity = 0.5  # 默认值
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    circularity = min(1.0, circularity)  # 限制在0-1之间
            
            # 综合评分
            quality = (color_ratio * 0.5 +                     # 50% 权重给颜色匹配
                      (1.0 - brightness_std / 100.0) * 0.3 +   # 30% 权重给亮度一致性
                      circularity * 0.2)                       # 20% 权重给形状
            
            return max(0.0, min(1.0, quality))  # 确保在0-1范围内
            
        except Exception as e:
            self._debug_log(f"球质量评估出错: {e}", 'hsv')
            return 0.3  # 出错时返回中等质量分数
    
    def _check_ball_circularity(self, frame, x, y, radius):
        """
        检查检测到的球的圆度，用于过滤噪点
        """
        try:
            # 确保在图像边界内
            h, w = frame.shape[:2]
            if x - radius < 0 or x + radius >= w or y - radius < 0 or y + radius >= h:
                return 0.0
            
            # 提取球的区域
            ball_region = frame[max(0, y-radius):min(h, y+radius), 
                              max(0, x-radius):min(w, x+radius)]
            
            if ball_region.size == 0:
                return 0.0
            
            # 转换为灰度
            gray_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2GRAY)
            
            # 二值化
            _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            # 找到最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter == 0 or area == 0:
                return 0.0
            
            # 计算圆度: 4π*面积/周长²
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 限制在0-1之间
            return max(0.0, min(1.0, circularity))
            
        except Exception as e:
            self._debug_log(f"圆度检查出错: {e}", 'hsv')
            return 0.5  # 出错时返回中等分数
    
    def _simulate_ball_detection(self, frame):
        """模拟球的检测（当其他方法失败时的备选）"""
        h, w = frame.shape[:2]
        detections = []
        
        self._debug_log(f"开始模拟球检测 (帧尺寸: {w}x{h})", 'simulation')
        
        # 模拟移动的球
        if self.sim_ball_pos is None:  # 初始化移动球
            self.sim_ball_pos = np.array([w / 2, h / 2], dtype=float)
            self.sim_ball_vel = np.array([np.random.uniform(-10, 10), np.random.uniform(-7, 7)], dtype=float)
            self._debug_log(f"初始化模拟球: 位置({self.sim_ball_pos[0]:.1f},{self.sim_ball_pos[1]:.1f}), 速度({self.sim_ball_vel[0]:.1f},{self.sim_ball_vel[1]:.1f})", 'simulation')
        else:
            old_pos = self.sim_ball_pos.copy()
            self.sim_ball_pos += self.sim_ball_vel
            
            # 边界反弹
            if not (0 < self.sim_ball_pos[0] < w - 1):
                self.sim_ball_vel[0] *= -0.85  # 减速并反向
                self.sim_ball_pos[0] = np.clip(self.sim_ball_pos[0], 0, w - 1)
                self._debug_log(f"模拟球X方向反弹: 速度{self.sim_ball_vel[0]:.1f}", 'simulation')
            if not (0 < self.sim_ball_pos[1] < h - 1):
                self.sim_ball_vel[1] *= -0.85  # 减速并反向
                self.sim_ball_pos[1] = np.clip(self.sim_ball_pos[1], 0, h - 1)
                self._debug_log(f"模拟球Y方向反弹: 速度{self.sim_ball_vel[1]:.1f}", 'simulation')
            
            # 随机小幅度改变速度
            if np.random.random() < 0.03:  # 3%概率改变方向
                old_vel = self.sim_ball_vel.copy()
                self.sim_ball_vel[0] += np.random.normal(0, 2)
                self.sim_ball_vel[1] += np.random.normal(0, 2)
                self.sim_ball_vel[0] = np.clip(self.sim_ball_vel[0], -15, 15)
                self.sim_ball_vel[1] = np.clip(self.sim_ball_vel[1], -10, 10)
                self._debug_log(f"模拟球随机变向: ({old_vel[0]:.1f},{old_vel[1]:.1f}) → ({self.sim_ball_vel[0]:.1f},{self.sim_ball_vel[1]:.1f})", 'simulation')
            
            self._debug_log(f"模拟球移动: ({old_pos[0]:.1f},{old_pos[1]:.1f}) → ({self.sim_ball_pos[0]:.1f},{self.sim_ball_pos[1]:.1f})", 'simulation')
        
        detection_probability = 0.95
        if np.random.random() > 0.05 and self.sim_ball_pos is not None:  # 95%的几率检测到球
            detected_x = self.sim_ball_pos[0] + np.random.normal(0, self.config['sim_ball_detection_noise'])
            detected_y = self.sim_ball_pos[1] + np.random.normal(0, self.config['sim_ball_detection_noise'])
            detections.append((int(detected_x), int(detected_y)))
            self._debug_log(f"模拟检测成功: 真实位置({self.sim_ball_pos[0]:.1f},{self.sim_ball_pos[1]:.1f}), 检测位置({int(detected_x)},{int(detected_y)})", 'simulation')
        else:
            self._debug_log(f"模拟检测失败 (概率事件: {1-detection_probability:.1%})", 'simulation')
        
        return detections
    
    def filter_static_candidates(self, detected_balls, frame_num):
        """
        改进的静止球过滤器 - 使用多帧历史和持续性检查
        """
        filtered_balls = []
        
        # 如果这是首帧：仅建立基准，不直接把所有检测当作“运动”
        if not self.prev_ball_positions:
            self.prev_ball_positions = detected_balls
            # 初始化球的历史追踪
            if not hasattr(self, 'ball_movement_history'):
                self.ball_movement_history = {}
            return []
        
        # 初始化球的移动历史字典（如果不存在）
        if not hasattr(self, 'ball_movement_history'):
            self.ball_movement_history = {}
        
        # 对每个检测到的球进行多帧分析
        for ball in detected_balls:
            is_moving = False
            should_include = False
            ball_key = f"{int(ball[0]/10)}_{int(ball[1]/10)}"  # 创建球的区域标识符
            
            # 计算与前一帧所有球的最小距离
            min_distance = float('inf')
            for prev_ball in self.prev_ball_positions:
                dist = np.sqrt((ball[0] - prev_ball[0])**2 + (ball[1] - prev_ball[1])**2)
                min_distance = min(min_distance, dist)
            
            # 更新球的移动历史
            if ball_key not in self.ball_movement_history:
                self.ball_movement_history[ball_key] = {
                    'positions': [ball],
                    'movements': [],
                    'frame_first_seen': frame_num,
                    'static_count': 0,
                    'moving_count': 0
                }
            else:
                history = self.ball_movement_history[ball_key]
                history['positions'].append(ball)
                history['movements'].append(min_distance)
                
                # 保持历史长度在合理范围内
                max_history_frames = self.config.get('static_ball_frames_threshold', 5) * 2
                if len(history['positions']) > max_history_frames:
                    history['positions'] = history['positions'][-max_history_frames:]
                    history['movements'] = history['movements'][-max_history_frames:]
            
            # 分析球的移动模式
            history = self.ball_movement_history[ball_key]
            
            # 检查最近几帧的移动情况
            recent_movements = history['movements'][-self.config.get('static_ball_frames_threshold', 5):]
            
            if len(recent_movements) >= self.config.get('static_ball_frames_threshold', 5):
                # 计算平均移动距离
                avg_movement = np.mean(recent_movements)
                max_movement = np.max(recent_movements)
                
                # 判断是否为运动球的多重条件
                movement_threshold = self.config['min_ball_movement']
                static_threshold = self.config['static_ball_movement_threshold_px']
                
                # 条件1: 平均移动距离超过阈值
                condition1 = avg_movement >= movement_threshold
                
                # 条件2: 至少有一次显著移动
                condition2 = max_movement >= movement_threshold * 1.8
                
                # 条件3: 最近的移动超过静止阈值
                condition3 = min_distance >= max(static_threshold, movement_threshold * 0.8)
                
                # 条件4: 检查移动的一致性（不是抖动）
                if len(recent_movements) >= 3:
                    movement_variance = np.var(recent_movements)
                    condition4 = movement_variance > (static_threshold ** 2)  # 有足够的变化
                else:
                    condition4 = True  # 数据不足时保守处理
                
                # 更严格的组合：平均移动达标且（存在显著移动或超过静止阈值）
                # 或者单次显著移动极大（容错极端快速位移）
                is_moving = (condition1 and (condition2 or condition3)) or condition2
                
                # 更新球的状态计数
                if is_moving:
                    history['moving_count'] += 1
                    history['static_count'] = 0  # 重置静止计数
                else:
                    history['static_count'] += 1
                    history['moving_count'] = max(0, history['moving_count'] - 1)  # 逐渐减少运动计数
                
                # 调试日志
                self._debug_log(
                    f"球 {ball_key}: 位置({ball[0]:.1f},{ball[1]:.1f}), "
                    f"最小距离={min_distance:.1f}px, 平均移动={avg_movement:.1f}px, "
                    f"最大移动={max_movement:.1f}px, 状态={'运动' if is_moving else '静止'}, "
                    f"运动计数={history['moving_count']}, 静止计数={history['static_count']}", 
                    'static'
                )

                # 需要达到一定的“运动计数”才纳入运动集合，抑制偶发抖动
                if is_moving and history['moving_count'] >= 2:
                    should_include = True
            else:
                # 数据不足：默认不判定为运动，除非与上一帧已跟踪球连续且接近
                is_moving = False
                if self.last_tracked_ball_coords is not None:
                    continuity_dist = np.linalg.norm(np.array(ball) - np.array(self.last_tracked_ball_coords))
                    if continuity_dist < self.config.get('max_ball_match_distance_px', 48):
                        is_moving = True
                        should_include = True
                        self._debug_log(
                            f"球 {ball_key}: 数据不足({len(recent_movements)}帧)，但与上次轨迹距离{continuity_dist:.1f}px，暂按连续运动处理",
                            'static'
                        )
                if not is_moving:
                    self._debug_log(
                        f"球 {ball_key}: 数据不足({len(recent_movements)}帧), 暂不判定运动（等待更多帧）",
                        'static'
                    )
            
            # 添加到过滤结果
            if should_include:
                filtered_balls.append(ball)
        
        # 清理过期的球历史记录
        current_ball_keys = set()
        for ball in detected_balls:
            current_ball_keys.add(f"{int(ball[0]/10)}_{int(ball[1]/10)}")
        
        # 移除长时间未见的球记录
        keys_to_remove = []
        for key, history in self.ball_movement_history.items():
            if key not in current_ball_keys:
                frames_since_last_seen = frame_num - history.get('frame_last_updated', frame_num)
                if frames_since_last_seen > self.config.get('max_lost_frames_for_track', 8):
                    keys_to_remove.append(key)
            else:
                history['frame_last_updated'] = frame_num
        
        for key in keys_to_remove:
            del self.ball_movement_history[key]
        
        # 更新前一帧球的位置
        self.prev_ball_positions = detected_balls
        
        self._debug_log(
            f"静止球过滤结果: {len(detected_balls)} → {len(filtered_balls)} 个球 "
            f"(过滤掉 {len(detected_balls) - len(filtered_balls)} 个静止球)", 
            'static'
        )
        
        return filtered_balls
    
    def advanced_ball_processing(self, all_detected_balls_current_frame, frame_num):
        """
        高级球处理：过滤静态球并追踪主要移动球，形成一致的轨迹
        返回：包含当前帧中单个活动球(x,y)坐标的列表，若无则为空列表
        更新：self.tracked_balls_history, self.static_ball_candidates, self.last_tracked_ball_coords
        """
        # 仅进行静止过滤：不开启轨迹追踪/绘制
        if self.config.get('static_filter_only', False):
            return self.filter_static_candidates(all_detected_balls_current_frame, frame_num)

        # 临时关闭追踪：直接返回输入的检测结果（不做静止过滤）
        if not self.config.get('ball_tracking_enabled', True):
            return all_detected_balls_current_frame
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
        """
        增强的球轨迹绘制 - 运动球用红色，轨迹用绿色
        """
        # 临时关闭追踪：不绘制轨迹
        if not self.config.get('ball_tracking_enabled', True):
            return frame
        # 绘制轨迹历史（连线）- 用绿色
        if len(self.tracked_balls_history) > 1:
            trajectory_points = []
            for entry in self.tracked_balls_history[-30:]:  # 显示最近30帧
                trajectory_points.append(tuple(map(int, entry['coords'])))
            
            # 绘制轨迹线段，带渐变效果 - 绿色轨迹
            for i in range(1, len(trajectory_points)):
                pt1 = trajectory_points[i-1]
                pt2 = trajectory_points[i]
                
                # 计算透明度和颜色强度 (越新的点越亮)
                alpha = max(0.3, i / len(trajectory_points))
                thickness = max(1, int(3 * alpha))
                
                # 使用亮绿色轨迹线
                color = (0, int(255 * alpha), 0)  # 绿色，透明度渐变
                cv2.line(frame, pt1, pt2, color, thickness)
            
            # 在轨迹点上绘制小圆点 - 黄绿色
            for i, point in enumerate(trajectory_points):
                alpha = max(0.4, i / len(trajectory_points))
                radius = max(2, int(4 * alpha))
                
                # 轨迹点用黄绿色
                color = (0, int(255 * alpha), int(200 * alpha))  # 黄绿色
                cv2.circle(frame, point, radius, color, -1)
        
        # 绘制当前运动球位置（如果有）- 用红色
        if self.last_tracked_ball_coords:
            current_pos = tuple(map(int, self.last_tracked_ball_coords))
            
            # 绘制大的红色圆圈表示当前运动球位置
            cv2.circle(frame, current_pos, 10, (0, 0, 255), 3)  # 红色空心圆
            cv2.circle(frame, current_pos, 5, (0, 0, 255), -1)  # 红色实心圆
            
            # 添加"MOVING"标签
            cv2.putText(frame, "MOVING", (current_pos[0]-30, current_pos[1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 在球周围绘制脉冲效果 - 橙红色
            if hasattr(self, 'frame_counter'):
                pulse_radius = 15 + int(4 * np.sin(self.frame_counter * 0.3))
                cv2.circle(frame, current_pos, pulse_radius, (0, 100, 255), 2)  # 橙红色脉冲圆
        
        # 绘制轨迹信息文本
        if len(self.tracked_balls_history) > 0:
            trajectory_length = len(self.tracked_balls_history)
            total_distance = 0
            
            # 计算轨迹总长度
            if len(self.tracked_balls_history) > 1:
                for i in range(1, len(self.tracked_balls_history)):
                    p1 = np.array(self.tracked_balls_history[i-1]['coords'])
                    p2 = np.array(self.tracked_balls_history[i]['coords'])
                    total_distance += np.linalg.norm(p2 - p1)
            
            # 在左下角显示轨迹信息
            info_y = frame.shape[0] - 100
            cv2.putText(frame, f"Moving Ball Trajectory: {trajectory_length} points", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Total Distance: {total_distance:.1f}px", 
                       (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示球的速度（如果有足够的数据）
            if len(self.tracked_balls_history) >= 2:
                recent_speed = self._calculate_ball_speed()
                cv2.putText(frame, f"Ball Speed: {recent_speed:.1f}px/frame", 
                           (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制监控区域边界
        if self.config['use_boundary'] and self.config.get('draw_boundary', False):
            x1 = self.config['boundary_x1']
            y1 = self.config['boundary_y1']
            x2 = self.config['boundary_x2']
            y2 = self.config['boundary_y2']
            
            # 绘制边界矩形 - 黄色
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 黄色边界
            
            # 在边界角落添加标签
            cv2.putText(frame, "TRACKING BOUNDARY", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 🚫 绘制屏蔽区域
        if self.config.get('use_mask_zones', False) and self.config.get('draw_mask_zones', False):
            for i, zone in enumerate(self.config.get('mask_zones', [])):
                x1 = zone['x']
                y1 = zone['y']
                x2 = x1 + zone['width']
                y2 = y1 + zone['height']
                
                zone_name = zone.get('name', f'MASK_{i+1}')
                
                # 绘制屏蔽区域矩形 - 红色
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色边框
                
                # 用半透明红色填充屏蔽区域  
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # 红色填充
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)  # 30%透明度混合
                
                # 添加屏蔽区域标签
                label_pos = (x1, y1 - 10 if y1 > 20 else y1 + zone['height'] + 20)
                cv2.putText(frame, f"MASKED: {zone_name}", label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 添加十字标记表示禁止
                center_x = x1 + zone['width'] // 2
                center_y = y1 + zone['height'] // 2
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 对角线1
                cv2.line(frame, (x1, y2), (x2, y1), (0, 0, 255), 2)  # 对角线2
        
        return frame
    
    def _calculate_ball_speed(self):
        """计算球的当前速度（基于最近几帧）"""
        if len(self.tracked_balls_history) < 2:
            return 0.0
        
        # 使用最近3-5帧计算平均速度
        recent_points = self.tracked_balls_history[-min(5, len(self.tracked_balls_history)):]
        
        if len(recent_points) < 2:
            return 0.0
        
        total_distance = 0
        frame_count = 0
        
        for i in range(1, len(recent_points)):
            p1 = np.array(recent_points[i-1]['coords'])
            p2 = np.array(recent_points[i]['coords'])
            distance = np.linalg.norm(p2 - p1)
            total_distance += distance
            frame_count += 1
        
        return total_distance / frame_count if frame_count > 0 else 0.0
    
    def draw_static_balls(self, frame):
        """
        在帧上绘制确认的静态球 - 用蓝色标识
        """
        # 临时关闭追踪：不绘制静态球
        if not self.config.get('ball_tracking_enabled', True):
            return frame
        static_count = 0
        for ball_id, static_info in self.static_ball_candidates.items():
            if static_info['frames_still'] >= self.config['static_ball_frames_threshold']:
                x, y = static_info['coords']
                
                # 用蓝色绘制静止球
                cv2.circle(frame, (int(x), int(y)), 8, (255, 100, 0), 3)  # 蓝色空心圆
                cv2.circle(frame, (int(x), int(y)), 4, (255, 100, 0), -1)  # 蓝色实心圆
                
                # 添加"STATIC"标签
                cv2.putText(frame, "STATIC", (int(x)-25, int(y)-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
                
                # 添加静止时间信息
                frames_still = static_info['frames_still']
                cv2.putText(frame, f"{frames_still}f", (int(x)-10, int(y)+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 0), 1)
                
                static_count += 1
        
        # 在右上角显示静止球统计
        if static_count > 0:
            cv2.putText(frame, f"Static Balls: {static_count}", 
                       (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        
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
    
    def _save_debug_frame(self, frame, detected_balls):
        """保存调试帧图像"""
        if not self.debug_config.get('save_debug_frames', False):
            return
        
        try:
            debug_frame = frame.copy()
            
            # 绘制检测到的球
            for i, ball in enumerate(detected_balls):
                cv2.circle(debug_frame, (int(ball[0]), int(ball[1])), 15, (0, 255, 0), 2)
                cv2.putText(debug_frame, f"Ball{i+1}", (int(ball[0])-20, int(ball[1])-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 绘制边界（如果启用）
            if self.config['use_boundary']:
                cv2.rectangle(debug_frame, 
                             (self.config['boundary_x1'], self.config['boundary_y1']),
                             (self.config['boundary_x2'], self.config['boundary_y2']), 
                             (0, 255, 255), 2)
                cv2.putText(debug_frame, "BOUNDARY", 
                           (self.config['boundary_x1'], self.config['boundary_y1']-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 添加信息文本
            info_text = f"Frame {self.frame_counter}: {len(detected_balls)} balls detected"
            cv2.putText(debug_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 保存图像
            debug_dir = "debug_frames"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            filename = f"{debug_dir}/ball_debug_frame_{self.frame_counter:04d}.jpg"
            cv2.imwrite(filename, debug_frame)
            self._debug_log(f"调试帧已保存: {filename}")
            
        except Exception as e:
            self._debug_log(f"保存调试帧失败: {e}")