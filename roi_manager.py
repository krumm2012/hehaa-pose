# roi_manager.py
import cv2
import numpy as np
import yaml
import time
from typing import List, Tuple, Optional, Dict, Any
import logging

class ROIManager:
    """
    兴趣区域管理器 - 支持4点描线的交互式选择和几何计算
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.roi_points = []  # 存储ROI的4个角点
        self.roi_polygon = None  # ROI多边形
        self.is_roi_set = False
        
        # 交互状态
        self.current_point_index = 0
        self.max_points = 4
        self.point_radius = 8
        self.line_thickness = 2
        
        # 颜色定义
        self.colors = {
            'roi_boundary': (0, 255, 255),    # 黄色 - ROI边界
            'roi_fill': (0, 255, 255),        # 黄色 - ROI填充（半透明）
            'current_point': (0, 0, 255),     # 红色 - 当前选择点
            'completed_point': (0, 255, 0),   # 绿色 - 已完成点
            'guide_line': (128, 128, 128),    # 灰色 - 引导线
            'text': (255, 255, 255),          # 白色 - 文字
            'inside_detection': (0, 255, 0),  # 绿色 - 区域内检测
            'outside_detection': (0, 0, 255)  # 红色 - 区域外检测
        }
        
        # 日志配置
        self.logger = logging.getLogger('ROIManager')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('🎯 [ROI管理器] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        # 坐标转换日志开关（默认关闭）
        self.log_coord_adjust = bool(
            self.config.get('roi_settings', {}).get('logging', {}).get('log_coordinate_adjustment', False)
        )
    
    def interactive_roi_selection(self, frame: np.ndarray, window_name: str = "ROI Selection") -> List[Tuple[int, int]]:
        """
        交互式ROI选择 - 用户通过鼠标点击选择4个点
        
        Args:
            frame: 输入视频帧
            window_name: 显示窗口名称
            
        Returns:
            List of (x, y) coordinates for the 4 ROI points
        """
        self.logger.info("开始交互式ROI选择")
        self.logger.info("请按顺序点击4个点来定义兴趣区域，按'r'重置，按'c'确认，按'q'退出")
        
        self.roi_points = []
        self.current_point_index = 0
        self.is_roi_set = False
        
        # 创建显示帧的副本
        display_frame = frame.copy()
        temp_frame = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal display_frame, temp_frame
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.current_point_index < self.max_points:
                    # 添加新点
                    self.roi_points.append((x, y))
                    self.current_point_index += 1
                    
                    self.logger.info(f"选择点 {self.current_point_index}: ({x}, {y})")
                    
                    # 更新显示
                    self._update_roi_display(temp_frame)
                    display_frame = temp_frame.copy()
                    cv2.imshow(window_name, display_frame)
                    
                    if self.current_point_index == self.max_points:
                        self.logger.info("已选择4个点，按'c'确认ROI或按'r'重新选择")
                        self.is_roi_set = True
                        self._create_roi_polygon()
            
            elif event == cv2.EVENT_MOUSEMOVE:
                # 显示鼠标位置和预览下一个点
                temp_display = temp_frame.copy()
                
                # 绘制鼠标位置
                cv2.circle(temp_display, (x, y), 3, self.colors['current_point'], -1)
                cv2.putText(temp_display, f"({x},{y})", (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
                
                # 如果已有点，显示到当前鼠标位置的预览线
                if self.roi_points and self.current_point_index < self.max_points:
                    cv2.line(temp_display, self.roi_points[-1], (x, y), 
                            self.colors['guide_line'], 1)
                
                display_frame = temp_display
                cv2.imshow(window_name, display_frame)
        
        # 设置鼠标回调
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # 初始显示
        self._update_roi_display(temp_frame)
        cv2.imshow(window_name, temp_frame)
        
        # 主循环
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.logger.info("用户取消ROI选择")
                break
            elif key == ord('r'):
                # 重置
                self.logger.info("重置ROI选择")
                self.roi_points = []
                self.current_point_index = 0
                self.is_roi_set = False
                temp_frame = frame.copy()
                self._update_roi_display(temp_frame)
                cv2.imshow(window_name, temp_frame)
            elif key == ord('c') and self.is_roi_set:
                # 确认ROI
                self.logger.info(f"确认ROI: {self.roi_points}")
                break
            elif key == 27:  # ESC键
                self.logger.info("用户按ESC退出")
                self.roi_points = []
                break
        
        cv2.destroyWindow(window_name)
        return self.roi_points
    
    def _update_roi_display(self, frame: np.ndarray):
        """更新ROI显示"""
        # 绘制已选择的点
        for i, point in enumerate(self.roi_points):
            color = self.colors['completed_point']
            cv2.circle(frame, point, self.point_radius, color, -1)
            cv2.putText(frame, f"P{i+1}", (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 绘制已完成的线段
        if len(self.roi_points) > 1:
            for i in range(len(self.roi_points) - 1):
                cv2.line(frame, self.roi_points[i], self.roi_points[i+1], 
                        self.colors['roi_boundary'], self.line_thickness)
        
        # 如果4个点都选择完毕，闭合多边形
        if len(self.roi_points) == self.max_points:
            cv2.line(frame, self.roi_points[-1], self.roi_points[0], 
                    self.colors['roi_boundary'], self.line_thickness)
            
            # 添加半透明填充
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(self.roi_points)], self.colors['roi_fill'])
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # 显示指导信息
        instructions = [
            f"点击选择第 {self.current_point_index + 1} 个点" if self.current_point_index < self.max_points else "按 'c' 确认 ROI",
            "按 'r' 重置",
            "按 'q' 退出"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
    
    def _create_roi_polygon(self):
        """创建ROI多边形"""
        if len(self.roi_points) == self.max_points:
            self.roi_polygon = np.array(self.roi_points, dtype=np.int32)
            self.logger.info("ROI多边形创建完成")
    
    def is_point_in_roi(self, point: Tuple[int, int]) -> bool:
        """
        检查点是否在ROI内
        
        Args:
            point: (x, y) 坐标
            
        Returns:
            True if point is inside ROI, False otherwise
        """
        if self.roi_polygon is None or not self.is_roi_set:
            return True  # 如果没有设置ROI，默认所有点都在内
        
        result = cv2.pointPolygonTest(self.roi_polygon, point, False)
        return result >= 0  # >= 0 表示在多边形内或边界上
    
    def filter_detections_by_roi(self, detections: List[Dict], detection_type: str = "general") -> List[Dict]:
        """
        根据ROI过滤检测结果
        
        Args:
            detections: 检测结果列表
            detection_type: 检测类型 ("pose", "ball", "racket", "general")
            
        Returns:
            过滤后的检测结果列表
        """
        if not self.is_roi_set or not detections:
            return detections
        
        filtered_detections = []
        
        for detection in detections:
            is_inside = False
            
            if detection_type == "pose":
                # 姿态检测：检查关键点是否在ROI内
                keypoints = detection if isinstance(detection, dict) else {}
                inside_count = 0
                total_count = 0
                
                for name, point in keypoints.items():
                    if point is not None:
                        total_count += 1
                        if self.is_point_in_roi(point):
                            inside_count += 1
                
                # 如果超过50%的关键点在ROI内，则认为整个姿态在ROI内
                is_inside = (inside_count / total_count) >= 0.5 if total_count > 0 else False
                
            elif detection_type == "ball":
                # 球检测：检查球心是否在ROI内
                if isinstance(detection, (tuple, list)) and len(detection) >= 2:
                    is_inside = self.is_point_in_roi((int(detection[0]), int(detection[1])))
                
            elif detection_type == "racket":
                # 球拍检测：检查边界框中心是否在ROI内
                if 'box' in detection:
                    box = detection['box']
                    center_x = int((box[0] + box[2]) / 2)
                    center_y = int((box[1] + box[3]) / 2)
                    is_inside = self.is_point_in_roi((center_x, center_y))
            
            else:
                # 通用检测：尝试多种方式判断
                if isinstance(detection, (tuple, list)) and len(detection) >= 2:
                    is_inside = self.is_point_in_roi((int(detection[0]), int(detection[1])))
                elif isinstance(detection, dict) and 'box' in detection:
                    box = detection['box']
                    center_x = int((box[0] + box[2]) / 2)
                    center_y = int((box[1] + box[3]) / 2)
                    is_inside = self.is_point_in_roi((center_x, center_y))
            
            if is_inside:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def draw_roi(self, frame: np.ndarray, show_fill: bool = True) -> np.ndarray:
        """
        在帧上绘制ROI
        
        Args:
            frame: 输入帧
            show_fill: 是否显示填充
            
        Returns:
            绘制了ROI的帧
        """
        if not self.is_roi_set or self.roi_polygon is None:
            return frame
        
        result_frame = frame.copy()
        
        # 绘制ROI边界
        cv2.polylines(result_frame, [self.roi_polygon], True, 
                     self.colors['roi_boundary'], self.line_thickness)
        
        # 绘制半透明填充
        if show_fill:
            overlay = result_frame.copy()
            cv2.fillPoly(overlay, [self.roi_polygon], self.colors['roi_fill'])
            cv2.addWeighted(overlay, 0.15, result_frame, 0.85, 0, result_frame)
        
        # 绘制角点
        for i, point in enumerate(self.roi_points):
            cv2.circle(result_frame, point, self.point_radius, 
                      self.colors['completed_point'], -1)
            cv2.putText(result_frame, f"P{i+1}", (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['completed_point'], 2)
        
        # 添加ROI标签
        if self.roi_points:
            label_pos = (self.roi_points[0][0], self.roi_points[0][1] - 20)
            cv2.putText(result_frame, "ROI ACTIVE", label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['roi_boundary'], 2)
        
        return result_frame
    
    def highlight_roi_detections(self, frame: np.ndarray, detections: List, detection_type: str = "general") -> np.ndarray:
        """
        高亮显示ROI内的检测结果
        
        Args:
            frame: 输入帧
            detections: 检测结果
            detection_type: 检测类型
            
        Returns:
            高亮显示检测结果的帧
        """
        if not detections:
            return frame
        
        result_frame = frame.copy()
        
        for detection in detections:
            if detection_type == "ball" and isinstance(detection, (tuple, list)):
                x, y = int(detection[0]), int(detection[1])
                is_inside = self.is_point_in_roi((x, y))
                color = self.colors['inside_detection'] if is_inside else self.colors['outside_detection']
                
                # 绘制球
                cv2.circle(result_frame, (x, y), 8, color, 3)
                cv2.circle(result_frame, (x, y), 4, color, -1)
                
                # 添加状态标签
                status = "IN ROI" if is_inside else "OUT ROI"
                cv2.putText(result_frame, status, (x+15, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            elif detection_type == "racket" and isinstance(detection, dict) and 'box' in detection:
                box = detection['box']
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                is_inside = self.is_point_in_roi((center_x, center_y))
                color = self.colors['inside_detection'] if is_inside else self.colors['outside_detection']
                
                # 绘制球拍框
                cv2.rectangle(result_frame, (box[0], box[1]), (box[2], box[3]), color, 3)
                
                # 添加状态标签
                status = "IN ROI" if is_inside else "OUT ROI"
                cv2.putText(result_frame, f"RACKET {status}", (box[0], box[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_frame
    
    def save_roi_config(self, config_path: str):
        """保存ROI配置到文件"""
        if not self.is_roi_set:
            self.logger.warning("ROI未设置，无法保存配置")
            return
        
        # 确保坐标格式为标准列表而不是元组，避免YAML序列化问题
        roi_points_list = [[int(point[0]), int(point[1])] for point in self.roi_points]
        
        roi_config = {
            'roi_enabled': True,
            'roi_points': roi_points_list,
            'roi_description': f"4-point ROI: {roi_points_list}",
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'created_by': "Tennis Analyzer System",
            'version': "1.0"
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(roi_config, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"ROI配置已保存到: {config_path}")
        except Exception as e:
            self.logger.error(f"保存ROI配置失败: {e}")
    
    def load_roi_config(self, config_path: str) -> bool:
        """从文件加载ROI配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                roi_config = yaml.safe_load(f)
            
            if roi_config.get('roi_enabled', False) and 'roi_points' in roi_config:
                roi_points = roi_config['roi_points']
                
                # 确保有4个点
                if len(roi_points) == self.max_points:
                    # 将坐标转换为元组格式 (x, y)
                    self.roi_points = []
                    for point in roi_points:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            self.roi_points.append((int(point[0]), int(point[1])))
                        else:
                            self.logger.error(f"无效的ROI点格式: {point}")
                            return False
                    
                    self._create_roi_polygon()
                    self.is_roi_set = True
                    self.logger.info(f"ROI配置已从 {config_path} 加载: {self.roi_points}")
                    return True
                else:
                    self.logger.warning(f"ROI点数不正确: 需要{self.max_points}个点，实际{len(roi_points)}个")
            else:
                self.logger.info(f"ROI配置文件中ROI未启用或缺少roi_points")
            
            return False
        except Exception as e:
            self.logger.error(f"加载ROI配置失败: {e}")
            return False
    
    def get_roi_stats(self) -> Dict[str, Any]:
        """获取ROI统计信息"""
        if not self.is_roi_set:
            return {"roi_enabled": False}
        
        # 计算ROI面积
        area = cv2.contourArea(self.roi_polygon) if self.roi_polygon is not None else 0
        
        # 计算ROI边界框
        x_coords = [p[0] for p in self.roi_points]
        y_coords = [p[1] for p in self.roi_points]
        bbox = {
            'x_min': min(x_coords),
            'y_min': min(y_coords),
            'x_max': max(x_coords),
            'y_max': max(y_coords)
        }
        
        return {
            'roi_enabled': True,
            'roi_points': self.roi_points,
            'roi_area': area,
            'roi_bbox': bbox,
            'roi_perimeter': cv2.arcLength(self.roi_polygon, True) if self.roi_polygon is not None else 0
        }
    
    def get_roi_mask(self, frame_shape):
        """获取ROI掩码
        
        Args:
            frame_shape: 帧的形状 (height, width)
            
        Returns:
            numpy.ndarray: ROI掩码，ROI内为255，ROI外为0
        """
        if not self.is_roi_set:
            return None
        
        height, width = frame_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if self.roi_polygon is not None:
            cv2.fillPoly(mask, [self.roi_polygon], 255)
        
        return mask
    
    def get_roi_bounding_box(self):
        """获取ROI的外接矩形
        
        Returns:
            tuple: (x1, y1, x2, y2) 外接矩形坐标，如果ROI未设置则返回None
        """
        if not self.is_roi_set or len(self.roi_points) == 0:
            return None
        
        x_coords = [p[0] for p in self.roi_points]
        y_coords = [p[1] for p in self.roi_points]
        
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        
        return (x1, y1, x2, y2)
    
    def adjust_detection_coordinates(self, detections, roi_offset, detection_type):
        """将ROI裁剪区域的检测结果坐标转换回原图坐标系
        
        Args:
            detections: 检测结果列表
            roi_offset: ROI区域在原图中的偏移量 (x_offset, y_offset)
            detection_type: 检测类型 ("pose", "ball", "racket")
            
        Returns:
            list: 坐标调整后的检测结果
        """
        if not detections or roi_offset == (0, 0):
            return detections
        
        x_offset, y_offset = roi_offset
        adjusted_detections = []
        
        if detection_type == "pose":
            # 姿态检测结果：每个人是一个关键点列表
            for person_keypoints in detections:
                adjusted_person = []
                for keypoint in person_keypoints:
                    if len(keypoint) >= 2:
                        # 调整x, y坐标，保持置信度不变
                        adjusted_keypoint = [
                            keypoint[0] + x_offset,  # x坐标
                            keypoint[1] + y_offset,  # y坐标
                        ]
                        if len(keypoint) > 2:
                            adjusted_keypoint.extend(keypoint[2:])  # 保持置信度等其他属性
                        adjusted_person.append(adjusted_keypoint)
                    else:
                        adjusted_person.append(keypoint)
                adjusted_detections.append(adjusted_person)
                
        elif detection_type == "ball":
            # 球检测结果：每个球是 [x, y, ...] 格式
            for ball in detections:
                if len(ball) >= 2:
                    adjusted_ball = [
                        ball[0] + x_offset,  # x坐标
                        ball[1] + y_offset,  # y坐标
                    ]
                    if len(ball) > 2:
                        adjusted_ball.extend(ball[2:])  # 保持其他属性
                    adjusted_detections.append(adjusted_ball)
                else:
                    adjusted_detections.append(ball)
                    
        elif detection_type == "racket":
            # 球拍检测结果：每个球拍可能是字典格式 {'box': [x1, y1, x2, y2], ...}
            for racket in detections:
                if isinstance(racket, dict) and 'box' in racket:
                    box = racket['box']
                    if len(box) >= 4:
                        adjusted_racket = racket.copy()
                        adjusted_racket['box'] = [
                            box[0] + x_offset,  # x1
                            box[1] + y_offset,  # y1
                            box[2] + x_offset,  # x2
                            box[3] + y_offset,  # y2
                        ]
                        adjusted_detections.append(adjusted_racket)
                    else:
                        adjusted_detections.append(racket)
                elif isinstance(racket, (list, tuple)) and len(racket) >= 4:
                    # 如果是直接的坐标列表格式
                    adjusted_racket = [
                        racket[0] + x_offset,  # x1
                        racket[1] + y_offset,  # y1
                        racket[2] + x_offset,  # x2
                        racket[3] + y_offset,  # y2
                    ]
                    if len(racket) > 4:
                        adjusted_racket.extend(racket[4:])
                    adjusted_detections.append(adjusted_racket)
                else:
                    adjusted_detections.append(racket)
        else:
            # 未知类型，返回原始检测结果
            return detections
        
        if self.log_coord_adjust:
            print(f"🔄 [坐标转换] {detection_type}: {len(detections)} → {len(adjusted_detections)} 个检测结果，偏移量: {roi_offset}")
        return adjusted_detections
