#!/usr/bin/env python3
"""
ball_tracker_enhanced.py
增强版球追踪器 - 多颜色空间融合 + 卡尔曼滤波

改进点:
1. 多颜色空间融合 (HSV + LAB + YCrCb)
2. 卡尔曼滤波器 (运动预测)
3. 自适应阈值 (光照适应)
4. 轨迹平滑 (消除抖动)
"""

import cv2
import numpy as np
import time
from collections import deque


class KalmanBallTracker:
    """卡尔曼滤波器用于球的运动预测"""
    
    def __init__(self):
        # 4个状态: [x, y, vx, vy]
        # 2个测量: [x, y]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)
        
        # 过程噪声
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # 测量噪声
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10
        
        # 初始化标志
        self.initialized = False
        
    def predict(self):
        """预测下一帧的位置"""
        if not self.initialized:
            return None
        
        prediction = self.kf.predict()
        return int(prediction[0]), int(prediction[1])
    
    def update(self, measurement):
        """更新卡尔曼滤波器"""
        if not self.initialized:
            # 首次初始化
            self.kf.statePre = np.array([
                [measurement[0]],
                [measurement[1]],
                [0],
                [0]
            ], dtype=np.float32)
            self.kf.statePost = self.kf.statePre.copy()
            self.initialized = True
        else:
            # 更新
            self.kf.correct(np.array([[measurement[0]], [measurement[1]]], dtype=np.float32))
        
        return int(self.kf.statePost[0]), int(self.kf.statePost[1])


class BallTrackerEnhanced:
    """增强版球追踪器"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # 卡尔曼滤波器
        self.kalman = KalmanBallTracker()
        
        # 轨迹历史
        self.trajectory = deque(maxlen=30)
        
        # 性能统计
        self.frame_count = 0
        self.detection_count = 0
        self.total_time = 0
        
        # 配置参数
        self.enable_kalman = self.config.get('enable_kalman', True)
        self.enable_multi_colorspace = self.config.get('enable_multi_colorspace', True)
        self.enable_adaptive_threshold = self.config.get('enable_adaptive_threshold', True)
        
        print("🎾 增强版球追踪器初始化完成")
        print(f"   卡尔曼滤波: {'✅' if self.enable_kalman else '❌'}")
        print(f"   多颜色空间: {'✅' if self.enable_multi_colorspace else '❌'}")
        print(f"   自适应阈值: {'✅' if self.enable_adaptive_threshold else '❌'}")
    
    def _detect_hsv(self, frame, adaptive=False):
        """HSV 颜色空间检测"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if adaptive and self.enable_adaptive_threshold:
            # 自适应阈值
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if brightness < 100:  # 暗
                lower = np.array([20, 80, 80], dtype=np.uint8)
                upper = np.array([45, 255, 255], dtype=np.uint8)
            elif brightness > 180:  # 亮
                lower = np.array([25, 100, 100], dtype=np.uint8)
                upper = np.array([40, 255, 255], dtype=np.uint8)
            else:  # 正常
                lower = np.array([22, 90, 90], dtype=np.uint8)
                upper = np.array([42, 255, 255], dtype=np.uint8)
        else:
            # 固定阈值
            lower = np.array([20, 50, 50], dtype=np.uint8)
            upper = np.array([70, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower, upper)
        return mask
    
    def _detect_lab(self, frame):
        """LAB 颜色空间检测"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # 网球在 LAB 空间的特征（更严格的范围）
        # L: 亮度 (中高，120-255)
        # A: 绿-红轴 (偏绿，100-135，网球的黄绿色)
        # B: 蓝-黄轴 (偏黄，140-200)
        lower = np.array([120, 100, 140], dtype=np.uint8)
        upper = np.array([255, 135, 200], dtype=np.uint8)
        
        mask = cv2.inRange(lab, lower, upper)
        return mask
    
    def _detect_ycrcb(self, frame):
        """YCrCb 颜色空间检测"""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # 网球在 YCrCb 空间的特征（更严格的范围）
        # Y: 亮度 (中高，120-255)
        # Cr: 红色分量 (低，0-110)
        # Cb: 蓝色分量 (低，0-110)
        lower = np.array([120, 0, 0], dtype=np.uint8)
        upper = np.array([255, 110, 110], dtype=np.uint8)
        
        mask = cv2.inRange(ycrcb, lower, upper)
        return mask
    
    def _fuse_masks(self, hsv_mask, lab_mask, ycrcb_mask):
        """融合多个颜色空间的掩码"""
        # 使用投票机制：至少2个颜色空间检测到才认为是球
        # 这样可以减少误检，提高精确度
        
        vote = (hsv_mask > 0).astype(np.uint8) + \
               (lab_mask > 0).astype(np.uint8) + \
               (ycrcb_mask > 0).astype(np.uint8)
        
        # 至少2个空间检测到
        combined = (vote >= 2).astype(np.uint8) * 255
        
        return combined
    
    def _morphology_operations(self, mask):
        """形态学操作 - 去噪和增强"""
        # 去除小噪点
        kernel_small = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 填充空洞
        kernel_medium = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
        
        # 膨胀以确保完整性
        mask = cv2.dilate(mask, kernel_medium, iterations=1)
        
        return mask
    
    def detect_ball(self, frame):
        """
        检测球的位置
        
        Returns:
            (x, y) 或 None
        """
        start_time = time.time()
        self.frame_count += 1
        
        # 缩小图像以加速处理
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # 预处理 - 降噪
        small_frame = cv2.GaussianBlur(small_frame, (3, 3), 0)
        
        # 多颜色空间检测
        if self.enable_multi_colorspace:
            hsv_mask = self._detect_hsv(small_frame, adaptive=True)
            lab_mask = self._detect_lab(small_frame)
            ycrcb_mask = self._detect_ycrcb(small_frame)
            mask = self._fuse_masks(hsv_mask, lab_mask, ycrcb_mask)
        else:
            mask = self._detect_hsv(small_frame, adaptive=True)
        
        # 形态学操作
        mask = self._morphology_operations(mask)
        
        # 卡尔曼预测
        predicted_pos = None
        if self.enable_kalman:
            predicted_pos = self.kalman.predict()
        
        # 圆检测
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=10,
            minRadius=3,
            maxRadius=30
        )
        
        detected_pos = None
        
        if circles is not None and len(circles) > 0:
            circles = circles[0]
            
            if predicted_pos is not None and self.enable_kalman:
                # 选择最接近预测位置的圆
                min_dist = float('inf')
                best_circle = None
                
                for circle in circles:
                    cx, cy = int(circle[0]), int(circle[1])
                    dist = np.sqrt((cx - predicted_pos[0]/scale)**2 + 
                                 (cy - predicted_pos[1]/scale)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_circle = circle
                
                if best_circle is not None:
                    x, y = int(best_circle[0] / scale), int(best_circle[1] / scale)
                    detected_pos = (x, y)
            else:
                # 选择第一个圆
                x, y = int(circles[0][0] / scale), int(circles[0][1] / scale)
                detected_pos = (x, y)
        
        # 更新卡尔曼滤波器
        if detected_pos is not None:
            if self.enable_kalman:
                detected_pos = self.kalman.update(detected_pos)
            
            self.trajectory.append(detected_pos)
            self.detection_count += 1
        else:
            # 使用预测位置
            if predicted_pos is not None and self.enable_kalman:
                self.trajectory.append(predicted_pos)
            else:
                self.trajectory.append(None)
        
        # 统计
        elapsed = time.time() - start_time
        self.total_time += elapsed
        
        return detected_pos
    
    def draw_ball(self, frame, position):
        """绘制球"""
        if position is not None:
            x, y = position
            cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    def draw_trajectory(self, frame):
        """绘制轨迹"""
        # 平滑轨迹
        valid_points = [p for p in self.trajectory if p is not None]
        
        if len(valid_points) >= 2:
            # 绘制轨迹线
            for i in range(len(valid_points) - 1):
                if valid_points[i] is not None and valid_points[i+1] is not None:
                    cv2.line(frame, valid_points[i], valid_points[i+1], 
                           (0, 255, 0), 2)
    
    def get_stats(self):
        """获取性能统计"""
        if self.frame_count == 0:
            return {
                'frames': 0,
                'detections': 0,
                'detection_rate': 0,
                'avg_time_ms': 0,
                'fps': 0
            }
        
        avg_time = self.total_time / self.frame_count
        
        return {
            'frames': self.frame_count,
            'detections': self.detection_count,
            'detection_rate': self.detection_count / self.frame_count * 100,
            'avg_time_ms': avg_time * 1000,
            'fps': 1.0 / avg_time if avg_time > 0 else 0
        }


def main():
    """测试增强版球追踪器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强版球追踪器测试')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, default='output/enhanced_test.mp4', 
                       help='输出视频路径')
    parser.add_argument('--max-frames', type=int, default=None, 
                       help='最大处理帧数')
    parser.add_argument('--no-kalman', action='store_true', 
                       help='禁用卡尔曼滤波')
    parser.add_argument('--no-multi-colorspace', action='store_true',
                       help='禁用多颜色空间')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'enable_kalman': not args.no_kalman,
        'enable_multi_colorspace': not args.no_multi_colorspace,
        'enable_adaptive_threshold': True
    }
    
    # 创建追踪器
    tracker = BallTrackerEnhanced(config)
    
    # 打开视频
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {args.video}")
        return
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 处理视频: {args.video}")
    print(f"   分辨率: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   总帧数: {total_frames}")
    
    # 创建输出视频
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # 处理视频
    frame_num = 0
    max_frames = args.max_frames or total_frames
    
    print(f"\n⏳ 开始处理...")
    
    while cap.isOpened() and frame_num < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # 检测球
        position = tracker.detect_ball(frame)
        
        # 绘制
        tracker.draw_ball(frame, position)
        tracker.draw_trajectory(frame)
        
        # 写入输出
        out.write(frame)
        
        # 进度
        if frame_num % 10 == 0:
            print(f"   进度: {frame_num/max_frames*100:.1f}% ({frame_num}/{max_frames})")
    
    # 清理
    cap.release()
    out.release()
    
    # 统计
    stats = tracker.get_stats()
    
    print(f"\n{'='*60}")
    print("✅ 测试完成！")
    print(f"{'='*60}")
    print(f"\n📊 性能统计:")
    print(f"   处理帧数: {stats['frames']}")
    print(f"   检测成功: {stats['detections']}/{stats['frames']}")
    print(f"   检测率: {stats['detection_rate']:.1f}%")
    print(f"   平均处理时间: {stats['avg_time_ms']:.2f} ms/帧")
    print(f"   平均 FPS: {stats['fps']:.2f}")
    print(f"\n📹 输出视频: {args.output}")


if __name__ == "__main__":
    main()
