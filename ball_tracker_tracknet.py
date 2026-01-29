#!/usr/bin/env python3
# ball_tracker_tracknet.py
# TrackNet 球追踪器 - 使用原始 Keras 模型

import sys
import os
import cv2
import numpy as np
from typing import Optional, Tuple, List
import queue

# 添加 tennis-tracking 到路径
sys.path.insert(0, 'tennis-tracking')
sys.path.insert(0, 'tennis-tracking/Models')

class BallTrackerTrackNet:
    """
    使用 TrackNet 深度学习模型进行球追踪
    """
    
    def __init__(self, config: dict = None):
        """
        初始化 TrackNet 追踪器
        
        Args:
            config: 配置字典
        """
        print("🎾 初始化 TrackNet 球追踪器...")
        
        self.config = config or {}
        self.input_size = (640, 360)  # TrackNet 输入尺寸
        self.n_classes = 256
        
        # 加载模型
        self._load_model()
        
        # 轨迹队列（保存最近8帧）
        self.trajectory_queue = queue.deque()
        for i in range(8):
            self.trajectory_queue.appendleft(None)
        
        # 性能统计
        self.frame_count = 0
        self.detection_count = 0
        
        print("✅ TrackNet 追踪器初始化成功")
    
    def _load_model(self):
        """
        加载 TrackNet 模型（channels_last 版本）
        """
        print("   加载模型架构...")
        
        from tracknet_channels_last import trackNet_channels_last
        
        # 创建模型
        self.model = trackNet_channels_last(
            n_classes=self.n_classes,
            input_height=self.input_size[1],
            input_width=self.input_size[0]
        )
        
        # 加载转换后的权重
        weights_path = 'models/tracknet_channels_last.h5'
        print(f"   加载权重: {weights_path}")
        
        try:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"权重文件不存在: {weights_path}")
            
            self.model.load_weights(weights_path)
            
            # 编译模型
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
            
            print(f"   模型参数: {self.model.count_params():,}")
            print("   ✅ 模型加载成功")
            
        except Exception as e:
            print(f"   ❌ 模型加载失败: {e}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理输入帧
        
        Args:
            frame: BGR 格式的输入帧
            
        Returns:
            预处理后的帧
        """
        # 保存原始尺寸
        self.original_size = (frame.shape[1], frame.shape[0])
        
        # 调整大小
        resized = cv2.resize(frame, self.input_size)
        
        # 转换为 float32
        img = resized.astype(np.float32)
        
        # TrackNet 原始使用 channels_first (NCHW)
        # 但 TensorFlow CPU 只支持 channels_last (NHWC)
        # 所以我们需要转置: (height, width, channels) -> (channels, height, width) -> (height, width, channels)
        # 实际上就保持原样即可，因为 cv2.resize 已经是 (H, W, C) 格式
        
        return img
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        检测球的位置
        
        Args:
            frame: 输入帧
            
        Returns:
            (x, y) 球的位置，如果未检测到返回 None
        """
        self.frame_count += 1
        
        # 预处理
        preprocessed = self.preprocess_frame(frame)
        
        try:
            # 推理
            prediction = self.model.predict(np.array([preprocessed]), verbose=0)[0]
            
            # 重塑为热力图: (height*width, n_classes) -> (height, width, n_classes)
            heatmap = prediction.reshape(
                (self.input_size[1], self.input_size[0], self.n_classes)
            ).argmax(axis=2)
            
            # 转换为 uint8
            heatmap = heatmap.astype(np.uint8)
            
            # 调整回原始尺寸
            heatmap_resized = cv2.resize(
                heatmap,
                self.original_size
            )
            
            # 二值化
            _, binary = cv2.threshold(heatmap_resized, 127, 255, cv2.THRESH_BINARY)
            
            # 查找圆形
            circles = cv2.HoughCircles(
                binary,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=1,
                param1=50,
                param2=2,
                minRadius=2,
                maxRadius=7
            )
            
            if circles is not None and len(circles) > 0:
                # 取第一个检测到的圆
                x, y = int(circles[0][0][0]), int(circles[0][0][1])
                
                # 更新轨迹队列
                self.trajectory_queue.appendleft((x, y))
                self.trajectory_queue.pop()
                
                self.detection_count += 1
                
                return (x, y)
            else:
                # 未检测到
                self.trajectory_queue.appendleft(None)
                self.trajectory_queue.pop()
                
                return None
                
        except Exception as e:
            print(f"❌ 检测失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def draw_ball(self, frame: np.ndarray, position: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        在帧上绘制球的位置和轨迹
        
        Args:
            frame: 输入帧
            position: 球的位置（如果为 None，使用最新检测结果）
            
        Returns:
            绘制后的帧
        """
        # 绘制轨迹（最近8帧）
        for i in range(8):
            pos = self.trajectory_queue[i]
            if pos is not None:
                # 轨迹点颜色渐变（黄色）
                cv2.circle(frame, pos, 2, (0, 255, 255), -1)
        
        # 绘制当前位置（更大的圆）
        if position is not None:
            cv2.circle(frame, position, 5, (0, 255, 0), -1)
        
        return frame
    
    def get_stats(self) -> dict:
        """
        获取性能统计
        
        Returns:
            统计信息字典
        """
        detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0
        
        return {
            'total_frames': self.frame_count,
            'detections': self.detection_count,
            'detection_rate': detection_rate
        }


def test_tracknet():
    """测试 TrackNet 追踪器"""
    print("=" * 60)
    print("🧪 TrackNet 追踪器测试")
    print("=" * 60)
    
    # 创建追踪器
    tracker = BallTrackerTrackNet()
    
    # 测试视频
    test_video = "data/16.10.mp4"
    
    if not os.path.exists(test_video):
        print(f"\n❌ 测试视频不存在: {test_video}")
        print("💡 请提供测试视频路径")
        return
    
    print(f"\n📹 处理视频: {test_video}")
    
    # 打开视频
    cap = cv2.VideoCapture(test_video)
    
    if not cap.isOpened():
        print("❌ 无法打开视频")
        return
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   分辨率: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   总帧数: {total_frames}")
    
    # 创建输出视频
    output_path = "output/tracknet_test.mp4"
    os.makedirs("output", exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\n⏳ 开始处理...")
    
    import time
    start_time = time.time()
    frame_count = 0
    
    # 只处理前100帧进行测试
    max_frames = min(100, total_frames)
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 检测球
        position = tracker.detect_ball(frame)
        
        # 绘制结果
        frame = tracker.draw_ball(frame, position)
        
        # 添加信息
        cv2.putText(
            frame,
            f"Frame: {frame_count}/{max_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        if position:
            cv2.putText(
                frame,
                f"Ball: ({position[0]}, {position[1]})",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )
        
        # 写入输出
        out.write(frame)
        
        # 显示进度
        if frame_count % 10 == 0:
            progress = frame_count / max_frames * 100
            print(f"   进度: {progress:.1f}% ({frame_count}/{max_frames})")
    
    # 计算性能
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time
    
    # 释放资源
    cap.release()
    out.release()
    
    # 获取统计
    stats = tracker.get_stats()
    
    # 输出结果
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    print(f"\n📊 性能统计:")
    print(f"   处理帧数: {frame_count}")
    print(f"   总时间: {elapsed_time:.2f} 秒")
    print(f"   平均 FPS: {avg_fps:.2f}")
    print(f"   检测成功: {stats['detections']}/{stats['total_frames']}")
    print(f"   检测率: {stats['detection_rate']:.1f}%")
    print(f"\n📹 输出视频: {output_path}")


if __name__ == "__main__":
    test_tracknet()
