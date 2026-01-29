#!/usr/bin/env python3
"""
视频 I/O Pipeline
并行化视频读取和写入，提升性能
"""

import cv2
import threading
import queue
import time
from typing import Tuple, Optional, Callable


class VideoIOPipeline:
    """
    视频 I/O Pipeline
    - 读取线程: 预读取帧到缓冲区
    - 主线程: 检测和处理
    - 写入线程: 异步写入视频
    """
    
    def __init__(self, input_path: str, output_path: str, fps: float, 
                 frame_size: Tuple[int, int], buffer_size: int = 3):
        """
        初始化 Pipeline
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            fps: 帧率
            frame_size: 帧尺寸 (width, height)
            buffer_size: 缓冲区大小（建议 2-4）
        """
        self.input_path = input_path
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.buffer_size = buffer_size
        
        # 队列
        self.read_queue = queue.Queue(maxsize=buffer_size)
        self.write_queue = queue.Queue(maxsize=buffer_size)
        
        # 线程
        self.read_thread = None
        self.write_thread = None
        
        # 统计
        self.frames_read = 0
        self.frames_written = 0
        self.start_time = None
        
        # VideoCapture 和 VideoWriter
        self.cap = None
        self.out = None
    
    def _read_thread_func(self):
        """读取线程函数"""
        self.cap = cv2.VideoCapture(self.input_path)
        
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # 读取完成，发送结束信号
                self.read_queue.put(None)
                break
            
            # 放入队列（会阻塞直到有空间）
            self.read_queue.put((frame_num, frame))
            self.frames_read += 1
            frame_num += 1
        
        self.cap.release()
    
    def _write_thread_func(self):
        """写入线程函数"""
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
        
        while True:
            item = self.write_queue.get()
            
            if item is None:  # 结束信号
                break
            
            frame_num, display_frame = item
            self.out.write(display_frame)
            self.frames_written += 1
        
        self.out.release()
    
    def start(self):
        """启动 Pipeline"""
        print("🚀 启动 I/O Pipeline...")
        
        self.start_time = time.time()
        
        # 启动读写线程
        self.read_thread = threading.Thread(target=self._read_thread_func, daemon=True)
        self.write_thread = threading.Thread(target=self._write_thread_func, daemon=True)
        
        self.read_thread.start()
        self.write_thread.start()
        
        print(f"   读取缓冲: {self.buffer_size} 帧")
        print(f"   写入缓冲: {self.buffer_size} 帧")
    
    def get_frame(self) -> Optional[Tuple[int, any]]:
        """
        获取下一帧
        
        Returns:
            (frame_num, frame) 或 None（结束）
        """
        item = self.read_queue.get()
        return item
    
    def put_frame(self, frame_num: int, display_frame):
        """
        提交处理后的帧到写入队列
        
        Args:
            frame_num: 帧号
            display_frame: 处理后的帧
        """
        self.write_queue.put((frame_num, display_frame))
    
    def finish(self):
        """完成处理，等待所有帧写入"""
        # 发送写入结束信号
        self.write_queue.put(None)
        
        # 等待写入线程完成
        if self.write_thread:
            self.write_thread.join()
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n✅ Pipeline 完成:")
        print(f"   读取帧数: {self.frames_read}")
        print(f"   写入帧数: {self.frames_written}")
        if total_time > 0:
            print(f"   总时间: {total_time:.2f}秒")
            print(f"   平均 FPS: {self.frames_written / total_time:.2f}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.finish()


# 使用示例
def example_with_pipeline():
    """使用 Pipeline 的示例"""
    
    # 模拟检测函数
    def process_frame(frame_num, frame):
        # 模拟检测耗时
        time.sleep(0.015)
        
        # 简单处理
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Frame: {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return display_frame
    
    # 使用 Pipeline
    with VideoIOPipeline(
        input_path="data/16.10.mp4",
        output_path="/tmp/pipeline_output.mp4",
        fps=25,
        frame_size=(2560, 1440),
        buffer_size=3
    ) as pipeline:
        
        frame_count = 0
        while True:
            # 获取帧
            item = pipeline.get_frame()
            if item is None:
                break
            
            frame_num, frame = item
            
            # 处理帧
            display_frame = process_frame(frame_num, frame)
            
            # 提交结果
            pipeline.put_frame(frame_num, display_frame)
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"   已处理 {frame_count} 帧")


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 视频 I/O Pipeline 测试")
    print("=" * 60)
    
    example_with_pipeline()
