#!/usr/bin/env python3
"""
async_detector.py
异步检测器 - 并行执行姿态、球、球拍检测以提升性能
"""

import concurrent.futures
from typing import Dict, Any, Optional, Tuple
import time
import numpy as np


class AsyncDetector:
    """异步检测器，并行执行多个检测任务"""
    
    def __init__(self, max_workers: int = 3):
        """
        初始化异步检测器
        
        Args:
            max_workers: 最大工作线程数（默认3：姿态、球、球拍）
        """
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="AsyncDetector"
        )
        self.max_workers = max_workers
        
        # 性能统计
        self.stats = {
            'total_detections': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
    
    def detect_async(
        self,
        frame: np.ndarray,
        roi_frame: np.ndarray,
        roi_offset: Tuple[int, int],
        pose_estimator,
        ball_tracker,
        racket_detector,
        frame_count: int = 0
    ) -> Dict[str, Any]:
        """
        并行执行所有检测任务
        
        Args:
            frame: 完整帧（用于姿态检测）
            roi_frame: ROI 区域帧（用于球和球拍检测）
            roi_offset: ROI 偏移量 (x_offset, y_offset)
            pose_estimator: 姿态估计器
            ball_tracker: 球追踪器
            racket_detector: 球拍检测器
            frame_count: 当前帧号
            
        Returns:
            {
                'keypoints': 姿态关键点,
                'balls': 球检测结果,
                'rackets': 球拍检测结果,
                'timing': 各模块耗时
            }
        """
        start_time = time.time()
        timing = {}
        
        try:
            # 提交并行任务
            pose_future = self.executor.submit(
                self._detect_pose,
                pose_estimator,
                frame,
                frame_count
            )
            
            ball_future = self.executor.submit(
                self._detect_ball,
                ball_tracker,
                roi_frame,
                roi_offset,
                frame_count
            )
            
            racket_future = self.executor.submit(
                self._detect_racket,
                racket_detector,
                roi_frame,
                roi_offset,
                frame_count
            )
            
            # 等待所有任务完成并获取结果
            keypoints, timing['pose'] = pose_future.result()
            balls, timing['ball'] = ball_future.result()
            rackets, timing['racket'] = racket_future.result()
            
            # 总耗时
            total_time = time.time() - start_time
            timing['total'] = total_time
            timing['parallel_efficiency'] = (
                sum([timing['pose'], timing['ball'], timing['racket']]) / total_time
            )
            
            # 更新统计
            self.stats['total_detections'] += 1
            self.stats['total_time'] += total_time
            self.stats['avg_time'] = self.stats['total_time'] / self.stats['total_detections']
            
            return {
                'keypoints': keypoints,
                'balls': balls,
                'rackets': rackets,
                'timing': timing
            }
            
        except Exception as e:
            print(f"❌ 异步检测错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回空结果
            return {
                'keypoints': [],
                'balls': [],
                'rackets': [],
                'timing': {'total': time.time() - start_time, 'error': str(e)}
            }
    
    def _detect_pose(self, pose_estimator, frame: np.ndarray, frame_count: int) -> Tuple[Any, float]:
        """姿态检测任务"""
        start = time.time()
        try:
            keypoints = pose_estimator.get_keypoints(frame)
            return keypoints, time.time() - start
        except Exception as e:
            print(f"⚠️ [帧{frame_count}] 姿态检测错误: {e}")
            return [], time.time() - start
    
    def _detect_ball(
        self,
        ball_tracker,
        roi_frame: np.ndarray,
        roi_offset: Tuple[int, int],
        frame_count: int
    ) -> Tuple[Any, float]:
        """球检测任务"""
        start = time.time()
        try:
            # BallTracker.predict_ball 只接受 roi_frame 参数
            balls = ball_tracker.predict_ball(roi_frame)
            return balls, time.time() - start
        except Exception as e:
            print(f"⚠️ [帧{frame_count}] 球检测错误: {e}")
            return [], time.time() - start
    
    def _detect_racket(
        self,
        racket_detector,
        roi_frame: np.ndarray,
        roi_offset: Tuple[int, int],
        frame_count: int
    ) -> Tuple[Any, float]:
        """球拍检测任务"""
        start = time.time()
        try:
            # RacketDetector.detect_rackets 只接受 roi_frame 参数
            rackets = racket_detector.detect_rackets(roi_frame)
            return rackets, time.time() - start
        except Exception as e:
            print(f"⚠️ [帧{frame_count}] 球拍检测错误: {e}")
            return [], time.time() - start
    
    def get_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {
            'total_detections': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
    
    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)
    
    def __del__(self):
        """析构函数，确保线程池关闭"""
        try:
            self.shutdown()
        except:
            pass


# 测试代码
if __name__ == "__main__":
    print("🧪 测试异步检测器")
    
    # 模拟检测器
    class MockDetector:
        def __init__(self, name, delay):
            self.name = name
            self.delay = delay
        
        def detect(self, *args, **kwargs):
            time.sleep(self.delay)
            return f"{self.name}_result"
    
    class MockPoseEstimator:
        def get_keypoints(self, frame):
            time.sleep(0.05)  # 模拟50ms
            return [{'keypoint': 'test'}]
    
    class MockBallTracker:
        def predict_ball(self, frame, offset):
            time.sleep(0.03)  # 模拟30ms
            return [(100, 100, 10, 0.9)]
    
    class MockRacketDetector:
        def detect_rackets(self, frame, offset):
            time.sleep(0.04)  # 模拟40ms
            return [{'bbox': [50, 50, 150, 150]}]
    
    # 创建异步检测器
    async_detector = AsyncDetector(max_workers=3)
    
    # 模拟帧
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    roi_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # 测试
    print("\n📊 顺序检测测试:")
    start = time.time()
    pose = MockPoseEstimator().get_keypoints(frame)
    ball = MockBallTracker().predict_ball(roi_frame, (0, 0))
    racket = MockRacketDetector().detect_rackets(roi_frame, (0, 0))
    sequential_time = time.time() - start
    print(f"   耗时: {sequential_time*1000:.2f}ms")
    
    print("\n📊 并行检测测试:")
    start = time.time()
    result = async_detector.detect_async(
        frame, roi_frame, (0, 0),
        MockPoseEstimator(),
        MockBallTracker(),
        MockRacketDetector(),
        0
    )
    parallel_time = result['timing']['total']
    print(f"   耗时: {parallel_time*1000:.2f}ms")
    print(f"   姿态: {result['timing']['pose']*1000:.2f}ms")
    print(f"   球: {result['timing']['ball']*1000:.2f}ms")
    print(f"   球拍: {result['timing']['racket']*1000:.2f}ms")
    print(f"   并行效率: {result['timing']['parallel_efficiency']:.2f}x")
    
    print(f"\n✅ 性能提升: {sequential_time/parallel_time:.2f}x")
    print(f"   顺序: {sequential_time*1000:.2f}ms")
    print(f"   并行: {parallel_time*1000:.2f}ms")
    print(f"   节省: {(sequential_time-parallel_time)*1000:.2f}ms")
    
    async_detector.shutdown()
