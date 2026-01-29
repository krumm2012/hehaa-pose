# pose_estimator.py
# 统一的姿态估计器接口 - 自动检测并加载 YOLOv8 或 YOLO26 模型

import os
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional

def create_pose_estimator(model_path: str, config: dict, roi_manager=None):
    """
    工厂函数：根据模型路径自动创建合适的姿态估计器
    
    Args:
        model_path: 模型文件路径
        config: 配置字典
        roi_manager: ROI 管理器 (可选)
        
    Returns:
        姿态估计器实例
    """
    # 检测模型类型
    if model_path.endswith('.mlpackage') or os.path.isdir(model_path):
        # Core ML 模型 (YOLO26)
        print("🔍 检测到 Core ML 模型格式，使用 YOLO26-pose 加载器")
        try:
            from pose_estimator_yolo26 import PoseEstimatorYOLO26
            return PoseEstimatorYOLO26(model_path, config, roi_manager)
        except ImportError as e:
            print(f"❌ 无法导入 YOLO26 加载器: {e}")
            print("💡 请安装 coremltools: pip install coremltools")
            raise
    
    elif model_path.endswith('.pt') or model_path.endswith('.onnx'):
        # PyTorch 或 ONNX 模型 (YOLOv8)
        print("🔍 检测到 PyTorch/ONNX 模型格式，使用 YOLOv8-pose 加载器")
        from ultralytics import YOLO
        return PoseEstimatorYOLOv8(model_path, config, roi_manager)
    
    else:
        raise ValueError(f"不支持的模型格式: {model_path}")


class PoseEstimatorYOLOv8:
    """YOLOv8-pose 姿态估计器 (原始实现)"""
    
    def __init__(self, model_path: str, config: dict, roi_manager=None):
        from ultralytics import YOLO
        
        print(f"🤖 加载 YOLOv8-pose 模型: {model_path}")
        self.model = YOLO(model_path)
        self.config = config
        self.roi_manager = roi_manager
        
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        self.skeleton = [
            ["right_shoulder", "right_elbow"],
            ["right_elbow", "right_wrist"],
            ["left_shoulder", "left_elbow"],
            ["left_elbow", "left_wrist"],
            ["right_shoulder", "left_shoulder"],
            ["right_hip", "left_hip"],
            ["right_shoulder", "right_hip"],
            ["left_shoulder", "left_hip"],
            ["right_hip", "right_knee"],
            ["right_knee", "right_ankle"],
            ["left_hip", "left_knee"],
            ["left_knee", "left_ankle"],
        ]
        
        self.colors = {
            "right_arm": (255, 140, 0),
            "left_arm": (135, 206, 235),
            "torso": (75, 0, 130),
            "legs": (50, 205, 50)
        }
        
        print("✅ YOLOv8-pose 模型加载成功")
    
    def get_keypoints(self, frame: np.ndarray) -> List[Dict]:
        """检测关键点"""
        results = self.model(frame, verbose=False)
        person_keypoints_list = []
        
        for result in results:
            if result.keypoints and result.keypoints.xy.numel() > 0:
                keypoints_xy = result.keypoints.xy[0].cpu().numpy()
                keypoints_conf = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else None
                
                if keypoints_conf is not None:
                    valid_kpts = {}
                    for i, name in enumerate(self.keypoint_names):
                        if keypoints_conf[i] > self.config.get('pose_confidence_threshold', 0.5):
                            valid_kpts[name] = (int(keypoints_xy[i, 0]), int(keypoints_xy[i, 1]))
                        else:
                            valid_kpts[name] = None
                    person_keypoints_list.append(valid_kpts)
                else:
                    valid_kpts = {name: (int(keypoints_xy[i,0]), int(keypoints_xy[i,1])) 
                                 for i, name in enumerate(self.keypoint_names)}
                    person_keypoints_list.append(valid_kpts)
        
        # ROI 过滤
        if self.roi_manager and self.roi_manager.is_roi_set:
            filtered_keypoints = []
            for keypoints in person_keypoints_list:
                filtered_keypoints_dict = self.roi_manager.filter_detections_by_roi([keypoints], "pose")
                filtered_keypoints.extend(filtered_keypoints_dict)
            person_keypoints_list = filtered_keypoints
        
        return person_keypoints_list
    
    def classify_swing(self, keypoints_dict: List[Dict]) -> str:
        """分类挥拍类型"""
        if not keypoints_dict:
            return "No Pose"
        
        kpts = keypoints_dict[0]
        lw = kpts.get("left_wrist")
        rw = kpts.get("right_wrist")
        ls = kpts.get("left_shoulder")
        rs = kpts.get("right_shoulder")
        le = kpts.get("left_elbow")
        re = kpts.get("right_elbow")
        
        if not all([lw, rw, ls, rs, le, re]):
            return "Incomplete Pose"
        
        # 双手反手
        wrist_dist = np.linalg.norm(np.array(lw) - np.array(rw))
        if wrist_dist < self.config.get('two_hand_wrist_distance_max_px', 50):
            body_center_x = (ls[0] + rs[0]) / 2
            avg_wrist_x = (lw[0] + rw[0]) / 2
            if (self.config.get('dominant_hand', 'right') == "right" and avg_wrist_x < body_center_x) or \
               (self.config.get('dominant_hand', 'right') == "left" and avg_wrist_x > body_center_x):
                return "Two-Handed Backhand"
        
        # 单手判断
        if self.config.get('dominant_hand', 'right') == "right":
            active_wrist, active_shoulder = rw, rs
        else:
            active_wrist, active_shoulder = lw, ls
        
        if (self.config.get('dominant_hand', 'right') == "right" and active_wrist[0] < active_shoulder[0]) or \
           (self.config.get('dominant_hand', 'right') == "left" and active_wrist[0] > active_shoulder[0]):
            return "Backhand"
        else:
            return "Forehand"
    
    def calculate_angle(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        """计算角度"""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 0.0
        angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
        return np.degrees(angle)
    
    def draw_keypoints(self, frame: np.ndarray, person_keypoints_list: List[Dict]) -> np.ndarray:
        """绘制关键点"""
        if not person_keypoints_list:
            return frame
        
        keypoints = person_keypoints_list[0]
        head_keypoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
        
        valid_pts = [pt for name, pt in keypoints.items() if pt is not None and name not in head_keypoints]
        if not valid_pts:
            return frame
        
        x_coords = [pt[0] for pt in valid_pts]
        y_coords = [pt[1] for pt in valid_pts]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        cv2.putText(frame, "tennis player", (x_max - 120, y_max),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 绘制骨架
        for connection in self.skeleton:
            name_a, name_b = connection
            if name_a in head_keypoints or name_b in head_keypoints:
                continue
            
            pt_a, pt_b = keypoints.get(name_a), keypoints.get(name_b)
            if pt_a and pt_b:
                if "arm" in name_a or "arm" in name_b or "wrist" in name_a or "wrist" in name_b or "elbow" in name_a or "elbow" in name_b:
                    color = self.colors["right_arm"] if ("right" in name_a or "right" in name_b) else self.colors["left_arm"]
                elif "hip" in name_a or "hip" in name_b or "shoulder" in name_a or "shoulder" in name_b:
                    color = self.colors["torso"]
                else:
                    color = self.colors["legs"]
                cv2.line(frame, pt_a, pt_b, color, 2)
        
        # 绘制关键点
        for name, pt in keypoints.items():
            if name in head_keypoints or pt is None:
                continue
            
            if "wrist" in name or "elbow" in name:
                color = self.colors["right_arm"] if "right" in name else self.colors["left_arm"]
            elif "shoulder" in name or "hip" in name:
                color = self.colors["torso"]
            elif "knee" in name or "ankle" in name:
                color = self.colors["legs"]
            else:
                color = (255, 0, 255)
            
            cv2.circle(frame, pt, 5, color, -1)
            point_id = self.keypoint_names.index(name)
            cv2.putText(frame, str(point_id), (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    @staticmethod
    def draw_keypoints_static(frame: np.ndarray, person_keypoints_list: List[Dict]) -> np.ndarray:
        """静态绘制函数"""
        if not person_keypoints_list:
            return frame
        
        head_keypoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        skeleton = [
            ["right_shoulder", "right_elbow"],
            ["right_elbow", "right_wrist"],
            ["left_shoulder", "left_elbow"],
            ["left_elbow", "left_wrist"],
            ["right_shoulder", "left_shoulder"],
            ["right_hip", "left_hip"],
            ["right_shoulder", "right_hip"],
            ["left_shoulder", "left_hip"],
            ["right_hip", "right_knee"],
            ["right_knee", "right_ankle"],
            ["left_hip", "left_knee"],
            ["left_knee", "left_ankle"],
        ]
        colors = {
            "right_arm": (255, 140, 0),
            "left_arm": (135, 206, 235),
            "torso": (75, 0, 130),
            "legs": (50, 205, 50)
        }
        
        keypoints = person_keypoints_list[0]
        valid_pts = [pt for name, pt in keypoints.items() if pt is not None and name not in head_keypoints]
        if not valid_pts:
            return frame
        
        for name_a, name_b in skeleton:
            if name_a in head_keypoints or name_b in head_keypoints:
                continue
            pt_a = keypoints.get(name_a)
            pt_b = keypoints.get(name_b)
            if pt_a and pt_b:
                if any(k in name_a+name_b for k in ["wrist", "elbow"]):
                    color = colors["right_arm"] if ("right" in name_a or "right" in name_b) else colors["left_arm"]
                elif any(k in name_a+name_b for k in ["hip", "shoulder"]):
                    color = colors["torso"]
                else:
                    color = colors["legs"]
                cv2.line(frame, pt_a, pt_b, color, 2)
        
        for name, pt in keypoints.items():
            if name in head_keypoints or pt is None:
                continue
            if "wrist" in name or "elbow" in name:
                color = colors["right_arm"] if "right" in name else colors["left_arm"]
            elif "shoulder" in name or "hip" in name:
                color = colors["torso"]
            elif "knee" in name or "ankle" in name:
                color = colors["legs"]
            else:
                color = (255, 0, 255)
            cv2.circle(frame, pt, 5, color, -1)
        
        return frame


# 默认使用工厂函数创建估计器
class PoseEstimator:
    """姿态估计器包装类 - 自动选择合适的实现"""
    
    def __new__(cls, model_path: str, config: dict, roi_manager=None):
        """使用工厂模式创建合适的估计器实例"""
        return create_pose_estimator(model_path, config, roi_manager)