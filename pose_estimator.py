# pose_estimator.py
from ultralytics import YOLO
import numpy as np
import cv2

# COCO Keypoint indices (may vary slightly based on model, check YOLOv8 docs)
# 0: nose, 1: left_eye, 2: right_eye, ..., 5: left_shoulder, 6: right_shoulder,
# 7: left_elbow, 8: right_elbow, 9: left_wrist, 10: right_wrist,
# 11: left_hip, 12: right_hip, ...

class PoseEstimator:
    def __init__(self, model_path, config):
        self.model = YOLO(model_path)
        self.config = config
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ] # Standard COCO 17 keypoints
        
        # 定义骨架连接，每对表示连接两个关键点的线
        self.skeleton = [
            ["right_shoulder", "right_elbow"],  # 右上臂
            ["right_elbow", "right_wrist"],     # 右前臂
            ["left_shoulder", "left_elbow"],    # 左上臂
            ["left_elbow", "left_wrist"],       # 左前臂
            ["right_shoulder", "left_shoulder"], # 肩膀连线
            ["right_hip", "left_hip"],          # 髋部连线
            ["right_shoulder", "right_hip"],    # 右侧躯干
            ["left_shoulder", "left_hip"],      # 左侧躯干
            ["right_hip", "right_knee"],        # 右大腿
            ["right_knee", "right_ankle"],      # 右小腿
            ["left_hip", "left_knee"],          # 左大腿
            ["left_knee", "left_ankle"],        # 左小腿
        ]
        
        # 不同骨架部分的颜色
        self.colors = {
            "right_arm": (255, 140, 0),    # 橙色 - 右臂
            "left_arm": (135, 206, 235),   # 天蓝色 - 左臂
            "torso": (75, 0, 130),         # 靛蓝色 - 躯干
            "legs": (50, 205, 50)          # 绿色 - 腿部
        }

    def get_keypoints(self, frame):
        results = self.model(frame, verbose=False) # verbose=False to reduce console output
        person_keypoints_list = []
        for result in results:
            if result.keypoints and result.keypoints.xy.numel() > 0:
                # Assuming one person of interest, or take the one with highest confidence
                # For multiple people, you'd need to track or select the main player
                keypoints_xy = result.keypoints.xy[0].cpu().numpy() # Get (x,y) for first detected person
                keypoints_conf = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else None

                # Filter by confidence
                if keypoints_conf is not None:
                    valid_kpts = {}
                    for i, name in enumerate(self.keypoint_names):
                        if keypoints_conf[i] > self.config['pose_confidence_threshold']:
                            valid_kpts[name] = (int(keypoints_xy[i, 0]), int(keypoints_xy[i, 1]))
                        else:
                            valid_kpts[name] = None # Mark as not detected with enough confidence
                    person_keypoints_list.append(valid_kpts)
                else: # If confidence scores are not available, take all
                    valid_kpts = {name: (int(keypoints_xy[i,0]), int(keypoints_xy[i,1])) for i, name in enumerate(self.keypoint_names)}
                    person_keypoints_list.append(valid_kpts)

        return person_keypoints_list # List of dictionaries, one per person

    def classify_swing(self, keypoints_dict):
        if not keypoints_dict:
            return "No Pose"

        # For simplicity, taking the first detected person's keypoints
        kpts = keypoints_dict[0]

        # Required keypoints for basic classification
        lw = kpts.get("left_wrist")
        rw = kpts.get("right_wrist")
        ls = kpts.get("left_shoulder")
        rs = kpts.get("right_shoulder")
        le = kpts.get("left_elbow")
        re = kpts.get("right_elbow")

        if not all([lw, rw, ls, rs, le, re]):
            return "Incomplete Pose"

        # --- Two-Handed Backhand Logic ---
        # Check if both wrists are close
        wrist_dist = np.linalg.norm(np.array(lw) - np.array(rw))
        if wrist_dist < self.config['two_hand_wrist_distance_max_px']:
            # Basic check: if racket (assumed near wrists) is on left side of body for right-hander
            # More robust: check which shoulder is forward, body rotation
            body_center_x = (ls[0] + rs[0]) / 2
            avg_wrist_x = (lw[0] + rw[0]) / 2
            if (self.config['dominant_hand'] == "right" and avg_wrist_x < body_center_x) or \
               (self.config['dominant_hand'] == "left" and avg_wrist_x > body_center_x):
                return "Two-Handed Backhand"
            # Could be two-handed forehand prep, but less common for a full swing label

        # --- One-Handed Logic ---
        # This is highly simplified and needs refinement based on swing phase (prep, contact, follow-through)
        # and relative positions of keypoints.
        # For a proper classification, you'd analyze the sequence of poses.
        # Here's a very basic idea, assuming mid-swing:

        if self.config['dominant_hand'] == "right":
            active_wrist, active_elbow, active_shoulder = rw, re, rs
            non_active_wrist, non_active_shoulder = lw, ls
        else:
            active_wrist, active_elbow, active_shoulder = lw, le, ls
            non_active_wrist, non_active_shoulder = rw, rs

        # Check if active arm is extended across the body (backhand) or on the dominant side (forehand)
        if (self.config['dominant_hand'] == "right" and active_wrist[0] < active_shoulder[0]) or \
           (self.config['dominant_hand'] == "left" and active_wrist[0] > active_shoulder[0]):
            # Further checks for backhand, e.g., arm angle
            # angle = self.calculate_angle(active_shoulder, active_elbow, active_wrist)
            # if self.config['backhand_wrist_elbow_angle_min'] < angle < self.config['backhand_wrist_elbow_angle_max']:
            return "Backhand"
        else:
            # Further checks for forehand
            # angle = self.calculate_angle(active_shoulder, active_elbow, active_wrist)
            # if self.config['forehand_wrist_elbow_angle_min'] < angle < self.config['forehand_wrist_elbow_angle_max']:
            return "Forehand"

        return "Undetermined Swing"

    def calculate_angle(self, p1, p2, p3): # p2 is the vertex
        """Calculates the angle between three points (p1-p2-p3)."""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0: return 0.0 # Avoid division by zero
        angle = np.arccos(dot_product / norm_product)
        return np.degrees(angle)

    def draw_keypoints(self, frame, person_keypoints_list):
        if not person_keypoints_list:
            return frame
            
        # 绘制第一个检测到的人的姿势（通常是主要运动员）
        keypoints = person_keypoints_list[0]
        
        # 定义头部关键点，我们将跳过这些点的绘制
        head_keypoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
        
        # 1. 首先获取人体框的范围，但排除头部关键点
        valid_pts = [pt for name, pt in keypoints.items() if pt is not None and name not in head_keypoints]
        if not valid_pts:
            return frame
            
        x_coords = [pt[0] for pt in valid_pts]
        y_coords = [pt[1] for pt in valid_pts]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 添加边距
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        # 2. 绘制人体框和标签
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        # 将标签移动到人体框的右下角
        cv2.putText(frame, "tennis player", (x_max - 120, y_max), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 3. 绘制关键点连接线（骨架），跳过任何连接头部关键点的线
        for connection in self.skeleton:
            name_a, name_b = connection
            
            # 跳过涉及头部关键点的连接
            if name_a in head_keypoints or name_b in head_keypoints:
                continue
                
            pt_a, pt_b = keypoints.get(name_a), keypoints.get(name_b)
            
            if pt_a and pt_b:
                # 根据连接的部位选择不同颜色
                if "arm" in name_a or "arm" in name_b or "wrist" in name_a or "wrist" in name_b or "elbow" in name_a or "elbow" in name_b:
                    if "right" in name_a or "right" in name_b:
                        color = self.colors["right_arm"]
                    else:
                        color = self.colors["left_arm"]
                elif "hip" in name_a or "hip" in name_b or "shoulder" in name_a or "shoulder" in name_b:
                    color = self.colors["torso"]
                else:
                    color = self.colors["legs"]
                
                cv2.line(frame, pt_a, pt_b, color, 2)
        
        # 4. 绘制关键点，但跳过头部关键点
        for name, pt in keypoints.items():
            # 跳过头部关键点
            if name in head_keypoints:
                continue
                
            if pt:
                # 为不同部位的关键点使用不同颜色
                if "wrist" in name or "elbow" in name:
                    if "right" in name:
                        color = self.colors["right_arm"]
                    else:
                        color = self.colors["left_arm"]
                elif "shoulder" in name or "hip" in name:
                    color = self.colors["torso"]
                elif "knee" in name or "ankle" in name:
                    color = self.colors["legs"]
                else:
                    color = (255, 0, 255)  # 紫色用于其他点
                
                # 关键点编号（对应图片中的编号）
                point_id = self.keypoint_names.index(name)
                
                # 绘制关键点（圆点）
                cv2.circle(frame, pt, 5, color, -1)
                
                # 显示关键点编号
                cv2.putText(frame, str(point_id), (pt[0] + 5, pt[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        return frame