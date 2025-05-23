# full_swing_analyzer.py
import numpy as np
import cv2 # For potential drawing or simple geometric helpers if needed

class FullSwingAnalyzer:
    def __init__(self, config):
        self.config = config
        # Keypoint names, ensure consistency with your pose_estimator
        self.kp_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        self.dominant_hand = self.config.get('dominant_hand', 'right')

    def _calculate_angle(self, p1, p2, p3): # p2 is the vertex
        """Calculates the angle between three points (p1-p2-p3)."""
        if not all([p1, p2, p3]): return None
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0: return 0.0
        angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0)) # Clip for precision issues
        return np.degrees(angle)

    def _get_midpoint(self, p1, p2):
        if not p1 or not p2: return None
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def _get_keypoint(self, person_kpts_dict, name):
        """Safely retrieves a keypoint, returning None if not found or invalid."""
        kp = person_kpts_dict.get(name)
        if kp and isinstance(kp, (tuple, list)) and len(kp) == 2:
            return kp
        return None

    def analyze_swing_components(self, keypoints_list, racket_results, ball_position, frame_dimensions):
        """
        分析挥拍动作的各个组成部分
        :param keypoints_list: 关键点列表
        :param racket_results: 球拍检测结果列表
        :param ball_position: 球的位置 (x, y)，可能是None
        :param frame_dimensions: 帧的尺寸 (height, width)
        :return: 分析结果字典
        """
        if not keypoints_list:
            return {}

        # 获取第一个检测到的人的关键点
        keypoints = keypoints_list[0]
        
        # 获取第一个检测到的球拍信息（如果有）
        racket_info = racket_results[0] if racket_results else None

        # 确保ball_position是有效的坐标
        if isinstance(ball_position, (list, tuple)) and len(ball_position) >= 2:
            ball_pos = ball_position
        else:
            ball_pos = None

        # 分析结果字典
        analysis = {
            'preparation': self._analyze_preparation(keypoints),
            'swing_motion': self._analyze_swing_motion(keypoints, racket_info, ball_pos),
            'footwork': self._analyze_footwork(keypoints),
            'power_indicators': self._analyze_power_indicators(keypoints)
        }

        return analysis

    def _analyze_preparation(self, keypoints):
        """分析准备阶段的指标"""
        metrics = {}
        
        # 分析肩部转动
        if all(keypoints.get(kp) for kp in ["left_shoulder", "right_shoulder"]):
            ls = np.array(keypoints["left_shoulder"])
            rs = np.array(keypoints["right_shoulder"])
            shoulder_vector = rs - ls
            vertical_vector = np.array([0, 1])
            shoulder_angle = self._calculate_angle_between_vectors(shoulder_vector, vertical_vector)
            metrics['shoulder_turn'] = f"{shoulder_angle:.1f}°"

        # 分析非惯用臂的扩展
        if self.config['dominant_hand'] == "right":
            non_dom_shoulder = keypoints.get("left_shoulder")
            non_dom_elbow = keypoints.get("left_elbow")
            non_dom_wrist = keypoints.get("left_wrist")
        else:
            non_dom_shoulder = keypoints.get("right_shoulder")
            non_dom_elbow = keypoints.get("right_elbow")
            non_dom_wrist = keypoints.get("right_wrist")

        if all([non_dom_shoulder, non_dom_elbow, non_dom_wrist]):
            arm_extension = self._calculate_angle(non_dom_shoulder, non_dom_elbow, non_dom_wrist)
            metrics['non_dominant_arm_extension'] = f"{arm_extension:.1f}°"

        return metrics

    def _analyze_swing_motion(self, keypoints, racket_info, ball_position):
        """分析挥拍动作的指标"""
        metrics = {}

        # 获取惯用手腕位置
        dom_wrist = keypoints.get("right_wrist" if self.config['dominant_hand'] == "right" else "left_wrist")
        
        if dom_wrist and ball_position:
            # 计算击球点相对于身体的位置
            contact_x_diff = ball_position[0] - dom_wrist[0]
            metrics['contact_point_position'] = "Front" if abs(contact_x_diff) < 50 else \
                                              "Side" if contact_x_diff < 0 else "Late"
            
            # 计算击球高度
            metrics['contact_height'] = "Low" if ball_position[1] > dom_wrist[1] + 50 else \
                                      "Mid" if abs(ball_position[1] - dom_wrist[1]) <= 50 else "High"

        # 分析手臂伸展度
        if self.config['dominant_hand'] == "right":
            dom_shoulder = keypoints.get("right_shoulder")
            dom_elbow = keypoints.get("right_elbow")
            dom_wrist = keypoints.get("right_wrist")
        else:
            dom_shoulder = keypoints.get("left_shoulder")
            dom_elbow = keypoints.get("left_elbow")
            dom_wrist = keypoints.get("left_wrist")

        if all([dom_shoulder, dom_elbow, dom_wrist]):
            arm_extension = self._calculate_angle(dom_shoulder, dom_elbow, dom_wrist)
            metrics['arm_extension'] = f"{arm_extension:.1f}°"

        return metrics

    def _analyze_footwork(self, keypoints):
        """分析脚步工作的指标"""
        metrics = {}

        # 计算站姿宽度
        left_ankle = keypoints.get("left_ankle")
        right_ankle = keypoints.get("right_ankle")
        if left_ankle and right_ankle:
            stance_width = np.linalg.norm(np.array(left_ankle) - np.array(right_ankle))
            metrics['stance_width'] = f"{stance_width:.1f}px"

            # 简单的站姿类型判断
            left_hip = keypoints.get("left_hip")
            right_hip = keypoints.get("right_hip")
            if left_hip and right_hip:
                hip_center_x = (left_hip[0] + right_hip[0]) / 2
                ankle_center_x = (left_ankle[0] + right_ankle[0]) / 2
                stance_offset = hip_center_x - ankle_center_x
                metrics['stance_type'] = "Open" if abs(stance_offset) > 30 else "Neutral"

        # 计算膝盖角度
        for side in ['left', 'right']:
            hip = keypoints.get(f"{side}_hip")
            knee = keypoints.get(f"{side}_knee")
            ankle = keypoints.get(f"{side}_ankle")
            if all([hip, knee, ankle]):
                knee_angle = self._calculate_angle(hip, knee, ankle)
                metrics[f'{side}_knee_angle'] = f"{knee_angle:.1f}°"

        return metrics

    def _analyze_power_indicators(self, keypoints):
        """分析力量指标"""
        metrics = {}

        # 计算髋肩分离度
        left_hip = keypoints.get("left_hip")
        right_hip = keypoints.get("right_hip")
        left_shoulder = keypoints.get("left_shoulder")
        right_shoulder = keypoints.get("right_shoulder")

        if all([left_hip, right_hip, left_shoulder, right_shoulder]):
            hip_vector = np.array(right_hip) - np.array(left_hip)
            shoulder_vector = np.array(right_shoulder) - np.array(left_shoulder)
            separation_angle = self._calculate_angle_between_vectors(hip_vector, shoulder_vector)
            metrics['hip_shoulder_separation'] = f"{separation_angle:.1f}°"

        return metrics

    def _calculate_angle_between_vectors(self, v1, v2):
        """计算两个向量之间的角度"""
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0: return 0.0
        angle = np.arccos(dot_product / norm_product)
        return np.degrees(angle)