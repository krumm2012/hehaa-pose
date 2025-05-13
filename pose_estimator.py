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
        for kpts_dict in person_keypoints_list:
            for name, pt in kpts_dict.items():
                if pt:
                    cv2.circle(frame, pt, 3, (0, 255, 0), -1)
                    # cv2.putText(frame, name, (pt[0]+5, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,0,0), 1)
        return frame