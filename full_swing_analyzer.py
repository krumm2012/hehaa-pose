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

    def analyze_swing_components(self, person_keypoints_list, associated_rackets_info, kf_estimated_ball_pos, frame_dimensions):
        """
        Analyzes various components of a tennis swing from a single frame.
        Returns a dictionary of analyzed metrics.
        """
        metrics = {
            "phase_estimation": "Unknown",
            "preparation": {},
            "swing_motion": {},
            "footwork": {},
            "power_indicators": {}
        }

        if not person_keypoints_list:
            return metrics # Not enough info if no player

        # For simplicity, analyze the first detected person
        player_kpts = person_keypoints_list[0]
        frame_h, frame_w = frame_dimensions

        # --- Dominant/Non-Dominant Side Assignment ---
        if self.dominant_hand == 'right':
            dom_shoulder = self._get_keypoint(player_kpts, "right_shoulder")
            dom_elbow = self._get_keypoint(player_kpts, "right_elbow")
            dom_wrist = self._get_keypoint(player_kpts, "right_wrist")
            dom_hip = self._get_keypoint(player_kpts, "right_hip")
            dom_knee = self._get_keypoint(player_kpts, "right_knee")
            dom_ankle = self._get_keypoint(player_kpts, "right_ankle")

            nondom_shoulder = self._get_keypoint(player_kpts, "left_shoulder")
            nondom_elbow = self._get_keypoint(player_kpts, "left_elbow")
            nondom_wrist = self._get_keypoint(player_kpts, "left_wrist")
            nondom_hip = self._get_keypoint(player_kpts, "left_hip")
        else: # Left-handed
            dom_shoulder = self._get_keypoint(player_kpts, "left_shoulder")
            dom_elbow = self._get_keypoint(player_kpts, "left_elbow")
            dom_wrist = self._get_keypoint(player_kpts, "left_wrist")
            # ... (assign other dominant/non-dominant parts for left-hander)
            nondom_shoulder = self._get_keypoint(player_kpts, "right_shoulder")
            nondom_elbow = self._get_keypoint(player_kpts, "right_elbow")
            nondom_wrist = self._get_keypoint(player_kpts, "right_wrist")
            nondom_hip = self._get_keypoint(player_kpts, "right_hip")


        # --- Racket Info ---
        # Assuming player_id 0 for the main player
        racket_info = associated_rackets_info.get(0)
        racket_box = racket_center = racket_state = None
        if racket_info:
            racket_box = racket_info.get('box')
            racket_state = racket_info.get('state')
            if racket_box:
                racket_center = ((racket_box[0] + racket_box[2]) / 2, (racket_box[1] + racket_box[3]) / 2)

        # --- Phase Estimation (Very Basic) ---
        # This is highly dependent on ball and racket state from racket_detector
        if racket_state:
            if "IMPACT" in racket_state:
                metrics["phase_estimation"] = "Impact Phase"
            elif "APPROACHING" in racket_state:
                metrics["phase_estimation"] = "Forward Swing / Approaching Ball"
            elif racket_center and dom_shoulder and \
                 ((self.dominant_hand == 'right' and racket_center[0] < dom_shoulder[0] - (racket_box[2]-racket_box[0])/2) or \
                  (self.dominant_hand == 'left' and racket_center[0] > dom_shoulder[0] + (racket_box[2]-racket_box[0])/2)):
                metrics["phase_estimation"] = "Backswing / Preparation"
            elif racket_center and dom_shoulder and \
                 ((self.dominant_hand == 'right' and racket_center[0] > dom_shoulder[0] + (racket_box[2]-racket_box[0])) or \
                  (self.dominant_hand == 'left' and racket_center[0] < dom_shoulder[0] - (racket_box[2]-racket_box[0]))):
                metrics["phase_estimation"] = "Follow-Through (Basic)"
            else:
                metrics["phase_estimation"] = "Idle / Other"


        # --- 1. 引拍 (Yǐnpāi - Preparation) ---
        prep = metrics["preparation"]
        if dom_shoulder and nondom_shoulder:
            shoulder_angle_to_horizontal = np.arctan2(dom_shoulder[1] - nondom_shoulder[1], dom_shoulder[0] - nondom_shoulder[0]) * 180 / np.pi
            prep["shoulder_turn_degrees"] = f"{abs(shoulder_angle_to_horizontal):.1f}" # Angle from horizontal
            # More qualitative: Is non-dominant shoulder pointing towards net (assuming net is to player's side)
            # This requires knowing camera orientation relative to court. For now, just the angle.

        if racket_center and dom_shoulder:
            racket_x_rel_shoulder = racket_center[0] - dom_shoulder[0]
            racket_y_rel_shoulder = racket_center[1] - dom_shoulder[1]
            prep["racket_pos_rel_dom_shoulder_px"] = f"({racket_x_rel_shoulder:.0f}, {racket_y_rel_shoulder:.0f})"
            if (self.dominant_hand == 'right' and racket_x_rel_shoulder < -20) or \
               (self.dominant_hand == 'left' and racket_x_rel_shoulder > 20):
                prep["racket_takeback"] = "Yes (Behind Shoulder)"
            else:
                prep["racket_takeback"] = "No / Forward"

        if nondom_elbow and nondom_shoulder and nondom_wrist:
            nondom_arm_angle = self._calculate_angle(nondom_shoulder, nondom_elbow, nondom_wrist)
            prep["nondom_arm_elbow_angle_deg"] = f"{nondom_arm_angle:.1f}" if nondom_arm_angle is not None else "N/A"
            if nondom_arm_angle is not None and nondom_arm_angle > 140:
                prep["nondom_arm_usage"] = "Extended (Good for balance/pointing)"
            elif nondom_arm_angle is not None:
                prep["nondom_arm_usage"] = "Bent"

        # --- 2. 挥拍 (Huīpāi - Swing Motion, including impact) ---
        swing = metrics["swing_motion"]
        if racket_center and kf_estimated_ball_pos:
            # Inferred contact point relative to body (at current frame if phase is "Impact")
            # This is very simplified. True contact point analysis is complex.
            body_center_x = self._get_midpoint(self._get_keypoint(player_kpts, "left_hip"), self._get_keypoint(player_kpts, "right_hip"))
            if body_center_x: body_center_x = body_center_x[0]

            if body_center_x and racket_center:
                if abs(racket_center[0] - kf_estimated_ball_pos[0]) < 30 and \
                   abs(racket_center[1] - kf_estimated_ball_pos[1]) < 30 : # If racket and ball are close
                    if (self.dominant_hand == 'right' and racket_center[0] > body_center_x + 20) or \
                       (self.dominant_hand == 'left' and racket_center[0] < body_center_x - 20):
                        swing["inferred_contact_point"] = "In Front of Body"
                    elif abs(racket_center[0] - body_center_x) <= 20 :
                        swing["inferred_contact_point"] = "Side of Body"
                    else:
                        swing["inferred_contact_point"] = "Behind Body (Late)"
                    swing["contact_height_ratio_frame"] = f"{racket_center[1] / frame_h:.2f}"
                else:
                    swing["inferred_contact_point"] = "Racket/Ball not at impact"
        else:
            swing["inferred_contact_point"] = "N/A (No Racket/Ball)"

        if dom_elbow and dom_shoulder and dom_wrist:
            dom_arm_angle = self._calculate_angle(dom_shoulder, dom_elbow, dom_wrist)
            swing["dom_arm_elbow_angle_deg"] = f"{dom_arm_angle:.1f}" if dom_arm_angle is not None else "N/A"
            if dom_arm_angle is not None and metrics["phase_estimation"] == "Impact Phase":
                swing["arm_extension_at_impact"] = "Extended" if dom_arm_angle > 130 else "Bent"


        # --- 3. 脚步 (Jiǎobù - Footwork) ---
        foot = metrics["footwork"]
        lknee = self._get_keypoint(player_kpts, "left_knee")
        rknee = self._get_keypoint(player_kpts, "right_knee")
        lankle = self._get_keypoint(player_kpts, "left_ankle")
        rankle = self._get_keypoint(player_kpts, "right_ankle")

        if lankle and rankle:
            stance_width = np.linalg.norm(np.array(lankle) - np.array(rankle))
            foot["stance_width_px"] = f"{stance_width:.0f}"
            # Qualitative stance width (needs calibration or relation to player height)
            # For now, just pixels.

            # Stance type (very simplified, based on ankle X positions relative to dominant side)
            # Assumes player is somewhat facing forward or sideways to camera.
            # A true stance analysis needs orientation relative to net/ball.
            if abs(lankle[0] - rankle[0]) < stance_width * 0.2: # Ankles almost aligned vertically
                foot["stance_type_guess"] = "Neutral/Square (approx)"
            elif (self.dominant_hand == 'right' and lankle[0] < rankle[0]) or \
                 (self.dominant_hand == 'left' and rankle[0] < lankle[0]):
                foot["stance_type_guess"] = "Open/Semi-Open (approx)"
            else:
                foot["stance_type_guess"] = "Closed (approx)"
        else:
            foot["stance_width_px"] = "N/A"
            foot["stance_type_guess"] = "N/A"

        if lknee and self._get_keypoint(player_kpts, "left_hip") and lankle:
            left_knee_angle = self._calculate_angle(self._get_keypoint(player_kpts, "left_hip"), lknee, lankle)
            foot["left_knee_angle_deg"] = f"{left_knee_angle:.1f}" if left_knee_angle is not None else "N/A"
        if rknee and self._get_keypoint(player_kpts, "right_hip") and rankle:
            right_knee_angle = self._calculate_angle(self._get_keypoint(player_kpts, "right_hip"), rknee, rankle)
            foot["right_knee_angle_deg"] = f"{right_knee_angle:.1f}" if right_knee_angle is not None else "N/A"


        # --- 4. 发力 (Fālì - Power Generation Indicators) ---
        power = metrics["power_indicators"]
        if lknee and rknee and (foot.get("left_knee_angle_deg", "N/A") != "N/A" and foot.get("right_knee_angle_deg", "N/A") != "N/A"):
            avg_knee_angle = (float(foot["left_knee_angle_deg"]) + float(foot["right_knee_angle_deg"])) / 2
            if avg_knee_angle < 140:
                power["leg_bend_indicator"] = "Significant (Good for power)"
            elif avg_knee_angle < 165:
                power["leg_bend_indicator"] = "Moderate"
            else:
                power["leg_bend_indicator"] = "Straight Legs (Less power from legs)"
        else:
            power["leg_bend_indicator"] = "N/A"

        # Hip-Shoulder Separation (simplified)
        # Compare alignment of hips vs shoulders. If different, indicates rotation/coil.
        if dom_hip and nondom_hip and dom_shoulder and nondom_shoulder:
            hip_line_vec = np.array(dom_hip) - np.array(nondom_hip)
            shoulder_line_vec = np.array(dom_shoulder) - np.array(nondom_shoulder)
            # Normalize (optional, but good for consistent dot product)
            hip_line_vec_norm = hip_line_vec / (np.linalg.norm(hip_line_vec) + 1e-6)
            shoulder_line_vec_norm = shoulder_line_vec / (np.linalg.norm(shoulder_line_vec) + 1e-6)
            
            dot_prod = np.dot(hip_line_vec_norm, shoulder_line_vec_norm)
            separation_angle_rad = np.arccos(np.clip(dot_prod, -1.0, 1.0))
            separation_angle_deg = np.degrees(separation_angle_rad)
            power["hip_shoulder_separation_deg"] = f"{separation_angle_deg:.1f}"
            if separation_angle_deg > 15: # Arbitrary threshold
                 power["body_coil_indicator"] = "Coiled (Potential for rotational power)"
            else:
                 power["body_coil_indicator"] = "Less Coil"
        else:
            power["hip_shoulder_separation_deg"] = "N/A"
            power["body_coil_indicator"] = "N/A"

        return metrics