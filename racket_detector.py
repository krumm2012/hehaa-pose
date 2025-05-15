# racket_detector.py
from ultralytics import YOLO
import cv2
import numpy as np

class RacketDetector:
    def __init__(self, model_path, config):
        self.model = YOLO(model_path)
        self.config = config
        self.target_class_id = 38  # COCO class ID for "tennis racket"
        self.confidence_threshold = self.config.get('racket_confidence_threshold', 0.3)

        # For associating racket to player and simple state tracking
        self.player_rackets = {} # {player_id: {'box': [], 'confidence': 0, 'last_seen_frame': 0, 'state': 'unknown'}}

    def detect_rackets(self, frame):
        """
        Detects tennis rackets in the given frame.
        Returns: A list of dictionaries, where each dictionary contains:
                 {'box': [x1, y1, x2, y2], 'confidence': conf, 'class_name': 'tennis racket'}
        """
        results = self.model(frame, verbose=False, classes=[self.target_class_id])
        detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

            for i in range(len(boxes)):
                if confidences[i] >= self.confidence_threshold and int(class_ids[i]) == self.target_class_id:
                    box = boxes[i].astype(int)
                    detections.append({
                        'box': list(box),
                        'confidence': float(confidences[i]),
                        'class_name': self.model.names[self.target_class_id]
                    })
        return detections

    def associate_racket_to_player(self, racket_detections, person_keypoints_list, frame_num):
        """
        Associates detected rackets to players based on proximity to hands.
        Updates self.player_rackets.
        For simplicity, assumes one primary player or the first detected player if multiple.
        """
        # Reset last_seen_frame for existing tracked rackets to check if they are still present
        for player_id in list(self.player_rackets.keys()):
            if frame_num - self.player_rackets[player_id]['last_seen_frame'] > self.config.get('racket_lost_threshold_frames', 5):
                # print(f"Player {player_id} racket lost.")
                del self.player_rackets[player_id]


        if not person_keypoints_list or not racket_detections:
            return self.player_rackets # Return current state if no players or rackets

        # Assuming player_id 0 for the primary player for now
        # A more robust system would track player IDs from pose estimation
        player_id = 0 # Simplified: first detected person
        player_kpts = person_keypoints_list[0] # Get keypoints for the first (assumed primary) player

        left_wrist = player_kpts.get("left_wrist")
        right_wrist = player_kpts.get("right_wrist")
        player_hands_coords = []
        if left_wrist: player_hands_coords.append(left_wrist)
        if right_wrist: player_hands_coords.append(right_wrist)

        if not player_hands_coords:
            return self.player_rackets # Player has no visible hands

        best_racket_for_player = None
        min_dist_to_hand = float('inf')

        for racket in racket_detections:
            racket_box = racket['box']
            racket_center_x = (racket_box[0] + racket_box[2]) / 2
            racket_center_y = (racket_box[1] + racket_box[3]) / 2
            racket_center = np.array([racket_center_x, racket_center_y])

            for hand_coord in player_hands_coords:
                dist = np.linalg.norm(racket_center - np.array(hand_coord))
                # Check if hand is somewhat within or very near the racket's vertical span
                hand_in_racket_y_span = racket_box[1] < hand_coord[1] < racket_box[3]
                # Check if hand is near the racket box in general
                # A simple check: distance from hand to racket center should be less than racket diagonal / 2
                racket_width = racket_box[2] - racket_box[0]
                racket_height = racket_box[3] - racket_box[1]
                max_association_dist = (racket_width + racket_height) * 0.75 # Heuristic for proximity

                if dist < min_dist_to_hand and dist < max_association_dist :
                    min_dist_to_hand = dist
                    best_racket_for_player = racket

        if best_racket_for_player:
            self.player_rackets[player_id] = {
                'box': best_racket_for_player['box'],
                'confidence': best_racket_for_player['confidence'],
                'last_seen_frame': frame_num,
                'state': 'associated_to_player' # Initial state
            }
        return self.player_rackets


    def determine_racket_state(self, player_id, ball_position_kf, frame_height):
        """
        Determines the state of the racket for a given player.
        States: 'near_hands', 'approaching_ball', 'potential_impact', 'follow_through', 'idle'
        This is a very heuristic and simplified state machine.
        """
        if player_id not in self.player_rackets:
            return 'unknown'

        racket_info = self.player_rackets[player_id]
        racket_box = racket_info['box']
        racket_center_x = (racket_box[0] + racket_box[2]) / 2
        racket_center_y = (racket_box[1] + racket_box[3]) / 2

        # Default state, can be refined
        current_state = racket_info.get('state', 'associated_to_player')

        if ball_position_kf:
            dist_racket_ball = np.linalg.norm(np.array([racket_center_x, racket_center_y]) - np.array(ball_position_kf))
            
            racket_width = racket_box[2] - racket_box[0]
            racket_height = racket_box[3] - racket_box[1]
            effective_racket_radius = max(racket_width, racket_height) / 2 # Approx

            # Distance thresholds (these need tuning based on camera perspective and scale)
            impact_threshold = effective_racket_radius + self.config.get('ball_radius_px', 10) # Approx ball radius
            approach_threshold = impact_threshold * 5 # Wider zone for approaching

            # Y position relative to typical contact zone (e.g. waist to shoulder height)
            contact_zone_y_min = frame_height * 0.4
            contact_zone_y_max = frame_height * 0.7


            if dist_racket_ball < impact_threshold and \
               contact_zone_y_min < racket_center_y < contact_zone_y_max and \
               contact_zone_y_min < ball_position_kf[1] < contact_zone_y_max:
                current_state = "IMPACT ZONE"
            elif dist_racket_ball < approach_threshold:
                # Further refine: if ball is in front of racket based on swing (complex)
                # For now, just proximity
                current_state = "APPROACHING BALL"
            else:
                # Could check racket speed/movement direction for 'follow_through' or 'preparing'
                # This requires tracking the racket's position over frames, not just current.
                current_state = "IDLE / OTHER"
        else:
            current_state = "IDLE / NO BALL"

        self.player_rackets[player_id]['state'] = current_state
        return current_state


    def draw_rackets(self, frame, racket_detections):
        """Draws all raw detected rackets (before association)."""
        for racket in racket_detections:
            box = racket['box']
            label = f"{racket['class_name']}: {racket['confidence']:.2f}"
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2) # Cyan
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return frame

    def draw_associated_rackets(self, frame):
        """Draws rackets associated with players and their state."""
        for player_id, racket_info in self.player_rackets.items():
            box = racket_info['box']
            state = racket_info.get('state', 'unknown')
            label = f"P{player_id} Racket: {state}"
            
            color = (255, 165, 0) # Orange for associated racket
            if "IMPACT" in state:
                color = (0,0,255) # Red for impact
            elif "APPROACHING" in state:
                color = (0,255,0) # Green for approaching

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame