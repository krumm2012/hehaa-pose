# main.py
# ... (other imports)
from racket_detector import RacketDetector
from full_swing_analyzer import FullSwingAnalyzer # <-- IMPORT NEW ANALYZER
import time

# ... (load_config, overlay_image_alpha, replace_player_head functions remain the same) ...

def main():
    # ... (config loading, image loading, video capture setup) ...
    print("开始加载配置...")
    config = load_config()
    video_path = config['video_input_path']
    replacement_head_path = config.get('judy_head_image_path')
    replacement_head_scale = config.get('judy_head_scale_factor', 1.5)

    replacement_image = None
    if replacement_head_path and os.path.exists(replacement_head_path):
        replacement_image = cv2.imread(replacement_head_path, cv2.IMREAD_UNCHANGED)
        # ... (image loading print statements)
    # ... (video capture setup, frame dimensions, output video writer)

    print("初始化姿势估计模块...")
    pose_module = PoseEstimator(config['yolo_pose_model_path'], config)
    print("初始化球追踪(KF)模块...")
    ball_module = BallTracker(config.get('tracknet_model_path', None), config)
    print("初始化球拍检测模块...")
    racket_module = RacketDetector(config['racket_yolo_model_path'], config)
    print("初始化完整挥拍分析模块...") # <-- NEW
    swing_analyzer = FullSwingAnalyzer(config) # <-- NEW

    frame_num = 0
    start_time = time.time()
    # ... (main loop starts) ...
    while True:
        ret, frame = cap.read()
        if not ret: print("视频处理完成!"); break

        display_frame = frame.copy()
        frame_dimensions = (frame_height, frame_width) # Pass to analyzer

        # 1. Pose Estimation
        person_keypoints_list = pose_module.get_keypoints(frame)

        # Optional: Head Replacement (no changes here)
        if replacement_image is not None:
            display_frame = replace_player_head(display_frame, person_keypoints_list, replacement_image, replacement_head_scale)
        if replacement_image is None or not person_keypoints_list :
             display_frame = pose_module.draw_keypoints(display_frame, person_keypoints_list)

        # 2. Swing Classification (basic type from pose_estimator)
        basic_swing_type = "No Swing"
        if person_keypoints_list:
            basic_swing_type = pose_module.classify_swing(person_keypoints_list)

        # 3. Racket Detection & Association
        raw_racket_detections = racket_module.detect_rackets(frame)
        associated_rackets_info = racket_module.associate_racket_to_player(raw_racket_detections, person_keypoints_list, frame_num)

        # 4. Ball Tracking
        raw_ball_detections = ball_module.predict_ball(frame)
        kf_estimated_ball_pos = ball_module.process_frame(raw_ball_detections, frame_num)

        # 5. Determine Racket State (based on ball, from racket_detector)
        player_id_for_state = 0 # Assuming player 0
        if player_id_for_state in associated_rackets_info:
            racket_module.determine_racket_state(player_id_for_state, kf_estimated_ball_pos, frame_height)

        # 6. Perform Full Swing Analysis <-- NEW
        full_analysis_metrics = {}
        if person_keypoints_list: # Only analyze if a person is detected
            full_analysis_metrics = swing_analyzer.analyze_swing_components(
                person_keypoints_list,
                associated_rackets_info, # Pass the associated racket info
                kf_estimated_ball_pos,
                frame_dimensions
            )

        # 7. Annotation
        # Draw associated rackets and their states
        display_frame = racket_module.draw_associated_rackets(display_frame)
        # Draw ball and trajectory
        display_frame = ball_module.draw_ball(display_frame, kf_estimated_ball_pos)
        display_frame = ball_module.draw_trajectory(display_frame)
        display_frame = ball_module.draw_static_balls(display_frame)

        # Display Text Info
        text_y_offset = 30
        line_height = 25

        cv2.putText(display_frame, f"Basic Swing: {basic_swing_type}", (text_y_offset, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        current_y = 50 + line_height

        if kf_estimated_ball_pos:
            cv2.putText(display_frame, f"Ball: ({int(kf_estimated_ball_pos[0])}, {int(kf_estimated_ball_pos[1])})", (text_y_offset, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2); current_y += line_height
        else:
            cv2.putText(display_frame, "Ball: Lost", (text_y_offset, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2); current_y += line_height

        if player_id_for_state in racket_module.player_rackets:
             cv2.putText(display_frame, f"Racket State: {racket_module.player_rackets[player_id_for_state]['state']}", (text_y_offset, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,0), 2); current_y += line_height
        
        # Display Full Swing Analysis Metrics <-- NEW
        cv2.putText(display_frame, "--- Full Swing Analysis ---", (text_y_offset, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,200), 2); current_y += line_height
        
        phase_est = full_analysis_metrics.get("phase_estimation", "N/A")
        cv2.putText(display_frame, f"Est. Phase: {phase_est}", (text_y_offset, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1); current_y += (line_height -5)


        for category, cat_metrics in full_analysis_metrics.items():
            if category == "phase_estimation": continue # Already displayed
            if isinstance(cat_metrics, dict) and cat_metrics:
                cv2.putText(display_frame, f"[{category.upper()}]:", (text_y_offset, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1); current_y += (line_height -5)
                for key, value in cat_metrics.items():
                    if current_y > frame_height - 20 : break # Avoid drawing off screen
                    display_text = f"  {key.replace('_', ' ').title()}: {value}"
                    cv2.putText(display_frame, display_text, (text_y_offset + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1); current_y += (line_height - 7)
            if current_y > frame_height - 20 : break
        
        # FPS display
        current_fps_val = (frame_num + 1) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        cv2.putText(display_frame, f"FPS: {current_fps_val:.1f}", (frame_width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Tennis Analysis", display_frame)
        if out: out.write(display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): print("用户按下'q'键，程序退出"); break
        frame_num += 1
    # ... (loop ends, resource release) ...
    end_time = time.time(); total_processing_time = end_time - start_time
    avg_fps = frame_num / total_processing_time if total_processing_time > 0 else 0
    print(f"处理结束. 总耗时: {total_processing_time:.2f} 秒. 平均 FPS: {avg_fps:.2f}")
    cap.release();
    if out: out.release()
    cv2.destroyAllWindows()
    print("程序结束")

if __name__ == "__main__":
    main()