# AI Coach Data Collection Roadmap

Date: 2026-05-01

## Purpose

This document records the current AI tennis-coach data collection status and the recommended next steps. It is intended to guide future iterations after the current event-level swing recognition and coach dataset export work.

Current dataset files:

- `data/output_video_coach_dataset.json`
- `data/output_pipe_coach_dataset.json`

Current schema:

```text
coach_dataset_v1.1
```

Current verified event summary:

```text
event_count = 3
stroke_type_counts = {"Forehand": 3}
```

## Current Pipeline

The current data pipeline uses existing per-frame analysis JSON and does not rerun inference.

```text
pipeline frame json
  -> swing_motion_features.py
  -> swing_event_segmenter.py
  -> swing_event_classifier.py
  -> swing_event_analyzer.py
  -> swing_coach_data_collector.py
  -> coach_dataset_v1.1 json
```

Main files:

- `swing_motion_features.py`: extracts frame-level motion, pose, ball, and racket features.
- `swing_event_segmenter.py`: segments complete swing events.
- `swing_event_classifier.py`: classifies event-level stroke type.
- `swing_event_analyzer.py`: writes event and frame audit outputs.
- `swing_event_video_renderer.py`: renders unified annotated videos.
- `swing_coach_data_collector.py`: exports AI coach dataset JSON.
- `test_swing_motion_pipeline.py`: regression tests for the swing event and coach data pipeline.

## Current Data Availability

For `data/output_video.json`:

```text
total_frames = 250
pose_frames = 249
ball_frames = 114
racket_frames = 118
ball_and_racket_same_frame = 54
```

Per event:

| Event | Frames | Pose | Ball | Racket | Ball+Racket |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 53 | 53 | 26 | 27 | 12 |
| 2 | 53 | 53 | 25 | 27 | 13 |
| 3 | 53 | 52 | 29 | 22 | 11 |

Interpretation:

- Body/pose features are highly usable.
- Ball and racket features are usable but not dense enough for high-confidence physical reconstruction.
- Contact frame estimation is available, but confidence varies because ball and racket are not always detected together near the true contact frame.

## Dataset Structure

Each event in `coach_dataset_v1.1` contains these sections:

- `frames`
- `ball`
- `racket`
- `body`
- `timing`
- `scores`
- `diagnosis_tags`
- `data_quality`
- `classification_evidence`
- `frame_trace`

### Frames

Current fields:

- `start`
- `start_frame`
- `preparation`
- `backswing_peak`
- `contact`
- `contact_frame`
- `contact_source`
- `contact_confidence`
- `peak`
- `peak_frame`
- `follow_through_peak`
- `end`
- `end_frame`

Purpose:

- Provides the event phase anchors needed for coaching analysis.
- `contact_frame` is currently estimated from minimum ball-racket distance, falling back to peak frame when needed.

### Ball

Currently available or partially available:

- `contact_point`
- `contact_confidence`
- `ball_racket_distance_at_contact`
- `ball_speed_at_contact`
- `incoming_ball_speed_avg`
- `outgoing_ball_speed_avg`
- `incoming_angle_deg`
- `outgoing_angle_deg`
- `ball_detection_frames`
- `trajectory_quality`

Reserved but currently missing:

- `estimated_spin`
- `spin_confidence`
- `landing_point`
- `landing_zone`
- `net_clearance_px`
- `bounce_frame`
- `shot_depth`
- `shot_direction`

### Racket

Currently available or partially available:

- `racket_center_at_contact`
- `racket_center_at_peak`
- `racket_speed_at_contact`
- `racket_speed_at_peak`
- `max_racket_speed`
- `max_racket_accel`
- `racket_path_angle_deg`
- `swing_path_type`
- `low_to_high_ratio`
- `racket_detection_frames`
- `racket_continuity_ratio`

Reserved but currently missing:

- `racket_face_angle_deg`
- `racket_face_confidence`
- `racket_lag_at_contact`

### Body

Currently available:

- `body_center_at_contact`
- `contact_point_relative_to_body`
- `right_wrist_relative_to_body`
- `left_wrist_relative_to_body`
- `active_wrist_x_offset_at_contact`
- `two_hand_distance_at_contact`
- `arm_extension_at_contact`
- `arm_extension_at_peak`
- `shoulder_turn_at_contact`
- `hip_shoulder_separation_at_contact`
- `pose_angles_at_contact`
- `pose_angles_at_peak`
- `center_movement_start_to_contact`
- `center_movement_contact_to_end`
- `weight_transfer`
- `balance_state`
- `stance_type`
- `unit_turn_quality`
- `contact_too_close_to_body`
- `late_contact`
- `stance_width_px`
- `stance_width_to_hip_ratio`

### Timing

Currently available:

- `duration_frames`
- `duration_seconds`
- `start_to_contact_frames`
- `contact_to_end_frames`
- `peak_to_contact_offset_frames`
- `phase_durations_frames`
- `start_to_contact_seconds`
- `contact_to_end_seconds`
- `recovery_time_frames`
- `recovery_time_seconds`
- `preparation_timing_quality`
- `tempo_consistency`

### Scores

Currently available:

- `contact_score`
- `racket_speed_score`
- `preparation_score`
- `follow_through_score`
- `power_transfer_score`
- `overall_score`
- `confidence`

These scores are heuristic and should be treated as first-pass coaching features, not final professional evaluation scores.

## Recently Completed Fields

The following 10 fields were added because they are feasible with current single-view video plus pose/ball/racket detections:

- `racket.low_to_high_ratio`
- `racket.swing_path_type`
- `body.weight_transfer`
- `body.balance_state`
- `body.stance_type`
- `body.unit_turn_quality`
- `body.contact_too_close_to_body`
- `body.late_contact`
- `timing.recovery_time_frames`
- `timing.tempo_consistency`

These fields no longer appear in `data_quality.missing_fields`.

## Remaining Missing Fields

Current aggregate missing fields in `output_video_coach_dataset.json`:

```text
ball.bounce_frame
ball.estimated_spin
ball.landing_point
ball.landing_zone
ball.net_clearance_px
ball.shot_depth
ball.shot_direction
ball.spin_confidence
racket.racket_center_at_peak
racket.racket_face_angle_deg
racket.racket_face_confidence
racket.racket_lag_at_contact
```

## Feasibility Analysis

### Feasible Next With Current Data

These can be attempted without new models, but should include confidence values:

1. `ball.shot_direction`

Current feasibility: medium

Method:

- Use outgoing ball trajectory after `contact_frame`.
- Estimate direction from post-contact ball positions.
- Output values such as `left`, `right`, `straight`, `unknown` in screen coordinates.

Limitations:

- Ball is visible in only about half the event frames.
- Direction is screen-relative unless court calibration is added.

2. `ball.bounce_frame`

Current feasibility: low to medium

Method:

- Detect local y-axis reversal or sudden speed/angle change in ball trajectory.
- Only attempt when there are enough post-contact ball points.

Limitations:

- Current camera angle and intermittent ball detections make bounce detection noisy.
- Should return `null` with low confidence if not enough points.

3. `racket.racket_center_at_peak`

Current feasibility: medium

Method:

- If racket is missing exactly at `peak_frame`, interpolate from nearest detected racket centers before and after peak.
- Add `racket_center_at_peak_source`: `detected`, `interpolated`, or `missing`.

Limitations:

- Event 1 and Event 3 currently miss exact racket center at peak.
- Interpolation should not be treated as equal to detection.

4. `racket.racket_lag_at_contact`

Current feasibility: medium

Method:

- Estimate distance between active wrist and racket center at contact.
- Add confidence based on whether racket center is detected or interpolated.

Limitations:

- Requires reliable active-hand selection and racket center continuity.

### Requires Court Calibration

These should wait until court geometry is available:

1. `ball.landing_point`
2. `ball.landing_zone`
3. `ball.net_clearance_px`
4. `ball.shot_depth`

Required additions:

- Court line detection or manually configured court points.
- Homography from image coordinates to court coordinates.
- Net line position.
- A calibrated definition of service boxes, baseline, sidelines, and depth zones.

Without calibration, these can only be screen-space approximations. They should not be used as real tennis court metrics.

### Requires Better Racket/Ball Models Or Extra Sensors

These are not reliable with current single-view 25 FPS detections:

1. `ball.estimated_spin`
2. `ball.spin_confidence`
3. `racket.racket_face_angle_deg`
4. `racket.racket_face_confidence`

Why:

- Spin needs ball texture/blur analysis, high frame rate, trajectory curvature, or specialized model support.
- Racket face angle requires more than a bounding box. It needs keypoints, segmentation, oriented bounding box, or a racket pose estimator.

## Recommended Implementation Order

### Phase 1: Improve Current Single-View Dataset

Goal: Fill the low-risk missing fields that can be estimated from current data.

Tasks:

1. Add post-contact `shot_direction` with confidence.
2. Add interpolated `racket_center_at_peak` and source metadata.
3. Add `racket_lag_at_contact` with confidence.
4. Add conservative `bounce_frame` candidate with confidence.
5. Expand `data_quality` so every estimated field records source and confidence.

Acceptance criteria:

- All new fields have tests.
- Low-confidence estimates are explicitly marked.
- No field silently pretends to be physically accurate when it is screen-space only.

### Phase 2: Court Calibration

Goal: Make ball landing and depth metrics meaningful.

Tasks:

1. Add court calibration config.
2. Support manual court keypoints first.
3. Add homography transform.
4. Convert ball pixels to court-space coordinates.
5. Compute `landing_point`, `landing_zone`, `shot_depth`, and `net_clearance_px`.

Acceptance criteria:

- Calibration can be saved and reused per camera setup.
- JSON records calibration version and confidence.
- Court-space metrics are disabled if calibration is missing.

### Phase 3: Racket Pose Upgrade

Goal: Estimate racket face and swing mechanics more accurately.

Tasks:

1. Evaluate oriented racket detection or segmentation.
2. Add racket keypoints or mask fitting.
3. Estimate racket long axis and face angle proxy.
4. Add confidence based on detection quality.

Acceptance criteria:

- `racket_face_angle_deg` is produced only when model confidence is sufficient.
- Poor detections remain `null` and appear in `missing_fields`.

### Phase 4: Spin Estimation Research

Goal: Decide whether spin is feasible from the available footage.

Tasks:

1. Test whether the ball is large/sharp enough near contact.
2. Explore trajectory-curvature spin proxy.
3. Evaluate high-FPS video requirement.
4. Decide whether to keep spin fields reserved or add a specialized model.

Acceptance criteria:

- Spin is not exposed as a coaching signal unless validated against known examples.

## Suggested Schema Improvements

Add per-field source and confidence for estimated values.

Example:

```json
{
  "racket_center_at_peak": [1611.5, 44.0],
  "racket_center_at_peak_source": "detected",
  "racket_center_at_peak_confidence": 0.9
}
```

For screen-space estimates:

```json
{
  "shot_direction": "right",
  "shot_direction_space": "screen",
  "shot_direction_confidence": 0.62
}
```

For court-space estimates after calibration:

```json
{
  "landing_point": {"x_m": 3.2, "y_m": 17.4},
  "landing_zone": "deep_crosscourt",
  "coordinate_space": "court",
  "calibration_id": "court_01_v1"
}
```

## Current Commands

Regenerate coach dataset:

```bash
python3 swing_coach_data_collector.py data/output_video.json
python3 swing_coach_data_collector.py data/output_pipe.json
```

Run tests:

```bash
python3 -m unittest -v test_swing_motion_pipeline.py test_swing_presence_detector.py test_detection_frame_context.py test_highlight_service.py
```

Syntax check:

```bash
python3 -m py_compile swing_coach_data_collector.py test_swing_motion_pipeline.py
```

## Current Verification

Latest verification:

```text
23 tests OK
```

## Notes For Future AI Coach Use

The current dataset is useful for early AI coach features such as:

- swing count
- event segmentation
- forehand/backhand classification for the current calibrated viewpoint
- contact confidence
- relative contact position
- preparation timing
- follow-through completeness
- stance width
- balance proxy
- weight-transfer proxy
- swing-path proxy

It is not yet sufficient for high-confidence claims about:

- exact landing point
- true shot depth
- net clearance
- spin
- racket face angle
- real-world speed in km/h

These should remain disabled or clearly marked as low-confidence until calibration/model support is added.
