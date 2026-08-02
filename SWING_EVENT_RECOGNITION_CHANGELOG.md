# Swing Event Recognition Change Log

## 2026-08-02 — Realtime overlap suppression and overlay re-recognition

- Aligned realtime peak duplicate suppression with the segmenter's 1.6-second minimum peak distance.
- Limited rolling analysis to frames after the latest immutable published event and defensively rejected intersecting event ranges.
- Added regression coverage for the observed `33–98` / `62–131` overlap while preserving a later non-overlapping Swing.
- Added opt-in recovery of analyzer-owned hollow ball markers and current/legacy racket rectangles for videos that are analyzed again.
- Kept overlay provenance in frame diagnostics and passed recovered candidates through existing ball/racket temporal selection.
- Added raw per-class model confidence and threshold diagnostics for both ball and racket calibration.
- Rejected small filled yellow balls, long ROI lines, and open pose strokes from overlay recovery.
- Verified 250 RTSP frames at approximately 25 FPS with four non-overlapping Swing ranges.

## 2026-08-02 — Session quality and drift dashboard

- Added the pure `build_session_quality_dashboard(events)` interface shared by offline analysis,
  realtime JSON, and both HTML reports.
- Separated visible-technique trend from capture/evidence quality so detection degradation is not
  presented as player regression.
- Added per-Swing trend series, recurring warning/advice counts, contact support, Coach latency,
  DeepSeek availability, and first-window versus recent-window indicators.
- Requires at least six Swing events before making a drift conclusion.
- Treats a 15% camera-scale change as a confounder and overlapping event ranges as an integrity
  blocker for technique drift.
- Verified the dashboard against 07.20, 16.10, and the local RTSP input at
  `rtsp://127.0.0.1:8554/input-video`.

## 2026-08-01 — Single-view biomechanics and Coach calibration

- Added one shared `single_view_visible_coach_v1` calibration policy for offline analysis,
  realtime local Coach, reports, overlays, and the DeepSeek sidecar.
- Added directly reviewable 2D metrics for shoulder-turn change and preparation knee flexion.
- Marked projected hip–shoulder separation, screen-space body translation, and translation-only
  balance as non-coaching proxies with explicit exclusion reasons.
- Stopped treating absent contact/racket evidence as a zero technique score.
- Split the 0–9 visible-technique score from event classification and capture quality, and added
  score uncertainty plus metric-level provenance.
- Gated preparation/follow-through advice on their own boundary/contact evidence confidence.
- Removed excluded numeric biomechanics and projected shoulder/hip angles from DeepSeek model
  evidence; the sidecar now receives only exclusion reasons and permitted advice candidates.
- Verified 07.20 as three Forehands with visible scores `6.69 / 3.80 / 5.72` (mean `5.40/9`),
  and preserved the three-Forehand result on 16.10.

Date: 2026-05-01

## Summary

This iteration redesigned swing counting from frame-label counting into an auditable event-level pipeline. The goal was to make swing recognition more precise, avoid counting follow-through/recovery as additional swings, and generate unified annotated videos plus frame-level records for review.

Final verified result on the current fixture `data/output_video.json` and `data/output_pipe.json`:

- Total swing events: 3
- Event types: 3 Forehand
- Important handedness rule: in this camera view, the image-left side corresponds to the player's real right-hand forehand side.

## Problem Background

The earlier recognition path was unstable for precise counting because it leaned too heavily on per-frame static labels such as `Forehand`, `Backhand`, and `Two-Handed Backhand`.

Observed issues:

- Follow-through and recovery frames were sometimes treated as new swing evidence.
- Static frame labels changed during one continuous action, causing one stroke to contain mixed labels.
- Two-hand proximity during recovery was incorrectly used as strong two-handed backhand evidence.
- The camera view flips the intuitive interpretation: image-left corresponds to the player's real right-hand forehand side.
- As a result, two events were previously misclassified as `Two-Handed Backhand`.

## Root Cause

The root cause was not only a threshold problem. The event classifier inherited noisy frame-level labels and then allowed post-impact/follow-through two-hand evidence to override the event type.

For the current video, event 1 and event 3 both had right-hand forehand evidence when interpreting image-left as the player's true right-hand side. However, later frames in each event showed hands moving close together and `Backhand`/`Two-Handed Backhand` frame labels, so the previous logic changed the event type to `Two-Handed Backhand`.

## New Architecture

The redesigned pipeline is split into four auditable layers.

1. Frame feature extraction

File: `swing_motion_features.py`

Extracts per-frame motion and geometry from the existing pipeline JSON. It does not run model inference.

Key fields:

- `wrist_speed`
- `wrist_accel`
- `racket_speed`
- `racket_accel`
- `ball_speed`
- `ball_racket_distance`
- `contact_score`
- `two_hand_distance`
- `two_hand_distance_body_width`
- `active_wrist_x_offset`
- `active_wrist_x_offset_body_width`
- `camera_facing_score`
- `arm_extension_deg`
- `shoulder_turn_deg`
- `hip_shoulder_sep_deg`

2. Event segmentation

File: `swing_event_segmenter.py`

Segments full swing events from the feature timeline.

Current behavior:

- Uses smoothed motion energy to find candidate swing peaks.
- Uses robust peak selection for long videos.
- Builds full event windows around peak frames.
- Refines `start_frame` from a stable quiet basin plus sustained motion/shoulder-turn onset instead of a fixed pre-peak offset.
- Emits `recovery_ready_transition` when recovery and preparation overlap without a trustworthy static ready frame.
- Makes phase labels contact-relative so `follow_through` cannot precede contact and `backswing` cannot follow it.
- Keeps fallback energy-island segmentation for short or synthetic inputs.
- Adds `peak_frame`, `phase_counts`, and per-frame `phase` traces.

3. Event classification

File: `swing_event_classifier.py`

Classifies an entire event instead of directly counting frame labels.

Current behavior:

- Treats frame labels as evidence, not ground truth.
- Uses a contact-centred window that excludes ready/recovery hand proximity.
- Records the configured player dominant hand instead of trying to infer identity from one swing.
- Infers whether the player faces the camera or faces away from anatomical shoulder projection.
- Maps the dominant-side wrist to forehand/backhand only after normalizing for that camera orientation.
- Normalizes wrist separation by shoulder width so two-hand evidence is stable across player distance and resolution.
- Requires both backhand-side and two-hand support before accepting `Two-Handed Backhand`.
- Emits auditable `evidence.classification_context` with player, camera, swing-side, and decision-rule evidence.

4. Offline analysis and video rendering

Files:

- `swing_event_analyzer.py`
- `swing_event_video_renderer.py`

Responsibilities:

- Read existing frame JSON output.
- Generate event-level JSON and CSV.
- Generate frame-level audit CSV.
- Render unified annotated videos from the original source video without rerunning inference.

## Changed Files

New files:

- `swing_motion_features.py`
- `swing_event_classifier.py`
- `swing_event_segmenter.py`
- `swing_event_analyzer.py`
- `swing_event_video_renderer.py`

Updated tests:

- `test_swing_motion_pipeline.py`

Generated outputs:

- `data/output_video_swing_events.json`
- `data/output_video_swing_events.csv`
- `data/output_video_swing_frames.csv`
- `data/output_video_swing_overlay_records.csv`
- `data/output_video_swing_annotated.mp4`
- `data/output_pipe_swing_events.json`
- `data/output_pipe_swing_events.csv`
- `data/output_pipe_swing_frames.csv`
- `data/output_pipe_swing_overlay_records.csv`
- `data/output_pipe_swing_annotated.mp4`

## Current Result

For both `output_video` and `output_pipe` frame JSON:

| Event | Frame Range | Peak Frame | Type | Notes |
| --- | ---: | ---: | --- | --- |
| 1 | 25-77 | 46 | Forehand | Corrected from previous false two-handed backhand |
| 2 | 97-149 | 118 | Forehand | Kept as forehand |
| 3 | 189-241 | 210 | Forehand | Corrected from previous false two-handed backhand |

Summary:

```text
swing_event_count = 3
swing_event_type_counts = {"Forehand": 3}
```

## Unified Video Overlay

The annotated videos now use one consistent event display format.

Overlay fields:

- `Event <id>/<total>: <type>`
- `Frame`
- `Phase`
- `Range`
- `Peak`
- `raw_label`
- `energy`
- `wrist`
- `racket`
- `2Hdist`
- `contact`
- `ball-racket`
- event progress bar
- peak frame border
- pose skeleton from saved JSON
- ball marker
- racket box

Output videos:

- `data/output_video_swing_annotated.mp4`
- `data/output_pipe_swing_annotated.mp4`

Both verified as:

```text
frames = 250
fps = 25.14
resolution = 2560x1440
```

## Commands

Run event analysis:

```bash
python3 swing_event_analyzer.py data/output_video.json
python3 swing_event_analyzer.py data/output_pipe.json
```

Render annotated videos:

```bash
python3 swing_event_video_renderer.py data/output_video.json
python3 swing_event_video_renderer.py data/output_pipe.json
```

Run regression tests:

```bash
python3 -m unittest -v test_swing_motion_pipeline.py test_swing_presence_detector.py test_detection_frame_context.py test_highlight_service.py
```

Syntax check:

```bash
python3 -m py_compile swing_event_segmenter.py swing_event_analyzer.py swing_event_video_renderer.py test_swing_motion_pipeline.py
```

## Verification

Latest verification:

```text
21 tests OK
```

Additional output validation:

```text
data/output_video_swing_annotated.mp4: 250 frames, 25.14 FPS, 2560x1440
data/output_pipe_swing_annotated.mp4: 250 frames, 25.14 FPS, 2560x1440
```

## Important Assumptions

This iteration encodes a camera-specific handedness rule:

```text
image-left = player's true right-hand forehand side
```

This rule is correct for the current analyzed video. For other camera angles, this should become a configurable setting instead of a hard-coded assumption.

## Known Limitations

- The current event type result is calibrated against the current fixture and camera orientation.
- The handedness rule should be moved into configuration before batch processing many videos with different camera setups.
- Ball-racket contact is still not always reliable because ball/racket detections can be missing around the peak frame.
- Current video rendering is an offline post-process. It does not yet integrate directly into `main_pipe.py` realtime output.

## Recommended Next Iterations

1. Add a config option for camera handedness mapping.

Example:

```yaml
swing_event_analysis:
  true_right_hand_screen_side: left
```

2. Integrate event analysis into `main_pipe.py` output mode after frame JSON generation.

3. Add more fixtures from different camera angles to avoid overfitting to this single video.

4. Improve contact confidence by combining ball trajectory, racket box continuity, and peak frame proximity.

5. Add a small event review HTML or CSV viewer to inspect `peak_frame`, `phase`, and feature evidence quickly.
