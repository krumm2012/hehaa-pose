# Swing Event Recognition Change Log

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
- `active_wrist_x_offset`
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
- Keeps fallback energy-island segmentation for short or synthetic inputs.
- Adds `peak_frame`, `phase_counts`, and per-frame `phase` traces.

3. Event classification

File: `swing_event_classifier.py`

Classifies an entire event instead of directly counting frame labels.

Current behavior:

- Treats frame labels as evidence, not ground truth.
- Uses a core window near the peak for primary classification.
- Uses post-impact two-hand evidence cautiously.
- Requires backhand support before accepting `Two-Handed Backhand` from two-hand proximity.
- Applies the current-camera handedness rule: image-left true-right evidence can override false two-handed backhand classification.

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
