# Swing Accuracy Closed Loop

## Pipeline

The current swing review flow is a four-step offline loop:

1. `main_pipe.py` generates frame JSON with pose, ball, racket, and detection diagnostics.
2. `swing_event_analyzer.py` converts frame labels into event-level swings.
3. `swing_coach_data_collector.py` expands each event into coach-oriented metrics.
4. `swing_event_video_renderer.py` and `swing_report_builder.py` produce review artifacts.

This keeps realtime detection separate from event-level review. The event JSON is the source of truth for counting and classification; the video and report are audit surfaces.

## Accuracy Risks Addressed

The event analyzer now exports `quality_flags` for every swing event:

- `pose_frame_ratio`
- `ball_frame_ratio`
- `racket_frame_ratio`
- `diagnostic_rejection_counts`
- `continuity_disabled_frames`
- `warnings`
- `review_recommended`

These flags make weak events visible instead of hiding them behind a single `confidence` score.

## Key Heuristics

Contact frame selection uses the highest `contact_score`, with a tie-break toward the frame closest to the event peak. This reduces early-contact bias when multiple frames have the same ball-racket distance score.

Event warnings are attached when the event has ball gaps, racket gaps, pose gaps, mirror handedness overrides, static-ball mask activity, mirror-ball rejections, or continuity disablement. These are not automatic failures; they are review prompts.

## JSON Outputs

`*_swing_events.json` now carries event-level quality flags.

`*_coach_dataset.json` copies those flags into each coach event and folds warnings into `diagnosis_tags`, so future AI coach prompts can explain why a recommendation is low or high confidence.

`*_swing_report.html` combines the swing annotated video, event timeline, score summary, diagnosis tags, quality warnings, and a compact JSON summary in one standalone page.

## Regression Videos

Primary regression clips:

- `data/players-video/03.15.mp4`
- `data/players-video/18.12.mp4`

Expected artifacts per clip:

- `*_swing_annotated.mp4`
- `*_swing_events.json`
- `*_coach_dataset.json`
- `*_swing_report.html`

## Known Limits

There is still no human-labeled truth file. Counts like "2 two-handed backhands and 1 forehand" must be treated as review targets until a label file exists.

Single-camera geometry cannot reliably estimate spin, landing depth, racket face angle, or true 3D body rotation. Those fields remain nullable or confidence-tagged.

Mirror-view handedness is still a camera-specific rule. It is now auditable through quality flags, but future multi-camera or calibrated court metadata would be needed for robust generalization.
