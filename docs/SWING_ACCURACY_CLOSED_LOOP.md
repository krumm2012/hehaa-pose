# Swing Accuracy Closed Loop

## Pipeline

The current swing review flow is a five-step offline loop:

1. `main_pipe.py` generates frame JSON with pose, ball, racket, and detection diagnostics.
2. `swing_event_analyzer.py` converts frame labels into event-level swings.
3. `swing_coach_data_collector.py` expands each event into coach-oriented metrics.
4. `swing_event_video_renderer.py` and `swing_report_builder.py` produce review artifacts.
5. `swing_evaluation.py` compares model events with human annotation JSON and writes objective accuracy metrics.

This keeps realtime detection separate from event-level review. The event JSON is the model source of truth for counting and classification; human annotation JSON is the review source of truth when available.

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

`*_swing_evaluation.json` is generated only when human annotations are available. V2 uses ordered temporal alignment instead of event ids and reports Precision, Recall, F1, stroke-type accuracy, contact/boundary error, interval IoU, and unmatched events.

## Manual Evaluation

After reviewing the full timeline, adding any missed swings, and downloading `swing_manual_annotations_v2.json` from the report page, run:

```bash
python3 swing_evaluation.py \
  --events data/players-video/results_20260517/03.15_closed_loop_swing_events.json \
  --annotations /Users/krum5539/Downloads/swing_manual_annotations_v2.json
```

The default output path is next to the event JSON, for example:

```text
data/players-video/results_20260517/03.15_closed_loop_swing_evaluation.json
```

Key fields:

- `precision`, `recall`, `f1`: detection quality, available only after `timeline_review_complete` is checked and no annotations remain marked `needs_review`.
- `stroke_type_accuracy`: event type agreement for valid human-labeled swings.
- `contact_accuracy`: ratio of contact frames within the configured tolerance.
- `contact_mean_abs_error_frames`: average absolute contact-frame error.
- `start_mean_abs_error_frames`, `end_mean_abs_error_frames`: boundary errors.
- `event_mean_iou`: temporal overlap between matched model and manual events.
- `manual_review_annotation_ids`: annotations the human reviewer marked for review.
- `model_review_event_ids`: events where quality flags already recommended review.
- `unmatched_model_event_ids`: model events not present in the annotation file.
- `unmatched_annotation_ids`: annotation events not present in model output.

V1 annotation files remain supported for compatibility, but they cannot measure true recall because they are tied to existing model event cards.

## Regression Videos

Primary regression clips:

- `data/players-video/03.15.mp4`
- `data/players-video/18.12.mp4`

Expected artifacts per clip:

- `*_swing_annotated.mp4`
- `*_swing_events.json`
- `*_coach_dataset.json`
- `*_swing_report.html`
- `*_swing_evaluation.json` when manual annotations are available

## Known Limits

Human-labeled truth files are still sparse. Counts like "2 two-handed backhands and 1 forehand" must be treated as review targets until enough annotation files exist across different camera views and players.

Single-camera geometry cannot reliably estimate spin, landing depth, racket face angle, or true 3D body rotation. Those fields remain nullable or confidence-tagged.

Mirror-view handedness is still a camera-specific rule. It is now auditable through quality flags, but future multi-camera or calibrated court metadata would be needed for robust generalization.
