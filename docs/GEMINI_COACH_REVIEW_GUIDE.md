# Gemini Coach Review Guide

This document explains how to use the generated swing video, report page, and JSON files for a professional tennis-coach review in Gemini.

## Purpose

Use Gemini as a professional tennis coach reviewer, not as the source of truth for detection. The local pipeline provides event segmentation, pose/ball/racket metrics, and quality warnings. Gemini should use those artifacts to produce a coaching evaluation, highlight uncertainty, and recommend what to fix first.

## Recommended Files

For each reviewed video, provide Gemini with:

1. `*_swing_annotated.mp4`
2. `*_swing_report.html`
3. `*_swing_events.json`
4. `*_coach_dataset.json`
5. Optional: `swing_manual_annotations.json` if human review has already corrected event labels.

Current review artifacts:

### 03.15

- Video: `data/players-video/results_20260517/03.15_closed_loop_swing_annotated.mp4`
- Report: `data/players-video/results_20260517/03.15_closed_loop_swing_report.html`
- Event JSON: `data/players-video/results_20260517/03.15_closed_loop_swing_events.json`
- Coach JSON: `data/players-video/results_20260517/03.15_closed_loop_coach_dataset.json`

### 18.12

- Video: `data/players-video/results_20260517/18.12_closed_loop_swing_annotated.mp4`
- Report: `data/players-video/results_20260517/18.12_closed_loop_swing_report.html`
- Event JSON: `data/players-video/results_20260517/18.12_closed_loop_swing_events.json`
- Coach JSON: `data/players-video/results_20260517/18.12_closed_loop_coach_dataset.json`

## How To Read The Report

Open `*_swing_report.html` locally in a browser.

The report contains:

- Video player: annotated swing video.
- Event cards: one card per detected swing event.
- Event frame range: `start`, `contact`, `peak`, `end`.
- Scores: overall score, contact score, preparation score, follow-through score.
- Diagnosis tags: automatic issue labels.
- Quality warnings: detection confidence problems that should affect trust.
- Manual annotation controls: dropdown and checkboxes for human correction.
- Manual annotation JSON: generated label data that can be downloaded.

## Important Quality Flags

Gemini should treat these as uncertainty signals:

- `ball_track_gaps`: ball tracking is incomplete inside the event.
- `racket_track_gaps`: racket tracking is incomplete inside the event.
- `pose_gaps`: pose is missing or unstable inside the event.
- `static_ball_mask_in_event`: static-ball filtering affected this event.
- `mirror_ball_rejection_in_event`: mirror/interference ball candidates were rejected.
- `mirror_handedness_rule_applied`: stroke type used the current mirror-view handedness rule.
- `ball_continuity_disabled`: ball continuity tracking was disabled for part of the event.
- `contact_frame_needs_review`: contact frame may need human review.

When these flags appear, Gemini should phrase coaching conclusions as lower confidence.

## What Gemini Should Evaluate

Ask Gemini to evaluate:

- Stroke type correctness: forehand, backhand, two-handed backhand, unclear.
- Preparation: unit turn, backswing timing, stance width, early racket preparation.
- Contact: contact position relative to body, late/early contact, spacing from body.
- Swing path: low-to-high path, racket lag, acceleration, follow-through.
- Balance and recovery: center movement, finish stability, recovery time.
- Ball/racket evidence: whether the ball and racket tracking support the coaching conclusion.
- Priority fixes: one or two changes that matter most for the next training session.

## Universal Gemini Prompt Template

Use this generic prompt for any student video. Replace the bracketed fields before sending it to Gemini.

```text
You are a professional tennis coach and video-analysis reviewer.

Context:
- Player level: [beginner / intermediate / advanced / unknown]
- Player handedness: [right-handed / left-handed / unknown]
- Camera note: [front view / side view / mirror view / unknown]
- Review goal: [count swings / technique feedback / training plan / competition review]
- Student focus, if any: [forehand / backhand / two-handed backhand / footwork / contact point / all strokes]

I will provide:
1. An annotated swing video.
2. A swing report HTML file, if available.
3. A swing event JSON file, if available.
4. A coach dataset JSON file, if available.
5. Optional human annotation JSON, if available.

Please evaluate the player's technique event by event.

Rules:
- The video is the primary visual evidence.
- Treat the event JSON and coach dataset as structured evidence, but verify visually against the video.
- If manual annotations are provided, prefer manual labels over model predictions.
- Use quality_flags as uncertainty signals. If ball/racket/pose tracking is weak, say the conclusion is low confidence.
- Do not invent 3D measurements, spin, landing depth, or racket face angle if the JSON says the field is null, unavailable, or low confidence.
- If the video and JSON disagree, explicitly list the disagreement and explain what should be reviewed manually.
- Separate coaching advice from data-quality concerns. Do not over-criticize the player when the detection evidence is weak.
- Give practical, court-ready advice suitable for the player's level.

For each swing event, output:
1. Event ID and predicted/manual stroke type.
2. Whether the event should count as a real swing.
3. Contact quality: early/late/too close/good spacing/unclear.
4. Preparation quality: unit turn, stance, backswing timing.
5. Swing path and follow-through quality.
6. Balance/recovery quality.
7. Data confidence: high/medium/low, with reasons from quality_flags.
8. One coaching cue for the next repetition.

Then produce:
- Overall summary.
- Top 3 technical issues.
- Top 3 training recommendations.
- A short coach-style feedback paragraph suitable for the student.
- A list of events that need manual review.
```

## Short Prompt Template

Use this shorter version when Gemini already has the files and you want a faster answer:

```text
Act as a professional tennis coach. Review the uploaded annotated swing video, report, event JSON, coach JSON, and optional manual annotations.

Evaluate each swing event for stroke type, whether it should count, contact quality, preparation, swing path, follow-through, balance, and recovery.

Use manual annotations over model predictions when provided. Use quality_flags to judge data confidence. If pose, ball, or racket tracking is weak, mark that event as lower confidence and avoid overclaiming.

Return:
1. Overall assessment.
2. Event-by-event review.
3. Top 3 technical issues.
4. Top 3 training recommendations.
5. One short coach-style paragraph for the student.
6. Events needing manual review.
```

## Suggested Output Format

Ask Gemini to respond in this structure:

```text
## Overall Assessment

## Event Review
### Event 1
- Stroke:
- Count as real swing:
- Contact:
- Preparation:
- Swing path:
- Balance/recovery:
- Data confidence:
- Coaching cue:

## Top Issues
1.
2.
3.

## Training Recommendations
1.
2.
3.

## Student Feedback

## Manual Review Needed
```

## Human Annotation Workflow

1. Open `*_swing_report.html`.
2. For each event, set the actual stroke type.
3. Check or uncheck `count_correct`, `valid_hit`, and `needs_review`.
4. Select issue tags when needed.
5. Add short notes only when checkboxes are not enough.
6. Click `下载标注 JSON`.
7. Give Gemini the downloaded `swing_manual_annotations.json` together with the video and JSON files.

## Current Known Limitations

- There is no full human-labeled ground-truth dataset yet.
- Single-view camera analysis cannot reliably measure true 3D rotation, spin, landing depth, or racket face angle.
- Mirror-view handedness is camera-specific.
- If ball or racket quality flags are present, contact-point coaching should be treated as medium or low confidence.

## Best Practice

Use Gemini for coaching interpretation and narrative feedback. Use the local JSON and human annotations for event counting, frame references, and auditability.
