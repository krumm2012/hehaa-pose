# Swing GPT-5.5 Review

## Status

GPT-5.5 reviewer agent unavailable in this execution.

The current tool policy only allows spawning sub-agents when the user explicitly asks for sub-agent delegation or parallel agent work. The request asked for GPT-5.5 analysis advice as part of the closed loop, but did not explicitly authorize a separate sub-agent execution. To avoid pretending an external review happened, this document records the local review instead.

## Local Review Findings

1. Event-level counting is the right source of truth. Frame labels are too noisy for final swing counts.
2. Contact frame selection should prefer ball-racket evidence but avoid choosing the first tied frame. This was changed to prefer the high-score frame nearest the event peak.
3. Accuracy should not be expressed as a single confidence value. Event-level `quality_flags` now expose pose, ball, racket, and detector-diagnostic weaknesses.
4. AI coach JSON should preserve uncertainty. `coach_dataset` now carries event quality warnings into both `quality_flags` and `diagnosis_tags`.
5. Human review needs a visual page. `swing_report_builder.py` creates a standalone report combining video, event timeline, scores, warnings, and JSON summary.

## Adoption

Adopted:

- Add event-level `quality_flags`.
- Add detector-diagnostic rejection counts to event quality.
- Improve contact frame tie-breaking.
- Add standalone HTML report.
- Document the lack of human truth labels as a remaining accuracy limit.

Not adopted yet:

- Train a learned swing classifier.
- Add multi-camera 3D reconstruction.
- Add external online model review.

## Next Review Hook

When explicit GPT-5.5 reviewer delegation is available, ask it to review:

- `swing_event_segmenter.py`
- `swing_event_classifier.py`
- `swing_coach_data_collector.py`
- `swing_report_builder.py`
- Two generated reports for `03.15.mp4` and `18.12.mp4`
