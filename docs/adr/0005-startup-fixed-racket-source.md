# ADR 0005: Select the racket source at startup

- Status: Accepted
- Date: 2026-10-18

## Context

The user requested two startup choices, with the new profile using only seven-
point racket observations and no per-frame switch back to legacy boxes. Legacy
boxes previously entered both racket arbitration and ball-support ranking.
The shared ball/racket model is still needed for ball inference; ignoring its
racket class does not remove its forward pass. The racket package has one class
and seven points, and cannot replace the separate human pose model.

## Decision

1. Keep `legacy` as the default. Add the startup choice `exclusive`, resolved to
   `active` plus `selection_policy: exclusive`. Keep `shadow` and mixed `active`
   only as backward-compatible diagnostic choices. No in-session source switch.
2. In the exclusive profile, the unified detector returns ball candidates and
   ignores legacy racket boxes before stateful ball selection. Complete ball
   selection once new observations arrive, with only current new-pose support.
3. Apply the existing same-frame, qualified-wrist/grip gate independently to
   front and back source-coordinate observations, retaining its confidence and
   distance thresholds. Give qualified candidates to the existing body/temporal
   resolver; a back-associated candidate cannot become a direct front detection. Keep raw source identity and retain
   mirror recovery/smoothing as derived, unobserved output with new provenance.
4. An empty result stays missing or explicitly derived. Never insert a legacy
   box. Assert source purity and current identity before publishing observations.
5. Require buffered/offline processing. Reject sampling realtime mode. Worker
   failure or timeout fails the exclusive session; legacy can be started anew.
6. Preserve Reader/Inference/Analyzer/Swing Runtime, body/ball models, source PTS,
   recording, reports, Coach, clips, and replay. Keep the legacy-racket field as
   an empty compatibility list for exclusive records. Audit racket and ball-
   support provenance independently from throughput and detection accuracy.

## Consequences

The profiles separate racket sources without retraining or replacing the
workflow. Four model calls per dual-view frame remain. No real-time performance
gain or accuracy improvement is implied by removing legacy candidate mixing.
Performance needs same-load measurements; preserve slow runs and do not use
sampling to hide backlog. See the startup-profile acceptance report.
