# ADR 0001: Versioned session data and asynchronous Swing runtime

- Status: Accepted
- Date: 2026-07-29

## Context

The live pipeline previously treated plain dictionaries as implicit FrameRecord
and SwingEvent contracts. Swing analysis, Coach generation, JSON/HTML updates,
and OSD rendering also shared the Analyzer frame loop. A malformed
biomechanics value could therefore terminate the Analyzer, leave Reader
backpressured, and still allow the parent process to report success.

Long-running sessions also rewrote event snapshots without an append-only
history, used nominal frame timestamps instead of processing timestamps, and
could reuse output directories across separate camera sessions.

## Decision

1. Keep all legacy fields and add versioned metadata through
   `analysis_data_contracts.py`.
2. Give every run a `session_id` and attach immutable session metadata to frame,
   event, diagnostics, and snapshot documents.
3. Record capture, inference, frame analysis, and event publication timestamps.
4. Run FrameRecord journaling, Swing segmentation, local Coach, and DeepSeek
   submission behind `RealtimeSwingRuntime`; OSD rendering only submits data.
5. Persist event creations and patches to an append-only JSONL journal while
   retaining JSON and HTML as backward-compatible snapshots.
6. Supervise Reader, Inference, and Analyzer as named child processes; any
   unexpected child failure stops the pipeline and returns a non-zero exit.
7. The local control panel creates a unique directory per session. Explicit CLI
   output paths retain their existing meaning.

## Consequences

- Existing report and renderer scripts remain compatible because `frames`,
  `events`, and their existing child fields are unchanged.
- Consumers can migrate incrementally by checking `schema_version`.
- Real end-to-end latency is auditable instead of inferred from `frame_id/fps`.
- A Swing data failure is surfaced through the runtime and process supervisor
  instead of silently stalling Reader.
- Snapshot writes remain simple and atomic; the JSONL event journal is the
  durable history for long-running sessions.
