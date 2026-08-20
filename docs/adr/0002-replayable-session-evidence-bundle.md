# ADR 0002: Replayable session evidence bundles

- Status: Accepted
- Date: 2026-08-01

## Context

The versioned session runtime persists FrameRecords, event snapshots, and an
append-only event journal, but those files were not tied together by an
integrity contract. An analyst could not prove which inputs produced an event,
detect a changed artifact, or replay the deterministic event and local Coach
stages without reconstructing command-line settings by hand.

The bundle must remain useful when a live stream is no longer reachable. It
must also avoid persisting RTSP credentials or model API keys.

## Decision

1. When evidence capture is enabled, always retain the FrameRecord JSONL and
   Swing event snapshot, even when live-mode output defaults would disable
   per-frame persistence.
2. Write one `tennis.replay-evidence-bundle.v1` manifest after pipeline
   shutdown. The manifest contains sanitized session/capture metadata,
   deterministic replay parameters, artifact roles, byte sizes, and SHA-256
   digests.
3. Treat `frame_journal` and `event_snapshot` as the minimum replay set.
   Reports, source video, frame snapshots, event journals, annotated video,
   and Swing clips are supplementary evidence.
4. Replay FrameRecords through `RealtimeSwingEventEngine`, reconstruct the
   local Coach configuration, and compare stable event semantics: identity,
   boundaries, contact/peak frames, stroke type, and Coach advice codes.
5. Keep DeepSeek outside deterministic replay. It is an asynchronous network
   sidecar and its credentials, endpoint configuration, and generated prose
   are not required to reproduce local event decisions.
6. The local control panel enables evidence bundles by default and creates the
   manifest inside the unique session output directory.

## Consequences

- A completed local-video or live-stream session can be audited after its
  original source becomes unavailable.
- Artifact changes are distinguishable from analysis drift.
- Replay intentionally starts after inference; reproducing detector output
  from source pixels remains a separate model/version provenance concern.
- Source videos may be external to the manifest directory. Their paths are
  marked external and their contents are still checksummed when available.
- Hashing a large source or annotated video adds shutdown I/O proportional to
  file size; it does not enter the live inference path.
