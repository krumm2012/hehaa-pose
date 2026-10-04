# ADR 0003: Separate source media time from runtime latency

- Status: Accepted
- Date: 2026-10-04

## Context

FrameRecord `timestamp` historically used `frame_id / fps`. Reader
`captured_at_unix_ns` is measured after `cap.read()` and describes receipt, not
sensor exposure. Using receipt time for file kinematics makes estimated motion
speed and peak intervals depend on decoding throughput and queue backpressure.
The realtime pipeline must remain lightweight and replayable.

## Decision

Add `source_time` with schema `tennis.source-time.v1` to FrameRecord; retain the
existing `tennis.frame.v1` contract and latency fields for compatibility:

- `source_kind`: `video_file` or `stream`.
- `source_frame_id`: input frame identity, including known discarded-frame gaps.
- `timestamp_seconds`: input media presentation time, estimated time, or null.
- `basis`: `media_pts`, `nominal_fps`, or `unavailable`.
- `quality`: `reported`, `estimated`, `duplicate`, `discontinuous`, or `unavailable`.
- `provider`/`reason`: time provenance or the reason measurement is unavailable.
- `exposure_time_verified`: false for the present OpenCV adapter.

For files, read `CAP_PROP_POS_MSEC` immediately after decoding and carry the
record through both inference paths and the analyzer into the append-only frame
journal. These are backend-reported timestamps of the input file; they do not
prove sensor exposure or identify an original exposure duplicated by a prior
transcode. Original-file mapping remains a separate future contract.

Finite nonnegative PTS, including the first zero, is retained without rounding.
Repeated or backward PTS is preserved and flagged, not replaced by nominal
time. Missing/invalid PTS may use explicitly estimated `frame_id / fps`. A
backend returning constant zero will be flagged as duplicate; it cannot supply
a trusted timeline. Invalid FPS and missing PTS leave time unavailable.

Streams currently have no device media clock adapter. Receipt and inference
times remain latency measurements; source time is unavailable. This abstains
only from kinetic peak timing, leaving detection, event candidates and display
running. Future device clocks require their own provenance and reset handling.

FrameRecord `timestamp` mirrors non-null source time for compatibility. When
source time is unavailable the legacy nominal value remains, but measurement
consumers must inspect `source_time` before using it. The two independent raw
pose observations remain in `kinematic_views`, without copying fused joints.

The kinetic analyzer uses a uniform source basis across an event. Duplicate,
backward, mixed or incomplete source-time records abstain; no wall-clock
fallback is permitted. Explicit nominal-FPS estimates cap heuristic evidence
quality at 0.45 and remain ineligible for technical coaching. Old journals
without the contract use their existing monotonic `timestamp`, labelled
`legacy_unverified`; they also ignore receipt clocks. All results retain the
unvalidated 2D status and are not accuracy probabilities.

Increment the kinetic policy version to `kinematic_cross_view_2d_v2_source_time`
so replay consumers can distinguish the corrected algorithm. Keep source
records unchanged during replay and never rewrite historical evidence.

## Validation and limits

Regression tests cover variable cadence, frame gaps, decoder-clock invariance,
duplicates, backward time, explicit fallback, unavailable stream time, mixed
bases, FrameProcessor/stamping and JSON replay preservation. Source time does
not change the definition of the image-plane proxy into true axial rotation.

Other event segmentation thresholds and legacy speed metrics that use FPS are
outside this change. Media PTS is not a claim of independently calibrated
exposure timing, and unique PTS cannot detect duplicated images/exposures
introduced before ingestion.
