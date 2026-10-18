# ADR 0004: Continuous processing with startup readiness and bounded backpressure

- Status: Accepted
- Date: 2026-10-18

## Context

The user accepts startup delay and requires every source frame to be processed.
The previous realtime policy dropped stale source frames and sampled the new
racket model. A low receipt-to-output latency could hide a growing input backlog.
HighGUI also blocked the Analyzer on every frame and during window initialization.

## Decision

1. Default to `buffered`; retain `realtime` as an explicit legacy freshness policy
   and `offline` as the unpaced local-file mode. Racket mode remains `legacy`.
2. Start local input only after inference, stateless Core ML warmup and Analyzer/
   display readiness. Warmup must not advance trackers or motion state.
3. Preserve decoded frame order. Keep three source shared-memory slots and apply
   backpressure rather than replacing or grabbing past a frame. Do not rebase
   the local nominal input schedule when processing falls behind.
4. In buffered/offline modes, an enabled racket worker runs on every frame and
   waits for its same-source result. Worker failure or a ten-second timeout fails
   the run, rather than reporting an incomplete fallback as a full success.
5. Isolate HighGUI in a supervised worker with one pending owned image. The
   default queue blocks and normal EOF drains it. Cancellation may stop early.
   Record submission and HighGUI return separately; neither proves physical
   screen presentation or sensor exposure time.
6. Retain source frame identity/PTS, all frame journals, event/Coach/report/clip
   workflows and immutable evidence. Add stage timing, input-schedule lag and
   display completion evidence. Continuous completeness and sustained realtime
   performance are separate gates.

## Consequences

Local videos can prove decoded source coverage. Upstream camera/network loss
remains unverified. Bounded application queues limit memory, but backpressure on
a live transport can move backlog upstream. When sustained capacity is below
the input rate, delay grows; fixed startup buffering cannot repair that deficit.
The three short local-video tests prove continuity, not a ten-minute soak or
independent model accuracy. See the pipeline performance report for measured
results; the old sampled-mode acceptance does not certify buffered realtime.
