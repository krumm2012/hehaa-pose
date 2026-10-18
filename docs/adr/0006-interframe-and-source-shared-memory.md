# ADR 0006: Opt-in source-slot IPC and two-stage inference

- Status: Accepted for experiments; production defaults unchanged
- Date: 2026-10-18

## Context

The user requested source shared-memory transport and inter-frame pipelining,
with no skipped frames and a fixed startup choice. A single-frame parallel
barrier is not proof that adding concurrency will improve hardware throughput.
The target remains sustained 25 FPS with bounded latency and full source coverage.

## Decision

1. Retain `barrier` scheduling and `queue` racket transport as defaults. Expose
   `--inference-schedule barrier|interframe` and `--racket-ipc queue|shared_memory`
   in the existing CLI/control panel; choices are fixed at startup.
2. Shared transport borrows existing Reader source slots only in buffered/offline
   modes. Queue messages contain source slot/name/shape/dtype/generation/PTS,
   never an ndarray. Preprocessing remains in the sidecar and is not zero-copy.
3. Append an eight-byte generation header to each source slot. Reader publishes
   the generation after the image. The sidecar checks identity, rejects replay,
   exposes a read-only view, and checks generation again after inference. Parent
   retains the slot until the same-frame result or worker termination, then
   aggregation/Analyzer owns it until returning it to q_free. Only the pipeline
   owner unlinks the shared memory. Stop/error cancels the run before reclamation.
4. Interframe scheduling requires exclusive racket selection and continuous
   processing. Stage A owns candidate detector calls and racket submission/join.
   Stage B owns same-frame body inference, association and ordered temporal state.
   One preparation thread plus two credits bounds in-flight inference frames to
   two, with a one-item prepared queue. Existing Reader slot counts are unchanged.
5. Candidate inference uses a shallow model-sharing wrapper with independent
   per-frame metadata and no ball selectors. Packets detach candidates and parse
   diagnostics. The aggregation wrapper alone owns ball history and selectors;
   body instances, dynamic body/mirror crops, racket resolver, biomechanical state
   and events advance in source order. No next-frame body crop is prepared early.
6. EOF drains both stages. Preparation exceptions propagate as session failure.
   Cancellation joins preparation before closing the sidecar/model resources.
   A supervision timeout fails the session rather than certifying partial output.
7. Record schedule, transport, preparation/aggregation timing and source identity.
   Compare queue/barrier, shared/barrier and shared/interframe sequentially on
   frozen local inputs, rotating order. Preserve reports/encoding/display; isolate
   external network/voice only for the documented core-chain comparison.

## Consequences

Extra concurrency can retain the same accelerator contention or make it worse.
The observed queue delay includes scheduling; it is not pure serialization cost.
Two frames in flight can add latency and cannot fix insufficient compute capacity.
Production acceptance still requires full-video repeats, no accumulating backlog,
resource/fault tests and the full workflow. A three-clip single-round experiment
must not promote the defaults or certify 25 FPS.
