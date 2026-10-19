# Frozen continuous 640 shadow candidate

This is an acceptance candidate, not a formal release or active-model quality receipt.

- Startup: `640_shadow`, `shadow`, buffered continuous processing, stride 1,
  barrier scheduling, shared source-frame transport, Court 02 project ROI.
- Preserve the legacy body pixel path (`pose_resize_policy: legacy`), body
  Core ML `ALL`, ball detector `ANE`, racket Core ML `ALL`. No frame sampling.
- Keep the existing OpenCV default thread budget. The sequential three-video
  comparison of 10 vs 1 thread produced identical business observations and
  events, but did not show a consistent speed benefit; no thread change adopted.
- Preserve display, main encoding, evidence replay, report refresh, Coach and
  event clips. External network/TTS isolation is tested separately while enabled.
- Default business mode remains legacy. Single-resize body policies remain opt-in
  engineering candidates; they are not part of this acceptance profile.

The first frozen candidate (`eec8297`) passed all 27 short runs but developed
latency tails during the uninterrupted session. Its failed videos and journals
are retained. Do not reuse that candidate's 27 passes for the next build.

The next candidate caches JSON for unchanged immutable event roots in the report
writer. Clip and sidecar updates already replace those roots; replacement events
are re-encoded and removed roots are released. Report refresh, full history,
atomic publication, models, input pixels, and event/Coach rules remain intact.
On the same captured 183-event, 32,950,044-byte document, repeated JSON+HTML
publication used about 187ms CPU instead of 846ms; parsed output hashes matched.
This isolated result is not an end-to-end acceptance receipt. Repeat the long
session and the complete matrix/workflow on the newly frozen commit.

Long-session video verification now gives FFprobe a frame-count-proportional
deadline. The original 30-second verifier deadline was insufficient for decoding
15,250 frames. This postprocessing repair does not change measured runtime clocks.

After committing the source, `scripts/validate_stable_shadow.py --output NEW_DIR`
freezes the commit, source/config/input hashes and all three model identities.
All subsequent phases must receive that directory's `frozen-inputs.json`:

1. Nine original clips, three distinct repetitions; all 27 strict gates must pass.
2. `scripts/validate_stable_shadow_soak.py --template FIRST_RUN/runtime.yaml
   --frozen-inputs NEW_DIR/frozen-inputs.json --output NEW_SOAK_DIR`.
3. `scripts/validate_stable_shadow_workflow.py --shadow-runs NEW_DIR
   --frozen-inputs NEW_DIR/frozen-inputs.json --output NEW_WORKFLOW_DIR`.

The soak uses one uninterrupted process with 61 references to the original
250-frame clip. Source PTS, completeness, resource growth and nominal latency
remain explicit. Local file tests do not certify upstream camera/network loss.
No failed measurement may be replaced by its pass flag. Do not combine phases
from different commits or build identities. Passing unit tests alone cannot
promote the candidate. Preserve all journals and failed videos; prune only
verified successful derivative videos according to the user's retention policy.
