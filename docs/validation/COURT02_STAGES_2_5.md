# Court02 stages 2–5

Current run: `court02_temporal_racket_20261005T161814Z_0ab2c2`.
Preparation directory: `data/analysis_results/kinematic_validation/court02_stages_2_5_20261006_v4`.

The preparation tool verifies the source, journal/event manifest hashes, session
identity and the calibration actually used by every frame. It creates separate
outputs and refuses to overwrite an existing iteration. Historical inputs remain
unchanged. Engineering readiness is distinct from independent validation.

## 2. Existing scale mapping (operator-authorized)

The user authorized reuse of ABCD and A′B′C′D′ on 2026-10-06.
AB / A′B′ are 3.3 m and AD / A′D′ are 4.8 m. Both views map their
A / A′ to (0,0), B / B′ to (3.3,0), C / C′ to (3.3,4.8) and
D / D′ to (0,4.8). This completes the revised floor-mapping requirement.
The source-bound frozen calibration is unchanged. Fit and round-trip checks
verify numerical consistency; independent physical scale accuracy is untested.
Pass `--reuse-existing-scale` to the preparation tool for this authorized scope.

### Optional independent physical validation

Record at least three non-collinear floor check points per evaluated view that
were not used to fit the calibration. Measure their X/Y positions relative to A
(along AB/AD, metres), record the instrument, uncertainty and physical evidence,
and identify them on the original frame. Existing four-corner fits and empty
scale reviews do not supply independent checks.

Use `scale_reference/index.html` and export the confirmed review. Evaluate it:

```sh
python3 scripts/evaluate_scale_checks.py \
  --review /absolute/path/independent_scale_checks.json \
  --calibration data/analysis_results/kinematic_validation/court02_newroi_iteration_20261005_v1/ground_calibration_snapshot.json \
  --source data/control_uploads/304b2cda45f90739ed77c6e3f98f8cd3.mp4 \
  --tolerance-m 0.05 \
  --output /absolute/path/new_scale_evaluation.json
```

The CLI also verifies the clicked PNG against the exact original source frame.
Reports include metre errors, pixel reprojection errors, conservative error plus
measurement uncertainty, distributed-check coverage and untested views. A
passing diagnostic does not approve physical body/3D measurements or coaching.
The 0.05 metre tolerance is a diagnostic parameter, not an agreed acceptance
standard. No automatic refitting occurs.

## 3. Automatic prelabels with complete human review

On 2026-10-06 the user requested automatic annotations with human-assisted
confirmation of every item. Active board:
`data/analysis_results/kinematic_validation/court02_all_frame_joints_20261006_v1/review/index.html`.
It contains the entire 250 source frames × 2 views × 4 shoulder/hip joints = 2000 prelabels,
all audited against the hash-bound current frame journal. 92 items have
review flags for short time intervals, temporal steps or left/right image-order
changes. See the per-item audit JSON. These are review priorities, not measured
anatomical position errors. Raw predictions remain unchanged.

Use **下一待审核项** to visit flagged/unreviewed items. Adjust a joint by clicking
the source image, confirm a suggestion individually, or confirm displayed
suggestions for the current frame and view. Every item must be reviewed or
marked unidentifiable before a complete confirmed export. The original model
suggestions, raw model scores and source identities remain separate from human
adjustments. Draft import and evaluation reject partial complete-review exports.
Local autosave stores only changed labels and backwards revision deltas, with
source metadata and suggestions supplied by the bound page. Legacy complete
drafts are restored and compacted under the same key; unrelated drafts are
never removed. If quota remains insufficient, the latest draft is retried without
local history. Failed edits remain in memory and can be exported. Before
refreshing a page that reported a failure, export its draft and then import it
after loading the fixed page.
Use `--assisted-joints --all-source-frames` in the preparation CLI to reproduce this workflow.

After receiving a confirmed export:

```sh
python3 scripts/compare_assisted_joint_review.py \
  --review /absolute/path/assisted_joint_review.json \
  --predictions data/analysis_results/kinematic_validation/court02_all_frame_joints_20261006_v1/predictions.json \
  --output /absolute/path/new_assisted_consistency.json
```

Human-assisted agreement measures correction/consistency, not independent
accuracy; accepting the same suggestions can produce circular zero errors.
The separate blind board and independent evaluator remain available when an
independent error benchmark is needed. The 15 pixel reporting parameter is not
an agreed coaching criterion. Additional body joints require separate review.

## Partial human review and automatic completion (2026-10-06)

Received `joint_draft_history-3.json`: latest revision 189, 144 reviewed labels,
including 14 unidentifiable joints. The received history and these decisions are
preserved verbatim. The remaining 1856 labels were conservatively processed:
1737 candidate positions and 119 null positions; total null positions are 133.
1621 candidates received a source-PTS-supported adjustment bounded to 2 pixels.
Only fresh high-score observations are eligible; ambiguous identity, proximity
to human-marked unknowns, and overlap proxies trigger abstention. No missing
joint is interpolated. Raw observations stay immutable. Candidate coordinates
use two decimals; inherited human values remain unchanged.

`court02_joint_optimization_20261006_v1/index.html` separates human and automatic
status and can navigate blank entries. Its revision-specific draft storage
prevents an older board's draft overwriting the completed result. Automatic
labels cannot be exported as complete human confirmation. Completion covers
assisted review, not independent visibility truth or position accuracy. Steps
4–5 retain their evidence requirements.

Reproduce into a new directory:

```sh
python3 scripts/optimize_assisted_joint_review.py \
  --history /absolute/path/joint_draft_history-3.json \
  --prelabels data/analysis_results/kinematic_validation/court02_all_frame_joints_20261006_v1/review/model_prelabels.json \
  --predictions data/analysis_results/kinematic_validation/court02_all_frame_joints_20261006_v1/predictions.json \
  --journal data/analysis_results/control_panel/court02_temporal_racket_20261005T161814Z_0ab2c2/court02_temporal_racket_frames.jsonl \
  --audit data/analysis_results/kinematic_validation/court02_all_frame_joints_20261006_v1/review/prelabel_audit.json \
  --output /absolute/path/new_optimization
python3 scripts/build_optimized_joint_review.py \
  --template data/analysis_results/kinematic_validation/court02_all_frame_joints_20261006_v1/review/index.html \
  --review /absolute/path/new_optimization/optimized_review.json \
  --image-prefix /artifacts/data/analysis_results/kinematic_validation/court02_all_frame_joints_20261006_v1/review/ \
  --output /absolute/path/new_optimization/index.html
```

## 4. Temporal and kinetic candidates

`temporal_baseline/report.html` contains the current source-PTS cadence audit and
replayed kinetic candidates. Independent labels must precede accepted parameter
tuning. Compare candidates on fixed observations and separate tuning/held-out
intervals. Assess position error, missing coverage, recovered versus observed
points, cadence sensitivity and peak displacement; lower jitter alone cannot
approve a candidate. Preserve raw observations and do not fill unsupported gaps.
A floor scale and 2D shoulder/hip labels cannot validate true axial 3D rotation.

## 5. Coach rule and score validation

`coach_reference_draft.json` requests independent coach identity, defined rubric
and version, score units/tolerance, event-level labels, unknown/visibility
decisions and held-out examples. Collect judgments before exposing model scores.
Three swings are not a general validation corpus. Validate each claimed rule and
score against the applicable observable metric and independent reference, then
review the results before adding any code-owned approved rule. The automatic
teaching allowlist remains empty until evidence supports an actual approval.
