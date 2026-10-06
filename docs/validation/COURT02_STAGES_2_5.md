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

## 3. Independent joint baseline

Use `independent_joints/index.html`, which hides model predictions. It covers
contiguous intervals from contact minus 0.56 seconds through contact plus 0.24
seconds around all three swings, including current candidate hip/shoulder peaks.
Label anatomical left/right shoulders and hips in both views, or explicitly mark
unidentifiable joints. Export with the independent annotator identity and review
confirmation. Existing model-assisted accepted points cannot replace this step.

```sh
python3 scripts/evaluate_joint_labels.py \
  --labels /absolute/path/joint_labels_draft.json \
  --predictions data/analysis_results/kinematic_validation/court02_stages_2_5_20261006_v4/predictions.json \
  --tolerance-px 15 \
  --output /absolute/path/new_joint_evaluation.json
```

Report completeness, visibility, missing qualified predictions and pixel errors
separately. The 15 pixel tolerance is a diagnostic parameter. More body joints
are needed when validating elbow/wrist/leg metrics.

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
