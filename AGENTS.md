## Agent skills

### Issue tracker

Issues and PRDs for this repo live as GitHub issues. See `docs/agents/issue-tracker.md`.

### Triage labels

The default five-label triage vocabulary is used. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository. See `docs/agents/domain.md`.

### Core ML & Dual-View Pose Lessons

- **Never set `CPU_AND_GPU` or `CPU_ONLY` on YOLO26 Core ML models** on Apple Silicon (M5 / macOS 26.7.1): `iOS15–18` models crash in `MPSGraphExecutable` MLIR pass (`SIGABRT`); `iOS26` avoids crash but is 4.3× slower (`41.97 ms` vs ANE `9.84 ms`). Keep `Det(ANE) + Pose(ALL)`.
- Keep `pose_resize_policy: direct_source` + `pose_preprocess_mode: letterbox` + float subpixel coordinates for measurement paths, and preserve both back-view guards (`_enforce_backview_anatomical_consistency` & `_suppress_backview_foreground_occluded_ankles`).
- Full benchmark details and trade-offs: see `docs/agents/pose-coreml-lessons.md`.
