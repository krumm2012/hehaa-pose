---
name: batch-swing-json
description: >-
  Batch analyzes tennis videos in a directory (or single file) using the
  offline headless Core ML pipeline and saves `<video_stem>_swing_events.json`
  directly into each video's directory. Use when the user asks to generate or
  batch export swing event JSON files for a folder of tennis videos.
---

# Batch Swing Event JSON Exporter (`batch-swing-json`)

Runs `main_pipe.py` in offline headless ANE-accelerated mode (`--algo2-dual-view --processing-mode offline --headless --racket-pose-mode legacy --pose-model yolo26n-pose.mlpackage --analyze-swings`) across all videos in a target directory and places `<video_stem>_swing_events.json` alongside each source video.

## Procedure

1. **Dry-run discovery first** (optional preview of files that will run vs skip):
   ```bash
   ./venv_yolo26/bin/python .agents/skills/batch-swing-json/scripts/batch_swing_json.py "<target_dir>" --dry-run
   ```
2. **Execute batch export**:
   ```bash
   ./venv_yolo26/bin/python .agents/skills/batch-swing-json/scripts/batch_swing_json.py "<target_dir>"
   ```
   - Script: [batch_swing_json.py](./scripts/batch_swing_json.py)
   - Must invoke with `./venv_yolo26/bin/python` (never system `python3`).
   - Videos run one at a time. See `--det-workers` below for the number of detector processes used within each video.
3. **Verify outputs**:
   - Confirm `<video_stem>_swing_events.json` exists in each video's directory, parses cleanly, and contains `events` (e.g., `jq '.events | length'`).
   - No `.mp4`, `.jpg`, `.html`, `.csv`, or `_diagnostics.json` sidecars are written into the target video directory (intermediate files are staged in a temporary directory and removed).

## Common Options

- **`--no-recursive`** scans only the immediate files in the target directory. The default also scans subdirectories. This option has no effect when the target is a single file.
- **`--force`** reruns analysis even if a valid `<video_stem>_swing_events.json` already exists. The script replaces the existing result after the new JSON passes validation. By default, it skips any existing JSON with an `events` array, including an empty array. Use `--force` to regenerate results after changing the model, camera profile, or frame limit.
- **`--pose-model <path>`** selects the human pose model. The default is `yolo26n-pose.mlpackage`. Use `--pose-model yolo26m-pose.mlpackage` when you want to try the larger model for harder poses. Paths can be absolute or relative to the repository root. This option selects the body keypoint model.
- **`--stream-id <id>`** selects the ROI and mirror calibration for the camera that recorded the video. The selected ID applies to every video in the batch. Run separate batches for videos from different cameras.

  With the current default config, a local video whose source matches no profile falls back to **Court 01 (`court01-main`)**. The program checks an explicit CLI or YAML stream ID first, then the input source, then the default profile. A filename such as `court02.mp4` does not select Court 02. An unknown ID can also fall back to Court 01. Check the startup `[ROI]` line with a normal run if you need to confirm the selected profile. A dry run only lists the files to process.

  | Recording camera | Stream ID |
  | --- | --- |
  | Court 01 Main Camera | `court01-main` |
  | Court 02 Main Camera | `court02-main` |
  | Court 03 Main Camera | `court03-main` |
  | Camera 04 Test Booth | `camera04-main` |

  For example, a batch of Court 02 recordings uses `--stream-id court02-main`. The IDs and default flag are defined in `configs/roi_config.yaml`.

- **`--det-workers N`** sets the number of separate Core ML ball and racket detector processes per video. `0` disables that process pool. The current default offline, headless, dual-view, legacy setup chooses `2` for the default small pose model and `1` for the larger pose model unless `pipeline_perf.det_workers` is set. Positive values require legacy dual view with live frame dropping disabled. Keep the automatic setting for normal exports.
- **`--shm-slots N`** sets the number of shared-memory slots for decoded video frames. Without a CLI or YAML override, headless mode uses `2 × det-workers + 4` slots when the detector pool runs, or `4` with no pool. For example, the default `2` workers use `8` slots. An explicit value must be at least `det-workers + 4` with the pool, or `3` without it. More slots consume more memory, especially for high-resolution video.
- **`--max-frames N`** limits each video to its first `N` processed frames. `0` means no cap and is the current default under `video_processing.max_frames`. For example, `--max-frames 300` gives about 12 seconds for a 25 FPS recording. A capped run writes the usual JSON filename, and a later full run skips that valid result by default. Use a temporary copy for previews or add `--force` when regenerating the full result.
