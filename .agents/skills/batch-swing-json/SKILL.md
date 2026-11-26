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
   - Videos run serially so each video saturates the Apple Silicon M5 ANE (`det-workers=2`) without multi-job contention.
3. **Verify outputs**:
   - Confirm `<video_stem>_swing_events.json` exists in each video's directory, parses cleanly, and contains `events` (e.g., `jq '.events | length'`).
   - No `.mp4`, `.jpg`, `.html`, `.csv`, or `_diagnostics.json` sidecars are written into the target video directory (intermediate files are staged in a temporary directory and removed).

## Common Options

- `--no-recursive`: Scan only the top level of `<target_dir>` instead of subdirectories.
- `--force`: Re-analyze and overwrite existing valid `<video_stem>_swing_events.json` files (default skips existing valid outputs).
- `--pose-model yolo26m-pose.mlpackage`: Switch to the heavier pose model when higher pose precision is needed (`yolo26n-pose.mlpackage` is the default).
- `--stream-id <id>`: Force a specific camera/ROI profile from `configs/roi_config.yaml`.
- `--det-workers N` / `--shm-slots N` / `--max-frames N`: Override ANE detection worker count, shared memory buffer slots, or frame limit.
