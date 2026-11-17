#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./venv_yolo26/bin/python}"

echo "[1/3] Checking frontend JS syntax..."
node -c static/js/*.js

echo "[2/3] Running frontend CJS unit tests..."
node --test tests/*.cjs

echo "[3/3] Running fast Python unit tests..."
"$PYTHON_BIN" -m unittest \
  test_local_control_panel.py \
  test_analysis_metric_delivery.py \
  test_swing_evaluation.py \
  test_swing_biomechanics.py \
  test_swing_report_builder.py \
  test_swing_session_summary.py \
  test_main_pipe_cli.py \
  test_pipeline_supervision.py \
  test_pipeline_runtime_optimizations.py \
  test_reader_runtime.py \
  test_pipeline_performance_controls.py

echo "All fast tests passed."
