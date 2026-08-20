#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${1:-/opt/homebrew/bin/python3.11}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python 3.10+ not found: $python_bin" >&2
  exit 1
fi

"$python_bin" -m venv "$workspace_dir/venv_qwen3_tts"
"$workspace_dir/venv_qwen3_tts/bin/python" -m pip install --upgrade pip
"$workspace_dir/venv_qwen3_tts/bin/python" -m pip install -r "$workspace_dir/requirements-tts-mlx.txt"
"$workspace_dir/venv_qwen3_tts/bin/python" - <<'PY'
from mlx_audio.tts.utils import load_model
load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
print("Qwen3-TTS MLX 0.6B is ready.")
PY
