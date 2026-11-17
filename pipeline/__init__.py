"""Modular runtime helpers and CLI for the multiprocess tennis analysis pipeline."""

from __future__ import annotations

import os


def create_output_directory(output_path: str) -> str:
    """Ensure the parent directory for an output artifact path exists."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_path
