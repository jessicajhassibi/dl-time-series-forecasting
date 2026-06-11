from __future__ import annotations

import os
from pathlib import Path


def find_last_checkpoint(parent_directory: str = "logs") -> Path | None:
    """Find the last checkpoint in the given directory."""
    if not os.path.exists(parent_directory):
        return None
    parent_path = Path(parent_directory)
    checkpoints = list(parent_path.glob("**/*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)
