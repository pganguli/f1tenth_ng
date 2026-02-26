"""Smoke test for RL training script."""

import subprocess
import sys
from pathlib import Path


def test_rl_script_help():
    script = Path(__file__).parent.parent / "train_rl.py"
    # should exit with 0 and print help text
    proc = subprocess.run([sys.executable, str(script), "--help"], capture_output=True)
    assert proc.returncode == 0
    assert b"Train RL cloud scheduler policy" in proc.stdout
