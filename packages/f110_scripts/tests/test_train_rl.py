"""Smoke test for RL training script."""

import subprocess
import sys
from pathlib import Path


def test_rl_script_help():
    # the tests live under packages/f110_scripts/tests, so walk up to the
    # package root before locating the script. the previous implementation
    # assumed the script would be under ``tests/src`` which doesn't exist and
    # caused an ENOENT error.
    package_root = Path(__file__).parents[1]
    script = package_root / "src" / "f110_scripts" / "train" / "train_rl.py"

    # should exit with 0 and print help text
    proc = subprocess.run([sys.executable, str(script), "--help"], capture_output=True)
    assert proc.returncode == 0
    assert b"Train RL cloud scheduler policy" in proc.stdout
