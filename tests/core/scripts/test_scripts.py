"""Shipped Python command-line scripts expose help without services or model weights."""

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = sorted(
    ".".join(path.relative_to(ROOT).with_suffix("").parts)
    for path in (ROOT / "scripts").rglob("*.py")
    if path.name != "__init__.py"
)


@pytest.mark.parametrize("module", SCRIPTS, ids=lambda module: module.rsplit(".", 1)[-1])
def test_help(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], cwd=ROOT,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage" in result.stdout.casefold()
