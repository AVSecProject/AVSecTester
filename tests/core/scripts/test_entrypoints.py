"""Repository tools resolve their imports through the documented module entry points."""

import subprocess
import sys

import pytest

from scripts import REPO_ROOT


@pytest.mark.parametrize("module", [
    "demos.carla.carla_insertion_demo",
    "demos.nurec.alpamayo_attack_demo",
    "demos.nurec.nurec_patch_demo",
    "demos.nuscenes.nuscenes_object_demo",
])
def test_help_requires_no_server_or_model_weights(module):
    result = subprocess.run(
        [sys.executable, "-m", f"scripts.{module}", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
