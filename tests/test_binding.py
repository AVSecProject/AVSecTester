"""Capability negotiation: a plugin's bindings resolve against a stack profile."""

from __future__ import annotations

import pytest
from avsectester.core.binding import (
    BindingSpec,
    IncompatiblePlugin,
    resolve,
    seams_downstream_of,
)
from avsectester.core.capability import Capability as Cap
from avsectester.core.capability import StackProfile
from avsectester.core.seams import SEAMS

# A phantom-obstacle attack realizable three ways, ranked by fidelity.
PHANTOM = (
    BindingSpec("raw_lidar", "points", requires={Cap.NEURAL_PERCEPTION, Cap.RAW_LIDAR}, fidelity=3),
    BindingSpec("perception_out", "detections", requires={Cap.NEURAL_PERCEPTION}, fidelity=2),
    BindingSpec("perception_input", "objects", requires={Cap.GT_PERCEPTION}, fidelity=1),
)


def test_perception_input_seam_is_registered():
    assert "perception_input" in SEAMS
    assert SEAMS["perception_input"].phase.value == "pre"


def test_gt_stack_resolves_to_object_level():
    gt = StackProfile.of(["perception_input", "perception_out"], [Cap.GT_PERCEPTION])
    assert resolve(PHANTOM, gt).seam == "perception_input"


def test_neural_stack_prefers_raw_lidar_by_fidelity():
    neural = StackProfile.of(
        ["raw_lidar", "perception_input", "perception_out"],
        [Cap.NEURAL_PERCEPTION, Cap.RAW_LIDAR],
    )
    assert resolve(PHANTOM, neural).seam == "raw_lidar"


def test_neural_without_raw_lidar_falls_back_to_detection_level():
    neural = StackProfile.of(["perception_out"], [Cap.NEURAL_PERCEPTION])
    assert resolve(PHANTOM, neural).seam == "perception_out"


def test_incompatible_stack_raises_with_reason():
    bare = StackProfile.of(["control_out"], [])
    with pytest.raises(IncompatiblePlugin) as ei:
        resolve(PHANTOM, bare)
    msg = str(ei.value)
    assert "not exposed" in msg and "control_out" in msg


def test_unknown_seam_rejected_at_declaration():
    with pytest.raises(ValueError, match="unknown seam"):
        BindingSpec("no_such_seam")


def test_seams_downstream_of():
    ds = seams_downstream_of("perception_out")
    assert "perception_out" in ds and "tracking_out" in ds
    assert "raw_lidar" not in ds and "perception_input" not in ds
    assert "perception_out" not in seams_downstream_of("perception_out", inclusive=False)
