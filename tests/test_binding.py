"""Plugin ↔ stack compatibility: seam-presence + capability checks, downstream helper."""

from __future__ import annotations

import pytest
from avsectester.core.binding import (
    IncompatiblePlugin,
    check_support,
    seams_downstream_of,
)
from avsectester.core.capability import Capability as Cap
from avsectester.core.capability import StackProfile
from avsectester.core.seams import SEAMS


def test_perception_input_seam_is_registered():
    assert "perception_input" in SEAMS
    assert SEAMS["perception_input"].phase.value == "pre"


def test_check_support_ok_when_all_present():
    profile = StackProfile.of(["perception_out", "tracking_out"], [Cap.NEURAL_PERCEPTION])
    check_support(("perception_out", "tracking_out"), frozenset({Cap.NEURAL_PERCEPTION}), profile)


def test_check_support_rejects_missing_seam():
    profile = StackProfile.of(["perception_out"], [Cap.NEURAL_PERCEPTION])
    with pytest.raises(IncompatiblePlugin, match="tracking_out"):
        check_support(("perception_out", "tracking_out"), frozenset(), profile)


def test_check_support_rejects_missing_capability():
    profile = StackProfile.of(["perception_out"], [Cap.GT_PERCEPTION])
    with pytest.raises(IncompatiblePlugin, match="neural_perception"):
        check_support(("perception_out",), frozenset({Cap.NEURAL_PERCEPTION}), profile)


def test_check_support_rejects_unknown_seam():
    with pytest.raises(ValueError, match="unknown seam"):
        check_support(("no_such_seam",), frozenset(), StackProfile.of(["perception_out"]))


def test_check_support_rejects_no_seams():
    with pytest.raises(IncompatiblePlugin, match="no seams"):
        check_support((), frozenset(), StackProfile.of(["perception_out"]))


def test_seams_downstream_of():
    ds = seams_downstream_of("perception_out")
    assert "perception_out" in ds and "tracking_out" in ds
    assert "raw_lidar" not in ds and "perception_input" not in ds
    assert "perception_out" not in seams_downstream_of("perception_out", inclusive=False)
