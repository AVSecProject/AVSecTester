"""Insertion placement and orientation specifications use explicit public types."""

import numpy as np
import pytest

from avsectester.attack.insertion.placement import (
    Orientation, WorldPlacement, orientation_from_spec, placement_from_spec,
)


def test_placement_from_spec():
    attached = placement_from_spec({"type": "avsectester.attack.insertion.placement:AttachedPlacement", "options": {
        "host": "lead", "anchor": "rear_center", "offset_m": [-.1, 0., .2],
    }})
    np.testing.assert_allclose(attached.local_position((4., 2., 1.5)), [-2.1, 0., .2])
    world = placement_from_spec({"type": "avsectester.attack.insertion.placement:WorldPlacement", "options": {"position_m": [8., 1., 0.]}})
    assert world == WorldPlacement((8., 1., 0.))
    with pytest.raises(ValueError, match="Unknown placement"):
        placement_from_spec({"type": "UnknownPlacement"})


def test_orientation_from_spec():
    orientation = orientation_from_spec({
        "type": "avsectester.attack.insertion.placement:Orientation",
        "options": {"mode": "face_ego"},
    })
    assert orientation == Orientation("face_ego")
    with pytest.raises(ValueError, match="full module:Class"):
        orientation_from_spec({"type": "UnknownOrientation"})
