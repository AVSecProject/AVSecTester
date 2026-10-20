"""Insertion geometry is shared by initial selection and runtime rendering."""

import numpy as np
import pytest

from avsectester.geometry.poses import ActorPose, rotation_matrix
from avsectester.attack.placement import WorldPlacement
from avsectester.attack.insertion import Insertion, PlaneAsset, PlaneSurface, resolve_insertion


def actor(position=(0, 0, 0), angles=(0, 0, 0), extent=(4, 2, 2)):
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix(angles)
    pose[:3, 3] = position
    return ActorPose(pose, extent)


def asset(color=255):
    texture = np.full((4, 4, 4), color, dtype=np.uint8)
    texture[..., 3] = 255
    return PlaneAsset(texture, 1.0, 0.5)


def test_custom_asset_surfaces_preserve_corners_and_texture():
    class TwoPanels:
        def planes(self):
            base = asset().planes()[0]
            return (base, PlaneSurface(base.corners + [1, 0, 0], base.texture))

    item = Insertion("custom", TwoPanels(), WorldPlacement((10, 0, 0)))
    resolved = resolve_insertion(item, {}, actor())
    surfaces = resolved.planes()
    assert len(surfaces) == 2
    first_corners, texture = surfaces[0].corners, surfaces[0].texture
    assert np.all(first_corners[:, 0] == 10)
    assert np.all(surfaces[1].corners[:, 0] == 11)
    assert texture.shape == (4, 4, 4)


def test_pose_and_texture_inputs_are_snapshots():
    original_pose = np.eye(4)
    original_texture = np.full((2, 2, 4), 255, dtype=np.uint8)
    pose = ActorPose(original_pose)
    panel = PlaneAsset(original_texture, 1, 1)
    original_pose[0, 3] = 100
    original_texture[:] = 0
    assert pose.transform[0, 3] == 0
    assert np.all(panel.texture == 255)
    with pytest.raises(ValueError):
        pose.transform[0, 3] = 3
