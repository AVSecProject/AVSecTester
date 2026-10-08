"""Insertion geometry is shared by initial selection and runtime rendering."""

import numpy as np
import pytest

from avsectester.insertion import (
    ActorPose,
    AttachedPlacement,
    Insertion,
    Orientation,
    PlaneAsset,
    PlaneSurface,
    WorldPlacement,
    resolve_insertion,
    rotation_matrix,
)


def actor(position=(0, 0, 0), angles=(0, 0, 0), extent=(4, 2, 2)):
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix(angles)
    pose[:3, 3] = position
    return ActorPose(pose, extent)


def asset(color=255):
    texture = np.full((4, 4, 4), color, dtype=np.uint8)
    texture[..., 3] = 255
    return PlaneAsset(texture, 1.0, 0.5)


def test_rear_installation_follows_named_host_and_relative_rotation():
    item = Insertion(
        "patch",
        asset(),
        AttachedPlacement("lead", "rear_center", (-0.1, 0, 0.5)),
        Orientation("follow_host", (0, 0, 180)),
    )
    victim = actor()
    first = resolve_insertion(item, {"lead": actor((10, 0, 1))}, victim)
    np.testing.assert_allclose(first.pose[:3, 3], [7.9, 0, 1.5])
    np.testing.assert_allclose(first.pose[:3, 0], [-1, 0, 0], atol=1e-12)

    second = resolve_insertion(item, {"lead": actor((20, 5, 1), (0, 0, 90))}, victim)
    np.testing.assert_allclose(second.pose[:3, 3], [20, 2.9, 1.5])
    np.testing.assert_allclose(second.pose[:3, 0], [0, -1, 0], atol=1e-12)


def test_fixed_world_rotation_preserves_full_attached_position_motion():
    item = Insertion(
        "panel",
        asset(),
        AttachedPlacement("host", offset_m=(0, 0, 1)),
        Orientation("fixed_world", (0, 0, 90)),
    )
    # A host rolled 90 degrees carries its roof installation point to its right.
    resolved = resolve_insertion(item, {"host": actor((5, 6, 7), (90, 0, 0))}, actor())
    np.testing.assert_allclose(resolved.pose[:3, 3], [5, 5, 7])
    np.testing.assert_allclose(resolved.pose[:3, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(resolved.pose[:3, 2], [0, 0, 1], atol=1e-12)


def test_world_placement_does_not_move_with_any_actor():
    sign = Insertion("sign", asset(), WorldPlacement((20, -3, 1.5)))
    first = resolve_insertion(sign, {"host": actor()}, actor())
    second = resolve_insertion(sign, {"host": actor((100, 20, 3))}, actor((4, 5, 0)))
    np.testing.assert_array_equal(first.pose, second.pose)


@pytest.mark.parametrize("victim_position", [(0, 0, 3), (7, -4, 0), (10, 0, 10)])
def test_face_victim_tracks_three_dimensional_target(victim_position):
    panel = Insertion("panel", asset(), WorldPlacement((10, 0, 0)), Orientation("face_victim"))
    result = resolve_insertion(panel, {}, actor(victim_position))
    direction = np.asarray(victim_position) - [10, 0, 0]
    direction = direction / np.linalg.norm(direction)
    np.testing.assert_allclose(result.pose[:3, 0], direction)
    np.testing.assert_allclose(result.pose[:3, :3].T @ result.pose[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(result.pose[:3, :3]) == pytest.approx(1)
    assert result.pose[2, 2] >= -1e-12


def test_face_victim_offset_is_victim_local():
    panel = Insertion(
        "panel",
        asset(),
        WorldPlacement((0, 0, 0)),
        Orientation("face_victim", target_offset_m=(2, 0, 0)),
    )
    result = resolve_insertion(panel, {}, actor((0, 0, 0), (0, 0, 90)))
    np.testing.assert_allclose(result.pose[:3, 0], [0, 1, 0], atol=1e-12)


def test_multiple_hosts_and_assets_remain_independent():
    first = Insertion("a", asset(50), AttachedPlacement("left"))
    second = Insertion("b", asset(100), AttachedPlacement("right", "top_center"))
    actors = {"left": actor((10, 3, 1)), "right": actor((20, -4, 1))}
    a, b = [resolve_insertion(item, actors, actor()) for item in (first, second)]
    np.testing.assert_allclose(a.pose[:3, 3], [10, 3, 1])
    np.testing.assert_allclose(b.pose[:3, 3], [20, -4, 2])
    assert a.host_id == "left" and b.host_id == "right"
    assert a.planes()[0].texture[0, 0, 0] == 50
    assert b.planes()[0].texture[0, 0, 0] == 100


def test_custom_asset_surfaces_and_compositor_unpacking():
    class TwoPanels:
        def planes(self):
            base = asset().planes()[0]
            return (base, PlaneSurface(base.corners + [1, 0, 0], base.texture))

    item = Insertion("custom", TwoPanels(), WorldPlacement((10, 0, 0)))
    resolved = resolve_insertion(item, {}, actor())
    surfaces = resolved.planes()
    assert len(surfaces) == 2
    first_corners, texture = surfaces[0]
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


def test_invalid_bindings_and_undefined_orientation_are_explicit():
    item = Insertion("patch", asset(), AttachedPlacement("missing"))
    with pytest.raises(KeyError, match="missing host"):
        resolve_insertion(item, {"other": actor()}, actor())
    with pytest.raises(ValueError, match="requires an attached"):
        Insertion("patch", asset(), WorldPlacement((0, 0, 0)), Orientation("follow_host"))
    with pytest.raises(ValueError, match="coincident"):
        resolve_insertion(
            Insertion("patch", asset(), WorldPlacement((0, 0, 0)), Orientation("face_victim")),
            {},
            actor(),
        )
    with pytest.raises(ValueError, match="rotation_deg must be zero"):
        Orientation("face_victim", (0, 0, 90))
    with pytest.raises(ValueError, match="Unknown anchor"):
        AttachedPlacement("host", "find_best_position")


@pytest.mark.parametrize("angles", [(90, 0, 0), (0, 90, 0), (0, 0, 90)])
def test_rotation_convention_matches_positive_right_hand_rule(angles):
    basis = np.array([0, 1, 0]) if angles[0] else np.array([1, 0, 0])
    expected = [0, 0, 1] if angles[0] else ([0, 0, -1] if angles[1] else [0, 1, 0])
    np.testing.assert_allclose(rotation_matrix(angles) @ basis, expected, atol=1e-12)
