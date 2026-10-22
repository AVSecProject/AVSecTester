"""World-object extents and transforms keep their bottom-centred physical meaning."""
import numpy as np
import pytest

from avsectester.rendering import ObjectAsset, PlacedObject, PlaneSurface


def test_plane_surface():
    corners = np.array([[0., -1., 3.], [0., 1., 3.], [0., 1., 1.], [0., -1., 1.]])
    texture = np.full((4, 8, 4), [220, 0, 0, 255], np.uint8)
    surface = PlaneSurface(corners, texture)
    corners[:] = 9.
    texture[:] = 0
    np.testing.assert_array_equal(surface.corners[0], [0., -1., 3.])
    np.testing.assert_array_equal(surface.texture[0, 0], [220, 0, 0, 255])
    with pytest.raises(ValueError, match="four finite"):
        PlaneSurface(np.zeros((3, 3)), surface.texture)


def test_object_asset():
    surface = PlaneSurface(np.array([[0., -1., 3.], [0., 1., 3.],
                                    [0., 1., 1.], [0., -1., 1.]]), np.full((4, 4, 4), 255, np.uint8))
    asset = ObjectAsset.from_surfaces((surface,), "stop_sign")
    assert asset.extent == (0., 2., 3.) and asset.category == "stop_sign"
    assert len(asset.surfaces) == 1
    assert ObjectAsset((surface,), (.1, 2., 3.), "sign").extent == (.1, 2., 3.)
    with pytest.raises(ValueError, match="at least one"):
        ObjectAsset.from_surfaces((), "sign")
    with pytest.raises(ValueError, match="nonempty"):
        ObjectAsset((surface,), (0., 2., 3.), "")
    with pytest.raises(ValueError, match="inside"):
        ObjectAsset((surface,), (0., 1., 3.), "sign")


def test_placed_object():
    surface = PlaneSurface(np.array([[0., -1., 2.], [0., 1., 2.],
                                    [0., 1., 0.], [0., -1., 0.]]), np.full((2, 2, 4), 255, np.uint8))
    asset = ObjectAsset.from_surfaces((surface,), "sign")
    pose = np.eye(4)
    pose[:3, 3] = (5., 2., 1.)
    placed = PlacedObject("sign", asset, pose)
    pose[:3, 3] = 0.
    np.testing.assert_array_equal(placed.world_from_object[:3, 3], [5., 2., 1.])
    invalid = np.eye(4)
    invalid[0, 0] = 2.
    with pytest.raises(ValueError, match="rigid"):
        PlacedObject("sign", asset, invalid)
    with pytest.raises(TypeError, match="harmonizer must be callable"):
        PlacedObject("sign", asset, np.eye(4), harmonizer="classic")
