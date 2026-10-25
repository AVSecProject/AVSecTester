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

    nominal = np.full_like(texture, 255)
    edited = PlaneSurface(surface.corners, surface.texture, nominal)
    nominal[:] = 0
    np.testing.assert_array_equal(edited.nominal_texture, 255)
    assert surface.nominal_texture is None
    nominal[..., 3] = 255
    nominal[0, 0, 3] = 0
    with pytest.raises(ValueError, match="texture's shape and alpha"):
        PlaneSurface(surface.corners, np.full_like(texture, 255), nominal)


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
    nominal = np.full((2, 2, 4), [0, 0, 0, 255], np.uint8)
    edited = ObjectAsset.from_surfaces((PlaneSurface(surface.corners, surface.texture, nominal),), "sign")
    world_surface, = PlacedObject("sign", edited, placed.world_from_object).world_surfaces()
    np.testing.assert_allclose(world_surface.corners[0], [5., 1., 3.])
    np.testing.assert_array_equal(world_surface.nominal_texture, nominal)
    invalid = np.eye(4)
    invalid[0, 0] = 2.
    with pytest.raises(ValueError, match="rigid"):
        PlacedObject("sign", asset, invalid)
    with pytest.raises(TypeError, match="harmonizer must be callable"):
        PlacedObject("sign", asset, np.eye(4), harmonizer="classic")


def test_from_panel():
    face = np.full((32, 16, 4), [50, 80, 100, 255], np.uint8)
    panel = ObjectAsset.from_panel(face, category="billboard", width_m=.9,
                                   mount_height_m=1.5, post_count=1)
    assert panel.category == "billboard" and len(panel.surfaces) == 2
    np.testing.assert_allclose(panel.extent, [.06, .9, 3.3])
    post, board = panel.surfaces
    np.testing.assert_array_equal(board.texture, face)
    np.testing.assert_allclose(board.corners, [[0., -.45, 3.3], [0., .45, 3.3],
                                              [0., .45, 1.5], [0., -.45, 1.5]])
    np.testing.assert_allclose(post.corners[:, 2], [2.4, 2.4, 0., 0.])
    assert np.all(post.corners[:, 0] == -.03)
    assert np.all(post.texture[..., 3] == 255)
    assert np.all(post.texture[0, 8, :3] > post.texture[0, 0, :3])
    np.testing.assert_array_equal(post.texture[0], post.texture[-1])
    supported = ObjectAsset.from_panel(face, category="poster", width_m=2., post_count=3)
    np.testing.assert_allclose([surface.corners[:, 1].mean() for surface in supported.surfaces[:-1]],
                               [-.7, 0., .7])
    face_only = ObjectAsset.from_panel(face, category="poster", width_m=1.2)
    assert len(face_only.surfaces) == 1
    nominal = np.full_like(face, 255)
    patched = ObjectAsset.from_panel(face, category="sign", width_m=.9, post_count=1,
                                     nominal_texture=nominal)
    assert patched.surfaces[0].nominal_texture is None
    np.testing.assert_array_equal(patched.surfaces[1].nominal_texture, nominal)
    np.testing.assert_allclose(face_only.extent, [0., 1.2, 2.4])
    with pytest.raises(ValueError, match="nonnegative integer"):
        ObjectAsset.from_panel(face, category="poster", width_m=1., post_count=-1)
