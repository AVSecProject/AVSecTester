"""Sign assets keep their face, post and bottom-centred world-object geometry."""
import numpy as np
import pytest

from avsectester.rendering import SignAsset, STOP_SIGN_FACE, post_rgba, stop_sign_asset


def test_sign_asset():
    face = np.full((32, 16, 4), [50, 80, 100, 255], np.uint8)
    sign = SignAsset(face=face)
    planes = tuple(sign.planes())
    assert len(planes) == 2
    assert all(plane.texture.shape[2] == 4 and plane.texture.dtype == np.uint8 for plane in planes)
    np.testing.assert_array_equal(planes[-1].texture, face)
    np.testing.assert_allclose(planes[-1].corners,
                              [[0., -.45, 3.3], [0., .45, 3.3],
                               [0., .45, 1.5], [0., -.45, 1.5]])
    asset = sign.object_asset("billboard")
    assert asset.category == "billboard" and len(asset.surfaces) == 2
    np.testing.assert_allclose(asset.extent, [.06, .9, 3.3])
    assert np.min(planes[0].corners[:, 2]) == 0.
    with pytest.raises(ValueError, match="Sign width_m"):
        SignAsset(face=face, width_m=0.)


def test_stop_sign_asset():
    assert STOP_SIGN_FACE.is_file()
    asset = stop_sign_asset()
    assert asset.category == "stop_sign" and len(asset.surfaces) == 2
    np.testing.assert_allclose(asset.extent, [.06, .9, 2.4])
    np.testing.assert_allclose(asset.surfaces[-1].corners,
                              [[0., -.45, 2.4], [0., .45, 2.4],
                               [0., .45, 1.5], [0., -.45, 1.5]])
    np.testing.assert_allclose(asset.surfaces[0].corners[:, 2], [1.95, 1.95, 0., 0.])
    lower = np.array([-asset.extent[0] / 2, -asset.extent[1] / 2, 0.])
    upper = np.array([asset.extent[0] / 2, asset.extent[1] / 2, asset.extent[2]])
    assert all(np.all(surface.corners >= lower) and np.all(surface.corners <= upper)
               for surface in asset.surfaces)
    face_only = stop_sign_asset(width_m=1.2, mount_height_m=0., post=False)
    assert len(face_only.surfaces) == 1
    np.testing.assert_allclose(face_only.extent, [0., 1.2, 1.2])


def test_post_rgba():
    texture = post_rgba(height_pixels=32, width_pixels=8)
    assert texture.shape == (32, 8, 4) and texture.dtype == np.uint8
    assert np.all(texture[..., 3] == 255)
    assert np.all(texture[0, 4, :3] > texture[0, 0, :3])
    np.testing.assert_array_equal(texture[0], texture[-1])

