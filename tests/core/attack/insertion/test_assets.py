"""Asset geometry and texture units are checked numerically through public APIs."""
import numpy as np
import pytest
from PIL import Image
from avsectester.attack.insertion import (CheckerboardTexture, ImageTexture, Insertion,
    PlaneAsset, resolve_insertion)
from avsectester.attack.placement import AttachedPlacement, Orientation, WorldPlacement
from avsectester.geometry.poses import ActorPose, rotation_matrix


def test_textures(tmp_path):
    checker = CheckerboardTexture(size=4, cells=2, colors=((0, 0, 0), (255, 0, 0)))
    rgba = checker.rgba()
    np.testing.assert_array_equal(rgba[:, :, 0], [[0, 0, 255, 255], [0, 0, 255, 255],
                                                [255, 255, 0, 0], [255, 255, 0, 0]])
    assert np.all(rgba[..., 3] == 255)
    path = tmp_path / "texture.png"
    Image.fromarray(rgba).save(path)
    np.testing.assert_array_equal(ImageTexture(path).rgba(), rgba)
    rgba[:] = 0
    assert checker.rgba().max() == 255


def test_plane_asset():
    asset = PlaneAsset(np.full((4, 8, 4), 255, np.uint8), width_m=4, height_m=2)
    surface, = asset.planes()
    np.testing.assert_array_equal(surface.corners,
        [[0, -2, 1], [0, 2, 1], [0, 2, -1], [0, -2, -1]])
    assert surface.texture.shape == (4, 8, 4)


def test_resolve_insertion():
    host_transform = np.eye(4)
    host_transform[:3, :3] = rotation_matrix((0, 0, 90))
    host_transform[:3, 3] = (10, 5, 1)
    host = ActorPose(host_transform, (4, 2, 2))
    ego = ActorPose(np.eye(4))
    asset = PlaneAsset(np.full((2, 2, 4), 255, np.uint8), 1, 1)
    mounted = Insertion("patch", asset, AttachedPlacement("lead", "rear_center", (-.1, 0, .5)),
                        Orientation("follow_host", (0, 0, 180)))
    resolved = resolve_insertion(mounted, {"lead": host}, ego)
    np.testing.assert_allclose(resolved.pose[:3, 3], [10, 2.9, 1.5])
    np.testing.assert_allclose(resolved.pose[:3, 0], [0, -1, 0], atol=1e-12)
    facing = resolve_insertion(Insertion("sign", asset, WorldPlacement((5, 0, 0)),
                                        Orientation("face_ego")), {}, ego)
    np.testing.assert_allclose(facing.pose[:3, 0], [-1, 0, 0])
    corners = facing.planes()[0].corners
    assert np.allclose(corners[:, 0], 5)
    with pytest.raises(KeyError, match="lead"):
        resolve_insertion(mounted, {}, ego)
