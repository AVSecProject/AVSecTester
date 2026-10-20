"""ObjectInsertionAttack asset geometry, textures and parameter validation."""
import math
import numpy as np
import pytest
from avsectester.attack.implemented_attacks.object_insertion_attack.assets import SignAsset
from avsectester.geometry.poses import ActorPose
from avsectester.attack.placement import Orientation, WorldPlacement
from avsectester.attack.insertion import Insertion, resolve_insertion


def test_sign_world_geometry_preserves_face_and_post_positions():
    yaw = 20
    insertion = Insertion("sign", SignAsset(), WorldPlacement((20, -5, 0)),
                          Orientation("fixed_world", (0, 0, 180 + yaw)))
    post, face = resolve_insertion(insertion, {}, ActorPose(np.eye(4))).planes()
    angle = math.radians(yaw)
    right = np.array([math.sin(angle), -math.cos(angle), 0])
    center = np.array([20, -5, 1.95])
    expected = np.array([center - right * .45 + [0, 0, .45],
                         center + right * .45 + [0, 0, .45],
                         center + right * .45 - [0, 0, .45],
                         center - right * .45 - [0, 0, .45]])
    np.testing.assert_allclose(face.corners, expected, atol=1e-12)
    assert face.texture[0, 0, 3] == 0
    np.testing.assert_allclose(post.corners[:2].mean(axis=0),
                               [20 + .03 * math.cos(angle), -5 + .03 * math.sin(angle), 1.95])
    assert post.corners[3, 2] == pytest.approx(0)

def test_all_asset_classes_construct_and_expose_independent_rgba_surfaces():
    from avsectester.attack.implemented_attacks.object_insertion_attack.assets import (
        PersonAsset, TrafficLightAsset,
    )
    person = np.zeros((120, 40, 4), np.uint8)
    person[:, 10:30] = [60, 70, 90, 255]
    assets = (SignAsset(), PersonAsset(person), PersonAsset(person, mode="standee"),
              TrafficLightAsset(lit="green"))
    for asset in assets:
        surfaces = tuple(asset.planes())
        assert surfaces and all(surface.texture.dtype == np.uint8 for surface in surfaces)
        assert all(surface.texture.shape[-1] == 4 for surface in surfaces)
    assert len(assets[1].planes()) == 3
    assert len(assets[2].planes()) == 1
    assert len(assets[3].planes()) == 2

@pytest.mark.parametrize("options", [{"lit": "blue"}, {"head_count": 0}, {"texture_height_pixels": 4}, {"gap_fraction": -1}])
def test_traffic_light_options_reject_unknown_or_invalid_values(options):
    from avsectester.attack.implemented_attacks.object_insertion_attack.assets import TrafficLightAsset
    with pytest.raises(ValueError):
        TrafficLightAsset(**options).planes()

def _person(h=120, w=40):
    """A synthetic cut-out: an opaque person-shaped column on a transparent background."""
    img = np.zeros((h, w, 4), np.uint8)
    img[:, 10:30, :3] = (60, 70, 90)
    img[:, 10:30, 3] = 255
    return img

def test_person_standee_and_billboard_geometry():
    from avsectester.attack.implemented_attacks.object_insertion_attack.assets import billboard, standee

    (surface,) = standee(_person(), height_m=1.75).planes()
    face = surface.corners  # no post
    assert np.isclose(face[0, 2] - face[3, 2], 1.75) and np.isclose(face[3, 2], 0.0)  # on the ground
    assert np.isclose(np.linalg.norm(face[1] - face[0]), 1.75 * 40 / 120)  # keeps the cut-out's aspect
    planes = billboard(_person(), width_m=1.4, mount_height_m=0.6).planes()
    assert len(planes) == 3  # two legs + the board
    legs = [p.corners for p in planes[:2]]
    assert not np.allclose(legs[0], legs[1]) and all(np.isclose(leg[3, 2], 0.0) for leg in legs)
    board = planes[-1].corners
    assert np.isclose(board[3, 2], 0.6) and np.isclose(np.linalg.norm(board[1] - board[0]), 1.4)

def test_poster_prints_person_on_opaque_paper():
    from avsectester.attack.implemented_attacks.object_insertion_attack.assets import poster_rgba

    p = poster_rgba(_person(), aspect=1.5)
    assert p.shape[0] / p.shape[1] == pytest.approx(1.5, rel=0.01)
    assert (p[..., 3] == 255).all()  # opaque board
    mid = p[p.shape[0] // 2, p.shape[1] // 2, :3]
    assert tuple(mid) == (60, 70, 90)  # the figure is printed in the centre
    assert tuple(p[p.shape[0] // 2, p.shape[1] // 8, :3]) == (228, 226, 218)  # paper around it

def test_traffic_lights_payload():
    from avsectester.attack.implemented_attacks.object_insertion_attack.assets import (
        roadside_rig,
        signal_head_rgba,
        traffic_lights_rgba,
    )

    head = signal_head_rgba("red", texture_height_pixels=300)
    assert head.shape[2] == 4 and (head[..., 3] == 255).any()  # RGBA, has opaque pixels
    # the lit (top, red) third is redder than the dark (bottom, green-off) third
    top = head[: head.shape[0] // 3][head[: head.shape[0] // 3, :, 3] > 0]
    bot = head[2 * head.shape[0] // 3:][head[2 * head.shape[0] // 3:, :, 3] > 0]
    assert top[:, 0].mean() > bot[:, 0].mean() + 20

    board = traffic_lights_rgba(head_count=3, lit="red", texture_height_pixels=240)
    assert (board[..., 3] == 255).all()  # opaque board behind the heads
    assert 0.7 < (board.shape[0] / board.shape[1]) < 1.0  # a wide-ish board (3 heads in a row)
    assert board.shape[1] > board.shape[0]  # wider than tall

    face = roadside_rig(board, mount_height_m=2.2).planes()[-1].corners
    assert np.isclose(face[3, 2], 2.2)  # bottom edge at the mount height


def test_traffic_light_texture_resolution_is_independent_of_world_width():
    from avsectester.attack.implemented_attacks.object_insertion_attack.assets import TrafficLightAsset

    smaller = TrafficLightAsset(head_count=2, texture_height_pixels=120, width_m=3.)
    larger = TrafficLightAsset(head_count=2, texture_height_pixels=240, width_m=3.)
    small_face, large_face = smaller.planes()[-1], larger.planes()[-1]
    assert large_face.texture.shape[0] > small_face.texture.shape[0]
    assert np.linalg.norm(small_face.corners[1] - small_face.corners[0]) == pytest.approx(3.)
    assert np.linalg.norm(large_face.corners[1] - large_face.corners[0]) == pytest.approx(3.)


def test_sign_post_count_changes_only_number_of_posts():
    single = SignAsset(post_count=1).planes()
    multiple = SignAsset(post_count=3).planes()
    assert len(single) == 2 and len(multiple) == 4
    np.testing.assert_array_equal(single[-1].corners, multiple[-1].corners)
