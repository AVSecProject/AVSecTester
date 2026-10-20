"""Native panel construction and RGBA textures remain explicit and independent of CARLA."""
import numpy as np

from avsectester.attack.insertion.assets import CheckerboardTexture, ImageTexture
from avsectester.attack.implemented_attacks.patch_attack.online import CarlaPatchPanel, DEFAULT_PROP


def test_checkerboard_is_rgba_and_alternates():
    image = CheckerboardTexture(size=64, cells=4, colors=((255, 0, 0), (255, 255, 255))).rgba()
    assert image.shape == (64, 64, 4) and image.dtype == np.uint8
    assert (image[..., 3] == 255).all()
    assert tuple(image[0, 0, :3]) == (255, 0, 0)
    assert tuple(image[0, 16, :3]) == (255, 255, 255)
    assert tuple(image[16, 16, :3]) == (255, 0, 0)


def test_panel_uses_explicit_transform_and_texture():
    texture = CheckerboardTexture().rgba()
    panel = CarlaPatchPanel(location=(-2., 0., .74), rotation=(90., 0., 0.), texture=texture)
    assert panel.prop == DEFAULT_PROP and panel.emissive
    np.testing.assert_array_equal(panel.texture, texture)


def test_image_texture_loads_rgba_once(tmp_path):
    from PIL import Image
    path = tmp_path / "patch.png"
    Image.fromarray(np.full((10, 20, 3), 128, np.uint8)).save(path)
    texture = ImageTexture(path)
    image = texture.rgba()
    assert image.shape == (10, 20, 4) and image.dtype == np.uint8
    image[:] = 0
    assert texture.rgba().any()


def test_local_patch_mount_uses_rotated_box_center_without_world_round_trip(monkeypatch):
    from contextlib import ExitStack
    from types import SimpleNamespace
    from avsectester.attack import AttackSegment
    from avsectester.attack.implemented_attacks.patch_attack import PatchAttack
    from avsectester.attack.placement import AttachedPlacement, Orientation
    from avsectester.closed_loop.datatypes import WorldSnapshot
    from avsectester.datasets import ActorState, FrameReference
    from avsectester.geometry.poses import rotation_matrix

    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix((12., 20., 33.))
    pose[:3, 3] = [1000., -2000., 4.]
    state = ActorState(pose, extent=(4., 2., 1.6), box_center=(.3, -.2, .74),
                       box_rotation=(5., 10., 15.))
    start, stop = FrameReference("s", 0, 0, 0.), FrameReference("s", 2, 2, .2)
    placement = AttachedPlacement("host", "rear_center", (-.01, .1, -.2))
    segment = AttackSegment("s", start, stop, roles={"host": ("host",)}, placement=placement,
                            user_data={"deployment": "world"})
    attack = PatchAttack(decision_steps=2, orientation=Orientation("follow_host", (0., 0., 180.)))
    captured = []
    monkeypatch.setattr(CarlaPatchPanel, "apply", lambda panel, *args, **kwargs: captured.append(panel))
    simulator_runtime = SimpleNamespace(native={"world": object(), "actors": {"host": object()}},
        state=lambda: WorldSnapshot(0., 0, actors={"host": state}), setup_tick=lambda: None)
    with ExitStack() as resources:
        attack.create_runtime(segment, attack.assets).setup(None)[0].apply(simulator_runtime, resources)
    panel, = captured
    flip = np.diag([1., -1., 1.])
    box_rotation = rotation_matrix(state.box_rotation)
    expected_position = flip @ (np.asarray(state.box_center)
                               + box_rotation @ placement.local_position(state.extent))
    np.testing.assert_allclose(panel.location, expected_position)
    correction = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
    expected_rotation = flip @ box_rotation @ rotation_matrix((0., 0., 180.)) @ flip @ correction
    pitch, yaw, roll = panel.rotation
    np.testing.assert_allclose(rotation_matrix((-roll, -pitch, yaw)), expected_rotation, atol=1e-12)


def test_native_patch_requires_host_extent(monkeypatch):
    from contextlib import ExitStack
    from types import SimpleNamespace
    import pytest
    from avsectester.attack import AttackSegment
    from avsectester.attack.implemented_attacks.patch_attack import PatchAttack
    from avsectester.attack.placement import AttachedPlacement
    from avsectester.closed_loop.datatypes import WorldSnapshot
    from avsectester.datasets import ActorState, FrameReference

    attack = PatchAttack(decision_steps=1)
    segment = AttackSegment("s", FrameReference("s", 0, 0, 0.), FrameReference("s", 1, 1, .1),
        placement=AttachedPlacement("host"), user_data={"deployment": "world"})
    simulator_runtime = SimpleNamespace(native={"world": object(), "actors": {"host": object()}},
        state=lambda: WorldSnapshot(0., 0, actors={"host": ActorState(np.eye(4))}))
    with ExitStack() as resources, pytest.raises(ValueError, match="extent"):
        attack.create_runtime(segment, attack.assets).setup(None)[0].apply(simulator_runtime, resources)
