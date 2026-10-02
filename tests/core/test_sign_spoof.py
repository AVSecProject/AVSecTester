"""Fake-sign attack + world-anchored insertion: camera models, plane rendering, placement, perturb seam."""

import math

import numpy as np
import pytest
from avsectester.simulators.camera_models import (
    FThetaCamera,
    PinholeCamera,
    make_pose,
    planar_rig_pose,
    quat_to_matrix,
)

# the NuRec camera_front_wide_120fov f-theta intrinsics + rig extrinsic (scene clipgt-01d503d4)
FTHETA = FThetaCamera(
    cx=956.11, cy=754.85,
    angle_to_pixeldist=(0, 944.4935, -10.977355, 32.704746, -77.398132, 32.516323),
    pixeldist_to_angle=(0, 1.0570853e-3, 2.9625948e-08, -9.3092818e-11, 1.6680121e-13, -6.9934230e-17),
    width=1920, height=1080, max_angle=1.35,
)
RIG_FROM_CAM = make_pose(quat_to_matrix(-0.496408641, 0.504378, -0.495750666, 0.503401), (2.016, -0.061, 1.591))


def _cam_from_world(x=0.0, y=0.0, yaw=0.0):
    return np.linalg.inv(planar_rig_pose(x, y, yaw) @ RIG_FROM_CAM)


def test_ftheta_project_unproject_roundtrip():
    rays = np.array([[0, 0, 1], [0.3, 0.1, 1], [-0.8, 0.3, 1], [1.2, -0.4, 1.0]])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    uv = FTHETA.project(rays * 7.0)
    assert np.allclose(uv[0], [FTHETA.cx, FTHETA.cy])  # optical axis -> principal point
    back = FTHETA.unproject(uv)
    assert np.degrees(np.arccos(np.clip((back * rays).sum(1), -1, 1))).max() < 0.01


def test_pinhole_roundtrip():
    cam = PinholeCamera(np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]), 640, 480)
    pts = np.array([[1.0, -0.5, 4.0], [0.0, 0.0, 2.0]])
    rays = cam.unproject(cam.project(pts))
    assert np.allclose(rays, pts / np.linalg.norm(pts, axis=1, keepdims=True))


def test_rig_extrinsic_is_forward_looking():
    """NuRec camera axes (x right, y down, z fwd) map to rig (-y, -z, +x): a point ahead of the ego
    lands near the image centre column, below the principal point when it is on the ground."""
    pts = np.array([[20.0, 0.0, 0.0]])  # 20 m ahead on the ground
    cam = (_cam_from_world()[:3, :3] @ pts.T).T + _cam_from_world()[:3, 3]
    assert cam[0, 2] > 17  # in front
    u, v = FTHETA.project(cam)[0]
    assert abs(u - FTHETA.cx) < 20 and v > FTHETA.cy


def test_roadside_sign_planes_geometry():
    from avsectester.attacks.sign_spoof import RoadsideSign

    s = RoadsideSign(x=20.0, y=-5.0, width=0.9, mount_height=1.5)
    (post, _), (face, tex) = s.planes()
    assert tex.shape[2] == 4 and tex[0, 0, 3] == 0  # RGBA, transparent outside the octagon
    assert np.allclose(face[:, 0], 20.0)  # yaw 0: the face is a plane of constant x
    assert face[0, 1] > face[1, 1]  # TL is on the viewer's left (+y) when facing the ego
    assert np.isclose(face[2, 2], 1.5) and np.isclose(face[0, 2] - face[3, 2], 0.9)
    assert np.isclose(post[3, 2], 0.0) and post[0, 0] > 20.0  # post: from the ground, behind the face
    turned = RoadsideSign(x=20.0, y=-5.0, yaw=math.radians(20)).planes()[-1][0]
    assert not np.allclose(turned[:, 0], 20.0)  # yaw rotates the face out of the x-plane


def test_render_plane_lands_where_projected_and_keeps_red():
    from avsectester.attacks.sign_spoof import RoadsideSign
    from avsectester.simulators.patch_insertion import (
        ClassicHarmonizer,
        PatchCompositor,
        render_plane,
    )

    frame = np.full((1080, 1920, 3), 90, np.uint8)
    sign = RoadsideSign(x=12.0, y=-4.0, post=False)
    corners, face = sign.planes()[0]
    _comp, mask = render_plane(frame, FTHETA, _cam_from_world(), corners, face)
    ys, xs = np.where(mask > 0)
    centre_world = corners.mean(axis=0)[None]
    c = (_cam_from_world()[:3, :3] @ centre_world.T).T + _cam_from_world()[:3, 3]
    u, v = FTHETA.project(c)[0]
    assert abs(xs.mean() - u) < 3 and abs(ys.mean() - v) < 3  # rendered footprint centred on the projection
    assert xs.mean() > FTHETA.cx  # right of centre (negative y)
    # chroma-preserving harmonization keeps the sign red; the default Lab transfer pulls it to grey
    kept = PatchCompositor(ClassicHarmonizer(preserve_chroma=True, blend="feather")).apply_planes(
        frame, FTHETA, _cam_from_world(), [(corners, face)])
    r, g, b = kept[mask > 0].astype(float).mean(axis=0)
    assert r > g + 30 and r > b + 30


def test_render_plane_behind_camera_is_noop():
    from avsectester.attacks.sign_spoof import RoadsideSign
    from avsectester.simulators.patch_insertion import render_plane

    frame = np.full((1080, 1920, 3), 90, np.uint8)
    corners, face = RoadsideSign(x=-10.0, y=-4.0, post=False).planes()[0]  # behind the ego
    comp, mask = render_plane(frame, FTHETA, _cam_from_world(), corners, face)
    assert not mask.any() and np.array_equal(comp, frame)


def test_frame_perturbation_rewrites_only_the_camera():
    from avsectester.plane import Observation
    from avsectester.simulators.patch_insertion import frame_perturbation

    rgb = np.zeros((4, 4, 3), np.uint8)
    obs = Observation(t=0.0, frame=0, sensor_data={"cam": rgb, "other": "lidar"})
    seen = frame_perturbation(lambda o, im: im + 7, camera="cam")(obs)
    assert (seen.sensor_data["cam"] == 7).all() and seen.sensor_data["other"] == "lidar"
    assert (obs.sensor_data["cam"] == 0).all()  # the backend's true observation is untouched


@pytest.mark.parametrize("custom_view", [False, True], ids=["default-view", "custom-view"])
def test_frame_perturbation_reads_and_rewrites_the_selected_camera(custom_view):
    from avsectester.plane import Observation
    from avsectester.simulators.patch_insertion import frame_perturbation

    other = np.full((4, 4, 3), 90, np.uint8)
    target = np.zeros((4, 4, 3), np.uint8)
    obs = Observation(t=0.0, frame=0, sensor_data={"other": other, "target": target})
    calls = []

    def view(observation):
        calls.append(observation)
        return observation.sensor_data["target"] + 3

    kwargs = {"base": view} if custom_view else {}
    seen = frame_perturbation(lambda o, rgb: rgb + 7, camera="target", **kwargs)(obs)

    assert (seen.sensor_data["target"] == (10 if custom_view else 7)).all()
    assert seen.sensor_data["other"] is other
    assert (other == 90).all() and (target == 0).all()
    if custom_view:
        assert len(calls) == 1 and calls[0] is obs


def test_lane_pick_and_hold():
    from avsectester.simulators.patch_insertion import detector_quad, hold_quad

    frame = np.zeros((100, 200, 3), np.uint8)
    lead = ([90, 40, 130, 60], 0.9, 8)  # spans the centre column (x=100): the ego-lane lead
    big_side = ([10, 30, 95, 90], 0.9, 3)  # larger, nearer, but in the next lane
    boxes = [[lead, big_side], [], [], [], [], [lead]]
    detect = lambda _rgb: boxes.pop(0)
    quad_of = hold_quad(detector_quad(detect, base=lambda _o: frame, pick="lane", aspect=1.0), frames=3)
    q0 = quad_of(None)
    assert q0 is not None and 90 < q0[:, 0].mean() < 130  # took the in-lane lead, not the bigger car
    assert np.allclose(q0[2] - q0[1], [0, q0[1, 0] - q0[0, 0]])  # aspect=1: square
    held = [quad_of(None) for _ in range(4)]
    assert all(np.array_equal(h, q0) for h in held[:3]) and held[3] is None  # held 3 misses, then dropped
    assert quad_of(None) is not None  # re-acquired


def _person(h=120, w=40):
    """A synthetic cut-out: an opaque person-shaped column on a transparent background."""
    img = np.zeros((h, w, 4), np.uint8)
    img[:, 10:30, :3] = (60, 70, 90)
    img[:, 10:30, 3] = 255
    return img


def test_person_standee_and_billboard_geometry():
    from avsectester.attacks.person_poster import billboard, standee

    ((face, _tex),) = standee(_person(), x=15.0, y=-3.0, height=1.75).planes()  # no post
    assert np.isclose(face[0, 2] - face[3, 2], 1.75) and np.isclose(face[3, 2], 0.0)  # on the ground
    assert np.isclose(np.linalg.norm(face[1] - face[0]), 1.75 * 40 / 120)  # keeps the cut-out's aspect
    planes = billboard(_person(), x=15.0, y=-6.0, width=1.4, mount_height=0.6).planes()
    assert len(planes) == 3  # two legs + the board
    legs = [p for p, _ in planes[:2]]
    assert not np.allclose(legs[0], legs[1]) and all(np.isclose(leg[3, 2], 0.0) for leg in legs)
    board = planes[-1][0]
    assert np.isclose(board[3, 2], 0.6) and np.isclose(np.linalg.norm(board[1] - board[0]), 1.4)


def test_poster_prints_person_on_opaque_paper():
    from avsectester.attacks.person_poster import poster_rgba

    p = poster_rgba(_person(), aspect=1.5)
    assert p.shape[0] / p.shape[1] == pytest.approx(1.5, rel=0.01)
    assert (p[..., 3] == 255).all()  # opaque board
    mid = p[p.shape[0] // 2, p.shape[1] // 2, :3]
    assert tuple(mid) == (60, 70, 90)  # the figure is printed in the centre
    assert tuple(p[p.shape[0] // 2, p.shape[1] // 8, :3]) == (228, 226, 218)  # paper around it


def test_traffic_lights_payload():
    from avsectester.attacks.traffic_light import (
        aspect_of,
        roadside_rig,
        signal_head_rgba,
        traffic_lights_rgba,
    )

    head = signal_head_rgba("red", height=300)
    assert head.shape[2] == 4 and (head[..., 3] == 255).any()  # RGBA, has opaque pixels
    # the lit (top, red) third is redder than the dark (bottom, green-off) third
    top = head[: head.shape[0] // 3][head[: head.shape[0] // 3, :, 3] > 0]
    bot = head[2 * head.shape[0] // 3:][head[2 * head.shape[0] // 3:, :, 3] > 0]
    assert top[:, 0].mean() > bot[:, 0].mean() + 20

    board = traffic_lights_rgba(n=3, lit="red", height=240)
    assert (board[..., 3] == 255).all()  # opaque board behind the heads
    assert 0.7 < aspect_of(board) < 1.0  # a wide-ish board (3 heads in a row)
    assert board.shape[1] > board.shape[0]  # wider than tall

    ((face, _), *_rest) = roadside_rig(board, x=26.0, y=-6.0, mount_height=2.2).planes()[-1:]
    assert np.isclose(face[3, 2], 2.2)  # bottom edge at the mount height
