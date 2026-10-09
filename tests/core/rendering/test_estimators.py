"""Visibility must include image truncation and distinguish evidence from approximation."""

import numpy as np
import pytest

from avsectester.insertion import (
    ActorPose,
    Insertion,
    Orientation,
    PlaneAsset,
    WorldPlacement,
    resolve_insertion,
)
from avsectester.rendering.cameras import FThetaCamera, PinholeCamera, camera_from_calibration
from avsectester.rendering.geometry import projected_silhouette
from avsectester.rendering.visibility import CuboidVisibilityEstimator, DepthVisibilityEstimator
from avsectester.scenarios.estimators import resolved_to_target
from avsectester.scenarios.scene import CameraCalib


CAM_FROM_WORLD = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1.0]])
CAMERA = PinholeCamera(np.array([[50, 0, 50], [0, 50, 40], [0, 0, 1.0]]), 100, 80)


def subject(position=(10, 0, 0), width=4, height=2, texture=None, id="panel"):
    if texture is None:
        texture = np.full((8, 8, 4), 255, dtype=np.uint8)
    return resolve_insertion(
        Insertion(
            id,
            PlaneAsset(texture, width, height),
            WorldPlacement(position),
            Orientation("fixed_world", (0, 0, 180)),
        ),
        {},
        ActorPose(np.eye(4)),
    )


def box(position, extent):
    pose = np.eye(4)
    pose[:3, 3] = position
    return ActorPose(pose, extent)


def estimate(item=None, occluders=None, camera=CAMERA, **kwargs):
    return CuboidVisibilityEstimator().estimate(
        item or subject(),
        camera,
        CAM_FROM_WORLD,
        occluders=occluders or {},
        camera_name="front",
        **kwargs,
    )


def test_complete_target_silhouette_includes_out_of_frame_pixels():
    clear = estimate()
    half_out = estimate(subject((10, 10, 0)))
    outside = estimate(subject((10, 30, 0)))
    assert clear.visibility.fraction == 1
    assert clear.reference_pixels == 200
    assert half_out.reference_pixels == clear.reference_pixels
    assert half_out.visibility.fraction == 0.5
    assert outside.visibility.fraction == 0
    assert outside.reference_pixels == 200
    assert not outside.visible_mask.any()


def test_depth_ordered_boxes_block_only_foreground_portion():
    half = estimate(occluders={"car": box((5, 1, 0), (1, 2, 4))})
    behind = estimate(occluders={"car": box((15, 0, 0), (1, 10, 10))})
    blocked = estimate(occluders={"car": box((5, 0, 0), (1, 10, 10))})
    assert half.visibility.fraction == 0.5
    assert half.visibility.source == "cuboid_estimate"
    assert behind.visibility.fraction == 1
    assert blocked.visibility.fraction == 0


def test_transparency_is_not_counted_in_target_silhouette():
    texture = np.full((8, 8, 4), 255, dtype=np.uint8)
    texture[:, :4, 3] = 0
    result = estimate(subject(texture=texture))
    assert result.reference_pixels == 100
    assert result.visible_mask.sum() == 100
    assert result.visibility.fraction == 1


def test_front_facing_texture_preserves_left_and_right():
    texture = np.full((8, 8, 4), 255, dtype=np.uint8)
    texture[:, :4, 3] = 0  # Left side of the authored texture is transparent.
    panel = subject(texture=texture)
    result = estimate(panel)
    assert not result.reference_mask[:, :50].any()
    assert result.reference_mask[:, 50:].sum() == 100
    corners = panel.planes()[0].corners
    projected = CAMERA.project(corners @ CAM_FROM_WORLD[:3, :3].T)
    np.testing.assert_allclose(projected, [[40, 35], [60, 35], [60, 45], [40, 45]])


def test_other_insertions_occlude_each_other_with_alpha():
    blocker = subject((5, 1, 0), width=2, height=4, id="other")
    result = estimate(other_insertions=(blocker, subject()))
    assert result.visibility.fraction == 0.5


def test_host_box_remains_an_occluder():
    # A panel accidentally inside the host is not reported fully visible.
    result = estimate(occluders={"host": box((10, 0, 0), (4, 4, 4))})
    assert result.visibility.fraction == 0
    same_id = estimate(subject(id="host"), occluders={"host": box((10, 0, 0), (4, 4, 4))})
    assert same_id.visibility.fraction == 0


@pytest.mark.parametrize("convention", ["z", "range"])
def test_depth_provider_compares_matching_depth_units(convention):
    plane = subject()
    depth = np.full((80, 100), 10.0)
    if convention == "range":
        xs, ys = np.meshgrid(np.arange(100) + 0.5, np.arange(80) + 0.5)
        rays = CAMERA.unproject(np.column_stack((xs.ravel(), ys.ravel())))
        depth /= rays[:, 2].reshape(80, 100)
    depth[:, :50] = 4
    result = DepthVisibilityEstimator().estimate(
        plane,
        CAMERA,
        CAM_FROM_WORLD,
        scene_depth=depth,
        depth_convention=convention,
        camera_name="front",
    )
    assert result.visibility.fraction == 0.5
    assert result.visibility.source == "depth_comparison"
    np.testing.assert_allclose(result.target_depth[result.reference_mask], 10)


def test_missing_depth_inside_target_is_unknown_but_elsewhere_is_irrelevant():
    depth = np.full((80, 100), np.nan)
    depth[35:45, 40:60] = 100
    estimator = DepthVisibilityEstimator()
    kwargs = dict(scene_depth=depth, depth_convention="z", camera_name="front")
    result = estimator.estimate(subject(), CAMERA, CAM_FROM_WORLD, **kwargs)
    assert result.visibility.fraction == 1
    depth[40, 50] = np.nan
    unknown = estimator.estimate(subject(), CAMERA, CAM_FROM_WORLD, **kwargs)
    assert unknown.visibility is None
    assert "Invalid depth" in unknown.reason


def test_behind_camera_is_zero_and_reference_budget_failure_is_unknown():
    assert estimate(subject((-10, 0, 0))).visibility.fraction == 0
    result = CuboidVisibilityEstimator(max_reference_pixels=100).estimate(
        subject(),
        CAMERA,
        CAM_FROM_WORLD,
        occluders={},
        camera_name="front",
    )
    assert result.visibility is None
    assert "max_reference_pixels" in result.reason
    oversized = estimate(subject(width=1e10, height=1e10))
    assert oversized.visibility is None
    assert "max_reference_pixels" in oversized.reason


def test_fisheye_rays_support_occlusion_and_offscreen_denominator():
    fisheye = FThetaCamera(50, 40, (0, 50), 100, 80, (0, 1 / 50), max_angle=1.5)
    clear = estimate(camera=fisheye)
    half = estimate(camera=fisheye, occluders={"car": box((5, 1, 0), (1, 2, 4))})
    truncation = estimate(subject((10, 15, 0), width=10), camera=fisheye)
    assert clear.visibility.fraction == 1
    assert half.visibility.fraction == 0.5
    assert 0 < truncation.visibility.fraction < 1
    assert truncation.reference_pixels > truncation.reference_mask.sum()


@pytest.mark.parametrize(
    "camera",
    [
        CAMERA,
        FThetaCamera(50, 40, (0, 50), 100, 80, (0, 1 / 50), max_angle=1.2),
    ],
)
def test_image_silhouette_agrees_with_visibility_reference(camera):
    texture = np.full((8, 8, 4), 255, np.uint8)
    texture[2:6, 2:6, 3] = 0
    item = subject((10, 12, 0), width=10, texture=texture)
    projected = projected_silhouette(item, camera, CAM_FROM_WORLD)
    np.testing.assert_array_equal(projected, estimate(item, camera=camera).reference_mask)


def test_disconnected_surfaces_do_not_fill_their_joint_bounding_box():
    from types import SimpleNamespace

    left = subject((10, 30, 0)).planes()
    right = subject((10, -30, 0)).planes()
    item = SimpleNamespace(planes=lambda: (*left, *right))
    assert not projected_silhouette(item, CAMERA, CAM_FROM_WORLD).any()
    transparent = subject(texture=np.zeros((8, 8, 4), np.uint8))
    assert not projected_silhouette(transparent, CAMERA, CAM_FROM_WORLD).any()


def test_calibration_adapters_and_projected_target_share_ego_geometry():
    camera_to_ego = np.linalg.inv(CAM_FROM_WORLD)
    calib = CameraCalib("front", 100, 80, model=(50, 50, 40), cam_to_ego=camera_to_ego)
    lens = camera_from_calibration(calib)
    np.testing.assert_allclose(lens.project([[0, 0, 10]]), [[50, 40]])
    candidate = resolved_to_target(subject(), ActorPose(np.eye(4)), {"front": calib})
    np.testing.assert_allclose(candidate.center, [10, 0, 0])
    np.testing.assert_allclose(candidate.box2d["front"], [40, 35, 60, 45])


def test_dataset_fisheye_wrapper_uses_optical_frame_and_inverts_forward_polynomial():
    from avsectester.scenarios.datasets.nurec import FThetaCamera as DatasetCamera

    model = DatasetCamera(
        100,
        80,
        np.array([50.0, 40.0]),
        np.array([0.0, 50.0]),
        1.5,
        np.linalg.inv(CAM_FROM_WORLD),
    )
    calib = CameraCalib("front", 100, 80, model, model.t_sensor_rig)
    camera = camera_from_calibration(calib)
    rays = np.array([[0, 0, 1], [0.3, -0.2, 1.0]])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    np.testing.assert_allclose(camera.unproject(camera.project(rays)), rays, atol=1e-10)
    assert estimate(camera=camera).visibility.fraction == 1
