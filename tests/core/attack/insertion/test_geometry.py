"""Ray/slab intersections and nearest opaque planar samples."""
import numpy as np
import pytest
from avsectester.attack.insertion.geometry import box_intersections, prepare_surface
from avsectester.attack.insertion import PlaneSurface
from avsectester.geometry.poses import ActorPose, rotation_matrix


def test_box_intersections():
    transform = np.eye(4)
    transform[:3, 3] = (0, 0, 5)
    rays = np.array([[0., 0., 1.], [1., 0., 0.], [0., 0., -1.]])
    distances = box_intersections(ActorPose(transform, (2, 2, 2)), rays, np.eye(4))
    np.testing.assert_array_equal(distances, [4., np.inf, np.inf])
    transform[:3, :3] = rotation_matrix((0, 0, 45))
    rotated = box_intersections(ActorPose(transform, (4, 2, 2)), rays, np.eye(4))
    assert rotated[0] == pytest.approx(4.)
    np.testing.assert_array_equal(box_intersections(ActorPose(np.eye(4), (2, 2, 2)),
                                                   rays, np.eye(4)), [1, 1, 1])
    with pytest.raises(ValueError, match="extent"):
        box_intersections(ActorPose(transform), rays, np.eye(4))


def test_surface_sample(camera):
    texture = np.full((8, 8, 4), [255, 0, 0, 255], np.uint8)
    surface = PlaneSurface(np.array([[-1., -1., 5.], [1., -1., 5.],
                                    [1., 1., 5.], [-1., 1., 5.]]), texture)
    sampler = prepare_surface(surface, camera, np.eye(4))
    rays = np.array([[0., 0., 1.], [1., 0., 1.]])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    distances, colors = sampler.sample(rays)
    assert distances[0] == pytest.approx(5.) and np.isinf(distances[1])
    np.testing.assert_array_equal(colors[0], [255, 0, 0, 255])
