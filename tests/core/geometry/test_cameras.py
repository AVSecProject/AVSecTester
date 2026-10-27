"""Pinhole and polynomial fisheye cameras share projection and inverse-ray numerics."""
import numpy as np
import pytest

from avsectester.geometry.cameras import (
    BODY_FROM_OPTICAL, FThetaCamera, WindshieldDistortion, camera_ray_mask,
)


def test_project(camera, lens):
    delta = lens.offset
    points = np.array([[0., 0., 5.], [1., 0., 5.], [0., -1., 5.]])
    pixels = camera.project(points)
    np.testing.assert_allclose(pixels[0], [50, 40])
    np.testing.assert_allclose(pixels[1], [50 + delta, 40])
    np.testing.assert_allclose(pixels[2], [50, 40 - delta])
    np.testing.assert_allclose(camera.project(points * 2), pixels)


def test_unproject(camera):
    points = np.array([[0., 0., 1.], [.3, -.2, 1.], [-.7, .5, 1.]])
    rays = camera.unproject(camera.project(points))
    np.testing.assert_allclose(rays, points / np.linalg.norm(points, axis=1, keepdims=True), atol=1e-10)


def test_camera_ray_mask(camera):
    rays = np.array([[0., 0., 1.], [0., 0., -1.], [np.nan, 0., 1.]])
    np.testing.assert_array_equal(camera_ray_mask(camera, rays), [True, False, False])


def test_body_from_optical():
    np.testing.assert_array_equal(BODY_FROM_OPTICAL @ [1., 2., 3., 1.], [3., -1., -2., 1.])


# Front wide camera of NuRec scene 023b7fcc (rig_trajectories.json), order-2 and order-4 polynomials.
SCENE_WINDSHIELD = WindshieldDistortion(
    (-0.00353792007, 1.00160002708, 4.7415298e-05, -0.00278471992, 0.00146400998, 0.00199894002),
    (0.00510897022, 0.00200385996, 0.00182037998, -0.00112585002, 0.00046490799, 1.01366996765,
     -0.00010489000, 0.00970146991, -0.00227264990, 0.01954500005, -0.00022134000, 0.01385449991,
     0.01996220089, -0.00391718978, 0.00865260977),
    (0.00351875136, 0.99840921164, -5.0843886e-05, 0.00275669992, -0.00142548024, -0.00200445950),
    (-0.00505727343, -0.00196955842, -0.00171427359, 0.00107042340, -0.00043105835, 0.98662668467,
     7.872558e-05, -0.00886319671, 0.00216872734, -0.01821293868, -0.00011304879, -0.01281992905,
     -0.01728768833, 0.00363773433, -0.00695833098))


def test_windshield_distortion():
    identity = WindshieldDistortion((0., 1., 0., 0., 0., 0.), (0., 0., 0., 1., 0., 0.),
                                    (0., 1., 0., 0., 0., 0.), (0., 0., 0., 1., 0., 0.))
    rays = np.array([[0., 0., 1.], [.3, -.2, 1.], [-.4, .25, 1.]])
    unit = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    np.testing.assert_allclose(identity.distort(rays), unit, atol=1e-12)
    # Blocks: c0 + c1*phi + c2*phi^2, then (c3 + c4*phi)*theta, then c5*theta^2.
    model = WindshieldDistortion((.01, 1., .1, 0., .2, 0.), (0., 0., 0., 1., 0., .3),
                                 identity.horizontal_poly_inverse, identity.vertical_poly_inverse)
    phi, theta = .2, .1
    ray = np.array([[np.sin(phi), np.sin(theta), np.sqrt(1 - np.sin(phi) ** 2 - np.sin(theta) ** 2)]])
    np.testing.assert_allclose(np.arcsin(model.distort(ray)[0, :2]),
                               [.01 + phi + .1 * phi ** 2 + .2 * phi * theta, theta + .3 * theta ** 2])
    lens = FThetaCamera(50, 40, (0, 50), 100, 80, (0, .02), max_angle=1.5)
    seen = FThetaCamera(50, 40, (0, 50), 100, 80, (0, .02), max_angle=1.5, windshield=model)
    np.testing.assert_allclose(seen.project(ray), lens.project(model.distort(ray)))
    # The calibrated inverse undoes the forward deflection; unprojection returns scene rays.
    np.testing.assert_allclose(SCENE_WINDSHIELD.undistort(SCENE_WINDSHIELD.distort(unit)), unit, atol=2e-4)
    scene = FThetaCamera(962.25, 744.12, (0., 940.31, 12.61, -30.50, -1.09, .31), 1920, 1080,
                         (0., 1.0637e-3, -1.7729e-8, 4.7888e-11, -1.2797e-14, 9.6638e-18),
                         max_angle=1.35, windshield=SCENE_WINDSHIELD)
    np.testing.assert_allclose(scene.unproject(scene.project(unit)), unit, atol=2e-4)
    with pytest.raises(ValueError, match="triangular"):
        WindshieldDistortion((0., 1.), (0., 0., 0., 1., 0., 0.), (0., 1., 0., 0., 0., 0.), (0., 0., 0., 1., 0., 0.))


def test_ftheta_in_view():
    camera = FThetaCamera(50., 40., (0., 50.), 100, 80, (0., .02), max_angle=.5)
    points = np.array([[0., 0., 1.],
                       [np.sin(.4), 0., np.cos(.4)],
                       [np.sin(.6), 0., np.cos(.6)],
                       [0., 0., -1.], [np.nan, 0., 1.]])
    np.testing.assert_array_equal(camera.in_view(points), [True, True, False, False, False])


def test_linear_screen_correction():
    camera = FThetaCamera(50., 40., (0., 50.), 100, 80, (0., .02), max_angle=1.5,
                          linear_cde=(1.1, .2, -.1))
    rays = np.array([[np.sin(.4), 0., np.cos(.4)],
                     [0., np.sin(.2), np.cos(.2)], [0., 0., 1.]])
    # Radial offsets are (20, 0), (0, 10) and (0, 0), before the affine screen correction.
    pixels = np.array([[72., 38.], [52., 50.], [50., 40.]])
    np.testing.assert_allclose(camera.project(rays), pixels, atol=1e-12)
    np.testing.assert_allclose(camera.unproject(pixels), rays, atol=1e-12)
