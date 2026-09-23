"""Scenario requirement DSL — the predicate selects the right target and enforces each constraint."""

from avsectester.scenarios import CameraCalib, EgoState, ObjectGT, SceneGT
from avsectester.scenarios.requirement import DistanceRange, ImageAreaFrac, ViewpointRear
from avsectester.scenarios.requirements import PHYSICAL_PATCH_HIDE_VEHICLE

CAM = {"front": CameraCalib(name="front", width=800, height=600)}


def _vehicle(track, x, box, yaw=0.0, vis=1.0, category="vehicle"):
    return ObjectGT(track_id=track, category=category, center=(x, 0.0, 0.0), extent=(4.5, 2.0, 1.5),
                    yaw=yaw, box2d={"front": box}, visibility=vis)


def _scene(objects, speed=5.0):
    return SceneGT(frame=0, t=0.0, ego=EgoState(speed=speed), cameras=CAM, objects=objects)


def test_requirement_matches_qualifying_vehicle_and_selects_nearest_ahead():
    near = _vehicle("near", 8.0, (330, 250, 470, 400))   # ~0.058 area frac, rear-facing, in view
    far = _vehicle("far", 60.0, (395, 295, 405, 305))    # too far, tiny box
    m = PHYSICAL_PATCH_HIDE_VEHICLE.match(_scene([far, near]))
    assert m is not None and m.target.track_id == "near" and m.camera == "front"


def test_requirement_rejects_when_a_constraint_fails():
    req = PHYSICAL_PATCH_HIDE_VEHICLE
    assert req.match(_scene([_vehicle("v", 40.0, (395, 295, 405, 305))])) is None  # DistanceRange fails
    assert req.match(_scene([_vehicle("v", 8.0, (330, 250, 470, 400), yaw=3.0)])) is None  # ViewpointRear
    assert req.match(_scene([_vehicle("v", 8.0, (330, 250, 470, 400), vis=0.2)])) is None  # MinVisibility
    assert req.match(_scene([_vehicle("v", 8.0, (0, 0, 800, 600))])) is None  # ImageAreaFrac too big
    assert req.match(_scene([_vehicle("v", 8.0, (330, 250, 470, 400), category="pedestrian")])) is None


def test_individual_constraints():
    v = _vehicle("v", 10.0, (300, 200, 500, 400))
    scene = _scene([v])
    assert DistanceRange(4, 25).holds(scene, v) and not DistanceRange(0, 5).holds(scene, v)
    assert ViewpointRear(35).holds(scene, v) and not ViewpointRear(5).holds(scene, _vehicle("w", 10, (1, 1, 2, 2), yaw=1.0))
    assert ImageAreaFrac(0.02, 0.5).holds(scene, v)  # 200*200 / (800*600) = 0.083
