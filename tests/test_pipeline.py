"""The driving pipeline is a real avstack ModularDrivingPipeline — verify it builds and the
forward-collision planner brakes, without needing CARLA/GPU (avstack modules only)."""

import numpy as np
import pytest

pytest.importorskip("avstack")


def _ego(pos_xyz, yaw_deg):
    from avstack.environment.objects import VehicleState
    from avstack.geometry import Attitude, Box3D, GlobalOrigin3D, Position
    from avstack.geometry import transformations as tforms

    q = tforms.transform_orientation([0, 0, np.radians(yaw_deg)], "euler", "quat")
    vs = VehicleState(obj_type="car", ID=0)
    p = Position(np.array(pos_xyz, dtype=float), GlobalOrigin3D)
    a = Attitude(q, GlobalOrigin3D)
    vs.set(0.0, p, Box3D(p, a, [1.5, 1.8, 4.0]), attitude=a)
    return vs


class _Obj:
    def __init__(self, xyz):
        from avstack.geometry import GlobalOrigin3D, Position

        self.position = Position(np.array(xyz, dtype=float), GlobalOrigin3D)


def test_modular_driving_pipeline_builds_from_config():
    import avstack.modules.control.vehicle
    import avstack.modules.perception.object3d
    import avstack.modules.pipeline
    import avstack.modules.planning.vehicle
    import avstack.modules.tracking.tracker3d  # noqa: F401
    from avstack.config import PIPELINE

    pipe = PIPELINE.build(
        dict(
            type="ModularDrivingPipeline",
            perception=dict(type="Passthrough3DObjectDetector"),
            tracking=dict(type="BasicBoxTracker3D"),
            planning=dict(type="ForwardCollisionPlanner", target_speed=6.0),
            control=dict(
                type="VehiclePIDController",
                args_lateral=dict(K_P=1.0, K_D=0.0, K_I=0.0),
                args_longitudinal=dict(K_P=0.5, K_D=0.0, K_I=0.05),
            ),
        )
    )
    assert type(pipe).__name__ == "ModularDrivingPipeline"


def test_forward_collision_planner_brakes_in_body_frame():
    import avstack.modules.planning.vehicle  # noqa: F401
    from avstack.modules.planning.types import WaypointPlan
    from avstack.modules.planning.vehicle import ForwardCollisionPlanner

    pl = ForwardCollisionPlanner(target_speed=6.0, brake_distance=12.0, brake_corridor=2.5)

    def speed(ego, objs):
        p = WaypointPlan()
        pl(p, ego, objects=objs)
        return p.top()[1].target_speed

    e = _ego([10, 10, 0], 90)  # facing +y in global
    assert speed(e, []) == 6.0  # nothing ahead -> cruise
    assert speed(e, [_Obj([10, 16, 0])]) == 0.0  # 6 m ahead in body frame -> brake
    assert speed(e, [_Obj([10, 4, 0])]) == 6.0  # behind -> cruise
    assert speed(e, [_Obj([16, 10, 0])]) == 6.0  # 6 m to the side -> cruise
