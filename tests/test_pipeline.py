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


@pytest.mark.parametrize("yaw", [0.0, np.pi / 2])
def test_phantom_propagates_to_track_plan_and_braking(
    make_pipeline,
    make_detections,
    make_ego,
    yaw,
):
    from avsectester.attacks import PhantomInjection
    from avstack.geometry import ReferenceFrame

    clean, attacked = make_pipeline(), make_pipeline()
    attacked.perception.register_post_hook(PhantomInjection())
    clean_commands, attacked_commands, attack_plans, confirmed = [], [], [], []
    origin = np.array([20.0, 10.0, 0.0])
    for frame in range(20):
        t = frame * 0.05
        xyz = origin + 5.0 * t * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        ego = make_ego(xyz=xyz, yaw=yaw, speed=5.0, t=t)
        ref = ReferenceFrame(ego.position.x, ego.attitude.q, ego.reference)
        # A genuine object outside the driving corridor establishes the sensor frame.
        # Each branch receives fresh data so attack mutation cannot contaminate the baseline.
        clean_commands.append(clean(make_detections([(30, 8, 0)], reference=ref, frame=frame), ego))
        attacked_commands.append(
            attacked(make_detections([(30, 8, 0)], reference=ref, frame=frame), ego)
        )
        attack_plans.append(attacked.plan.top()[1].target_speed)
        confirmed.append(len(attacked.tracking.tracks_confirmed))

    assert all(command.throttle > 0 and command.brake == 0 for command in clean_commands)
    assert clean.plan.top()[1].target_speed == 6.0
    assert confirmed[0] == 0  # a single observation should not immediately confirm the phantom
    assert confirmed[-1] > len(clean.tracking.tracks_confirmed)
    assert attack_plans[0] == 6.0
    assert attack_plans[-1] == 0.0
    assert attacked_commands[0].brake == 0
    assert attacked_commands[-1].throttle == 0
    assert attacked_commands[-1].brake > 0


def test_tracking_hook_replacement_reaches_planner(make_pipeline, make_detections, make_ego):
    from avstack.datastructs import DataContainer

    clean, attacked = make_pipeline(), make_pipeline()
    observed_track_counts = []

    def hide_tracks(tracks):
        observed_track_counts.append(len(tracks))
        # Replace the output without changing the tracker's internal tracks.
        return (DataContainer(tracks.frame, tracks.timestamp, [], tracks.source_identifier),)

    attacked.tracking.register_post_hook(hide_tracks)
    for frame in range(20):
        ego = make_ego(speed=5.0, t=frame * 0.05)
        clean_command = clean(make_detections([(6, 0, 0)], frame=frame), ego)
        attacked_command = attacked(make_detections([(6, 0, 0)], frame=frame), ego)

    assert observed_track_counts[-1] > 0
    assert len(attacked.tracking.tracks_confirmed) == len(clean.tracking.tracks_confirmed) > 0
    assert clean.plan.top()[1].target_speed == 0.0
    assert clean_command.throttle == 0.0
    assert clean_command.brake > 0.0
    assert attacked.plan.top()[1].target_speed == 6.0
    assert attacked_command.throttle > 0.0
    assert attacked_command.brake == 0.0


def test_planning_hook_replacement_reaches_control_and_next_frame(
    make_pipeline, make_detections, make_ego
):
    from avstack.modules.planning.types import Waypoint, WaypointPlan

    pipe = make_pipeline()
    planner_inputs, replacement_plans = [], []

    def record_plan_input(*args, **kwargs):
        planner_inputs.append(args[0])
        return args, kwargs

    def replace_with_stop(plan):
        distance, waypoint = plan.top()
        assert waypoint.target_speed == 6.0
        # A fresh plan and waypoint ensure the original cruise plan is not mutated.
        replacement = WaypointPlan()
        replacement.push(distance, Waypoint(waypoint.target_point, target_speed=0.0))
        replacement_plans.append(replacement)
        return (replacement,)

    pipe.planning.register_pre_hook(record_plan_input)
    pipe.planning.register_post_hook(replace_with_stop)
    for frame in range(3):
        previous_plan = pipe.plan
        command = pipe(make_detections(frame=frame), make_ego(speed=5.0, t=frame * 0.05))

        assert planner_inputs[-1] is previous_plan
        assert pipe.plan is replacement_plans[-1]
        assert pipe.plan is not previous_plan
        assert previous_plan.top()[1].target_speed == 6.0
        assert pipe.plan.top()[1].target_speed == 0.0
        assert command.throttle == 0.0
        assert command.brake > 0.0
