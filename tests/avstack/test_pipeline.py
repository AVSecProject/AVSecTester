"""The driving pipeline is a real avstack ModularDrivingPipeline — verify it builds and the
forward-collision planner brakes, without needing CARLA/GPU (avstack modules only)."""

import numpy as np
import pytest



@pytest.mark.parametrize(
    "objects,expected_speed",
    [([], 6.0), ([(10, 16, 0)], 0.0), ([(10, 4, 0)], 6.0), ([(16, 10, 0)], 6.0)],
    ids=["empty", "ahead", "behind", "adjacent-lane"],
)
def test_forward_collision_planner_brakes_in_body_frame(make_ego, objects, expected_speed):
    from avstack.modules.planning.types import WaypointPlan
    from avstack.modules.planning.vehicle import ForwardCollisionPlanner

    planner = ForwardCollisionPlanner(target_speed=6.0, brake_distance=12.0, brake_corridor=2.5)
    ego = make_ego(xyz=(10, 10, 0), yaw=np.pi / 2)
    plan = planner(WaypointPlan(), ego, objects=[make_ego(xyz=xyz) for xyz in objects])
    assert plan.top()[1].target_speed == expected_speed


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


def test_modular_stack_forwards_observation_and_converts_control(monkeypatch, pipeline_config):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from avsectester.plane import Control, Observation
    from avsectester.stacks.modular import ModularAVStack
    from avstack.config import PIPELINE

    pipeline = Mock(return_value=SimpleNamespace(
        throttle=np.float32(0.25), steer=np.float32(-0.5), brake=np.float32(0.0)))
    monkeypatch.setattr(PIPELINE, "build", Mock(return_value=pipeline))
    stack = ModularAVStack(pipeline_config)
    observation = Observation(t=1.0, frame=20, sensor_data={"lidar": object()},
                              vehicle_state=object())

    control = stack(observation)

    pipeline.assert_called_once_with(observation.sensor_data, observation.vehicle_state)
    assert control == Control(throttle=0.25, steer=-0.5, brake=0.0)
    assert all(type(value) is float for value in (control.throttle, control.steer, control.brake))


def test_modular_stack_counts_detections_after_attack(pipeline_config, make_detections, make_ego):
    from avsectester.plane import Control, Observation
    from avsectester.stacks.modular import ModularAVStack

    stack = ModularAVStack(pipeline_config)
    stack.attach("perception", {"type": "PhantomInjection"})
    stack.instrument(stages=("perception",))  # capture last, so it sees the attacked output
    counts = []
    for frame in range(3):
        detections = make_detections([(30, 8, 0)], frame=frame)
        observation = Observation(t=frame * 0.05, frame=frame, sensor_data=detections,
                                  vehicle_state=make_ego(t=frame * 0.05))
        assert isinstance(stack(observation), Control)
        counts.append(len(stack.component_log()["perception"]))

    assert counts == [2, 2, 2]  # original object plus injected phantom, every frame


def test_runtime_bridges_replace_real_stage_data_and_capture_final_outputs(
    pipeline_config, make_detections, make_ego
):
    from types import SimpleNamespace

    from avstack.datastructs import DataContainer
    from avsectester.plane import Observation
    from avsectester.runtime import CallInputs
    from avsectester.stacks.modular import ModularAVStack

    stack = ModularAVStack(pipeline_config)
    stack.instrument(("perception", "control"))
    stack.attach("perception", {"type": "PhantomInjection"})
    modules = {name: getattr(stack.pipeline, name) for name in ("perception", "tracking", "planning", "control")}
    original_hooks = {name: (tuple(module.pre_hooks), tuple(module.post_hooks)) for name, module in modules.items()}
    calls = []

    class Session:
        def emit(self, stage, value, sensor=None):
            calls.append(stage)
            if stage.endswith(".pre"):
                assert isinstance(value, CallInputs)
            if stage == "perception.pre":
                return CallInputs(args=(make_detections([(30, 8, 0)]),), kwargs=value.kwargs)
            if stage == "perception.post":
                assert len(value) == 2  # configured native phantom ran before the runtime transform
                return DataContainer(value.frame, value.timestamp, [], value.source_identifier)
            if stage == "tracking.pre":
                assert len(value.args[0]) == 0
                assert "platform" in value.kwargs
            if stage == "control.post":
                return SimpleNamespace(throttle=0.0, steer=0.2, brake=0.8)
            return value

    observation = Observation(t=0.0, frame=0, sensor_data=make_detections(), vehicle_state=make_ego())
    with stack.runtime_context(Session()):
        control = stack(observation)

    assert calls == [f"{stage}.{phase}" for stage in modules for phase in ("pre", "post")]
    assert control.throttle == 0.0 and control.brake == 0.8 and control.steer == 0.2
    assert len(stack.component_log()["perception"]) == 0
    assert stack.component_log()["control"].brake == 0.8
    for name, module in modules.items():
        assert (tuple(module.pre_hooks), tuple(module.post_hooks)) == original_hooks[name]
    assert stack.native["pipeline"] is stack.pipeline
    assert stack.models["perception"] is stack.pipeline.perception


def test_runtime_bridges_cleanup_after_invalid_pre_result(pipeline_config, make_detections, make_ego):
    from avsectester.plane import Observation
    from avsectester.stacks.modular import ModularAVStack

    stack = ModularAVStack(pipeline_config)

    class Session:
        def emit(self, stage, value, sensor=None):
            return object()

    observation = Observation(t=0.0, frame=0, sensor_data=make_detections(), vehicle_state=make_ego())
    with pytest.raises(TypeError, match="perception.pre must return CallInputs"):
        with stack.runtime_context(Session()):
            stack(observation)
    assert all(not module.pre_hooks and not module.post_hooks for module in stack.native.values() if hasattr(module, "pre_hooks"))


def test_native_hook_attached_during_reset_runs_before_runtime_defense_and_capture(
    pipeline_config, make_detections, make_ego
):
    from avstack.datastructs import DataContainer
    from avsectester.backend import WorldBackend, run
    from avsectester.plane import Observation
    from avsectester.runtime import Hook, Plugin, Runtime
    from avsectester.stacks.modular import ModularAVStack

    class Backend(WorldBackend):
        def observation(self, frame):
            return Observation(
                t=frame * 0.05, frame=frame,
                sensor_data=make_detections([(30, 8, 0)], frame=frame),
                vehicle_state=make_ego(t=frame * 0.05), ego_speed=5.0,
            )

        def reset(self):
            return self.observation(0)

        def step(self, control):
            return self.observation(1)

    class InstallNativeAttack(Plugin):
        def reset(self, context):
            context.stack.attach("perception", {"type": "PhantomInjection"})

        def transform(self, observation, context):
            return observation

    defense_inputs = []

    def hide_detections(detections, context):
        defense_inputs.append(len(detections))
        return DataContainer(
            detections.frame, detections.timestamp, [], detections.source_identifier,
            source_reference=detections.source_reference,
        )

    stack = ModularAVStack(pipeline_config)
    stack.instrument(("perception",))
    runtime = Runtime([
        Hook("observation", InstallNativeAttack()),
        Hook("perception.post", hide_detections),
    ])
    run(Backend(), stack, frames=1, runtime=runtime)

    assert defense_inputs == [2]  # real detection plus the newly installed phantom
    assert len(stack.component_log()["perception"]) == 0
    assert len(stack.pipeline.perception.post_hooks) == 2  # native attack and capture survive


def test_instrumentation_is_idempotent_and_reset_clears_telemetry(
    pipeline_config, make_detections, make_ego
):
    from avsectester.plane import Observation
    from avsectester.stacks.modular import ModularAVStack

    stack = ModularAVStack(pipeline_config)
    stack.instrument(("perception",))
    stack.instrument(("perception",))
    assert len(stack.pipeline.perception.post_hooks) == 1
    obs = Observation(t=0.0, frame=0, sensor_data=make_detections(), vehicle_state=make_ego())
    stack(obs)
    assert stack.component_log()["perception"] is not None
    stack.reset(obs)
    assert stack.component_log()["perception"] is None


def test_stage_capabilities_exclude_modules_without_actual_hook_dispatch(pipeline_config):
    from avsectester.stacks.modular import ModularAVStack

    pipeline_config["tracking"] = {"type": "GroundTruthTracker"}
    stack = ModularAVStack(pipeline_config)
    # GroundTruthTracker inherits hook lists but bypasses the decorated call.
    assert "tracking.pre" not in stack.supported_stages
    assert "tracking.post" not in stack.supported_stages
    assert "perception.pre" in stack.supported_stages
    with pytest.raises(ValueError, match="Cannot instrument unavailable stages"):
        stack.instrument(("tracking",))
    with pytest.raises(ValueError, match="No hookable"):
        stack.attach("tracking", {"type": "PhantomInjection"})
