"""Exercise the real runner while replacing only its simulator- and stack-facing objects.

After the interface split, ``run_scenario`` assembles a :class:`~avsectester.scenario.CarlaBackend`
(client + ego + traffic, senses/actuates) and a :class:`~avsectester.scenario.ModularAVStack` (the AV
box), and drives them with :func:`avsectester.backend.run`. These tests mock ``CARLA.build`` (the
simulator objects) and ``PIPELINE.build`` (the AV pipeline) so the runner's own logic — frame
recording, hook order + attacked detection counts, replay-spawn capture, NPC handling, cleanup on
failure, and the sensor-delivery wait — is exercised without a live simulator.
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from avsectester import scenario
from avstack.modules.base import BaseModule


@pytest.fixture
def simulation(monkeypatch):
    clock = SimpleNamespace(frame=100, t=10.0)

    def snapshot():
        return SimpleNamespace(
            frame=clock.frame, timestamp=SimpleNamespace(elapsed_seconds=clock.t)
        )

    def tick():
        clock.frame += 1
        clock.t += 0.05

    client = SimpleNamespace(
        world=SimpleNamespace(get_snapshot=Mock(side_effect=snapshot)),
        tick=Mock(side_effect=tick),
        close=Mock(),
    )

    # --- the AV stack pipeline (what PIPELINE.build returns): a callable that fires perception's
    #     post-hooks on a real detection and returns a fixed control ---
    perception = BaseModule(name="test-perception")
    tracking = BaseModule(name="test-tracking")
    control = SimpleNamespace(throttle=0.0, brake=0.7, steer=-0.2)
    stage_outputs = []

    def pipeline_call(sensor_data, vehicle_state):
        stage_outputs.append(perception._apply_post_hooks(["real-detection"]))
        return control

    pipeline = Mock(side_effect=pipeline_call)
    pipeline.perception = perception
    pipeline.tracking = tracking
    monkeypatch.setattr(scenario.PIPELINE, "build", Mock(return_value=pipeline))

    # --- the simulator objects (what CARLA.build returns): ego senses + actuates, no driving ---
    def spawned_actor(index):
        transform = SimpleNamespace(
            location=SimpleNamespace(x=10.0 + index, y=-2.0, z=0.5),
            rotation=SimpleNamespace(pitch=1.0, yaw=37.0, roll=-3.0),
        )
        return SimpleNamespace(type_id="vehicle.test", get_transform=Mock(return_value=transform))

    ego = SimpleNamespace(
        actor=spawned_actor(0),
        initialize=Mock(),
        destroy=Mock(),
        apply_control=Mock(),
        get_pose=Mock(
            return_value=SimpleNamespace(
                position=SimpleNamespace(x=0.0), attitude=SimpleNamespace(q=1.0)
            )
        ),
        reference=SimpleNamespace(x=None, q=None),
        sensor_data_manager=SimpleNamespace(
            empty=Mock(return_value=False), pop=Mock(return_value={"lidar-0-0": "cloud"})
        ),
        get_object_state=Mock(
            return_value=SimpleNamespace(
                velocity=SimpleNamespace(norm=lambda: (clock.frame - 100) * 0.5)
            )
        ),
    )
    npcs = [
        SimpleNamespace(actor=spawned_actor(i + 1), initialize=Mock(), destroy=Mock())
        for i in range(2)
    ]
    built_npcs = []

    def build(config, default_args=None):
        if config["type"] == "CarlaClient":
            return client
        assert default_args == {"client": client}
        if config["type"] == "CarlaMobileActor":
            return ego
        assert config["type"] == "CarlaNpc"
        npc = npcs[len(built_npcs)]
        built_npcs.append(npc)
        return npc

    for actor in [ego, *npcs]:
        actor.spawn_transform = actor.actor.get_transform.return_value
        actor.actor.get_transform.side_effect = AssertionError(
            "Actor state is stale before ticking"
        )

    registry = Mock(side_effect=build)
    monkeypatch.setattr(scenario.CARLA, "build", registry)
    sleep = Mock()
    monkeypatch.setattr(scenario.time, "sleep", sleep)
    config = {
        "client": {"type": "CarlaClient"},
        "ego": {"type": "CarlaMobileActor", "pipeline": {"type": "ModularDrivingPipeline"}},
        "npcs": {"count": 2, "npc_type": "vehicle", "spawn_start": 3},
    }
    return SimpleNamespace(
        config=config,
        client=client,
        ego=ego,
        npcs=npcs,
        pipeline=pipeline,
        registry=registry,
        sleep=sleep,
        stage_outputs=stage_outputs,
    )


def test_runner_records_requested_frames_and_initializes_actors(simulation):
    sim = simulation
    trace = scenario.run_scenario(sim.config, frames=3)
    # reset ticks once (ego stationary) then each step ticks + records the post-step state.
    assert [record.frame for record in trace.records] == [0, 1, 2]
    assert [record.t for record in trace.records] == pytest.approx([10.10, 10.15, 10.20])
    assert [record.speed for record in trace.records] == [1.0, 1.5, 2.0]
    assert [record.n_detections for record in trace.records] == [1, 1, 1]
    assert all((r.throttle, r.brake, r.steer) == (0.0, 0.7, -0.2) for r in trace.records)
    assert sim.client.tick.call_count == 4  # 1 in reset + 3 steps
    assert sim.ego.apply_control.call_count == 3
    sim.ego.initialize.assert_called_once_with(10.0, 100)
    sim.ego.destroy.assert_called_once_with()
    for npc in sim.npcs:
        npc.initialize.assert_called_once_with(10.0, 100)
        npc.destroy.assert_called_once_with()
    sim.client.close.assert_called_once()
    sim.sleep.assert_not_called()


def test_runner_preserves_hook_order_and_counts_attacked_output(simulation, monkeypatch):
    sim = simulation

    def append_hook(name):
        def hook(data):
            data.append(name)
            return (data,)

        return hook

    hooks = [append_hook("phantom-a"), append_hook("phantom-b"), append_hook("tracking-only")]
    build_hook = Mock(side_effect=hooks)
    monkeypatch.setattr(scenario.HOOKS, "build", build_hook)
    attacks = [
        {"stage": "perception", "hook": {"type": "A"}},
        {"stage": "perception", "hook": {"type": "B"}},
        {"stage": "tracking", "hook": {"type": "C"}},
    ]
    trace = scenario.run_scenario(sim.config, attacks=attacks, frames=2)
    assert sim.stage_outputs == [["real-detection", "phantom-a", "phantom-b"]] * 2
    assert [r.n_detections for r in trace.records] == [3, 3]
    assert sim.pipeline.tracking.post_hooks == [hooks[2]]
    assert [call.args[0] for call in build_hook.call_args_list] == [a["hook"] for a in attacks]


def test_runner_records_actual_spawns_before_driving_without_mutating_input(simulation):
    sim = simulation
    before = deepcopy(sim.config)

    def advance_world():
        # Moving actors after setup must not alter the recorded spawn config.
        for actor in [sim.ego, *sim.npcs]:
            actor.spawn_transform.location.x = 999

    sim.client.tick.side_effect = advance_world
    trace = scenario.run_scenario(sim.config, frames=2)
    replay = trace.replay_scenario
    for spec, x in zip([replay["ego"], *replay["npcs"]], [10.0, 11.0, 12.0]):
        assert spec["spawn_transform"] == {
            "location": {"x": x, "y": -2.0, "z": 0.5},
            "rotation": {"pitch": 1.0, "yaw": 37.0, "roll": -3.0},
        }
    assert replay["ego"]["vehicle"] == "vehicle.test"
    assert all(npc["npc_type"] == "vehicle.test" for npc in replay["npcs"])
    assert replay["client"]["strict_spawn"] is True
    assert sim.registry.call_args_list[0].args[0]["strict_spawn"] is False
    assert sim.config == before
    sim.ego.actor.get_transform.assert_not_called()


def test_compact_npc_config_is_expanded(simulation):
    sim = simulation
    scenario.run_scenario(sim.config, frames=1)
    configs = [call.args[0] for call in sim.registry.call_args_list]
    assert configs[2:] == [
        {"type": "CarlaNpc", "spawn": 3, "npc_type": "vehicle"},
        {"type": "CarlaNpc", "spawn": 4, "npc_type": "vehicle"},
    ]


@pytest.mark.parametrize(
    "npcs",
    [
        None,
        [],
        {"count": 0},
        [{"type": "CarlaNpc", "spawn": 9, "npc_type": "vehicle"}],
    ],
)
def test_optional_and_explicit_npc_config(simulation, npcs):
    sim = simulation
    sim.config["npcs"] = npcs
    before = deepcopy(sim.config)
    scenario.run_scenario(sim.config, frames=1)
    assert sim.config == before
    assert sim.registry.call_count == (3 if isinstance(npcs, list) and npcs else 2)
    if isinstance(npcs, list) and npcs:
        assert sim.registry.call_args.args[0] == npcs[0]


@pytest.mark.parametrize("failure_site", ["world", "control"])
def test_tick_failure_propagates_and_destroys_all_actors(simulation, failure_site):
    sim = simulation
    target = sim.client.tick if failure_site == "world" else sim.ego.apply_control
    target.side_effect = RuntimeError("tick failed")
    with pytest.raises(RuntimeError, match="tick failed"):
        scenario.run_scenario(sim.config, frames=3)
    sim.ego.destroy.assert_called_once_with()
    for npc in sim.npcs:
        npc.destroy.assert_called_once_with()


def test_npc_destroy_failure_does_not_prevent_remaining_cleanup(simulation):
    sim = simulation
    sim.npcs[0].destroy.side_effect = RuntimeError("already destroyed")
    scenario.run_scenario(sim.config, frames=1)
    sim.npcs[1].destroy.assert_called_once_with()


@pytest.mark.parametrize("failure_site", ["initialize", "npc_build"])
def test_setup_failure_cleans_up_already_created_actors(simulation, failure_site):
    sim = simulation
    if failure_site == "initialize":
        sim.ego.initialize.side_effect = RuntimeError("setup failed")
    else:
        sim.registry.side_effect = [sim.client, sim.ego, sim.npcs[0], RuntimeError("setup failed")]
    with pytest.raises(RuntimeError, match="setup failed"):
        scenario.run_scenario(sim.config, frames=1)
    sim.ego.destroy.assert_called_once_with()
    sim.npcs[0].destroy.assert_called_once_with()


def test_ego_destroy_failure_still_cleans_up_npcs(simulation):
    sim = simulation
    sim.ego.destroy.side_effect = RuntimeError("ego destroy failed")
    with pytest.raises(RuntimeError, match="ego destroy failed"):
        scenario.run_scenario(sim.config, frames=1)
    for npc in sim.npcs:
        npc.destroy.assert_called_once_with()


def test_runner_waits_for_delayed_sensor_data(simulation):
    sim = simulation
    # reset's observe finds data immediately; the single step's observe waits two polls for it.
    sim.ego.sensor_data_manager.empty.side_effect = [False, True, True, False]
    trace = scenario.run_scenario(sim.config, frames=1, settle_iters=4)
    assert len(trace.records) == 1
    assert sim.sleep.call_count == 2
    sim.ego.apply_control.assert_called_once()


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="Exhausting the sensor wait currently proceeds to observe instead of reporting timeout",
)
def test_missing_sensor_data_reports_timeout_and_cleans_up(simulation):
    sim = simulation
    sim.ego.sensor_data_manager.empty.return_value = True
    with pytest.raises(TimeoutError):
        scenario.run_scenario(sim.config, frames=1, settle_iters=3)


@pytest.mark.parametrize("gpu", [None, 0, 2])
def test_gpu_override_changes_only_perception_device(gpu):
    config = {"ego": {"pipeline": {"perception": {"gpu": 1, "model": "pointpillars"}}}}
    expected = deepcopy(config)
    if gpu is not None:
        expected["ego"]["pipeline"]["perception"]["gpu"] = gpu
    assert scenario.set_perception_gpu(config, gpu) is config
    assert config == expected
