"""Portable attacks complete public preparation, execution and offline evaluation."""
from dataclasses import replace
import numpy as np
import pytest

from avsectester.attack.implemented_attacks.patch_attack.attack import PatchAttack
from avsectester.attack.placement import AttachedPlacement
from avsectester.closed_loop.datatypes import WorldSnapshot
from avsectester.closed_loop.services import DrivingAdapter
from avsectester.closed_loop.simulator import SensorView
from avsectester.datasets import ActorState, FrameReference
from avsectester.evaluation import evaluate
from avsectester.experiment import Experiment, ExperimentConfiguration, TreatmentConfiguration
from avsectester.geometry.poses import ActorPose

def test_attack_prepares_executes_and_evaluates(memory_experiment):
    Data = memory_experiment.Data
    Driver = memory_experiment.Driver
    Sim = memory_experiment.Sim
    attack = PatchAttack(decision_steps=2, max_windows=1)
    config = ExperimentConfiguration(Data(), Sim(), DrivingAdapter(lambda seed: Driver()),
        attacks={"test": attack}, treatments=(TreatmentConfiguration("clean"),
        TreatmentConfiguration("attack", "test")), strict=True)
    prepared = Experiment.prepare(config)
    assert len(prepared.metadata["test"]) == 1
    result = Experiment.execute(prepared)
    assert not result.failures
    assert len(result.runs) == 2
    assert all(len(run.log.steps) == 2 for run in result.runs)
    report = evaluate(result, config)
    assert len(report.groups) == 1
    clean, attacked = [run.log for run in result.runs]
    assert any(e.get("outcome") == "applied" for e in attacked.steps[0].effects)

def test_patch_deploys_exported_texture_at_box_center_and_cleans_up(monkeypatch, memory_experiment):
    Data = memory_experiment.Data
    host_state = memory_experiment.host_state
    from contextlib import ExitStack
    from types import SimpleNamespace
    from avsectester.attack import AttackSegment
    from avsectester.attack.implemented_attacks.patch_attack.online import CarlaPatchPanel

    origin = np.eye(4)
    origin[0, 3] = 10.
    state = host_state()
    state = replace(state, box_center=(0., 0., 0.74))
    actor = SimpleNamespace(get_transform=lambda: SimpleNamespace(get_matrix=lambda: origin))
    destroyed, ticks, panels = [], [], []
    spawned = SimpleNamespace(is_alive=True, destroy=lambda: destroyed.append(True))

    def deploy(panel, world, target, *, tick, on_spawn):
        panels.append(panel)
        assert target is actor
        on_spawn(spawned)
        tick()
        return [spawned]

    monkeypatch.setattr(CarlaPatchPanel, "apply", deploy)
    simulator_runtime = SimpleNamespace(native={"actors": {"lead": actor}, "world": object()},
        state=lambda: WorldSnapshot(0., 0, actors={"lead": state},
            ego=ActorState(np.eye(4), extent=(4., 2., 2.))), setup_tick=lambda: ticks.append(True))
    segment = AttackSegment("s", Data.timeline[0], Data.timeline[2],
                            placement=AttachedPlacement("lead", "rear_center", (-.01, 0., 0.)),
                            user_data={"deployment": "world"})
    attack = PatchAttack(decision_steps=2)
    artifact = np.full((9, 11, 4), 127, np.uint8)
    runtime = attack.create_runtime(segment, artifact)
    assert runtime.setup_ticks == 1 and runtime.step(None) == ()
    with ExitStack() as resources:
        runtime.setup(None)[0].apply(simulator_runtime, resources)
        assert ticks == [True] and not destroyed
        np.testing.assert_array_equal(panels[0].texture, artifact)
        np.testing.assert_allclose(panels[0].location, [-2.01, 0., 0.74])
        np.testing.assert_allclose(panels[0].rotation, [90., 0., 0.], atol=1e-8)
    assert destroyed == [True]

def test_patch_selection_uses_explicit_host_and_distance_without_replacing_them(memory_experiment):
    Data = memory_experiment.Data
    Driver = memory_experiment.Driver
    Sim = memory_experiment.Sim
    for attack in (PatchAttack(decision_steps=2, candidate_hosts=["absent"]),
                   PatchAttack(decision_steps=2, max_distance=5.)):
        config = ExperimentConfiguration(Data(), Sim(), DrivingAdapter(lambda seed: Driver()),
                                         attacks={"patch": attack}, strict=True)
        prepared = Experiment.prepare(config)
        assert not prepared.metadata["patch"]

def test_physical_selection_does_not_skip_frames_for_deployment_ticks(memory_experiment):
    CAMERA = memory_experiment.CAMERA
    CAM_FROM_WORLD = memory_experiment.CAM_FROM_WORLD
    host_state = memory_experiment.host_state
    from contextlib import nullcontext
    from types import SimpleNamespace
    from avsectester.closed_loop.simulators.carla import CarlaSimulator
    frames = tuple(FrameReference("s", i, i, i * .1) for i in range(6))
    rendered = []
    view = SensorView("front", np.zeros((80, 100, 3), np.uint8), CAMERA, CAM_FROM_WORLD,
                      {"lead": host_state().box_pose()}, ActorPose(np.eye(4)), 0., 0)

    def render(request):
        assert request.effects == ()
        rendered.append(request.frame.position)
        return SimpleNamespace(view=view)

    scenario = SimpleNamespace(services=SimpleNamespace(simulator=object.__new__(CarlaSimulator),
        session=lambda seed: nullcontext(SimpleNamespace(render=render))),
        scenario_id="s", seed=0, frames=frames, record_reason=lambda reason: None)
    selected = PatchAttack(decision_steps=2).build_metadata(scenario)
    assert rendered == [0, 1, 2, 3, 4]
    assert [(segment.start.position, segment.stop.position) for segment in selected] == [(0, 2), (2, 4)]
    assert all(segment.user_data["deployment"] == "world" for segment in selected)

def test_selection_keeps_earlier_segments_when_target_later_crosses_camera_plane(memory_experiment):
    Data = memory_experiment.Data
    Driver = memory_experiment.Driver
    Sim = memory_experiment.Sim
    World = memory_experiment.World
    attack = PatchAttack(decision_steps=2)
    class LongData(Data):
        timeline = tuple(FrameReference("s", i, i, i * .1) for i in range(6))

    class CrossingWorld(World):
        def sensor_view(self, sensor):
            view = super().sensor_view(sensor)
            if self.i < 3:
                return view
            r = 1 / np.sqrt(2)
            camera_x = 8. if self.i == 3 else 20.
            camera = np.array([[r, -r, 0., -camera_x*r], [0., 0., -1., 0.],
                               [r, r, 0., -camera_x*r], [0., 0., 0., 1.]])
            return replace(view, cam_from_world=camera)

    class CrossingSim(Sim):
        def create_runtime(self, seed=None):
            return CrossingWorld(self)

    config = ExperimentConfiguration(LongData(), CrossingSim(), DrivingAdapter(lambda seed: Driver()),
                                     attacks={"attack": attack}, strict=True)
    prepared = Experiment.prepare(config)
    assert [(s.start.position, s.stop.position) for s in prepared.metadata["attack"]] == [(0, 2)]
    assert prepared.selections[0].status == "selected"
    # Assert the fixture exercises unknown camera-plane visibility, not merely a zero score.
    runtime = CrossingSim().create_runtime()
    runtime.i = 3
    from avsectester.attack.insertion.effect import InsertionAttackEffect
    evidence = InsertionAttackEffect("front", attack.insertion(attack.assets, host="lead")).evidence(runtime.sensor_view("front"))
    assert evidence.visibility is None and "camera plane" in evidence.reason

def test_selection_does_not_hide_invalid_sensor_configuration(memory_experiment):
    Data = memory_experiment.Data
    Driver = memory_experiment.Driver
    Sim = memory_experiment.Sim
    attack = PatchAttack(decision_steps=2, sensor="unknown-camera")
    config = ExperimentConfiguration(Data(), Sim(), DrivingAdapter(lambda seed: Driver()),
                                     attacks={"attack": attack}, strict=True)
    with pytest.raises(KeyError, match="unknown-camera"):
        Experiment.prepare(config)
