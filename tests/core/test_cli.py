"""CLI configuration, treatment routing and observable experiment outcomes."""

from copy import deepcopy
from dataclasses import replace
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from typer.testing import CliRunner

from avsectester import carla_runner
from avsectester import cli
from avsectester.backend import AVStack
from avsectester.experiment import ExecutionMode, TreatmentMode
from avsectester.plane import Control, StateEstimate
from avsectester.preparation import Intervention
from avsectester.runtime import Hook, Plugin
from avsectester.simulators.nurec import NuRecBackend, StubRenderer


@pytest.fixture
def attack_trace(make_trace):
    def build(*args):
        trace = make_trace(*args)
        trace.provenance["interventions"] = [{"kind": "attack"}]
        return trace
    return build


@pytest.fixture
def cli_config(tmp_path):
    config = {
        "frames": 6,
        "client": {"type": "CarlaClient", "seed": 0},
        "ego": {
            "type": "CarlaMobileActor",
            "pipeline": {"perception": {"type": "MMDetObjectDetector3D", "gpu": 1}},
        },
        "attacks": [{"stage": "perception", "hook": {"type": "PhantomInjection"}}],
    }
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config


@pytest.fixture
def user_methods(monkeypatch):
    events, inputs, localizers = [], [], []

    class Localizer:
        def __init__(self):
            self.history = []
            localizers.append(self)

        def __call__(self, observation, context):
            self.history.append(observation.frame)
            return StateEstimate(observation.vehicle_state, observation.ego_speed)

    class SpeedPlugin(Plugin):
        def __init__(self, speed):
            self.speed = speed

        def reset(self, context):
            events.append(("reset", self.speed))

        def transform(self, observation, context):
            assert context.ground_truth.ego_speed == 2
            events.append(("input", observation.ego_speed))
            return replace(observation, ego_speed=self.speed)

        def close(self):
            events.append(("close", self.speed))

    class Stack(AVStack):
        def __init__(self):
            self.seen = []
            inputs.append(self.seen)

        def __call__(self, observation):
            self.seen.append(observation.ego_speed)
            return Control()

    class Defense(Intervention):
        def hooks(self, artifacts):
            def defend(command, context):
                context.record_output("defense", "applied")
                events.append(("defense", context.step))
                return Control(brake=1)

            return [Hook("command", defend)]

    class SpeedFilter(Plugin):
        def __init__(self, max_speed=1.0):
            self.max_speed = max_speed

        def transform(self, observation, context):
            return replace(observation, ego_speed=min(observation.ego_speed, self.max_speed))

    class FilterMethod(Intervention):
        def __init__(self, max_speed=1.0):
            self.max_speed = max_speed

        def hooks(self, artifacts):
            return [Hook("observation", SpeedFilter(self.max_speed))]

    module = ModuleType("avsec_test_cli_methods")
    module.SpeedPlugin, module.Stack, module.Localizer, module.Defense = (
        SpeedPlugin, Stack, Localizer, Defense,
    )
    module.SpeedFilter, module.FilterMethod = SpeedFilter, FilterMethod
    module.NoopAttack = lambda: lambda value, context: value
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return SimpleNamespace(
        path=module.__name__ + ":", events=events, inputs=inputs, localizers=localizers,
        config={
            "backend": {
                "type": "NuRecBackend", "renderer": {"type": "stub", "height": 2, "width": 2},
                "ego0": {"speed": 2},
            },
            "stack": {"factory": module.__name__ + ":Stack"},
        },
    )


@pytest.mark.parametrize("args,frames,gpu", [
    ([], 6, 1), (["--frames", "10", "--gpu", "0"], 10, 0), (["--frames", "0"], 0, 1),
])
def test_cli_loads_config_and_uses_the_shared_experiment_entry(
    monkeypatch, cli_config, make_trace, attack_trace, args, frames, gpu,
):
    path, config = cli_config
    clean, attacked = make_trace([0, 2, 5]), attack_trace([0, 1, 0], [0, 1, 1])
    execute = Mock(return_value={"clean": clean, "attack": attacked})
    monkeypatch.setattr(cli, "run_configured", execute)
    result = CliRunner().invoke(cli.app, ["run", str(path), *args])

    assert result.exit_code == 0, result.output
    expected = deepcopy(config)
    expected["ego"]["pipeline"]["perception"]["gpu"] = gpu
    execute.assert_called_once()
    assert execute.call_args.args == (expected, frames)
    assert execute.call_args.kwargs["mode"] is ExecutionMode.PAIR
    assert execute.call_args.kwargs["treatment"] is TreatmentMode.ATTACK
    assert execute.call_args.kwargs["provenance"] == {"config_path": str(path.resolve())}
    assert result.output.index("[clean]") < result.output.index("[attacked]")
    assert result.output.index("[scenario] seed=0") < result.output.index("[pair]")
    assert "peak_speed=5.00" in result.output and "ATTACK SUCCEEDED" in result.output
    assert yaml.safe_load(path.read_text()) == config


@pytest.mark.parametrize("clean_speeds,attacked_speeds,brakes,code,verdict", [
    ([0, 2, 5], [0, 1, 0], [0, 1, 1], 0, "ATTACK SUCCEEDED"),
    ([0, 2, 5], [0, 2, 5], [0, 0, 0], 3, "no impact"),
    ([0, 0.1, 0.1], [0, 0, 0], [0, 1, 1], 4, "INCONCLUSIVE"),
])
def test_cli_exit_code_matches_verdict(
    monkeypatch, cli_config, make_trace, attack_trace, clean_speeds, attacked_speeds, brakes, code, verdict,
):
    path, _ = cli_config
    monkeypatch.setattr(cli, "run_configured", Mock(return_value={
        "clean": make_trace(clean_speeds), "attack": attack_trace(attacked_speeds, brakes),
    }))
    result = CliRunner().invoke(cli.app, ["run", str(path)])
    assert result.exit_code == code, result.output
    assert verdict in result.output


@pytest.mark.parametrize("treatment", ["clean", "attack", "defense", "attack-defense"])
def test_cli_forwards_single_treatment_without_pair_metrics(
    monkeypatch, cli_config, make_trace, treatment,
):
    path, config = cli_config
    config.update(mode="pair", treatment="clean")
    path.write_text(yaml.safe_dump(config))
    execute = Mock(return_value={treatment: make_trace([0])})
    monkeypatch.setattr(cli, "run_configured", execute)
    result = CliRunner().invoke(cli.app, [
        "run", str(path), "--mode", "single", "--treatment", treatment,
    ])
    assert result.exit_code == 0, result.output
    assert execute.call_args.kwargs["mode"] is ExecutionMode.SINGLE
    assert execute.call_args.kwargs["treatment"] is TreatmentMode(treatment)
    assert "INCONCLUSIVE" not in result.output and "no impact" not in result.output


def test_cli_optional_plot_creates_file(monkeypatch, cli_config, make_trace, attack_trace, tmp_path):
    path, _ = cli_config
    monkeypatch.setattr(cli, "run_configured", Mock(return_value={
        "clean": make_trace([0, 2, 5]), "attack": attack_trace([0, 1, 0], [0, 1, 1]),
    }))
    output = tmp_path / "plots" / "impact.png"
    result = CliRunner().invoke(cli.app, ["run", str(path), "--plot", str(output)])
    assert result.exit_code == 0, result.output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert str(output) in result.output


@pytest.mark.parametrize("failure", [RuntimeError("server unavailable"), ValueError("handler failed")])
@pytest.mark.parametrize("seed", [0, None])
def test_cli_runtime_errors_preserve_seed_and_have_execution_exit_code(
    monkeypatch, cli_config, failure, seed,
):
    import typer
    from avsectester.simulators import carla

    path, config = cli_config
    config["client"]["seed"] = seed
    path.write_text(yaml.safe_dump(config))
    generated = Mock(return_value=987)
    monkeypatch.setattr(carla.secrets, "randbelow", generated)
    echo = Mock(wraps=typer.echo)
    monkeypatch.setattr(typer, "echo", echo)

    def fail(configured_scene, frames, **kwargs):
        expected_seed = 987 if seed is None else seed
        assert configured_scene["client"]["seed"] == expected_seed
        echo.assert_any_call(f"[scenario] seed={expected_seed}")
        raise failure

    monkeypatch.setattr(cli, "run_configured", fail)
    result = CliRunner().invoke(cli.app, ["run", str(path)])
    assert result.exit_code == 1
    assert "Execution error" in result.output and str(failure) in result.output
    assert generated.call_count == int(seed is None)
    assert yaml.safe_load(path.read_text()) == config


@pytest.mark.parametrize("invalid", ["-1", "malformed-yaml", "single-plot"])
def test_cli_rejects_invalid_options_before_experiment_assembly(monkeypatch, cli_config, invalid):
    path, _ = cli_config
    args = []
    if invalid == "-1":
        args = ["--frames", "-1"]
    elif invalid == "malformed-yaml":
        path.write_text("runtime: [")
    else:
        args = ["--mode", "single", "--plot", "unused.png"]
    execute = Mock()
    monkeypatch.setattr(cli, "run_configured", execute)
    result = CliRunner().invoke(cli.app, ["run", str(path), *args])
    assert result.exit_code == 2
    execute.assert_not_called()


def test_cli_rejects_unknown_runtime_options_before_constructing_backend(monkeypatch, cli_config):
    path, config = cli_config
    config["runtime"] = {"unused_option": True}
    path.write_text(yaml.safe_dump(config))
    construct = Mock(side_effect=AssertionError("Backend must not be created"))
    monkeypatch.setattr(carla_runner, "CarlaBackend", construct)
    result = CliRunner().invoke(cli.app, ["run", str(path)])
    assert result.exit_code == 2, result.output
    assert "unused_option" in result.output
    construct.assert_not_called()


def test_cli_consumes_physical_patches_through_world_setup(monkeypatch, cli_config):
    from avsectester.attacks.patch.physical_patch import PhysicalPatch

    path, config = cli_config
    config.pop("attacks")
    config["patches"] = [{"target": "lead", "texture": {"pattern": "checkerboard"}}]
    path.write_text(yaml.safe_dump(config))
    world, lead, panel = object(), object(), SimpleNamespace(destroy=Mock(), is_alive=True)
    deployed = []

    class Backend(NuRecBackend):
        replay_scenario = config

        @property
        def native(self):
            return {"world": world, "actors": {"lead": lead}}

        def _hold_tick(self):
            pass

    def deploy(patch, supplied_world, target, *, tick, on_spawn):
        assert supplied_world is world and target is lead
        deployed.append(patch.texture)
        on_spawn(panel)
        return [panel]

    monkeypatch.setattr(PhysicalPatch, "apply", deploy)
    monkeypatch.setattr(carla_runner, "CarlaBackend", lambda _: Backend(
        {"ego0": {"speed": 2}}, renderer=StubRenderer(height=2, width=2),
    ))
    monkeypatch.setattr(carla_runner, "_stack", lambda _: NeutralStack())
    result = CliRunner().invoke(cli.app, ["run", str(path), "--frames", "1"])
    assert result.exit_code == 3, result.output
    assert deployed == [{"pattern": "checkerboard"}]
    panel.destroy.assert_called_once()


class NeutralStack(AVStack):
    def __call__(self, observation):
        return Control()


def test_cli_custom_factory_runs_real_nurec_pair_and_localization(user_methods, tmp_path):
    env = user_methods
    config = {
        "backend": {
            "type": "NuRecBackend", "renderer": {"type": "stub", "height": 2, "width": 2},
            "ego0": {"speed": 2},
        },
        "stack": {"factory": env.path + "Stack"},
        "runtime": {
            "localizer": {"factory": env.path + "Localizer"},
            "hooks": [
                {"stage": "observation", "factory": env.path + "SpeedPlugin", "options": {"speed": speed}}
                for speed in (9, 6)
            ],
        },
    }
    path = tmp_path / "nurec.yaml"
    path.write_text(yaml.safe_dump(config))
    result = CliRunner().invoke(cli.app, ["run", str(path), "--frames", "2"])
    assert result.exit_code == 3, result.output
    assert env.inputs == [[2, 2], [6, 6]]
    assert [event for event in env.events if event[0] == "input"] == [
        ("input", 2), ("input", 9), ("input", 2), ("input", 9),
    ]
    for speed in (9, 6):
        assert env.events.count(("reset", speed)) == env.events.count(("close", speed)) == 1
    assert len(env.localizers) == 2 and env.localizers[0] is not env.localizers[1]
    assert env.localizers[0].history == env.localizers[1].history == [0, 1]


def test_cli_custom_carla_stack_needs_no_embedded_modular_pipeline(
    monkeypatch, cli_config, user_methods,
):
    path, config = cli_config
    config["ego"].pop("pipeline")
    config.pop("attacks")
    config["stack"] = {"factory": user_methods.path + "Stack"}
    config["interventions"] = [{"kind": "defense", "factory": user_methods.path + "Defense"}]
    config.update(mode="single", treatment="defense")
    path.write_text(yaml.safe_dump(config))
    backend = NuRecBackend({"ego0": {"speed": 2}}, renderer=StubRenderer(height=2, width=2))
    backend.replay_scenario = config
    monkeypatch.setattr(carla_runner, "CarlaBackend", lambda _: backend)
    result = CliRunner().invoke(cli.app, ["run", str(path), "--frames", "2"])
    assert result.exit_code == 0, result.output
    assert user_methods.inputs == [[2, 1.6]]
    assert user_methods.events == [("defense", 0), ("defense", 1)]
    assert "outputs=2" in result.output


@pytest.mark.parametrize("treatment,disabled_method", [
    ("defense", False), ("clean", False), ("attack", False), ("attack", True),
], ids=["defense-side-effects", "repeated-clean", "empty-attack", "disabled-defense"])
@pytest.mark.parametrize("speed", [2, 0.1], ids=["driving", "stationary-baseline"])
def test_non_attack_pairs_report_comparisons_without_attack_outcomes(
    monkeypatch, user_methods, tmp_path, treatment, disabled_method, speed,
):
    from avsectester import metric

    config = deepcopy(user_methods.config)
    config["backend"]["ego0"]["speed"] = speed
    config["conditions"] = {"sensors": [{"type": "GaussianNoise", "severity": 0.1}]}
    if treatment == "defense" or disabled_method:
        config["interventions"] = [{"kind": "defense", "factory": user_methods.path + "Defense"}]
    path = tmp_path / "pair.yaml"
    path.write_text(yaml.safe_dump(config))
    args = ["run", str(path), "--treatment", treatment, "--frames", "6"]
    plot = Mock(wraps=metric.plot_impact)
    monkeypatch.setattr(metric, "plot_impact", plot)
    if treatment == "defense" and speed == 2:
        args += ["--plot", str(tmp_path / "defense.png")]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "ATTACK" not in result.output and "INCONCLUSIVE" not in result.output
    assert "no impact" not in result.output
    assert "final_speed=" in result.output
    if treatment == "defense":
        assert "Defense side effects" in result.output and "defense: final_speed= 0.00" in result.output
        assert len(user_methods.events) == 6
        if speed == 2:
            assert plot.call_args.kwargs["show_verdict"] is False
            assert plot.call_args.kwargs["treated_label"] == "defense"
            assert "Defense side effects" in plot.call_args.kwargs["title"]
    else:
        assert "baseline: final_speed=" in result.output and user_methods.events == []
        if treatment == "attack":
            assert "Comparing two baseline runs" in result.output


def test_combined_pair_labels_the_attack_with_defense_and_keeps_the_verdict(
    monkeypatch, user_methods, tmp_path,
):
    from avsectester import metric

    config = deepcopy(user_methods.config)
    config["runtime"] = {"hooks": [{"stage": "observation", "factory": user_methods.path + "NoopAttack"}]}
    config["interventions"] = [{"kind": "defense", "factory": user_methods.path + "Defense"}]
    path = tmp_path / "combined.yaml"
    path.write_text(yaml.safe_dump(config))
    plot = Mock(wraps=metric.plot_impact)
    monkeypatch.setattr(metric, "plot_impact", plot)
    output = tmp_path / "combined.png"
    result = CliRunner().invoke(cli.app, [
        "run", str(path), "--treatment", "attack-defense", "--frames", "6", "--plot", str(output),
    ])
    assert result.exit_code == 0, result.output
    assert "ATTACK SUCCEEDED" in result.output and "[attack+defense]" in result.output
    assert "attack+defense: final_speed=" in result.output
    assert plot.call_args.kwargs["treated_label"] == "attack+defense"
    assert plot.call_args.kwargs["show_verdict"] is True
    assert output.read_bytes().startswith(b"\x89PNG")


def test_raw_defense_configuration_points_to_the_method_format(user_methods, tmp_path):
    config = deepcopy(user_methods.config)
    config["runtime"] = {"hooks": [{
        "stage": "observation", "factory": user_methods.path + "SpeedFilter",
    }]}
    path = tmp_path / "filter.yaml"
    path.write_text(yaml.safe_dump(config))
    result = CliRunner().invoke(cli.app, ["run", str(path), "--treatment", "defense", "--frames", "2"])
    assert result.exit_code == 2, result.output
    assert "runtime.hooks" in result.output and "kind: defense" in result.output
    assert "interventions" in result.output and user_methods.inputs == []

    config.pop("runtime")
    config["interventions"] = [{"kind": "defense", "factory": user_methods.path + "FilterMethod"}]
    path.write_text(yaml.safe_dump(config))
    result = CliRunner().invoke(cli.app, [
        "run", str(path), "--mode", "single", "--treatment", "defense", "--frames", "2",
    ])
    assert result.exit_code == 0, result.output
    assert user_methods.inputs == [[1, 1]]
    result = CliRunner().invoke(cli.app, ["run", str(path), "--treatment", "attack", "--frames", "2"])
    assert result.exit_code == 0, result.output
    assert user_methods.inputs == [[1, 1], [2, 2], [2, 2]]
    assert "Comparing two baseline runs" in result.output


@pytest.mark.parametrize("invalid", [
    "stage", "command", "world-condition", "backend", "renderer", "dynamics", "backend-option",
])
def test_cli_classifies_preflight_errors_before_reset(monkeypatch, user_methods, tmp_path, invalid):
    config = {
        "backend": {"type": "NuRecBackend"},
        "stack": {"factory": user_methods.path + "Stack"},
    }
    if invalid == "stage":
        config["runtime"] = {"hooks": [{
            "stage": "percepton.post", "factory": user_methods.path + "SpeedPlugin",
            "options": {"speed": 6},
        }]}
    elif invalid == "command":
        config["stack"] = {"type": "AlpamayoAVStack", "options": {"checkpoint_path": "unused"}}
    elif invalid == "world-condition":
        config["conditions"] = {"world": {"weather": "ClearNoon"}}
    elif invalid == "backend":
        config["backend"]["type"] = "UnknownBackend"
    elif invalid == "backend-option":
        config["backend"]["dtt"] = 0.1
    else:
        config["backend"][invalid] = {"type": "unknown"}
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    reset = Mock(side_effect=AssertionError("World must not reset"))
    monkeypatch.setattr(NuRecBackend, "reset", reset)
    result = CliRunner().invoke(cli.app, ["run", str(path), "--frames", "1"])
    assert result.exit_code == 2, result.output
    assert "Configuration error" in result.output
    reset.assert_not_called()
