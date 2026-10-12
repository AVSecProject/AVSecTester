"""User-facing command behavior with controlled experiment outcomes."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
import yaml
from avsectester import carla_runner as scenario
from avsectester.cli import app
from typer.testing import CliRunner


@pytest.fixture
def cli_config(tmp_path):
    config = {
        "frames": 6,
        "client": {"type": "CarlaClient", "seed": 0},
        "ego": {
            "type": "CarlaMobileActor",
            "pipeline": {
                "perception": {"type": "MMDetObjectDetector3D", "gpu": 1},
            },
        },
        "attacks": [{"stage": "perception", "hook": {"type": "PhantomInjection"}}],
    }
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config


@pytest.mark.parametrize(
    "args,frames,gpu",
    [
        ([], 6, 1),
        (["--frames", "10", "--gpu", "0"], 10, 0),
        (["--frames", "0"], 0, 1),
    ],
)
def test_cli_loads_config_and_runs_clean_then_attacked(
    monkeypatch,
    cli_config,
    make_trace,
    args,
    frames,
    gpu,
):
    path, config = cli_config
    clean, attacked = make_trace([0, 2, 5]), make_trace([0, 1, 0], [0, 1, 1])
    run = Mock(return_value=(clean, attacked))
    monkeypatch.setattr(scenario, "run_scenario_pair", run)
    result = CliRunner().invoke(app, ["run", str(path), *args])

    assert result.exit_code == 0, result.output
    run.assert_called_once()
    expected = deepcopy(config)
    expected["ego"]["pipeline"]["perception"]["gpu"] = gpu
    assert run.call_args.args == (expected,)
    assert run.call_args.kwargs["frames"] == frames
    assert run.call_args.kwargs["clean_runtime"].hooks == ()
    attacked_hooks = run.call_args.kwargs["attacked_runtime"].hooks
    assert [hook.stage for hook in attacked_hooks] == ["perception.post"]
    assert attacked_hooks[0].handler.config == config["attacks"][0]["hook"]
    assert result.output.index("[clean]") < result.output.index("[attacked]")
    assert result.output.index("[scenario] seed=0") < result.output.index("[pair]")
    assert "peak_speed=5.00" in result.output
    assert "ATTACK SUCCEEDED" in result.output
    assert yaml.safe_load(path.read_text()) == config


@pytest.mark.parametrize(
    "clean_speeds,attacked_speeds,brakes,code,verdict",
    [
        ([0, 2, 5], [0, 1, 0], [0, 1, 1], 0, "ATTACK SUCCEEDED"),
        ([0, 2, 5], [0, 2, 5], [0, 0, 0], 3, "no impact"),
        ([0, 0.1, 0.1], [0, 0, 0], [0, 1, 1], 4, "INCONCLUSIVE"),
    ],
)
def test_cli_exit_code_matches_verdict(
    monkeypatch,
    cli_config,
    make_trace,
    clean_speeds,
    attacked_speeds,
    brakes,
    code,
    verdict,
):
    path, _ = cli_config
    monkeypatch.setattr(
        scenario,
        "run_scenario_pair",
        Mock(
            return_value=(
                make_trace(clean_speeds),
                make_trace(attacked_speeds, brakes),
            )
        ),
    )
    result = CliRunner().invoke(app, ["run", str(path)])
    assert result.exit_code == code, result.output
    assert verdict in result.output


def test_cli_optional_plot_creates_file(monkeypatch, cli_config, make_trace, tmp_path):
    path, _ = cli_config
    monkeypatch.setattr(
        scenario,
        "run_scenario_pair",
        Mock(
            return_value=(
                make_trace([0, 2, 5]),
                make_trace([0, 1, 0], [0, 1, 1]),
            )
        ),
    )
    output = tmp_path / "plots" / "impact.png"
    result = CliRunner().invoke(app, ["run", str(path), "--plot", str(output)])
    assert result.exit_code == 0, result.output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert str(output) in result.output


@pytest.mark.parametrize("failure,code,label", [
    (RuntimeError("server unavailable"), 1, "Execution error"),
    (ValueError("handler failed"), 1, "Execution error"),
])
@pytest.mark.parametrize("seed", [0, None])
def test_cli_runtime_errors_are_distinct_from_experiment_outcomes(
    monkeypatch, cli_config, failure, code, label, seed,
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

    def fail(configured_scene, **kwargs):
        # A failure during preparation must not hide the chosen seed from the caller.
        expected_seed = 987 if seed is None else seed
        assert configured_scene["client"]["seed"] == expected_seed
        echo.assert_any_call(f"[scenario] seed={expected_seed}")
        raise failure

    monkeypatch.setattr(scenario, "run_scenario_pair", fail)
    result = CliRunner().invoke(app, ["run", str(path)])
    assert result.exit_code == code
    assert label in result.output and str(failure) in result.output
    assert generated.call_count == (1 if seed is None else 0)
    assert yaml.safe_load(path.read_text()) == config


@pytest.mark.parametrize("invalid", ["-1", "malformed-yaml", "unconsumed-runtime"])
def test_cli_rejects_invalid_configuration_without_running(monkeypatch, cli_config, invalid):
    path, config = cli_config
    args = []
    if invalid == "-1":
        args = ["--frames", "-1"]
    elif invalid == "malformed-yaml":
        path.write_text("runtime: [")
    else:
        config["runtime"] = {"unused_option": True}
        path.write_text(yaml.safe_dump(config))
    pair = Mock()
    monkeypatch.setattr(scenario, "run_scenario_pair", pair)
    result = CliRunner().invoke(app, ["run", str(path), *args])
    assert result.exit_code == 2
    pair.assert_not_called()


def test_cli_consumes_physical_patches(monkeypatch, cli_config, make_trace):
    from avsectester.attacks.patch.physical_patch import PhysicalPatchPlugin

    path, config = cli_config
    config.pop("attacks")
    config["patches"] = [{"target": "lead", "texture": {"pattern": "checkerboard"}}]
    path.write_text(yaml.safe_dump(config))
    pair = Mock(return_value=(make_trace([2, 2]), make_trace([2, 2])))
    monkeypatch.setattr(scenario, "run_scenario_pair", pair)
    result = CliRunner().invoke(app, ["run", str(path)])
    assert result.exit_code == 3
    hook, = pair.call_args.kwargs["attacked_runtime"].hooks
    assert hook.stage == "world.setup" and isinstance(hook.handler, PhysicalPatchPlugin)
    assert hook.handler.patches == config["patches"]
    assert "1 intervention hook" in result.output


def test_cli_custom_factory_executes_through_real_pair_and_runtime(monkeypatch, cli_config):
    import sys
    from types import ModuleType
    from dataclasses import replace
    from avsectester.backend import AVStack
    from avsectester.plane import Control, StateEstimate
    from avsectester.runtime import Plugin
    from avsectester.simulators.nurec import NuRecBackend, StubRenderer

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

    module = ModuleType("research_handlers")
    module.make_speed = SpeedPlugin
    module.Localizer = Localizer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path, config = cli_config
    config.pop("attacks")
    config["runtime"] = {"localizer": {"factory": "research_handlers:Localizer"}, "hooks": [
        {"stage": "observation", "factory": "research_handlers:make_speed", "options": {"speed": speed}}
        for speed in (9, 6)
    ]}
    path.write_text(yaml.safe_dump(config))
    backend = NuRecBackend({"ego0": {"speed": 2}}, renderer=StubRenderer(height=2, width=2))
    backend.replay_scenario = config
    monkeypatch.setattr(scenario, "CarlaBackend", lambda *a, **kw: backend)
    monkeypatch.setattr(scenario, "_stack", lambda config: Stack())
    result = CliRunner().invoke(app, ["run", str(path), "--frames", "2"])
    assert result.exit_code == 3, result.output
    assert inputs == [[2, 2], [6, 6]]
    assert [event for event in events if event[0] == "input"] == [
        ("input", 2), ("input", 9), ("input", 2), ("input", 9),
    ]
    assert events.count(("reset", 9)) == events.count(("close", 9)) == 1
    assert events.count(("reset", 6)) == events.count(("close", 6)) == 1
    assert len(localizers) == 2 and localizers[0] is not localizers[1]
    assert localizers[0].history == localizers[1].history == [0, 1]
