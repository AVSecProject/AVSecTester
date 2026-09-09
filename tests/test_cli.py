"""User-facing command behavior with controlled experiment outcomes."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
import yaml
from avsectester import scenario
from avsectester.cli import app
from typer.testing import CliRunner


@pytest.fixture
def cli_config(tmp_path):
    config = {
        "frames": 6,
        "client": {"type": "CarlaClient"},
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
    run = Mock(side_effect=[clean, attacked])
    monkeypatch.setattr(scenario, "run_scenario", run)
    result = CliRunner().invoke(app, ["run", str(path), *args])

    assert result.exit_code == 0, result.output
    assert run.call_count == 2
    expected = deepcopy(config)
    expected["ego"]["pipeline"]["perception"]["gpu"] = gpu
    assert run.call_args_list[0].args == (expected,)
    assert run.call_args_list[0].kwargs == {"attacks": None, "frames": frames}
    assert run.call_args_list[1].args == (expected,)
    assert run.call_args_list[1].kwargs == {"attacks": config["attacks"], "frames": frames}
    assert result.output.index("[clean]") < result.output.index("[attacked]")
    assert "peak_speed=5.00" in result.output
    assert "ATTACK SUCCEEDED" in result.output
    assert yaml.safe_load(path.read_text()) == config


@pytest.mark.parametrize(
    "clean_speeds,attacked_speeds,brakes,code,verdict",
    [
        ([0, 2, 5], [0, 1, 0], [0, 1, 1], 0, "ATTACK SUCCEEDED"),
        ([0, 2, 5], [0, 2, 5], [0, 0, 0], 1, "no impact"),
        ([0, 2, 5], [0, 2, 2], [0, 1, 1], 1, "ATTACK INDUCED BRAKING"),
        ([0, 0.1, 0.1], [0, 0, 0], [0, 1, 1], 2, "INCONCLUSIVE"),
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
        "run_scenario",
        Mock(
            side_effect=[
                make_trace(clean_speeds),
                make_trace(attacked_speeds, brakes),
            ]
        ),
    )
    result = CliRunner().invoke(app, ["run", str(path)])
    assert result.exit_code == code, result.output
    assert verdict in result.output


def test_cli_optional_plot_creates_file(monkeypatch, cli_config, make_trace, tmp_path):
    pytest.importorskip("matplotlib")
    path, _ = cli_config
    monkeypatch.setattr(
        scenario,
        "run_scenario",
        Mock(
            side_effect=[
                make_trace([0, 2, 5]),
                make_trace([0, 1, 0], [0, 1, 1]),
            ]
        ),
    )
    output = tmp_path / "plots" / "impact.png"
    result = CliRunner().invoke(app, ["run", str(path), "--plot", str(output)])
    assert result.exit_code == 0, result.output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert str(output) in result.output


@pytest.mark.parametrize("config_text", ["ego: [", "", "{}"])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="CLI does not validate scenario structure or turn configuration exceptions into user errors",
)
def test_invalid_config_reports_a_readable_error_without_running(
    monkeypatch,
    tmp_path,
    config_text,
    make_trace,
):
    path = tmp_path / "invalid.yaml"
    path.write_text(config_text)
    run = Mock(return_value=make_trace([0, 2, 5]))
    monkeypatch.setattr(scenario, "run_scenario", run)
    result = CliRunner().invoke(app, ["run", str(path)])
    assert result.exit_code != 0
    run.assert_not_called()
    assert "error" in result.output.lower()


@pytest.mark.parametrize("frames", ["0", "-1"])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="CLI accepts nonpositive frame counts (zero falls back to config; negative is passed on)",
)
def test_nonpositive_frames_are_rejected_before_running(
    monkeypatch,
    cli_config,
    make_trace,
    frames,
):
    path, _ = cli_config
    run = Mock(return_value=make_trace([0, 2, 5]))
    monkeypatch.setattr(scenario, "run_scenario", run)
    result = CliRunner().invoke(app, ["run", str(path), "--frames", frames])
    assert result.exit_code != 0
    run.assert_not_called()
