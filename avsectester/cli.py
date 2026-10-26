"""Prepare, execute and evaluate experiments through the public experiment API."""

from __future__ import annotations

from pathlib import Path
import shlex
import shutil
import subprocess
import time
from typing import Any, NoReturn, TYPE_CHECKING

import typer
import yaml  # type: ignore[import-untyped]

from avsectester import __version__
from avsectester.closed_loop.simulators.nurec.server import DEFAULT_NRE_IMAGE
from avsectester.experiment import (
    ConfigurationMismatchError, Experiment, ExperimentConfiguration, ExperimentResult,
    PreparedExperiment, load_configuration,
)
from avsectester.experiment.engine import validate_configuration_environment
from avsectester.experiment.output_directory import run_directory
from avsectester.validation import require_positive_number

if TYPE_CHECKING:
    from avsectester.closed_loop.simulators.nurec import NuRecRenderer

app = typer.Typer(help="Adversarial security testing for AV systems.")
nre_app = typer.Typer(help="Start and stop the configured local NuRec renderer.")
app.add_typer(nre_app, name="nre")
_CONFIGURATION_ERRORS = (ValueError, TypeError, ImportError, AttributeError, FileNotFoundError,
                         yaml.YAMLError)


def _configuration(path: Path) -> ExperimentConfiguration:
    try:
        configuration = load_configuration(path)
        if configuration.runs.output is None:
            raise ValueError("CLI experiments require an output directory in runs.output")
        validate_configuration_environment(configuration)
        return configuration
    except _CONFIGURATION_ERRORS as error:
        _configuration_error(error)
    except KeyboardInterrupt:
        typer.echo("Configuration loading interrupted.", err=True)
        raise typer.Exit(130) from None
    except Exception as error:
        typer.echo(f"Configuration load error: {type(error).__name__}: {error}", err=True)
        raise typer.Exit(1) from error


def _configuration_error(error: Exception) -> NoReturn:
    typer.echo(f"Configuration error: {type(error).__name__}: {error}", err=True)
    raise typer.Exit(2) from error


def _evaluate_result(result: ExperimentResult, configuration: ExperimentConfiguration) -> Any:
    from avsectester.experiment import evaluate
    from avsectester.experiment.html_report import write_offline_html

    assert configuration.runs.output is not None
    write_offline_html(configuration.runs.output)
    return evaluate(result, configuration)


def _print_report(report: Any, result: ExperimentResult,
                  configuration: ExperimentConfiguration) -> None:
    for notice in result.notices:
        typer.echo(notice)
    typer.echo(report.format_summary())
    assert configuration.runs.output is not None
    typer.echo(f"Offline HTML: {configuration.runs.output / 'offline_report.html'}")
    typer.echo(f"Evaluation HTML: {configuration.runs.output / 'evaluation_report.html'}")
    for failure in result.failures:
        assert configuration.runs.output is not None
        destination = run_directory(configuration.runs.output, failure.segment_index,
                                    failure.seed, failure.treatment)
        typer.echo(f"Failed run: {destination}: {failure.error or failure.log.status}", err=True)
    for selection in result.selection_errors:
        typer.echo(f"Selection error: {selection.attack_name}/{selection.scenario_id}: {selection.error}",
                   err=True)


def _exit_code(result: ExperimentResult, report: Any) -> int:
    return 1 if result.failures or result.selection_errors or report.metric_errors else 0


def _partial_report(configuration: ExperimentConfiguration, saved_this_call: bool) -> None:
    if not saved_this_call:
        return
    assert configuration.runs.output is not None
    try:
        partial = ExperimentResult.load(configuration.runs.output, configuration)
        report = _evaluate_result(partial, configuration)
        _print_report(report, partial, configuration)
    except KeyboardInterrupt:
        typer.echo("Partial evaluation interrupted.", err=True)
    except Exception as error:
        typer.echo(f"Partial evaluation unavailable: {type(error).__name__}: {error}", err=True)


@app.command()
def version() -> None:
    """Print the installed AVSecTester version."""
    typer.echo(f"avsectester {__version__}")


@app.command()
def run(
    config: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Experiment YAML file.",
    ),
    prepared: Path | None = typer.Option(
        None, exists=True, file_okay=False, readable=True,
        help="Reuse metadata and artifacts saved in this directory.",
    ),
    resume: bool = typer.Option(False, help="Resume preparation and unfinished runs in runs.output."),
) -> None:
    """Prepare attacks, execute treatments and write the configured evaluation report."""
    if resume and prepared is not None:
        _configuration_error(ValueError("--resume and --prepared cannot be used together"))
    configuration = _configuration(config)
    output = configuration.runs.output
    assert output is not None
    if not resume and output.exists() and (not output.is_dir() or any(output.iterdir())):
        _configuration_error(ValueError(f"Experiment output already exists: {output}"))
    preparation = None
    source = prepared
    if resume and (output / "prepared.json").is_file():
        source = output
    if source is not None:
        try:
            preparation = PreparedExperiment.load(configuration, source)
        except ConfigurationMismatchError as error:
            _configuration_error(error)
        except KeyboardInterrupt:
            typer.echo("Preparation loading interrupted.", err=True)
            raise typer.Exit(130) from None
        except Exception as error:
            typer.echo(f"Preparation load error: {type(error).__name__}: {error}", err=True)
            raise typer.Exit(1) from error
    saved_this_call = resume and (output / "prepared.json").is_file()
    try:
        if preparation is None:
            preparation = Experiment.prepare(configuration, resume=resume)
            # Preparation writes its manifest last. A failed prepare must never evaluate a
            # manifest left by an earlier invocation in the same output directory.
            saved_this_call = (output / "prepared.json").is_file()
        else:
            if not resume:
                preparation.save(output)
            saved_this_call = True
        result = Experiment.execute(preparation, resume=resume)
    except ConfigurationMismatchError as error:
        _configuration_error(error)
    except KeyboardInterrupt:
        typer.echo("Preparation or execution interrupted.", err=True)
        _partial_report(configuration, saved_this_call)
        if saved_this_call:
            typer.echo(f"Output directory: {output}")
        raise typer.Exit(130) from None
    except Exception as error:
        typer.echo(f"Execution error: {type(error).__name__}: {error}", err=True)
        _partial_report(configuration, saved_this_call)
        if saved_this_call:
            typer.echo(f"Output directory: {output}")
        raise typer.Exit(1) from error
    try:
        report = _evaluate_result(result, configuration)
        _print_report(report, result, configuration)
    except KeyboardInterrupt:
        typer.echo("Evaluation interrupted; report files may be incomplete. Saved run logs remain "
                   "available for evaluation.", err=True)
        typer.echo(f"Output directory: {output}")
        raise typer.Exit(130) from None
    except Exception as error:
        typer.echo(f"Evaluation error: {type(error).__name__}: {error}", err=True)
        typer.echo(f"Output directory: {output}")
        raise typer.Exit(1) from error
    typer.echo(f"Output directory: {configuration.runs.output}")
    raise typer.Exit(_exit_code(result, report))


@app.command()
def evaluate(
    config: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Experiment YAML file.",
    ),
) -> None:
    """Recompute evaluation from saved run logs without running a simulator or reading a dataset."""
    configuration = _configuration(config)
    assert configuration.runs.output is not None
    if not (configuration.runs.output / "prepared.json").is_file():
        typer.echo(f"No saved experiment results found: {configuration.runs.output / 'prepared.json'}", err=True)
        raise typer.Exit(1)
    try:
        result = ExperimentResult.load(configuration.runs.output, configuration)
    except ConfigurationMismatchError as error:
        _configuration_error(error)
    except KeyboardInterrupt:
        typer.echo("Result loading interrupted.", err=True)
        raise typer.Exit(130) from None
    except Exception as error:
        typer.echo(f"Result load error: {type(error).__name__}: {error}", err=True)
        raise typer.Exit(1) from error
    try:
        report = _evaluate_result(result, configuration)
        _print_report(report, result, configuration)
    except KeyboardInterrupt:
        typer.echo("Evaluation interrupted; report files may be incomplete. Saved run logs remain "
                   "available for evaluation.", err=True)
        typer.echo(f"Output directory: {configuration.runs.output}")
        raise typer.Exit(130) from None
    except Exception as error:
        typer.echo(f"Evaluation error: {type(error).__name__}: {error}", err=True)
        typer.echo(f"Output directory: {configuration.runs.output}")
        raise typer.Exit(1) from error
    typer.echo(f"Output directory: {configuration.runs.output}")
    raise typer.Exit(_exit_code(result, report))


def _nre_configuration(config: Path) -> tuple[ExperimentConfiguration, NuRecRenderer]:
    from avsectester.closed_loop.simulators.nurec import NuRecRenderer

    try:
        configuration = load_configuration(config)
        renderer = getattr(configuration.simulator, "renderer", None)
        if not isinstance(renderer, NuRecRenderer):
            raise ValueError("NRE commands require NuRecRenderer")
        return configuration, renderer
    except _CONFIGURATION_ERRORS as error:
        _configuration_error(error)


def _nre_launch_spec(config: Path, gpu: str, name: str | None, image: str,
                     cache_size: int) -> tuple[ExperimentConfiguration, list[str], str]:
    from avsectester.closed_loop.simulators.nurec.server import (
        default_container_name, renderer_command,
    )

    configuration, renderer = _nre_configuration(config)
    try:
        chosen_name = name or default_container_name(renderer)
        command = renderer_command(configuration.dataset, renderer, gpu=gpu,
                                   name=chosen_name, image=image, cache_size=cache_size)
        return configuration, command, chosen_name
    except _CONFIGURATION_ERRORS as error:
        _configuration_error(error)


def _docker_available() -> None:
    if shutil.which("docker") is None:
        typer.echo("docker is not on PATH; use nre serve --dry-run to print the command.", err=True)
        raise typer.Exit(1)


def _docker_call(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *arguments], capture_output=True, text=True, check=False)


def _remove_nre(name: str) -> None:
    _docker_call(["stop", name])
    _docker_call(["rm", name])


@nre_app.command("serve")
def nre_serve(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    gpu: str = typer.Option(..., help="Host GPU index passed to Docker --gpus device=N."),
    name: str | None = typer.Option(None, help="Container name; default avsectester-nre-PORT."),
    image: str = typer.Option(DEFAULT_NRE_IMAGE),
    cache_size: int = typer.Option(2, min=1),
    timeout: float = typer.Option(600., min=0.001),
    dry_run: bool = typer.Option(False, help="Only print the command; create no files or containers."),
) -> None:
    """Start the configured local NRE and wait for its Serving on log message."""
    try:
        require_positive_number(timeout, "timeout")
    except _CONFIGURATION_ERRORS as error:
        _configuration_error(error)
    configuration, command, chosen_name = _nre_launch_spec(config, gpu, name, image, cache_size)
    if dry_run:
        typer.echo(shlex.join(command))
        return
    _docker_available()
    if _docker_call(["inspect", chosen_name]).returncode == 0:
        typer.echo(f"Container already exists: {chosen_name}; choose --name or stop it explicitly.", err=True)
        raise typer.Exit(1)
    renderer = getattr(configuration.simulator, "renderer", None)
    assert renderer is not None  # Checked by _nre_launch_spec.
    if renderer.probe_directory is not None:
        Path(renderer.probe_directory).mkdir(parents=True, exist_ok=True)
    created = False
    try:
        launched = subprocess.run(command, capture_output=True, text=True, check=False)
        # The name did not exist before this invocation, so failed startup containers belong to it.
        created = True
        if launched.returncode:
            raise RuntimeError(launched.stderr.strip() or launched.stdout.strip() or "docker run failed")
        deadline = time.monotonic() + timeout
        logs = ""
        while True:
            response = _docker_call(["logs", "--tail", "50", chosen_name])
            logs = response.stdout + response.stderr
            state = _docker_call(["inspect", "--format", "{{.State.Running}}", chosen_name])
            if state.returncode or state.stdout.strip() != "true":
                raise RuntimeError("NRE container exited before becoming ready")
            if "Serving on" in logs:
                typer.echo(f"NRE ready: endpoint={renderer.endpoint}, GPU={gpu}, "
                           f"scenes={len(configuration.dataset.usdz_paths)}, container={chosen_name}")
                typer.echo(f"Stop command: {shlex.join(['avsectester', 'nre', 'stop', str(config), '--name', chosen_name])}")
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"NRE startup timed out after {timeout:g} seconds")
            time.sleep(min(.5, max(0., deadline - time.monotonic())))
    except BaseException as error:
        if created:
            response = _docker_call(["logs", "--tail", "50", chosen_name])
            logs = response.stdout + response.stderr
            typer.echo("\n".join(logs.splitlines()[-50:]), err=True)
            _remove_nre(chosen_name)
        if isinstance(error, KeyboardInterrupt):
            typer.echo("NRE startup interrupted.", err=True)
            raise typer.Exit(130) from None
        if not isinstance(error, Exception):
            raise
        typer.echo(f"NRE startup error: {error}", err=True)
        raise typer.Exit(1) from error


@nre_app.command("stop")
def nre_stop(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    name: str | None = typer.Option(None),
) -> None:
    """Stop and remove the local renderer container selected by this configuration."""
    from avsectester.closed_loop.simulators.nurec.server import default_container_name

    _, renderer = _nre_configuration(config)
    try:
        chosen_name = name or default_container_name(renderer)
    except _CONFIGURATION_ERRORS as error:
        _configuration_error(error)
    _docker_available()
    stopped = _docker_call(["stop", chosen_name])
    if stopped.returncode:
        typer.echo(stopped.stderr.strip() or f"NRE container does not exist: {chosen_name}", err=True)
        raise typer.Exit(1)
    removed = _docker_call(["rm", chosen_name])
    if removed.returncode:
        typer.echo(removed.stderr.strip() or f"Unable to remove NRE container: {chosen_name}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Stopped and removed NRE container: {chosen_name}")


if __name__ == "__main__":
    app()
