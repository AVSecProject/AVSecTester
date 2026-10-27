"""Prepare, execute and evaluate experiments through the public experiment API."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shlex
from typing import Any, NoReturn, TYPE_CHECKING

import typer
import yaml  # type: ignore[import-untyped]

from avsectester import __version__
from avsectester.closed_loop.simulators.nurec.server import (
    DEFAULT_NRE_IMAGE, DEFAULT_TRAFFIC_IMAGE, local_services, start_container, stop_container,
)
from avsectester.experiment import (
    ConfigurationMismatchError, Experiment, ExperimentConfiguration, ExperimentResult,
    PreparedExperiment, load_configuration,
)
from avsectester.experiment.engine import validate_configuration_environment
from avsectester.experiment.output_directory import run_directory
from avsectester.validation import require_positive_number

if TYPE_CHECKING:
    from avsectester.closed_loop.simulators.nurec import AlpaSimTraffic, NuRecRenderer

app = typer.Typer(help="Adversarial security testing for AV systems.")
'Command-line application and its service subcommands.'
nre_app = typer.Typer(help="Start and stop the configured local NuRec renderer.")
app.add_typer(nre_app, name="nre")
trafficsim_app = typer.Typer(help="Start and stop configured local CATK traffic.")
app.add_typer(trafficsim_app, name="trafficsim")
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
    nre_gpu: str | None = typer.Option(None, help="Host GPU for an automatically started NRE."),
    nre_name: str | None = typer.Option(None, help="Name of an automatically started NRE container."),
    trafficsim_gpu: str | None = typer.Option(None, help="Host GPU for automatically started traffic."),
    trafficsim_models: Path | None = typer.Option(None, file_okay=False, help="Local CATK weights directory."),
    trafficsim_name: str | None = typer.Option(None, help="Name of an automatically started traffic container."),
    service_timeout: float = typer.Option(600., min=.001, help="Service startup timeout in seconds."),
) -> None:
    """Prepare attacks, execute treatments and write the configured evaluation report."""
    if resume and prepared is not None:
        _configuration_error(ValueError("--resume and --prepared cannot be used together"))
    configuration = _configuration(config)
    output = configuration.runs.output
    assert output is not None
    if output.exists() and not output.is_dir():
        _configuration_error(ValueError(f"Experiment output must be a directory: {output}"))
    if not resume and output.exists() and any(output.iterdir()):
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
    if preparation is None:
        try:
            Experiment.validate(configuration, resume=resume)
        except _CONFIGURATION_ERRORS as error:
            _configuration_error(error)
        except KeyboardInterrupt:
            typer.echo("Preparation validation interrupted.", err=True)
            raise typer.Exit(130) from None
        except Exception as error:
            typer.echo(f"Preparation validation error: {type(error).__name__}: {error}", err=True)
            raise typer.Exit(1) from error

    saved_this_call = resume and (output / "prepared.json").is_file()
    phase = "services"
    try:
        with local_services(configuration.dataset, configuration.simulator,
                            nre_gpu=nre_gpu, nre_name=nre_name,
                            trafficsim_gpu=trafficsim_gpu, trafficsim_models=trafficsim_models,
                            trafficsim_name=trafficsim_name, timeout_s=service_timeout, log=typer.echo):
            phase = "execution"
            if preparation is None:
                preparation = Experiment.prepare(configuration, resume=resume)
                saved_this_call = (output / "prepared.json").is_file()
            else:
                if not resume:
                    preparation.save(output)
                saved_this_call = True
            result = Experiment.execute(preparation, resume=resume)
            phase = "evaluation"
            report = _evaluate_result(result, configuration)
            _print_report(report, result, configuration)
    except ConfigurationMismatchError as error:
        _configuration_error(error)
    except KeyboardInterrupt:
        if phase == "services":
            typer.echo("Service startup interrupted.", err=True)
        elif phase == "evaluation":
            typer.echo("Evaluation interrupted; report files may be incomplete. Saved run logs remain "
                       "available for evaluation.", err=True)
        else:
            typer.echo("Preparation or execution interrupted.", err=True)
            _partial_report(configuration, saved_this_call)
        if saved_this_call:
            typer.echo(f"Output directory: {output}")
        raise typer.Exit(130) from None
    except Exception as error:
        if phase == "services" and isinstance(error, _CONFIGURATION_ERRORS):
            _configuration_error(error)
        label = {"services": "Service", "execution": "Execution", "evaluation": "Evaluation"}[phase]
        typer.echo(f"{label} error: {type(error).__name__}: {error}", err=True)
        if phase == "execution":
            _partial_report(configuration, saved_this_call)
        if saved_this_call:
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
        default_renderer_container_name, renderer_command,
    )

    configuration, renderer = _nre_configuration(config)
    try:
        chosen_name = name or default_renderer_container_name(renderer)
        command = renderer_command(configuration.dataset, renderer, gpu=gpu,
                                   name=chosen_name, image=image, cache_size=cache_size)
        return configuration, command, chosen_name
    except _CONFIGURATION_ERRORS as error:
        _configuration_error(error)


def _start_service(command: list[str], *, name: str, service: str, timeout: float,
                   dry_run: bool, ready_log: str, readiness: Callable[[float], Any] | None = None,
                   prepare: Callable[[], Any] | None = None) -> None:
    """Adapt lifecycle exceptions and dry-run output to CLI exit conventions."""
    if dry_run:
        typer.echo(shlex.join(command))
        return
    try:
        start_container(command, name=name, ready_log=ready_log, timeout_s=timeout,
                        readiness=readiness, prepare=prepare, log=typer.echo)
    except KeyboardInterrupt:
        typer.echo(f"{service} startup interrupted.", err=True)
        raise typer.Exit(130) from None
    except Exception as error:
        typer.echo(f"{service} startup error: {error}", err=True)
        raise typer.Exit(1) from error


def _stop_service(name: str, service: str) -> None:
    try:
        stop_container(name)
    except Exception as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Stopped and removed {service} container: {name}")


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
    renderer = getattr(configuration.simulator, "renderer", None)
    assert renderer is not None  # Checked by _nre_launch_spec.

    def prepare() -> None:
        if renderer.probe_directory is not None:
            Path(renderer.probe_directory).mkdir(parents=True, exist_ok=True)

    _start_service(command, name=chosen_name, service="NRE", timeout=timeout, dry_run=dry_run,
                     ready_log="Serving on", prepare=prepare)
    if not dry_run:
        typer.echo(f"NRE ready: endpoint={renderer.endpoint}, GPU={gpu}, "
                   f"scenes={len(configuration.dataset.usdz_paths)}, container={chosen_name}")
        typer.echo(f"Stop it with: {shlex.join(['nre', 'stop', str(config), '--name', chosen_name])}")


@nre_app.command("stop")
def nre_stop(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    name: str | None = typer.Option(None),
) -> None:
    """Stop and remove the local renderer container selected by this configuration."""
    from avsectester.closed_loop.simulators.nurec.server import default_renderer_container_name

    _, renderer = _nre_configuration(config)
    try:
        chosen_name = name or default_renderer_container_name(renderer)
    except _CONFIGURATION_ERRORS as error:
        _configuration_error(error)
    _stop_service(chosen_name, "NRE")


def _traffic_configuration(config: Path) -> tuple[ExperimentConfiguration, AlpaSimTraffic]:
    from avsectester.closed_loop.simulators.nurec import AlpaSimTraffic

    try:
        configuration = load_configuration(config)
        traffic = getattr(configuration.simulator, "traffic", None)
        if not isinstance(traffic, AlpaSimTraffic):
            raise ValueError("Trafficsim commands require simulator.options.traffic: AlpaSimTraffic")
        return configuration, traffic
    except _CONFIGURATION_ERRORS as error:
        _configuration_error(error)


@trafficsim_app.command("serve")
def trafficsim_serve(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    gpu: str = typer.Option(..., help="Host GPU index passed to Docker --gpus device=N."),
    models: Path = typer.Option(..., file_okay=False, help="Directory containing CATK weights and tokens."),
    name: str | None = typer.Option(None, help="Container name; default avsectester-trafficsim-PORT."),
    image: str = typer.Option(DEFAULT_TRAFFIC_IMAGE),
    timeout: float = typer.Option(600., min=0.001),
    dry_run: bool = typer.Option(False, help="Only print the command; create no files or containers."),
) -> None:
    """Start CATK traffic and wait for its startup log and successful metadata RPC."""
    from avsectester.closed_loop.simulators.nurec.server import (
        default_traffic_container_name, traffic_command,
    )

    configuration, traffic = _traffic_configuration(config)
    try:
        require_positive_number(timeout, "timeout")
        chosen_name = name or default_traffic_container_name(traffic)
        command = traffic_command(configuration.dataset, traffic, gpu=gpu, name=chosen_name,
                                  image=image, models=models)
        if not dry_run and not models.is_dir():
            raise ValueError(f"Traffic model directory does not exist: {models}")
    except _CONFIGURATION_ERRORS as error:
        _configuration_error(error)

    def ready(remaining: float) -> Any:
        return traffic.get_metadata(timeout_s=min(1., remaining))

    _start_service(command, name=chosen_name, service="trafficsim", timeout=timeout, dry_run=dry_run,
                     ready_log="Starting CATK traffic gRPC server on", readiness=ready)
    if not dry_run:
        typer.echo(f"Trafficsim ready: endpoint={traffic.endpoint}, GPU={gpu}, "
                   f"scenes={len(configuration.dataset.usdz_paths)}, container={chosen_name}")
        typer.echo(f"Stop it with: {shlex.join(['trafficsim', 'stop', str(config), '--name', chosen_name])}")


@trafficsim_app.command("stop")
def trafficsim_stop(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    name: str | None = typer.Option(None),
) -> None:
    """Stop the configured traffic container without scanning the dataset or model paths."""
    from avsectester.closed_loop.simulators.nurec.server import default_traffic_container_name

    _, traffic = _traffic_configuration(config)
    try:
        chosen_name = name or default_traffic_container_name(traffic)
    except _CONFIGURATION_ERRORS as error:
        _configuration_error(error)
    _stop_service(chosen_name, "trafficsim")


if __name__ == "__main__":
    app()


__all__ = [
    'app',
]
