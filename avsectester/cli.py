"""Prepare, execute and evaluate experiments through the public experiment API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

import typer
import yaml  # type: ignore[import-untyped]

from avsectester import __version__
from avsectester.experiment import (
    ConfigurationMismatchError, Experiment, ExperimentConfiguration, ExperimentResult,
    PreparedExperiment, load_configuration,
)
from avsectester.experiment.engine import validate_configuration_environment
from avsectester.experiment.models import run_directory, runs_directory

app = typer.Typer(help="Adversarial security testing for AV systems.")
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
    from avsectester.evaluation.report import evaluate

    return evaluate(result, configuration)


def _print_report(report: Any, result: ExperimentResult,
                  configuration: ExperimentConfiguration) -> None:
    for notice in result.notices:
        typer.echo(notice)
    typer.echo(report.format_summary())
    for failure in result.failures:
        assert configuration.runs.output is not None
        destination = run_directory(configuration.runs.output, failure.attack_name, failure.segment_index,
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
) -> None:
    """Prepare attacks, execute treatments and write the configured evaluation report."""
    configuration = _configuration(config)
    output = configuration.runs.output
    assert output is not None
    if runs_directory(output).exists():
        _configuration_error(ValueError(f"Run output already exists: {runs_directory(output)}"))
    preparation = None
    if prepared is not None:
        try:
            preparation = PreparedExperiment.load(configuration, prepared)
        except ConfigurationMismatchError as error:
            _configuration_error(error)
        except KeyboardInterrupt:
            typer.echo("Preparation loading interrupted.", err=True)
            raise typer.Exit(130) from None
        except Exception as error:
            typer.echo(f"Preparation load error: {type(error).__name__}: {error}", err=True)
            raise typer.Exit(1) from error
    saved_this_call = False
    try:
        if preparation is None:
            preparation = Experiment.prepare(configuration)
            # Preparation writes its manifest last. A failed prepare must never evaluate a
            # manifest left by an earlier invocation in the same output directory.
            saved_this_call = (output / "prepared.json").is_file()
        else:
            preparation.save(output)
            saved_this_call = True
        result = Experiment.execute(preparation)
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


if __name__ == "__main__":
    app()
