"""Run configured CARLA comparisons through the shared runtime and paired runner."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from . import __version__
from .interventions import configured_runtime_pair

app = typer.Typer(help="Adversarial security testing for AV systems.")


@app.command()
def version() -> None:
    """Print the installed AVSecTester version."""
    typer.echo(f"avsectester {__version__}")


@app.command()
def run(
    config: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True,
        help="Scenario YAML (see configs/carla_scenario.yaml).",
    ),
    frames: int | None = typer.Option(None, min=0, help="Override the scenario frame count."),
    gpu: int | None = typer.Option(None, min=0, help="Override the perception CUDA device."),
    plot: str | None = typer.Option(None, help="Save a clean-vs-attacked driving-impact plot."),
) -> None:
    """Run one prepared clean/attacked pair in CARLA, including configured runtime hooks."""
    from .carla_runner import run_scenario_pair
    from .metric import impact, plot_impact
    from .simulators.carla import resolve_scenario_seed, set_perception_gpu

    try:
        scenario = yaml.safe_load(config.read_text())
        if not isinstance(scenario, dict):
            raise ValueError("Scenario YAML must contain a mapping")
        for key in ("client", "ego"):
            if not isinstance(scenario.get(key), dict):
                raise ValueError(f"Scenario requires a {key!r} configuration mapping")
        if not isinstance(scenario["ego"].get("pipeline"), dict):
            raise ValueError("Scenario ego requires a pipeline configuration mapping")
        n = frames if frames is not None else scenario.get("frames", 40)
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError("frames must be a nonnegative integer")
        scenario = resolve_scenario_seed(set_perception_gpu(scenario, gpu))
        clean_runtime, attacked_runtime = configured_runtime_pair(scenario)
    except (ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(2) from error
    except OSError as error:
        typer.echo(f"Cannot read scenario: {error}", err=True)
        raise typer.Exit(2) from error

    try:
        typer.echo(f"[scenario] seed={scenario['client']['seed']}")
        typer.echo(f"[pair] {n} frames per run, {len(attacked_runtime.hooks)} intervention hook(s)")
        if not attacked_runtime.hooks:
            typer.echo("[pair] No intervention hooks configured. Comparing two baseline runs.")
        clean, attacked = run_scenario_pair(
            scenario, frames=n, clean_runtime=clean_runtime, attacked_runtime=attacked_runtime,
        )
        typer.echo(
            f"[clean]    mean_detections={clean.mean_detections:.1f} "
            f"peak_speed={clean.peak_speed:.2f} final_speed={clean.final_speed:.2f} "
            f"brake_frames={clean.braking_frames}"
        )
        typer.echo(
            f"[attacked] mean_detections={attacked.mean_detections:.1f} "
            f"final_speed={attacked.final_speed:.2f} brake_frames={attacked.braking_frames}"
        )
        result = impact(clean, attacked)
        saved = plot_impact(clean, attacked, plot, result=result) if plot else None
        typer.echo(str(result))
        if saved:
            typer.echo(f"[output] wrote driving-impact figure: {saved}")
    except ValueError as error:
        # Stage validation precedes reset and identifies configuration errors explicitly.
        from .runtime import UnsupportedStageError

        if isinstance(error, UnsupportedStageError):
            typer.echo(f"Configuration error: {error}", err=True)
            raise typer.Exit(2) from error
        typer.echo(f"Execution error: {error}", err=True)
        raise typer.Exit(1) from error
    except Exception as error:
        typer.echo(f"Execution error: {error}", err=True)
        raise typer.Exit(1) from error

    # Keep experiment outcomes distinct from runtime (1) and configuration (2) failures.
    raise typer.Exit(0 if result.attack_succeeded else 4 if not result.clean_drove else 3)


if __name__ == "__main__":
    app()
