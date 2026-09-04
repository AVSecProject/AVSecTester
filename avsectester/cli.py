"""AVSecTester command-line interface — the one entry point for the demo.

``avsectester run <scenario.yaml>`` is the whole thing: it builds the scenario from config (an
avcarla ego running an avstack ``ModularDrivingPipeline`` in real CARLA), drives it **clean** then
**attacked**, scores the difference with the impact metric, optionally saves a driving-impact plot,
and exits 0 (attack succeeded) / 2 (inconclusive) / 1 (no impact). The scenario is built entirely
from avstack/avcarla registries (see :mod:`avsectester.scenario`); attacks are avstack hooks declared
in the config.
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from . import __version__

app = typer.Typer(help="Adversarial security testing for AV systems.")


@app.command()
def version() -> None:
    """Print the AVSecTester version."""
    typer.echo(f"avsectester {__version__}")


@app.command()
def run(
    config: Path = typer.Argument(..., help="Scenario YAML (see configs/carla_scenario.yaml)."),
    frames: int = typer.Option(None, help="Override the scenario frame count (default 40)."),
    gpu: int = typer.Option(
        None, help="Override the perception CUDA device (e.g. 1 on a host where CARLA holds GPU 0)."
    ),
    plot: str = typer.Option(
        None,
        "--plot",
        help="Save a clean-vs-attacked driving-impact plot (e.g. results/impact.png).",
    ),
) -> None:
    """Run a scenario clean then attacked in real CARLA and report the attack's driving impact.

    Needs a running CARLA server (docs/DOCKER.md) and the carla-vehicle weights
    (scripts/fetch_models.sh). This is the end-to-end demo.
    """
    from .metric import impact
    from .scenario import run_scenario, set_perception_gpu

    scenario = set_perception_gpu(yaml.safe_load(Path(config).read_text()), gpu)
    n = frames or scenario.get("frames", 40)
    attacks = scenario.get("attacks", [])

    typer.echo(f"[clean]    {n} frames ...")
    clean = run_scenario(scenario, attacks=None, frames=n)
    typer.echo(
        f"[clean]    mean_detections={clean.mean_detections:.1f} peak_speed={clean.peak_speed:.2f} "
        f"final_speed={clean.final_speed:.2f} brake_frames={clean.braking_frames}"
    )

    typer.echo(f"[attacked] {n} frames with {len(attacks)} attack hook(s) ...")
    attacked = run_scenario(scenario, attacks=attacks, frames=n)
    typer.echo(
        f"[attacked] mean_detections={attacked.mean_detections:.1f} "
        f"final_speed={attacked.final_speed:.2f} brake_frames={attacked.braking_frames}"
    )

    result = impact(clean, attacked)

    if plot:
        from .viz import plot_impact

        plot_impact(clean, attacked, plot, result=result)

    typer.echo(str(result))  # ends in "=> <verdict>"
    if plot:
        typer.echo(f"[plot]     saved {plot}")

    # exit code encodes the verdict for scripting: 0 success, 2 inconclusive, 1 no meaningful impact
    raise typer.Exit(0 if result.attack_succeeded else 2 if not result.clean_drove else 1)


if __name__ == "__main__":
    app()
