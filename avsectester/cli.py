"""AVSecTester command-line interface.

A single verb — ``run`` — takes a scenario config, drives it clean then attacked in real CARLA, and
prints the impact. The scenario is built entirely from avstack/avcarla registries (see
:mod:`avsectester.scenario`); attacks are avstack hooks declared in the config.
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
    frames: int = typer.Option(None, help="Override the scenario frame count."),
    gpu: int = typer.Option(
        None, help="Override the perception CUDA device (e.g. 1 on a host where CARLA holds GPU 0)."
    ),
) -> None:
    """Run a scenario clean then attacked in CARLA and print the attack's driving impact."""
    from .metric import impact
    from .scenario import run_scenario, set_perception_gpu

    scenario = set_perception_gpu(yaml.safe_load(Path(config).read_text()), gpu)
    n = frames or scenario.get("frames", 40)
    attacks = scenario.get("attacks", [])

    typer.echo(f"[clean]    running {n} frames ...")
    clean = run_scenario(scenario, attacks=None, frames=n)
    typer.echo(f"[attacked] running {n} frames with {len(attacks)} attack hook(s) ...")
    attacked = run_scenario(scenario, attacks=attacks, frames=n)

    result = impact(clean, attacked)
    typer.echo(str(result))
    raise typer.Exit(0 if result.attack_succeeded else 1)


if __name__ == "__main__":
    app()
