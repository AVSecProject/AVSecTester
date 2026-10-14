"""Run configured single or paired experiments through prepared scenarios."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from . import __version__
from .experiment import ConfigurationError, ExecutionMode, TreatmentMode, run_configured

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
    plot: str | None = typer.Option(None, help="Save a paired driving-impact plot."),
    mode: ExecutionMode | None = typer.Option(None, help="single or pair (default: pair)."),
    treatment: TreatmentMode | None = typer.Option(None, help="clean, attack, defense or attack-defense."),
) -> None:
    """Run a prepared scenario with the selected attack and defense treatment."""
    from .metric import impact, plot_impact
    from .simulators.carla import resolve_scenario_seed, set_perception_gpu

    try:
        scenario = yaml.safe_load(config.read_text())
        if not isinstance(scenario, dict):
            raise ValueError("Scenario YAML must contain a mapping")
        execution = mode or ExecutionMode(scenario.get("mode", "pair"))
        selected = treatment or TreatmentMode(scenario.get("treatment", "attack"))
        if plot is not None and execution is ExecutionMode.SINGLE:
            raise ValueError("--plot requires pair mode")
        backend_spec = scenario.get("backend", {})
        if not isinstance(backend_spec, dict):
            raise ValueError("backend must be a mapping")
        is_carla = backend_spec.get("type", "CarlaBackend") in {"carla", "CarlaBackend"}
        if is_carla:
            for key in ("client", "ego"):
                if not isinstance(scenario.get(key), dict):
                    raise ValueError(f"Scenario requires a {key!r} configuration mapping")
            if scenario.get("stack") is None and not isinstance(scenario["ego"].get("pipeline"), dict):
                raise ValueError("Scenario ego requires a pipeline configuration mapping")
            if scenario.get("stack") is not None and gpu is not None:
                raise ValueError("Configure a custom stack's device in stack.options instead of --gpu")
            scenario = resolve_scenario_seed(set_perception_gpu(scenario, gpu))
        elif gpu is not None:
            raise ValueError("--gpu overrides CARLA perception. Configure stack.options.device for NuRec.")
        n = frames if frames is not None else scenario.get("frames", 40)
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError("frames must be a nonnegative integer")
    except (ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(2) from error
    except OSError as error:
        typer.echo(f"Cannot read scenario: {error}", err=True)
        raise typer.Exit(2) from error

    try:
        if is_carla:
            typer.echo(f"[scenario] seed={scenario['client']['seed']}")
        typer.echo(f"[{execution.value}] {n} frames per run, treatment={selected.value}")
        options = {}
        if is_carla:
            from . import carla_runner
            options = {
                "backend_factory": None if scenario.get("selection") is not None else
                    lambda config: carla_runner.CarlaBackend(config),
                "stack_factory": lambda: carla_runner._stack(scenario),
                "run_fn": carla_runner._logged_run,
            }
        traces = run_configured(
            scenario, n, mode=execution, treatment=selected,
            provenance={"config_path": str(config.resolve())}, **options,
        )
        has_attack = False
        if execution is ExecutionMode.PAIR:
            clean = traces["clean"]
            treated = next(trace for label, trace in traces.items() if label != "clean")
            has_attack = selected in {TreatmentMode.ATTACK, TreatmentMode.COMBINED} and (
                any(item.get("kind") == "attack" for item in treated.provenance.get("interventions", ()))
                or bool((scenario.get("runtime") or {}).get("hooks"))
            )
        for label, trace in traces.items():
            displayed = {"attack": "attacked", "attack-defense": "attack+defense"}.get(label, label)
            if execution is ExecutionMode.PAIR and label == "attack" and not has_attack:
                displayed = "baseline"
            typer.echo(
                f"[{displayed}] mean_detections={trace.mean_detections:.1f} "
                f"peak_speed={trace.peak_speed:.2f} final_speed={trace.final_speed:.2f} "
                f"brake_frames={trace.braking_frames} outputs={len(trace.outputs)}"
            )
        if execution is ExecutionMode.SINGLE:
            return
        if selected is TreatmentMode.DEFENSE:
            typer.echo("[pair] Defense side effects relative to clean.")
        elif selected is TreatmentMode.CLEAN:
            typer.echo("[pair] Baseline comparison.")
        elif not has_attack:
            typer.echo("[pair] No intervention hooks configured. Comparing two baseline runs.")
        result = impact(clean, treated)
        treated_label = (
            "attack+defense" if selected is TreatmentMode.COMBINED else
            "attacked" if has_attack else
            "defense" if selected is TreatmentMode.DEFENSE else "baseline"
        )
        title = "Defense side effects (clean vs defense)" if selected is TreatmentMode.DEFENSE else None
        saved = plot_impact(
            clean, treated, plot, result=result, title=title,
            treated_label=treated_label, show_verdict=has_attack,
        ) if plot else None
        comparison = result.comparison(treated_label)
        typer.echo(f"{comparison}\n=> {result.verdict}" if has_attack else comparison)
        if saved:
            typer.echo(f"[output] wrote driving-impact figure: {saved}")
    except ValueError as error:
        # Stage validation precedes reset and identifies configuration errors explicitly.
        from .runtime import UnsupportedStageError
        from .backend import CommandCompatibilityError
        from .conditions import ConditionConfigurationError
        from .scenarios.setup import ScenarioConfigurationError

        if isinstance(error, (UnsupportedStageError, CommandCompatibilityError,
                              ConditionConfigurationError, ScenarioConfigurationError, ConfigurationError)):
            typer.echo(f"Configuration error: {error}", err=True)
            raise typer.Exit(2) from error
        typer.echo(f"Execution error: {error}", err=True)
        raise typer.Exit(1) from error
    except Exception as error:
        typer.echo(f"Execution error: {error}", err=True)
        raise typer.Exit(1) from error

    # Keep experiment outcomes distinct from runtime (1) and configuration (2) failures.
    if not has_attack:
        return
    raise typer.Exit(0 if result.attack_succeeded else 4 if not result.clean_drove else 3)


if __name__ == "__main__":
    app()
