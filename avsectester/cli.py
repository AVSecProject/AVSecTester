"""AVSecTester command-line interface."""

from __future__ import annotations

import typer

from . import __version__
from .config import ATTACKS, DEFENSES, ENVIRONMENTS, HAVE_AVSTACK, METRICS, SYSTEMS

app = typer.Typer(help="Adversarial security testing for AV systems.")


def _load_plugins() -> None:
    """Import plugin packages so their @register_module classes self-register."""
    import avsectester.attacks
    import avsectester.defenses
    import avsectester.envs
    import avsectester.metrics  # noqa: F401


def _entries(reg) -> list[str]:
    d = getattr(reg, "module_dict", None) or getattr(reg, "_entries", {})
    return sorted(d)


@app.command()
def version() -> None:
    """Print the AVSecTester version and whether the avstack stack is available."""
    typer.echo(f"avsectester {__version__}")
    typer.echo(f"avstack stack available: {HAVE_AVSTACK}")


@app.command()
def registry() -> None:
    """List registered plugins (environments / systems / attacks / defenses / metrics)."""
    _load_plugins()
    for name, reg in [("environments", ENVIRONMENTS), ("systems", SYSTEMS),
                      ("attacks", ATTACKS), ("defenses", DEFENSES), ("metrics", METRICS)]:
        typer.echo(f"{name}: {_entries(reg)}")


@app.command()
def run(config_path: str, report_path: str = "") -> None:
    """Run an experiment (clean vs attacked [vs defended]) from a YAML config."""
    from pathlib import Path

    import yaml

    from .core import run_experiment
    from .reports import render_report

    _load_plugins()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    metric = METRICS.build(dict(cfg.get("metric") or {"type": "ImpactMetric"}))
    attack = ATTACKS.build(dict(cfg["attack"])) if cfg.get("attack") else None
    defense = DEFENSES.build(dict(cfg["defense"])) if cfg.get("defense") else None
    result = run_experiment(
        make_env=lambda: ENVIRONMENTS.build(dict(cfg["environment"])),
        make_system=lambda: SYSTEMS.build(dict(cfg["system"])),
        metric=metric, attack=attack, defense=defense,
    )
    report = render_report(cfg.get("name", config_path), result)
    typer.echo(report)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(report)
        typer.echo(f"\n(report written to {report_path})")
    raise typer.Exit(code=0 if result.metrics.get("impacted") else 1)


if __name__ == "__main__":
    app()
