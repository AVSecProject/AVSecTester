"""AVSecTester command-line interface.

Thin entry point; real orchestration lands with the runner (Phase 2+).
"""

from __future__ import annotations

import typer

from . import __version__
from .config import ATTACKS, BACKENDS, DEFENSES, HAVE_AVSTACK, METRICS

app = typer.Typer(help="Closed-loop adversarial stress-testing for AV systems.")


@app.command()
def version() -> None:
    """Print the AVSecTester version and whether the avstack stack is available."""
    typer.echo(f"avsectester {__version__}")
    typer.echo(f"avstack stack available: {HAVE_AVSTACK}")


@app.command()
def registry() -> None:
    """List registered plugins (attacks / defenses / backends / metrics)."""
    # Import plugin packages so their @register_module classes self-register.
    import avsectester.attacks
    import avsectester.backends.carla_backend
    import avsectester.backends.dataset_backend
    import avsectester.backends.mock_backend
    import avsectester.defenses
    import avsectester.metrics  # noqa: F401

    for name, reg in [
        ("attacks", ATTACKS),
        ("defenses", DEFENSES),
        ("backends", BACKENDS),
        ("metrics", METRICS),
    ]:
        # avstack's Registry exposes `module_dict`; our fallback shim uses `_entries`.
        entries = getattr(reg, "module_dict", None)
        if entries is None:
            entries = getattr(reg, "_entries", {})
        typer.echo(f"{name}: {sorted(entries)}")


@app.command()
def validate(spec_path: str) -> None:
    """Validate a security-experiment YAML spec against the schema."""
    from .core import ExperimentSpec

    spec = ExperimentSpec.from_yaml(spec_path)
    typer.echo(f"OK: {spec.name}")


@app.command()
def run(spec_path: str, report_path: str = "") -> None:
    """Run a security experiment (clean vs attacked) and print the escalation report."""
    # import plugin packages so backends/attacks/defenses/metrics self-register
    import avsectester.attacks
    import avsectester.backends.carla_backend
    import avsectester.backends.dataset_backend
    import avsectester.backends.mock_backend
    import avsectester.defenses
    import avsectester.metrics  # noqa: F401

    from .core import ExperimentSpec
    from .core.engine import ExperimentRunner
    from .reports import render_report

    spec = ExperimentSpec.from_yaml(spec_path)
    result = ExperimentRunner(spec).run()
    report = render_report(result)
    typer.echo(report)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(report)
        typer.echo(f"\n(report written to {report_path})")
    raise typer.Exit(code=0 if result.metrics["escalated"] else 1)


if __name__ == "__main__":
    app()
