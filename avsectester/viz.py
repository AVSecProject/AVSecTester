"""Visualization: turn a clean-vs-attacked run into a driving-impact figure.

The whole security story is in the two :class:`~avsectester.scenario.Trace`s the run already
collects, so the plot needs no extra sensors — it draws ego speed and brake command over time for
the clean and attacked runs on shared axes, making "clean cruises / attacked brakes to a stop"
visible at a glance. matplotlib is a lazy import (install with ``pip install -e ".[viz]"``).
"""

from __future__ import annotations

from .metric import Impact, impact
from .scenario import Trace


def plot_impact(
    clean: Trace,
    attacked: Trace,
    path: str,
    title: str = "Phantom-detection attack — driving impact",
    result: Impact | None = None,
) -> str:
    """Render clean vs attacked (ego speed + brake command over frames) to ``path``; returns it."""
    import os

    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    result = result or impact(clean, attacked)
    cf = [r.frame for r in clean.records]
    af = [r.frame for r in attacked.records]

    fig, (ax_s, ax_b) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # -- speed
    ax_s.plot(cf, [r.speed for r in clean.records], color="#2a7", lw=2, label="clean")
    ax_s.plot(af, [r.speed for r in attacked.records], color="#d33", lw=2, label="attacked")
    ax_s.axhline(0.5, color="#888", ls=":", lw=1)
    ax_s.set_ylabel("ego speed (m/s)")
    ax_s.legend(loc="upper right")
    ax_s.grid(alpha=0.25)

    # -- brake command, shaded
    ax_b.fill_between(af, [r.brake for r in attacked.records], color="#d33", alpha=0.25, step="mid")
    ax_b.plot(af, [r.brake for r in attacked.records], color="#d33", lw=2, label="attacked")
    ax_b.plot(cf, [r.brake for r in clean.records], color="#2a7", lw=2, label="clean")
    ax_b.set_ylabel("brake command")
    ax_b.set_xlabel("frame")
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.legend(loc="upper left")
    ax_b.grid(alpha=0.25)

    # -- verdict caption
    fig.text(0.5, 0.005, result.verdict, ha="center", fontsize=11,
             color="#d33" if result.attack_succeeded else "#555")

    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
