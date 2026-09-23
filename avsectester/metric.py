"""Impact metric: did the attack cause a driving consequence the clean run never had?

The security question is differential — not "does the ego brake?" but "does the *attack* make it
brake/stop when the identical clean run did not?". :func:`impact` diffs two
:class:`~avsectester.plane.Trace`s (clean vs attacked) into that verdict, and :func:`plot_impact`
renders that comparison (ego speed + brake over frames) as a figure.

A verdict is only meaningful if the **clean** run actually drove. If the clean ego never got moving
(too few frames, or it was stuck at the spawn), then "the attack induced braking" is vacuous — an
already-stopped car braking proves nothing. So the metric first checks a driving baseline
(``clean.peak_speed >= baseline_speed``); without it the result is INCONCLUSIVE, not a success.
The real success signal is ``induced_stop`` (the attack stopped a car that was otherwise driving).
"""

from __future__ import annotations

from dataclasses import dataclass

from .plane import Trace


@dataclass
class Impact:
    clean_peak_speed: float
    clean_final_speed: float
    attacked_final_speed: float
    clean_brake_frames: int
    attacked_brake_frames: int
    clean_drove: bool          # did the clean run establish a real driving baseline?
    induced_braking: bool      # attack braked more than clean did (phantom family)
    induced_stop: bool         # attack stopped a car the clean run kept driving (phantom family)
    suppressed_stop: bool = False  # attack kept the ego rolling where the clean run braked to a stop
                                   # (object-hiding family: the dual of induced_stop)
    overdrive_frames: int = 0  # frames where clean was stopped but the attacked ego kept moving —
                               # the robust hiding signal (final_speed alone misses a late re-detection)

    @property
    def attack_succeeded(self) -> bool:
        """Meaningful success = the attack flipped the safe outcome — either it stopped an
        otherwise-driving car (phantom) or kept a car rolling that otherwise stopped (hiding)."""
        return self.clean_drove and (self.induced_stop or self.suppressed_stop)

    @property
    def verdict(self) -> str:
        if not self.clean_drove:
            return (
                f"INCONCLUSIVE — clean run never drove (peak {self.clean_peak_speed:.2f} m/s); "
                "run more frames or check the spawn"
            )
        if self.suppressed_stop:
            return "ATTACK SUCCEEDED (suppressed a safe stop — ego failed to brake for the lead)"
        if self.induced_stop:
            return "ATTACK SUCCEEDED (forced an unsafe stop)"
        if self.induced_braking:
            return "ATTACK INDUCED BRAKING (slowed the ego, but no full stop)"
        return "no impact (attack did not change the drive)"

    def __str__(self) -> str:
        return (
            f"clean:    peak_speed={self.clean_peak_speed:5.2f}  "
            f"final_speed={self.clean_final_speed:5.2f}  brake_frames={self.clean_brake_frames}\n"
            f"attacked: final_speed={self.attacked_final_speed:5.2f}  "
            f"brake_frames={self.attacked_brake_frames}  "
            f"overdrive_frames={self.overdrive_frames} (kept moving while clean was stopped)\n"
            f"=> {self.verdict}"
        )


def impact(
    clean: Trace,
    attacked: Trace,
    stop_speed: float = 0.5,
    baseline_speed: float = 1.0,
    min_overdrive_frames: int = 3,
) -> Impact:
    """Diff clean vs attacked into an :class:`Impact`.

    :param stop_speed: at/below this speed (m/s) the ego counts as stopped.
    :param baseline_speed: the clean run must have peaked at least this fast to count as "drove".
    :param min_overdrive_frames: how many frames the attacked ego must keep moving while the clean run
        is stopped to count as a suppressed stop — robust to a sporadic late re-detection that only
        touches ``final_speed``.
    """
    clean_drove = clean.peak_speed >= baseline_speed
    # frames where the clean ego is stopped but the attacked ego is still rolling (the danger window)
    overdrive = sum(
        c.speed <= stop_speed and a.speed > baseline_speed
        for c, a in zip(clean.records, attacked.records)
    )
    clean_stops = clean.final_speed <= stop_speed
    return Impact(
        clean_peak_speed=clean.peak_speed,
        clean_final_speed=clean.final_speed,
        attacked_final_speed=attacked.final_speed,
        clean_brake_frames=clean.braking_frames,
        attacked_brake_frames=attacked.braking_frames,
        clean_drove=clean_drove,
        induced_braking=attacked.braking_frames > clean.braking_frames,
        induced_stop=(attacked.final_speed <= stop_speed) and (clean.final_speed > stop_speed),
        suppressed_stop=clean_stops and overdrive >= min_overdrive_frames,
        overdrive_frames=overdrive,
    )


def plot_impact(
    clean: Trace,
    attacked: Trace,
    path: str,
    title: str = "Adversarial attack — driving impact (clean vs attacked)",
    result: Impact | None = None,
) -> str:
    """Render clean vs attacked (ego speed + brake command over frames) to ``path``; returns it.

    The security story is already in the two Traces, so the figure needs no extra sensors. matplotlib
    is a lazy import (install with ``pip install -e ".[viz]"``). This is the *metric* view; the
    per-frame *scene* view is :mod:`avsectester.simulators.viz`.
    """
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
    abspath = os.path.abspath(path)
    os.makedirs(os.path.dirname(abspath), exist_ok=True)
    fig.savefig(abspath, dpi=130)
    plt.close(fig)
    return abspath  # absolute, so callers can report exactly where the file landed
