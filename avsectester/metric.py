"""Impact metric: did the attack cause a driving consequence the clean run never had?

The security question is differential — not "does the ego brake?" but "does the *attack* make it
brake/stop when the identical clean run did not?". :func:`impact` diffs two
:class:`~avsectester.scenario.Trace`s (clean vs attacked) into that verdict.

A verdict is only meaningful if the **clean** run actually drove. If the clean ego never got moving
(too few frames, or it was stuck at the spawn), then "the attack induced braking" is vacuous — an
already-stopped car braking proves nothing. So the metric first checks a driving baseline
(``clean.peak_speed >= baseline_speed``); without it the result is INCONCLUSIVE, not a success.
The real success signal is ``induced_stop`` (the attack stopped a car that was otherwise driving).
"""

from __future__ import annotations

from dataclasses import dataclass

from .scenario import Trace


@dataclass
class Impact:
    clean_peak_speed: float
    clean_final_speed: float
    attacked_final_speed: float
    clean_brake_frames: int
    attacked_brake_frames: int
    clean_drove: bool          # did the clean run establish a real driving baseline?
    induced_braking: bool      # attack braked more than clean did
    induced_stop: bool         # attack stopped a car the clean run kept driving

    @property
    def attack_succeeded(self) -> bool:
        """Meaningful success = the attack stopped an otherwise-driving car."""
        return self.clean_drove and self.induced_stop

    @property
    def verdict(self) -> str:
        if not self.clean_drove:
            return (
                f"INCONCLUSIVE — clean run never drove (peak {self.clean_peak_speed:.2f} m/s); "
                "run more frames or check the spawn"
            )
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
            f"brake_frames={self.attacked_brake_frames}\n"
            f"=> {self.verdict}"
        )


def impact(
    clean: Trace,
    attacked: Trace,
    stop_speed: float = 0.5,
    baseline_speed: float = 1.0,
) -> Impact:
    """Diff clean vs attacked into an :class:`Impact`.

    :param stop_speed: at/below this speed (m/s) the ego counts as stopped.
    :param baseline_speed: the clean run must have peaked at least this fast to count as "drove".
    """
    clean_drove = clean.peak_speed >= baseline_speed
    return Impact(
        clean_peak_speed=clean.peak_speed,
        clean_final_speed=clean.final_speed,
        attacked_final_speed=attacked.final_speed,
        clean_brake_frames=clean.braking_frames,
        attacked_brake_frames=attacked.braking_frames,
        clean_drove=clean_drove,
        induced_braking=attacked.braking_frames > clean.braking_frames,
        induced_stop=(attacked.final_speed <= stop_speed) and (clean.final_speed > stop_speed),
    )
