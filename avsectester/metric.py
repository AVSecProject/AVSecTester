"""Impact metric: did the attack cause a driving consequence the clean run never had?

The security question is differential — not "does the ego brake?" but "does the *attack* make it
brake/stop when the identical clean run did not?". :func:`impact` diffs two :class:`~avsectester
.scenario.Trace`s (clean vs attacked) into that verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

from .scenario import Trace


@dataclass
class Impact:
    clean_final_speed: float
    attacked_final_speed: float
    clean_brake_frames: int
    attacked_brake_frames: int
    induced_braking: bool
    induced_stop: bool

    @property
    def attack_succeeded(self) -> bool:
        return self.induced_stop or self.induced_braking

    def __str__(self) -> str:
        verdict = "ATTACK SUCCEEDED" if self.attack_succeeded else "no impact"
        stop = " (forced an unsafe stop)" if self.induced_stop else ""
        return (
            f"clean:    final_speed={self.clean_final_speed:5.2f}  "
            f"brake_frames={self.clean_brake_frames}\n"
            f"attacked: final_speed={self.attacked_final_speed:5.2f}  "
            f"brake_frames={self.attacked_brake_frames}\n"
            f"=> {verdict}{stop}"
        )


def impact(clean: Trace, attacked: Trace, stop_speed: float = 0.5) -> Impact:
    return Impact(
        clean_final_speed=clean.final_speed,
        attacked_final_speed=attacked.final_speed,
        clean_brake_frames=clean.braking_frames,
        attacked_brake_frames=attacked.braking_frames,
        induced_braking=attacked.braking_frames > clean.braking_frames,
        induced_stop=(attacked.final_speed <= stop_speed) and (clean.final_speed > stop_speed),
    )
