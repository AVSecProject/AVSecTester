"""Attack-robustness harness: the clean-vs-attacked-under-corruption grid + its aggregation.

A tiny closed-loop stub exercises the whole path without a simulator: a bright "lead" is rendered each
frame; the stack brakes when it detects the lead (mean brightness over a threshold); the attack hides the
lead (blacks the frame) so the ego keeps rolling — a *suppressed stop*, the object-hiding success signal.
Two deterministic corruptions make the effect legible: one reveals the hidden lead (neutralises the
attack), one does not (attack survives). This tests the harness wiring, the impact scoring, and the ASR /
resilience aggregation.
"""

import math
from dataclasses import replace

import numpy as np
from avsectester.backend import AVStack, WorldBackend
from avsectester.evaluation.robustness import ConditionResult, RobustnessReport, evaluate_robustness
from avsectester.plane import Control, Observation
from avsectester.scenarios.source import ScenarioInstance, ScenarioSource
from avsectester.simulators.augment import AugmentationPipeline, Corruption


class _ApproachBackend(WorldBackend):
    """Ego at 10 m/s facing a lead; braking sheds 5 m/s/frame. Renders a bright front frame (the lead)."""

    def __init__(self):
        self.speed = 10.0
        self.t = 0.0
        self.frame = 0

    def _obs(self):
        return Observation(t=self.t, frame=self.frame, ego_speed=self.speed,
                           sensor_data={"front": np.full((8, 8, 3), 200, np.uint8)})

    def reset(self):
        self.speed, self.t, self.frame = 10.0, 0.0, 0
        return self._obs()

    def step(self, control):
        if control.brake > 0:
            self.speed = max(0.0, self.speed - 5.0)
        self.frame += 1
        self.t += 0.1
        return self._obs()


class _BrakeIfLeadSeen(AVStack):
    """Brake when the front frame looks bright enough to be a lead; otherwise accelerate."""

    def __call__(self, obs):
        rgb = obs.sensor_data["front"]
        return Control(brake=1.0) if rgb.mean() > 50 else Control(throttle=1.0)


def _hide_attack(match, backend):
    """The attack: black out the front frame so the lead is not detected (object-hiding)."""
    def perturb(obs):
        data = dict(obs.sensor_data)
        data["front"] = np.zeros_like(data["front"])
        return replace(obs, sensor_data=data)
    return perturb


class _Match:                       # duck-types ScenarioMatch (harness only reads .camera)
    camera = "front"


class _StubSource(ScenarioSource):
    def __init__(self, n=1):
        self.n = n

    def scenarios(self, req, limit=None):
        for i in range(self.n if limit is None else min(self.n, limit)):
            yield ScenarioInstance(make_backend=lambda: _ApproachBackend(), target=_Match(),
                                   provenance={"scene_id": f"stub-{i}"})


class _LiftBlack(Corruption):       # a corruption that reveals the hidden lead -> neutralises the attack
    def apply(self, rgb, rng):
        return np.clip(rgb.astype(int) + 100, 0, 255).astype(np.uint8)


class _Mild(Corruption):            # a corruption that does not cross the detection threshold from black
    def apply(self, rgb, rng):
        return np.clip(rgb.astype(int) + 1, 0, 255).astype(np.uint8)


def _report(n=1):
    return evaluate_robustness(
        _StubSource(n), req=None, attack_for=_hide_attack, stack=_BrakeIfLeadSeen, frames=8,
        conditions=[AugmentationPipeline([_LiftBlack()]), AugmentationPipeline([_Mild()])])


def test_harness_scores_asr_and_resilience_per_condition():
    r = _report()
    assert set(r.per_condition) == {"clean", "_LiftBlack@0.50", "_Mild@0.50"}
    assert r.baseline_asr == 1.0                              # attack succeeds clean (suppressed stop)
    assert r.per_condition["_Mild@0.50"].asr == 1.0          # survives the mild corruption
    assert r.per_condition["_LiftBlack@0.50"].asr == 0.0     # neutralised when the lead is revealed
    assert r.resilience("_Mild@0.50") == 1.0
    assert r.resilience("_LiftBlack@0.50") == 0.0
    assert math.isclose(r.mean_resilience, 0.5)               # mean over the two corruptions
    assert "mean resilience" in r.summary()


def test_aggregates_over_multiple_scenarios():
    r = _report(n=3)
    assert r.per_condition["clean"].n == 3 and r.per_condition["clean"].successes == 3
    assert len(r.rows) == 3 * 3                               # 3 scenarios x 3 conditions


def test_default_conditions_use_common_corruptions():
    # with no explicit conditions, the suite is the common-corruptions set (+ the clean baseline)
    r = evaluate_robustness(_StubSource(), req=None, attack_for=_hide_attack,
                            stack=_BrakeIfLeadSeen, frames=8, severity=0.5)
    from avsectester.simulators.augment import CORRUPTIONS
    assert len(r.per_condition) == len(CORRUPTIONS) + 1       # + clean baseline
    assert r.baseline_asr == 1.0


def test_report_aggregation_is_pure():
    r = RobustnessReport()
    r.record("s1", "clean", True)
    r.record("s2", "clean", False)
    r.record("s1", "fog", True)
    assert r.per_condition["clean"].asr == 0.5 and r.baseline_asr == 0.5
    assert r.resilience("fog") == 1.0                          # fog ASR 1.0 / baseline 0.5, clamped to 1
    assert ConditionResult("x").asr == 0.0                     # empty condition -> 0, no div-by-zero