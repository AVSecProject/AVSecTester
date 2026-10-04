"""Attack-robustness harness: the clean-vs-attacked-under-corruption grid + its aggregation.

A tiny closed-loop stub exercises the whole path without a simulator: a bright "lead" is rendered each
frame. The stack brakes when it detects the lead (mean brightness over a threshold). The attack hides the
lead (blacks the frame) so the ego keeps rolling — a *suppressed stop*, the object-hiding success signal.
Two deterministic corruptions make the effect legible: one reveals the hidden lead (neutralises the
attack), one does not (attack survives). This tests the harness wiring, the impact scoring, and the ASR /
resilience aggregation.
"""

import math
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from avsectester.backend import AVStack, WorldBackend
from avsectester.evaluation.robustness import ConditionResult, RobustnessReport, evaluate_robustness
from avsectester.plane import Control, Observation
from avsectester.scenarios.source import ScenarioInstance, ScenarioSource
from avsectester.simulators.augment import AugmentationPipeline, Corruption


class _ApproachBackend(WorldBackend):
    """Ego at 10 m/s facing a lead. Braking sheds 5 m/s/frame. Renders a bright front frame (the lead)."""

    def __init__(self):
        self.speed = 10.0
        self.t = 0.0
        self.frame = 0

    def _obs(self):
        return Observation(
            t=self.t,
            frame=self.frame,
            ego_speed=self.speed,
            sensor_data={"front": np.full((8, 8, 3), 200, np.uint8)},
        )

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
    """Brake when the front frame looks bright enough to be a lead. Otherwise accelerate."""

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


class _Match:  # duck-types ScenarioMatch (harness only reads .camera)
    camera = "front"


class _StubSource(ScenarioSource):
    def __init__(self, n=1, backend_factory=_ApproachBackend):
        self.n = n
        self.backend_factory = backend_factory

    def scenarios(self, req, limit=None):
        for i in range(self.n if limit is None else min(self.n, limit)):
            yield ScenarioInstance(
                make_backend=self.backend_factory,
                target=_Match(),
                provenance={"scene_id": f"stub-{i}"},
            )


class _LiftBlack(Corruption):  # a corruption that reveals the hidden lead -> neutralises the attack
    def apply(self, rgb, rng):
        return np.clip(rgb.astype(int) + 100, 0, 255).astype(np.uint8)


class _Mild(Corruption):  # a corruption that does not cross the detection threshold from black
    def apply(self, rgb, rng):
        return np.clip(rgb.astype(int) + 1, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("n", [1, 3])
def test_harness_scores_each_condition_and_aggregates_scenarios(n):
    r = evaluate_robustness(
        _StubSource(n),
        req=None,
        attack_for=_hide_attack,
        stack=_BrakeIfLeadSeen,
        frames=8,
        conditions=[AugmentationPipeline([_LiftBlack()]), AugmentationPipeline([_Mild()])],
    )
    assert r.per_condition["clean"].n == n
    assert r.per_condition["clean"].successes == n
    assert len(r.rows) == n * 3
    assert set(r.per_condition) == {"clean", "_LiftBlack@0.50", "_Mild@0.50"}
    assert r.baseline_asr == 1.0  # attack succeeds clean (suppressed stop)
    assert r.per_condition["_Mild@0.50"].asr == 1.0  # survives the mild corruption
    assert r.per_condition["_LiftBlack@0.50"].asr == 0.0  # neutralised when the lead is revealed
    assert r.resilience("_Mild@0.50") == 1.0
    assert r.resilience("_LiftBlack@0.50") == 0.0
    assert math.isclose(r.mean_resilience, 0.5)  # mean over the two corruptions
    assert "mean resilience" in r.summary()


def test_default_conditions_use_common_corruptions():
    # with no explicit conditions, the suite is the common-corruptions set (+ the clean baseline)
    r = evaluate_robustness(
        _StubSource(),
        req=None,
        attack_for=_hide_attack,
        stack=_BrakeIfLeadSeen,
        frames=8,
        severity=0.5,
    )
    from avsectester.simulators.augment import CORRUPTIONS

    assert len(r.per_condition) == len(CORRUPTIONS) + 1  # + clean baseline
    assert r.baseline_asr == 1.0


def test_excluded_attempts_do_not_change_valid_denominator():
    report = RobustnessReport()
    for status in ("success", "failure", "skipped", "inconclusive"):
        report.record(status, "clean", status, reason=status)
    condition = report.per_condition["clean"]
    assert condition.n == 2 and condition.asr == 0.5
    assert condition.skipped == condition.inconclusive == 1
    assert [row.status for row in report.rows] == ["success", "failure", "skipped", "inconclusive"]
    assert report.rows[-1].reason == "inconclusive"
    assert "inconclusive" in report.summary()
    report.record("fog-success", "fog", "success")
    assert report.resilience("fog") == 1.0  # preserve the existing ratio cap
    assert math.isnan(ConditionResult("empty").asr)


def test_skipped_attack_does_not_run_and_closes_backend():
    backend = _ApproachBackend()
    backend.close = Mock()
    backend.prepare_clean_attack_pair = Mock()
    source = _StubSource(backend_factory=lambda: backend)
    run = Mock(side_effect=AssertionError("Skipped experiments must not run"))
    report = evaluate_robustness(
        source, None, lambda *a: None, _BrakeIfLeadSeen, 8, conditions=[], run_fn=run
    )
    assert report.rows[0].status == "skipped"
    assert report.per_condition["clean"].n == 0
    assert math.isnan(report.baseline_asr)
    assert "N/A" in report.summary()
    backend.prepare_clean_attack_pair.assert_not_called()
    backend.close.assert_called_once()


def test_clean_that_never_drove_is_inconclusive():
    class StoppedBackend(_ApproachBackend):
        def reset(self):
            super().reset()
            self.speed = 0.0
            return self._obs()

    source = _StubSource(backend_factory=StoppedBackend)
    report = evaluate_robustness(source, None, _hide_attack, _BrakeIfLeadSeen, 8, conditions=[])
    assert report.rows[0].status == "inconclusive"
    assert "never drove" in report.rows[0].reason
    assert report.per_condition["clean"].n == 0
    assert math.isnan(report.baseline_asr)


@pytest.mark.parametrize("fail_run", [1, 2])
def test_backend_is_closed_when_either_run_fails(fail_run):
    backend = _ApproachBackend()
    backend.prepare_clean_attack_pair = Mock()
    backend.close = Mock()
    source = _StubSource(backend_factory=lambda: backend)
    from avsectester.plane import Trace

    run = Mock(side_effect=[Trace()] * (fail_run - 1) + [RuntimeError("simulation failed")])

    with pytest.raises(RuntimeError, match="simulation failed"):
        evaluate_robustness(
            source, None, _hide_attack, _BrakeIfLeadSeen, 8, conditions=[], run_fn=run
        )
    backend.prepare_clean_attack_pair.assert_called_once()
    backend.close.assert_called_once()


def test_harness_pairs_corruptions_across_different_simulator_frame_ids():
    from avsectester.simulators.augment import GaussianNoise

    class OffsetBackend(_ApproachBackend):
        def __init__(self):
            super().__init__()
            self.prepared = False
            self.resets = 0

        def prepare_clean_attack_pair(self):
            self.prepared = True

        def reset(self):
            assert self.prepared
            super().reset()
            self.frame = 100 if self.resets == 0 else 250
            self.resets += 1
            return self._obs()

    runs = []

    class CapturingStack(_BrakeIfLeadSeen):
        def __init__(self):
            self.observations = []
            runs.append(self.observations)

        def __call__(self, obs):
            self.observations.append(obs)
            return super().__call__(obs)

    pipeline = AugmentationPipeline([GaussianNoise()], seed=42)
    evaluate_robustness(
        _StubSource(backend_factory=OffsetBackend),
        None,
        lambda match, backend: lambda obs: obs,
        CapturingStack,
        frames=3,
        conditions=[pipeline],
    )
    assert len(runs) == 4  # baseline and corruption, each with clean and attacked runs
    clean, attacked = runs[2:]
    assert [obs.frame for obs in clean] == [100, 101, 102]
    assert [obs.frame for obs in attacked] == [250, 251, 252]
    for c, a in zip(clean, attacked, strict=True):
        np.testing.assert_array_equal(c.sensor_data["front"], a.sensor_data["front"])
    assert not np.array_equal(clean[0].sensor_data["front"], clean[1].sensor_data["front"])
