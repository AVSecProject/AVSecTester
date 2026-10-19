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

from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.simulator import WorldBackend
from avsectester.conditions import ScenarioConditions, SensorCondition
from avsectester.evaluation.robustness import ConditionResult, RobustnessReport, evaluate_robustness
from avsectester.closed_loop.datatypes import Control, Observation
from avsectester.scenarios.source import ScenarioInstance, ScenarioSource
from avsectester.perturbations.image import AugmentationPipeline, Corruption
from avsectester.preparation import Intervention, prepare_intervention
from avsectester.closed_loop.hooks import Hook, Plugin


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
                factory=self.backend_factory,
                match=_Match(),
                provenance={"scene_id": f"stub-{i}"},
            )


class _LiftBlack(Corruption):  # a corruption that reveals the hidden lead -> neutralises the attack
    def apply(self, rgb, rng):
        return np.clip(rgb.astype(int) + 100, 0, 255).astype(np.uint8)


class _Mild(Corruption):  # a corruption that does not cross the detection threshold from black
    def apply(self, rgb, rng):
        return np.clip(rgb.astype(int) + 1, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("use_runtime", [False, True])
def test_harness_scores_each_condition_and_aggregates_scenarios(use_runtime):
    from avsectester.closed_loop.hooks import Hook, Runtime

    def attack_for(match, backend):
        attack = _hide_attack(match, backend)
        if not use_runtime:
            return attack

        def hide(observation, context):
            assert context.case is match
            return attack(observation)

        return Runtime([Hook("observation", hide)])

    n = 3
    r = evaluate_robustness(
        _StubSource(n),
        req=None,
        attack_for=attack_for,
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
    from avsectester.perturbations.image import CORRUPTIONS

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
    from avsectester.closed_loop.datatypes import Trace

    run = Mock(side_effect=[Trace()] * (fail_run - 1) + [RuntimeError("simulation failed")])

    with pytest.raises(RuntimeError, match="simulation failed"):
        evaluate_robustness(
            source, None, _hide_attack, _BrakeIfLeadSeen, 8, conditions=[], run_fn=run
        )
    backend.prepare_clean_attack_pair.assert_called_once()
    backend.close.assert_called_once()


def test_harness_pairs_corruptions_across_different_simulator_frame_ids():
    from avsectester.perturbations.image import GaussianNoise

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


@pytest.mark.parametrize("reuse_artifacts", [False, True])
def test_method_prepares_once_per_case_and_creates_fresh_handlers_across_conditions(reuse_artifacts):
    class HideMethod(Intervention):
        def __init__(self):
            self.prepared, self.handlers = [], []

        def prepare(self, scenario):
            self.prepared.append(scenario)
            return {"camera": scenario.match.camera}

        def hooks(self, artifacts):
            class Handler(Plugin):
                def reset(self, context):
                    self.steps = []

                def transform(self, observation, context):
                    self.steps.append(context.step)
                    return replace(observation, sensor_data={
                        **observation.sensor_data,
                        artifacts["camera"]: np.zeros_like(observation.sensor_data[artifacts["camera"]]),
                    })

            handler = Handler()
            self.handlers.append(handler)
            return [Hook("observation", handler)]

    method = HideMethod()
    attack = method
    if reuse_artifacts:
        attack = prepare_intervention(method, next(_StubSource().scenarios(None)))
    report = evaluate_robustness(
        _StubSource(2), None, attack, _BrakeIfLeadSeen, 8,
        conditions=[AugmentationPipeline([_Mild()])],
    )
    assert len(method.prepared) == (1 if reuse_artifacts else 2)
    assert len(method.handlers) == 4 and len({id(handler) for handler in method.handlers}) == 4
    assert all(handler.steps == list(range(8)) for handler in method.handlers)
    assert all(row.status == "success" for row in report.rows)


def test_robustness_preserves_world_conditions_and_independent_sensor_state():
    configured, histories, traces, events = [], [], [], []

    class Backend(_ApproachBackend):
        def configure_conditions(self, world):
            configured.append(dict(world))
            self.weather = world["weather"]
            events.append("conditions")

        def prepare_clean_attack_pair(self):
            events.append("prepare")

        def reset(self):
            events.append("reset")
            return super().reset()

    class Condition(Plugin):
        def reset(self, context):
            self.steps = []
            histories.append(self.steps)

        def transform(self, observation, context):
            self.steps.append(context.step)
            return observation

    class Source(_StubSource):
        def scenarios(self, req, limit=None):
            for scenario in super().scenarios(req, limit):
                yield replace(scenario, conditions=ScenarioConditions(
                    world={"weather": "cloudy"}, sensors=(SensorCondition(Condition),), seed=21,
                ))

    def capture(*args, **kwargs):
        from avsectester.backend import run

        trace = run(*args, **kwargs)
        traces.append(trace)
        return trace

    def callback(match, backend):
        assert backend.weather == "cloudy"
        events.append("callback")
        return _hide_attack(match, backend)

    evaluate_robustness(
        Source(backend_factory=Backend), None, callback, _BrakeIfLeadSeen, 3,
        conditions=[AugmentationPipeline([_Mild()], seed=7)], run_fn=capture,
    )
    # Each condition constructs one world. Its existing condition layer is not applied twice.
    assert configured == [{"weather": "cloudy"}] * 2
    assert events == ["conditions", "callback", "prepare", "reset", "reset"] * 2
    assert len(histories) == 4 and all(history == [0, 1, 2] for history in histories)
    assert all(trace.provenance["conditions"]["world"] == {"weather": "cloudy"} for trace in traces)
    assert len(traces[0].provenance["conditions"]["sensors"]) == 1
    assert len(traces[2].provenance["conditions"]["sensors"]) == 2
    assert traces[2].provenance["conditions"]["sensors"][-1]["seed"] == 7


def test_callback_failure_is_closed_once_before_pair_preparation():
    backend = _ApproachBackend()
    backend.prepare_clean_attack_pair = Mock()
    backend.close = Mock()
    callback = Mock(side_effect=RuntimeError("cannot construct attack"))
    with pytest.raises(RuntimeError, match="cannot construct attack"):
        evaluate_robustness(
            _StubSource(backend_factory=lambda: backend), None, callback,
            _BrakeIfLeadSeen, 3, conditions=[],
        )
    callback.assert_called_once()
    backend.prepare_clean_attack_pair.assert_not_called()
    backend.close.assert_called_once()


def test_skipped_callback_cleanup_failure_is_not_reported_as_a_skipped_experiment():
    backend = _ApproachBackend()
    backend.close = Mock(side_effect=RuntimeError("cleanup failed"))
    backend.prepare_clean_attack_pair = Mock()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        evaluate_robustness(
            _StubSource(backend_factory=lambda: backend), None, lambda *args: None,
            _BrakeIfLeadSeen, 3, conditions=[],
        )
    backend.close.assert_called_once()
    backend.prepare_clean_attack_pair.assert_not_called()


def test_robustness_rejects_defense_prepared_definition():
    from avsectester.attacks.adaptive import SpeedFeedbackAttack

    scenario = next(_StubSource().scenarios(None))
    definition = prepare_intervention(SpeedFeedbackAttack(5), scenario, kind="defense")
    with pytest.raises(ValueError, match="requires an attack"):
        evaluate_robustness(_StubSource(), None, definition, _BrakeIfLeadSeen, 3, conditions=[])
