"""The minimal core runtime: Environment/System/Attack + run/run_experiment (no avstack)."""

from __future__ import annotations

import pytest
from avsectester.core import (
    Attack,
    Environment,
    Frame,
    Outcome,
    Seam,
    System,
    run,
    run_experiment,
)
from avsectester.core.metric import Metric


class ToyEnv(Environment):
    def __init__(self, n: int = 5) -> None:
        self.n, self.i = n, 0

    def reset(self) -> Frame:
        self.i = 0
        return Frame(index=0)

    def step(self, control=None):
        self.i += 1
        return Frame(index=self.i, meta={"control": control}), self.i >= self.n


class ToySystem(System):
    seams = (Seam.PERCEPTION_OUT,)

    def process(self, frame: Frame) -> Outcome:
        payload = self.fire(Seam.PERCEPTION_OUT, ["x"], frame)
        return Outcome(control=frame.index, record={"n": len(payload), "frame": frame.index})


class AppendAttack(Attack):
    seams = (Seam.PERCEPTION_OUT,)

    def apply(self, payload, ctx):
        return [*payload, "atk"]


def test_run_collects_one_record_per_tick():
    tr = run(ToyEnv(5), ToySystem())
    assert len(tr) == 5 and tr.records[0]["n"] == 1


def test_attack_fires_at_its_seam():
    sys = ToySystem()
    sys.attach(AppendAttack())
    tr = run(ToyEnv(3), sys)
    assert all(r["n"] == 2 for r in tr.records)  # attack appended at perception_out


def test_ctx_carries_frame_and_seam():
    seen = {}

    class Spy(Attack):
        seams = (Seam.PERCEPTION_OUT,)

        def apply(self, payload, ctx):
            seen["seam"], seen["frame"] = ctx.seam, ctx.frame.index
            return payload

    sys = ToySystem()
    sys.attach(Spy())
    run(ToyEnv(2), sys)
    assert seen["seam"] == Seam.PERCEPTION_OUT


def test_attach_rejects_unexposed_seam():
    class Bad(Attack):
        seams = (Seam.CONTROL_OUT,)

        def apply(self, payload, ctx):
            return payload

    with pytest.raises(ValueError, match="control_out"):
        ToySystem().attach(Bad())


def test_run_experiment_clean_vs_attacked():
    class CountMetric(Metric):
        def compute(self, clean, attacked):
            c = sum(r["n"] for r in clean.records)
            a = sum(r["n"] for r in attacked.records)
            return {"impacted": a > c}

    res = run_experiment(lambda: ToyEnv(4), ToySystem, CountMetric(), attack=AppendAttack())
    assert res.metrics["impacted"] is True
    assert len(res.clean) == 4 and len(res.attacked) == 4
