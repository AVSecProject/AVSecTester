"""Verify the hook adapter against avstack's REAL hook machinery.

The pure-contract tests exercise :class:`HookAdapter` / :class:`MonitorAdapter` directly
(core-only, no avstack). The integration tests build genuine ``@apply_hooks``-decorated
``avstack.BaseModule`` subclasses and assert the adapters compose correctly through
``_apply_pre_hooks`` / ``_apply_post_hooks`` (the fiddly re-splat + length-1-unwrap rules).
Those are skipped when the avstack stack is not installed.
"""
from __future__ import annotations

import pytest
from avsectester.core.escalation import Stage
from avsectester.hooks import (
    HookAdapter,
    MonitorAdapter,
    Phase,
    RunContext,
    Seam,
    attach,
    attach_monitor,
)


class AppendPlugin:
    """Clean AVSecTester-style plugin: append a tag to a list payload."""

    def __init__(self, tag: str = "phantom") -> None:
        self.tag = tag

    def apply(self, payload, ego_state=None, ctx=None, **kwargs):
        return list(payload) + [self.tag]


# ---------------------------------------------------------------------------
# Pure-contract tests (no avstack): the adapter honors avstack's return shapes.
# ---------------------------------------------------------------------------
def test_pre_adapter_returns_args_kwargs_and_splices_payload():
    adapter = HookAdapter(AppendPlugin("x"), Seam("s", Phase.PRE, Stage.SENSOR, "c"), RunContext())
    ret = adapter(["a", "b"], frame=1)
    assert ret == ((["a", "b", "x"],), {"frame": 1})  # (args_tuple, kwargs_dict)


def test_post_adapter_returns_one_tuple():
    adapter = HookAdapter(AppendPlugin("z"), Seam("s", Phase.POST, Stage.TRACKING, "c"), RunContext())
    assert adapter(["t"]) == (["t", "z"],)  # chain-safe 1-tuple


def test_pre_seam_missing_payload_raises():
    adapter = HookAdapter(AppendPlugin(), Seam("s", Phase.PRE, Stage.SENSOR, "c", arg_index=2), RunContext())
    with pytest.raises(IndexError):
        adapter("only-one-arg")


def test_plugin_reads_ctx_updated_each_tick():
    ctx = RunContext()
    seen: list = []

    class Spy:
        def apply(self, payload, ego_state=None, ctx=None, **kwargs):
            seen.append((ctx.frame, ego_state))
            return payload

    adapter = HookAdapter(Spy(), Seam("s", Phase.PRE, Stage.SENSOR, "c"), ctx)
    ctx.tick(1, 0.05, ego_state="EGO1")
    adapter(["a"])
    ctx.tick(2, 0.10, ego_state="EGO2")
    adapter(["a"])
    assert seen == [(1, "EGO1"), (2, "EGO2")]


def test_monitor_records_stage_and_passes_through():
    ctx = RunContext()
    adapter = MonitorAdapter(Seam("s", Phase.POST, Stage.TRACKING, "tracker"), ctx)
    out = adapter(["t1", "t2"])
    assert out == (["t1", "t2"],)  # unchanged, wrapped for the chain
    recs = [r for r in ctx.trace.records if r.stage == Stage.TRACKING.value]
    assert len(recs) == 1
    assert recs[0].outputs == {"count": 2}
    assert recs[0].component == "tracker"


# ---------------------------------------------------------------------------
# Integration tests against the real @apply_hooks decorator.
# ---------------------------------------------------------------------------
def _dummy_modules():
    """Build genuine avstack BaseModule subclasses (skips if avstack is absent)."""
    pytest.importorskip("avstack")
    from avstack.modules.base import BaseModule
    from avstack.utils.decorators import apply_hooks

    class DummyDetector(BaseModule):
        def __init__(self, **kw):
            super().__init__(name="dummy_detector", **kw)

        @apply_hooks
        def __call__(self, data, frame=0):
            # "detect": echo the (possibly perturbed) input plus a marker that proves the
            # frame kwarg flowed through the pre-hook untouched.
            return list(data) + [f"det@{frame}"]

    class DummyTracker(BaseModule):
        def __init__(self, **kw):
            super().__init__(name="dummy_tracker", **kw)

        @apply_hooks
        def __call__(self, dets):
            return list(dets)

    return DummyDetector, DummyTracker


def test_pre_hook_perturbs_input_seen_by_the_module():
    DummyDetector, _ = _dummy_modules()
    ctx = RunContext()
    det = DummyDetector()
    attach(det, AppendPlugin("phantom"), "raw_lidar", ctx)  # raw_lidar is PRE arg0
    out = det(["a", "b"], frame=3)
    # phantom injected into the INPUT before detection; frame kwarg preserved end to end
    assert out == ["a", "b", "phantom", "det@3"]


def test_post_hooks_replace_output_and_run_in_registration_order():
    _, DummyTracker = _dummy_modules()
    ctx = RunContext()
    trk = DummyTracker()
    attach(trk, AppendPlugin("a1"), "tracking_out", ctx)  # POST
    attach(trk, AppendPlugin("a2"), "tracking_out", ctx)  # POST, second
    out = trk(["t"])
    assert out == ["t", "a1", "a2"]  # order preserved
    assert not isinstance(out, tuple)  # avstack unwrapped the final 1-tuple to a bare value


def test_monitor_traces_real_module_output_without_changing_it():
    _, DummyTracker = _dummy_modules()
    ctx = RunContext()
    trk = DummyTracker()
    attach_monitor(trk, "tracking_out", ctx)
    ctx.tick(5, 0.25)
    out = trk(["t1", "t2", "t3"])
    assert out == ["t1", "t2", "t3"]  # untouched by the monitor
    recs = ctx.trace.records
    assert len(recs) == 1
    assert (recs[0].frame, recs[0].stage, recs[0].outputs) == (5, Stage.TRACKING.value, {"count": 3})


def test_attack_then_defense_ordering_through_real_pipeline():
    DummyDetector, _ = _dummy_modules()
    ctx = RunContext()
    det = DummyDetector()

    class DropTag:
        """Defense: remove a specific injected tag from the payload."""

        def __init__(self, tag):
            self.tag = tag

        def apply(self, payload, ego_state=None, ctx=None, **kwargs):
            return [x for x in payload if x != self.tag]

    attach(det, AppendPlugin("phantom"), "raw_lidar", ctx)  # inject first
    attach(det, DropTag("phantom"), "raw_lidar", ctx)  # then sanitize
    out = det(["a"], frame=0)
    assert out == ["a", "det@0"]  # defense removed the injected phantom before detection
