"""Bridge AVSecTester plugins onto avstack's native hook machinery.

avstack modules run pre/post hooks with a specific calling convention
(``avstack/modules/base.py``)::

    pre-hook:   hook(*args, **kwargs) -> (args, kwargs)   # MUST return the (args, kwargs) pair
    post-hook:  hook(*ret)           -> ret               # re-splatted every iteration

``_apply_post_hooks`` runs ``args = hook(*args)`` in a loop and only unwraps a *final*
length-1 tuple, so a chain-safe post-hook must return ``(value,)``: that survives the next
iteration's ``*args`` re-splat and is unwrapped back to a bare value at the end.

AVSecTester plugins keep a clean, avstack-agnostic contract ::

    apply(payload, ego_state=..., ctx=...) -> payload

ALL knowledge of avstack's calling convention lives here, in the framework layer. Attacks,
defenses, and monitors never import it, so they stay a clean, extractable subtree.

Two adapters, both attachable to any avstack ``BaseModule`` at runtime via
``register_pre_hook`` / ``register_post_hook`` (see :func:`attach` / :func:`attach_monitor`):

- :class:`HookAdapter` wraps an attack/defense plugin (it may modify the payload).
- :class:`MonitorAdapter` observes only, recording per-stage I/O into the run trace.

The ego pose a plugin needs is not carried by a module's own call signature, so it comes
from a per-run :class:`RunContext` the backend updates each tick.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .core.escalation import Stage

# Seam vocabulary lives in ``core`` (see core/seams.py); this module maps seams onto
# avstack's concrete pre/post-hook machinery. Re-exported here for backward compatibility.
from .core.seams import SEAMS, Phase, Seam, resolve_seam
from .monitors.trace import ComponentIO, Trace

__all__ = [  # public surface (incl. seam names re-exported from core.seams)
    "SEAMS",
    "ComponentIO",
    "HookAdapter",
    "MonitorAdapter",
    "Phase",
    "RunContext",
    "Seam",
    "Stage",
    "Trace",
    "attach",
    "attach_monitor",
    "resolve_seam",
]


@dataclass
class RunContext:
    """Per-run, per-tick state shared with hooks.

    The backend calls :meth:`tick` at the top of each step (before invoking the pipeline)
    so adapters can read the current ``frame`` / ``t`` / ``ego_state`` / ``ground_truth``.
    This is how a plugin gets the ego pose that a module's call signature does not carry.
    Monitors append per-stage records into ``trace``; defenses append per-tick
    :class:`~avsectester.core.interfaces.DefenseOutcome` telemetry into ``defense_outcomes``.
    """

    run_id: str = "run"
    frame: int = 0
    t: float = 0.0
    ego_state: Any = None
    ground_truth: Any = None
    seam: str = ""  # the seam currently firing (set by HookAdapter before each apply)
    trace: Trace = field(default=None)  # type: ignore[assignment]
    defense_outcomes: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.trace is None:
            self.trace = Trace(self.run_id)

    def tick(
        self,
        frame: int,
        t: float,
        ego_state: Any = None,
        ground_truth: Any = None,
    ) -> None:
        self.frame = frame
        self.t = t
        self.ego_state = ego_state
        self.ground_truth = ground_truth

    def record(self, stage: Stage | str, component: str, outputs: Any) -> None:
        stage_name = stage.value if isinstance(stage, Stage) else stage
        self.trace.add(ComponentIO(self.frame, stage_name, component, outputs=outputs))


class HookAdapter:
    """Wrap an AVSecTester plugin as an avstack pre/post hook.

    Confines avstack's calling convention to this class. The wrapped ``plugin`` only needs
    ``apply(payload, ego_state=..., ctx=...) -> payload``. A ``PRE`` adapter returns
    ``(args, kwargs)``; a ``POST`` adapter returns ``(value,)`` (chain-safe; avstack's
    final length-1 rule unwraps it to a bare value).
    """

    def __init__(self, plugin: Any, seam: Seam, ctx: RunContext) -> None:
        self.plugin = plugin
        self.seam = seam
        self.ctx = ctx

    def _invoke(self, payload: Any) -> Any:
        self.ctx.seam = self.seam.name  # tell the plugin which seam is firing
        return self.plugin.apply(payload, ego_state=self.ctx.ego_state, ctx=self.ctx)

    def __call__(self, *args: Any, **kwargs: Any):
        if self.seam.phase is Phase.PRE:
            i = self.seam.arg_index
            if i >= len(args):
                raise IndexError(
                    f"seam {self.seam.name!r} expects a payload at arg[{i}] but the call "
                    f"had {len(args)} positional args"
                )
            new = self._invoke(args[i])
            return args[:i] + (new,) + args[i + 1 :], kwargs
        payload = args[0] if len(args) == 1 else args
        return (self._invoke(payload),)


def _default_extract(payload: Any) -> dict[str, Any]:
    """Summarize an arbitrary module payload for the trace (count when it is sized)."""
    try:
        return {"count": len(payload)}
    except TypeError:
        return {}


class MonitorAdapter:
    """Observe-only hook: record a module's I/O into the run trace, pass it through untouched.

    ``extract`` maps the payload to the small dict stored on the trace record (defaults to a
    ``{"count": len(payload)}`` summary). Works at either phase; a ``PRE`` monitor observes
    the input and returns ``(args, kwargs)`` unchanged, a ``POST`` monitor observes the
    output and returns ``(value,)``.
    """

    def __init__(
        self,
        seam: Seam,
        ctx: RunContext,
        extract: Callable[[Any], Any] | None = None,
    ) -> None:
        self.seam = seam
        self.ctx = ctx
        self.extract = extract or _default_extract

    def _observe(self, payload: Any) -> None:
        self.ctx.record(self.seam.stage, self.seam.component, self.extract(payload))

    def __call__(self, *args: Any, **kwargs: Any):
        if self.seam.phase is Phase.PRE:
            i = self.seam.arg_index
            if i < len(args):
                self._observe(args[i])
            return args, kwargs
        payload = args[0] if len(args) == 1 else args
        self._observe(payload)
        return (payload,)


_as_seam = resolve_seam  # backward-compatible alias


def _register(module: Any, phase: Phase, adapter: Any) -> None:
    if phase is Phase.PRE:
        module.register_pre_hook(adapter)
    else:
        module.register_post_hook(adapter)


def attach(module: Any, plugin: Any, seam: Seam | str, ctx: RunContext) -> HookAdapter:
    """Wrap ``plugin`` for ``seam`` and register it on the avstack ``module`` at runtime.

    Registration order is preserved by avstack, so attaching an attack before a defense
    means the defense sees the already-perturbed payload.
    """
    seam = _as_seam(seam)
    adapter = HookAdapter(plugin, seam, ctx)
    _register(module, seam.phase, adapter)
    return adapter


def attach_monitor(
    module: Any,
    seam: Seam | str,
    ctx: RunContext,
    extract: Callable[[Any], Any] | None = None,
) -> MonitorAdapter:
    """Register a :class:`MonitorAdapter` on ``module`` to trace ``seam`` into ``ctx``."""
    seam = _as_seam(seam)
    adapter = MonitorAdapter(seam, ctx, extract)
    _register(module, seam.phase, adapter)
    return adapter
