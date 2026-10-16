"""Minimal online attack protocol with one state owner and data-only callbacks.

The candidate-selection and preparation APIs are introduced in later phases. Explicit
world actions still use existing deployment hooks; StageAction replaces a pipeline value.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Literal

from .plane import Observation, WorldSnapshot
from .runtime import Hook, StageContext, UnsupportedStageError


@dataclass(frozen=True)
class ResultSubscription:
    """Declare a result read by an attack; component results name their exact stage."""

    kind: Literal["observation", "physical_state", "component"]
    stage: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"observation", "physical_state", "component"}:
            raise ValueError(f"Unknown attack result kind: {self.kind!r}")
        if self.kind == "component":
            if not isinstance(self.stage, str) or not self.stage.endswith(".post"):
                raise ValueError("A component result subscription needs a component .post stage")
        elif self.stage is not None:
            raise ValueError("Only component subscriptions specify a stage")


@dataclass(frozen=True)
class AttackContext:
    """Independent snapshots of subscribed results, with a read-only physical clock.

    This context deliberately carries no backend, stack, native SDK or time-advancing
    service. Nested data can be modified locally without changing another callback.
    """

    step: int
    stage: str
    sim_time: float
    frame: int
    observation: Observation | None = None
    physical_state: WorldSnapshot | None = None
    component_output: Any = None


@dataclass(frozen=True)
class StageAction:
    """Replace a value at a current or forthcoming pipeline stage in this decision."""

    stage: str
    value: Any

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("StageAction.stage must be a nonempty string")


class AttackRuntime:
    """One online attack instance per run; all callbacks share this object's state."""

    reads: Sequence[ResultSubscription] = ()

    def start(self, context: AttackContext) -> Sequence[StageAction]:
        return ()

    def step(self, context: AttackContext) -> Sequence[StageAction]:
        return ()

    def on_component(self, context: AttackContext) -> Sequence[StageAction]:
        return ()

    def stop(self, context: AttackContext) -> None:
        pass


_STAGE_ORDER = (
    "sensors.post", "localization.pre", "localization.post", "observation",
    "perception.pre", "perception.post", "tracking.pre", "tracking.post",
    "planning.pre", "planning.post", "control.pre", "control.post",
    "policy.pre", "policy.post", "command",
)
_COMPONENT_STAGES = frozenset(
    stage for stage in _STAGE_ORDER
    if stage.endswith(".post") and stage not in {"sensors.post", "localization.post"}
)


class AttackDispatcher:
    """Internal callback routing; attack histories stay on AttackRuntime instances.

    Validate before reset, append ``hooks`` to the run's HookSet, then call start, one
    step per decision before Vehicle I/O, and stop in a finally block. Data actions are
    consumed by those hooks in pipeline order. ``requests`` and ``receipts`` distinguish
    requests from pipeline replacements; ``actions`` retains the legacy request view.
    """

    def __init__(
        self, attacks: Iterable[AttackRuntime] = (), *,
        observation_copy: Callable[[Observation], Observation] = deepcopy,
    ) -> None:
        self.attacks = tuple(attacks)
        self._copy_observation = observation_copy
        if len({id(attack) for attack in self.attacks}) != len(self.attacks):
            raise ValueError("An AttackRuntime instance must occur only once in a run")
        self._attack_labels = {
            id(attack): f"{type(attack).__qualname__}[{index}]"
            for index, attack in enumerate(self.attacks)
        }
        self._reads: dict[int, tuple[ResultSubscription, ...]] = {}
        for attack in self.attacks:
            for method in ("start", "step", "on_component", "stop"):
                if not callable(getattr(attack, method, None)):
                    raise TypeError(f"AttackRuntime must provide {method}(context)")
            reads = tuple(attack.reads)
            if any(not isinstance(item, ResultSubscription) for item in reads):
                raise TypeError("AttackRuntime.reads must contain ResultSubscription instances")
            self._reads[id(attack)] = reads
        self._available: frozenset[str] | None = None
        self._started: list[AttackRuntime] = []
        self._context: AttackContext | None = None
        self._pending: dict[str, tuple[int, StageAction, str]] = {}
        self._last_step = -1
        self._has_started = False
        self._stopped = False
        self._next_action_id = 0
        self.actions: list[tuple[int, str, StageAction]] = []
        self.requests: list[dict[str, Any]] = []
        self.receipts: list[dict[str, Any]] = []

    def validate(self, available_stages: Iterable[str]) -> None:
        available = frozenset(available_stages)
        for reads in self._reads.values():
            for item in reads:
                if item.kind == "component" and (
                    item.stage not in available or item.stage not in _COMPONENT_STAGES
                ):
                    raise UnsupportedStageError(f"Unavailable component result: {item.stage!r}")
        self._available = available

    @property
    def hooks(self) -> tuple[Hook, ...]:
        if self._available is None:
            raise RuntimeError("Validate attack subscriptions before binding hooks")
        if not self.attacks:
            return ()
        return tuple(Hook(stage, self._at_stage) for stage in _STAGE_ORDER if stage in self._available)

    def _for_attack(self, attack: AttackRuntime, context: AttackContext) -> AttackContext:
        reads = self._reads[id(attack)]
        kinds = {item.kind for item in reads}
        component = any(item.kind == "component" and item.stage == context.stage for item in reads)
        return replace(
            context,
            observation=(
                self._copy_observation(context.observation)
                if "observation" in kinds and context.observation is not None else None
            ),
            physical_state=deepcopy(context.physical_state) if "physical_state" in kinds else None,
            component_output=deepcopy(context.component_output) if component else None,
        )

    def _enqueue(
        self, actions: Sequence[StageAction], context: AttackContext,
        attack: AttackRuntime, callback: str,
    ) -> tuple[StageAction, ...]:
        if actions is None:
            raise TypeError("Attack callbacks must return a sequence of StageAction, not None")
        result = tuple(actions)
        issuer = f"{self._attack_labels[id(attack)]}.{callback}"
        current_rank = _STAGE_ORDER.index(context.stage) if context.stage in _STAGE_ORDER else -1
        for action in result:
            if not isinstance(action, StageAction):
                raise TypeError("Attack callbacks must return StageAction instances")
            if action.stage not in _STAGE_ORDER or action.stage not in (self._available or ()):
                raise UnsupportedStageError(f"Unsupported attack action stage: {action.stage!r}")
            if _STAGE_ORDER.index(action.stage) < current_rank:
                raise ValueError(f"Attack action cannot target an elapsed stage: {action.stage!r}")
        for action in result:
            pending = self._pending.get(action.stage)
            if pending is not None:
                raise ValueError(
                    f"Conflicting pending replacements for stage {action.stage!r}: "
                    f"{pending[2]} and {issuer}"
                )
            copied = self._copy_action(action)
            action_id = self._next_action_id
            self._next_action_id += 1
            self._pending[copied.stage] = (action_id, copied, issuer)
            self.requests.append({
                "action_id": action_id, "step": context.step, "stage": context.stage,
                "sim_time": context.sim_time, "frame": context.frame,
                "target_stage": copied.stage, "action": self._copy_action(copied),
            })
            self.actions.append((context.step, context.stage, self._copy_action(copied)))
        return result

    def drain_records(self) -> dict[str, list[dict[str, Any]]]:
        """Transfer request/receipt records while keeping pending execution and unique IDs.

        The runner drains after recording initialization, decisions and cleanup. Direct
        callers retain accumulating lists unless they explicitly request this transfer.
        """
        result = {"requests": list(self.requests), "receipts": list(self.receipts)}
        self.requests.clear()
        self.receipts.clear()
        self.actions.clear()
        return result

    def _copy_action(self, action: StageAction) -> StageAction:
        value = action.value
        return StageAction(
            action.stage,
            self._copy_observation(value) if isinstance(value, Observation) else deepcopy(value),
        )

    def start(self, context: AttackContext) -> tuple[StageAction, ...]:
        if self._available is None:
            raise RuntimeError("Validate attack subscriptions before starting")
        if self._has_started or self._stopped:
            raise RuntimeError("Attack dispatcher can only start once")
        self._has_started = True
        self._context = context
        actions: list[StageAction] = []
        for attack in self.attacks:
            # Include partially started attacks in cleanup if their callback fails.
            self._started.append(attack)
            actions.extend(self._enqueue(
                attack.start(self._for_attack(attack, context)), context, attack, "start",
            ))
        return tuple(actions)

    def step(self, context: AttackContext) -> tuple[StageAction, ...]:
        if self._context is None or self._stopped:
            raise RuntimeError("Attack dispatcher is not active")
        if context.step != self._last_step + 1:
            raise ValueError("AttackRuntime.step must run exactly once per consecutive decision")
        if self._last_step >= 0 and self._pending:
            raise RuntimeError(f"Attack actions were not consumed: {sorted(self._pending)}")
        self._last_step = context.step
        self._context = context
        actions: list[StageAction] = []
        for attack in self.attacks:
            actions.extend(self._enqueue(
                attack.step(self._for_attack(attack, context)), context, attack, "step",
            ))
        return tuple(actions)

    def finish_decision(self) -> None:
        """Reject actions whose declared stage was skipped, including on the last step."""
        if self._pending:
            raise RuntimeError(f"Attack actions were not consumed: {sorted(self._pending)}")

    def _consume(self, value: Any, context: StageContext) -> Any:
        pending = self._pending.get(context.stage)
        if pending is not None:
            action_id, action, _issuer = pending
            value = self._copy_action(action).value
            # Retain a pending action if its value cannot be copied. Cleanup then records
            # not_applied rather than losing it or recording a successful replacement.
            self.receipts.append({
                "action_id": action_id, "step": context.step, "stage": context.stage,
                "sim_time": context.sim_time, "frame": context.ground_truth.frame,
                "target_stage": action.stage, "outcome": "applied", "reason": None,
            })
        self._pending.pop(context.stage, None)
        return value

    def _at_stage(self, value: Any, context: StageContext) -> Any:
        value = self._consume(value, context)
        if self._context is None or self._stopped:
            return value
        for attack in self.attacks:
            if not any(
                read.kind == "component" and read.stage == context.stage
                for read in self._reads[id(attack)]
            ):
                continue
            data = replace(
                self._context, stage=context.stage, sim_time=context.sim_time,
                frame=context.ground_truth.frame, physical_state=context.ground_truth,
                component_output=value,
            )
            self._enqueue(
                attack.on_component(self._for_attack(attack, data)), data,
                attack, f"on_component({context.stage})",
            )
            value = self._consume(value, context)
        return value

    def stop(self, context: AttackContext) -> None:
        if self._stopped:
            return
        self._stopped = True
        # Stop is also called for zero decisions and partial failures. An unvisited stage
        # never applied its action, regardless of whether the run ended successfully.
        for action_id, action, _issuer in sorted(self._pending.values(), key=lambda item: item[0]):
            self.receipts.append({
                "action_id": action_id, "step": context.step, "stage": context.stage,
                "sim_time": context.sim_time, "frame": context.frame,
                "target_stage": action.stage, "outcome": "not_applied",
                "reason": "run_ended_before_target_stage",
            })
        self._pending.clear()
        failure: BaseException | None = None
        for attack in reversed(self._started):
            try:
                attack.stop(self._for_attack(attack, context))
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(f"Attack cleanup also failed: {error!r}")
        if failure is not None:
            raise failure
