"""The online phase: one AttackRuntime per run and the attack effects it returns.

Effects are classified by where they act in the closed loop:

* ``WorldAttackEffect`` changes the simulated world. The simulator applies it at setup or
  before a world step, and it is visible from the next capture.
* ``SensorAttackEffect`` changes one captured sensor's data before Vehicle I/O, using the
  simulator's sensor view. It applies to the current decision.
* ``AVStackAttackEffect`` changes the value at a stage of the driving system (the ``AVStack``,
  including its Vehicle I/O and command) in the current decision.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from avsectester.closed_loop.datatypes import Observation, WorldSnapshot
from avsectester.closed_loop.hooks import Hook, StageContext, UnsupportedStageError

if TYPE_CHECKING:
    from avsectester.closed_loop.simulator import SensorView, SimulatorRuntime


@dataclass(frozen=True)
class ResultSubscription:
    """Declare a closed-loop result read by an attack runtime."""

    kind: Literal["observation", "physical_state"]

    def __post_init__(self) -> None:
        if self.kind not in {"observation", "physical_state"}:
            raise ValueError(f"Unknown attack result kind: {self.kind!r}")


@dataclass(frozen=True)
class OnlineContext:
    """Independent snapshots of subscribed results, with a read-only physical clock.

    The framework creates it for every online-phase callback. It deliberately carries no
    simulator, driving system, native SDK or time-advancing service. Nested data can be
    modified locally without changing another callback.
    """

    step: int
    stage: str
    sim_time: float
    frame: int
    observation: Observation | None = None
    physical_state: WorldSnapshot | None = None
    _output_recorder: Callable[[OnlineContext, str, Any], None] | None = field(
        default=None, repr=False, compare=False,
    )

    def record_output(self, name: str, value: Any) -> None:
        """Save a named copy of ``value``, such as an attack decision, in this run's log."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Output name must be a nonempty string")
        if self._output_recorder is None:
            raise RuntimeError("Outputs can only be recorded during a run")
        self._output_recorder(self, name, value)


class AttackEffect:
    """Base of everything the online phase applies. Subclasses state where it acts."""


class WorldAttackEffect(AttackEffect, ABC):
    """Change the simulated world. The simulator applies it; it is visible from the next capture.

    ``setup_ticks`` is the number of ``simulator.setup_tick()`` calls that ``apply`` makes when
    the effect is returned by ``AttackRuntime.setup``. Ticks are unavailable after setup.
    Register the removal of created world objects on ``resources``; it runs when the run ends.
    """

    setup_ticks: int = 0

    @abstractmethod
    def apply(self, simulator_runtime: SimulatorRuntime, resources: ExitStack) -> None:
        """Change the world through the simulator's native access."""


class SensorAttackEffect(AttackEffect, ABC):
    """Change one captured sensor's data before Vehicle I/O, in the current decision.

    ``sensor`` names the simulator sensor. ``apply`` receives the current data, which earlier
    effects on the same sensor may have changed, and the capture's view, whose ``data`` is
    the unchanged capture. It returns the changed data in the same representation.
    """

    sensor: str

    @abstractmethod
    def apply(self, data: Any, view: SensorView) -> Any:
        """Return the changed sensor data."""

    def render_metadata(self) -> Mapping[str, Any]:
        """Data-only evidence from the most recent application, included in its receipt."""
        return {}


@dataclass(frozen=True)
class AVStackAttackEffect(AttackEffect):
    """Change the value at a driving-system stage in the current decision.

    ``apply(value, context)`` returns the value used downstream. It may be a bound method of
    the runtime, so a decision that depends on a component output keeps its state there.
    """

    stage: str
    apply: Callable[[Any, OnlineContext], Any]

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("AVStackAttackEffect.stage must be a nonempty string")
        if not callable(self.apply):
            raise TypeError("AVStackAttackEffect.apply must be callable")


class AttackRuntime:
    """One online-phase instance per run; every callback shares this object's state.

    ``setup`` runs after the simulator reset and before the first capture, and returns only
    world attack effects. ``step`` runs once per decision before Vehicle I/O. ``stop`` runs
    at the end of the run, also after failures and with zero decisions. ``setup_ticks`` is the
    simulator tick budget used by the world effects that ``setup`` returns.
    """

    subscriptions: Sequence[ResultSubscription] = ()
    setup_ticks: int = 0

    def setup(self, context: OnlineContext) -> Sequence[WorldAttackEffect]:
        return ()

    def step(self, context: OnlineContext) -> Sequence[AttackEffect]:
        return ()

    def stop(self, context: OnlineContext) -> None:
        pass


# Stages an AVStack attack effect can target, in their order within one decision.
_STAGE_ORDER = (
    "sensors.post", "localization.pre", "localization.post", "observation",
    "perception.pre", "perception.post", "tracking.pre", "tracking.post",
    "planning.pre", "planning.post", "control.pre", "control.post",
    "policy.pre", "policy.post", "command",
)


def _category(effect: AttackEffect) -> str:
    for kind, name in ((WorldAttackEffect, "world"), (SensorAttackEffect, "sensor"),
                       (AVStackAttackEffect, "avstack")):
        if isinstance(effect, kind):
            return name
    raise TypeError("Attack callbacks must return WorldAttackEffect, SensorAttackEffect or "
                    "AVStackAttackEffect instances")


class _SetupHook:
    """The simulator.setup participant that runs AttackRuntime.setup inside the simulator reset."""

    def __init__(self, dispatcher: AttackDispatcher) -> None:
        self.dispatcher = dispatcher
        self.setup_ticks = dispatcher.setup_ticks

    def __call__(self, value: None, context: StageContext) -> None:
        truth = context.physical_state
        simulator_runtime = context.simulator_runtime
        self.dispatcher.setup(
            OnlineContext(0, "setup", truth.t, truth.frame, physical_state=truth),
            simulator_runtime,
        )


class AttackDispatcher:
    """Internal: apply the effects of the run's attack runtimes at their points of action.

    Validate before reset, then append ``hooks`` to the run's HookSet. ``setup`` runs inside
    the reset through the simulator.setup hook. ``step`` runs
    once per decision before ``apply_sensor_effects`` and Vehicle I/O; ``stop`` runs in a
    finally block. Every effect gets a request record and a receipt.
    """

    def __init__(
        self, attacks: Iterable[AttackRuntime] = (), *,
        observation_copy: Callable[[Observation], Observation] = deepcopy,
        output_recorder: Callable[[OnlineContext, str, Any], None] | None = None,
    ) -> None:
        self.attacks = tuple(attacks)
        self._copy_observation = observation_copy
        self._output_recorder = output_recorder
        if len({id(attack) for attack in self.attacks}) != len(self.attacks):
            raise ValueError("An AttackRuntime instance must occur only once in a run")
        self._labels = {
            id(attack): f"{type(attack).__qualname__}[{index}]"
            for index, attack in enumerate(self.attacks)
        }
        self._subscriptions: dict[int, frozenset[str]] = {}
        for attack in self.attacks:
            for method in ("setup", "step", "stop"):
                if not callable(getattr(attack, method, None)):
                    raise TypeError(f"AttackRuntime must provide {method}(context)")
            subscriptions = tuple(attack.subscriptions)
            if any(not isinstance(item, ResultSubscription) for item in subscriptions):
                raise TypeError(
                    "AttackRuntime.subscriptions must contain ResultSubscription instances")
            ticks = getattr(attack, "setup_ticks", 0)
            if not isinstance(ticks, int) or isinstance(ticks, bool) or ticks < 0:
                raise ValueError("AttackRuntime.setup_ticks must be a nonnegative integer")
            self._subscriptions[id(attack)] = frozenset(item.kind for item in subscriptions)
        self._available: frozenset[str] | None = None
        self._started: list[AttackRuntime] = []
        self._resources = ExitStack()
        self._last_context: OnlineContext | None = None
        self._step_contexts: dict[int, OnlineContext] = {}
        self._world: list[tuple[int, WorldAttackEffect, AttackRuntime]] = []
        self._sensor: list[tuple[int, SensorAttackEffect, AttackRuntime]] = []
        self._avstack: list[tuple[int, AVStackAttackEffect, AttackRuntime]] = []
        self._last_step = -1
        self._has_setup = False
        self._stopped = False
        self._next_id = 0
        self.requests: list[dict[str, Any]] = []
        self.receipts: list[dict[str, Any]] = []

    @property
    def setup_ticks(self) -> int:
        return sum(attack.setup_ticks for attack in self.attacks)

    @property
    def started(self) -> bool:
        return bool(self._started) and not self._stopped

    def validate(self, available_stages: Iterable[str]) -> None:
        self._available = frozenset(available_stages)

    @property
    def hooks(self) -> tuple[Hook, ...]:
        if self._available is None:
            raise RuntimeError("Validate attack runtimes before binding hooks")
        if not self.attacks:
            return ()
        hooks = []
        if "simulator.setup" in self._available:
            hooks.append(Hook("simulator.setup", _SetupHook(self)))
        hooks.extend(Hook(stage, self._at_stage) for stage in _STAGE_ORDER if stage in self._available)
        if "simulator.step.pre" in self._available:
            hooks.append(Hook("simulator.step.pre", self._at_world_step))
        return tuple(hooks)

    def _for_attack(self, attack: AttackRuntime, context: OnlineContext) -> OnlineContext:
        kinds = self._subscriptions[id(attack)]
        return replace(
            context,
            _output_recorder=self._output_recorder,
            observation=(
                self._copy_observation(context.observation)
                if "observation" in kinds and context.observation is not None else None
            ),
            physical_state=deepcopy(context.physical_state) if "physical_state" in kinds else None,
        )

    def _request(self, effect: AttackEffect, context: OnlineContext, attack: AttackRuntime,
                 callback: str) -> int:
        effect_id = self._next_id
        self._next_id += 1
        self.requests.append({
            "effect_id": effect_id, "step": context.step, "stage": context.stage,
            "sim_time": context.sim_time, "frame": context.frame,
            "issuer": f"{self._labels[id(attack)]}.{callback}",
            "category": _category(effect), "target": _target(effect),
            "effect": type(effect).__qualname__,
        })
        return effect_id

    def _receipt(self, effect_id: int, effect: AttackEffect, step: int, stage: str,
                 sim_time: float, frame: int, outcome: str = "applied",
                 reason: str | None = None) -> None:
        self.receipts.append({
            "effect_id": effect_id, "step": step, "stage": stage, "sim_time": sim_time,
            "frame": frame, "category": _category(effect), "target": _target(effect),
            "outcome": outcome, "reason": reason,
        })

    def drain_records(self) -> dict[str, list[dict[str, Any]]]:
        """Transfer request/receipt records while keeping pending effects and unique IDs.

        The runner drains after recording initialization, decisions and cleanup. Direct
        callers retain accumulating lists unless they explicitly request this transfer.
        """
        result = {"requests": list(self.requests), "receipts": list(self.receipts)}
        self.requests.clear()
        self.receipts.clear()
        return result

    def setup(self, context: OnlineContext, simulator_runtime: SimulatorRuntime) -> None:
        """Run every ``setup`` and apply its world effects before the first capture."""
        if self._available is None:
            raise RuntimeError("Validate attack runtimes before setup")
        if self._has_setup or self._stopped:
            raise RuntimeError("Attack runtimes can only be set up once")
        self._has_setup = True
        self._last_context = context
        for attack in self.attacks:
            # Include partially set-up attacks in cleanup if their callback fails.
            self._started.append(attack)
            effects = attack.setup(self._for_attack(attack, context))
            if effects is None:
                raise TypeError("AttackRuntime.setup must return a sequence of effects, not None")
            for effect in tuple(effects):
                if not isinstance(effect, WorldAttackEffect):
                    raise TypeError("AttackRuntime.setup returns only WorldAttackEffect instances")
                effect_id = self._request(effect, context, attack, "setup")
                effect.apply(simulator_runtime, self._resources)
                truth = simulator_runtime.state()
                self._receipt(effect_id, effect, context.step, "simulator.setup", truth.t, truth.frame)

    def step(self, context: OnlineContext) -> None:
        if not self._has_setup or self._stopped:
            raise RuntimeError("Attack runtimes are not active")
        if context.step != self._last_step + 1:
            raise ValueError("AttackRuntime.step must run exactly once per consecutive decision")
        pending = [*self._world, *self._sensor, *self._avstack]
        if pending:
            raise RuntimeError(f"Attack effects were not applied: {sorted(i for i, *_ in pending)}")
        self._last_step = context.step
        self._last_context = context
        for attack in self.attacks:
            attack_context = self._for_attack(attack, context)
            effects = attack.step(attack_context)
            if effects is None:
                raise TypeError("AttackRuntime.step must return a sequence of effects, not None")
            effects = tuple(effects)
            for effect in effects:
                _category(effect)  # reject other return values before any request
                if (isinstance(effect, WorldAttackEffect)
                        and "simulator.step.pre" not in (self._available or ())):
                    raise UnsupportedStageError("This simulator cannot apply world attack effects")
                if isinstance(effect, AVStackAttackEffect) and (
                    effect.stage not in _STAGE_ORDER or effect.stage not in (self._available or ())
                ):
                    raise UnsupportedStageError(f"Unsupported attack effect stage: {effect.stage!r}")
            for effect in effects:
                effect_id = self._request(effect, context, attack, "step")
                if isinstance(effect, WorldAttackEffect):
                    self._world.append((effect_id, effect, attack))
                elif isinstance(effect, SensorAttackEffect):
                    self._sensor.append((effect_id, effect, attack))
                elif isinstance(effect, AVStackAttackEffect):
                    self._avstack.append((effect_id, effect, attack))
                    self._step_contexts[id(attack)] = attack_context

    def record_end(self, context: OnlineContext) -> None:
        """Remember the post-step clock for a later stop context."""
        self._last_context = context

    def apply_sensor_effects(self, observation: Observation, simulator_runtime: Any) -> Observation:
        """Apply this decision's sensor effects to the captured observation, in issue order."""
        if not self._sensor:
            return observation
        from avsectester.closed_loop.datatypes import write_camera_rgb

        views: dict[str, SensorView] = {}
        data: dict[str, Any] = {}
        pending, self._sensor = self._sensor, []
        context = self._last_context
        assert context is not None
        for effect_id, effect, _attack in pending:
            if effect.sensor not in views:
                views[effect.sensor] = simulator_runtime.sensor_view(effect.sensor)
            view = views[effect.sensor]
            if view.sensor not in data:
                data[view.sensor] = deepcopy(view.data)
            data[view.sensor] = effect.apply(data[view.sensor], view)
            self._receipt(effect_id, effect, context.step, "capture", context.sim_time, context.frame)
            self.receipts[-1]["render"] = dict(effect.render_metadata())
        for sensor, value in data.items():
            observation = write_camera_rgb(observation, sensor, value)
        return observation

    def finish_decision(self) -> None:
        """Reject AVStack effects whose stage was skipped in this decision."""
        if self._avstack:
            raise RuntimeError(
                f"Attack effects were not applied: {sorted({e.stage for _, e, _ in self._avstack})}")

    def _at_stage(self, value: Any, context: StageContext) -> Any:
        remaining = []
        for effect_id, effect, attack in self._avstack:
            if effect.stage != context.stage:
                remaining.append((effect_id, effect, attack))
                continue
            truth = context.physical_state
            base = self._step_contexts[id(attack)]
            value = effect.apply(value, replace(
                base, stage=context.stage, sim_time=truth.t, frame=truth.frame,
                physical_state=deepcopy(truth) if base.physical_state is not None else None,
            ))
            self._receipt(effect_id, effect, context.step, context.stage, truth.t, truth.frame)
        self._avstack = remaining
        return value

    def _at_world_step(self, value: None, context: StageContext) -> None:
        from avsectester.closed_loop.simulator import SimulatorRuntime

        pending, self._world = self._world, []
        if not pending:
            return
        simulator_runtime = context.simulator_runtime
        if not isinstance(simulator_runtime, SimulatorRuntime):
            raise UnsupportedStageError("This simulator cannot apply world attack effects")
        for effect_id, effect, _attack in pending:
            effect.apply(simulator_runtime, self._resources)
            truth = context.physical_state
            self._receipt(effect_id, effect, context.step, context.stage, truth.t, truth.frame)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        context = replace(self._last_context, stage="stop") if self._last_context else None
        # Stop also runs for zero decisions and partial failures. An effect whose point of
        # action was never reached was not applied, whether or not the run succeeded.
        pending = sorted([*self._world, *self._sensor, *self._avstack], key=lambda item: item[0])
        for effect_id, effect, _attack in pending:
            assert context is not None
            self._receipt(effect_id, effect, context.step, context.stage, context.sim_time,
                          context.frame, "not_applied", "run_ended_before_point_of_action")
        self._world, self._sensor, self._avstack = [], [], []
        failure: BaseException | None = None
        for attack in reversed(self._started):
            try:
                assert context is not None
                attack.stop(self._for_attack(attack, context))
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(f"Attack cleanup also failed: {error!r}")
        try:
            self._resources.close()
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                failure.add_note(f"World effect cleanup also failed: {error!r}")
        if failure is not None:
            raise failure


def _target(effect: AttackEffect) -> str:
    if isinstance(effect, AVStackAttackEffect):
        return effect.stage
    if isinstance(effect, SensorAttackEffect):
        return effect.sensor
    return "world"
