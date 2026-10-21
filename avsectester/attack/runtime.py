"""The online phase: one AttackRuntime per run and the attack actions it returns.

Actions are classified by where they act in the closed loop:

* ``WorldAttackAction`` changes the simulated world. The simulator applies it at setup or
  before a world step, and it is visible from the next capture.
* ``SensorAttackAction`` changes one captured sensor's data before Vehicle I/O, using the
  simulator's sensor view. It applies to the current decision.
* ``AVStackAttackAction`` changes a driving-system value from ``localization.pre`` through
  ``command`` in the current decision.
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


class AttackAction:
    """Base of everything the online phase applies. Subclasses state where it acts."""


class WorldAttackAction(AttackAction, ABC):
    """Change the simulated world. The simulator applies it; it is visible from the next capture.

    ``setup_ticks`` is the number of ``simulator.setup_tick()`` calls that ``apply`` makes when
    the action is returned by ``AttackRuntime.setup``. Ticks are unavailable after setup.
    Register the removal of created world objects on ``resources``; it runs when the run ends.
    """

    setup_ticks: int = 0

    @abstractmethod
    def apply(self, simulator_runtime: SimulatorRuntime, resources: ExitStack) -> None:
        """Change the world through the simulator's native access."""


class SensorAttackAction(AttackAction, ABC):
    """Change one captured sensor's data before Vehicle I/O, in the current decision.

    ``sensor`` names the simulator sensor. ``apply`` receives the current data, which earlier
    actions on the same sensor may have changed, and the capture's view, whose ``data`` is
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
class AVStackAttackAction(AttackAction):
    """Change the value at a driving-system stage in the current decision.

    ``apply(value, context)`` returns the value used downstream. It may be a bound method of
    the runtime, so a decision that depends on a component output keeps its state there.
    """

    stage: str
    apply: Callable[[Any, OnlineContext], Any]

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("AVStackAttackAction.stage must be a nonempty string")
        if not callable(self.apply):
            raise TypeError("AVStackAttackAction.apply must be callable")


class AttackRuntime:
    """One online-phase instance per run; every callback shares this object's state.

    ``setup`` runs after the simulator reset and before the first capture, and returns only
    world attack actions. ``step`` runs once per decision before Vehicle I/O. ``stop`` runs
    at the end of the run, also after failures and with zero decisions. ``setup_ticks`` is the
    simulator tick budget used by the world actions that ``setup`` returns.
    """

    subscriptions: Sequence[ResultSubscription] = ()
    setup_ticks: int = 0

    def setup(self, context: OnlineContext) -> Sequence[WorldAttackAction]:
        return ()

    def step(self, context: OnlineContext) -> Sequence[AttackAction]:
        return ()

    def stop(self, context: OnlineContext) -> None:
        pass


# Stages an AVStack attack action can target, in their order within one decision.
_STAGE_ORDER = (
    "localization.pre", "localization.post", "observation",
    "perception.pre", "perception.post", "tracking.pre", "tracking.post",
    "planning.pre", "planning.post", "control.pre", "control.post",
    "policy.pre", "policy.post", "command",
)


def _category(action: AttackAction) -> str:
    for kind, name in ((WorldAttackAction, "world"), (SensorAttackAction, "sensor"),
                       (AVStackAttackAction, "avstack")):
        if isinstance(action, kind):
            return name
    raise TypeError("Attack callbacks must return WorldAttackAction, SensorAttackAction or "
                    "AVStackAttackAction instances")


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


def chain_sensor_actions(
    observation: Observation, simulator_runtime: Any, actions: Sequence[SensorAttackAction], *,
    views: dict[str, SensorView] | None = None,
    on_applied: Callable[[SensorAttackAction], None] | None = None,
) -> Observation:
    """Apply sensor actions in issue order to copies of their captures and write them back.

    Each action receives the output of the previous action on the same physical sensor, so
    actions that address one sensor through different names chain on a single copy; the sensor
    view's own data is never modified. ``views`` holds sensor views already read for this
    capture. ``on_applied(action)`` runs after each action.
    """
    from avsectester.closed_loop.datatypes import write_camera_rgb, write_sensor_data

    views = {} if views is None else views
    data: dict[str, Any] = {}
    camera_sensors: set[str] = set()
    for action in actions:
        if action.sensor not in views:
            views[action.sensor] = simulator_runtime.sensor_view(action.sensor)
        view = views[action.sensor]
        if view.sensor not in data:
            data[view.sensor] = deepcopy(view.data)
            if view.camera is not None:
                camera_sensors.add(view.sensor)
        data[view.sensor] = action.apply(data[view.sensor], view)
        if on_applied is not None:
            on_applied(action)
    for sensor, value in data.items():
        write = write_camera_rgb if sensor in camera_sensors else write_sensor_data
        observation = write(observation, sensor, value)
    return observation


class AttackDispatcher:
    """Internal: apply the actions of the run's attack runtimes at their points of action.

    Validate before reset, then append ``hooks`` to the run's HookSet. ``setup`` runs inside
    the reset through the simulator.setup hook. ``step`` runs
    once per decision before ``apply_sensor_actions`` and Vehicle I/O; ``stop`` runs in a
    finally block. Every action gets a request record and a receipt.
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
        self._world: list[tuple[int, WorldAttackAction, AttackRuntime]] = []
        self._sensor: list[tuple[int, SensorAttackAction, AttackRuntime]] = []
        self._avstack: list[tuple[int, AVStackAttackAction, AttackRuntime]] = []
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

    def _check_actions(self, actions: Iterable[AttackAction]) -> None:
        assert self._available is not None
        for action in actions:
            _category(action)
            if isinstance(action, WorldAttackAction) and "simulator.step.pre" not in self._available:
                raise UnsupportedStageError("This simulator cannot apply world attack actions")
            if isinstance(action, AVStackAttackAction):
                if action.stage == "sensors.post":
                    raise UnsupportedStageError(
                        "AVStackAttackAction cannot target sensors.post; "
                        "use SensorAttackAction to modify sensor data")
                if action.stage not in _STAGE_ORDER or action.stage not in self._available:
                    raise UnsupportedStageError(f"Unsupported attack action stage: {action.stage!r}")

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

    def _request(self, action: AttackAction, context: OnlineContext, attack: AttackRuntime,
                 callback: str) -> int:
        action_id = self._next_id
        self._next_id += 1
        self.requests.append({
            "action_id": action_id, "step": context.step, "stage": context.stage,
            "sim_time": context.sim_time, "frame": context.frame,
            "issuer": f"{self._labels[id(attack)]}.{callback}",
            "category": _category(action), "target": _target(action),
            "action": type(action).__qualname__,
        })
        return action_id

    def _receipt(self, action_id: int, action: AttackAction, step: int, stage: str,
                 sim_time: float, frame: int, outcome: str = "applied",
                 reason: str | None = None) -> None:
        self.receipts.append({
            "action_id": action_id, "step": step, "stage": stage, "sim_time": sim_time,
            "frame": frame, "category": _category(action), "target": _target(action),
            "outcome": outcome, "reason": reason,
        })

    def drain_records(self) -> dict[str, list[dict[str, Any]]]:
        """Transfer request/receipt records while keeping pending actions and unique IDs.

        The runner drains after recording initialization, decisions and cleanup. Direct
        callers retain accumulating lists unless they explicitly request this transfer.
        """
        result = {"requests": list(self.requests), "receipts": list(self.receipts)}
        self.requests.clear()
        self.receipts.clear()
        return result

    def setup(self, context: OnlineContext, simulator_runtime: SimulatorRuntime) -> None:
        """Run every ``setup`` and apply its world actions before the first capture."""
        if self._available is None:
            raise RuntimeError("Validate attack runtimes before setup")
        if self._has_setup or self._stopped:
            raise RuntimeError("Attack runtimes can only be set up once")
        self._has_setup = True
        self._last_context = context
        for attack in self.attacks:
            # Include partially set-up attacks in cleanup if their callback fails.
            self._started.append(attack)
            actions = attack.setup(self._for_attack(attack, context))
            if actions is None:
                raise TypeError("AttackRuntime.setup must return a sequence of actions, not None")
            for action in tuple(actions):
                if not isinstance(action, WorldAttackAction):
                    raise TypeError("AttackRuntime.setup returns only WorldAttackAction instances")
                action_id = self._request(action, context, attack, "setup")
                action.apply(simulator_runtime, self._resources)
                truth = simulator_runtime.state()
                self._receipt(action_id, action, context.step, "simulator.setup", truth.t, truth.frame)

    def step(self, context: OnlineContext) -> None:
        if not self._has_setup or self._stopped:
            raise RuntimeError("Attack runtimes are not active")
        if context.step != self._last_step + 1:
            raise ValueError("AttackRuntime.step must run exactly once per consecutive decision")
        pending = [*self._world, *self._sensor, *self._avstack]
        if pending:
            raise RuntimeError(f"Attack actions were not applied: {sorted(i for i, *_ in pending)}")
        self._last_step = context.step
        self._last_context = context
        for attack in self.attacks:
            attack_context = self._for_attack(attack, context)
            actions = attack.step(attack_context)
            if actions is None:
                raise TypeError("AttackRuntime.step must return a sequence of actions, not None")
            actions = tuple(actions)
            self._check_actions(actions)
            for action in actions:
                action_id = self._request(action, context, attack, "step")
                if isinstance(action, WorldAttackAction):
                    self._world.append((action_id, action, attack))
                elif isinstance(action, SensorAttackAction):
                    self._sensor.append((action_id, action, attack))
                elif isinstance(action, AVStackAttackAction):
                    self._avstack.append((action_id, action, attack))
                    self._step_contexts[id(attack)] = attack_context

    def record_end(self, context: OnlineContext) -> None:
        """Remember the post-step clock for a later stop context."""
        self._last_context = context

    def apply_sensor_actions(self, observation: Observation, simulator_runtime: Any) -> Observation:
        """Apply this decision's sensor actions to the captured observation, in issue order."""
        if not self._sensor:
            return observation
        pending, self._sensor = self._sensor, []
        context = self._last_context
        assert context is not None
        action_ids = iter(action_id for action_id, _action, _attack in pending)

        def receipt(action: SensorAttackAction) -> None:
            self._receipt(next(action_ids), action, context.step, "capture", context.sim_time,
                          context.frame)
            self.receipts[-1]["render"] = dict(action.render_metadata())

        return chain_sensor_actions(observation, simulator_runtime,
                                    [action for _id, action, _attack in pending], on_applied=receipt)

    def finish_decision(self) -> None:
        """Reject AVStack actions whose stage was skipped in this decision."""
        if self._avstack:
            raise RuntimeError(
                f"Attack actions were not applied: {sorted({e.stage for _, e, _ in self._avstack})}")

    def _at_stage(self, value: Any, context: StageContext) -> Any:
        remaining = []
        for action_id, action, attack in self._avstack:
            if action.stage != context.stage:
                remaining.append((action_id, action, attack))
                continue
            truth = context.physical_state
            base = self._step_contexts[id(attack)]
            value = action.apply(value, replace(
                base, stage=context.stage, sim_time=truth.t, frame=truth.frame,
                physical_state=deepcopy(truth) if base.physical_state is not None else None,
            ))
            self._receipt(action_id, action, context.step, context.stage, truth.t, truth.frame)
        self._avstack = remaining
        return value

    def _at_world_step(self, value: None, context: StageContext) -> None:
        from avsectester.closed_loop.simulator import SimulatorRuntime

        pending, self._world = self._world, []
        if not pending:
            return
        simulator_runtime = context.simulator_runtime
        if not isinstance(simulator_runtime, SimulatorRuntime):
            raise UnsupportedStageError("This simulator cannot apply world attack actions")
        for action_id, action, _attack in pending:
            action.apply(simulator_runtime, self._resources)
            truth = context.physical_state
            self._receipt(action_id, action, context.step, context.stage, truth.t, truth.frame)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        context = replace(self._last_context, stage="stop") if self._last_context else None
        # Stop also runs for zero decisions and partial failures. An action whose point of
        # action was never reached was not applied, whether or not the run succeeded.
        pending = sorted([*self._world, *self._sensor, *self._avstack], key=lambda item: item[0])
        for action_id, action, _attack in pending:
            assert context is not None
            self._receipt(action_id, action, context.step, context.stage, context.sim_time,
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
                failure.add_note(f"World action cleanup also failed: {error!r}")
        if failure is not None:
            raise failure


def _target(action: AttackAction) -> str:
    if isinstance(action, AVStackAttackAction):
        return action.stage
    if isinstance(action, SensorAttackAction):
        return action.sensor
    return "world"
