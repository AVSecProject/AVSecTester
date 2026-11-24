"""Apply a run's attack actions at their closed-loop execution stages.

The dispatcher owns execution order and receipts. Each AttackRuntime owns its attack state.
Sensor actions chain on detached capture data before the driving system's ObservationPipeline.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from avsectester.validation import require_nonnegative_integer, timestamp_to_microseconds
from avsectester.attack.runtime import (
    AttackAction, AttackRuntime, AVStackAttackAction, OnlineContext, ResultSubscription,
    SensorAttackAction, WorldAttackAction,
)
from avsectester.closed_loop.hooks import (
    EXTERNAL_CONTROL_INPUT_ERROR, Hook, StageContext, UnsupportedStageError,
)
from avsectester.data_types import Observation, RosMessage

if TYPE_CHECKING:
    from avsectester.closed_loop.simulators.messages import MessageStage
    from avsectester.closed_loop.simulators.base import SensorView, SimulatorRuntime


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
    actions that address one sensor through different names chain on a single copy. The sensor
    view's own data is never modified. ``views`` holds sensor views already read for this
    capture. ``on_applied(action)`` runs after each action.
    """
    from avsectester.data_types import write_camera_rgb, write_sensor_data

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
    once per decision before ``apply_sensor_actions`` and ObservationPipeline. ``stop`` runs in a
    finally block. Every action gets a request record and a receipt.
    """

    def __init__(
        self, attacks: Iterable[AttackRuntime] = (), *,
        observation_copy: Callable[[Observation], Observation] = deepcopy,
        output_recorder: Callable[[OnlineContext, str, Any], None] | None = None,
    ) -> None:
        self.attacks = tuple(attacks)
        self._callback_lock: AbstractContextManager[Any] | None = None
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
            require_nonnegative_integer(ticks, "AttackRuntime.setup_ticks")
            self._subscriptions[id(attack)] = frozenset(item.kind for item in subscriptions)
        self._available: frozenset[str] | None = None
        self._world_only = False
        self._message_stages: dict[str, str | None] = {}
        self._message_sensor_topics: dict[str, str] = {}
        self._started: list[AttackRuntime] = []
        self._resources = ExitStack()
        self._last_context: OnlineContext | None = None
        self._step_contexts: dict[int, OnlineContext] = {}
        self._world: list[tuple[int, WorldAttackAction, AttackRuntime]] = []
        self._sensor: list[tuple[int, SensorAttackAction, AttackRuntime]] = []
        self._persistent: list[tuple[int, SensorAttackAction, AttackRuntime, int, int | None]] = []
        self._applied_persistent: set[int] = set()
        self._avstack: list[tuple[int, AVStackAttackAction, AttackRuntime]] = []
        # A decision's message actions remain current until the next decision replaces them.
        self._message_sensor: list[tuple[int, SensorAttackAction, AttackRuntime]] = []
        self._message_persistent: list[tuple[int, SensorAttackAction, AttackRuntime, int, int | None]] = []
        self._message_stage: list[tuple[int, AVStackAttackAction, AttackRuntime]] = []
        self._message_stage_applied: set[int] = set()
        self._last_step = -1
        self._has_setup = False
        self._stopped = False
        self._next_id = 0
        self.requests: list[dict[str, Any]] = []
        self.receipts: list[dict[str, Any]] = []

    def bind_callback_lock(self, lock: AbstractContextManager[Any]) -> None:
        """Bind the owning session's shared lock once, before setup."""
        if self._callback_lock is not None or self._has_setup:
            raise RuntimeError("Bind the attack callback lock once before setup")
        self._callback_lock = lock

    def _require_callback_lock(self) -> AbstractContextManager[Any]:
        if self._callback_lock is None:
            raise RuntimeError("Bind the attack callback lock before running callbacks")
        return self._callback_lock

    @property
    def setup_ticks(self) -> int:
        return sum(attack.setup_ticks for attack in self.attacks)

    @property
    def started(self) -> bool:
        return bool(self._started) and not self._stopped

    def validate(self, available_stages: Iterable[str], *, world_only: bool = False,
                 message_stages: Mapping[str, MessageStage] | None = None) -> None:
        self._available = frozenset(available_stages)
        self._world_only = world_only
        self._message_stages = {topic: stage.sensor for topic, stage in (message_stages or {}).items()
                                if stage.modifiable}
        self._message_sensor_topics = {sensor: topic for topic, sensor in self._message_stages.items()
                                       if sensor is not None}

    def _check_actions(self, actions: Iterable[AttackAction]) -> None:
        assert self._available is not None
        for action in actions:
            _category(action)
            message_stage = isinstance(action, AVStackAttackAction) and action.stage in self._message_stages
            message_sensor = (isinstance(action, SensorAttackAction)
                              and action.sensor in self._message_sensor_topics)
            if (self._world_only and not isinstance(action, WorldAttackAction)
                    and not (message_stage or message_sensor)):
                raise UnsupportedStageError(
                    "External control supports world actions and actions on modifiable ROS topics. "
                    + EXTERNAL_CONTROL_INPUT_ERROR)
            if isinstance(action, WorldAttackAction) and "simulator.step.pre" not in self._available:
                raise UnsupportedStageError("This simulator cannot apply world attack actions")
            if isinstance(action, AVStackAttackAction):
                if action.stage == "sensors.post":
                    raise UnsupportedStageError(
                        "AVStackAttackAction cannot target sensors.post. "
                        "Use SensorAttackAction to modify sensor data")
                if (not message_stage and action.stage not in _STAGE_ORDER) or action.stage not in self._available:
                    raise UnsupportedStageError(f"Unsupported attack action stage: {action.stage!r}")
            if isinstance(action, SensorAttackAction) and action.active_for_us is not None:
                require_nonnegative_integer(action.active_for_us, "SensorAttackAction.active_for_us")

    @property
    def hooks(self) -> tuple[Hook, ...]:
        if self._available is None:
            raise RuntimeError("Validate attack runtimes before binding hooks")
        if not self.attacks:
            return ()
        hooks = []
        if "simulator.setup" in self._available:
            hooks.append(Hook("simulator.setup", _SetupHook(self)))
        if not self._world_only:
            hooks.extend(Hook(stage, self._at_stage) for stage in _STAGE_ORDER if stage in self._available)
        hooks.extend(Hook(stage, self._at_message_stage) for stage in self._message_stages)
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
            "details": dict(action.receipt_details()) if outcome == "applied" else {},
        })

    def records(self) -> dict[str, list[dict[str, Any]]]:
        """Data-only request and receipt records, ready for serialization."""
        return {"requests": list(self.requests), "receipts": list(self.receipts)}

    def drain_records(self) -> dict[str, list[dict[str, Any]]]:
        """Transfer request/receipt records while keeping pending actions and unique IDs.

        The runner drains after recording initialization, decisions and cleanup. Direct
        callers retain accumulating lists unless they explicitly request this transfer.
        """
        result = self.records()
        self.requests.clear()
        self.receipts.clear()
        return result

    def setup(self, context: OnlineContext, simulator_runtime: SimulatorRuntime) -> None:
        """Run every ``setup`` and apply its world actions before the first capture."""
        with self._require_callback_lock():
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
                actions = tuple(actions)
                if self._world_only:
                    self._check_actions(actions)
                for action in actions:
                    if not isinstance(action, WorldAttackAction):
                        raise TypeError("AttackRuntime.setup returns only WorldAttackAction instances")
                    action_id = self._request(action, context, attack, "setup")
                    action.apply(simulator_runtime, self._resources)
                    truth = simulator_runtime.state()
                    self._receipt(action_id, action, context.step, "simulator.setup", truth.t, truth.frame)

    def step(self, context: OnlineContext) -> None:
        self.issue_step(context, self.plan_step(context))

    def plan_step(self, context: OnlineContext
                  ) -> list[tuple[AttackRuntime, OnlineContext, tuple[AttackAction, ...]]]:
        """Evaluate decision callbacks before replacing the active message actions."""
        with self._require_callback_lock():
            planned = []
            for attack in self.attacks:
                attack_context = self._for_attack(attack, context)
                actions = attack.step(attack_context)
                if actions is None:
                    raise TypeError("AttackRuntime.step must return a sequence of actions, not None")
                actions = tuple(actions)
                self._check_actions(actions)
                planned.append((attack, attack_context, actions))
            return planned

    def issue_step(self, context: OnlineContext,
                   planned: Sequence[tuple[AttackRuntime, OnlineContext, tuple[AttackAction, ...]]]) -> None:
        """Register the next decision's actions without invoking user callbacks."""
        if not self._has_setup or self._stopped:
            raise RuntimeError("Attack runtimes are not active")
        if context.step != self._last_step + 1:
            raise ValueError("AttackRuntime.step must run exactly once per consecutive decision")
        pending = [*self._world, *self._sensor, *self._avstack, *self._message_sensor, *self._message_stage]
        if pending:
            raise RuntimeError(f"Attack actions were not applied: {sorted(i for i, *_ in pending)}")
        self._last_step = context.step
        self._last_context = context
        for attack, attack_context, actions in planned:
            for action in actions:
                action_id = self._request(action, context, attack, "step")
                if isinstance(action, WorldAttackAction):
                    self._world.append((action_id, action, attack))
                elif isinstance(action, SensorAttackAction):
                    message = action.sensor in self._message_sensor_topics
                    if action.active_for_us == 0:
                        (self._message_sensor if message else self._sensor).append((action_id, action, attack))
                    else:
                        start = context.decision_time_us
                        if start is None:
                            start = timestamp_to_microseconds(context.sim_time)
                        end = None if action.active_for_us is None else start + action.active_for_us
                        (self._message_persistent if message else self._persistent).append(
                            (action_id, action, attack, start, end))
                elif isinstance(action, AVStackAttackAction):
                    message = action.stage in self._message_stages
                    (self._message_stage if message else self._avstack).append((action_id, action, attack))
                    self._step_contexts[id(attack)] = attack_context

    def record_end(self, context: OnlineContext) -> None:
        """Remember the post-step clock for a later stop context."""
        self._last_context = context

    def discard_missing_sensor_actions(
        self, observation: Observation, simulator_runtime: SimulatorRuntime,
    ) -> None:
        """Resolve sensor targets and reject one-shot actions without an issuing-time frame."""
        context = self._last_context
        assert context is not None
        for _identity, action, _attack, _start, _end in self._persistent:
            simulator_runtime.canonical_sensor_id(action.sensor)
        remaining = []
        for action_id, action, attack in self._sensor:
            sensor = simulator_runtime.canonical_sensor_id(action.sensor)
            if sensor in observation.sensor_data:
                remaining.append((action_id, action, attack))
            else:
                self._receipt(action_id, action, context.step, "capture", observation.t,
                              observation.frame, "not_applied", "sensor_not_acquired_at_issue_time")
        self._sensor = remaining

    def apply_sensor_actions(self, observation: Observation, simulator_runtime: Any) -> Observation:
        """Apply one-shot and active duration actions to this timestamp's due sensors."""
        with self._require_callback_lock():
            time_us = simulator_runtime.clock.time_us
            for identity, action, _attack, _start, end in self._persistent:
                if end is not None and time_us >= end:
                    if identity not in self._applied_persistent:
                        context = self._last_context
                        assert context is not None
                        self._receipt(identity, action, context.step, "capture", observation.t,
                                      observation.frame, "not_applied", "no_frame_during_active_interval")
                    self._applied_persistent.discard(identity)
            self._persistent = [item for item in self._persistent if item[4] is None or time_us < item[4]]
            pending = list(self._sensor)
            pending.extend((identity, action, attack) for identity, action, attack, start, end in self._persistent
                           if start <= time_us and simulator_runtime.canonical_sensor_id(action.sensor)
                           in observation.sensor_data)
            pending.sort(key=lambda item: item[0])
            if not pending:
                return observation
            self._sensor = []
            context = self._last_context
            assert context is not None
            action_ids = iter(action_id for action_id, _action, _attack in pending)

            def receipt(action: SensorAttackAction) -> None:
                identity = next(action_ids)
                if action.active_for_us != 0:
                    self._applied_persistent.add(identity)
                self._receipt(identity, action, context.step, "capture", observation.t,
                              observation.frame)

            return chain_sensor_actions(observation, simulator_runtime,
                                        [action for _id, action, _attack in pending], on_applied=receipt)

    def finish_decision(self, *, skip_driver: bool = False, every_sensor_due: bool = True) -> None:
        """Reject AVStack actions whose stage was skipped in this decision.

        A driver skipped during force_gt, or a decision where some sensor had no frame due,
        records them as not applied. Otherwise a skipped stage raises RuntimeError.
        """
        if not self._avstack:
            return
        if every_sensor_due and not skip_driver:
            raise RuntimeError(
                f"Attack actions were not applied: {sorted({e.stage for _, e, _ in self._avstack})}")
        context = self._last_context
        assert context is not None
        reason = "driving_system_skipped_during_force_gt" if skip_driver else "stage_not_executed"
        for identity, action, _attack in self._avstack:
            self._receipt(identity, action, context.step, action.stage, context.sim_time,
                          context.frame, "not_applied", reason)
        self._avstack.clear()

    @staticmethod
    def _message_result(action: AttackAction, value: RosMessage, result: Any, topic: str) -> RosMessage:
        if not isinstance(result, RosMessage) or result.type != value.type:
            raise TypeError(f"{topic}: {type(action).__name__} must return RosMessage({value.type!r}, ...)")
        return result

    def apply_message_sensor_actions(self, topic: str, value: RosMessage, view: SensorView,
                                     time_us: int) -> RosMessage:
        """Apply issued sensor actions to one native message in issue order."""
        with self._require_callback_lock():
            context = self._last_context
            assert context is not None
            pending = [(identity, action) for identity, action, _ in self._message_sensor
                       if action.sensor == view.sensor]
            self._message_sensor = [item for item in self._message_sensor if item[1].sensor != view.sensor]
            pending.extend((identity, action) for identity, action, _, start, end in self._message_persistent
                           if action.sensor == view.sensor and start <= time_us and (end is None or time_us < end))
            for identity, action in sorted(pending, key=lambda item: item[0]):
                point = replace(view, data=value)
                value = self._message_result(action, value, action.apply(value, point), topic)
                if action.active_for_us != 0:
                    self._applied_persistent.add(identity)
                self._receipt(identity, action, context.step, topic, view.time, view.frame)
            return value

    def _at_message_stage(self, value: Any, context: StageContext) -> Any:
        with self._require_callback_lock():
            truth = context.physical_state
            for identity, action, attack in self._message_stage:
                if action.stage != context.stage:
                    continue
                base = self._step_contexts[id(attack)]
                result = action.apply(value, replace(
                    base, stage=context.stage, sim_time=truth.t, frame=truth.frame,
                    physical_state=deepcopy(truth),
                    _output_recorder=lambda _context, name, datum: context.record_output(name, datum)))
                value = self._message_result(action, value, result, context.stage)
                self._message_stage_applied.add(identity)
                self._receipt(identity, action, context.step, context.stage, truth.t, truth.frame)
            return value

    def finish_interval(self, context: OnlineContext, *, final: bool) -> None:
        """Close a decision interval for actions on intercepted native messages.

        One-shot sensor and stage actions without a message in the interval are recorded as not
        applied. Sensor actions with a duration remain until it ends or the run's final interval.
        Without message actions this does nothing.
        """
        topics = self._message_sensor_topics
        for identity, action, _ in self._message_sensor:
            self._receipt(identity, action, context.step, topics[action.sensor], context.sim_time,
                          context.frame, "not_applied", "sensor_not_acquired_at_issue_time")
        self._message_sensor.clear()
        for identity, stage_action, _ in self._message_stage:
            if identity not in self._message_stage_applied:
                self._receipt(identity, stage_action, context.step, stage_action.stage, context.sim_time,
                              context.frame, "not_applied", "stage_not_executed")
        self._message_stage.clear()
        self._message_stage_applied.clear()
        remaining = []
        now = timestamp_to_microseconds(context.sim_time)
        for item in self._message_persistent:
            identity, action, _, _, end = item
            if not final and (end is None or now < end):
                remaining.append(item)
                continue
            if identity not in self._applied_persistent:
                self._receipt(identity, action, context.step, topics[action.sensor], context.sim_time,
                              context.frame, "not_applied", "no_frame_during_active_interval")
            self._applied_persistent.discard(identity)
        self._message_persistent = remaining

    def _at_stage(self, value: Any, context: StageContext) -> Any:
        with self._require_callback_lock():
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
        with self._require_callback_lock():
            from avsectester.closed_loop.simulators.base import SimulatorRuntime

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
        with self._require_callback_lock():
            if self._stopped:
                return
            self._stopped = True
            context = replace(self._last_context, stage="stop") if self._last_context else None
            # Stop also runs for zero decisions and partial failures. An action whose point of
            # action was never reached was not applied, whether or not the run succeeded.
            persistent = (*self._persistent, *self._message_persistent)
            pending = sorted([*self._world, *self._sensor, *self._avstack, *self._message_sensor,
                              *self._message_stage,
                              *((identity, action, attack) for identity, action, attack, _start, _end
                                in persistent if identity not in self._applied_persistent)],
                             key=lambda item: item[0])
            for action_id, action, _attack in pending:
                assert context is not None
                self._receipt(action_id, action, context.step, context.stage, context.sim_time,
                              context.frame, "not_applied", "run_ended_before_point_of_action")
            self._world, self._sensor, self._avstack = [], [], []
            self._message_sensor, self._message_stage = [], []
            self._persistent.clear()
            self._message_persistent.clear()
            self._applied_persistent.clear()
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
