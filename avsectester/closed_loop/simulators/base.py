"""The simulator: a configuration (``Simulator``) and its live worlds (``SimulatorRuntime``).

``Simulator`` is what a user configures: settings and declarations, with no world state. Each
run or service operation creates one ``SimulatorRuntime`` through ``create_runtime``. The
runtime owns the world, clock, current capture and native connections until it is closed.
Concrete runtimes implement native initialization, one native advance and one native capture.
The runtime base owns scenario-start reset, pre-roll, deployment windows and capture ordering.
It also owns the process-local reservation of the simulated world and its read-only clock.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
import math
from numbers import Real
from operator import itemgetter
from threading import Lock
from typing import Any, TYPE_CHECKING

from avsectester.closed_loop.hooks import _HookBindingMixin
from avsectester.validation import timestamp_to_microseconds
from avsectester.closed_loop.timing import (
    HistoryRequirement, SensorTiming, TimingCapabilities, TimingConfiguration,
    seconds_to_microseconds, validate_decision_period,
)
from avsectester.closed_loop.simulators.object_composition import compose_capture
from avsectester.data_types import ActorState
from avsectester.datasets.base import FrameData, FrameReference, ScenarioStart
from avsectester.geometry.poses import interpolate_transform, validated_transform, validated_vector
from avsectester.validation import require_nonnegative_integer
from avsectester.data_types import (
    Control, Observation, WorldSnapshot, is_external_control, validate_command_modes,
)

if TYPE_CHECKING:
    import numpy as np

    from avsectester.closed_loop.simulators.messages import MessageChannel, MessageStage
    from avsectester.geometry.cameras import Camera
    from avsectester.rendering import (
        CompositePlan, CompositeResult, Harmonizer, MeshAsset, ObjectAsset, ObjectComposition, ObjectVisibility, PlacedObject,
    )


def validated_camera_aliases(camera_aliases: Mapping[str, str]) -> dict[str, str]:
    """Copy native-to-output camera names after checking names and output uniqueness."""
    if not isinstance(camera_aliases, Mapping) or any(
        not isinstance(native, str) or not native or not isinstance(alias, str) or not alias
        for native, alias in camera_aliases.items()
    ):
        raise ValueError("camera_aliases must map nonempty native camera names to nonempty aliases")
    if len(set(camera_aliases.values())) != len(camera_aliases):
        raise ValueError("camera_aliases must use unique output names")
    return dict(camera_aliases)


_reservation_lock = Lock()
_reservation_owners: dict[Hashable, tuple[object, int]] = {}


@contextmanager
def _claim_world(key: Hashable, owner: object) -> Iterator[None]:
    """Nonblocking ownership primitive shared by runtimes and execution operations.

    One owner may nest reservations of the same resource. Independent owners fail
    immediately. The final context exit releases that resource for another owner.
    This process-local guard cannot reserve a server against other processes.
    """
    with _reservation_lock:
        previous, depth = _reservation_owners.get(key, (owner, 0))
        if previous is not owner:
            raise RuntimeError(f"Simulator resource is already in use: {key!r}")
        _reservation_owners[key] = (owner, depth + 1)
    try:
        yield
    finally:
        with _reservation_lock:
            _, depth = _reservation_owners[key]
            if depth == 1:
                del _reservation_owners[key]
            else:
                _reservation_owners[key] = (owner, depth - 1)


@dataclass(frozen=True)
class DeploymentPlan:
    """Shared tick budget for all treatments, including frame data generation.

    Participants are the simulator.setup hooks bound by the runner, including attack runtimes
    whose setup returns world attack actions. They may consume ticks through setup_tick.
    The remainder is executed by the simulator. Neither path can exceed this budget.
    Selection metadata never enters the simulator.
    """

    ticks: int = 0

    def __post_init__(self) -> None:
        require_nonnegative_integer(self.ticks, "deployment ticks")


def can_deploy_at(ticks: int, deployment: DeploymentPlan, capture_ticks: int) -> bool:
    """Whether deployment can preserve this selected decision boundary.

    Frame zero permits a held-origin setup. Later boundaries need enough preceding ticks
    for the shared deployment budget and the simulator's capture latency.
    """
    require_nonnegative_integer(ticks, "ticks")
    require_nonnegative_integer(capture_ticks, "capture ticks")
    return ticks == 0 or ticks >= deployment.ticks + capture_ticks


@dataclass(frozen=True)
class ClockSnapshot:
    """Physical simulator time and frame, never a model-visible timestamp."""

    t: float
    frame: int
    scheduled_time_us: int | None = None

    @property
    def time_us(self) -> int:
        """Integer event time, retaining the native floating timestamp separately."""
        return timestamp_to_microseconds(self.t) if self.scheduled_time_us is None else self.scheduled_time_us


@dataclass(frozen=True)
class SensorGeometry:
    """Physical optical-camera geometry at one capture, without image or SDK handles."""

    sensor: str
    camera: Camera
    cam_from_world: np.ndarray


@dataclass(frozen=True)
class SensorView:
    """One sensor capture, including inserted world objects.

    ``data`` is the sensor's captured data. A camera supplies RGB uint8 (H, W, 3).
    ``camera`` projects optical camera points to pixels and unprojects pixels to rays, hiding pinhole versus f-theta
    differences. ``cam_from_world`` maps world points to optical camera axes (x right, y down,
    z forward). Both are None for non-camera sensors. ``sensor_from_world`` maps world points
    to sensor body axes (x forward, y left, z up), including for a camera. It is not the optical
    transform. It is None when the sensor pose is unavailable. ``actors`` holds world poses by
    stable actor ID and ``ego`` is the physical ego state. Use ``box_transform()`` for
    world bounding-box centers. Each state's ``pose`` is its actor origin.
    ``depth`` is optical-axis depth in meters per pixel, when the simulator measures it.
    Without it, occlusion is estimated from actor cuboids and ``depth`` remains None after
    object composition. Existing measured depth includes composed objects. ``objects`` contains
    the placed world objects whose projection lies in this camera's current image, keyed by
    object ID, including fully occluded ones. They are immutable and shared, not copied.
    ``object_masks`` has the same keys and maps each object to its visible boolean HxW image
    mask. Its pixel count equals that object's ``ObjectVisibility.visible_pixels``.
    ``composition`` exists only for composite insertion. It holds the image beneath the objects
    and the plan that composed them into ``data``. ``plan.apply(background).image`` reproduces
    that capture. Otherwise ``composition`` is None.
    ``object_occlusion`` states how the simulator hides composed objects behind its scene:
    ``"geometry"`` uses this view's depth or actor cuboids, which ``composite_plan`` reproduces.
    ``"rendered_probes"`` measures scene occlusion with opaque probe copies, and ``"scene_render"``
    renders objects together with the scene. ``occlusion_tolerance_m`` applies only to composite
    insertion. Scene content hides a composed object only when it lies more than about this
    distance in front of the object.
    ``native`` gives simulator-specific access and gives up portability. World units are meters, with x forward,
    y left, z up.
    """

    sensor: str
    data: Any
    camera: Camera | None
    cam_from_world: np.ndarray | None
    actors: Mapping[str, ActorState]
    ego: ActorState
    time: float
    frame: int
    sensor_from_world: np.ndarray | None = None
    depth: np.ndarray | None = None
    native: Mapping[str, Any] = field(default_factory=dict)
    objects: Mapping[str, PlacedObject] = field(default_factory=dict)
    composition: ObjectComposition | None = None
    object_occlusion: str = "geometry"
    occlusion_tolerance_m: float = 0.1
    object_masks: Mapping[str, np.ndarray] = field(default_factory=dict)

    def composite_plan(self, objects: Sequence[PlacedObject]) -> CompositePlan:
        """Plan composing ``objects`` into this camera view exactly as simulator captures do.

        The plan uses this view's camera, its measured depth or, without depth, its actor
        cuboids, and its occlusion tolerance. ``plan.composite`` therefore reproduces
        the deployed capture differentiably in surface colors. Plan on a view captured
        without these objects, for example a clean ``ClosedLoopSession.render`` result. Only
        ``"geometry"`` object occlusion can be planned here. Otherwise capture a view with the
        objects inserted, then call ``plan_composite`` using its camera, ``cam_from_world``
        and ``object_masks`` as scene visibility.
        """
        from avsectester.rendering import plan_composite

        if self.camera is None or self.cam_from_world is None:
            raise ValueError(f"Sensor {self.sensor!r} has no optical camera geometry")
        if self.object_occlusion != "geometry":
            raise ValueError(f"Sensor {self.sensor!r} hides objects with {self.object_occlusion!r} occlusion. "
                             "Capture the view with the objects inserted and call plan_composite "
                             "with its object_masks")
        objects = tuple(objects)
        present = sorted({placed.object_id for placed in objects} & self.objects.keys())
        if present:
            raise ValueError(f"View already contains objects {present}. Plan on a view captured without them")
        return plan_composite(self.camera, self.cam_from_world, objects,
                              scene_depth=self.depth, occluders=self.actors,
                              occlusion_tolerance_m=self.occlusion_tolerance_m)


@dataclass(frozen=True)
class Initialization:
    """Source replay inputs only. No attack origin, score or placement enters the simulator."""

    expected: FrameData
    resolved_scenario: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameDataSequence:
    """Detached generated frames and the resolved scenario and provenance that produced them."""

    frames: tuple[FrameData, ...]
    resolved_scenario: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def at(self, frame: FrameReference) -> FrameData:
        """Return frame data at the requested recorded time."""
        if frame.position >= len(self.frames) or self.frames[frame.position].frame != frame:
            raise ValueError("Reference cache does not match the requested source frame")
        return deepcopy(self.frames[frame.position])


class UnsupportedStartError(ValueError):
    """A correctly configured simulator cannot start at this particular scenario start.

    ``Simulator.validate_start`` raises it for problems of the recorded data, for example a
    recording whose navigation route cannot be built. Experiment selection skips such segments
    and records the reason. Other ``validate_start`` errors indicate a configuration problem.
    """


@dataclass(frozen=True)
class DecisionInterval:
    """One native propagation, available to extension hooks until its endpoint is published.

    ``report_times_us`` contains intermediate report times without either endpoint.
    ``start_state`` is physical truth after ``simulator.step.pre`` has run.
    """

    start_us: int
    end_us: int
    report_times_us: tuple[int, ...]
    force_gt: bool
    coerce_dynamic_state: bool
    start_state: WorldSnapshot


@dataclass(frozen=True)
class RecordedAcquisition:
    """One recorded-history read with its observation and physical evidence.

    ``sensor_geometry`` contains the acquired optical cameras. ``object_visibility`` maps
    inserted-object IDs to camera visibility. Both mappings are empty for a pose-only read.
    Values are detached from later live captures and recorded-history reads.
    """

    observation: Observation
    physical_state: WorldSnapshot
    sensor_geometry: Mapping[str, SensorGeometry] = field(default_factory=dict)
    object_visibility: Mapping[str, Mapping[str, ObjectVisibility]] = field(default_factory=dict)


@dataclass(frozen=True)
class _InsertedObject:
    placed: PlacedObject
    velocity: tuple[float, float, float]
    anchor_us: int | None = None
    nodes: tuple[tuple[int, np.ndarray], ...] = ()

    def anchored_at(self, time_us: int) -> _InsertedObject:
        return replace(self, anchor_us=time_us, nodes=((time_us, self.placed.world_from_object),))

    def pose_at(self, time_us: int) -> np.ndarray:
        initial_pose = self.placed.world_from_object
        if self.anchor_us is None or time_us == self.anchor_us:
            return initial_pose
        if time_us < self.anchor_us:
            if not any(self.velocity):
                return initial_pose
            pose = initial_pose.copy()
            elapsed = (time_us - self.anchor_us) / 1e6
            pose[:3, 3] += tuple(elapsed * speed for speed in self.velocity)
            return pose
        if time_us >= self.nodes[-1][0]:
            return self.nodes[-1][1]
        right = bisect_left(self.nodes, time_us, key=itemgetter(0))
        end, target = self.nodes[right]
        start, origin = self.nodes[right - 1]
        if time_us == end or origin is target:
            return target
        return interpolate_transform(origin, target, (time_us - start) / (end - start))

    def velocity_at(self, time_us: int) -> tuple[float, float, float]:
        if self.anchor_us is None or time_us <= self.anchor_us:
            return self.velocity
        if time_us > self.nodes[-1][0]:
            return (0., 0., 0.)
        right = bisect_left(self.nodes, time_us, key=itemgetter(0))
        end, target = self.nodes[right]
        start, origin = self.nodes[right - 1]
        x, y, z = (target[:3, 3] - origin[:3, 3]) / ((end - start) / 1e6)
        return (float(x), float(y), float(z))

    def placed_at(self, time_us: int) -> PlacedObject:
        pose = self.pose_at(time_us)
        if pose is self.placed.world_from_object:
            return self.placed
        return replace(self.placed, world_from_object=pose)

    def state_at(self, time_us: int) -> ActorState:
        asset = self.placed.asset
        return ActorState(self.pose_at(time_us).copy(), self.velocity_at(time_us), asset.category,
                          asset.extent, (0., 0., asset.extent[2] / 2))


@dataclass
class _SensorSlot:
    time_us: int
    observation: Observation
    physical_state: WorldSnapshot
    build_view: Callable[[], SensorView]
    geometry: SensorGeometry | None
    objects: Mapping[str, PlacedObject]
    render_request: Any = None
    view: SensorView | None = None
    composed: bool = False
    visibility: dict[str, ObjectVisibility] = field(default_factory=dict)
    error: BaseException | None = None


class Simulator(ABC):
    """A configured simulator: settings and declarations, without world state.

    It can be reused. ``create_runtime`` creates one live world per run or service operation,
    and the caller closes it. Declarations describe every runtime of this simulator.
    ``occlusion_tolerance_m`` applies to inserted world objects: scene content hides one only
    when it lies more than about this distance in front of it. Simulators that insert objects
    accept it as a constructor option.
    """

    dataset_kinds: tuple[str, ...] = ()
    'Dataset kinds supported by this simulator.'
    can_render: bool = False
    'Whether arbitrary-view rendering is supported.'
    can_replay: bool = False
    'Whether recorded-tick replay is supported.'
    can_observe_past: bool = False
    'Whether observations before a selected start can be reconstructed.'
    can_start_runs_after_origin: bool = True
    'Whether closed-loop runs can start after the scenario origin. Rendering and preparation are unaffected.'
    can_insert_objects: bool = False
    'Whether world objects can be inserted into sensor rendering.'
    can_repeat_runs: bool = False
    'Whether a treatment may run repeatedly because outcomes at fixed settings can vary.'
    supported_world_settings: frozenset[str] = frozenset()
    'World-setting keys accepted by the simulator.'
    simulator_version = "1"
    'Version identifier included in simulator compatibility summaries.'
    # Native ticks that make simulator.setup actions visible in the first capture. A simulator
    # that renders on every tick (CARLA) needs one. One that renders at capture time needs none.
    capture_ticks = 0
    'Simulator ticks consumed by a sensor capture.'
    occlusion_tolerance_m: float = 0.1
    'Depth tolerance in meters for deciding whether an inserted object is occluded.'

    @abstractmethod
    def create_runtime(self, seed: int | None = None) -> SimulatorRuntime:
        """Create a new live world with this configuration. ``seed`` seeds its random choices."""

    @property
    def past_steps(self) -> int:
        """Recorded decision steps required before a selected start."""
        return 0

    @property
    def observation_lead_steps(self) -> int:
        """Recorded decision steps needed before the earliest observation this simulator renders.

        They add to the driving system's history. ``past_steps`` remains the simulator's own
        history before a selected start.
        """
        return 0

    @property
    def decision_time_step(self) -> float | None:
        """Decision interval in seconds. None leaves the interval unrestricted."""
        return None

    @property
    def timing_capabilities(self) -> TimingCapabilities:
        """Declare native timing limits without opening a world."""
        return TimingCapabilities()

    @property
    def command_modes(self) -> frozenset[str] | None:
        """Command representations consumed by this simulator, or None if undeclared."""
        return None

    @property
    def supported_stages(self) -> frozenset[str]:
        """Hook stages supported by the component."""
        return frozenset({"simulator.setup", "simulator.step.pre", "simulator.step.post"})

    @property
    def message_stages(self) -> Mapping[str, MessageStage]:
        """Declare native message stages by their public topic names."""
        return {}

    @property
    def identity(self) -> Mapping[str, Any]:
        """Configuration identity used to check saved-preparation compatibility."""
        identity = {"simulator": type(self).__name__, "version": self.simulator_version,
                    "time_step": self.decision_time_step, "capture_ticks": self.capture_ticks}
        if self.can_insert_objects:
            identity["occlusion_tolerance_m"] = self.occlusion_tolerance_m
        return identity

    def validate_world_settings(self, settings: Mapping[str, Any]) -> None:
        """Check world-setting names and values without creating a live world.

        Simulator implementations extend this with checks that need no native connection.
        Actor existence and other live capabilities are checked during initialization.
        """
        if not isinstance(settings, Mapping):
            raise TypeError("Simulator world settings must be a mapping")
        unsupported = set(settings) - self.supported_world_settings
        if unsupported:
            raise ValueError(f"{type(self).__name__} does not support world settings "
                             f"{sorted(unsupported)}. Use explicitly configured sensors transforms instead")

    def validate_start(self, start: ScenarioStart | None) -> None:
        """Reject a scenario start this simulator cannot reset to, before any world exists.

        Raise ``UnsupportedStartError`` when the configuration is valid but this start's data
        cannot be used, so experiment selection can skip it.
        """
        if start is None:
            return
        if not isinstance(start, ScenarioStart):
            raise TypeError("Simulator reset requires ScenarioStart, never selection metadata")
        if start.kind not in self.dataset_kinds:
            raise ValueError(f"{type(self).__name__} requires {'/'.join(self.dataset_kinds)} ScenarioStart")
        dt = start.payload.get("time_step")
        if dt is not None and dt != self.decision_time_step:
            raise ValueError(f"Dataset time_step={dt} differs from simulator decision time step "
                             f"{self.decision_time_step}")

    def validate_start_timing(self, start: ScenarioStart | None, timing: TimingConfiguration) -> None:
        """Check a recorded start under the selected frame clocks without opening a world."""
        self.validate_start(start)

    @property
    def decision_ticks(self) -> int:
        """Number of native ticks in one decision, or one without a native base tick."""
        tick = self.timing_capabilities.base_tick_us
        if tick is None:
            return 1
        if self.decision_time_step is None:
            raise ValueError("A native-tick simulator must declare its decision interval")
        return seconds_to_microseconds(self.decision_time_step) // tick

    def observation_lead_steps_for(self, timing: TimingConfiguration) -> int:
        """Recorded decision steps preceding a capture under the selected frame clocks."""
        return self.observation_lead_steps

    def validate_timing(self, timing: TimingConfiguration) -> None:
        """Reject unsupported clocks or startup motion without opening a simulator world."""
        if not isinstance(timing, TimingConfiguration):
            raise TypeError("timing must be TimingConfiguration")
        capabilities = self.timing_capabilities
        if (timing.force_gt_duration_us and timing.skip_driver_during_force_gt
                and "trajectory" not in (self.command_modes or ())):
            raise ValueError("skip_driver_during_force_gt requires trajectory commands during force_gt")
        if timing.force_gt_duration_us and not capabilities.recorded_ego_motion:
            raise ValueError(f"{type(self).__name__} does not support force_gt_duration_us")
        if timing.pose_reporting_interval_us and not capabilities.motion_reports:
            raise ValueError(f"{type(self).__name__} does not support intermediate pose reporting")
        if any(sensor.shutter_duration_us for sensor in timing.sensors.values()) and not capabilities.camera_exposure:
            raise ValueError(f"{type(self).__name__} supports only instantaneous camera exposure")
        declared = self.decision_time_step
        period = timing.decision_period_us
        if period is not None and declared is not None and period != seconds_to_microseconds(declared):
            raise ValueError("timing.decision_period_us must equal the dataset and simulator decision period")
        period = period if period is not None else (seconds_to_microseconds(declared) if declared is not None else None)
        if period is None:
            return
        validate_decision_period(timing, period)
        if (any(sensor.resolved_frame_interval_us(period) != period for sensor in timing.sensors.values())
                and not capabilities.captures_between_decisions):
            raise ValueError(f"{type(self).__name__} cannot acquire between decisions")
        tick = capabilities.base_tick_us
        if tick is not None:
            values = [period, timing.pose_reporting_interval_us, timing.planner_delay_us,
                      timing.force_gt_duration_us,
                      *(sensor.resolved_frame_interval_us(period) for sensor in timing.sensors.values())]
            if any(value % tick for value in values):
                raise ValueError("Frame, reporting, planner delay and force-GT periods must align with simulator base ticks")

    def _validate_run_start(self, start: ScenarioStart | None) -> None:
        if start is not None and not self.can_start_runs_after_origin and start.frame.position > 0:
            raise UnsupportedStartError(
                f"{type(self).__name__} can start closed-loop runs only at the scenario origin")

    def validate_startup(self, start: ScenarioStart | None, duration_us: int) -> None:
        """Check the closed-loop start and recorded-motion coverage for force_gt_duration_us.

        A zero duration needs no recording. Missing support raises ``ValueError`` and
        a disallowed start or insufficient recorded motion raises ``UnsupportedStartError``.
        """
        self._validate_run_start(start)
        require_nonnegative_integer(duration_us, "startup duration")
        if not duration_us:
            return
        if not self.timing_capabilities.recorded_ego_motion:
            raise ValueError(f"{type(self).__name__} does not support recorded ego startup motion")
        if not isinstance(start, ScenarioStart):
            raise ValueError("force_gt_duration_us requires a recorded ScenarioStart")
        required_end_us = timestamp_to_microseconds(start.frame.timestamp) + duration_us
        if self._recorded_end_time_us(start) < required_end_us:
            raise UnsupportedStartError("Recorded ego motion does not cover the force-GT interval")

    def _recorded_end_time_us(self, start: ScenarioStart) -> int:
        period = start.payload.get("time_step", self.decision_time_step)
        if not start.recorded_ego or period is None:
            raise UnsupportedStartError("Force-GT interval requires recorded ego motion and its time grid")
        remaining = len(start.recorded_ego) - 1 - start.frame.position
        return timestamp_to_microseconds(start.frame.timestamp) + remaining * seconds_to_microseconds(period)


class SimulatorRuntime(_HookBindingMixin, ABC):
    """One live world of a ``Simulator``: world state, clock, actors, sensors and sensor views.

    A runtime is created by ``Simulator.create_runtime`` and closed by its creator, also as a
    context manager. ``reset`` starts a run and may be called again for a further run on the
    same world. Declarations come from ``simulator``.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if {"reset", "step", "advance_interval", "capture_at", "sensor_view", "sensor_geometry", "start_replay", "advance_replay", "observe_recorded_at"} & cls.__dict__.keys():
            raise TypeError("SimulatorRuntime subclasses implement native operations, "
                            "not reset/step/sensor_view/start_replay/advance_replay/observe_recorded_at")

    def __init__(self, simulator: Simulator) -> None:
        self.simulator = simulator
        self._timing = TimingConfiguration()
        self._history_requirement = HistoryRequirement()
        self._state_cache: WorldSnapshot | None = None
        self._capture: Observation | None = None
        self._clock_time_us: int | None = None
        self._interval: DecisionInterval | None = None
        self._published_state: WorldSnapshot | None = None
        self._sensor_slots: dict[str, _SensorSlot] = {}
        self._objects: dict[str, _InsertedObject] = {}
        self._capture_object_records: dict[str, _InsertedObject] = {}
        self._capture_objects: dict[str, PlacedObject] = {}
        self._start_objects: dict[str, _InsertedObject] = {}
        self._object_origin_time_us: int | None = None
        self._object_move_interval: tuple[int, int] | None = None
        self._composited_capture: Observation | None = None
        self._object_visibility: dict[str, dict[str, ObjectVisibility]] = {}
        self._composition_error: BaseException | None = None
        self._world_initialized = False
        self._closed_loop_run = False
        self._resetting = False
        self._advancing = False
        self._start: ScenarioStart | None = None
        self._deployment = DeploymentPlan()
        self._in_setup = False
        self._setup_ticks_used = 0
        self._exclusive = False
        self._world_owner: object | None = None
        # True while generate_frame_data replays a start: the runtime keeps following the
        # start's prescribed motion instead of handing the ego to a driving system.
        self._generating_frame_data = False
        self.initialization_report: dict[str, Any] = {}

    def __enter__(self) -> SimulatorRuntime:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # Declarations of the configured simulator.
    @property
    def dataset_kinds(self) -> tuple[str, ...]:
        """Dataset kinds supported by this simulator."""
        return self.simulator.dataset_kinds

    @property
    def can_render(self) -> bool:
        """Whether arbitrary-view rendering is supported."""
        return self.simulator.can_render

    @property
    def can_replay(self) -> bool:
        """Whether recorded-tick replay is supported."""
        return self.simulator.can_replay

    @property
    def can_observe_past(self) -> bool:
        """Whether observations before a selected start can be reconstructed."""
        return self.simulator.can_observe_past

    @property
    def can_insert_objects(self) -> bool:
        """Whether world objects can be inserted into sensor rendering."""
        return self.simulator.can_insert_objects

    @property
    def closed_loop_run(self) -> bool:
        """Whether the current reset is for a closed-loop run, until reset or close."""
        return self._closed_loop_run

    @property
    def supported_world_settings(self) -> frozenset[str]:
        """World-setting keys accepted by the simulator."""
        return self.simulator.supported_world_settings

    @property
    def capture_ticks(self) -> int:
        """Simulator ticks consumed by a sensor capture."""
        return self.simulator.capture_ticks

    @property
    def decision_time_step(self) -> float | None:
        """Decision interval in seconds. None leaves the interval unrestricted."""
        return self.simulator.decision_time_step

    @property
    def command_modes(self) -> frozenset[str] | None:
        """Control command modes this simulator runtime accepts, or None when undeclared."""
        return self.simulator.command_modes

    @property
    def supported_stages(self) -> frozenset[str]:
        return self.simulator.supported_stages

    @property
    def message_stages(self) -> Mapping[str, MessageStage]:
        """Native message stages declared by the configured simulator."""
        return self.simulator.message_stages

    @property
    def message_channel(self) -> MessageChannel | None:
        """Transport for native message stages in this open run, if configured."""
        return None

    @property
    def message_snapshot(self) -> tuple[int, WorldSnapshot] | None:
        """Latest completed native step's clock and detached state for message processing."""
        return None

    @property
    def message_sensor_poses(self) -> Mapping[str, np.ndarray]:
        """Fixed ego-relative transforms of the native sensors used by message stages."""
        return {}

    @property
    def identity(self) -> Mapping[str, Any]:
        """Configuration identity used to check saved-preparation compatibility."""
        return self.simulator.identity

    @property
    def termination_details(self) -> Mapping[str, Any]:
        """Simulator diagnostics saved with the run's termination event."""
        return {}

    def decision_telemetry(self) -> Mapping[str, Any]:
        """Native evidence recorded at each decision without issuing a driving command."""
        return {}

    def interval_telemetry(self) -> Mapping[str, Any]:
        """Native evidence from a completed decision interval, read after endpoint capture."""
        return {}

    def validate_start(self, start: ScenarioStart | None) -> None:
        """Check dataset and simulator constraints for a recorded scenario start."""
        self.simulator.validate_start_timing(start, self._timing)

    def validate_command(self, command: Control) -> None:
        """Reject active command fields that this simulator would silently ignore."""
        validate_command_modes(command, self.command_modes)

    def copy_observation(self, observation: Observation) -> Observation:
        """Detach the complete mutable data graph before model-visible processing.

        Native sensor captures that cannot be copied require a runtime override documenting
        which buffers are borrowed and how to replace them safely. Do not copy SDK handles.
        """
        return deepcopy(observation)

    def interpolate_vehicle_state(self, left: Any, right: Any, fraction: float, time_us: int) -> Any:
        """Interpolate model-visible actor, rigid-matrix or planar dataclass state.

        Translation and numeric motion fields interpolate linearly. Rotations take the shortest
        arc. Extensions override this operation for their native vehicle-state representation.
        """
        import numpy as np

        if type(left) is not type(right):
            raise TypeError("Pose reports must use the same vehicle-state representation")
        if left is None:
            return None
        if isinstance(left, ActorState):
            velocity = (None if left.velocity is None or right.velocity is None else
                        tuple((1 - fraction) * a + fraction * b for a, b in zip(left.velocity, right.velocity)))
            return replace(left, pose=interpolate_transform(left.pose, right.pose, fraction), velocity=velocity)
        if isinstance(left, np.ndarray) and left.shape == (4, 4):
            return interpolate_transform(left, right, fraction)
        if is_dataclass(left) and not isinstance(left, type) and all(
                hasattr(left, name) for name in ("x", "y", "yaw")):
            values: dict[str, Any] = {}
            for item in fields(left):
                a, b = getattr(left, item.name), getattr(right, item.name)
                if isinstance(a, Real) and isinstance(b, Real) and not isinstance(a, bool):
                    values[item.name] = float(a) + fraction * (float(b) - float(a))
            first_yaw, last_yaw = float(getattr(left, "yaw")), float(getattr(right, "yaw"))
            values["yaw"] = first_yaw + fraction * math.atan2(math.sin(last_yaw - first_yaw),
                                                            math.cos(last_yaw - first_yaw))
            if hasattr(left, "t"):
                values["t"] = time_us / 1e6
            return replace(left, **values)
        raise TypeError(f"Cannot interpolate vehicle-state type {type(left).__name__}")

    def close(self) -> None:
        """Release the world and native connections. Optional."""
        self._closed_loop_run = False
        self._state_cache = None
        self._objects.clear()
        self._capture_object_records.clear()
        self._capture_objects.clear()
        self._start_objects.clear()
        self._object_origin_time_us = None
        self._object_move_interval = None
        self._object_visibility.clear()
        self._composited_capture = None
        self._composition_error = None
        self._capture = None
        self._clock_time_us = None
        self._interval = None
        self._published_state = None
        self._sensor_slots.clear()
        self._world_initialized = False

    @property
    def world_key(self) -> Hashable:
        """Identify the mutable world owned by this runtime.

        Runtimes that affect the same external world must return the same key. Independent
        in-process worlds use the default key unique to this runtime.
        """
        return ("simulator", id(self))

    @contextmanager
    def claim_world(self, owner: object) -> Iterator[None]:
        """Claim the world for an owner until exit, allowing that owner's nested claims.

        Independent owners fail immediately. The caller must close native resources before
        releasing the claim. Claims coordinate only runtimes in this process.
        """
        previous = self._world_owner
        with _claim_world(self.world_key, owner):
            self._world_owner = owner
            try:
                yield
            finally:
                self._world_owner = previous

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Claim the world and allow only one operation on this runtime at a time.

        Unlike ``claim_world``, this also rejects nested operations on the same runtime,
        even when they have the same owner.
        """
        if self._exclusive:
            raise RuntimeError("Simulator already has an active operation")
        owner = self._world_owner if self._world_owner is not None else self
        with _claim_world(self.world_key, owner):
            self._exclusive = True
            try:
                yield
            finally:
                self._exclusive = False

    @property
    def clock(self) -> ClockSnapshot:
        """Return the current ``ClockSnapshot`` containing simulator time and native frame."""
        if not (self._world_initialized or self._resetting or self._advancing):
            raise RuntimeError("Reset the simulator before requesting its clock")
        reported = self._reported_interval_state()
        native = (self._native_clock() if reported is None
                  else ClockSnapshot(reported.t, reported.frame))
        return replace(native, scheduled_time_us=self._clock_time_us)

    def _reported_interval_state(self) -> WorldSnapshot | None:
        """Latest reported state of a pending interval, read outside reset and advance."""
        if self._interval is None or self._advancing or self._resetting:
            return None
        return self._published_state

    def _native_clock(self) -> ClockSnapshot:
        """Read native time and frame without copying a full world snapshot when supported."""
        state = self._cached_state()
        return ClockSnapshot(state.t, state.frame)

    def _restore_clock(self, time_us: int) -> None:
        """Restore a checkpoint clock and discard interval and acquisition state."""
        self._interval = None
        self._published_state = None
        self._sensor_slots.clear()
        self._capture_object_records.clear()
        self._capture_objects.clear()
        self._capture = None
        self.invalidate_state()
        self._clock_time_us = time_us

    def invalidate_state(self) -> None:
        """Discard cached physical truth after explicitly changing the live world.

        Built-in world mutation methods call this automatically. Native extensions that mutate
        a simulator resource directly must call it before reading the changed physical state.
        """
        self._state_cache = None

    def _cached_state(self) -> WorldSnapshot:
        if self._state_cache is None:
            self._state_cache = self._state()
        return self._state_cache

    def state(self) -> WorldSnapshot:
        """Read detached physical state without a capture, callback or simulator tick.

        Available after reset and during reset and step, so stage handlers can read it.
        """
        if not (self._world_initialized or self._resetting or self._advancing):
            raise RuntimeError("Reset the simulator before requesting its state")
        objects = {}
        if self._objects:
            if self._object_origin_time_us is None:
                # Setup insertions have no time anchor until the origin capture.
                time_us = 0
            elif self._advancing and self._interval is not None:
                time_us = self._interval.start_us
            else:
                time_us = self.clock.time_us
            objects = {identity: item.state_at(time_us) for identity, item in self._objects.items()}
        reported = self._reported_interval_state()
        state = deepcopy(self._cached_state() if reported is None else reported)
        return replace(state, objects=objects)

    @property
    def timing_capabilities(self) -> TimingCapabilities:
        """Static timing limits declared by this runtime's configured simulator."""
        return self.simulator.timing_capabilities

    @property
    def timing(self) -> TimingConfiguration:
        """Timing configuration selected by ``configure_timing`` for this runtime."""
        return self._timing

    @property
    def history_requirement(self) -> HistoryRequirement:
        """Recorded context selected by ``configure_timing`` for this runtime."""
        return self._history_requirement

    def _state_at(self, time_us: int) -> WorldSnapshot:
        """Read detached truth within the last completed motion interval without advancing."""
        if isinstance(time_us, bool) or not isinstance(time_us, int):
            raise ValueError("State time must be integer microseconds")
        self._validate_native_time(time_us)
        if time_us == self.clock.time_us:
            return self.state()
        interval = self._interval
        if interval is None or not interval.start_us <= time_us <= interval.end_us:
            raise ValueError("State time lies outside the last advanced interval")
        if time_us == interval.start_us:
            return deepcopy(interval.start_state)
        if time_us == interval.end_us and not self.timing_capabilities.motion_reports:
            return deepcopy(self._cached_state())
        if not self.timing_capabilities.motion_reports:
            raise ValueError(f"{type(self).__name__} cannot report intermediate motion")
        return self._reported_state_at(time_us)

    def _validate_native_time(self, time_us: int) -> None:
        tick = self.timing_capabilities.base_tick_us
        origin = self._interval.start_us if self._interval is not None else self.clock.time_us
        if tick is not None and (time_us - origin) % tick:
            raise ValueError("Acquisition and state times must align with simulator base ticks")

    def _reported_state_at(self, time_us: int) -> WorldSnapshot:
        raise NotImplementedError("This simulator does not report intermediate motion")

    def _activate_reported_state(self, time_us: int, state: WorldSnapshot) -> None:
        """Select a reported pose for acquisition without updating the controller."""
        if self._interval is None or time_us != self._interval.end_us:
            raise NotImplementedError("This simulator cannot acquire intermediate reports")

    @property
    def sensors(self) -> frozenset[str]:
        """All sensor IDs for which this runtime can provide a current SensorView."""
        return frozenset()

    @property
    def camera_sensors(self) -> frozenset[str]:
        """Explicit camera IDs whose current physical geometry can be recorded."""
        return frozenset()

    def sensor_geometry(self, sensor: str) -> SensorGeometry:
        """Read geometry of the sensor's latest acquisition, or native geometry before capture."""
        sensor = self.canonical_sensor_id(sensor)
        slot = self._sensor_slots.get(sensor)
        if slot is not None:
            if slot.geometry is None:
                raise ValueError(f"Sensor {sensor!r} has no camera geometry")
            return deepcopy(slot.geometry)
        return self._sensor_geometry(sensor)

    def _sensor_geometry(self, sensor: str) -> SensorGeometry:
        """Native optical geometry without image or depth work when supported.

        Implementations override this for projection-only requests before an acquisition.
        """
        view = self._sensor_view(sensor)
        if view.camera is None or view.cam_from_world is None:
            raise ValueError(f"Sensor {sensor!r} has no optical camera geometry")
        return SensorGeometry(view.sensor, view.camera, deepcopy(view.cam_from_world))

    def sensor_view(self, sensor: str) -> SensorView:
        """Return a detached view of the sensor's latest capture without acquiring again."""
        slot = self._current_slot(sensor)
        self._compose_slot(slot)
        return self._detached_view(self._slot_view(slot))

    def canonical_sensor_id(self, sensor: str) -> str:
        """Resolve a native sensor name or configured alias to its observation ID."""
        resolved = sensor if sensor in self.sensors else getattr(
            self.simulator, "camera_aliases", {}).get(sensor, sensor)
        if resolved not in self.sensors:
            raise ValueError(f"Unknown sensor {sensor!r}")
        return resolved

    def frame_range_us(self, sensor: str, end_time_us: int) -> range:
        """Read a frame's shutter or native scan interval at its integer end timestamp."""
        sensor = self.canonical_sensor_id(sensor)
        if isinstance(end_time_us, bool) or not isinstance(end_time_us, int):
            raise ValueError("Frame end must be integer microseconds")
        return self._frame_range_us(sensor, end_time_us)

    def _frame_range_us(self, sensor: str, end_time_us: int) -> range:
        timing = self._timing.sensors.get(sensor, SensorTiming())
        return range(end_time_us - timing.shutter_duration_us, end_time_us)

    def _current_slot(self, sensor: str) -> _SensorSlot:
        if not self._world_initialized:
            raise RuntimeError("Reset the simulator before requesting sensor views")
        sensor = self.canonical_sensor_id(sensor)
        slot = self._sensor_slots.get(sensor)
        if slot is None:
            if self._capture is None or sensor not in self._capture.sensor_data:
                raise ValueError(f"Unknown or not yet acquired sensor {sensor!r}")
            slot = self._make_sensor_slot(sensor)
        return slot

    def _latest_render_request(self, sensor: str) -> Any:
        """Detach the latest acquired native request without building a sensor view."""
        sensor = self.canonical_sensor_id(sensor)
        if sensor not in self._sensor_slots:
            raise KeyError(f"No current capture for sensor {sensor!r}")
        return deepcopy(self._sensor_slots[sensor].render_request)

    def _detached_view(self, view: SensorView) -> SensorView:
        # Frozen placed objects can own harmonizer resources and remain shared.
        shared = {id(placed): placed for placed in view.objects.values()}
        return deepcopy(view, shared)

    def _make_sensor_slot(self, sensor: str, truth: WorldSnapshot | None = None) -> _SensorSlot:
        assert self._capture is not None
        observation = replace(self._capture, sensor_data={sensor: self._capture.sensor_data[sensor]},
                              calibration={sensor: self._capture.calibration[sensor]}
                              if sensor in self._capture.calibration else {})
        truth = self.state() if truth is None else truth
        geometry = self._sensor_geometry(sensor) if sensor in self.camera_sensors else None
        time_us = ClockSnapshot(truth.t, truth.frame, self._clock_time_us).time_us
        slot = _SensorSlot(time_us, observation, truth,
                           self._capture_view_builder(sensor, observation, truth), geometry,
                           dict(self._capture_objects), self._capture_render_request(sensor))
        self._sensor_slots[sensor] = slot
        return slot

    def _capture_render_request(self, sensor: str) -> Any:
        """Freeze optional renderer metadata independently of a full sensor view."""
        return None

    def _capture_view_builder(self, sensor: str, observation: Observation,
                              truth: WorldSnapshot) -> Callable[[], SensorView]:
        """Freeze native view inputs now. The returned builder must not read live state."""
        view = self._sensor_view(sensor)
        return lambda: view

    @staticmethod
    def _slot_view(slot: _SensorSlot) -> SensorView:
        if slot.view is None:
            slot.view = slot.build_view()
        return slot.view

    def _sensor_view(self, sensor: str) -> SensorView:
        """Native capture and geometry, before framework object composition."""
        raise NotImplementedError(f"{type(self).__name__} does not provide sensor views")

    def _supports_asset(self, asset: ObjectAsset | MeshAsset) -> bool:
        """Whether this runtime can render the supplied world-object asset."""
        from avsectester.rendering import ObjectAsset

        return isinstance(asset, ObjectAsset)

    def _require_object_insertion(self) -> None:
        if not self.can_insert_objects:
            raise NotImplementedError(f"{type(self.simulator).__name__} cannot insert objects")

    def _objects_at(self, time_us: int) -> dict[str, PlacedObject]:
        """Evaluate the live world's inserted objects at a scheduled time."""
        return {identity: item.placed_at(time_us) for identity, item in self._objects.items()}

    def _capture_objects_at(self, time_us: int) -> dict[str, PlacedObject]:
        """Evaluate the current batch's inserted objects at a second time during acquisition."""
        return {identity: item.placed_at(time_us)
                for identity, item in self._capture_object_records.items()}

    def _start_objects_at(self, time_us: int) -> dict[str, PlacedObject]:
        """Evaluate the objects present at the origin using their initial motion."""
        return {identity: item.placed_at(time_us) for identity, item in self._start_objects.items()}

    def _recorded_object_time_us(self, time_us: int) -> int:
        """Convert a recorded offset relative to the origin to the runtime object clock."""
        assert self._object_origin_time_us is not None
        return self._object_origin_time_us + time_us

    def insert_object(self, object_id: str, asset: ObjectAsset | MeshAsset, world_from_object: np.ndarray, *,
                      harmonizer: Harmonizer | None = None,
                      velocity: Sequence[float] | np.ndarray = (0., 0., 0.)) -> None:
        """Register a physical world object, visible from the next sensor capture.

        Velocity is a finite world-frame three-vector in meters per second. It describes motion
        up to the insertion time, or up to the origin capture for objects inserted during reset.
        """
        from avsectester.rendering import PlacedObject

        self._require_object_insertion()
        if not (self._world_initialized or self._resetting or self._advancing):
            raise RuntimeError("Reset the simulator before inserting objects")
        if not self._supports_asset(asset):
            raise TypeError(
                f"{type(self.simulator).__name__} does not support asset type {type(asset).__name__}")
        if harmonizer is not None and not callable(harmonizer):
            raise TypeError("An object harmonizer must be callable or None")
        placed = PlacedObject(object_id, asset, world_from_object, harmonizer)
        if object_id in self._objects:
            raise ValueError(f"Object {object_id!r} is already inserted")
        vx, vy, vz = map(float, validated_vector(velocity, "velocity"))
        item = _InsertedObject(placed, (vx, vy, vz))
        if self._object_origin_time_us is not None:
            item = item.anchored_at(self.clock.time_us)
        self._objects[object_id] = item
        self.invalidate_state()

    def move_object(self, object_id: str, world_from_object: np.ndarray) -> None:
        """Set an inserted object's end pose once during ``simulator.step.pre``.

        Raises ``NotImplementedError`` when object insertion is unsupported, ``RuntimeError``
        outside this stage, ``KeyError`` for an unknown object and ``ValueError`` for an invalid
        pose or another move of the same object in this interval.
        """
        self._require_object_insertion()
        if self._object_move_interval is None:
            raise RuntimeError("move_object is only available during simulator.step.pre")
        start_us, end_us = self._object_move_interval
        if object_id not in self._objects:
            raise KeyError(f"Unknown inserted object {object_id!r}")
        item = self._objects[object_id]
        nodes = item.nodes
        if nodes[-1][0] == end_us:
            raise ValueError(f"Object {object_id!r} already has a move in this interval")
        if nodes[-1][0] < start_us:
            nodes += ((start_us, item.pose_at(start_us)),)
        self._objects[object_id] = replace(item, nodes=(*nodes, (end_us, validated_transform(world_from_object))))
        self.invalidate_state()

    def remove_object(self, object_id: str) -> None:
        """Remove a world object. The previous capture remains unchanged."""
        self._require_object_insertion()
        if object_id not in self._objects:
            raise KeyError(f"Unknown inserted object {object_id!r}")
        del self._objects[object_id]
        self.invalidate_state()

    def object_visibility(self, object_id: str) -> Mapping[str, ObjectVisibility]:
        """Visibility of an object in the latest capture, for every camera whose image contains it.

        A camera whose image contains the object's projection has an entry even when actors or
        scene depth hide every pixel (``visible_pixels == 0``). Raises ``KeyError`` for an object
        that is not part of the latest capture, and re-raises a failed composition.
        """
        if not self._world_initialized or self._capture is None:
            raise RuntimeError("Reset the simulator before reading object visibility")
        if object_id not in self._capture_objects:
            if object_id in self._objects:
                raise KeyError(f"Object {object_id!r} was inserted after the latest capture. "
                               "It is visible from the next capture")
            raise KeyError(f"Unknown inserted object {object_id!r}")
        self._compose_capture()
        return deepcopy(self._object_visibility[object_id])

    def _snapshot_capture_objects(self) -> None:
        # Render hooks may edit the live world during native acquisition.
        self._capture_object_records = dict(self._objects)
        self._capture_objects = self._objects_at(self.clock.time_us) if self._objects else {}

    def _take_capture(self, sensor_ids: Sequence[str] | None = None,
                      truth: WorldSnapshot | None = None) -> None:
        """Acquire native sensor data once and snapshot the objects visible to that capture."""
        self._snapshot_capture_objects()
        self._composited_capture = None
        self._object_visibility = {identity: {} for identity in self._capture_objects}
        self._composition_error = None
        self._capture = None
        self._capture = self._observe() if sensor_ids is None else self._observe_sensors(sensor_ids)
        truth = self.state() if truth is None else truth
        for sensor in self._capture.sensor_data.keys() & self.sensors:
            self._make_sensor_slot(sensor, truth)

    def _compose_slot(self, slot: _SensorSlot) -> None:
        if slot.composed:
            return
        if slot.error is not None:
            raise slot.error
        if not slot.objects or slot.geometry is None:
            slot.composed = True
            return
        try:
            geometry = slot.geometry
            sensor = geometry.sensor
            observation, views, visibility = compose_capture(
                slot.observation, slot.objects, (sensor,), lambda sensor: geometry,
                lambda sensor: self._slot_view(slot), self._compose_objects)
            slot.observation = observation
            slot.view = views.get(sensor, slot.view)
            slot.visibility = {identity: values[sensor] for identity, values in visibility.items()
                               if sensor in values}
            slot.composed = True
        except BaseException as error:
            slot.error = error
            raise

    def _observe_sensors(self, sensor_ids: Sequence[str]) -> Observation:
        observation = self._observe()
        selected = set(sensor_ids)
        return replace(observation,
                       sensor_data={key: value for key, value in observation.sensor_data.items() if key in selected},
                       calibration={key: value for key, value in observation.calibration.items() if key in selected})

    def _compose_capture(self) -> None:
        if self._composited_capture is not None:
            return
        if self._composition_error is not None:
            raise self._composition_error
        assert self._capture is not None
        try:
            observation = self._capture
            for sensor in sorted(observation.sensor_data.keys() & self.sensors):
                slot = self._sensor_slots.get(sensor) or self._make_sensor_slot(sensor)
                self._compose_slot(slot)
                observation = replace(observation, sensor_data={
                    **observation.sensor_data, sensor: slot.observation.sensor_data[sensor]})
                for identity, visibility in slot.visibility.items():
                    self._object_visibility.setdefault(identity, {})[sensor] = visibility
            self._composited_capture = observation
        except BaseException as error:
            self._composition_error = error
            raise

    def control_actor(self, actor_id: str, control: Control) -> None:
        """Drive another actor with an actuator command from the next world step.

        The actor stops following its prescribed or traffic-manager motion for the rest of the
        run. Call it from a ``WorldAttackAction``. The ego belongs to the driving system.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot control actors")

    def set_actor_state(self, actor_id: str, state: ActorState) -> None:
        """Move another actor to a canonical world state. It is visible from the next capture.

        The actor stops following its prescribed or traffic-manager motion for the rest of the
        run. Call it from a ``WorldAttackAction``. The ego belongs to the driving system.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot set actor states")

    def reset(self, start: ScenarioStart | None = None, initialization: Any = None, *,
              deployment: DeploymentPlan | None = None,
              world: Mapping[str, Any] | None = None, closed_loop_run: bool = False) -> None:
        """Initialize the world at ``start`` or the configured origin and capture the first frame.

        The order is native initialization, pre-roll to the deployment window, simulator.setup
        and its deployment ticks, the capture ticks, then the first capture. Read it with
        ``observe``. With ``closed_loop_run=True``, reject unsupported run starts before opening
        the world and expose the run phase to setup hooks.
        """
        if start is None and initialization is not None:
            raise ValueError("An initialization plan requires ScenarioStart")
        self.validate_start(start)
        if closed_loop_run:
            self.simulator._validate_run_start(start)
        # The simulator owns only native world settings. Sensor environments belong to
        # the runner/services after capture. Rejecting envelopes avoids silently losing them.
        world_environment = dict(world or {})
        self.simulator.validate_world_settings(world_environment)
        plan = deployment if deployment is not None else DeploymentPlan()
        position = self._replay_position(start)
        # The deployment window ends at the selected capture: participant ticks plus the
        # simulator's capture ticks. It is carved out of the pre-roll, never appended to it.
        window = plan.ticks + self.capture_ticks
        if not can_deploy_at(position, plan, self.capture_ticks):
            raise ValueError("Selected boundary is earlier than its deployment window")
        self.invalidate_state()
        self._world_initialized = False
        self._clock_time_us = None
        self._interval = None
        self._published_state = None
        self._sensor_slots.clear()
        self._capture = None
        self._objects.clear()
        self._capture_object_records.clear()
        self._capture_objects.clear()
        self._start_objects.clear()
        self._object_origin_time_us = None
        self._object_move_interval = None
        self._composited_capture = None
        self._object_visibility.clear()
        self._composition_error = None
        self.initialization_report = {}
        self._start, self._deployment = start, plan
        self._closed_loop_run = closed_loop_run
        self._resetting = True
        try:
            self._initialize_world(start, initialization, world_environment)
            for _ in range(max(0, position - window)):
                self._replay_tick()
            self._setup_ticks_used = 0
            self._in_setup = True
            try:
                self._emit("simulator.setup", None)
                while self._setup_ticks_used < plan.ticks:
                    before = self._setup_ticks_used
                    self.setup_tick()
                    if self._setup_ticks_used != before + 1:
                        raise RuntimeError("setup_tick must consume exactly one deployment tick")
            finally:
                self._in_setup = False
            for _ in range(self.capture_ticks):
                self._window_tick(position)
            self._before_selected_capture(position)
            self.invalidate_state()
            origin_us = self.clock.time_us
            self._object_origin_time_us = origin_us
            self._objects = {identity: item.anchored_at(origin_us)
                             for identity, item in self._objects.items()}
            self._take_capture()
            self._start_objects = dict(self._capture_object_records)
            self._world_initialized = True
            try:
                self._check_initialization(initialization)
            except BaseException:
                self._world_initialized = False
                self._capture = None
                raise
        finally:
            self._resetting = False

    def setup_tick(self) -> None:
        """Advance one setup tick without requesting a driving decision."""
        if not self._in_setup:
            raise RuntimeError("setup_tick is only available during simulator.setup")
        if self._setup_ticks_used >= self._deployment.ticks:
            raise RuntimeError("Deployment participant exceeded the shared tick budget")
        self._window_tick(self._replay_position(self._start))
        self._setup_ticks_used += 1

    def step(self, control: Control | None) -> None:
        """Advance one decision and capture through the scheduled simulator operations.

        External-control runtimes accept None, leaving ego control to the external system.
        """
        endpoint = self.clock.time_us + self.decision_period_us
        self.advance_interval(endpoint, control)
        self._acquire_at(endpoint)

    @property
    def decision_period_us(self) -> int:
        """Declared decision interval in integer microseconds."""
        period = self.simulator.decision_time_step
        if period is None and self._start is not None:
            period = self._start.payload.get("time_step")
        if period is None:
            raise ValueError("Simulator must declare a decision interval")
        return seconds_to_microseconds(period)

    def configure_timing(self, timing: TimingConfiguration, *,
                         history: HistoryRequirement = HistoryRequirement()) -> None:
        """Set acquisition clocks before resetting a run. Never advance a live world.

        Rates must fit the simulator's native tick. Decision period remains the dataset grid.
        Sensor IDs and loaded scan windows are checked after reset when they are available.
        """
        self.simulator.validate_timing(timing)
        self._timing = timing
        self._history_requirement = history

    def advance_interval(self, time_us: int, control: Control | None, *,
                   report_times_us: Sequence[int] = (), force_gt: bool = False,
                   coerce_dynamic_state: bool = False) -> None:
        """Propagate one complete decision interval, retaining reports for on-demand reads.

        The controller runs once. The published clock remains at the last acquisition.
        ``capture_at`` publishes subsequent report times in order. Simulator step hooks
        run before propagation and when its endpoint is published, before acquisition.
        ``control=None`` follows recorded ego motion unless ``command_modes`` is empty,
        in which case the external driving system controls propagation.
        """
        if isinstance(time_us, bool) or not isinstance(time_us, int):
            raise ValueError("Advance time must be integer microseconds")
        if not self._world_initialized:
            raise RuntimeError("Reset the simulator before stepping")
        if self._interval is not None:
            raise RuntimeError("Publish the previous interval endpoint before propagating again")
        follows_recording = force_gt or (control is None and not is_external_control(self.command_modes))
        if follows_recording:
            if not self.timing_capabilities.recorded_ego_motion:
                raise ValueError(f"{type(self.simulator).__name__} cannot advance recorded ego motion")
        if control is not None:
            self.validate_command(control)
        if time_us != self.clock.time_us + self.decision_period_us:
            raise ValueError("advance_interval must end at the next decision time")
        origin_us = self.clock.time_us
        for report_time in report_times_us:
            if (isinstance(report_time, bool) or not isinstance(report_time, int)
                    or not origin_us <= report_time <= time_us):
                raise ValueError("Report times must lie within the decision interval")
            if report_time not in (origin_us, time_us) and not self.timing_capabilities.motion_reports:
                raise ValueError(f"{type(self).__name__} cannot report intermediate motion")
            tick = self.timing_capabilities.base_tick_us
            if tick is not None and (report_time - origin_us) % tick:
                raise ValueError("Motion report times must align with simulator base ticks")
        self._capture = None
        self._world_initialized = False
        self._advancing = True
        try:
            # Only simulator.step.pre can set object poses at this interval's end.
            self._object_move_interval = (origin_us, time_us)
            try:
                self._emit("simulator.step.pre", None)
            finally:
                self._object_move_interval = None
            self._interval = DecisionInterval(origin_us, time_us,
                tuple(sorted(set(report_times_us) - {origin_us, time_us})),
                force_gt, coerce_dynamic_state, self.state())
            if follows_recording:
                self._advance_force_gt_native(control)
            else:
                self._advance_native(control)
            self.invalidate_state()
            end_state = self._completed_state()
            self._state_cache = end_state
            self._completed_interval(end_state)
        finally:
            self._advancing = False
        self._world_initialized = True
        self._published_state = deepcopy(self._interval.start_state)

    def _completed_state(self) -> WorldSnapshot:
        return self.state()

    def _advance_force_gt_native(self, control: Control | None) -> None:
        self._advance_recorded_native()

    def recorded_reference(self, time_us: int, end_us: int) -> Control:
        """Recorded reference in the rig frame at its production time, for force-GT control."""
        raise NotImplementedError("This simulator has no recorded trajectory command")

    def _advance_recorded_native(self) -> None:
        raise NotImplementedError("This simulator cannot advance recorded ego motion")

    def capture_at(self, time_us: int, sensor_ids: Sequence[str] | None = None) -> Observation:
        """Acquire sensors at the current integer time without advancing the world.

        Repeated reads reuse the same acquisition. A sensor selection returns only those
        payloads and their calibrations. The full acquisition remains available to views.
        """
        if sensor_ids is not None:
            sensor_ids = tuple(self.canonical_sensor_id(sensor) for sensor in sensor_ids)
            unknown = set(sensor_ids) - self.sensors
            if unknown:
                raise ValueError(f"Unknown capture sensors: {sorted(unknown)}")
        self._preflight_capture(time_us, sensor_ids)
        self._acquire_at(time_us, sensor_ids)
        observation = self.observe()
        if sensor_ids is None:
            return observation
        selected = set(sensor_ids)
        return replace(observation,
                       sensor_data={key: value for key, value in observation.sensor_data.items() if key in selected},
                       calibration={key: value for key, value in observation.calibration.items() if key in selected})

    def _preflight_capture(self, time_us: int, sensor_ids: Sequence[str] | None) -> None:
        """Validate native availability before publishing or replacing an acquisition."""
        if isinstance(time_us, bool) or not isinstance(time_us, int):
            raise ValueError("Capture time must be integer microseconds")
        self._validate_native_time(time_us)
        if (self._interval is not None and time_us != self._interval.end_us and sensor_ids != ()
                and not self.timing_capabilities.captures_between_decisions):
            raise ValueError(f"{type(self).__name__} cannot acquire between decisions")

    def _acquire_at(self, time_us: int, sensor_ids: Sequence[str] | None = None) -> None:
        if isinstance(time_us, bool) or not isinstance(time_us, int):
            raise ValueError("Capture time must be integer microseconds")
        if not self._world_initialized:
            raise RuntimeError("Reset the simulator before capturing")
        self._validate_native_time(time_us)
        if time_us < self.clock.time_us:
            raise ValueError("Capture time cannot precede the current time")
        endpoint = self._interval.end_us if self._interval is not None else None
        endpoint_truth: WorldSnapshot | None = None
        if self._interval is not None and (time_us != self.clock.time_us or self._capture is None):
            if time_us <= self.clock.time_us or time_us > self._interval.end_us:
                raise ValueError("Capture time must advance within the pending interval")
            capability = (self.timing_capabilities.motion_reports if sensor_ids == ()
                          else self.timing_capabilities.captures_between_decisions)
            if time_us != self._interval.end_us and not capability:
                raise ValueError(f"{type(self).__name__} cannot acquire between decisions")
            reported = self._state_at(time_us)
            self._activate_reported_state(time_us, reported)
            self._published_state = reported
            self._clock_time_us = time_us
            self._capture = None
            if time_us == self._interval.end_us:
                self._interval = None
                self._published_state = None
                try:
                    self._emit("simulator.step.post", None)
                except BaseException:
                    self._world_initialized = False
                    raise
                self.invalidate_state()
                endpoint_truth = self.state()
                self._state_cache = endpoint_truth
            self._capture = None
        elif time_us != self.clock.time_us:
            raise ValueError("This simulator can capture only at its current time")
        if self._capture is None:
            try:
                truth = endpoint_truth if time_us == endpoint else self.state()
                self._take_capture(sensor_ids, truth)
            except BaseException:
                self._world_initialized = False
                raise
        else:
            missing = (self.sensors if sensor_ids is None else set(sensor_ids)) - self._capture.sensor_data.keys()
            if missing:
                previous = self._capture
                self._snapshot_capture_objects()
                self._capture = self._observe_sensors(tuple(sorted(missing)))
                truth = self.state()
                for sensor in self._capture.sensor_data.keys() & self.sensors:
                    self._make_sensor_slot(sensor, truth)
                self._capture = replace(previous,
                    sensor_data={**previous.sensor_data, **self._capture.sensor_data},
                    calibration={**previous.calibration, **self._capture.calibration})
                self._composited_capture = None
                self._object_visibility = {identity: {} for identity in self._capture_objects}

    def observe(self) -> Observation:
        """Return an independent copy of the current capture. Never advance or recapture."""
        if not self._world_initialized or self._capture is None:
            raise RuntimeError("Reset the simulator before observing")
        self._compose_capture()
        assert self._composited_capture is not None
        return self.copy_observation(self._composited_capture)

    def _recorded_pose_state(self, time_us: int) -> WorldSnapshot:
        return self._recorded_state(time_us)

    def observe_recorded_at(self, time_us: int, sensor_ids: Sequence[str] | None = None) -> RecordedAcquisition:
        """Read an acquisition before the origin without advancing the live world.

        Recorded poses and actors are interpolated only by simulators declaring intermediate
        acquisition support. Start objects are present, but online actions and hooks are absent.
        Times on the dataset grid preserve the native recorded observation exactly. An empty
        ``sensor_ids`` sequence reads only the recorded pose without loading or rendering sensors.
        The returned RecordedAcquisition contains the observation, physical state, camera geometry
        and inserted-object visibility. Geometry contains only the acquired optical cameras.
        Geometry and visibility are empty for a pose-only read. Returned values are independent
        of the live world and later recorded-history reads.
        """
        if isinstance(time_us, bool) or not isinstance(time_us, int) or time_us >= 0:
            raise ValueError("Recorded acquisition time must be negative integer microseconds")
        if not self._world_initialized or self._start is None or not self.can_observe_past:
            raise NotImplementedError(f"{type(self).__name__} cannot supply recorded acquisition history")
        period = self.decision_period_us
        if sensor_ids is not None:
            sensor_ids = tuple(self.canonical_sensor_id(sensor) for sensor in sensor_ids)
        if time_us % period and not self.timing_capabilities.captures_between_decisions:
            raise ValueError(f"{type(self).__name__} cannot interpolate recorded acquisitions")
        if self._start.frame.position * period + time_us < 0:
            raise ValueError("Recorded acquisition precedes available source history")
        if sensor_ids is not None and not sensor_ids:
            state = self._recorded_pose_state(time_us)
            return RecordedAcquisition(
                Observation(state.t, state.frame, vehicle_state=deepcopy(state.vehicle_state),
                            ego_speed=state.ego_speed), deepcopy(state))
        observation = self._observe_recorded(time_us, sensor_ids)
        object_time_us = self._recorded_object_time_us(time_us)
        visibility: dict[str, dict[str, ObjectVisibility]] = {}
        if self._start_objects:
            observation, _, visibility = compose_capture(
                observation, self._start_objects_at(object_time_us),
                self.camera_sensors & observation.sensor_data.keys(),
                lambda sensor: self._recorded_sensor_geometry(time_us, sensor),
                lambda sensor: self._recorded_sensor_view(time_us, sensor, observation),
                self._compose_objects)
        if sensor_ids is not None:
            selected = set(sensor_ids)
            observation = replace(observation,
                sensor_data={key: value for key, value in observation.sensor_data.items() if key in selected},
                calibration={key: value for key, value in observation.calibration.items() if key in selected})
        state = deepcopy(self._recorded_state(time_us))
        objects = {identity: item.state_at(object_time_us)
                   for identity, item in self._start_objects.items()}
        state = replace(state, objects=objects)
        geometry_sensors = observation.sensor_data if sensor_ids is None else sensor_ids
        geometry = {sensor: deepcopy(self._recorded_sensor_geometry(time_us, sensor))
                    for sensor in geometry_sensors
                    if sensor in observation.sensor_data and sensor in self.camera_sensors}
        self._recorded_read_complete(time_us)
        return RecordedAcquisition(observation, state, geometry, visibility)

    def _recorded_read_complete(self, time_us: int) -> None:
        """Release acquisition-only caches after transferring a recorded payload."""

    def _recorded_state(self, time_us: int) -> WorldSnapshot:
        raise NotImplementedError("This simulator cannot supply recorded acquisition truth")

    def _observe_recorded(self, time_us: int, sensors: Sequence[str] | None) -> Observation:
        """Capture selected recorded sensors without hooks or changing live state."""
        raise NotImplementedError("This simulator cannot observe the recorded past")

    def _recorded_sensor_geometry(self, time_us: int, sensor: str) -> SensorGeometry:
        """Camera geometry of a recorded past capture. Required to compose start objects into it."""
        raise NotImplementedError("This simulator cannot compose world objects into recorded captures")

    def _recorded_sensor_view(self, time_us: int, sensor: str, observation: Observation) -> SensorView:
        """Camera view of a recorded past capture whose data comes from ``observation``."""
        raise NotImplementedError("This simulator cannot compose world objects into recorded captures")

    def _compose_objects(
        self, view: SensorView, objects: Sequence[PlacedObject],
    ) -> tuple[CompositeResult, ObjectComposition | None]:
        """Compose objects into a camera capture and describe the composite when applicable."""
        import numpy as np

        from avsectester.rendering import ObjectComposition

        plan = self._composite_plan(view, objects)
        background = np.asarray(view.data)
        return plan.apply(background), ObjectComposition(np.array(background, copy=True), plan)

    def _composite_plan(self, view: SensorView, objects: Sequence[PlacedObject]) -> CompositePlan:
        """
        Plan composing ``objects`` into a native capture view. Measured occlusion overrides this.
        """
        return view.composite_plan(objects)

    def start_replay(self, start: ScenarioStart, initialization: Initialization | None = None, *,
                     world: Mapping[str, Any] | None = None,
                     deployment: DeploymentPlan | None = None) -> None:
        """Reset for reading prescribed motion, without hooks or a driving system.

        Bound hooks are rejected. This initial capture is available even when ``can_replay``
        is False. Only advancing beyond it requires replay support.
        """
        if self._dispatcher is not None:
            raise RuntimeError("Replay requires a runtime without hooks")
        self._generating_frame_data = True
        try:
            self.reset(start, initialization, world=world, deployment=deployment)
        finally:
            self._generating_frame_data = False

    def advance_replay(self, count: int) -> None:
        """Advance prescribed motion and capture each frame, without hooks or a driver.

        Requires ``can_replay``, an initialized runtime and no bound hooks. A failed advance
        invalidates the current capture until the runtime is reset again.
        """
        require_nonnegative_integer(count, "replay steps")
        if not self.can_replay:
            raise NotImplementedError(f"{type(self.simulator).__name__} cannot advance replay")
        if not self._world_initialized or self._dispatcher is not None:
            raise RuntimeError("Replay requires an initialized runtime without hooks")
        if self._interval is not None:
            raise RuntimeError("Publish the pending interval endpoint before advancing replay")
        self._generating_frame_data = True
        self._clock_time_us = None
        try:
            for _ in range(count):
                self._capture = None
                self._replay_grid_step()
                self.invalidate_state()
                self._take_capture()
        except BaseException:
            self._world_initialized = False
            raise
        finally:
            self._generating_frame_data = False

    def _replay_position(self, start: ScenarioStart | None) -> int:
        return 0

    def _window_tick(self, position: int) -> None:
        """Advance one deployment-window tick: continue replay, or hold a p=0/plain origin."""
        if position:
            self._replay_tick()
        else:
            self._hold_tick()
        self.invalidate_state()

    @abstractmethod
    def _initialize_world(self, start: ScenarioStart | None, initialization: Any,
                          environment: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def _advance_native(self, control: Control | None) -> None: ...

    def _completed_interval(self, end_state: WorldSnapshot) -> None:
        """Keep native motion reports needed by later acquisitions."""

    @abstractmethod
    def _observe(self) -> Observation: ...

    @abstractmethod
    def _state(self) -> WorldSnapshot:
        """Return an independent snapshot of the physical state, never the model-visible data.

        It must work at every emitted stage, including ``simulator.setup`` during reset.
        """

    def _replay_tick(self) -> None:
        """Advance one frame along the start's prescribed motion, without hooks or a driver."""
        raise NotImplementedError("This simulator does not replay prescribed motion")

    def _replay_grid_step(self) -> None:
        for _ in range(self.simulator.decision_ticks):
            self._replay_tick()
            self.invalidate_state()

    def _hold_tick(self) -> None:
        raise ValueError("This simulator does not support physical deployment ticks")

    def _before_selected_capture(self, position: int) -> None:
        pass

    def _check_initialization(self, initialization: Any) -> None:
        if initialization is not None:
            raise ValueError("This simulator does not support replay initialization")

    def generate_frame_data(
        self, start: ScenarioStart, frames: Sequence[FrameReference], *,
        world: Mapping[str, Any] | None = None, deployment: DeploymentPlan | None = None,
    ) -> FrameDataSequence:
        """Record actual frame data by replaying ``start``'s prescribed motion over ``frames``.

        Used when the dataset has no frame data, such as a CARLA recipe. ``frames`` are the
        consecutive dataset frames beginning at ``start.frame``. ``world`` holds native world
        settings only, as for ``reset``. The runtime resets as for a run (same deployment plan
        and world settings), then advances one frame at a time along the
        prescribed motion and records each frame. No hook, attack or driving system runs.
        A runtime supports generation by implementing ``_replay_tick`` and
        ``_capture_frame_data``.
        """

        if type(self)._capture_frame_data is SimulatorRuntime._capture_frame_data:
            raise NotImplementedError(f"{type(self.simulator).__name__} cannot generate frame data")
        frames = tuple(frames)
        if not frames or frames[0] != start.frame:
            raise ValueError("Frame data generation must begin at the start frame")
        if any(frame.scenario_id != start.scenario_id or frame.position != start.frame.position + i
               for i, frame in enumerate(frames)):
            raise ValueError("Frame data generation needs consecutive frames of the start's scenario")
        self._generating_frame_data = True
        try:
            self.reset(start, world=world, deployment=deployment)
            records = [self._capture_frame_data(frames[0])]
            for frame in frames[1:]:
                self._replay_grid_step()
                self._take_capture()
                records.append(self._capture_frame_data(frame))
        finally:
            self._generating_frame_data = False
        if any(record.frame != frame for record, frame in zip(records, frames)):
            raise ValueError("Captured frame data must describe the requested frames")
        return FrameDataSequence(tuple(records), self._replayed_scenario(), {
            "simulator": dict(self.identity), "deployment_ticks": self._deployment.ticks,
            **self._frame_data_provenance(),
        })

    def _capture_frame_data(self, frame: FrameReference) -> FrameData:
        """Record the current capture and physical state as the frame data of ``frame``."""
        raise NotImplementedError(f"{type(self.simulator).__name__} cannot generate frame data")

    def _replayed_scenario(self) -> Mapping[str, Any]:
        """The resolved configuration that reproduces the generated frames, if any."""
        return {}

    def _frame_data_provenance(self) -> Mapping[str, Any]:
        """Simulator-specific provenance of generated frame data."""
        return {}


__all__ = [
    'DecisionInterval',
    'DeploymentPlan',
    'ClockSnapshot',
    'SensorGeometry',
    'SensorView',
    'RecordedAcquisition',
    'Initialization',
    'FrameDataSequence',
    'UnsupportedStartError',
    'Simulator',
    'SimulatorRuntime',
]
