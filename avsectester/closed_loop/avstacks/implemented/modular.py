"""ModularAVStack — an AVStack backed by an avstack ``ModularDrivingPipeline``.

The modular (white-box) AVStack: perception -> tracking -> planning -> control, built from config.
Heavy avstack imports are lazy (done when the pipeline is built), so this module — and the
``avsectester.closed_loop.avstacks`` package — imports without avstack, matching `AlpamayoAVStack`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any

from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.closed_loop.avstacks import driving_adapter
from avsectester.closed_loop.hooks import CallInputs, Plugin, RunContext, StageContext
from avsectester.data_types import Control, Detection, Observation
from avsectester.serialization import DEFAULT_REGISTRY

if TYPE_CHECKING:
    from avsectester.closed_loop.hooks import StageEmitter


def _snapshot_sensor(value: Any, capture_child: Callable[[Any], Any]) -> Any:
    result = copy(value)
    # The allowlist is also used by persistence. Copy wrappers and references,
    # but snapshot the explicitly supported SDK payload through its own registration.
    for name, child in _sensor(value).items():
        if name != "native_type":
            setattr(result, name, capture_child(child))
    # These are optional derived image caches owned by native wrappers.
    for name in ("_rgb_image", "depth_in_meters"):
        if hasattr(value, name):
            setattr(result, name, capture_child(getattr(value, name)))
    return result


def _snapshot_native(value: Any, capture_child: Callable[[Any], Any]) -> Any:
    """Keep native in-memory behavior while capturing children through adapter registrations.

    ``deepcopy`` of an outer DataContainer bypasses the ReferenceFrame registration and follows
    every reverse reference in the live global frame. Native record fields are captured
    recursively here instead. Persistence continues to use explicit field allowlists.
    """
    result = copy(value)
    for name, child in vars(value).items():
        vars(result)[name] = capture_child(child)
    return result


def _snapshot_reference(value: Any, capture_child: Callable[[Any], Any]) -> Any:
    """Copy only an ancestor chain; descendants and integrated caches are not pose data."""
    result = copy(value)
    state = vars(result)
    for name, child in vars(value).items():
        if name == "_point_from":
            state[name] = []
        elif name == "_global_integrated":
            state[name] = None
        else:
            state[name] = capture_child(child)
    return result


def _fields(value: Any, *names: str) -> dict[str, Any]:
    """Copy an explicit field allowlist, never an object's attribute dictionary."""
    return {
        "native_type": f"{type(value).__module__}.{type(value).__qualname__}",
        **{name: getattr(value, name) for name in names},
    }


def _quaternion(value: Any) -> list[float]:
    return [float(value.w), float(value.x), float(value.y), float(value.z)]


def _reference(value: Any) -> dict[str, Any]:
    result = _fields(value, "reference", "handedness", "n_prec", "from_frame", "to_frame",
                     "timestamp")
    result.update({name: getattr(value, name).tolist() for name in ("x", "v", "acc")})
    result.update(q_wxyz=_quaternion(value.q), angular_wxyz=_quaternion(value.ang))
    # Never follow _point_from: it contains reverse links and potentially an
    # entire live sensor tree. The parent chain alone defines the coordinates.
    return result


def _vector(value: Any) -> dict[str, Any]:
    return {**_fields(value, "reference", "n_prec"), "x": value.x.tolist()}


def _rotation(value: Any) -> dict[str, Any]:
    return {**_fields(value, "reference", "n_prec"), "q_wxyz": _quaternion(value.q)}


def _box3d(value: Any) -> dict[str, Any]:
    return {**_fields(value, "ID", "obj_type", "position", "attitude", "where_is_t"),
            "hwl": [float(value.h), float(value.w), float(value.l)]}


def _camera(value: Any) -> dict[str, Any]:
    return {**_fields(value, "reference", "fov_horizontal", "fov_vertical", "square_pixels",
                      "channel_order"),
            "P": value.P.tolist(), "img_shape": list(value.img_shape)}


def _semantic_camera(value: Any) -> dict[str, Any]:
    return {**_camera(value), "colors": value.colors,
            "tags": [{"id": key, "name": name} for key, name in value.tags.items()]}


def _sensor(value: Any) -> dict[str, Any]:
    result = _fields(value, "timestamp", "frame", "source_ID", "source_name",
                     "source_identifier", "data", "calibration")
    # These are declared SensorData/subclass fields, not arbitrary kwargs.
    for name in ("depth", "in_front", "levar", "encoding", "n_features", "flipy"):
        if hasattr(value, name):
            result[name] = getattr(value, name)
    # Image and point arrays stay arrays: RunLog places them in payload shards.
    return result


def _detection(value: Any) -> dict[str, Any]:
    return {**_fields(value, "ID", "source_identifier", "obj_type", "score", "data",
                      "reference"), "noise": value.noise.tolist()}


def _track(value: Any) -> dict[str, Any]:
    return {
        **_fields(value, "ID", "obj_type", "t0", "t", "dt_coast", "n_updates", "n_missed",
                  "score", "reference", "attitude"),
        "x": value.x.tolist(), "P": value.P.tolist(),
    }


def _box_track(value: Any) -> dict[str, Any]:
    return {**_track(value), "box": value.box, "velocity": value.velocity}


def _waypoints(value: Any) -> dict[str, Any]:
    # Reading the heap must not pop entries or alter the planner's state.
    entries = sorted(value.heap, key=lambda pair: pair[0])
    return {**_fields(value, "min_waypoints", "dist_arrived", "max_dist", "max_size"),
            "waypoints": [{"priority": value.mult * priority, "waypoint": item}
                          for priority, item in entries]}


_NATIVE_RECORDS: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
    ('avstack.datastructs', 'DataContainer', lambda value: _fields(value, "frame", "timestamp", "source_identifier", "source_reference", "data")),
    ('avstack.geometry.refchoc', 'ReferenceFrame', _reference),
    ('avstack.geometry.refchoc', 'PassiveReferenceFrame', lambda value: _fields(value, "frame_id", "timestamp")),
    ('avstack.geometry.refchoc', 'Vector', _vector),
    ('avstack.geometry.refchoc', 'Rotation', _rotation),
    ('avstack.geometry.datastructs', 'Pose', lambda value: _fields(value, "position", "attitude")),
    ('avstack.geometry.datastructs', 'Twist', lambda value: _fields(value, "linear", "angular")),
    ('avstack.geometry.datastructs', 'PointMatrix2D', lambda value: _fields(value, "x", "calibration")),
    ('avstack.geometry.datastructs', 'PointMatrix3D', lambda value: _fields(value, "x", "calibration")),
    ('avstack.geometry.bbox', 'Box3D', _box3d),
    ('avstack.geometry.bbox', 'Box2D', lambda value: {**_fields(value, "ID", "obj_type", "calibration"), "xyxy": value.box2d.tolist()}),
    ('avstack.geometry.bbox', 'SegMask2D', lambda value: _fields(value, "ID", "mask", "calibration")),
    ('avstack.environment.objects', 'ObjectState', lambda value: _fields(value, "ID", "obj_type", "score", "t", "position", "velocity", "acceleration",
                      "attitude", "angular_velocity", "box", "occlusion", "visible_fraction")),
    ('avstack.calibration', 'Calibration', lambda value: _fields(value, "reference")),
    ('avstack.calibration', 'CameraCalibration', _camera),
    ('avstack.calibration', 'SemanticSegmentationCalibration', _semantic_camera),
    ('avstack.calibration', 'RadarCalibration', lambda value: _fields(value, "reference", "fov_horizontal", "fov_vertical")),
    ('avstack.sensors', 'SensorData', _sensor),
    ('avstack.modules.perception.detections', 'Detection_', _detection),
    ('avstack.modules.tracking.tracks', 'TrackBase', _track),
    ('avstack.modules.tracking.tracks', 'BasicBoxTrack2D', _box_track),
    ('avstack.modules.tracking.tracks', 'BasicBoxTrack3D', _box_track),
    ('avstack.modules.tracking.tracks', 'BasicJointBoxTrack', lambda value: _fields(value, "track_2d", "track_3d")),
    ('avstack.modules.tracking.tracks', 'GroupTrack', lambda value: _fields(value, "state", "members")),
    ('avstack.modules.planning.types', 'WaypointPlan', _waypoints),
    ('avstack.modules.planning.types', 'Waypoint', lambda value: _fields(value, "target_point", "target_speed")),
    ('avstack.modules.control.types', 'VehicleControlSignal', lambda value: _fields(value, "timestamp", "throttle", "steer", "brake", "hand_brake", "reverse",
                      "manual_gear_shift", "bounded")),
)

for _module, _name, _encode in _NATIVE_RECORDS:
    _snapshot = (_snapshot_sensor if _module == "avstack.sensors" else
                 _snapshot_reference if (_module, _name) ==
                 ("avstack.geometry.refchoc", "ReferenceFrame") else _snapshot_native)
    DEFAULT_REGISTRY.register_native(_module, _name, _encode, snapshot=_snapshot)


class NativeHook(Plugin):
    """Adapt a registered avstack hook's tuple convention to an ordered stage hook.

    Import the implementation's registration module explicitly before running. The native
    handler is built afresh on reset and released when its enclosing hook session closes.
    """

    def __init__(self, config: Mapping[str, Any], *, phase: str = "post") -> None:
        if phase not in {"pre", "post"}:
            raise ValueError("NativeHook phase must be 'pre' or 'post'")
        self.config = deepcopy(dict(config))
        self.phase = phase
        self.native: Any = None

    def reset(self, context: RunContext) -> None:
        from avstack.config import HOOKS

        self.native = HOOKS.build(deepcopy(self.config))

    def transform(self, value: Any, context: StageContext) -> Any:
        if self.phase == "pre":
            if not isinstance(value, CallInputs):
                raise TypeError("Native pre-hooks require CallInputs")
            args, kwargs = self.native(*value.args, **value.kwargs)
            return CallInputs(args=args, kwargs=kwargs)
        result = self.native(value)
        if not isinstance(result, tuple) or len(result) != 1:
            raise TypeError("Native post-hooks must return one output in a tuple")
        return result[0]

    def close(self) -> None:
        native, self.native = self.native, None
        close = getattr(native, "close", None)
        if close is not None:
            close()


def register_carla_dataset() -> type:
    """Return the CARLA dataset class, registering its vendored declaration if needed.

    CARLA-trained PointPillars configurations require this class even for inference,
    where MMDetection3D builds a lazy dataset to obtain metadata. Existing registrations
    are preserved. Only the dataset module is loaded, leaving the installed detector
    package and its CUDA operators unchanged.
    """
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    from mmdet3d.registry import DATASETS

    existing = DATASETS.get("CarlaDataset")
    if existing is not None:
        return existing

    # The installed package supplies Det3DDataset and the module's relative imports.
    importlib.import_module("mmdet3d.datasets")
    name = "mmdet3d.datasets.carla_dataset"
    if importlib.util.find_spec(name) is not None:
        importlib.import_module(name)
    else:
        import avstack

        source = (
            Path(avstack.__file__).resolve().parents[1]
            / "third_party/mmdetection3d/mmdet3d/datasets/carla_dataset.py"
        )
        if not source.is_file():
            raise FileNotFoundError(
                f"CARLA dataset declaration not found: {source}. Initialize the nested "
                "third_party/mmdetection3d submodule in third_party/avstack-core."
            )
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise

    registered = DATASETS.get("CarlaDataset")
    if registered is None:
        raise RuntimeError(f"{name} did not register CarlaDataset")
    return registered


def modular_driving_adapter(factory, *, components=(), scorers=()):
    """Expose explicitly selected native Modular components and existing detector scorers.

    Component inputs are {args: tuple, kwargs: dict}, matching that native component's
    call signature. MMDet scorer inputs follow the existing scorer API and TargetSpec.
    Declaring a scorer is an explicit assertion that the factory supplies that model.
    """
    bindings = []
    for name in components:
        def evaluate(avstack, inputs, name=name):
            if name not in avstack.native:
                raise ValueError(f"Driving system does not expose component {name!r}")
            return avstack.native[name](*inputs.get("args", ()), **inputs.get("kwargs", {}))
        bindings.append((driving_adapter.ComponentDescriptor(
            name, {"args": "native positional arguments", "kwargs": "native keyword arguments"},
            {"value": "native component output"}, {"value": "adapter codecs"},
        ), evaluate))
    scores = []
    for name in scorers:
        if name not in {"mmdet.inference", "mmdet.rpn_objectness"}:
            raise ValueError(f"Unknown built-in scorer {name!r}")
        differentiable = name == "mmdet.rpn_objectness"

        def score(avstack, inputs, target, differentiable=differentiable):
            from avsectester.closed_loop.avstacks.implemented.scorers import MMDetInferenceScorer, MMDetRPNObjectnessScorer

            model = avstack.models["perception"]
            scorer = MMDetRPNObjectnessScorer(model) if differentiable else MMDetInferenceScorer(model)
            return scorer.target_score(inputs, target)
        scores.append((driving_adapter.ScorerDescriptor(
            name, {"image": "CHW float tensor [0,1]" if differentiable else "HWC RGB uint8"},
            {"score": "scalar confidence"}, {"score": "tensor" if differentiable else "float"},
            ("image",) if differentiable else (), (), "minimize",
        ), score))
    return driving_adapter.DrivingAdapter(factory, components=bindings, scorers=scores)


def _remove_hook(hooks: list, target: Any) -> None:
    """Remove our own hook without disturbing hooks installed by the caller."""
    hooks[:] = [hook for hook in hooks if hook is not target]


class _StagePreHook:
    def __init__(self, avstack: ModularAVStack, stage: str) -> None:
        self.avstack = avstack
        self.stage = stage

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        from avsectester.closed_loop.hooks import CallInputs

        inputs = self.avstack._emit(self.stage, CallInputs(args=args, kwargs=kwargs))
        if not isinstance(inputs, CallInputs):
            raise TypeError(f"{self.stage} must return CallInputs")
        return inputs.args, inputs.kwargs


class _StagePostHook:
    def __init__(self, avstack: ModularAVStack, stage: str) -> None:
        self.avstack = avstack
        self.stage = stage

    def __call__(self, output: Any) -> tuple[Any]:
        # avstack post-hooks return a tuple of positional outputs.
        return (self.avstack._emit(self.stage, output),)


class _StageCapture:
    """A generic avstack post-hook that keeps a pipeline stage's most recent output (pass-through).

    One capture mechanism for detections and component analysis: it snapshots the stage output before
    downstream modules can mutate it, and returns the original so the pipeline is unaffected."""

    def __init__(self) -> None:
        self.last: Any = None

    def __call__(self, output: Any, *args: Any, **kwargs: Any) -> tuple[Any]:
        self.last = DEFAULT_REGISTRY.snapshot(output)
        return (output,)


def _register_avstack_modules() -> None:
    """Import the avstack perception, tracking, planning and control modules."""
    import avstack.modules.control.vehicle
    import avstack.modules.perception.object3d
    import avstack.modules.pipeline
    import avstack.modules.planning.vehicle
    import avstack.modules.tracking.tracker3d  # noqa: F401  (BasicBoxTracker3D)


class ModularAVStack(AVStack):
    """An `avsectester.closed_loop.avstacks.base.AVStack` backed by an avstack ``ModularDrivingPipeline``.

    Observation -> the pipeline (perception -> tracking -> planning -> control) -> a Control command.
    CARLA-trained MMDetection3D checkpoint configurations register their dataset metadata
    automatically before the pipeline is built.
    Runtime transforms replace each available stage's inputs and outputs. Use NativeHook
    to adapt explicitly registered native hooks to the stage chain.
    `instrument` optionally captures component outputs for `component_log`.
    Perception is always captured when supported, allowing `detections` to return
    canonical boxes without additional inference.
    """

    provides_detections = True

    def __init__(self, pipeline_cfg: dict) -> None:
        _register_avstack_modules()
        from avstack.config import PIPELINE

        perception = pipeline_cfg.get("perception", {})
        if (
            isinstance(perception, Mapping)
            and perception.get("type") == "MMDetObjectDetector3D"
            and perception.get("dataset", "kitti").lower().startswith("carla-")
            and not perception.get("deploy", False)
        ):
            # PointPillars inference also needs its custom dataset metadata.
            register_carla_dataset()
        self.pipeline = PIPELINE.build(deepcopy(pipeline_cfg))
        self._captures: dict[str, _StageCapture] = {}
        self._perception_capture: _StageCapture | None = None
        self._last_detections: tuple[Detection, ...] | None = None
        self._detection_reference: Any = None
        self._detections_ready = False
        perception_module = self._stage_modules().get("perception")
        if perception_module is not None:
            self._perception_capture = _StageCapture()
            perception_module.register_post_hook(self._perception_capture)

    def reset(self, observation: Observation, past: Sequence[Observation] = ()) -> None:
        """Clear telemetry. Use a fresh AVStack for independent pipeline histories.

        avstack modules do not share a complete reset contract for neural model,
        tracker, planner and controller state. A custom adapter can override reset
        when its modules provide one. Constructing a fresh ModularAVStack is the
        default way to begin an independent experiment.
        """
        for capture in self._captures.values():
            capture.last = None
        if self._perception_capture is not None:
            self._perception_capture.last = None
        self._last_detections = None
        self._detection_reference = None
        self._detections_ready = False

    def _stage_modules(self) -> dict[str, Any]:
        """Stages whose effective call actually dispatches avstack hooks."""
        stages = {}
        for name in ("perception", "tracking", "planning", "control"):
            module = getattr(self.pipeline, name, None)
            # BaseModule owns hook lists even when a subclass bypasses apply_hooks.
            # Only effective calls marked by the hook dispatcher advertise support.
            call = getattr(type(module), "__call__", None)
            if (
                getattr(call, "__avstack_hooks__", False)
                and isinstance(getattr(module, "pre_hooks", None), list)
                and isinstance(getattr(module, "post_hooks", None), list)
            ):
                stages[name] = module
        return stages

    @property
    def command_modes(self) -> frozenset[str]:
        return frozenset({"actuator"})

    @property
    def supported_stages(self) -> frozenset[str]:
        return frozenset(
            f"{name}.{phase}"
            for name in self._stage_modules()
            for phase in ("pre", "post")
        )

    @property
    def native(self) -> Mapping[str, Any]:
        """The pipeline and its actual modules, including non-hookable modules."""
        return {
            "pipeline": self.pipeline,
            **{
                name: module
                for name in ("perception", "tracking", "planning", "control")
                if (module := getattr(self.pipeline, name, None)) is not None
            },
        }

    @property
    def models(self) -> Mapping[str, Any]:
        """Public handles for the built components and their loaded models."""
        return {
            name: getattr(module, "model", module)
            for name, module in self.native.items()
            if name != "pipeline"
        }

    @contextmanager
    def bind_hooks(self, dispatcher: StageEmitter) -> Iterator[None]:
        """Bridge hook callbacks for one run, preserving native hook order."""
        installed = []
        with super().bind_hooks(dispatcher):
            try:
                for name, module in self._stage_modules().items():
                    pre = _StagePreHook(self, f"{name}.pre")
                    post = _StagePostHook(self, f"{name}.post")
                    module.pre_hooks.append(pre)
                    # Instrumentation must observe the final, modified output.
                    index = next(
                        (i for i, hook in enumerate(module.post_hooks) if isinstance(hook, _StageCapture)),
                        len(module.post_hooks),
                    )
                    module.post_hooks.insert(index, post)
                    installed.extend(((module.pre_hooks, pre), (module.post_hooks, post)))
                yield
            finally:
                for hooks, hook in installed:
                    _remove_hook(hooks, hook)

    def instrument(self, stages: tuple[str, ...] = ("perception", "tracking", "planning", "control")) -> None:
        """Capture named stage outputs after attack hooks, enabling `component_log`."""
        modules = self._stage_modules()
        missing = set(stages) - modules.keys()
        if missing:
            raise ValueError(f"Cannot instrument unavailable stages: {sorted(missing)}")
        for stage, capture in self._captures.items():
            module = modules.get(stage)
            if module is not None:
                _remove_hook(module.post_hooks, capture)
        if self._perception_capture is not None:
            _remove_hook(modules["perception"].post_hooks, self._perception_capture)
        self._captures = {
            stage: (self._perception_capture or _StageCapture()) if stage == "perception" else _StageCapture()
            for stage in stages
        }
        for stage, capture in self._captures.items():
            getattr(self.pipeline, stage).register_post_hook(capture)
        if self._perception_capture is not None and "perception" not in self._captures:
            modules["perception"].register_post_hook(self._perception_capture)

    def component_log(self) -> dict[str, Any] | None:
        """Return the most recent per-stage outputs after ``instrument``, or None otherwise.

        Outputs are snapshots of native values;
        canonical evaluation boxes are separately available from `detections`."""
        return {stage: cap.last for stage, cap in self._captures.items()} if self._captures else None

    def detections(self) -> Sequence[Detection] | None:
        """Boxes from the current decision in observation-time ego coordinates.

        Results are detached before advancing the world. A detector with no boxes returns
        an empty tuple; an unavailable perception stage returns None.
        """
        if self._detections_ready:
            return self._last_detections
        self._detections_ready = True
        capture = self._perception_capture
        if capture is None:
            return None
        output = capture.last
        if output is None:
            self._last_detections = ()
            return self._last_detections
        if not hasattr(output, "__iter__"):
            return None
        try:
            items = tuple(output)
            if not items:
                self._last_detections = ()
                return self._last_detections
            if self._detection_reference is None:
                return None
            detections = []
            for item in items:
                native_box = getattr(item, "data", None)
                if not callable(getattr(native_box, "change_reference", None)):
                    return None
                box = native_box.change_reference(self._detection_reference, inplace=False)
                detections.append(Detection(
                    position=tuple(float(value) for value in box.center.x),
                    extent=(float(box.l), float(box.w), float(box.h)),
                    yaw=float(box.yaw), category=item.obj_type,
                    score=None if item.score is None else float(item.score),
                ))
        except (AttributeError, TypeError, ValueError, OverflowError):
            # Native perception can return masks, class logits or custom box objects.
            # Telemetry conversion must never interrupt an otherwise valid decision.
            return None
        self._last_detections = tuple(detections)
        return self._last_detections

    def _remember_capture_reference(self, observation: Observation) -> None:
        """Detach the physical body parent of a sensor's installation reference.

        Native detection coordinates retain the sensor reference. Localization modifies
        vehicle_state, which is deliberately not read by this recording-only conversion.
        """
        self._detection_reference = None
        self._last_detections = None
        self._detections_ready = False
        if not isinstance(observation.sensor_data, Mapping):
            return
        for name, payload in observation.sensor_data.items():
            calibration = observation.calibration.get(name, getattr(payload, "calibration", None))
            reference = getattr(calibration, "reference", None)
            if reference is None:
                reference = getattr(payload, "source_reference", None)
            if reference is not None:
                reference = getattr(reference, "reference", None)
                # Without a vehicle body parent, the detections cannot be expressed in the ego frame.
                if reference is None or getattr(reference, "is_global_origin", False):
                    continue
                self._detection_reference = DEFAULT_REGISTRY.snapshot(reference)
                break

    def __call__(self, observation: Observation) -> Control:
        """Run the native perception-to-control pipeline once using the current capture reference."""
        self._remember_capture_reference(observation)
        if self._perception_capture is not None:
            hooks = self.pipeline.perception.post_hooks
            _remove_hook(hooks, self._perception_capture)
            hooks.append(self._perception_capture)
        ctrl = self.pipeline(observation.sensor_data, observation.vehicle_state)
        return Control(
            throttle=float(ctrl.throttle), steer=float(ctrl.steer), brake=float(ctrl.brake)
        )


__all__ = [
    'NativeHook',
    'register_carla_dataset',
    'modular_driving_adapter',
    'ModularAVStack',
]
