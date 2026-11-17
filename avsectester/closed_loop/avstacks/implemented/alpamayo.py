"""Alpamayo camera policy behind the common driving-system interface.

The installed AlpaSim Alpamayo15Model consumes camera frames, ego pose history and navigation,
then returns candidate positions and rotations. The adapter retains bounded frame and pose
caches and exposes the selected trajectory through Control.trajectory. Coordinates and
waypoint timestamps remain those of the model's planning origin for controller anchoring.

Model dependencies load lazily. A checkpoint comes from an explicit constructor option or
AVSECTESTER_ALPAMAYO_CHECKPOINT. An injected policy can replace checkpoint loading.
"""

from __future__ import annotations

from avsectester.validation import timestamp_to_microseconds

import math
import os
import sys
from collections import deque
from collections.abc import Iterable, Mapping
from inspect import signature
from typing import Any, ClassVar
from pathlib import Path

from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.closed_loop.timing import HistoryRequirement
from avsectester.closed_loop.avstacks import driving_adapter
from avsectester.data_types import Control, DeliveredAcquisition, Observation, TimedStateEstimate
from avsectester.geometry.poses import rotation_to_quaternion
from avsectester.serialization import DEFAULT_REGISTRY
from avsectester.validation import require_nonnegative_integer, require_positive_integer

# A seeded stack loads AlpaSim with force_determinism, which enables deterministic PyTorch
# algorithms and sets this variable only when the model loads. CuBLAS honours it only when it is
# set before the process's first GPU matrix product, which other work in the same experiment,
# such as detector optimization, may already have run. Configuration loading imports this
# module before any of that work, so the default is in place in time.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _model_prediction(value: Any) -> dict[str, Any]:
    import numpy as np
    from alpasim_utils.geometry import Trajectory

    # NVIDIA AlpaSim models/base.py, inspected at
    # affc2eab209fa43bdfa2f26c0f8d437922d78a68. Arrays remain separate payloads.
    result = {"native_type": f"{type(value).__module__}.{type(value).__qualname__}",
              **{name: getattr(value, name) for name in (
                  "candidate_positions", "candidate_rotations", "selected_index",
                  "reasoning_text", "selected_plan")}}
    for name in ("model_t0_us", "pose_local_to_rig_t0", "waypoint_timestamps_us"):
        result[name] = getattr(value, name, None)
    plan = result["selected_plan"]
    if isinstance(plan, Trajectory):
        plan_record = _alpasim_trajectory(plan)
        times = result["waypoint_timestamps_us"]
        plan_times = plan_record["timestamps_us"]
        if (isinstance(times, np.ndarray) and times.dtype == plan_times.dtype
                and times.shape == plan_times.shape and np.array_equal(times, plan_times)):
            # The native trajectory getter creates an array, so share its recording value.
            result["waypoint_timestamps_us"] = plan_times
        result["selected_plan"] = plan_record
    return result


def _alpasim_pose(value: Any) -> dict[str, Any]:
    position, quaternion = value.to_proto()
    return {"position": list(position), "q_wxyz": list(quaternion)}


def _alpasim_trajectory(value: Any) -> dict[str, Any]:
    # AlpaSim geometry uses scipy's xyzw quaternion convention.
    return {"timestamps_us": value.timestamps_us, "positions": value.positions,
            "quaternions_xyzw": value.quaternions}


DEFAULT_REGISTRY.register_native('alpasim_driver.models.base', 'ModelPrediction', _model_prediction)
DEFAULT_REGISTRY.register_native(
    "alpasim_driver.models.base", "CameraFrame",
    lambda value: {"native_type": f"{type(value).__module__}.{type(value).__qualname__}",
                   "timestamp_us": value.timestamp_us, "image": value.image},
)
DEFAULT_REGISTRY.register_native('alpasim_utils.geometry', 'Pose', _alpasim_pose)
DEFAULT_REGISTRY.register_native('alpasim_utils.geometry', 'Trajectory', _alpasim_trajectory)


def alpamayo_driving_adapter(factory):
    """Bind one ``policy`` component (observation → Control) and no scorers."""
    def evaluate(avstack, inputs):
        if not isinstance(inputs, Mapping) or set(inputs) != {"observation"}:
            raise TypeError("Alpamayo policy inputs require observation")
        observation = inputs["observation"]
        avstack.reset(observation)
        return avstack(observation)

    return driving_adapter.DrivingAdapter(factory, components=((driving_adapter.ComponentDescriptor(
        "policy", {"observation": "Observation with camera deliveries and recorded pose history"},
        {"command": "Control with rig-frame trajectory"}, {"command": "Control"},
    ), evaluate),), history=AlpamayoAVStack.required_history(getattr(factory, "avstack_options", {})),
        decision_time_step=AlpamayoAVStack.decision_time_step)


ALPASIM_CAMERA_IDS = (
    "camera_front_wide_120fov", "camera_front_tele_30fov",
    "camera_cross_left_120fov", "camera_cross_right_120fov",
)
'Cameras of AlpaSim\'s Alpamayo 1.5 driver configuration.'


class AlpamayoAVStack(AVStack):
    """Camera policy with timestamped acquisition history and model-visible navigation.

    The framework supplies pose history before the latest delivered image and camera captures
    for the configured context. Defaults are four context frames and a subsample factor of one.
    Frames are selected newest-first by ``frame_subsample``, then passed oldest-first.
    Pose estimates follow the controller reporting clock.
    The model resamples the timestamped pose history at its required spacing.
    ``camera_ids`` defaults to AlpaSim's Alpamayo 1.5 cameras: front wide, front tele and both
    cross cameras.
    ``seed`` is the run seed, which configuration loading supplies. Like AlpaSim's driver, a
    seeded stack loads the model with ``force_determinism`` and seeds each decision from the
    run seed and the decision count since reset. Unlike AlpaSim's ``seed + count``, the
    derived seeds of distinct run seeds do not overlap. ``None`` leaves sampling unseeded. An
    injected ``model`` keeps its own determinism setting.
    The checkpoint is explicit or supplied by the environment.
    """

    provides_detections = False
    decision_time_step: ClassVar[float | None] = 0.1

    @classmethod
    def required_history(cls, options: Mapping[str, Any]) -> HistoryRequirement:
        """Recorded pose duration and camera samples needed by the configured context."""
        defaults = signature(cls).parameters
        length = options.get("context_length", defaults["context_length"].default)
        subsample = options.get("frame_subsample", defaults["frame_subsample"].default)
        require_positive_integer(length, "context_length")
        require_positive_integer(subsample, "frame_subsample")
        return HistoryRequirement(duration_us=1_500_000, samples=(length - 1) * subsample + 1)

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        camera_ids: list[str] | None = None,
        device: str = "cuda:0",
        context_length: int = 4,
        frame_subsample: int = 1,
        output_frequency_hz: int = 10,
        model: Any | None = None,
        seed: int | None = None,
    ) -> None:
        self.checkpoint_path = (
            checkpoint_path if checkpoint_path is not None
            else os.environ.get("AVSECTESTER_ALPAMAYO_CHECKPOINT")
        )
        self.camera_ids = camera_ids or list(ALPASIM_CAMERA_IDS)
        self.device = device
        require_positive_integer(context_length, "context_length")
        require_positive_integer(frame_subsample, "frame_subsample")
        self.context_length = context_length
        self.frame_subsample = frame_subsample
        self.history = self.required_history({"context_length": context_length,
                                              "frame_subsample": frame_subsample})
        require_positive_integer(output_frequency_hz, "output_frequency_hz")
        self.output_frequency_hz = output_frequency_hz
        if seed is not None:
            require_nonnegative_integer(seed, "seed")
        self.seed = seed
        # Separate adapters may share loaded weights while retaining independent input history.
        self._model = model
        self._owns_model = model is None
        self._closed = False
        history_length = self.history.samples
        self._frames: dict[str, deque[DeliveredAcquisition]] = {c: deque(maxlen=history_length) for c in self.camera_ids}
        self._poses: deque[TimedStateEstimate] = deque()
        self._ready = False
        self._prev_plan = None
        self._inference_count = 0
        self.last_reasoning: str | None = None
        self._last_log: dict[str, Any] | None = None

    @property
    def command_modes(self) -> frozenset[str]:
        return frozenset({"trajectory"})

    @property
    def supported_stages(self) -> frozenset[str]:
        """Expose ``policy.pre`` and ``policy.post`` around the end-to-end policy."""
        return frozenset(("policy.pre", "policy.post"))

    @property
    def native(self) -> Mapping[str, Any]:
        return {"model": self._model, "camera_history": self._frames}

    @property
    def models(self) -> Mapping[str, Any]:
        """The loaded policy model. Empty until reset or the first inference loads it."""
        return {} if self._model is None else {"policy": self._model}

    def _load(self):
        """
        Lazily build the Alpamayo model from the driver package (heavy: torch + a 10B checkpoint).
        """
        if not self.checkpoint_path:
            raise ValueError(
                "Provide checkpoint_path or set AVSECTESTER_ALPAMAYO_CHECKPOINT to the model location"
            )
        import torch
        from alpasim_driver.models.alpamayo1_5_model import Alpamayo15Model
        from alpasim_driver.schema import ModelConfig

        cfg = ModelConfig(
            model_type="alpamayo1_5",
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            num_trajectory_samples=1,
            force_determinism=self.seed is not None,
        )
        self._model = Alpamayo15Model.from_config(
            model_cfg=cfg,
            device=torch.device(self.device),
            camera_ids=self.camera_ids,
            context_length=self.context_length,
            output_frequency_hz=self.output_frequency_hz,
        )

    def reset(self, observation: Observation) -> None:
        """Clear context and validate recorded inputs in the first decision's deliveries."""
        if self._closed:
            raise RuntimeError("AlpamayoAVStack is closed")
        self._ready = False
        for buffer in self._frames.values():
            buffer.clear()
        self._poses.clear()
        self._prev_plan = None
        self._inference_count = 0
        self.last_reasoning = None
        self._last_log = None
        estimates = observation.estimated_states()
        for index, estimate in enumerate(estimates):
            self._observation_pose(estimate)
            if index and estimate.time_us <= estimates[index - 1].time_us:
                raise ValueError("Alpamayo state estimate times must increase")
        latest_image_times = []
        for camera in self.camera_ids:
            if observation.sensor_data.get(camera) is None:
                raise ValueError(f"Alpamayo observation is missing camera {camera!r}")
            samples = observation.delivered(camera)
            if len(samples) < self.history.samples:
                raise ValueError(f"Alpamayo camera {camera!r} requires {self.history.samples} actual images")
            if any(sample.data is None for sample in samples):
                raise ValueError(f"Alpamayo observation is missing camera {camera!r}")
            latest_image_times.append(samples[-1].time_us)
        if (not estimates or estimates[0].time_us > self._history_start_us(latest_image_times)
                or estimates[-1].time_us < max(latest_image_times)):
            raise ValueError("Alpamayo requires 1.5 seconds of recorded pose history before "
                             "the latest delivered image")
        if self._model is None:
            self._load()
        self._ready = True

    def close(self) -> None:
        """Release this driving system's references and any model it loaded itself.

        Closing is terminal and idempotent. An injected model is borrowed: another
        driving system may share it, so its close method and CUDA allocator are untouched.
        """
        if self._closed:
            return
        self._closed = True
        self._ready = False
        self._poses.clear()
        model, self._model = self._model, None
        owned = self._owns_model and model is not None
        for history in self._frames.values():
            history.clear()
        self._prev_plan = None
        self._last_log = None
        self.last_reasoning = None
        self._inference_count = 0
        failure = None
        try:
            if owned and callable(getattr(model, "close", None)):
                model.close()
        except BaseException as error:
            failure = error
            raise
        finally:
            # Drop the last wrapper-owned model reference before returning cached CUDA
            # blocks. Do not import torch, initialize CUDA, move weights, or collect GC.
            del model
            torch = sys.modules.get("torch")
            try:
                if (owned and str(self.device).split(":", 1)[0] == "cuda" and torch is not None
                        and torch.cuda.is_initialized()):
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
            except BaseException as cleanup:
                if failure is None:
                    raise
                failure.add_note(f"CUDA cache release also failed: {cleanup!r}")

    EPOCH_US = 10_000_000  # fixed offset for nonnegative protobuf timestamps
    'Timestamp offset in microseconds used by the trajectory interface.'

    @classmethod
    def _timestamp(cls, time: float) -> int:
        if not math.isfinite(time):
            raise ValueError("Alpamayo observation time must be finite")
        timestamp = timestamp_to_microseconds(time) + cls.EPOCH_US
        if timestamp < 0:
            raise ValueError("Alpamayo observation time precedes its nonnegative timestamp epoch")
        return timestamp

    @classmethod
    def _observation_pose(cls, observation: Observation | TimedStateEstimate) -> tuple[float, float, float, float]:
        if not isinstance(observation, (Observation, TimedStateEstimate)):
            raise TypeError("Alpamayo pose history requires timed state estimates")
        state = observation.vehicle_state
        try:
            pose = (float(state.x), float(state.y), float(state.yaw), float(observation.t))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Alpamayo requires vehicle_state with finite x, y and yaw") from error
        if not all(math.isfinite(value) for value in pose):
            raise ValueError("Alpamayo requires finite ego poses and observation times")
        cls._timestamp(pose[3])
        return pose

    def _history_start_us(self, latest_image_times: Iterable[int]) -> int:
        # AlpaSim anchors planning to the newest image across every configured camera.
        return max(latest_image_times) - self.history.duration_us

    def _append_observation(self, observation: Observation) -> None:
        for camera in self.camera_ids:
            if camera not in observation.sensor_data:
                if camera in observation.deliveries:
                    raise ValueError(f"Alpamayo observation is missing camera {camera!r}")
                continue
            for acquisition in observation.delivered(camera):
                if acquisition.data is None:
                    raise ValueError(f"Alpamayo observation is missing camera {camera!r}")
                self._frames[camera].append(acquisition)
        for estimate in observation.estimated_states():
            self._observation_pose(estimate)
            self._poses.append(estimate)
        earliest = self._history_start_us(frames[-1].time_us for frames in self._frames.values() if frames)
        # Retain the sample immediately before the history boundary for model interpolation.
        while len(self._poses) > 1 and self._poses[1].time_us <= earliest:
            self._poses.popleft()

    def ingest(self, observation: Observation) -> None:
        """Retain force_gt input deliveries and poses without running policy inference."""
        if self._closed:
            raise RuntimeError("AlpamayoAVStack is closed")
        if not self._ready:
            raise RuntimeError("Reset AlpamayoAVStack before ingesting observations")
        self._append_observation(observation)

    def _prediction_input(self, observation: Observation):
        import numpy as np
        from alpasim_driver.models.base import CameraFrame, DriveCommand, PredictionInput
        from alpasim_driver.navigation import determine_command_from_route
        from alpasim_grpc.v0 import common_pb2, egodriver_pb2

        self._append_observation(observation)
        earliest = self._history_start_us(frames[-1].time_us for frames in self._frames.values() if frames)
        if (not self._poses or self._poses[0].time_us > earliest
                or self._poses[-1].time_us < earliest + self.history.duration_us):
            raise ValueError("Alpamayo requires actual ego poses covering 1.5 seconds before "
                             "the latest delivered image")
        camera_images = {}
        for camera in self.camera_ids:
            frames = list(self._frames[camera])
            required = self.history.samples
            if len(frames) < required:
                raise ValueError(f"Alpamayo camera {camera!r} requires {required} actual images")
            camera_images[camera] = [
                CameraFrame(timestamp_us=self._timestamp(sample.t), image=sample.data)
                for sample in frames[-required::self.frame_subsample]
            ]
        poses = [self._observation_pose(estimate) for estimate in self._poses]
        # Like AlpaSim's driver: horizontal speed and longitudinal acceleration of the reported state.
        state = observation.vehicle_state
        speed = math.hypot(observation.ego_speed, float(getattr(state, "lateral_speed", 0.0)))
        acceleration = float(getattr(state, "acceleration", 0.0))
        route = None
        if observation.route is None:
            command = DriveCommand.STRAIGHT
        else:
            points = np.asarray(observation.route, dtype=float)
            if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
                raise ValueError("Alpamayo route must be a finite (N, 3) array in ego coordinates")
            route = egodriver_pb2.Route(timestamp_us=self._timestamp(observation.t),
                waypoints=[common_pb2.Vec3(x=float(x), y=float(y), z=float(z)) for x, y, z in points])
            command = determine_command_from_route(route)
        return PredictionInput(
            camera_images=camera_images, command=command, speed=speed, acceleration=acceleration,
            ego_pose_history=[self._pose_at(x, y, yaw, self._timestamp(time))
                              for x, y, yaw, time in poses],
            inference_seed=self._inference_seed(), previous_plan=self._prev_plan, route=route,
        )

    def _inference_seed(self) -> int:
        """This decision's sampling seed: independent streams for distinct run seeds."""
        if self.seed is None:
            return self._inference_count
        import numpy as np

        return int(np.random.SeedSequence((self.seed, self._inference_count)).generate_state(1)[0])

    @staticmethod
    def _pose_at(x: float, y: float, yaw: float, t_us: int):
        from alpasim_grpc.v0 import common_pb2

        return common_pb2.PoseAtTime(
            pose=common_pb2.Pose(
                vec=common_pb2.Vec3(x=x, y=y, z=0.0),
                quat=common_pb2.Quat(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2)),
            ),
            timestamp_us=t_us,
        )

    def __call__(self, observation: Observation) -> Control:
        """Ingest deliveries and infer a rig-frame trajectory from cached cameras and ego poses."""
        if self._closed:
            raise RuntimeError("AlpamayoAVStack is closed")
        if not self._ready:
            raise RuntimeError("Reset AlpamayoAVStack with recorded deliveries before requesting a decision")
        inputs = self._emit("policy.pre", self._prediction_input(observation))
        prediction = self._emit("policy.post", self._model.predict(inputs))
        policy_record = DEFAULT_REGISTRY.snapshot(prediction)
        self._inference_count += 1
        self._prev_plan = getattr(prediction, "selected_plan", None)
        self.last_reasoning = getattr(prediction, "reasoning_text", None)  # chain-of-causation text, if any
        command = Control(trajectory=self._to_waypoints(prediction, observation))
        # Record the already-produced policy output and command without extra inference.
        import numpy as np

        route = inputs.route
        route_waypoints = (None if route is None else np.asarray(
            [[point.x, point.y, point.z] for point in route.waypoints], dtype=float).reshape(-1, 3))
        self._last_log = {
            "policy": policy_record, "command": DEFAULT_REGISTRY.snapshot(command),
            "route_waypoints": route_waypoints, "navigation_command": inputs.command.name,
            "navigation_note": ("No route supplied; using STRAIGHT" if route is None
                                else "Navigation command derived from observation.route"),
            "history_pose_count": len(inputs.ego_pose_history),
            "selected_capture_ids": {camera: [sample.capture_id for sample in
                list(self._frames[camera])[-self.history.samples::self.frame_subsample]]
                for camera in self.camera_ids},
            "selected_image_times": {camera: [(frame.timestamp_us - self.EPOCH_US) / 1e6
                                               for frame in frames]
                                     for camera, frames in inputs.camera_images.items()},
        }
        return command

    def component_log(self) -> dict[str, Any] | None:
        """Already-produced policy prediction and Control under ``policy`` and ``command``.

        Also includes route waypoints, semantic navigation command, a navigation note,
        real-history pose count, selected capture IDs and image times on the simulator clock.
        Recording does not request additional inference.
        """
        return self._last_log

    def _to_waypoints(self, prediction: Any, observation: Observation) -> list:
        """Prepend the model's t0 rig origin and retain predicted poses and exact timestamps."""
        import numpy as np

        index = int(prediction.selected_index)
        positions = np.asarray(prediction.candidate_positions)[index]
        rotations = np.asarray(prediction.candidate_rotations)[index]
        if (positions.ndim != 2 or positions.shape[1:] != (3,)
                or rotations.shape != (len(positions), 3, 3) or not np.isfinite(positions).all()):
            raise ValueError("Alpamayo prediction needs (T, 3) positions and (T, 3, 3) rotations")
        model_origin = prediction.model_t0_us
        origin = (timestamp_to_microseconds(observation.t) if model_origin is None
                  else int(model_origin) - self.EPOCH_US)
        provided_times = prediction.waypoint_timestamps_us
        if provided_times is None:
            times = [origin + round((index + 1) * 1e6 / self.output_frequency_hz)
                     for index in range(len(positions))]
        else:
            provided_times = np.asarray(provided_times)
            if provided_times.shape != (len(positions),):
                raise ValueError("Alpamayo waypoint timestamps must match the predicted positions")
            times = [int(time) - self.EPOCH_US for time in provided_times]
        if any(second <= first for first, second in zip([origin, *times], times)):
            raise ValueError("Alpamayo waypoint timestamps must increase after model t0")
        trajectory = [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), origin), *[
            (tuple(float(value) for value in position), rotation_to_quaternion(rotation), time)
            for position, rotation, time in zip(positions, rotations, times)
        ]]
        return trajectory


__all__ = [
    'alpamayo_driving_adapter',
    'AlpamayoAVStack',
]
