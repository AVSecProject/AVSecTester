"""Alpamayo end-to-end camera policy as an `avsectester.closed_loop.avstacks.base.AVStack`.

Wraps AlpaSim's own Alpamayo model (``alpasim_driver.models.alpamayo1_5_model.Alpamayo15Model``) so
it plugs into ``ClosedLoopRunner``: an `avsectester.data_types.Observation`
(a rendered camera + ego state) in, a `avsectester.data_types.Control` (the planned trajectory) out.

Alpamayo is a trajectory-output policy: ``predict(PredictionInput) -> ModelPrediction`` with
``candidate_positions`` of shape ``(K, T, 3)`` in the **rig frame**. We keep a short temporal buffer
of camera frames + ego poses (the model's context) and emit the selected candidate as
``Control.trajectory``; the simulator's configured ego dynamics tracks that plan.

Runs in the AlpaSim **driver env** (Python 3.12 + ``alpasim_driver`` and the Alpamayo repos), NOT the
CARLA/avsec 3.11 env — so all heavy imports are lazy and this module imports fine without them. The
checkpoint is supplied explicitly or through ``AVSECTESTER_ALPAMAYO_CHECKPOINT``.
"""

from __future__ import annotations

import math
import os
import sys
from collections import deque
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar
from pathlib import Path

from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.closed_loop.avstacks import driving_adapter
from avsectester.data_types import Control, Observation
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
    # NVIDIA AlpaSim models/base.py, inspected at
    # affc2eab209fa43bdfa2f26c0f8d437922d78a68. Arrays remain separate payloads.
    result = {"native_type": f"{type(value).__module__}.{type(value).__qualname__}",
              **{name: getattr(value, name) for name in (
                  "candidate_positions", "candidate_rotations", "selected_index",
                  "reasoning_text", "selected_plan")}}
    for name in ("model_t0_us", "pose_local_to_rig_t0", "waypoint_timestamps_us"):
        result[name] = getattr(value, name, None)
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
    """Bind one ``policy`` component (observation + past → Control) and no scorers."""
    def evaluate(avstack, inputs):
        if not isinstance(inputs, Mapping) or set(inputs) != {"observation", "past"}:
            raise TypeError("Alpamayo policy inputs require observation and past")
        observation, past = inputs["observation"], inputs["past"]
        avstack.reset(observation, past=past)
        return avstack(observation)

    return driving_adapter.DrivingAdapter(factory, components=((driving_adapter.ComponentDescriptor(
        "policy", {"observation": "Observation with configured camera IDs and ego pose",
                   "past": "Sequence[Observation] covering the previous 15 decisions"},
        {"command": "Control with rig-frame trajectory"}, {"command": "Control"},
    ), evaluate),), past_steps=AlpamayoAVStack.past_steps,
        decision_time_step=AlpamayoAVStack.decision_time_step)


ALPASIM_CAMERA_IDS = (
    "camera_front_wide_120fov", "camera_front_tele_30fov",
    "camera_cross_left_120fov", "camera_cross_right_120fov",
)
'Cameras of AlpaSim\'s Alpamayo 1.5 driver configuration.'


class AlpamayoAVStack(AVStack):
    """A policy using real 10 Hz observations, image context and model-visible navigation.

    ``past_steps=15`` supplies the 16 poses spanning 1.5 s required by AlpaSim's
    ``alpamayo_base.py`` NUM_HISTORY_STEPS=16 and HISTORY_TIME_STEP=0.1. Images require
    ``(context_length - 1) * frame_subsample`` past decisions, which must fit that history.
    ``reset`` receives the recorded past; neither missing images nor poses are fabricated.
    ``camera_ids`` defaults to AlpaSim's Alpamayo 1.5 cameras: front wide, front tele and both
    cross cameras.
    ``seed`` is the run seed, which configuration loading supplies. Like AlpaSim's driver, a
    seeded stack loads the model with ``force_determinism`` and seeds each decision from the
    run seed and the decision count since reset. Unlike AlpaSim's ``seed + count``, the
    derived seeds of distinct run seeds do not overlap. ``None`` leaves sampling unseeded. An
    injected ``model`` keeps its own determinism setting.
    """

    provides_detections = False
    past_steps: ClassVar[int] = 15
    decision_time_step: ClassVar[float | None] = 0.1

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
        if (context_length - 1) * frame_subsample > self.past_steps:
            raise ValueError("The image context requires more than 15 past decisions; "
                             "reduce context_length or frame_subsample")
        require_positive_integer(output_frequency_hz, "output_frequency_hz")
        self.output_frequency_hz = output_frequency_hz
        if seed is not None:
            require_nonnegative_integer(seed, "seed")
        self.seed = seed
        # Separate adapters may share loaded weights while retaining independent input history.
        self._model = model
        self._owns_model = model is None
        self._closed = False
        history_length = (context_length - 1) * frame_subsample + 1
        self._frames: dict[str, deque] = {c: deque(maxlen=history_length) for c in self.camera_ids}
        self._poses: deque[tuple[float, float, float, float]] = deque(maxlen=self.past_steps + 1)
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
        """Lazily build the Alpamayo model from the driver package (heavy: torch + a 10B checkpoint)."""
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

    def reset(self, observation: Observation, past: Sequence[Observation] = ()) -> None:
        """Initialize policy context from recorded observations before the selected origin."""
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
        history = list(past)
        if len(history) < self.past_steps:
            raise ValueError("Alpamayo requires 15 recorded past observations before its first decision")
        all_observations = [*history, observation]
        for index, item in enumerate(all_observations):
            self._observation_pose(item)
            for camera in self.camera_ids:
                if item.sensor_data.get(camera) is None:
                    raise ValueError(f"Alpamayo observation is missing camera {camera!r}")
            if index:
                self._validate_interval(all_observations[index - 1].t, item.t)
        if self._model is None:
            self._load()
        for item in history[-self.past_steps:]:
            self._append_observation(item)
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

    @staticmethod
    def _validate_interval(first: float, second: float) -> None:
        if round(second * 1e6) - round(first * 1e6) != 100000:
            raise ValueError("Alpamayo requires a 0.1 second decision interval")

    @classmethod
    def _timestamp(cls, time: float) -> int:
        if not math.isfinite(time):
            raise ValueError("Alpamayo observation time must be finite")
        timestamp = round(time * 1e6) + cls.EPOCH_US
        if timestamp < 0:
            raise ValueError("Alpamayo observation time precedes its nonnegative timestamp epoch")
        return timestamp

    @classmethod
    def _observation_pose(cls, observation: Observation) -> tuple[float, float, float, float]:
        if not isinstance(observation, Observation):
            raise TypeError("Alpamayo history must contain Observation values")
        state = observation.vehicle_state
        try:
            pose = (float(state.x), float(state.y), float(state.yaw), float(observation.t))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Alpamayo requires vehicle_state with finite x, y and yaw") from error
        if not all(math.isfinite(value) for value in pose):
            raise ValueError("Alpamayo requires finite ego poses and observation times")
        cls._timestamp(pose[3])
        return pose

    def _append_observation(self, observation: Observation) -> None:
        from alpasim_driver.models.base import CameraFrame

        pose = self._observation_pose(observation)
        if self._poses:
            self._validate_interval(self._poses[-1][3], pose[3])
        images = {}
        for camera in self.camera_ids:
            image = observation.sensor_data.get(camera)
            if image is None:
                raise ValueError(f"Alpamayo observation is missing camera {camera!r}")
            images[camera] = deepcopy(image)
        self._poses.append(pose)
        timestamp = self._timestamp(observation.t)
        for camera, image in images.items():
            self._frames[camera].append(CameraFrame(timestamp_us=timestamp, image=image))

    def _prediction_input(self, observation: Observation):
        import numpy as np
        from alpasim_driver.models.base import DriveCommand, PredictionInput
        from alpasim_driver.navigation import determine_command_from_route
        from alpasim_grpc.v0 import common_pb2, egodriver_pb2

        self._append_observation(observation)
        if len(self._poses) != self.past_steps + 1:
            raise ValueError("Alpamayo requires 16 actual ego poses covering the last 1.5 seconds")
        camera_images = {}
        for camera in self.camera_ids:
            frames = list(self._frames[camera])
            required = (self.context_length - 1) * self.frame_subsample + 1
            if len(frames) < required:
                raise ValueError(f"Alpamayo camera {camera!r} requires {required} actual images")
            camera_images[camera] = frames[::self.frame_subsample]
        poses = list(self._poses)
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
        """Append the current capture and infer a rig-frame trajectory from camera and ego history."""
        if self._closed:
            raise RuntimeError("AlpamayoAVStack is closed")
        if not self._ready:
            raise RuntimeError("Reset AlpamayoAVStack with recorded past before requesting a decision")
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
            "selected_image_times": {camera: [(frame.timestamp_us - self.EPOCH_US) / 1e6
                                               for frame in frames]
                                     for camera, frames in inputs.camera_images.items()},
        }
        return command

    def component_log(self) -> dict[str, Any] | None:
        """Already-produced policy prediction and Control under ``policy`` and ``command``.

        Also includes route waypoints, semantic navigation command, a navigation note,
        real-history pose count and the selected image times on the simulator clock.
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
        origin = (round(observation.t * 1e6) if model_origin is None
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
        return [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), origin), *[
            (tuple(float(value) for value in position), rotation_to_quaternion(rotation), time)
            for position, rotation, time in zip(positions, rotations, times)
        ]]


__all__ = [
    'alpamayo_driving_adapter',
    'AlpamayoAVStack',
]
