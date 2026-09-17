"""Alpamayo end-to-end camera policy as an :class:`~avsectester.backend.AVStack`.

Wraps AlpaSim's own Alpamayo model (``alpasim_driver.models.alpamayo1_5_model.Alpamayo15Model``) so
it plugs into ``run(backend, stack, frames)`` unchanged: an :class:`~avsectester.plane.Observation`
(a rendered camera + ego state) in, a :class:`~avsectester.plane.Control` (the planned trajectory) out.

Alpamayo is a trajectory-output policy: ``predict(PredictionInput) -> ModelPrediction`` with
``candidate_positions`` of shape ``(K, T, 3)`` in the **rig frame**. We keep a short temporal buffer
of camera frames + ego poses (the model's context) and emit the selected candidate as
``Control.trajectory``; the world backend's trajectory follower turns that into motion.

Runs in the AlpaSim **driver env** (Python 3.12 + ``alpasim_driver`` and the Alpamayo repos), NOT the
CARLA/avsec 3.10 env — so all heavy imports are lazy and this module imports fine without them. The
checkpoint defaults to the already-downloaded local model to avoid any re-download.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from avsectester.backend import AVStack
from avsectester.plane import Control, Observation

# already on disk (see /workspace/hdd/models/huggingface); a local path skips any HF download
DEFAULT_CHECKPOINT = "/workspace/hdd/models/huggingface/nvidia/Alpamayo-1.5-10B"


class AlpamayoAVStack(AVStack):
    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        camera_ids: list[str] | None = None,
        device: str = "cuda:0",
        context_length: int = 4,
        output_frequency_hz: int = 10,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.camera_ids = camera_ids or ["camera_front_wide_120fov"]
        self.device = device
        self.context_length = context_length
        self.output_frequency_hz = output_frequency_hz
        self._model = None
        self._frames: dict[str, deque] = {c: deque(maxlen=context_length) for c in self.camera_ids}
        self._prev_plan = None
        self._iseed = 0

    def _load(self):
        """Lazily build the Alpamayo model from the driver package (heavy: torch + a 10B checkpoint)."""
        import torch
        from alpasim_driver.models.alpamayo1_5_model import Alpamayo15Model
        from alpasim_driver.schema import ModelConfig

        cfg = ModelConfig(
            model_type="alpamayo1_5",
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            num_trajectory_samples=1,
        )
        self._model = Alpamayo15Model.from_config(
            model_cfg=cfg,
            device=torch.device(self.device),
            camera_ids=self.camera_ids,
            context_length=self.context_length,
            output_frequency_hz=self.output_frequency_hz,
        )

    def reset(self, observation: Observation) -> None:
        if self._model is None:
            self._load()
        for buf in self._frames.values():
            buf.clear()
        self._prev_plan = None
        self._iseed = 0

    HISTORY_SPAN_S = 1.6  # must exceed the model's required 1.5 s ego-history span
    EPOCH_US = 10_000_000  # offset so backward-history timestamps stay positive (proto fixed64)

    def _prediction_input(self, obs: Observation):
        from alpasim_driver.models.base import CameraFrame, DriveCommand, PredictionInput

        t_us = int(obs.t * 1e6) + self.EPOCH_US
        for cam in self.camera_ids:
            image = obs.sensor_data.get(cam)  # HWC uint8 (np or torch) rendered by NuRec
            if image is not None:
                self._frames[cam].append(CameraFrame(timestamp_us=t_us, image=image))
        # camera context: pad to context_length at startup; ego history: span HISTORY_SPAN_S backward
        return PredictionInput(
            camera_images={c: self._pad(self._frames[c]) for c in self.camera_ids},
            command=DriveCommand.STRAIGHT,
            speed=float(obs.ego_speed),
            acceleration=0.0,
            ego_pose_history=self._ego_history_backward(obs, t_us),
            inference_seed=self._iseed,
            previous_plan=self._prev_plan,
            route=None,
        )

    def _pad(self, items) -> list:
        """Left-pad a temporal buffer to ``context_length`` by repeating its earliest element."""
        items = list(items)
        if not items:
            return items
        return [items[0]] * (self.context_length - len(items)) + items

    def _ego_history_backward(self, obs: Observation, t_us: int) -> list:
        """Ego poses spanning HISTORY_SPAN_S before now, constant-velocity backward from the current
        state (a valid startup prior; a longer run can substitute the real accumulated trajectory)."""
        import math

        s = obs.vehicle_state
        x, y, yaw = getattr(s, "x", 0.0), getattr(s, "y", 0.0), getattr(s, "yaw", 0.0)
        speed = float(obs.ego_speed)
        dt = 1.0 / self.output_frequency_hz
        n = int(self.HISTORY_SPAN_S / dt) + 1
        return [
            self._pose_at(
                x - speed * math.cos(yaw) * (k * dt),
                y - speed * math.sin(yaw) * (k * dt),
                yaw,
                t_us - int(k * dt * 1e6),
            )
            for k in range(n, -1, -1)  # oldest (-HISTORY_SPAN_S) .. current (0)
        ]

    @staticmethod
    def _pose_at(x: float, y: float, yaw: float, t_us: int):
        import math

        from alpasim_grpc.v0 import common_pb2

        return common_pb2.PoseAtTime(
            pose=common_pb2.Pose(
                vec=common_pb2.Vec3(x=x, y=y, z=0.0),
                quat=common_pb2.Quat(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2)),
            ),
            timestamp_us=t_us,
        )

    def __call__(self, observation: Observation) -> Control:
        if self._model is None:
            self._load()
        prediction = self._model.predict(self._prediction_input(observation))
        self._iseed += 1
        self._prev_plan = getattr(prediction, "selected_plan", None)
        return Control(trajectory=self._to_waypoints(prediction, observation))

    def _to_waypoints(self, prediction: Any, obs: Observation) -> list:
        """Selected candidate (T,3) rig-frame positions -> [((x,y,z),(w,x,y,z),t_us), ...]."""
        import numpy as np

        k = int(getattr(prediction, "selected_index", 0))
        positions = np.asarray(prediction.candidate_positions)[k]  # (T, 3), rig frame
        # emit waypoint times in the backend's sim clock (not the model's epoch'd time) at the
        # model's output frequency, so the trajectory follower interpolates consistently.
        base_us = int(obs.t * 1e6)
        dt_us = int(1e6 / self.output_frequency_hz)
        return [
            (tuple(float(v) for v in xyz), (1.0, 0.0, 0.0, 0.0), base_us + (i + 1) * dt_us)
            for i, xyz in enumerate(positions)
        ]
