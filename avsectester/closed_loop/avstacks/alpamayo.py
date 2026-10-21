"""Alpamayo end-to-end camera policy as an :class:`~avsectester.closed_loop.avstack.AVStack`.

Wraps AlpaSim's own Alpamayo model (``alpasim_driver.models.alpamayo1_5_model.Alpamayo15Model``) so
it plugs into ``ClosedLoopRunner``: an :class:`~avsectester.closed_loop.datatypes.Observation`
(a rendered camera + ego state) in, a :class:`~avsectester.closed_loop.datatypes.Control` (the planned trajectory) out.

Alpamayo is a trajectory-output policy: ``predict(PredictionInput) -> ModelPrediction`` with
``candidate_positions`` of shape ``(K, T, 3)`` in the **rig frame**. We keep a short temporal buffer
of camera frames + ego poses (the model's context) and emit the selected candidate as
``Control.trajectory``; the simulator's trajectory follower turns that into motion.

Runs in the AlpaSim **driver env** (Python 3.12 + ``alpasim_driver`` and the Alpamayo repos), NOT the
CARLA/avsec 3.11 env — so all heavy imports are lazy and this module imports fine without them. The
checkpoint is supplied explicitly or through ``AVSECTESTER_ALPAMAYO_CHECKPOINT``.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from collections.abc import Mapping
from typing import Any

from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.datatypes import Control, Observation


class AlpamayoAVStack(AVStack):
    provides_detections = False

    def __init__(
        self,
        checkpoint_path: str | None = None,
        camera_ids: list[str] | None = None,
        device: str = "cuda:0",
        context_length: int = 4,
        output_frequency_hz: int = 10,
        model: Any | None = None,
    ) -> None:
        from avsectester.runlog.codecs import default_codecs
        from avsectester.runlog import Recorder

        self.checkpoint_path = (
            checkpoint_path if checkpoint_path is not None
            else os.environ.get("AVSECTESTER_ALPAMAYO_CHECKPOINT")
        )
        self.camera_ids = camera_ids or ["camera_front_wide_120fov"]
        self.device = device
        self.context_length = context_length
        self.output_frequency_hz = output_frequency_hz
        # Separate adapters may share loaded weights while retaining independent input history.
        self._model = model
        self._owns_model = model is None
        self._closed = False
        self._frames: dict[str, deque] = {c: deque(maxlen=context_length) for c in self.camera_ids}
        self._prev_plan = None
        self._iseed = 0
        self.last_reasoning: str | None = None
        self._last_log: dict[str, Any] | None = None
        self._log_capture = Recorder(codecs=default_codecs())

    @property
    def command_modes(self) -> frozenset[str]:
        return frozenset({"trajectory"})

    @property
    def supported_stages(self) -> frozenset[str]:
        """The end-to-end policy has no separate perception or tracking stage."""
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
        )
        self._model = Alpamayo15Model.from_config(
            model_cfg=cfg,
            device=torch.device(self.device),
            camera_ids=self.camera_ids,
            context_length=self.context_length,
            output_frequency_hz=self.output_frequency_hz,
        )

    def reset(self, observation: Observation) -> None:
        if self._closed:
            raise RuntimeError("AlpamayoAVStack is closed")
        if self._model is None:
            self._load()
        for buf in self._frames.values():
            buf.clear()
        self._prev_plan = None
        self._iseed = 0
        self.last_reasoning = None
        self._last_log = None

    def close(self) -> None:
        """Release this adapter's references and any model it loaded itself.

        Closing is terminal and idempotent. An injected model is borrowed: another
        adapter may share it, so its close method and CUDA allocator are untouched.
        """
        if self._closed:
            return
        self._closed = True
        model, self._model = self._model, None
        owned = self._owns_model and model is not None
        for history in self._frames.values():
            history.clear()
        self._prev_plan = None
        self._last_log = None
        self.last_reasoning = None
        self._iseed = 0
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
        if self._closed:
            raise RuntimeError("AlpamayoAVStack is closed")
        if self._model is None:
            self._load()
        inputs = self._emit("policy.pre", self._prediction_input(observation))
        prediction = self._emit("policy.post", self._model.predict(inputs))
        policy_record = self._log_capture.capture(prediction)
        self._iseed += 1
        self._prev_plan = getattr(prediction, "selected_plan", None)
        self.last_reasoning = getattr(prediction, "reasoning_text", None)  # chain-of-causation text, if any
        command = Control(trajectory=self._to_waypoints(prediction, observation))
        # Record the already-produced policy output and command without extra inference.
        self._last_log = {"policy": policy_record, "command": self._log_capture.capture(command)}
        return command

    def component_log(self) -> dict[str, Any] | None:
        """Already-produced policy prediction and Control under ``policy`` and ``command``.

        The prediction retains candidate positions, selected index and available reasoning;
        recording does not request additional inference.
        """
        return self._last_log

    def _to_waypoints(self, prediction: Any, obs: Observation) -> list:
        """Selected candidate (T,3) rig-frame positions -> [((x,y,z),(w,x,y,z),t_us), ...]."""
        import numpy as np

        k = int(getattr(prediction, "selected_index", 0))
        positions = np.asarray(prediction.candidate_positions)[k]  # (T, 3), rig frame
        # emit waypoint times in the simulator's clock (not the model's epoch'd time) at the
        # model's output frequency, so the trajectory follower interpolates consistently.
        base_us = int(obs.t * 1e6)
        dt_us = int(1e6 / self.output_frequency_hz)
        return [
            (tuple(float(v) for v in xyz), (1.0, 0.0, 0.0, 0.0), base_us + (i + 1) * dt_us)
            for i, xyz in enumerate(positions)
        ]
