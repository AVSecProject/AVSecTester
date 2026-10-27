"""AlpaSim CATK configuration, stateless RPC and owned live traffic sessions."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import uuid

import numpy as np

from avsectester.geometry.poses import (
    interpolate_transform, make_pose, quaternion_to_rotation, rotation_to_quaternion,
    validated_transform, validated_vector,
)
from avsectester.validation import (
    require_nonnegative_integer, require_positive_integer, require_positive_number,
)

TRAFFIC_STEP_US = 100_000


def _grpc_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import grpc
        from alpasim_grpc.v0 import common_pb2, traffic_pb2, traffic_pb2_grpc
    except ImportError as error:
        raise ImportError("AlpaSim traffic requires alpasim_grpc and grpcio; use the NuRec image") from error
    return grpc, common_pb2, traffic_pb2, traffic_pb2_grpc


def _call(method: Any, request: Any, timeout_s: float) -> Any:
    grpc, *_ = _grpc_modules()
    require_positive_number(timeout_s, "timeout_s")
    try:
        return method(request, timeout=timeout_s)
    except grpc.RpcError as error:
        raise RuntimeError(f"AlpaSim traffic RPC {error.code()}: {error.details()}") from error


def _trajectory_message(points: Sequence[tuple[int, np.ndarray]]) -> Any:
    _grpc, common, _traffic, _service = _grpc_modules()
    poses = []
    previous = None
    for timestamp, transform in points:
        require_nonnegative_integer(timestamp, "trajectory timestamp_us")
        if previous is not None and timestamp <= previous:
            raise ValueError("Traffic trajectory timestamps must be strictly increasing")
        previous = timestamp
        pose = validated_transform(transform)
        w, x, y, z = rotation_to_quaternion(pose[:3, :3])
        poses.append(common.PoseAtTime(
            timestamp_us=timestamp, pose=common.Pose(
                vec=common.Vec3(x=float(pose[0, 3]), y=float(pose[1, 3]), z=float(pose[2, 3])),
                quat=common.Quat(w=w, x=x, y=y, z=z))))
    if not poses:
        raise ValueError("Traffic trajectories must contain at least one pose")
    return common.Trajectory(poses=poses)


@dataclass(frozen=True)
class TrafficTrack:
    """A recorded box-center trajectory and its native dimensions and class."""

    track_id: str
    trajectory: tuple[tuple[int, np.ndarray], ...]
    extent: tuple[float, float, float]
    label_class: str
    is_static: bool

    def __post_init__(self) -> None:
        dimensions = np.asarray(self.extent, dtype=float)
        if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or (dimensions <= 0).any():
            raise ValueError("Traffic track extent must contain three positive dimensions")


def _traffic_window(trajectory: Sequence[tuple[int, np.ndarray]], first_us: int,
                    last_us: int) -> tuple[tuple[int, np.ndarray], ...]:
    """Clip a recorded track, interpolating only at the window boundaries."""
    if not trajectory:
        return ()
    times = np.asarray([int(stamp) for stamp, _ in trajectory], dtype=np.int64)
    first_us, last_us = max(first_us, int(times[0])), min(last_us, int(times[-1]))
    if first_us > last_us:
        return ()

    def at(stamp: int) -> np.ndarray:
        position = int(np.searchsorted(times, stamp))
        if position < len(times) and times[position] == stamp:
            return trajectory[position][1].copy()
        left, right = trajectory[position - 1], trajectory[position]
        return interpolate_transform(left[1], right[1], (stamp - left[0]) / (right[0] - left[0]))

    timestamps = sorted({first_us, last_us, *(int(stamp) for stamp in times
                                             if first_us < stamp < last_us)})
    return tuple((stamp, at(stamp)) for stamp in timestamps)


class AlpaSimTraffic:
    """Configure stateless RPC and open owned CATK sessions; seed is only logged."""

    def __init__(
        self, endpoint: str = "127.0.0.1:6200",
        history_steps: int = 16, rpc_timeout_s: float = 300.,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be a nonempty host:port string")
        require_positive_integer(history_steps, "history_steps")
        require_positive_number(rpc_timeout_s, "rpc_timeout_s")
        self.endpoint = endpoint
        self.history_steps = history_steps
        self.rpc_timeout_s = rpc_timeout_s

    def get_metadata(self, *, timeout_s: float | None = None) -> Any:
        """Fetch TrafficModuleMetadata over a short-lived connection."""
        grpc, common, _traffic, service = _grpc_modules()
        with grpc.insecure_channel(self.endpoint) as channel:
            return _call(service.TrafficServiceStub(channel).get_metadata, common.Empty(),
                         self.rpc_timeout_s if timeout_s is None else timeout_s)

    def get_available_scenes(self, *, timeout_s: float | None = None) -> tuple[str, ...]:
        """Return the scene IDs served by this endpoint."""
        grpc, common, _traffic, service = _grpc_modules()
        with grpc.insecure_channel(self.endpoint) as channel:
            result = _call(service.TrafficServiceStub(channel).get_available_scenes, common.Empty(),
                           self.rpc_timeout_s if timeout_s is None else timeout_s)
        return tuple(result.scene_ids)

    def served_scenes(self, timeout_s: float = 2.0) -> tuple[str, ...] | None:
        """Query scene support; unavailable endpoints and deadline expiry return None."""
        require_positive_number(timeout_s, "timeout_s")
        grpc, *_ = _grpc_modules()
        try:
            return self.get_available_scenes(timeout_s=timeout_s)
        except RuntimeError as error:
            cause = error.__cause__
            if (isinstance(cause, grpc.RpcError)
                    and cause.code() in {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}):
                return None
            raise

    def open_session(
        self, *, scene_id: str, random_seed: int, start_us: int, end_us: int,
        ego_extent: tuple[float, float, float],
        ego_trajectory: Sequence[tuple[int, np.ndarray]],
        tracks: Sequence[TrafficTrack],
    ) -> TrafficSession:
        """Validate support and open a session with EGO first and clipped native tracks."""
        require_nonnegative_integer(random_seed, "random_seed")
        require_nonnegative_integer(start_us, "start_us")
        require_nonnegative_integer(end_us, "end_us")
        if scene_id not in self.get_available_scenes():
            raise RuntimeError(f"AlpaSim traffic does not serve scene {scene_id!r}")
        history_us = int(self.get_metadata().minimum_history_length_us)
        expected_us = self.history_steps * TRAFFIC_STEP_US
        if history_us != expected_us:
            raise RuntimeError(f"AlpaSim traffic minimum_history_length_us={history_us}, expected {expected_us}")
        grpc, common, traffic, service = _grpc_modules()
        ego = TrafficTrack("EGO", tuple(ego_trajectory), ego_extent, "", False)
        first_us = start_us - (self.history_steps - 1) * TRAFFIC_STEP_US
        selected = []
        for track in tracks:
            if any(right[0] <= left[0] for left, right in zip(track.trajectory, track.trajectory[1:])):
                raise ValueError("Traffic trajectory timestamps must be strictly increasing")
            clipped = _traffic_window(track.trajectory, first_us, end_us)
            if clipped:
                selected.append(TrafficTrack(track.track_id, clipped, track.extent,
                                             track.label_class, track.is_static))
        objects = [traffic.ObjectTrajectory(
            object_id=track.track_id,
            aabb=common.AABB(size_x=track.extent[0], size_y=track.extent[1], size_z=track.extent[2]),
            trajectory=_trajectory_message(track.trajectory), is_static=track.is_static,
            label_class=track.label_class) for track in (ego, *selected)]
        session_uuid = str(uuid.uuid4())
        handover = start_us + TRAFFIC_STEP_US
        channel = grpc.insecure_channel(self.endpoint)
        try:
            _call(service.TrafficServiceStub(channel).start_session, traffic.TrafficSessionRequest(
                session_uuid=session_uuid, scene_id=scene_id, random_seed=random_seed,
                logged_object_trajectories=objects, handover_time_us=handover), self.rpc_timeout_s)
        except BaseException:
            channel.close()
            raise
        return TrafficSession(channel, session_uuid, handover, selected,
                              history_steps=self.history_steps, rpc_timeout_s=self.rpc_timeout_s)


class TrafficSession:
    """One live CATK session, its channel and complete constant-velocity override history."""

    def __init__(self, channel: Any, session_uuid: str, handover_time_us: int,
                 tracks: Sequence[TrafficTrack], *, history_steps: int, rpc_timeout_s: float) -> None:
        self.uuid = session_uuid
        self.handover_time_us = handover_time_us
        self.track_ids = tuple(track.track_id for track in tracks)
        self._static_ids = {track.track_id for track in tracks if track.is_static}
        self._history_steps = history_steps
        self._rpc_timeout_s = rpc_timeout_s
        self._channel = channel
        self._override_history: dict[str, list[tuple[int, np.ndarray]]] = {}

    def record_override(self, track_id: str, time_us: int,
                        world_from_box: np.ndarray, velocity: Sequence[float]) -> None:
        """Record an override, backfilling only its first constant-velocity history."""
        if self._channel is None:
            raise RuntimeError(f"AlpaSim traffic session {self.uuid!r} is not open")
        if track_id not in self.track_ids:
            raise ValueError(f"Unknown traffic session track {track_id!r}")
        require_nonnegative_integer(time_us, "time_us")
        pose = validated_transform(world_from_box)
        speed = validated_vector(velocity, "velocity")
        if track_id not in self._override_history:
            history = []
            for k in range(self._history_steps - 1, 0, -1):
                past = pose.copy()
                past[:3, 3] -= speed * (k * TRAFFIC_STEP_US / 1e6)
                history.append((time_us - k * TRAFFIC_STEP_US, past))
            self._override_history[track_id] = history
        self._append_override(track_id, time_us, pose)

    def _append_override(self, track_id: str, time_us: int, pose: np.ndarray) -> None:
        history = self._override_history[track_id]
        sample = (time_us, pose.copy())
        if history and history[-1][0] == time_us:
            history[-1] = sample
        else:
            history.append(sample)

    def simulate(self, time_query_us: int, ego_world_from_box: np.ndarray,
                 override_world_from_box: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Send current EGO and full override histories; return current movable poses.

        Static, unknown and overridden tracks are ignored. Any error closes the session;
        a returned movable track must contain the exact requested timestamp.
        """
        try:
            if self._channel is None:
                raise RuntimeError(f"AlpaSim traffic session {self.uuid!r} is not open")
            require_nonnegative_integer(time_query_us, "time_query_us")
            if set(override_world_from_box) != set(self._override_history):
                raise ValueError("override keys must match the recorded traffic overrides exactly")
            for identity, pose in override_world_from_box.items():
                self._append_override(identity, time_query_us, validated_transform(pose))
            _grpc, _common, traffic, service = _grpc_modules()
            updates = {"EGO": ((time_query_us, ego_world_from_box),),
                       **{key: tuple(value) for key, value in self._override_history.items()}}
            request = traffic.TrafficRequest(
                session_uuid=self.uuid, time_query_us=time_query_us,
                object_trajectory_updates=[traffic.ObjectTrajectoryUpdate(
                    object_id=key, trajectory=_trajectory_message(points))
                    for key, points in updates.items()])
            response = _call(service.TrafficServiceStub(self._channel).simulate, request,
                             self._rpc_timeout_s)
            result = {}
            for update in response.object_trajectory_updates:
                identity = update.object_id
                if (identity not in self.track_ids or identity in self._static_ids
                        or identity in self._override_history):
                    continue
                matches = [point.pose for point in update.trajectory.poses
                           if point.timestamp_us == time_query_us]
                if not matches:
                    raise RuntimeError(f"AlpaSim traffic track {identity!r} has no pose at {time_query_us}")
                pose = matches[0]
                result[identity] = validated_transform(make_pose(
                    quaternion_to_rotation(pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z),
                    (pose.vec.x, pose.vec.y, pose.vec.z)))
            return result
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(f"Closing the traffic session also failed: {cleanup_error}")
            raise

    def close(self) -> None:
        """Close once and release the channel even when the close RPC fails."""
        channel, self._channel = self._channel, None
        if channel is None:
            return
        try:
            _grpc, _common, traffic, service = _grpc_modules()
            _call(service.TrafficServiceStub(channel).close_session,
                  traffic.TrafficSessionCloseRequest(session_uuid=self.uuid), self._rpc_timeout_s)
        finally:
            channel.close()


__all__ = [
    'TrafficTrack',
    'AlpaSimTraffic',
    'TrafficSession',
]
