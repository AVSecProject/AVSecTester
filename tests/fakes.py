"""Small deterministic worlds and drivers shared by public-API tests.

These are alternative implementations of the framework's extension interfaces. Tests use the
real experiment, simulator lifecycle, hooks, recorder and services around them. Native boundary
doubles below replace only software and hardware outside this repository.
"""
from __future__ import annotations

from copy import deepcopy
from collections import deque
from threading import Thread
from time import monotonic, sleep, time_ns
from types import ModuleType, SimpleNamespace
import sys
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from avsectester.attack import Attack
from avsectester.attack.runtime import AttackRuntime
from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState
from avsectester.closed_loop.simulators.messages import MessageChannel
from avsectester.closed_loop.timing import HistoryRequirement, TimingCapabilities
from avsectester.data_types import ActorState, Control, Detection, Observation, WorldSnapshot
from avsectester.closed_loop.simulators.base import SensorGeometry, SensorView, Simulator, SimulatorRuntime
from avsectester.closed_loop.simulators.nurec.renderer import Renderer
from avsectester.datasets import Dataset, FrameData, FrameReference, ScenarioDescriptor, ScenarioStart
from avsectester.geometry.cameras import BODY_FROM_OPTICAL, PinholeCamera
from avsectester.geometry.poses import interpolate_transform
from avsectester.maps import Lane, ScenarioMap, StopLine, TrafficControl


CAMERA = PinholeCamera(np.array([[50., 0., 50.], [0., 50., 40.], [0., 0., 1.]]), 100, 80)
CAM_FROM_WORLD = np.linalg.inv(BODY_FROM_OPTICAL)


def actor_state(position=(10., 0., 1.), speed=0., *, yaw=0., extent=(4., 2., 2.), category="vehicle"):
    pose = np.eye(4)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    pose[:2, :2] = [[cosine, -sine], [sine, cosine]]
    pose[:3, 3] = position
    return ActorState(pose, (speed, 0., 0.), category, extent)


class MemoryDataset(Dataset):
    """A calibrated camera, one stationary lead vehicle and an evenly spaced timeline."""

    def __init__(self, scenario_ids=("scene",), decision_steps=6, time_step=.1,
                 maps=None, scenarios_error=None, ego_position=(0., 0., 0.)):
        self.scenario_ids = tuple(scenario_ids)
        self.decision_steps, self.time_step = decision_steps, time_step
        self.maps = dict(maps or {})
        self.scenarios_error = scenarios_error
        self.ego_position = np.asarray(ego_position, dtype=float)

    def scenario_map(self, scenario_id):
        if scenario_id not in self.scenario_ids:
            raise KeyError(scenario_id)
        return self.maps[scenario_id] if scenario_id in self.maps else super().scenario_map(scenario_id)

    def scenarios(self):
        if self.scenarios_error == "interrupt":
            raise KeyboardInterrupt()
        if self.scenarios_error is not None:
            raise RuntimeError(self.scenarios_error)
        return iter(ScenarioDescriptor(name, MemoryDataset.__name__) for name in self.scenario_ids)

    def frames(self, scenario_id):
        if scenario_id not in self.scenario_ids:
            raise KeyError(scenario_id)
        return tuple(FrameReference(scenario_id, i, i, i * self.time_step)
                     for i in range(self.decision_steps + 1))

    def frame_data(self, scenario_id, frame):
        frame = self.checked_frame(scenario_id, frame)
        return FrameData(frame, actors={"lead": actor_state()},
                         ego=actor_state(self.ego_position + [frame.timestamp, 0., 0.], speed=1.))

    def scenario_start(self, scenario_id, frame):
        frame = self.checked_frame(scenario_id, frame)
        return ScenarioStart(
            MemoryDataset.__name__, scenario_id, frame, {"time_step": self.time_step},
            recorded_ego=tuple(self.frame_data(scenario_id, item).ego
                               for item in self.frames(scenario_id)),
            scenario_map=self.scenario_map(scenario_id),
        )

    def native(self, scenario_id):
        return {"scenario_id": scenario_id}


class StopApproachDataset(MemoryDataset):
    """A real Dataset extension with a straight approach and a recorded pause."""

    def __init__(self, scenario_map, stopped=True):
        self.positions = np.r_[np.arange(59.), [58., 58.], np.arange(59., 81.)]
        self.stopped = stopped
        super().__init__(decision_steps=len(self.positions) - 1, maps={"scene": scenario_map})

    def frame_data(self, scenario_id, frame):
        frame = self.checked_frame(scenario_id, frame)
        speed = 10.
        if self.stopped and 58 <= frame.position <= 60:
            speed = 0.
        return FrameData(frame, ego=actor_state((self.positions[frame.position], 0., 0.), speed))


def stop_approach_map(*, line_kind="stop_sign", facing=True, line_offset_y=0.,
                      linked_sign=True, sign_x=60.):
    """Map geometry for a straight recorded stop approach."""
    pose = np.diag([-1., -1., 1., 1.])
    pose[:3, 3] = [sign_x, -3., 2.]
    if not facing:
        pose[:3, :3] = np.eye(3)
    control = TrafficControl("sign", "stop_sign", "MUTCD_R1_STOP", pose, (.05, .8, .8),
                             "face_center", lane_ids=("lane",))
    lane = Lane("lane", np.array([[0., 0., 0.], [80., 0., 0.]]),
                np.array([[0., 3., 0.], [80., 3., 0.]]),
                np.array([[0., -3., 0.], [80., -3., 0.]]))
    line = StopLine("line", np.array([[60., -3. + line_offset_y, 0.],
                                     [60., 3. + line_offset_y, 0.]]), "ENTRY",
                    kind=line_kind, lane_ids=("lane",),
                    control_id="sign" if linked_sign else None)
    return ScenarioMap(lanes={"lane": lane}, traffic_controls={"sign": control},
                       stop_lines={"line": line})


class MemoryImageRenderer(Renderer):
    """An in-memory renderer with a chosen optical lens model."""

    cameras = ["front"]

    def __init__(self, camera, camera_id="front", rig_from_camera=None, provide_geometry=True):
        self.camera = camera
        self.cameras = [camera_id]
        self.rig_from_camera = rig_from_camera
        self.provide_geometry = provide_geometry

    def camera_model(self, camera=None):
        if camera is not None and camera not in self.cameras:
            raise ValueError(f"Unknown camera {camera!r}")
        return self.camera if self.provide_geometry else None

    def cam_from_world(self, world_from_rig, camera=None):
        rig_from_camera = BODY_FROM_OPTICAL.copy()
        rig_from_camera[2, 3] = 1.5
        if self.rig_from_camera is not None:
            rig_from_camera = self.rig_from_camera
        return np.linalg.inv(world_from_rig @ rig_from_camera)

    def render(self, world_from_rig, timestamp_us, camera, dynamic_objects=(), **exposure):
        return np.full((self.camera.height, self.camera.width, 3), 30, np.uint8)


class MemorySimulator(Simulator):
    """An image-producing actuator world, with explicit optional external failures."""

    dataset_kinds = (MemoryDataset.__name__,)
    can_render = True
    can_replay = True
    can_observe_past = True
    can_insert_objects = True
    command_modes = frozenset({"actuator"})
    supported_world_settings = frozenset({"weather", "lighting", "fail", "interrupt"})

    def __init__(self, time_step=.1, capture_ticks=0, world_key=None,
                 extra_sensors=None, camera_geometries=None, sensor_depths=None, actors=None, runtime_error=None, start_error=None,
                 can_replay=True, can_observe_past=True, can_insert_objects=True, can_render=True,
                 observation_lead_steps=0, captures_between_decisions=True, can_repeat_runs=False):
        self.time_step, self.capture_ticks = time_step, capture_ticks
        self.captures_between_decisions = captures_between_decisions
        self._observation_lead_steps = observation_lead_steps
        self.world_key = world_key
        self.runtime_error = runtime_error
        self.start_error = start_error
        self.can_render = can_render
        self.can_replay = can_replay
        self.can_observe_past = can_observe_past
        self.can_insert_objects = can_insert_objects
        self.can_repeat_runs = can_repeat_runs
        self.extra_sensors = deepcopy(extra_sensors or {})
        self.camera_geometries = deepcopy(camera_geometries if camera_geometries is not None else {
            "front": SensorGeometry("front", CAMERA, CAM_FROM_WORLD),
        })
        self.sensor_depths = deepcopy(sensor_depths or {})
        self.initial_actors = deepcopy(actors if actors is not None else {"lead": actor_state()})

    @property
    def observation_lead_steps(self):
        return self._observation_lead_steps

    @property
    def decision_time_step(self):
        return self.time_step

    @property
    def timing_capabilities(self):
        return TimingCapabilities(captures_between_decisions=self.captures_between_decisions,
                                  motion_reports=True, recorded_ego_motion=self.can_replay,
                                  camera_exposure=True)

    @property
    def identity(self):
        return {**super().identity, "world_key": self.world_key}

    def validate_start(self, start):
        super().validate_start(start)
        if self.start_error is not None:
            raise RuntimeError(self.start_error)

    def create_runtime(self, seed=None):
        if self.runtime_error is not None:
            raise RuntimeError(self.runtime_error)
        return MemorySimulatorRuntime(self, seed)


class MemorySimulatorRuntime(SimulatorRuntime):
    """Native operations for the shared world. The production base owns reset and step."""

    def __init__(self, simulator, seed=None):
        super().__init__(simulator)
        self._native = {"objects": {}, "world": {}}

    @property
    def world_key(self):
        return self.simulator.world_key if self.simulator.world_key is not None else ("memory", id(self))

    @property
    def native(self):
        return self._native

    def _initialize_world(self, start, initialization, environment):
        self.source_start = start
        self.motion_start = self.motion_end = None
        self.motion_start_us = self.motion_end_us = None
        self.position, self.distance, self.speed = 0, 0., 1.
        self.actors = deepcopy(self.simulator.initial_actors)
        self._native = {"objects": {}, "world": deepcopy(environment)}

    def _replay_position(self, start):
        return 0 if start is None else start.frame.position

    def _replay_tick(self):
        self.position += 1
        self.distance += self.simulator.time_step

    def _hold_tick(self):
        pass

    def _check_initialization(self, initialization):
        # Memory frame data and replay follow the same exact prescribed motion.
        pass

    def _observe(self):
        return self._observe_sensors(tuple(self.sensors))

    def _sensor_data(self, selected, frame_position):
        color = 20 + int(frame_position)
        images = {sensor: np.full((geometry.camera.height, geometry.camera.width, 3), color, np.uint8)
                  for sensor, geometry in self.simulator.camera_geometries.items() if sensor in selected}
        return {**images, **{key: deepcopy(value) for key, value in self.simulator.extra_sensors.items()
                            if key in selected}}

    def _observe_sensors(self, sensor_ids):
        return Observation(self.state().t, self.position, self._sensor_data(set(sensor_ids), self.position),
                           ego_speed=self.speed)

    def _recorded_ego(self, position):
        left = int(position)
        actor = self.source_start.recorded_ego[left]
        if position == left:
            return actor
        after = self.source_start.recorded_ego[left + 1]
        fraction = position - left
        return replace(actor, pose=interpolate_transform(actor.pose, after.pose, fraction),
                       velocity=tuple((1 - fraction) * np.asarray(actor.velocity)
                                      + fraction * np.asarray(after.velocity)))

    def _recorded_position(self, time_us):
        grid, remainder = divmod(time_us, self.decision_period_us)
        return self.source_start.frame.position + grid + remainder / self.decision_period_us

    def _recorded_pose_state(self, time_us):
        position = self._recorded_position(time_us)
        ego = self._recorded_ego(position)
        speed = float(np.linalg.norm(ego.velocity))
        state = PlanarEgoState(x=float(ego.pose[0, 3]), y=float(ego.pose[1, 3]),
                              yaw=float(np.arctan2(ego.pose[1, 0], ego.pose[0, 0])),
                              speed=speed, t=time_us / 1e6)
        return WorldSnapshot(time_us / 1e6, time_us // self.decision_period_us,
                             state, speed, ego=ego)

    def _recorded_state(self, time_us):
        return replace(self._recorded_pose_state(time_us), actors=deepcopy(self.simulator.initial_actors))

    def _observe_recorded(self, time_us, sensors):
        position = self._recorded_position(time_us)
        state = self._recorded_pose_state(time_us)
        selected = self.sensors if sensors is None else set(sensors)
        return Observation(state.t, state.frame, self._sensor_data(selected, position),
                           vehicle_state=state.vehicle_state, ego_speed=state.ego_speed)

    def _recorded_sensor_geometry(self, time_us, sensor):
        distance = self._recorded_ego(self._recorded_position(time_us)).pose[0, 3]
        return self._camera_geometry(sensor, distance)

    def _recorded_sensor_view(self, time_us, sensor, observation):
        geometry = self._recorded_sensor_geometry(time_us, sensor)
        return SensorView(sensor, observation.sensor_data[sensor], geometry.camera, geometry.cam_from_world,
                          {}, self._recorded_ego(self._recorded_position(time_us)), observation.t, observation.frame,
                          depth=deepcopy(self.simulator.sensor_depths.get(sensor)),
                          sensor_from_world=BODY_FROM_OPTICAL @ geometry.cam_from_world)

    def _state(self):
        return self._memory_state()

    def _memory_state(self):
        return WorldSnapshot((self.position * self.decision_period_us) / 1e6, self.position,
                             ego_speed=self.speed, actors=deepcopy(self.actors),
                             ego=actor_state((self.distance, 0., 0.), self.speed))

    def _advance_native(self, control):
        world = self.native["world"]
        if world.get("interrupt"):
            raise KeyboardInterrupt("memory simulation interrupted")
        if world.get("fail"):
            raise RuntimeError("memory simulation failed")
        self.motion_start = self._memory_state()
        self.motion_start_us = self._interval.start_us
        rain = .5 if world.get("weather") == "rain" else 1.
        self.speed = max(0., self.speed + (control.throttle * rain - control.brake) * self.simulator.time_step)
        self.distance += self.speed * self.simulator.time_step
        self.position += 1
        self.motion_end = self._memory_state()
        self.motion_end_us = self._interval.end_us

    def _advance_recorded_native(self):
        self.motion_start = self._memory_state()
        self.motion_start_us = self._interval.start_us
        next_position = self.position + 1
        state = self.source_start.recorded_ego[next_position]
        self.position = next_position
        self.distance = float(state.pose[0, 3])
        self.speed = float(np.linalg.norm(state.velocity))
        self.motion_end = self._memory_state()
        self.motion_end_us = self._interval.end_us

    def _reported_state_at(self, time_us):
        first, last = self.motion_start, self.motion_end
        if time_us == self.motion_end_us:
            return deepcopy(last)
        fraction = (time_us - self.motion_start_us) / (self.motion_end_us - self.motion_start_us)
        pose = interpolate_transform(first.ego.pose, last.ego.pose, fraction)
        speed = first.ego_speed + fraction * (last.ego_speed - first.ego_speed)
        ego = replace(first.ego, pose=pose, velocity=(speed, 0., 0.))
        return replace(first, t=time_us / 1e6, ego_speed=speed, ego=ego,
                       actors=deepcopy(last.actors), objects=deepcopy(last.objects))

    def _activate_reported_state(self, time_us, state):
        self.position = state.frame
        self.speed = state.ego_speed
        self.distance = float(state.ego.pose[0, 3])
        self.actors = deepcopy(state.actors)

    @property
    def sensors(self):
        return frozenset({*self.simulator.camera_geometries, *self.simulator.extra_sensors})

    @property
    def camera_sensors(self):
        return frozenset(self.simulator.camera_geometries)

    def _sensor_geometry(self, sensor):
        return self._camera_geometry(sensor, self.distance)

    def _camera_geometry(self, sensor, distance):
        if sensor not in self.camera_sensors:
            raise ValueError(f"Unknown memory camera: {sensor}")
        geometry = self.simulator.camera_geometries[sensor]
        transform = geometry.cam_from_world.copy()
        transform[:, 3] -= transform[:, 0] * distance
        return SensorGeometry(sensor, geometry.camera, transform)

    def _capture_view_builder(self, sensor, observation, state):
        if sensor not in self.sensors:
            raise ValueError(f"Unknown memory sensor: {sensor}")
        data = observation.sensor_data[sensor]
        body_from_world = np.eye(4)
        body_from_world[0, 3] = -self.distance
        camera, cam_from_world = None, None
        if sensor in self.camera_sensors:
            geometry = self._sensor_geometry(sensor)
            camera, cam_from_world = geometry.camera, geometry.cam_from_world
            body_from_world = BODY_FROM_OPTICAL @ cam_from_world
        elif not isinstance(data, np.ndarray):
            data = getattr(data, "data", data)
        view = SensorView(sensor, data, camera, cam_from_world,
                          state.actors,
                          state.ego, state.t, state.frame,
                          depth=deepcopy(self.simulator.sensor_depths.get(sensor)),
                          sensor_from_world=body_from_world)
        return lambda: view

    def set_actor_state(self, actor_id, state):
        if actor_id not in self.actors:
            raise KeyError(actor_id)
        self.actors[actor_id] = deepcopy(state)
        self.invalidate_state()

    def control_actor(self, actor_id, control):
        state = self.actors[actor_id]
        self.actors[actor_id] = replace(state, velocity=(float(control.throttle), 0., 0.))
        self.invalidate_state()

    def _capture_frame_data(self, frame):
        state = self.state()
        return FrameData(frame, actors=state.actors, ego=state.ego)

    def close(self):
        super().close()
        self.native["objects"].clear()


class ExternallyControlledMemorySimulator(MemorySimulator):
    """A world whose ego motion comes from an external controller."""

    command_modes = frozenset()
    can_start_runs_after_origin = False

    def create_runtime(self, seed=None):
        return ExternallyControlledMemoryRuntime(self, seed)


class ExternallyControlledMemoryRuntime(MemorySimulatorRuntime):
    """Model an external controller through the simulator extension contract."""

    def _advance_native(self, control):
        if control is not None:
            raise ValueError("External control does not accept framework commands")
        super()._advance_native(Control(throttle=.2))

    def decision_telemetry(self):
        return {"ego_control": {"throttle": .2, "brake": 0., "steer": 0.}}


class MemoryMessageChannel(MessageChannel):
    """An in-process message transport controlled by the simulated peer."""

    def __init__(self, reports=(), on_finish=None):
        self.controls = []
        self.reports = reports or ({"topics": {}, "tf": {}},)
        self.on_finish = on_finish
        self.handler = None

    @property
    def relay_count(self):
        return len(self.reports)

    def start(self, handler):
        self.handler = handler

    def deliver(self, frame, *, relay_index=0):
        return self.handler({**frame, "relay_index": relay_index, "read_ns": time_ns()})

    def send(self, frame):
        self.controls.append(frame["kind"])
        if frame["kind"] == "finish":
            if self.on_finish is not None:
                self.on_finish(self)
            for index, report in enumerate(self.reports):
                self.deliver({"kind": "statistics", **report}, relay_index=index)

    def check(self):
        pass


class MemoryMessageSimulator(ExternallyControlledMemorySimulator):
    """A closed-loop memory simulator with native message stages."""

    def __init__(self, stages, *, channel=None, tick_callback=None):
        super().__init__()
        self._stages = stages
        self.channel = channel
        self.tick_callback = tick_callback

    @property
    def message_stages(self):
        return self._stages

    @property
    def supported_stages(self):
        return super().supported_stages | self._stages.keys()

    def create_runtime(self, seed=None):
        return MemoryMessageRuntime(self, seed)


class MemoryMessageRuntime(ExternallyControlledMemoryRuntime):
    @property
    def message_channel(self):
        return self.simulator.channel

    @property
    def message_snapshot(self):
        return self.clock.time_us, self.state()

    @property
    def message_sensor_poses(self):
        return {stage.sensor: np.eye(4) for stage in self.message_stages.values() if stage.sensor}

    def _advance_native(self, control):
        if self.simulator.tick_callback is not None:
            self.simulator.tick_callback()
        super()._advance_native(control)


class MemoryDrivingSystem:
    """A hosted driver with simulated readiness and a detached lifecycle report."""

    def __init__(self, ready_at_s=.7, *, startup_error=None, run_error=None):
        self.ready_at_s = ready_at_s
        self.startup_error = startup_error
        self.run_error = run_error
        self.sessions = []

    @property
    def identity(self):
        return {"ready_at_s": self.ready_at_s}

    @property
    def carla_publishes_tf(self):
        return False

    def start(self, context):
        session = MemoryDrivingSession(context, self.ready_at_s, self.startup_error,
                                             self.run_error)
        self.sessions.append(session)
        return session


class MemoryDrivingSession:
    def __init__(self, context, ready_at_s, startup_error, run_error):
        self.context = context
        self.ready_at_s = ready_at_s
        self.startup_error = startup_error
        self.run_error = run_error
        self.ready_s = self.engaged_s = None
        self.sim_time = None
        self.pose = None
        self.advanced = self.stopped = False

    def poll(self, ego_pose, sim_time):
        self.pose = ego_pose.copy()
        self.sim_time = sim_time
        if self.sim_time < self.ready_at_s:
            return False
        if self.startup_error:
            raise TimeoutError(self.startup_error)
        if self.ready_s is None:
            self.ready_s = sim_time
        return True

    def engage(self):
        self.engaged_s = self.sim_time

    def check(self):
        if self.run_error:
            raise RuntimeError(self.run_error)
        self.advanced = True

    def telemetry(self):
        return {"mode": "AUTONOMOUS" if self.advanced else "STOP"}

    def report(self):
        return {"ready_s": self.ready_s, "engaged_s": self.engaged_s, "stopped": self.stopped}

    def stop(self):
        self.stopped = True

class FrameDriver(AVStack):
    """Emit perception only for steps that deliver a front camera frame."""

    command_modes = frozenset({"actuator"})
    supported_stages = frozenset({"perception.post"})

    def reset(self, observation):
        pass

    def __call__(self, observation):
        if "front" in observation.sensor_data:
            self._emit("perception.post", [])
        return Control(throttle=.2)


class ConstantAVStack(AVStack):
    """A deterministic driver exposing one actual component-stage boundary."""

    command_modes = frozenset({"actuator"})
    supported_stages = frozenset({"perception.post"})

    def __init__(self, command=None, perception_output=None,
                 fail_on_position=None, interrupt_on_position=None):
        self.command = Control(throttle=.25) if command is None else command
        self.perception_output = [] if perception_output is None else perception_output
        self.fail_on_position, self.interrupt_on_position = fail_on_position, interrupt_on_position
        self.output = None

    def reset(self, observation):
        self.output = None

    def __call__(self, observation):
        if observation.frame == self.interrupt_on_position:
            raise KeyboardInterrupt("memory driver interrupted")
        if observation.frame == self.fail_on_position:
            raise RuntimeError("memory driver failed")
        self.output = self._emit("perception.post", deepcopy(self.perception_output))
        return replace(self.command)

    def component_log(self):
        return {"perception": self.output}

    def detections(self):
        if isinstance(self.output, (tuple, list)) and all(isinstance(item, Detection) for item in self.output):
            return tuple(self.output)
        return None


class HistoryAVStack(ConstantAVStack):
    """A driver that records its actual warmup inputs in public component telemetry."""

    history = HistoryRequirement(samples=3)

    def reset(self, observation):
        super().reset(observation)
        self.initial_observation = deepcopy(observation)

    def component_log(self):
        return {**super().component_log(), "initial_observation": self.initial_observation}


def decelerating_command(speed, t, acceleration=-2.):
    """A local trajectory with constant acceleration and 0.1 s samples over five seconds."""
    times = np.arange(51) * .1
    positions = float(speed) * times + acceleration * times ** 2 / 2
    return Control(trajectory=[
        ((float(position), 0., 0.), (1., 0., 0., 0.), round((t + time) * 1e6))
        for time, position in zip(times, positions)
    ])


class DeceleratingAVStack(AVStack):
    """Replan a -2 m/s² trajectory from the current observed speed every decision."""

    command_modes = frozenset({"trajectory"})
    history = HistoryRequirement(duration_us=1_500_000)

    def __call__(self, observation):
        return decelerating_command(observation.ego_speed, observation.t)


class StopCommandAVStack(DeceleratingAVStack):
    """Emit a delayed stop command, optionally selecting its delay from captured pixels."""

    history = HistoryRequirement(samples=16)

    def __init__(self, stop_after=3, image_dependent=False, stop_by_start=None,
                 invalid_timestamps=False):
        self.stop_after = stop_after
        self.image_dependent = image_dependent
        self.stop_by_start = dict(stop_by_start or {})
        self.invalid_timestamps = invalid_timestamps
        self.decision = 0
        self.delay = stop_after

    def reset(self, observation):
        self.decision = 0
        self.delay = self.stop_after

    def __call__(self, observation):
        if self.decision == 0:
            self.delay = self.stop_by_start.get(round(observation.vehicle_state.x), self.stop_after)
            if self.image_dependent:
                image = np.asarray(next(iter(observation.sensor_data.values())))
                self.delay = 2 + int(image.astype(np.uint64).sum()) % 4
        tail_speed = .25 if self.delay is not None and self.decision >= self.delay else 10.
        self.decision += 1
        stamp = round(observation.t * 1e6)
        end = stamp + 100000 if self.invalid_timestamps else stamp + 1100000
        return Control(trajectory=[
            ((0., 0., 0.), (1., 0., 0., 0.), stamp),
            ((1., 0., 0.), (1., 0., 0., 0.), stamp + 100000),
            ((1. + tail_speed, 0., 0.), (1., 0., 0., 0.), end),
        ])


class MetadataOnlyAttack(Attack):
    """A user attack that accepts supplied segments and has no scenario selector."""

    def __init__(self, required_key=None):
        self.required_key = required_key

    def validate_segment(self, segment):
        if self.required_key is not None and self.required_key not in segment.user_data:
            raise ValueError(f"Missing required user_data key {self.required_key!r}")


class MinimalAttack(Attack):
    """Select deterministic consecutive windows and record per-run progress."""

    def __init__(self, decision_steps=2, max_segments=2, assets=None, fail_scenarios=(), stop_marker_path=None,
                 variants_per_window=1):
        self.decision_steps, self.max_segments, self.assets = decision_steps, max_segments, assets
        self.fail_scenarios, self.stop_marker_path = tuple(fail_scenarios), stop_marker_path
        self.variants_per_window = variants_per_window

    def build_metadata(self, scenario):
        if scenario.scenario_id in self.fail_scenarios:
            raise RuntimeError("synthetic selection failed")
        from avsectester.attack.segments import split_segments
        windows = split_segments(scenario.frames, lambda frame: True,
                                 decision_steps=self.decision_steps)[:self.max_segments]
        if self.variants_per_window == 1:
            return windows
        return tuple(replace(segment, user_data={"variant": variant})
                     for segment in windows for variant in range(self.variants_per_window))

    def run_offline(self, context):
        return self.assets

    def create_runtime(self, segment, artifacts):
        return MinimalAttackRuntime(segment, self.stop_marker_path)


class MinimalAttackRuntime(AttackRuntime):
    """Record output through the real recorder, with optional durable cleanup evidence."""

    def __init__(self, segment, stop_marker_path=None):
        self.segment, self.stop_marker_path = segment, stop_marker_path
        self.decision = 0

    def step(self, context):
        context.record_output("progress", {"decision": self.decision, "frame": context.frame})
        self.decision += 1
        return ()

    def stop(self, context):
        if self.stop_marker_path is not None:
            path = Path(self.stop_marker_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps({"scenario": self.segment.scenario_id,
                                         "start": self.segment.start.position}) + "\n")


# Native dependency boundaries used by the real simulator implementations.


def proto_world_pose(pose):
    """Read the external RPC pose fields for independent boundary expectations."""
    quaternion = np.array([pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z])
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ]
    result[:3, 3] = [pose.vec.x, pose.vec.y, pose.vec.z]
    return result


class NuRecRpcError(Exception):
    """A gRPC status error raised by the NuRec service boundary."""

    def __init__(self, code, details):
        super().__init__(details)
        self._code, self._details = code, details

    def code(self):
        return getattr(sys.modules["grpc"].StatusCode, self._code, self._code)

    def details(self):
        return self._details


def container_running(state_path):
    """Whether a Docker fake container state is absent (no container check) or running."""
    if state_path is None:
        return True
    try:
        state = json.loads(state_path.read_text())
    except FileNotFoundError:
        return False
    return state.get("running", False) if isinstance(state, dict) else state is True


def decode_gaussian_ply(data):
    """Read the binary float rows of a NuRec Gaussian PLY."""
    _, body = data.split(b"end_header\n", 1)
    return np.frombuffer(body, "<f4").reshape(-1, 14)


TRACK_FROM_PLY = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


class NuRecRpcBoundary:
    """A SensorsimService boundary with a real NuRecRenderer.

    Inserted Gaussian assets are drawn at their listed poses, mapped into track frames as the
    measured NRE convention does, except at pixels in ``hidden``, where the scene covers them.
    ``asset_directories`` maps server asset paths to local ones. A render naming a scene or
    coverage track that was not inserted fails as NRE does. With ``scene_tracks`` set, the same
    applies to any other track outside it.
    ``returned_resolution_hw`` is None for the native resolution, an (H, W) tuple, or a
    callable mapping the integer frame-start timestamp in microseconds to an (H, W) tuple. With
    ``service_container`` set, scene queries answer UNAVAILABLE unless that Docker fake state
    reports a running container. LiDAR returns ``lidar_response`` and retains scan requests
    in ``lidar_requests``. An empty dynamic-object list renders every inserted track at its
    stored trajectory. Nonempty lists render only their named tracks. Repeated track IDs
    use the first listed pose. Repeated insertions append copies and increment
    ``insertion_counts``. The miniature rasterizer widens the fringe of duplicate copies
    so duplicate insertion changes the image.
    """

    TRACK_PADDING_M = np.array([1., 1., .25])

    def __init__(self, returned_resolution_hw=None, *, service_container=None):
        self.returned_resolution_hw = returned_resolution_hw
        self.service_container = service_container
        self.last_request = None
        self.requests = []
        self.lidar_requests = []
        self.lidar_response = SimpleNamespace(
            point_xyzs_buffer=np.array([[4., 1., -1.95], [8., -2., -1.]], np.float32).tobytes(),
            point_intensities_buffer=b"", point_xyzs=[], point_intensities=[], num_points=2)
        self.available_scenes = ("scene",)
        self.scene_query_error = None
        self.rpc_error = NuRecRpcError
        self.scene_query_timeouts = []
        self.tracks = {}
        self.insertion_counts = {}
        self.inserted_tracks = {}
        self.stored_trajectories = {}
        self.scene_tracks = None
        self.hidden = None
        self.asset_directories = {}
        self.mount = SimpleNamespace(vec=SimpleNamespace(x=0., y=0., z=1.5),
            quat=SimpleNamespace(w=.5, x=-.5, y=.5, z=-.5))
        self.spec = SimpleNamespace(resolution_h=12, resolution_w=16, ftheta_param=SimpleNamespace(
            principal_point_x=8., principal_point_y=6., angle_to_pixeldist_poly=[0., 8.],
            pixeldist_to_angle_poly=[0., .125], max_angle=1.5,
            linear_cde=SimpleNamespace(linear_c=1., linear_d=0., linear_e=0.)),
            WhichOneof=lambda group: None)

    def get_available_scenes(self, request, timeout=None):
        self.requests.append(request)
        self.scene_query_timeouts.append(timeout)
        if not container_running(self.service_container):
            raise self.rpc_error("UNAVAILABLE", "renderer is not running")
        if self.scene_query_error is not None:
            raise self.rpc_error(*self.scene_query_error)
        return SimpleNamespace(scene_ids=list(self.available_scenes))

    def get_available_cameras(self, request):
        self.requests.append(request)
        return SimpleNamespace(available_cameras=[SimpleNamespace(logical_id="front",
            intrinsics=self.spec, rig_to_camera=self.mount)])

    def get_available_trajectories(self, request):
        self.requests.append(request)
        pose = SimpleNamespace(vec=SimpleNamespace(x=0., y=0., z=.3),
                               quat=SimpleNamespace(w=1., x=0., y=0., z=0.))
        return SimpleNamespace(available_trajectories=[SimpleNamespace(trajectory=SimpleNamespace(
            poses=[SimpleNamespace(timestamp_us=0, pose=pose)]))])

    def _listed_tracks(self, request):
        listed = list(request.dynamic_objects)
        first = {}
        for item in listed:
            first.setdefault(item.track_id, item)
        listed = list(first.values())
        if not listed:
            listed = [SimpleNamespace(track_id=key, pose_pair=SimpleNamespace(
                start_pose=track.trajectory.poses[0].pose,
                end_pose=track.trajectory.poses[-1].pose))
                for key, track in self.inserted_tracks.items()]
        unknown = [item.track_id for item in listed if item.track_id not in self.tracks
                   and (item.track_id.startswith(("avsectester_scene_", "avsectester_coverage_"))
                        or self.scene_tracks is not None and item.track_id not in self.scene_tracks)]
        if unknown:
            raise self.rpc_error("INVALID_ARGUMENT", f"{unknown[0]!r} is not in list")
        return listed

    def render_rgb(self, request):
        from io import BytesIO
        from PIL import Image
        from avsectester.geometry.cameras import FThetaCamera
        self.requests.append(request)
        self.last_request = request
        returned = self.returned_resolution_hw
        if callable(returned):
            returned = returned(request.frame_start_us)
        height, width = returned or (
            request.resolution_h,
            round(request.resolution_h * self.spec.resolution_w / self.spec.resolution_h),
        )
        listed = self._listed_tracks(request)
        image = np.full((height, width, 3), 30, np.uint8)
        camera = FThetaCamera(8.5, 6.5, (0., 8.), 16, 12, (0., .125), 1.5)
        cam_from_world = np.linalg.inv(proto_world_pose(request.sensor_pose.start_pose))
        for item in listed:
            if item.track_id not in self.tracks:
                continue
            cam_from_track = cam_from_world @ proto_world_pose(item.pose_pair.start_pose)
            for copy_index, (points, colours) in enumerate(self.tracks[item.track_id]):
                optical = points @ cam_from_track[:3, :3].T + cam_from_track[:3, 3]
                front = optical[:, 2] > 0
                pixels = np.floor(camera.project(optical[front])).astype(int)
                offsets = ((0, 0),) if copy_index == 0 else (
                    (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
                for offset in offsets:
                    shifted = pixels + offset
                    inside = (shifted >= 0).all(axis=1) & (shifted < [width, height]).all(axis=1)
                    for (x, y), colour in zip(shifted[inside], colours[front][inside]):
                        if self.hidden is None or not self.hidden[y, x]:
                            image[y, x] = colour
        stream = BytesIO()
        Image.fromarray(image).save(stream, format="PNG")
        return SimpleNamespace(image_bytes=stream.getvalue())

    def render_lidar(self, request):
        """Return native scan data plus listed scene assets in the ending sensor frame."""
        self.requests.append(request)
        self.lidar_requests.append(request)
        listed = self._listed_tracks(request)
        clouds = []
        sensor_from_world = np.linalg.inv(proto_world_pose(request.sensor_pose.end_pose))
        for item in listed:
            if item.track_id not in self.tracks:
                continue
            sensor_from_track = sensor_from_world @ proto_world_pose(item.pose_pair.end_pose)
            for points, _ in self.tracks[item.track_id]:
                points = points @ sensor_from_track[:3, :3].T + sensor_from_track[:3, 3]
                clouds.append(points[np.linalg.norm(points, axis=1) < 200.])
        if not any(len(points) for points in clouds):
            return self.lidar_response
        native = self.lidar_response
        base = (np.frombuffer(native.point_xyzs_buffer, np.float32).reshape(-1, 3)
                if native.point_xyzs_buffer else np.asarray(native.point_xyzs).reshape(-1, 3))
        points = np.concatenate([base, *clouds]).astype(np.float32)
        return SimpleNamespace(point_xyzs_buffer=points.tobytes(), point_xyzs=[],
                               point_intensities_buffer=b"", point_intensities=[],
                               num_points=len(points))

    def edit_assets(self, request):
        self.requests.append(request)
        for track in request.insert:
            self.inserted_tracks[track.id] = track
            self.stored_trajectories[track.id] = tuple(
                (pose.timestamp_us, proto_world_pose(pose.pose)) for pose in track.trajectory.poses)
            directory, name = track.asset_id.rsplit("/", 1)
            data = (Path(self.asset_directories.get(directory, directory)) / name).read_bytes()
            rows = decode_gaussian_ply(data)
            size = np.array([track.object_size.size_x, track.object_size.size_y, track.object_size.size_z])
            scale = (size - self.TRACK_PADDING_M).max()
            colours = np.clip((rows[:, 3:6] * .28209479177387814 + .5) * 255, 0, 255).round().astype(np.uint8)
            self.tracks.setdefault(track.id, []).append(
                (rows[:, :3] @ TRACK_FROM_PLY.T * scale, colours))
            self.insertion_counts[track.id] = self.insertion_counts.get(track.id, 0) + 1
        return SimpleNamespace(success=True, message="Success")

    def _unary_unary(self, method, request_serializer, response_deserializer):
        if not method.endswith("/edit_assets"):
            raise NotImplementedError(method)
        return lambda request, timeout=None: self.edit_assets(request)

    def install(self, monkeypatch, *, passthrough_endpoints=()):
        boundary = self
        native_grpc = None
        native_common = None
        if passthrough_endpoints:
            import grpc as native_grpc
            from alpasim_grpc.v0 import common_pb2 as native_common
            native_channel = native_grpc.insecure_channel
            self.rpc_error = type("SensorsimRpcError", (NuRecRpcError, native_grpc.RpcError), {})

        class Channel:
            def close(self):
                pass

            def unary_unary(self, *args, **kwargs):
                return boundary._unary_unary(*args, **kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        common = native_common or SimpleNamespace(Pose=SimpleNamespace, Quat=SimpleNamespace,
                                                  Vec3=SimpleNamespace, Empty=SimpleNamespace)
        sensor = SimpleNamespace(LidarRenderRequest=SimpleNamespace,
            LidarSpec=SimpleNamespace, PANDAR128=0,
            AvailableCamerasRequest=SimpleNamespace,
            AvailableTrajectoriesRequest=SimpleNamespace, RGBRenderRequest=SimpleNamespace,
            PosePair=SimpleNamespace, DynamicObject=SimpleNamespace,
            ImageFormat=SimpleNamespace(JPEG=1))
        modules = {
            "grpc": {"insecure_channel": lambda endpoint, **kwargs: Channel(),
                     "RpcError": NuRecRpcError,
                     "StatusCode": SimpleNamespace(INVALID_ARGUMENT="INVALID_ARGUMENT",
                         UNAVAILABLE="UNAVAILABLE", DEADLINE_EXCEEDED="DEADLINE_EXCEEDED",
                         PERMISSION_DENIED="PERMISSION_DENIED")},
            "alpasim_grpc": {}, "alpasim_grpc.v0": {"common_pb2": common,
                "sensorsim_pb2": sensor,
                "sensorsim_pb2_grpc": SimpleNamespace(SensorsimServiceStub=lambda channel: self)},
        }
        if native_grpc is not None:
            grpc_values = dict(vars(native_grpc))
            grpc_values.update(
                insecure_channel=lambda endpoint, **kwargs: (
                    native_channel(endpoint, **kwargs) if endpoint in passthrough_endpoints else Channel()),
            )
            modules = {"grpc": grpc_values}
            import alpasim_grpc.v0 as protocol
            monkeypatch.setattr(protocol, "sensorsim_pb2", sensor, raising=False)
            monkeypatch.setattr(protocol, "sensorsim_pb2_grpc",
                                SimpleNamespace(SensorsimServiceStub=lambda channel: self), raising=False)
        for name, values in modules.items():
            module = ModuleType(name)
            vars(module).update(values)
            monkeypatch.setitem(sys.modules, name, module)
        return self


class CarlaVector:
    def __init__(self, x=0., y=0., z=0.):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def norm(self):
        return float(np.linalg.norm((self.x, self.y, self.z)))


class CarlaTransform:
    def __init__(self, location=None, rotation=None):
        self.location = location or CarlaVector()
        self.rotation = rotation or SimpleNamespace(pitch=0., yaw=0., roll=0.)

    def get_matrix(self):
        pitch, yaw, roll = np.deg2rad([self.rotation.pitch, self.rotation.yaw, self.rotation.roll])
        cosine_pitch, cosine_yaw, cosine_roll = np.cos([pitch, yaw, roll])
        sine_pitch, sine_yaw, sine_roll = np.sin([pitch, yaw, roll])
        pose = np.eye(4)
        pose[:3, :3] = [
            [cosine_pitch * cosine_yaw,
             cosine_yaw * sine_pitch * sine_roll - sine_yaw * cosine_roll,
             -cosine_yaw * sine_pitch * cosine_roll - sine_yaw * sine_roll],
            [cosine_pitch * sine_yaw,
             sine_yaw * sine_pitch * sine_roll + cosine_yaw * cosine_roll,
             -sine_yaw * sine_pitch * cosine_roll + cosine_yaw * sine_roll],
            [sine_pitch, -cosine_pitch * sine_roll, cosine_pitch * cosine_roll],
        ]
        pose[:3, 3] = (self.location.x, self.location.y, self.location.z)
        return pose

    def get_inverse_matrix(self):
        return np.linalg.inv(self.get_matrix())

    def get_forward_vector(self):
        return CarlaVector(1., 0., 0.)

    def get_right_vector(self):
        return CarlaVector(0., 1., 0.)


class CarlaActor:
    def __init__(self, world, actor_id, vehicle="vehicle.test", transform=None):
        self.world, self.id, self.type_id = world, actor_id, vehicle
        self.transform = transform or CarlaTransform()
        self.velocity = CarlaVector()
        self.control = SimpleNamespace(throttle=0., brake=0., steer=0.)
        self.bounding_box = SimpleNamespace(extent=CarlaVector(2., 1., .7),
            location=CarlaVector(.2, 0., .7), rotation=SimpleNamespace(roll=0., pitch=0., yaw=0.))
        self.physics = SimpleNamespace(wheels=[SimpleNamespace(friction_force_multiplier=2.)])
        self.simulate_physics = True
        self.enable_gravity = True
        self.ros_enabled = False
        self.destroyed = False
        self.light_state = None
        self.attached_to = None
        self.textures = {}
        self.attributes = {}
        self.capture_callback = None

    def listen(self, callback):
        self.capture_callback = callback

    def stop(self):
        self.capture_callback = None

    @property
    def is_alive(self):
        return not self.destroyed

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        return self.velocity

    def set_transform(self, transform):
        self.world.actor_update(self, "set_transform")
        self.transform = transform

    def set_target_velocity(self, velocity):
        self.world.actor_update(self, "set_target_velocity")
        self.velocity = velocity

    def set_target_angular_velocity(self, velocity):
        self.world.actor_update(self, "set_target_angular_velocity")

    def set_simulate_physics(self, enabled):
        self.world.actor_update(self, "set_simulate_physics")
        self.simulate_physics = enabled

    def set_enable_gravity(self, enabled):
        self.world.actor_update(self, "set_enable_gravity")
        self.enable_gravity = enabled

    def set_autopilot(self, enabled, port):
        self.world.actor_update(self, "set_autopilot")

    def apply_control(self, control):
        self.world.actor_update(self, "apply_control")
        self.control = control

    def get_control(self):
        return self.control

    def enable_for_ros(self):
        self.ros_enabled = True

    def get_physics_control(self):
        return self.physics

    def apply_physics_control(self, physics):
        self.world.actor_update(self, "apply_physics_control")
        self.physics = physics

    def set_light_state(self, state):
        self.world.actor_update(self, "set_light_state")
        self.light_state = state

    def destroy(self):
        self.destroyed = True
        self.world.destroyed_ids.append(self.id)


class CarlaTrafficLight(CarlaActor):
    def __init__(self, world, actor_id, signal_id, state="Red"):
        super().__init__(world, actor_id, "traffic.traffic_light")
        self.signal_id = signal_id
        self.state = state
        self.elapsed_s = 0.

    def get_opendrive_id(self):
        return self.signal_id

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def get_elapsed_time(self):
        return self.elapsed_s


class CarlaSensorManager:
    def __init__(self):
        self.data = {"front": deque()}

    def empty(self):
        return not any(self.data.values())

    def has_data(self, name):
        return bool(self.data[name])

    def pop(self, name):
        return self.data[name].popleft()


class CarlaLidarCapture:
    """The native SDK buffer can be read as bytes but cannot be deep-copied."""

    __module__ = "carla.libcarla"

    def __init__(self, frame, timestamp):
        self.frame, self.timestamp = frame, timestamp
        self.raw_data = memoryview(np.array([[5., -2., .5, .8]], np.float32).tobytes())

    def __deepcopy__(self, memo):
        raise TypeError("Native CARLA capture cannot be deep-copied")


class CarlaImageData(SimpleNamespace):
    """ImageData's RGB accessor follows its replaceable pixel buffer."""

    @property
    def rgb_image(self):
        return self.data


CARLA_CAMERA_WIDTH, CARLA_CAMERA_HEIGHT = 16, 12
CARLA_CAMERA_ATTRIBUTES = {
    "image_size_x": str(CARLA_CAMERA_WIDTH), "image_size_y": str(CARLA_CAMERA_HEIGHT),
    "fov": "90", "lens_k": "0", "lens_kcube": "0", "lens_circle_multiplier": "0",
}
CARLA_CAMERA_PROJECTION = np.array([
    [CARLA_CAMERA_WIDTH / 2, 0., CARLA_CAMERA_WIDTH / 2, 0.],
    [0., CARLA_CAMERA_WIDTH / 2, CARLA_CAMERA_HEIGHT / 2, 0.],
    [0., 0., 1., 0.],
])


class CarlaMobileActor:
    def __init__(self, client, specification=None, include_rgb=True, **options):
        specification = options if specification is None else specification
        self.client = client
        self.specification = deepcopy(specification)
        spawn = specification.get("spawn", 0)
        if "spawn_transform" in specification:
            spec = specification["spawn_transform"]
            transform = CarlaTransform(CarlaVector(**spec["location"]), SimpleNamespace(**spec["rotation"]))
        else:
            transform = CarlaTransform(CarlaVector(float(spawn) * 3, 0., 0.))
        blueprint = client.world.get_blueprint_library().find(
            specification.get("vehicle", specification.get("npc_type", "vehicle.test")))
        self.actor = client.world.try_spawn_actor(blueprint, transform)
        self.spawn_transform = transform
        self.reference = SimpleNamespace(x=np.zeros(3), q=np.array([1., 0., 0., 0.]))
        self.timestamp = 0.
        self.sensor_data_manager = CarlaSensorManager()
        native_camera = SimpleNamespace(type_id="sensor.camera.rgb",
            get_transform=self.actor.get_transform,
            attributes=dict(CARLA_CAMERA_ATTRIBUTES))
        self.sensors = {"front_camera": SimpleNamespace(source_identifier="front", frame0=0, t0=0.,
            object=native_camera, reference=SimpleNamespace(as_carla_transform=lambda local=True: CarlaTransform()),
            imsize=(CARLA_CAMERA_HEIGHT, CARLA_CAMERA_WIDTH), P=CARLA_CAMERA_PROJECTION.copy())}
        if not include_rgb:
            self.sensors.clear()
            self.sensor_data_manager.data.clear()
        for sensor_config in specification.get("sensors", ()):
            if sensor_config["type"].startswith("Configured"):
                from avcarla.config import CARLA

                sensor = CARLA.build(sensor_config, default_args={"client": client, "parent": self})
                key = "top_lidar" if "Lidar" in sensor_config["type"] else "front_camera"
                self.sensors[key] = sensor
                self.sensor_data_manager.data.setdefault(sensor.source_identifier, deque())
            elif sensor_config["type"] == "CarlaLidar":
                def lidar_transform():
                    location = self.actor.get_transform().location
                    return CarlaTransform(CarlaVector(location.x + 1., location.y + 2., 1.5),
                                          SimpleNamespace(pitch=0., yaw=90., roll=0.))

                self.sensors["top_lidar"] = SimpleNamespace(source_identifier="lidar", frame0=0, t0=0.,
                    object=SimpleNamespace(type_id="sensor.lidar.ray_cast", get_transform=lidar_transform))
                self.sensor_data_manager.data["lidar"] = deque()
        client.world.vehicles.append(self)

    def initialize(self, timestamp, frame):
        for sensor in self.sensors.values():
            sensor.frame0, sensor.t0 = frame, timestamp

    def apply_control(self, control):
        self.actor.apply_control(control)

    def get_pose(self):
        vector = self.actor.transform.location
        return SimpleNamespace(position=SimpleNamespace(x=np.array([vector.x, -vector.y, vector.z])),
                               attitude=SimpleNamespace(q=np.array([1., 0., 0., 0.])))

    def get_object_state(self):
        pose = self.get_pose()
        return SimpleNamespace(position=pose.position, attitude=pose.attitude, velocity=self.actor.velocity)

    def capture(self, frame, timestamp, frame_offset=0):
        # Identical physical scenes have identical pixels even when reset setup ticks
        # give their captures different native clock/frame metadata.
        color = int(round(self.actor.transform.location.x * 1000)) % 255
        if "front_camera" in self.sensors:
            sensor = self.sensors["front_camera"]
            payload = CarlaImageData(frame=frame - sensor.frame0 + frame_offset,
                timestamp=timestamp - sensor.t0,
                data=np.full((CARLA_CAMERA_HEIGHT, CARLA_CAMERA_WIDTH, 3), color, np.uint8),
                calibration=SimpleNamespace(reference=SimpleNamespace(reference=self.reference)))
            self.sensor_data_manager.data["front"].append(payload)
        if "top_lidar" in self.sensors:
            lidar = self.sensors["top_lidar"]
            payload = SimpleNamespace(frame=frame - lidar.frame0 + frame_offset,
                timestamp=timestamp - lidar.t0,
                data=CarlaLidarCapture(frame, timestamp), flipy=True, n_features=4)
            self.sensor_data_manager.data["lidar"].append(payload)

    def destroy(self):
        self.actor.destroy()


class CarlaNativeSensor:
    def __init__(self, client, parent, *, do_spawn=True, name="front", **options):
        self.client, self.parent = client, parent
        self.source_identifier = "lidar" if "lidar" in name.lower() else "front"
        kind = "sensor.lidar.ray_cast" if self.source_identifier == "lidar" else "sensor.camera.rgb"
        self.bp = client.world.get_blueprint_library().find(kind)
        for key, value in CARLA_CAMERA_ATTRIBUTES.items():
            self.bp.set_attribute(key, value)
        self.reference = SimpleNamespace(as_carla_transform=lambda local=True: CarlaTransform())
        self.frame0, self.t0 = 0, 0.
        self.imsize = (CARLA_CAMERA_HEIGHT, CARLA_CAMERA_WIDTH)
        self.P = CARLA_CAMERA_PROJECTION.copy()
        if do_spawn:
            self.spawn()

    def spawn(self):
        self.object = self.client.world.spawn_actor(
            self.bp, CarlaTransform(), attach_to=self.parent.actor)


class CarlaBlueprint:
    """CARLA actor blueprint fields consumed by sensor construction."""

    def __init__(self, identity):
        self.id = identity
        self.attributes = {}

    def set_attribute(self, name, value):
        self.attributes[name] = str(value)

    def has_attribute(self, name):
        if name.startswith("noise_"):
            return self.id in {"sensor.lidar.ray_cast", "sensor.other.imu", "sensor.other.gnss",
                               "sensor.other.autoware_gnss"}
        return name in {"role_name", "ros_name", "ros_topic_name", "ros2_ackermann_control",
                        "ros2_autoware_control", "post_process_profile", "image_size_x", "image_size_y",
                        "fov", "sensor_tick", "mass", "mesh_path"}


class CarlaWorld:
    def __init__(self, config, frame_offset=0, ground_height=None, ground_offset=(0., 0., 0.),
                 external_ego_throttle=None, ego_attribute_overrides=None, tick_callback=None,
                 publish_tf=True, world_settings_overrides=None):
        self.frame, self.time = 0, 0.
        self.dt = 1 / config.get("rate", 10.)
        self.actors, self.vehicles = [], []
        self.actors.append(CarlaTrafficLight(self, 10000, "signal-1"))
        self.config, self.frame_offset = config, frame_offset
        self.weather = None
        self.layers = []
        self.ground_height = ground_height
        self.ground_offset = ground_offset
        self.external_ego_throttle = external_ego_throttle
        self.ego_attribute_overrides = ego_attribute_overrides or {}
        self.tick_callback = tick_callback
        self.publish_tf = publish_tf
        self.world_settings_overrides = world_settings_overrides or {}
        self.command_events = []
        self.direct_actor_updates = []
        self.destroyed_ids = []
        self.traffic_reset_pending = False
        self.applying_batch = False

    def actor_update(self, actor, operation):
        if not self.applying_batch:
            self.direct_actor_updates.append((actor.id, operation))

    def get_snapshot(self):
        return SimpleNamespace(frame=self.frame,
            timestamp=SimpleNamespace(elapsed_seconds=self.time),
            find=lambda identity: next((actor for actor in self.actors if actor.id == identity), None))

    def get_settings(self):
        settings = {"synchronous_mode": self.config.get("synchronous", True),
                    "fixed_delta_seconds": self.dt,
                    "substepping": self.config.get("substepping", True),
                    "max_substep_delta_time": self.config.get("max_substep_delta_time", .01),
                    "max_substeps": self.config.get("max_substeps", 10)}
        return SimpleNamespace(**{
            name: self.world_settings_overrides.get(name, value) for name, value in settings.items()
        })

    def set_publish_tf(self, enabled):
        self.publish_tf = enabled

    def get_publish_tf(self):
        return self.world_settings_overrides.get("publish_tf", self.publish_tf)

    def tick(self):
        self.command_events.append(("tick", self.frame + 1))
        self.frame += 1
        self.time += self.dt
        for actor in self.actors:
            if isinstance(actor, CarlaTrafficLight):
                if self.traffic_reset_pending:
                    actor.set_state("Red")
                    actor.elapsed_s = 0.
                actor.elapsed_s += self.dt
            if (self.ground_height is not None and actor.type_id.startswith("vehicle.")
                    and actor.simulate_physics and actor.enable_gravity):
                location, rotation = actor.transform.location, actor.transform.rotation
                if location.z > self.ground_height:
                    location.z = max(self.ground_height, location.z - .1)
                    if location.z == self.ground_height:
                        dx, dy, dyaw = self.ground_offset
                        location.x += dx
                        location.y += dy
                        rotation.yaw += dyaw
                        rotation.pitch = 2.
                        rotation.roll = -1.
            if (self.external_ego_throttle is not None
                    and actor.attributes.get("role_name") in {"hero", "ego"}
                    and actor.attributes.get("ros2_ackermann_control") == "true"):
                # Native ROS control arrives after the framework batch and before movement.
                actor.control = SimpleNamespace(
                    throttle=self.external_ego_throttle, brake=0., steer=0., hand_brake=False,
                    reverse=False, manual_gear_shift=False, gear=1)
            speed = max(0., actor.velocity.x + (actor.control.throttle - actor.control.brake) * self.dt)
            actor.velocity = CarlaVector(speed, actor.velocity.y, actor.velocity.z)
            actor.transform.location.x += speed * self.dt
        self.traffic_reset_pending = False
        for actor in self.vehicles:
            actor.capture(self.frame, self.time, self.frame_offset)
        for actor in self.actors:
            if actor.type_id == "sensor.camera.depth" and actor.capture_callback is not None:
                width = int(actor.attributes.get("image_size_x", CARLA_CAMERA_WIDTH))
                height = int(actor.attributes.get("image_size_y", CARLA_CAMERA_HEIGHT))
                encoded = round(10. / 1000. * (256 ** 3 - 1))
                bgra = np.empty((height, width, 4), np.uint8)
                bgra[:] = [(encoded >> 16) & 255, (encoded >> 8) & 255, encoded & 255, 255]
                actor.capture_callback(SimpleNamespace(
                    frame=self.frame, timestamp=self.time, width=width, height=height,
                    raw_data=bgra.tobytes()))
        if self.tick_callback is not None:
            self.tick_callback()
        return self.frame

    def reset_all_traffic_lights(self):
        self.traffic_reset_pending = True

    def get_actors(self):
        return list(self.actors)

    def get_blueprint_library(self):
        vehicles = [SimpleNamespace(id="vehicle.a"), SimpleNamespace(id="vehicle.b")]
        return SimpleNamespace(filter=lambda pattern: (
            [CarlaBlueprint(pattern)] if pattern.startswith("static.prop") else vehicles),
            find=CarlaBlueprint)

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        actor = CarlaActor(self, len(self.actors) + 1, blueprint.id, transform)
        actor.attributes = dict(getattr(blueprint, "attributes", {}))
        actor.attached_to = attach_to
        self.actors.append(actor)
        return actor

    def try_spawn_actor(self, blueprint, transform):
        actor = self.spawn_actor(blueprint, transform)
        if actor.type_id.startswith("vehicle."):
            actor.attributes.update(self.ego_attribute_overrides)
        return actor

    def get_names_of_all_objects(self):
        return [f"object-{actor.id}" for actor in self.actors
                if actor.type_id.startswith("static.prop") and actor.is_alive]

    def apply_color_texture_to_object(self, name, parameter, texture):
        actor = next(actor for actor in self.actors if name == f"object-{actor.id}")
        actor.textures[parameter] = texture.pixels.copy()

    def apply_float_color_texture_to_object(self, name, parameter, texture):
        self.apply_color_texture_to_object(name, parameter, texture)

    def set_weather(self, weather):
        self.weather = weather

    def get_map(self):
        return SimpleNamespace(name=self.config.get("map_name", "Town03_Opt"))

    def unload_map_layer(self, layer):
        self.layers.append(("unload", layer))

    def load_map_layer(self, layer):
        self.layers.append(("load", layer))


class CarlaNativeClient:
    def __init__(self, config, *, batch_callback=None, **world_options):
        map_name = config.get("map_name", "")
        if isinstance(map_name, str) and "/" in map_name:
            raise RuntimeError(f"Map not found: {map_name}")
        self.config = config
        self.world = CarlaWorld(config, **world_options)
        self.client = self
        self.command_error = None
        self.batch_callback = batch_callback
        self.spawn_points = [CarlaTransform(CarlaVector(i * 3, 0, 0)) for i in range(6)]
        self.traffic_manager_port = config.get("traffic_manager_port", 8000)

    def scene_settings(self):
        name = self.config.get("map_name")
        return {"map_name": f"Carla/Maps/{name.rsplit('/', 1)[-1]}"} if name else {}

    def tick(self):
        return self.world.tick()

    def apply_batch_sync(self, commands, do_tick):
        self.world.command_events.append(("batch", tuple(commands), do_tick))
        results = []
        self.world.applying_batch = True
        try:
            for command in commands:
                actor = next(actor for actor in self.world.actors if actor.id == command.actor_id)
                error = self.command_error
                if error is None:
                    getattr(actor, command.method)(*command.arguments)
                results.append(SimpleNamespace(error=error, has_error=lambda error=error: error is not None))
        finally:
            self.world.applying_batch = False
        if self.batch_callback is not None:
            self.batch_callback()
        return results

    def close(self):
        pass


class CarlaTexture:
    def __init__(self, width, height):
        self.pixels = np.zeros((height, width, 4), np.float32)

    def set(self, x, y, color):
        self.pixels[y, x] = color


def wait_until(predicate, *, timeout=3.):
    """Poll an external boundary with a finite deadline."""
    deadline = monotonic() + timeout
    while not predicate():
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("The test boundary did not reach the expected state")
        sleep(min(.01, remaining))


class ControllableClock:
    """A wall clock whose sleeping and work both advance simulated wall time."""

    def __init__(self):
        self.now = 0.
        self.waits = []

    def install(self, monkeypatch, component):
        monkeypatch.setattr(sys.modules[component.__module__], "time", self)

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.waits.append(duration)
        self.advance(duration)

    def advance(self, duration):
        self.now += duration


class RelayMemoryDrivingSystem(MemoryDrivingSystem):
    """A memory driving system publishing through a real relay connection."""

    def __init__(self, socket_path, interception):
        super().__init__()
        self.socket_path = socket_path
        self.interception = interception

    def start(self, context):
        from avsectester.closed_loop.ros_interception.channel import RelayChannel, RelayConnection

        session = super().start(context)
        channel = RelayChannel(self.socket_path)
        peer = RelayConnection(channel.path)
        session.interception_channel = channel
        self.peer = peer
        self.replies = []
        peer.send({"kind": "ready"})
        peer.flush(2.)
        wait_until(lambda: channel.ready)
        original_stop = session.stop

        def respond():
            while peer.wait(3.):
                for frame in peer.drain():
                    if frame["kind"] == "reply":
                        self.replies.append(frame)
                        peer.send({"kind": "published", "id": frame["id"], "sent_ns": time_ns()})
                    elif frame["kind"] == "finish":
                        peer.send({"kind": "statistics", "topics": {}, "tf": {}})
                        peer.flush(2.)
                        return

        worker = Thread(target=respond, daemon=True)
        worker.start()

        def stop():
            worker.join(3.)
            peer.close()
            channel.close()
            original_stop()

        session.stop = stop
        return session


class CarlaNativeBoundary:
    """Registry/client boundary used by the real CARLA simulator, without a server."""

    def __init__(self, frame_offset=0, include_rgb=True, ground_height=None, *, ground_offset=(0., 0., 0.),
                 external_ego_throttle=None, ego_attribute_overrides=None, tick_callback=None,
                 batch_callback=None, publish_tf=True, world_settings_overrides=None):
        self.include_rgb = include_rgb
        self.batch_callback = batch_callback
        self.world_options = dict(frame_offset=frame_offset, ground_height=ground_height,
                                  ground_offset=ground_offset, external_ego_throttle=external_ego_throttle,
                                  ego_attribute_overrides=ego_attribute_overrides,
                                  tick_callback=tick_callback, publish_tf=publish_tf,
                                  world_settings_overrides=world_settings_overrides)
        self.module_dict = {"CarlaMobileActor": CarlaMobileActor, "CarlaRgbCamera": CarlaNativeSensor,
                            "CarlaLidar": CarlaNativeSensor}

    def get(self, name):
        return self.module_dict.get(name)

    def register_module(self, name=None, force=False):
        def register(component):
            key = name or component.__name__
            if key in self.module_dict and not force:
                raise KeyError(key)
            self.module_dict[key] = component
            return component
        return register

    def build(self, config, default_args=None):
        if default_args is None:
            return CarlaNativeClient(
                config, batch_callback=self.batch_callback, **self.world_options)
        if config["type"] in self.module_dict:
            options = {key: value for key, value in config.items() if key != "type"}
            if "parent" not in default_args:
                options["include_rgb"] = self.include_rgb
            return self.module_dict[config["type"]](**default_args, **options)
        return CarlaMobileActor(default_args["client"], config, self.include_rgb)

    def install(self, monkeypatch, rgb_camera=None):
        if rgb_camera is not None:
            self.module_dict["CarlaRgbCamera"] = rgb_camera
        carla = ModuleType("carla")
        carla.Vector3D = carla.Location = CarlaVector
        carla.LidarMeasurement = CarlaLidarCapture
        carla.Transform = CarlaTransform
        carla.Rotation = lambda **values: SimpleNamespace(**values)
        carla.TextureColor = carla.TextureFloatColor = CarlaTexture
        carla.Color = carla.FloatColor = lambda r, g, b, a: (r, g, b, a)
        carla.MaterialParameter = SimpleNamespace(Diffuse="Diffuse", Emissive="Emissive",
            AO_Roughness_Metallic_Emissive="material")
        carla.AttachmentType = SimpleNamespace(Rigid="Rigid")
        from enum import IntFlag
        carla.VehicleLightState = IntFlag("VehicleLightState", {
            "NONE": 0, "Position": 1, "LowBeam": 2, "HighBeam": 4, "Brake": 8})
        carla.VehicleControl = lambda throttle=0., brake=0., steer=0., hand_brake=False, reverse=False, manual_gear_shift=False, gear=0: SimpleNamespace(
            throttle=throttle, brake=brake, steer=steer, hand_brake=hand_brake, reverse=reverse,
            manual_gear_shift=manual_gear_shift, gear=gear)
        methods = {"ApplyTransform": "set_transform", "ApplyTargetVelocity": "set_target_velocity",
            "ApplyTargetAngularVelocity": "set_target_angular_velocity",
            "SetSimulatePhysics": "set_simulate_physics", "SetEnableGravity": "set_enable_gravity",
            "ApplyVehicleControl": "apply_control", "SetAutopilot": "set_autopilot",
            "ApplyVehiclePhysicsControl": "apply_physics_control", "SetVehicleLightState": "set_light_state"}
        carla.command = SimpleNamespace(**{name: (lambda actor_id, *arguments, method=method:
            SimpleNamespace(actor_id=actor_id, method=method, arguments=arguments))
            for name, method in methods.items()})
        carla.WeatherParameters = lambda **values: SimpleNamespace(**values)
        carla.MapLayer = SimpleNamespace(Buildings="Buildings")
        monkeypatch.setitem(sys.modules, "carla", carla)
        monkeypatch.setitem(sys.modules, "avcarla", ModuleType("avcarla"))
        monkeypatch.setitem(sys.modules, "avcarla.config", SimpleNamespace(CARLA=self))
        return self


def alpamayo_model_boundary(base, monkeypatch):
    """Replace Alpamayo checkpoint construction while using native policy records."""
    class PolicyModel:
        def __init__(self):
            self.closed = False
            self.last_input = None
            self.prediction = base.ModelPrediction(
                candidate_positions=np.array([[[1., 0., 0.], [2., 0., 0.]]]),
                candidate_rotations=np.tile(np.eye(3), (1, 2, 1, 1)),
                selected_index=0, selected_plan="forward", reasoning_text="clear",
            )

        def predict(self, inputs):
            self.last_input = inputs
            return self.prediction

        def close(self):
            self.closed = True

    class ModelLoader:
        @classmethod
        def from_config(cls, **options):
            model = PolicyModel()
            model.options = options
            return model

    # Keep real AlpaSim protobufs, navigation and model input/output records. Only weights
    # and checkpoint-backed construction are replaced.
    name = "alpasim_driver.models.alpamayo1_5_model"
    module = ModuleType(name)
    module.Alpamayo15Model = ModelLoader
    monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(create=PolicyModel)


def camera_detector_boundary(api, monkeypatch):
    """Replace checkpoint-backed image inference, leaving the real camera policy intact."""
    import torch
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from avstack.config import MODELS
    import avstack.modules.perception.object2dfv  # noqa: F401

    class ImageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(1.))
            self.data_preprocessor = DetDataPreprocessor()
            self.boxes = [[8., 8., 16., 16.]]
            self.labels = [0]
            self.last_images = []

        def extract_feat(self, images):
            values = torch.nn.functional.avg_pool2d(images.mean(dim=1, keepdim=True), 4)
            return (values * self.gain / 255.,)

        def rpn_head(self, features):
            return (list(features), [])

        def predict(self, rgb):
            return SimpleNamespace(pred_instances=SimpleNamespace(
                bboxes=torch.tensor(self.boxes).reshape(-1, 4),
                scores=torch.full((len(self.boxes),), .9), labels=torch.tensor(self.labels),
            ))

    def predict(detector, images):
        if isinstance(images, list):
            detector.last_images = [image.copy() for image in images]
            return [detector.predict(image) for image in images]
        return detector.predict(images)

    model = ImageModel()
    # Only the learned detector is replaced. Preprocessing, scorer and driving adapter stay real.
    monkeypatch.setitem(MODELS.module_dict, "MMDetObjectDetector2D",
                        lambda **options: SimpleNamespace(model=model))
    monkeypatch.setattr(api, "inference_detector", lambda detector, images: predict(detector, images))
    return model


def yolox_detection_model(predictions=None, *, image_weight=0.):
    """Create raw YOLOX predictions with optional image-dependent objectness."""
    import torch

    if predictions is None:
        probabilities = [0.] * 80
        probabilities[11] = 1.
        predictions = [[50., 40., 4., 4., .9, *probabilities]]

    class DetectionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(float(image_weight)))
            self.last_input = None

        def forward(self, images):
            self.last_input = images.detach().clone()
            rows = images.new_tensor(predictions).reshape(1, -1, 85).expand(len(images), -1, -1)
            objectness = rows[..., 4:5] + images.mean(dim=(1, 2, 3))[:, None, None] / 255 * self.weight
            return torch.cat((rows[..., :4], objectness, rows[..., 5:]), dim=-1)

    return DetectionModel()


class TrafficServiceBoundary:
    """A real in-process gRPC TrafficService with deterministic responses and request records.

    Session, simulate and close requests are copied into sessions, requests and
    closed. Tests may set simulate_response(request), rpc_errors[method] =
    (grpc.StatusCode, details), or delays[method] to replace this native boundary. With
    ``service_container`` set, scene queries answer UNAVAILABLE unless that Docker fake state
    reports a running container.
    """

    def __init__(self, scene_ids=("scene",), minimum_history_length_us=1_600_000,
                 simulate_response=None, service_container=None):
        self.scene_ids = tuple(scene_ids)
        self.service_container = service_container
        self.minimum_history_length_us = minimum_history_length_us
        self.simulate_response = simulate_response
        self.sessions = []
        self.requests = []
        self.closed = []
        self.rpc_errors = {}
        self.delays = {}
        self.endpoint = None
        self._server = None

    def capture_channels(self, monkeypatch):
        """Observe actual gRPC channel cleanup at the native transport boundary."""
        import grpc

        create = grpc.insecure_channel
        self.channels = []

        class Channel:
            def __init__(self, *args, **kwargs):
                self.native = create(*args, **kwargs)
                self.closed = False
                owner.channels.append(self)

            def __getattr__(self, name):
                return getattr(self.native, name)

            def close(self):
                self.closed = True
                self.native.close()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        owner = self
        monkeypatch.setattr(grpc, "insecure_channel", Channel)

    def __enter__(self):
        from concurrent.futures import ThreadPoolExecutor
        import time

        import grpc
        from alpasim_grpc.v0 import common_pb2, traffic_pb2, traffic_pb2_grpc

        owner = self

        def before(method, context):
            time.sleep(owner.delays.get(method, 0.))
            if method in owner.rpc_errors:
                code, details = owner.rpc_errors[method]
                context.abort(code, details)

        class Service(traffic_pb2_grpc.TrafficServiceServicer):
            def get_metadata(self, request, context):
                before("get_metadata", context)
                return traffic_pb2.TrafficModuleMetadata(
                    minimum_history_length_us=owner.minimum_history_length_us)

            def get_available_scenes(self, request, context):
                before("get_available_scenes", context)
                if not container_running(owner.service_container):
                    context.abort(grpc.StatusCode.UNAVAILABLE, "traffic is not running")
                return common_pb2.AvailableScenesReturn(scene_ids=owner.scene_ids)

            def start_session(self, request, context):
                owner.sessions.append(deepcopy(request))
                before("start_session", context)
                return common_pb2.SessionRequestStatus()

            def close_session(self, request, context):
                owner.closed.append(deepcopy(request))
                before("close_session", context)
                return common_pb2.Empty()

            def simulate(self, request, context):
                owner.requests.append(deepcopy(request))
                before("simulate", context)
                if owner.simulate_response is not None:
                    return owner.simulate_response(request)
                # A deterministic replay response at the requested instant. The
                # runtime tests can replace it with controlled reactive poses.
                session = next(item for item in owner.sessions if item.session_uuid == request.session_uuid)
                updates = []
                for obj in session.logged_object_trajectories:
                    points = obj.trajectory.poses
                    if not points:
                        continue
                    timestamp = request.time_query_us
                    if not obj.is_static and not (points[0].timestamp_us <= timestamp <= points[-1].timestamp_us):
                        continue
                    pose = min(points, key=lambda point: abs(point.timestamp_us - timestamp)).pose
                    updates.append(traffic_pb2.ObjectTrajectoryUpdate(
                        object_id=obj.object_id, trajectory=common_pb2.Trajectory(poses=[
                            common_pb2.PoseAtTime(timestamp_us=timestamp, pose=pose)])))
                return traffic_pb2.TrafficReturn(object_trajectory_updates=updates)

        self._server = grpc.server(ThreadPoolExecutor(max_workers=4))
        traffic_pb2_grpc.add_TrafficServiceServicer_to_server(Service(), self._server)
        port = self._server.add_insecure_port("127.0.0.1:0")
        self.endpoint = f"127.0.0.1:{port}"
        self._server.start()
        return self

    def __exit__(self, *exc):
        if self._server is not None:
            self._server.stop(0).wait()


def docker_process_boundary(tmp_path, monkeypatch, *, extension=None):
    """A deterministic Docker process boundary.

    DOCKER_MODE selects ready, race, run-fail, exit, timeout, interrupt or stop-fail.
    Container state and every command argument are stored under tmp_path.
    Interrupt mode assumes the Docker subprocess's parent is the test process.
    A DOCKER_STATE marker ensures that only its first log request sends SIGINT.
    """
    import os
    import sys

    directory = tmp_path / "bin"
    directory.mkdir()
    state = tmp_path / "containers"
    state.mkdir()
    calls = tmp_path / "docker.jsonl"
    docker = directory / "docker"
    docker.write_text(f"#!{sys.executable}\n" + r'''
import json, os, sys, uuid
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["DOCKER_CALLS"], "a") as stream:
    stream.write(json.dumps(args) + "\n")
root = Path(os.environ["DOCKER_STATE"])
mode = os.environ.get("DOCKER_MODE", "ready")
def read(p):
    try:
        return json.loads(p.read_text())
    except (ValueError, FileNotFoundError):
        return None

def locate(key):
    direct=root/key
    if direct.exists():
        return direct
    for path in root.iterdir():
        data = read(path)
        if isinstance(data, dict) and data.get("id") == key:
            return path
    return direct
if args[0] == "run":
    if mode == "run-fail":
        print("injected docker run failure", file=sys.stderr)
        raise SystemExit(1)
    name=args[args.index("--name")+1]
    labels=dict(args[i+1].split("=",1) for i,a in enumerate(args) if a=="--label")
    if mode=="race":labels["avsectester.service-owner"]="someone-else"
    identity="id-"+uuid.uuid4().hex
    state = {"id":identity,"owner":labels.get("avsectester.service-owner",""),
             "labels":labels,"running":mode!="exit"}
    (root/name).write_text(json.dumps(state))
    print(identity)
    raise SystemExit(1 if mode == "race" else 0)
if args[0] not in {"inspect", "logs", "stop", "rm"}:
    import runpy

    extension = os.environ.get("DOCKER_EXTENSION")
    if extension:
        runpy.run_path(extension, init_globals={"args": args, "locate": locate, "read": read,
                                               "calls": Path(os.environ["DOCKER_CALLS"])})
    raise SystemExit(f"Unsupported test Docker command: {args[0]}")
p=locate(args[-1]);data=read(p)
if isinstance(data,bool):data={"id":"seed-"+p.name,"owner":"","running":data}
if args[0] == "inspect":
    if data is None:raise SystemExit(1)
    fmt=args[args.index("--format")+1] if "--format" in args else ""
    if ".Config.Labels" in fmt:
        import re
        labels=data.get("labels") or {"avsectester.service-owner":data.get("owner","")}
        keys=re.findall(r'index \.Config\.Labels "([^"]+)"',fmt)
        print(" ".join([data["id"],*(labels.get(key) or "<no value>" for key in keys)]))
    elif ".State.Running" in fmt:print(str(data["running"]).lower())
    else:print(json.dumps(data))
elif args[0] == "logs":
    marker = root / ".interrupted"
    if mode == "interrupt" and not marker.exists():
        import signal
        marker.touch()
        os.kill(os.getppid(), signal.SIGINT)
    print("Serving on\nStarting CATK traffic gRPC server on" if mode == "ready" else "startup diagnostic")
elif args[0] == "stop":
    if data is None:
        print("No such container", file=sys.stderr)
        raise SystemExit(1)
    data["running"] = False
    p.write_text(json.dumps(data))
    if mode == "stop-fail":
        print("stop failed", file=sys.stderr)
        raise SystemExit(1)
elif args[0] == "rm":
    if data is None:raise SystemExit(1)
    p.unlink()
''')
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("DOCKER_STATE", str(state))
    monkeypatch.setenv("DOCKER_CALLS", str(calls))
    if extension is not None:
        monkeypatch.setenv("DOCKER_EXTENSION", str(extension))
    return state, calls
