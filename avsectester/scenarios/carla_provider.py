"""Live CARLA candidate preparation and same-frame depth visibility.

These opt-in helpers own their preview actors and sensors. The normal CARLA backend's
reset behavior is unchanged. Multi-frame previews require an explicit scripted advance
callback, without a driving model or attack feedback.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from queue import Queue

import numpy as np

from avsectester.scenarios.carla_gt import carla_rgb_sensor, carla_scene_gt
from avsectester.scenarios.context import FilterContext
from avsectester.scenarios.estimators import DepthVisibilityEstimator, camera_from_calibration
from avsectester.simulators.carla import CarlaBackend


_PINHOLE_ATTRIBUTES = {"lens_k": "0", "lens_kcube": "0", "lens_circle_multiplier": "0"}


def register_pinhole_camera():
    """Register the selection backend's undistorted RGB sensor without importing CARLA eagerly.

    The class stays in this repository. The ordinary avcarla RGB sensor and other backends keep
    their original behavior. Sensor factories reuse the registered subclass and its calibration.
    """
    from avcarla.config import CARLA
    from avcarla.sensors import CarlaRgbCamera

    existing = CARLA.module_dict.get("PinholeRgbCamera")
    if existing is not None:
        if existing.__module__ != __name__:
            raise ValueError("PinholeRgbCamera is already registered by another module")
        return existing

    @CARLA.register_module()
    class PinholeRgbCamera(CarlaRgbCamera):
        """An ideal pinhole RGB sensor for calibrated depth-based insertion."""

        def __init__(self, *args, do_spawn=True, **kwargs):
            super().__init__(*args, do_spawn=False, **kwargs)
            for name, value in _PINHOLE_ATTRIBUTES.items():
                self.bp.set_attribute(name, value)
            if do_spawn:
                self.spawn()

    return PinholeRgbCamera


def _require_pinhole_sensor(sensor):
    """Reject camera geometry that the pinhole visibility provider cannot represent."""
    for name in _PINHOLE_ATTRIBUTES:
        value = float(sensor.attributes.get(name, -1 if name == "lens_k" else 0))
        if not np.isfinite(value) or value != 0:
            raise ValueError(
                f"Selection requires an undistorted camera ({name}=0), got {value}. "
                "Use PinholeRgbCamera or supply a custom calibrated visibility provider."
            )


def decode_carla_depth(image):
    """Decode CARLA's 24-bit depth image to forward camera depth in metres."""
    bgra = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
    channels = bgra.astype(np.float64)
    return (channels[..., 2] + 256 * channels[..., 1] + 65536 * channels[..., 0]) * (
        1000.0 / 16777215
    )


def depth_visibility(context, insertion_id, camera):
    """Evaluate the inserted surface against a matching native depth snapshot."""
    depth = context.native.get("depth", {}).get(camera)
    if depth is None:
        return None
    calibration = context.scene.cameras[camera]
    resolved = context.resolved_insertions
    return DepthVisibilityEstimator(tolerance_m=0.0002).estimate(
        resolved[insertion_id],
        camera_from_calibration(calibration),
        np.linalg.inv(context.world_from_ego @ calibration.cam_to_ego),
        scene_depth=depth,
        depth_convention="z",
        camera_name=camera,
        other_insertions=tuple(item for key, item in resolved.items() if key != insertion_id),
    )


def carla_filter_context(backend, *, camera="front", depth=None):
    """Expose normalized geometry and the original backend, world and actors.

    ``depth`` must belong to the current world snapshot and the selected RGB camera's
    optical pose and calibration. :class:`CarlaSelectionBackend` acquires it explicitly.
    Custom providers can instead supply their own renderer or visibility callback.
    """
    world = backend.client.world
    sensor = carla_rgb_sensor(backend, camera)
    return FilterContext(
        scene=carla_scene_gt(backend, camera),
        backend=backend,
        metadata={"scenario": backend.scenario, "replay_scenario": backend.replay_scenario},
        native={
            "world": world,
            "actors": world.get_actors(),
            "client": backend.client,
            "depth": {} if depth is None else {camera: np.array(depth, copy=True)},
            "camera_sources": {camera: sensor.source_identifier},
            "camera_frame_offsets": {camera: sensor.frame0},
        },
        visibility_provider=depth_visibility,
    )


class CarlaSelectionBackend(CarlaBackend):
    """CARLA backend with a registered depth sensor for insertion selection/rendering.

    Reset creates the ordinary scene, then advances one stationary setup frame to acquire
    the additional sensor. Every replay uses this same preparation path. This setup frame
    is outside the returned experiment. Depth and geometry use the same CARLA frame ID.

    This opt-in backend replaces stock ``CarlaRgbCamera`` configurations with
    ``PinholeRgbCamera`` in its own configuration copy. Both paired runs therefore use
    undistorted cameras matching the pinhole projection. Custom RGB classes are preserved
    and must expose zero lens distortion or use a different visibility provider.
    """

    def __init__(self, scenario, *, camera="front", **kwargs):
        scenario = deepcopy(scenario)
        for sensor in scenario.get("ego", {}).get("sensors", ()):
            if sensor.get("type") == "CarlaRgbCamera":
                sensor["type"] = "PinholeRgbCamera"
        super().__init__(scenario, **kwargs)
        self.selection_camera = camera
        self.depth = None
        self.depth_frame = None
        self._depth_queue = None

    def reset(self):
        import carla

        register_pinhole_camera()
        super().reset()
        rgb = carla_rgb_sensor(self, self.selection_camera)
        _require_pinhole_sensor(rgb.object)
        world = self.client.world
        blueprint = world.get_blueprint_library().find("sensor.camera.depth")
        for name in (
            "image_size_x",
            "image_size_y",
            "fov",
            "lens_k",
            "lens_kcube",
            "lens_circle_falloff",
            "lens_circle_multiplier",
            "lens_x_size",
            "lens_y_size",
        ):
            if name in rgb.object.attributes:
                blueprint.set_attribute(name, rgb.object.attributes[name])
        blueprint.set_attribute("sensor_tick", "0")
        relative = np.asarray(self.ego.actor.get_transform().get_inverse_matrix()) @ np.asarray(
            rgb.object.get_transform().get_matrix()
        )
        rotation = carla.Rotation(
            pitch=float(np.rad2deg(np.arcsin(np.clip(relative[2, 0], -1, 1)))),
            yaw=float(np.rad2deg(np.arctan2(relative[1, 0], relative[0, 0]))),
            roll=float(np.rad2deg(np.arctan2(-relative[2, 1], relative[2, 2]))),
        )
        sensor = world.spawn_actor(
            blueprint,
            carla.Transform(carla.Location(*relative[:3, 3]), rotation),
            attach_to=self.ego.actor,
        )
        self._resources.callback(sensor.destroy)
        self._resources.callback(sensor.stop)
        self._depth_queue = Queue()
        sensor.listen(self._depth_queue.put)
        self.client.tick()
        return self._observe()

    def _observe(self):
        observation = super()._observe()
        if self._depth_queue is not None:
            sample = self._depth_queue.get(timeout=30)
            while sample.frame < observation.frame:
                sample = self._depth_queue.get(timeout=30)
            if sample.frame != observation.frame:
                raise RuntimeError("CARLA depth and observation frames differ")
            self.depth = decode_carla_depth(sample)
            self.depth_frame = sample.frame
        return observation

    def selection_context(self):
        if self.depth_frame != self.client.world.get_snapshot().frame:
            raise RuntimeError("Refresh the backend observation before requesting a context")
        return carla_filter_context(self, camera=self.selection_camera, depth=self.depth)

    def close(self):
        try:
            super().close()
        finally:
            self._depth_queue = None
            self.depth = self.depth_frame = None


class CarlaCandidateProvider:
    """Prepare real candidates with an optional prescribed initial sequence.

    ``advance_initial(backend, index)`` must advance one prescribed frame and return an
    observation, e.g. by replaying recorded controls or poses. It runs only during scene
    selection. A selected experiment starts again at the first captured frame. The live
    context is evaluated once within its context manager. Filters run before the next
    prescribed frame is advanced, including filters that call native CARLA APIs.
    """

    def __init__(self, *, initial_frames=1, advance_initial=None, backend_factory=None):
        if (
            isinstance(initial_frames, bool)
            or not isinstance(initial_frames, int)
            or initial_frames < 1
        ):
            raise ValueError("initial_frames must be a positive integer")
        if initial_frames > 1 and advance_initial is None:
            raise ValueError(
                "Multiple initial frames require a prescribed advance_initial callback"
            )
        self.initial_frames = initial_frames
        self.advance_initial = advance_initial
        self.backend_factory = backend_factory or CarlaSelectionBackend

    @contextmanager
    def __call__(self, scenario):
        backend = self.backend_factory(deepcopy(scenario))
        try:
            backend.prepare_clean_attack_pair()
            backend.reset()
            replay = deepcopy(backend.replay_scenario)
            first = backend.selection_context()
            consumed = False

            def initial_contexts():
                nonlocal consumed
                if consumed:
                    raise RuntimeError("Prepare a new CARLA candidate before evaluating it again")
                consumed = True
                yield first
                for index in range(1, self.initial_frames):
                    self.advance_initial(backend, index)
                    yield backend.selection_context()

            first.initial_contexts = initial_contexts
            # Capture configuration, not the temporary live backend that closes on exit.
            first.native = {
                **first.native,
                "make_backend": lambda: self.backend_factory(deepcopy(replay)),
            }
            yield first
        finally:
            backend.close()
