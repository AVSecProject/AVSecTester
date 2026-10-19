"""Live CARLA candidate preparation and same-frame depth visibility.

These opt-in helpers own their preview actors and sensors. The normal CARLA backend's
reset behavior is unchanged. Multi-frame previews require an explicit scripted advance
callback, without a driving model or attack feedback.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import numpy as np

from avsectester.scenarios.carla_gt import carla_rgb_sensor, carla_scene_gt
from avsectester.scenarios.context import FilterContext
from avsectester.geometry.cameras import camera_from_calibration
from avsectester.rendering.visibility import DepthVisibilityEstimator
from avsectester.closed_loop.simulators.carla import (  # noqa: F401  (re-exported)
    CarlaSimulator, _require_pinhole_sensor, decode_carla_depth, register_pinhole_camera,
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


def carla_filter_context(backend, *, camera="front", depth=None, observation=None):
    """Expose normalized geometry and the original backend, world and actors.

    ``depth`` must belong to the current world snapshot and the selected RGB camera's
    optical pose and calibration. A ``CarlaSimulator(depth_camera=...)`` runtime acquires it explicitly.
    Custom providers can instead supply their own renderer or visibility callback.
    """
    world = backend.client.world
    sensor = carla_rgb_sensor(backend, camera)
    return FilterContext(
        scene=carla_scene_gt(backend, camera),
        backend=backend,
        metadata={"scenario": backend.scenario, "replay_scenario": backend.replay_scenario},
        native={
            **({"observation": observation} if observation is not None else {}),
            "world": world,
            "actors": world.get_actors(),
            "client": backend.client,
            "depth": {} if depth is None else {camera: np.array(depth, copy=True)},
            "camera_sources": {camera: sensor.source_identifier},
            "camera_frame_offsets": {camera: sensor.frame0},
        },
        visibility_provider=depth_visibility,
    )


class CarlaCandidateProvider:
    """Prepare real candidates with an optional prescribed initial sequence.

    ``advance_initial(backend, index)`` must advance one prescribed frame and return an
    observation, e.g. by replaying recorded controls or poses. It runs only during scene
    selection. A selected experiment starts again at the first captured frame. The live
    context is evaluated once within its context manager. Filters run before the next
    prescribed frame is advanced, including filters that call native CARLA APIs.
    """

    def __init__(self, *, initial_frames=1, advance_initial=None, backend_factory=None, conditions=None):
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
        from avsectester.conditions import ScenarioConditions

        self.conditions = conditions or ScenarioConditions()
        self.initial_frames = initial_frames
        self.advance_initial = advance_initial
        self.backend_factory = backend_factory or _depth_runtime

    @contextmanager
    def __call__(self, scenario):
        backend = self.backend_factory(deepcopy(scenario))
        try:
            self.conditions.configure_backend(backend)
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
            replay_config = deepcopy(replay)
            if "weather" in self.conditions.world:
                replay_config.pop("weather", None)
            first.native = {
                **first.native,
                "make_backend": lambda: self.backend_factory(deepcopy(replay_config)),
            }
            yield first
        finally:
            backend.close()


def _depth_runtime(scenario):
    """A live CARLA world whose front camera has aligned depth, for selection and insertion."""
    return CarlaSimulator(scenario, depth_camera="front").create_runtime()
