"""Legacy NuRec insertion adapter retained until concrete attacks migrate to Attack effects."""

from avsectester.rendering.compositor import InsertionRenderer


def insertion_renderer(simulator_runtime, scene, data, insertions, bindings=None, compositor=None):
    """Configure a host-bound insertion callable without connecting to the renderer.

    ``bindings`` accepts a selected match's role-to-track-ID tuples. Actor poses are sampled at
    the effective render timestamp and camera pose. Model-visible time and localization do
    not move the geometry. An explicit render-viewpoint change updates camera geometry while
    the victim's physical pose still comes from the runtime's physical state.
    """
    import numpy as np
    from avsectester.geometry.poses import ActorPose
    from avsectester.rendering.types import InsertionGeometry
    from avsectester.rendering.visibility import CuboidVisibilityEstimator

    renderer = simulator_runtime.renderer
    objects = {obj.track_id: obj for obj in scene.objects}
    selected = {
        role: tuple(objects[track_id] for track_id in ids)
        for role, ids in (bindings or {}).items()
    }
    from avsectester.scenarios.context import FilterContext
    from avsectester.geometry.cameras import FThetaCamera as Lens

    aliases = FilterContext(scene, bindings=selected).actor_aliases
    local_center = np.eye(4)
    local_center[:3, 3] = scene.ego.center
    path = scene.source["usdz_path"]

    def geometry(observation):
        pose = simulator_runtime.state().vehicle_state
        request = simulator_runtime.render_request(data.camera_id)
        adapter.camera = renderer.camera_model(request.camera)
        actors = data.actor_poses(path, renderer.timestamp_us(request.pose))
        for alias, track_id in aliases.items():
            if track_id in actors:
                if alias in actors and alias != track_id:
                    raise ValueError(f"Role {alias!r} conflicts with an actor track ID")
                actors[alias] = actors[track_id]
        victim = ActorPose(renderer.rig_transform(pose) @ local_center, scene.ego.extent)
        return InsertionGeometry(
            actors, victim, renderer.cam_from_world(request.pose, request.camera),
        )

    # Recorded intrinsics let the adapter be registered before reset. Each geometry
    # call then selects the loaded camera that produced the actual render request.
    recorded = scene.cameras[data.camera_alias].model
    camera = Lens(
        cx=float(recorded.principal_point[0]), cy=float(recorded.principal_point[1]),
        angle_to_pixeldist=tuple(recorded.angle_to_pixeldist),
        width=recorded.width, height=recorded.height, max_angle=recorded.max_angle,
        linear_cde=tuple(recorded.linear_cde),
    )
    adapter = NuRecInsertions(
        insertions,
        camera,
        geometry,
        compositor,
        visibility_estimator=CuboidVisibilityEstimator(),
        camera_name=data.camera_alias,
    )
    return adapter



class NuRecInsertions(InsertionRenderer):
    """Insertion renderer with optional known-cuboid visibility for NuRec observations.

    ``geometry(observation)`` returns InsertionGeometry with actor/victim poses and the
    world-to-camera matrix. Use with ``frame_perturbation`` to change the model input.
    """

    def __init__(
        self,
        insertions,
        camera,
        geometry,
        compositor=None,
        visibility_estimator=None,
        camera_name="front",
    ):
        def evidence(observation, resolved, state):
            return {
                item.id: visibility_estimator.estimate(
                    item,
                    self.camera,
                    state.cam_from_world,
                    occluders=state.actors,
                    other_insertions=tuple(other for other in resolved if other.id != item.id),
                    camera_name=camera_name,
                )
                for item in resolved
            }

        super().__init__(
            insertions,
            camera,
            geometry,
            compositor,
            evidence_provider=evidence if visibility_estimator is not None else None,
        )
