"""Legacy CARLA insertion adapter retained until concrete attacks migrate to Attack effects."""

def insertion_perturbation(simulator_runtime, insertions, *, bindings=None, camera="front",
                           compositor=None):
    """Legacy observation transform rendering explicit insertions into a CARLA camera image."""
    return _insertion_renderer(simulator_runtime, insertions, bindings=bindings, camera=camera,
                               compositor=compositor)


def _insertion_renderer(simulator_runtime, insertions, *, bindings=None, camera="front",
                        compositor=None):
    """Insert explicit, stable host-bound or world-fixed surfaces into model input.

    The runtime must provide ``selection_context()`` with current same-frame depth, as a
    runtime of ``CarlaSimulator(depth_camera=...)`` does.
    Geometry is updated each frame. Selection filters are never rerun here.
    Pass ``bindings=case.match.binding_ids`` to preserve selected role aliases across
    resets. Native logical host names such as ``lead`` need no additional mapping.
    """
    import numpy as np

    from avsectester.geometry.cameras import camera_from_calibration
    from avsectester.rendering.visibility import DepthVisibilityEstimator
    from avsectester.rendering.types import InsertionGeometry
    from avsectester.closed_loop.datatypes import read_camera_rgb, write_camera_rgb
    from avsectester.rendering.compositor import InsertionRenderer

    renderer = None
    current = None
    selected_ids = {}
    for role, ids in (bindings or {}).items():
        if isinstance(ids, str):
            raise TypeError("Each role binding must be a sequence of actor IDs, not one string")
        selected_ids[role] = tuple(ids)

    def geometry(observation):
        context = current
        calibration = context.scene.cameras[camera]
        return InsertionGeometry(context.actors, context.victim, np.linalg.inv(
            context.world_from_ego @ calibration.cam_to_ego
        ))

    def evidence_provider(observation, resolved, geometry_data):
        depth = current.native.get("depth", {}).get(camera)
        if depth is None:
            raise ValueError("CARLA insertion needs aligned depth for the selected camera")
        return {
            item.id: DepthVisibilityEstimator(tolerance_m=0.0002).estimate(
                item, renderer.camera, geometry_data.cam_from_world, scene_depth=depth,
                depth_convention="z", camera_name=camera,
                other_insertions=tuple(other for other in resolved if other.id != item.id),
            ) for item in resolved
        }

    def perturb(observation):
        nonlocal renderer, current
        current = simulator_runtime.selection_context()
        true_frame = simulator_runtime.state().frame
        if current.scene.frame != true_frame:
            raise ValueError("Insertion geometry and the physical world must use the same frame")
        if selected_ids:
            available = {actor.track_id: actor for actor in current.scene.objects}
            missing = {track_id for ids in selected_ids.values() for track_id in ids if track_id not in available}
            if missing:
                raise KeyError(f"Selected insertion actors unavailable: {sorted(missing)}")
            current = current.derive(bindings={
                role: tuple(available[track_id] for track_id in ids)
                for role, ids in selected_ids.items()
            })
        source_key = current.native.get("camera_sources", {}).get(camera, camera)
        if source_key not in observation.sensor_data:
            raise ValueError(f"Missing insertion camera {camera!r} (sensor {source_key!r})")
        payload_frame = getattr(observation.sensor_data[source_key], "frame", None)
        frame_offset = current.native.get("camera_frame_offsets", {}).get(camera)
        if payload_frame is not None and frame_offset is not None:
            if payload_frame + frame_offset != true_frame:
                raise ValueError("Insertion camera payload and geometry must use the same frame")
        got = read_camera_rgb(observation, source_key)
        if got is None:
            raise ValueError(f"Missing insertion camera {camera!r}")
        key, rgb = got
        if renderer is None:
            renderer = InsertionRenderer(
                insertions, camera_from_calibration(current.scene.cameras[camera]), geometry,
                compositor=compositor, evidence_provider=evidence_provider,
            )
        else:
            # A paired reset recreates sensors. Never retain a previous sensor's calibration.
            renderer.camera = camera_from_calibration(current.scene.cameras[camera])
        rendered = renderer(observation, rgb)
        perturb.evidence = renderer.evidence
        perturb.resolved = renderer.resolved
        return write_camera_rgb(observation, key, rendered)

    return perturb
