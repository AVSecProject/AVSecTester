"""Validate NuRec ground alignment offline, with optional depth edges over NRE images.

Rig and nearby raw clipgt road points are compared vertically with mesh_ground. Road
distance is measured horizontally to the continuous polyline of every recorded rig sample.
Camera-depth errors are informational: grazing angles amplify small height offsets, and
foreground geometry can occlude road points. Only an explicit endpoint enables NRE overlays.
No scene transform is fitted and world_to_nre is not applied.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
from pathlib import Path
import sys

import numpy as np

from avsectester.datasets.nurec import NuRecDataset
from avsectester.geometry.cameras import camera_ray_mask
from avsectester.geometry.poses import transform
from avsectester.geometry.mesh import MeshRaycaster


def _percentile(values, fraction):
    """Linear sample quantile, retaining missing-hit infinities without inf-minus-inf."""
    if not len(values):
        return np.inf
    ordered = np.sort(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    if lower == upper:
        return ordered[lower]
    if not np.isfinite(ordered[upper]):
        return np.inf
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def road_points(dataset, scenario_id):
    """Read every finite original lane-rail and road-boundary point without resampling."""
    import pyarrow.parquet as parquet

    points, counts = [], {}
    with dataset.open_archive(dataset.usdz_path(scenario_id)) as archive:
        for layer, fields in (("lane", ("left_rail", "right_rail")),
                              ("road_boundary", ("location",))):
            name = f"clipgt/{layer}.parquet"
            if name not in archive.namelist():
                counts[layer] = 0
                continue
            rows = parquet.read_table(io.BytesIO(archive.read(name))).to_pylist()
            layer_points = []
            for row in rows:
                if (row.get("key") or {}).get("map_id") is None:
                    continue
                values = row.get(layer) or {}
                for field in fields:
                    for point in values.get(field) or ():
                        coordinates = [point.get(axis) for axis in ("x", "y", "z")]
                        if (all(value is not None for value in coordinates)
                                and np.isfinite(coordinates).all()):
                            layer_points.append(coordinates)
            points.extend(layer_points)
            counts[layer] = len(layer_points)
    if not points:
        raise ValueError("Road validation needs finite clipgt lane or road_boundary points")
    return np.asarray(points, dtype=np.float64), counts


def ground_raycaster(dataset, scenario_id):
    """Load only mesh_ground, without requiring a camera or a neural renderer."""
    import trimesh

    with dataset.open_archive(dataset.usdz_path(scenario_id)) as archive:
        if "mesh_ground.ply" not in archive.namelist():
            raise ValueError("Ground alignment validation requires mesh_ground.ply")
        mesh = trimesh.load(io.BytesIO(archive.read("mesh_ground.ply")), file_type="ply", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("mesh_ground.ply is not a triangle mesh")
    return MeshRaycaster(np.asarray(mesh.vertices), np.asarray(mesh.faces))


def vertical_heights(points, raycaster):
    """Return ground-z minus point-z from rays starting three metres above each point."""
    origins = np.array(points, dtype=np.float64, copy=True)
    origins[:, 2] += 3.
    distances = raycaster.distances(origins, np.tile([0., 0., -1.], (len(origins), 1)))
    return 3. - distances


def trajectory_distances(points, trajectory):
    """Exact horizontal distance to the continuous polyline through every recorded rig pose."""
    if not len(trajectory):
        raise ValueError("Road validation needs a recorded rig trajectory")
    points = np.asarray(points)[:, :2]
    trajectory = np.asarray(trajectory)[:, :2]
    distances = np.linalg.norm(points - trajectory[0], axis=1)
    for first, second in zip(trajectory, trajectory[1:]):
        vector = second - first
        length_squared = vector @ vector
        if length_squared == 0:
            continue
        fractions = np.clip((points - first) @ vector / length_squared, 0., 1.)
        offsets = points - first - fractions[:, None] * vector
        distances = np.minimum(distances, np.linalg.norm(offsets, axis=1))
    return distances


def road_heights(points, counts, raycaster, trajectory, road_radius_m):
    """Evaluate road height near the whole trajectory, reporting hits over all source points."""
    distances = trajectory_distances(points, trajectory)
    heights = vertical_heights(points, raycaster)
    finite = np.isfinite(heights)
    selected = distances <= road_radius_m
    point_count = int(np.count_nonzero(selected))
    measured = heights[selected & finite]
    hit_fraction = len(measured) / point_count if point_count else 0.
    median = float(np.median(measured)) if len(measured) else None
    absolute_median = float(np.median(np.abs(measured))) if len(measured) else None
    bands = {}
    for name, lower, upper in (("0-10m", 0., 10.), ("10-25m", 10., 25.),
            ("25-50m", 25., 50.), ("50-100m", 50., 100.), ("100+m", 100., np.inf)):
        in_band = (distances >= lower) & (distances < upper)
        bands[name] = {"point_count": int(np.count_nonzero(in_band)),
                       "hit_count": int(np.count_nonzero(in_band & finite))}
    return {
        "road_radius_m": road_radius_m, "trajectory_samples": len(trajectory),
        "distance_definition": "Minimum horizontal distance to the complete recorded rig polyline",
        "total_source_points": len(points), "source_point_counts": counts,
        "point_count": point_count, "hit_count": len(measured), "hit_fraction": hit_fraction,
        "missing_intersections": point_count - len(measured),
        "all_point_distance_bands": bands,
        "band_interval_convention": "Lower bound inclusive, upper bound exclusive; last band includes 100 m",
        "statistics_support": "Finite ground intersections; missing points are reported separately",
        "median_signed_height_m": median, "median_absolute_height_m": absolute_median,
        "p90_absolute_height_m": float(np.quantile(np.abs(measured), .9)) if len(measured) else None,
        "absolute_median_limit_m": .1,
        "minimum_hit_fraction": .9,
        "passed": hit_fraction >= .9 and absolute_median is not None and absolute_median < .1,
    }


def road_depth_errors(dataset, scenario_id, frames, camera, rig_from_camera, points):
    """Return relative mesh-depth errors binned by Euclidean camera distance."""
    vertices, faces = dataset.scene_mesh(scenario_id)
    raycaster = MeshRaycaster(vertices, faces)
    del vertices, faces
    bands = {(0, 10): [], (10, 25): [], (25, 50): []}
    counts = {band: 0 for band in bands}
    for frame in frames:
        ego = dataset.frame_data(scenario_id, frame).ego
        if ego is None:
            raise ValueError("Road validation needs recorded rig poses")
        world_from_camera = ego.pose @ rig_from_camera
        camera_points = transform(np.linalg.inv(world_from_camera), points)
        ranges = np.linalg.norm(camera_points, axis=1)
        eligible = camera_ray_mask(camera, camera_points) & (ranges < 50.)
        camera_points, ranges = camera_points[eligible], ranges[eligible]
        pixels = camera.project(camera_points)
        inside = (np.isfinite(pixels).all(axis=1)
                  & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width)
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height))
        pixels, camera_points, ranges = pixels[inside], camera_points[inside], ranges[inside]
        if not len(pixels):
            continue
        rays = camera.unproject(np.floor(pixels) + .5)
        supported = camera_ray_mask(camera, rays)
        rays, camera_points, ranges = rays[supported], camera_points[supported], ranges[supported]
        directions = rays @ world_from_camera[:3, :3].T
        origins = np.broadcast_to(world_from_camera[:3, 3], directions.shape)
        depth = raycaster.distances(origins, directions) * rays[:, 2]
        errors = np.abs(depth - camera_points[:, 2]) / camera_points[:, 2]
        for band in bands:
            lower, upper = band
            selected = (ranges >= lower) & (ranges < upper)
            bands[band].extend(errors[selected].tolist())
            counts[band] += int(np.count_nonzero(~np.isfinite(depth[selected])))
    del raycaster
    gc.collect()
    result = {}
    for band, values in bands.items():
        median = _percentile(values, .5)
        percentile = _percentile(values, .9)
        result[f"{band[0]}-{band[1]}m"] = {
            "samples": len(values), "missing_intersections": counts[band],
            "median_relative_error": float(median) if np.isfinite(median) else None,
            "p90_relative_error": float(percentile) if np.isfinite(percentile) else None,
            "informational_only": True,
        }
    return result


def recorded_trajectory(dataset, scenario_id):
    """Every recorded camera-time rig pose, independently of the selected diagnostic frames."""
    timestamps = dataset.camera_timestamps(scenario_id)
    origins = np.stack([dataset.ego_state(scenario_id, round(timestamp * 1e6)).pose[:3, 3]
                        for timestamp in timestamps])
    return timestamps, origins


def ground_heights(timestamps, origins, raycaster):
    """Record signed ground height at every recorded camera-time rig position."""
    heights = vertical_heights(origins, raycaster)
    finite = heights[np.isfinite(heights)]
    return {
        "trajectory_samples": len(origins),
        "missing_intersections": int(np.count_nonzero(~np.isfinite(heights))),
        "minimum_height_m": float(finite.min()) if len(finite) else None,
        "maximum_height_m": float(finite.max()) if len(finite) else None,
        "height_range_m": float(np.ptp(finite)) if len(finite) else None,
        "maximum_absolute_height_m": float(np.max(np.abs(finite))) if len(finite) else None,
        "absolute_height_limit_m": .1,
        "samples": [{"timestamp_us": round(float(timestamp) * 1e6),
                     "rig_position": origin.tolist(),
                     "signed_height_m": float(height) if np.isfinite(height) else None}
                    for timestamp, origin, height in zip(timestamps, origins, heights)],
        "passed": bool(len(origins) and len(finite) == len(origins) and (np.abs(finite) <= .1).all()),
    }


def depth_overlays(dataset, scenario_id, frames, endpoint, output, edge_threshold):
    """Use the actual simulator's mesh-plus-actor depth and NRE-rendered RGB images."""
    from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator
    from avsectester.evaluation.visualization import save_image

    renderer = NuRecRenderer(endpoint=endpoint, cameras=[dataset.camera_id])
    simulator = NuRecSimulator({"dt": dataset.time_step}, renderer, depth="mesh")
    images = []
    with simulator.create_runtime() as runtime:
        for frame in frames:
            runtime.reset(dataset.scenario_start(scenario_id, frame))
            view = runtime.sensor_view(dataset.camera_id)
            depth = view.depth
            if depth is None:
                raise ValueError("NuRec mesh mode did not provide depth")
            finite = np.isfinite(depth)
            edges = np.zeros(depth.shape, dtype=bool)
            horizontal = finite[:, 1:] & finite[:, :-1]
            vertical = finite[1:] & finite[:-1]
            delta_horizontal = np.zeros(depth[:, 1:].shape)
            delta_vertical = np.zeros(depth[1:].shape)
            np.subtract(depth[:, 1:], depth[:, :-1], out=delta_horizontal, where=horizontal)
            np.subtract(depth[1:], depth[:-1], out=delta_vertical, where=vertical)
            edges[:, 1:] |= (finite[:, 1:] != finite[:, :-1]) | (np.abs(delta_horizontal) > edge_threshold)
            edges[1:] |= (finite[1:] != finite[:-1]) | (np.abs(delta_vertical) > edge_threshold)
            image = view.data.copy()
            image[edges] = [255, 0, 255]
            destination = output / f"depth_edges_{frame.position:05d}.png"
            save_image(image, destination)
            images.append({"position": frame.position, "timestamp": frame.timestamp,
                           "image": str(destination)})
    return images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usdz", type=Path, required=True, help="NuRec reconstruction archive")
    parser.add_argument("--output", type=Path, required=True, help="JSON report and overlay directory")
    parser.add_argument("--endpoint", help="Generate overlays using this running NRE gRPC endpoint")
    parser.add_argument("--camera-id", default="camera_front_wide_120fov")
    parser.add_argument("--frames", type=int, default=5, help="Evenly spaced camera-depth/overlay frames")
    parser.add_argument("--time-step", type=float, default=.1)
    parser.add_argument("--road-radius-m", type=float, default=10.,
                        help="Road-height evaluation radius around the full rig trajectory, in metres")
    parser.add_argument("--edge-threshold", type=float, default=.5, help="Depth edge threshold in metres")
    arguments = parser.parse_args(argv)
    if arguments.frames <= 0 or not np.isfinite(arguments.edge_threshold) or arguments.edge_threshold <= 0:
        parser.error("frames and edge-threshold must be positive")
    if not np.isfinite(arguments.road_radius_m) or arguments.road_radius_m <= 0:
        parser.error("road-radius-m must be positive and finite")
    arguments.output.mkdir(parents=True, exist_ok=True)
    report = {"usdz": str(arguments.usdz), "passed": False,
              "distance_bands": "Euclidean camera range; optical-axis z error",
              "coordinate_transform": "identity",
              "height_definition": "Ground z minus point z = 3 m minus downward hit distance",
              "camera_depth_note": (
                  "Camera-depth bands are informational only. Road points are seen at grazing "
                  "angles, so centimetre-scale height offsets can produce several percent depth "
                  "error. Mesh guardrails and vehicles can also occlude the projected road points."
              )}
    try:
        dataset = NuRecDataset([str(arguments.usdz)], camera_id=arguments.camera_id,
                               time_step=arguments.time_step)
        scenario_id = next(dataset.scenarios()).id
        timeline = dataset.frames(scenario_id)
        frames = [timeline[int(position)] for position in np.unique(
            np.linspace(0, len(timeline) - 1, arguments.frames).astype(int))]
        raycaster = ground_raycaster(dataset, scenario_id)
        timestamps, trajectory = recorded_trajectory(dataset, scenario_id)
        report["ground_height"] = ground_heights(timestamps, trajectory, raycaster)
        points, counts = road_points(dataset, scenario_id)
        report["road_height"] = road_heights(points, counts, raycaster, trajectory, arguments.road_radius_m)
        del raycaster
        gc.collect()
        camera, rig_from_camera = dataset.camera_geometry(scenario_id)
        report["camera_calibration"] = "Native USDZ camera parameters and extrinsic; no fitted coefficients"
        try:
            camera.unproject(np.array([[camera.width / 2, camera.height / 2]]))
        except NotImplementedError as error:
            report["road_depth"] = {
                "status": "unavailable", "informational_only": True, "reason": str(error),
            }
        else:
            report["road_depth"] = road_depth_errors(
                dataset, scenario_id, frames, camera, rig_from_camera, points)
        if arguments.endpoint is None:
            report["overlays"] = {"status": "skipped", "reason": "No --endpoint supplied; no NRE service used"}
        else:
            images = depth_overlays(dataset, scenario_id, frames, arguments.endpoint,
                                    arguments.output, arguments.edge_threshold)
            report["overlays"] = {"status": "completed", "images": images}
        report["passed"] = report["ground_height"]["passed"] and report["road_height"]["passed"]
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], file=sys.stderr)
    destination = arguments.output / "depth_validation.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Report: {destination}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
