#!/usr/bin/env python
"""Paint the white demo car mesh, close its side gap and add wheels.

The vehicle-insertion demo supplies ``real_car.ply``, a white mesh without wheels. This script
writes ``real_car_painted.ply`` next to it with the same body geometry and these changes:

* Faces are subdivided to at most 5 cm so that painted regions have sharp boundaries.
* The body is blue and window panes are dark. Lower trim, tail lights, plate, headlights and
  grille have their own colors.
* Simple shading is baked into the colors, with upward faces brighter and low parts darker.
* A small side gap is closed. The bottom opening stays open because sensors above the body
  never see it.
* The body is scaled per axis to 4.2 m long, 1.9 m wide and 1.6 m high, measured from the
  ground, and centered horizontally on the origin.
* Four tires with rims are added under the scaled wheel arches, touching z = 0. Their radius
  follows the height scale so that they stay round.

Window panes, wheel positions and light regions are specific to ``real_car.ply``. The output
stores one color per face and loads with ``load_mesh_asset(..., origin="file")``.

    docker compose run --rm --entrypoint python nurec-experiment \
        -m scripts.preparation.paint_demo_vehicle
"""

import argparse
from pathlib import Path

import numpy as np

MODELS = (Path(__file__).resolve().parents[2]
          / "avsectester/implemented_attacks/vehicle_insertion_attack/models")
BODY = np.array([52, 78, 120])
GLASS = np.array([26, 32, 40])
TRIM = np.array([38, 38, 40])
TAIL_LIGHT = np.array([150, 18, 18])
HEAD_LIGHT = np.array([215, 215, 205])
PLATE = np.array([205, 205, 195])
GRILLE = np.array([24, 24, 26])
TIRE = np.array([22, 22, 22])
RIM = np.array([125, 127, 132])
WHEEL_CENTERS_X = (-1.2, 1.22)
WHEEL_RADIUS_M, WHEEL_WIDTH_M, WHEEL_Y_M, RIM_RADIUS_M = .34, .22, .91, .21
MAX_EDGE_M = .05
TARGET_SIZE_M = np.array([4.2, 1.9, 1.6])


def boundary_loops(mesh):
    """Chain the mesh's open boundary edges into vertex loops."""
    unique, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    adjacent = {}
    for a, b in unique[counts == 1]:
        adjacent.setdefault(a, []).append(b)
        adjacent.setdefault(b, []).append(a)
    seen, loops = set(), []
    for start in adjacent:
        if start in seen:
            continue
        loop, previous, current = [start], None, start
        seen.add(start)
        while True:
            following = [n for n in adjacent[current] if n != previous and n not in seen]
            if not following:
                break
            previous, current = current, following[0]
            loop.append(current)
            seen.add(current)
        loops.append(loop)
    return loops


def glass_faces(mesh):
    """Window panes: large faces in the cabin band facing sideways, forward or backward."""
    centroid, normal, area = mesh.triangles_center, mesh.face_normals, mesh.area_faces
    cabin = (centroid[:, 2] > 1.3) & (centroid[:, 2] < 1.9)
    windshield = cabin & (normal[:, 0] > .5) & (normal[:, 2] > .5)
    rear_window = cabin & (normal[:, 0] < -.7)
    side_window = cabin & (np.abs(normal[:, 1]) > .85) & (area > .05)
    return windshield | rear_window | side_window


def shaded(colors, normal, z):
    """Brighten upward faces and darken low parts."""
    shade = .62 + .38 * np.clip(normal[:, 2], 0, 1)
    shade *= .8 + .2 * np.clip((z - .37) / .6, 0, 1)
    return np.clip(colors * shade[:, None], 0, 255)


def body_colors(centroid, normal, glass):
    """Region colors for subdivided body faces."""
    _, y, z = centroid.T
    colors = np.tile(BODY, (len(centroid), 1)).astype(float)
    colors[z < .52] = TRIM
    rear, front = normal[:, 0] < -.7, normal[:, 0] > .7
    colors[rear & (z > .85) & (z < 1.05) & (np.abs(y) > .55) & (np.abs(y) < .98)] = TAIL_LIGHT
    colors[rear & (z > .58) & (z < .72) & (np.abs(y) < .26)] = PLATE
    colors[front & (z > .8) & (z < .98) & (np.abs(y) > .55) & (np.abs(y) < .95)] = HEAD_LIGHT
    colors[front & (z > .6) & (z < .8) & (np.abs(y) < .45)] = GRILLE
    colors[glass] = GLASS
    return shaded(colors, normal, z)


def wheels(center, scale):
    """Tires with rim discs on their outer faces, as meshes with face colors.

    Wheel positions move with the body transform. Radii use the height scale.
    """
    import trimesh

    rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    radius, width = WHEEL_RADIUS_M * scale[2], WHEEL_WIDTH_M * scale[1]
    result = []
    for center_x in WHEEL_CENTERS_X:
        for side in (-1, 1):
            tire = trimesh.creation.cylinder(radius, width, sections=40, transform=rotation)
            rim = trimesh.creation.cylinder(RIM_RADIUS_M * scale[2], .01, sections=40,
                                            transform=rotation)
            rim.apply_translation([0, side * (width / 2 + .005), 0])
            x, y, _ = (np.array([center_x, side * WHEEL_Y_M, 0.]) - center) * scale
            for mesh, color in ((tire, TIRE), (rim, RIM)):
                mesh.apply_translation([x, y, radius])
                colors = np.tile(color, (len(mesh.faces), 1)).astype(float)
                result.append((mesh.vertices, mesh.faces,
                               shaded(colors, mesh.face_normals, mesh.triangles_center[:, 2])))
    return result


def paint(source: Path, output: Path) -> None:
    import trimesh

    original = trimesh.load(source, process=False)
    mesh = trimesh.Trimesh(original.vertices, original.faces[original.area_faces > 1e-9],
                           process=False)
    glass = glass_faces(mesh)
    for loop in boundary_loops(mesh):
        points = mesh.vertices[loop]
        if points[:, 2].max() < .5:
            continue
        center = len(mesh.vertices)
        fill = [[loop[i], loop[(i + 1) % len(loop)], center] for i in range(len(loop))]
        mesh = trimesh.Trimesh(np.vstack((mesh.vertices, points.mean(axis=0))),
                               np.vstack((mesh.faces, fill)), process=False)
        glass = np.concatenate((glass, np.zeros(len(fill), bool)))
    vertices, faces, index = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=MAX_EDGE_M, return_index=True)
    body = trimesh.Trimesh(vertices, faces, process=False)
    body_rgb = body_colors(body.triangles_center, body.face_normals, glass[index])
    # Regions are painted in source coordinates. Sizes are measured from the ground at z = 0.
    lower, upper = body.bounds
    center = np.array([(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, 0.])
    scale = TARGET_SIZE_M / np.array([upper[0] - lower[0], upper[1] - lower[1], upper[2]])
    pieces = [((vertices - center) * scale, faces, body_rgb)]
    pieces.extend(wheels(center, scale))
    offset, all_vertices, all_faces, all_colors = 0, [], [], []
    for piece_vertices, piece_faces, piece_colors in pieces:
        all_vertices.append(piece_vertices)
        all_faces.append(piece_faces + offset)
        all_colors.append(piece_colors)
        offset += len(piece_vertices)
    colors = np.concatenate(all_colors).round().astype(np.uint8)
    painted = trimesh.Trimesh(
        np.concatenate(all_vertices), np.concatenate(all_faces),
        face_colors=np.column_stack((colors, np.full(len(colors), 255, np.uint8))), process=False)
    painted.export(output)
    print(f"Wrote {output} with {len(painted.faces)} faces")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=MODELS / "real_car.ply")
    parser.add_argument("--output", type=Path, default=MODELS / "real_car_painted.ply")
    options = parser.parse_args()
    paint(options.source, options.output)


if __name__ == "__main__":
    main()
