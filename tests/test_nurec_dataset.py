"""nuRec ``.usdz`` -> ``SceneGT``: the label reader + requirement filter, tested without a real artifact.

nuRec scenes ship as license-gated multi-GB ``.usdz`` files, so these tests build a **synthetic** ``.usdz``
(a ZIP holding ``sequence_tracks.json`` + ``rig_trajectories.json`` in the exact schema
``alpasim_utils.scenario`` reads) with a *known* geometry, and assert the reader recovers the ego-frame
ground truth and that the requirement predicate then runs on it. This validates parsing + the
NRE->world->rig composition + the pinhole projection end-to-end, offline and on CPU.
"""

import json
import math
import zipfile

import numpy as np
from avsectester.scenarios.datasets.nurec import NuRecDataset, scene_from_tracks
from avsectester.scenarios.requirements import PHYSICAL_PATCH_HIDE_VEHICLE as REQ
from avsectester.scenarios.source import DatasetFilter

I4 = np.eye(4).tolist()


def _quat_z(theta):  # xyzw quaternion for a yaw rotation of theta about +z
    return [0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2)]


def _make_usdz(path, tracks, world_to_nre=I4, t_rig_worlds=(I4, I4), ts=(1_000, 2_000)):
    """Write a minimal ``.usdz`` (ZIP). ``tracks`` = list of dicts {id,label,dims,pos,quat}."""
    seq = {
        "tracks_data": {
            "tracks_id": [t["id"] for t in tracks],
            "tracks_label_class": [t["label"] for t in tracks],
            "tracks_flags": [["CONTROLLABLE"] for _ in tracks],
            "tracks_timestamps_us": [list(ts) for _ in tracks],
            "tracks_poses": [[list(t["pos"]) + list(t["quat"])] * len(ts) for t in tracks],
        },
        "cuboidtracks_data": {"cuboids_dims": [list(t["dims"]) for t in tracks]},
    }
    rig = {
        "world_to_nre": {"matrix": world_to_nre},
        "camera_calibrations": {"u-front": {"logical_sensor_name": "camera_front_wide_120fov"}},
        "rig_trajectories": [{
            "sequence_id": "clipgt-test",
            "T_rig_world_timestamps_us": list(ts),
            "T_rig_worlds": list(t_rig_worlds),
        }],
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("sequence_tracks.json", json.dumps({"seq0": seq}))
        zf.writestr("rig_trajectories.json", json.dumps(rig))
    return str(path)


def _vehicle(pos=(6.0, 0.0, 0.8), quat=None, dims=(4.5, 2.0, 1.6), label="car", id="a0"):
    return {"id": id, "label": label, "dims": dims, "pos": pos, "quat": quat or [0.0, 0.0, 0.0, 1.0]}


def test_scene_from_tracks_recovers_ego_frame_gt():
    # world == nre == rig (all identity): a car 6 m dead ahead, rear facing us
    tracks = [{"track_id": "a0", "label_class": "car", "dims": (4.5, 2.0, 1.6),
               "pos": np.array([6.0, 0.0, 0.8]), "quat": np.array([0.0, 0.0, 0.0, 1.0])}]
    from avsectester.scenarios.datasets.nurec import _pinhole_k
    scene = scene_from_tracks(tracks, np.eye(4), np.eye(4), _pinhole_k(1920, 1080, 120.0),
                              1920, 1080, 1_000, ego_speed=5.0)
    assert len(scene.objects) == 1
    o = scene.objects[0]
    assert o.category == "vehicle" and abs(o.distance - 6.0) < 0.05 and o.ahead
    assert abs(o.yaw) < 1e-6 and o.center[:2] == (6.0, 0.0)      # heading == ego => rear faces us
    assert "front" in o.box2d and REQ.match(scene) is not None    # a close rear-facing lead qualifies


def test_nurec_dataset_reads_usdz_and_filters(tmp_path):
    p = _make_usdz(tmp_path / "clipgt-x.usdz", [_vehicle()])
    ds = NuRecDataset([p], width=1920, height=1080)
    scenes = list(ds.scenes())
    assert len(scenes) == 1 and scenes[0].source["dataset"] == "nurec"
    o = scenes[0].objects[0]
    assert o.category == "vehicle" and abs(o.distance - 6.0) < 0.05
    assert len(list(DatasetFilter(ds).scenarios(REQ))) == 1        # this real-schema scene qualifies


def test_nre_to_world_to_rig_composition(tmp_path):
    # ego (rig) sits at world x=2 facing +x, so world->rig is a -2 m x-translation; an actor at world
    # x=8 must land at rig x=6. world_to_nre is identity here (tracks already in world coords).
    t_rig_world = np.eye(4)
    t_rig_world[0, 3] = -2.0
    p = _make_usdz(tmp_path / "clipgt-c.usdz", [_vehicle(pos=(8.0, 0.0, 0.8))],
                   t_rig_worlds=(t_rig_world.tolist(), t_rig_world.tolist()))
    o = next(NuRecDataset([p], width=1920, height=1080).scenes()).objects[0]
    assert abs(o.center[0] - 6.0) < 1e-6 and abs(o.center[1]) < 1e-6   # T_rig_world applied correctly


def test_facing_and_label_filters(tmp_path):
    # a vehicle facing away (heading -x => yaw pi) fails ViewpointRear; a pedestrian is not a target
    facing = _make_usdz(tmp_path / "f.usdz", [_vehicle(quat=_quat_z(math.pi))])
    o = next(NuRecDataset([facing], width=1920, height=1080).scenes()).objects[0]
    assert abs(abs(o.yaw) - math.pi) < 1e-6 and REQ.match(next(NuRecDataset([facing]).scenes())) is None
    ped = _make_usdz(tmp_path / "p.usdz", [_vehicle(label="pedestrian")])
    assert next(NuRecDataset([ped], width=1920, height=1080).scenes()).objects == []
