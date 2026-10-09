"""Offline regressions for experiment method selection and saved driving results."""

import importlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.metric import impact
from avsectester.simulators.nurec import StubRenderer


@pytest.fixture
def demo():
    return importlib.import_module("scripts.demos.nurec.alpamayo_attack_demo")


@pytest.mark.parametrize("failure_stage", ["load", "inference"])
def test_experiment_pctnet_failure_raises_without_classic_fallback(demo, monkeypatch, failure_stage):
    harmonizer = demo.make_harmonizer("libcom", gpu=0)

    def fail(*args):
        raise RuntimeError(f"controlled {failure_stage} failure")

    def unexpected_fallback(*args):
        pytest.fail("An experiment labelled libcom must not run the classic harmonizer")

    monkeypatch.setattr(harmonizer, "_fallback", unexpected_fallback)
    if failure_stage == "load":
        monkeypatch.setattr(harmonizer, "_load", fail)
    else:
        monkeypatch.setattr(harmonizer, "_load", lambda: fail)
        harmonizer._dev = "cpu"

    frame = np.full((16, 16, 3), 90, np.uint8)
    mask = np.full((16, 16), 255, np.uint8)
    with pytest.raises(RuntimeError, match=f"PCTNet harmonization failed: controlled {failure_stage}"):
        harmonizer(frame, mask, frame)


def test_saved_steps_pair_reasoning_with_post_step_trace(demo, monkeypatch, tmp_path):
    class Renderer(StubRenderer):
        def __init__(self, **kwargs):
            super().__init__(cameras=kwargs["cameras"], height=8, width=8)

        def camera_model(self):
            return None  # The synthetic insertion below needs no geometry.

    class Model:
        def predict(self, observation):
            attacked = bool(observation.sensor_data[demo.CAM].any())
            speed = ([1.0, 0.0] if attacked else [3.0, 2.0])[observation.frame]
            return SimpleNamespace(
                candidate_positions=np.array([[[speed * 0.1, 0.0, 0.0]]]),
                reasoning_text=f"{'attacked' if attacked else 'clean'} step {observation.frame}",
            )

    def load_model(stack):
        stack._model = Model()

    # Keep the real stack adapter, run loop and trajectory follower. Replace only external inputs.
    monkeypatch.setattr(demo, "NuRecRenderer", Renderer)
    monkeypatch.setattr(demo.AlpamayoAVStack, "_load", load_model)
    monkeypatch.setattr(demo.AlpamayoAVStack, "_prediction_input", lambda self, obs: obs)
    def insertion_renderer(*args, compositor, **kwargs):
        assert compositor.harmonize is False
        return lambda obs, rgb: rgb + 10

    dataset = SimpleNamespace(insertion_renderer=insertion_renderer)
    monkeypatch.setattr(demo, "nurec_source", lambda *a: (dataset, SimpleNamespace(source={"scene_id": "test", "timestamp_us": 12345})))
    plotted = {}

    def capture_plot(clean, attacked, path, title):
        plotted.update(clean=clean, attacked=attacked)

    monkeypatch.setattr(demo, "plot_speed", capture_plot)
    monkeypatch.setattr(
        demo.sys, "argv",
        ["alpamayo_attack_demo.py", "--usdz", "test.usdz", "--frames", "2", "--speed", "5", "--harmonizer", "none",
         "--out", str(tmp_path)],
    )
    assert demo.main() == 0

    saved = json.loads((tmp_path / "trace.json").read_text())
    for label, speeds in (("clean", [3.0, 2.0]), ("attacked", [1.0, 0.0])):
        rows = saved[label]
        assert len(rows) == 2
        assert [row["input_t"] for row in rows] == pytest.approx([0.0, 0.1])
        assert [row["t"] for row in rows] == pytest.approx([0.1, 0.2])
        assert [row["speed"] for row in rows] == pytest.approx(speeds)
        assert [row["reasoning"] for row in rows] == [f"{label} step 0", f"{label} step 1"]
        for row, record in zip(rows, plotted[label].records, strict=True):
            assert {key: row[key] for key in asdict(record)} == asdict(record)
    assert saved["verdict"] == str(impact(plotted["clean"], plotted["attacked"]))


@pytest.mark.parametrize("yaw", [0, 0.3, -0.5])
def test_rear_attachment_preserves_vehicle_local_mount_and_texture_orientation(demo, yaw):
    from scripts.common.demo_common import rear_insertion
    from avsectester.insertion import ActorPose, resolve_insertion
    from avsectester.rendering.cameras import planar_rig_pose

    host = ActorPose(planar_rig_pose(20, -3, yaw, 1), (4, 2, 1.5))
    texture = np.full((16, 16, 4), 255, np.uint8)
    insertion = rear_insertion(texture, "lead", host, width_frac=0.85, height_frac=0.6)
    resolved = resolve_insertion(insertion, {"lead": host}, ActorPose(np.eye(4)))
    # The installation spans the same fraction of the rear box in the host frame.
    local = np.linalg.inv(host.transform)
    corners = resolved.planes()[0].corners
    corners = (local[:3, :3] @ corners.T).T + local[:3, 3]
    np.testing.assert_allclose(corners, [[-2, .85, .45], [-2, -.85, .45],
                                       [-2, -.85, -.45], [-2, .85, -.45]], atol=1e-12)
    assert resolved.host_id == "lead"


def test_carla_demo_binds_after_reset_and_keeps_the_same_host(demo, monkeypatch):
    from scripts.common.demo_common import carla_rear_perturbation
    from avsectester.insertion import ActorPose, resolve_insertion
    from avsectester.plane import Observation
    from avsectester.rendering.cameras import planar_rig_pose
    from avsectester.simulators import carla
    from avsectester.simulators.patch_insertion import PatchCompositor

    state = {}
    received = []
    backend = SimpleNamespace(selection_context=lambda: SimpleNamespace(actors=state))

    def adapter(backend, insertions, compositor):
        received.extend(insertions)
        return lambda obs: resolve_insertion(insertions[0], state, ActorPose(np.eye(4)))

    monkeypatch.setattr(carla, "insertion_perturbation", adapter)
    perturb = carla_rear_perturbation(backend, np.full((8, 8, 4), 255, np.uint8),
                                     PatchCompositor(harmonize=False))
    # Constructing the demo callback must not require actors before backend.reset().
    assert received == []
    state["lead"] = ActorPose(planar_rig_pose(10, 0, 0), (4, 2, 2))
    initial = perturb(Observation(0, 0))
    state["lead"] = ActorPose(planar_rig_pose(15, 2, 0.5), (4, 2, 2))
    state["nearer"] = ActorPose(planar_rig_pose(3, 0, 0), (5, 3, 2))
    moved = perturb(Observation(.1, 1))
    assert len(received) == 1
    assert initial.host_id == moved.host_id == "lead"
    np.testing.assert_allclose(np.linalg.inv(state["lead"].transform) @ moved.pose,
                               np.linalg.inv(planar_rig_pose(10, 0, 0)) @ initial.pose, atol=1e-12)


def test_nurec_demo_uses_metadata_start_time_without_changing_cruise_dynamics(demo, monkeypatch, tmp_path):
    from scripts.demos.nurec import nurec_object_demo as objects

    calls = []

    class Renderer(StubRenderer):
        def __init__(self, **kwargs):
            calls.append(kwargs)
            super().__init__(cameras=kwargs["cameras"], height=8, width=8)

    monkeypatch.setattr(objects, "NuRecRenderer", Renderer)
    scene = SimpleNamespace(source={"scene_id": "clipgt-test", "timestamp_us": 87654321})
    inputs = []

    def perturb(backend):
        inputs.append(backend)
        return lambda obs: obs

    clean = objects.drive("test", scene, 2, 3.7, tmp_path / "clean")
    attacked = objects.drive("test", scene, 2, 3.7, tmp_path / "attacked", perturb=perturb)
    assert len(clean) == len(attacked) == 2
    assert calls[0] == calls[1] == {
        "endpoint": "test", "scene_id": "clipgt-test", "cameras": [objects.CAM],
        "start_timestamp_us": 87654321,
    }
    assert inputs[0].pose.speed == pytest.approx(3.7)
    assert inputs[0].pose.t == pytest.approx(.2)


@pytest.mark.parametrize("name", ["stop", "standee", "billboard"])
def test_nuscenes_demo_keeps_candidate_positions_and_rejects_annotated_overlap(demo, name):
    from scripts.demos.nuscenes import nuscenes_object_demo as objects
    from avsectester.rendering.cameras import PinholeCamera
    from avsectester.simulators.patch_insertion import render_resolved

    camera = PinholeCamera(np.array([[400, 0, 400], [0, 400, 300], [0, 0, 1]]), 800, 600)
    transform = np.array([[0, -1, 0, 0], [0, 0, -1, 1.5], [1, 0, 0, 0], [0, 0, 0, 1.]])
    frame = np.full((600, 800, 3), 90, np.uint8)
    person = np.full((60, 20, 4), [50, 100, 150, 255], np.uint8)
    insertion = objects.free_spot(name, frame, camera, transform, [], person)
    assert insertion.placement.position_m == (*objects.SPOTS[name][0], 0)
    rendered, mask = render_resolved(frame, camera, transform, objects.resolved_object(insertion))
    np.testing.assert_array_equal(mask, objects.footprint(frame, camera, transform, insertion))
    assert np.any(rendered[mask > 0] != frame[mask > 0])
    assert objects.free_spot(name, frame, camera, transform, [(0, 0, 800, 600)], person) is None
