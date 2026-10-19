"""CARLA demo entry points retain their deployment, driving and saved-image behavior."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import yaml

from avsectester.closed_loop.simulators.nurec import NuRecSimulatorRuntime, NuRecSimulator, StubRenderer


class CameraWorld(NuRecSimulatorRuntime):
    def __init__(self, config, initial_color=0):
        class Renderer(StubRenderer):
            def __init__(self):
                super().__init__(height=4, width=6)
                self.color = initial_color

            def render(self, pose, camera):
                return np.full((4, 6, 3), self.color, np.uint8)

        super().__init__(NuRecSimulator({"dt": 0.1}, renderer=Renderer()))
        self.original_config = config
        self.prepares = self.resets = self.closes = 0
        self.lead = object()

    def prepare_clean_attack_pair(self):
        self.prepares += 1

    def _initialize_world(self, source, initialization, environment):
        self.resets += 1
        return super()._initialize_world(source, initialization, environment)

    def close(self):
        self.closes += 1
        super().close()

    def _hold_tick(self):
        pass

    @property
    def native(self):
        return {"world": self, "actors": {"lead": self.lead}}


@pytest.fixture
def scene_file(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump({
        "client": {"seed": 31}, "ego": {}, "lead": {"gap": 9},
        "patches": [{"target": "lead", "texture": {"pattern": "checkerboard"}}],
    }))
    return path


def test_physical_patch_demo_prepares_once_and_deploys_texture_before_saved_camera_input(
    monkeypatch, scene_file, tmp_path,
):
    from avsectester.attacks.patch.physical_patch import PhysicalPatch
    from scripts.demos.carla import carla_patch_demo as demo

    worlds, deployments = [], []
    panel = SimpleNamespace(destroy=Mock(), is_alive=True)

    def make_world(config):
        world = CameraWorld(config)
        worlds.append(world)
        return world

    def apply(patch, world, target, *, tick, on_spawn):
        assert target is world.lead and world.resets == 1
        rgba = patch._rgba()
        deployments.append((patch, rgba))
        world.renderer.color = 200
        on_spawn(panel)
        return [panel]

    texture_path = tmp_path / "texture.png"
    Image.fromarray(np.full((8, 8, 4), [120, 10, 5, 255], np.uint8)).save(texture_path)
    output = tmp_path / "physical"
    monkeypatch.setattr(demo, "OUT", output)
    monkeypatch.setattr(demo, "CarlaSimulator", make_world)
    monkeypatch.setattr(PhysicalPatch, "apply", apply)
    monkeypatch.setattr(demo.sys, "argv", [
        "carla_patch_demo", "--config", str(scene_file), "--frames", "2",
        "--gap", "6", "--texture", str(texture_path), "--tex", "16", "--no-detect",
    ])
    assert demo.main() == 0
    assert len(worlds) == 1
    world = worlds[0]
    assert world.original_config["lead"]["gap"] == 6
    assert "patches" not in world.original_config
    assert world.prepares == 0 and world.resets == world.closes == 1
    patch, rgba = deployments[0]
    assert patch.emissive and rgba.shape == (16, 16, 4)
    np.testing.assert_array_equal(rgba[0, 0], [120, 10, 5, 255])
    panel.destroy.assert_called_once()
    with Image.open(output / "seq" / "frame_0000.png") as image:
        assert np.asarray(image)[..., :3].min() == 200
    assert (output / "seq" / "frame_0001.png").exists()
    assert (output / "patch.gif").exists() and (output / "patch_filmstrip.png").exists()


def test_camera_patch_demo_pairs_initial_scene_and_changes_model_input_before_driving(
    monkeypatch, scene_file, tmp_path,
):
    from scripts.demos.carla import patch_driving_demo as demo

    worlds, placements = [], []

    def make_world(config):
        world = CameraWorld(config, initial_color=90)
        worlds.append(world)
        return world

    def make_patch(world, texture, compositor):
        assert world.prepares == 1 and world.resets == 0
        assert texture.shape == (16, 16, 4) and compositor.harmonize is False
        placements.append(world)

        def perturb(observation):
            assert world.resets == 2
            return replace(observation, sensor_data={
                key: np.zeros_like(value) for key, value in observation.sensor_data.items()
            })

        return perturb

    def detect(image):
        return [((1, 1, 4, 2), 1.0, "car")] if image.any() else []

    plotted = {}

    def plot(clean, attacked, path, **kwargs):
        plotted.update(clean=clean, attacked=attacked, path=path)

    output = tmp_path / "driving"
    monkeypatch.setattr(demo, "OUT", output)
    class DepthCarla:
        def __init__(self, config, depth_camera=None):
            assert depth_camera == "front"
            self.config = config

        def create_runtime(self, seed=None):
            return make_world(self.config)

    monkeypatch.setattr(demo, "CarlaSimulator", DepthCarla)
    monkeypatch.setattr(demo, "carla_rear_perturbation", make_patch)
    monkeypatch.setattr(demo, "build_detector", lambda gpu: detect)
    monkeypatch.setattr(demo, "plot_impact", plot)
    monkeypatch.setattr(demo.sys, "argv", [
        "patch_driving_demo", "--config", str(scene_file), "--frames", "2",
        "--gap", "8", "--tex", "16", "--harmonizer", "none",
    ])
    assert demo.main() == 0
    assert len(worlds) == 1 and placements == worlds
    world = worlds[0]
    assert world.original_config["lead"]["gap"] == 8
    assert "patches" not in world.original_config
    assert world.prepares == 1 and world.resets == 2 and world.closes == 1
    assert [record.brake for record in plotted["clean"].records] == [1, 1]
    assert [record.throttle for record in plotted["attacked"].records] == [0.55, 0.55]
    assert plotted["clean"].provenance["treatment"] == "clean"
    assert plotted["attacked"].provenance["treatment"] == "attacked"
    with Image.open(output / "patched" / "frame_0000.png") as image:
        assert not np.asarray(image)[..., :3].any()
    for name in ("clean", "patched"):
        assert (output / name / "frame_0001.png").exists()
        assert (output / f"{name}.gif").exists()
        assert (output / f"{name}_filmstrip.png").exists()
