"""Opt-in CARLA selection, replay and sensor insertion without a neural checkpoint."""

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.carla
def test_selected_case_replays_and_inserts_into_same_frame_observation(request, tmp_path):
    import yaml
    from PIL import Image

    from avsectester.insertion import AttachedPlacement, Insertion, Orientation, PlaneAsset
    from avsectester.plane import Control
    from avsectester.scenarios import CarlaScenarioBuilder, RoleSpec, ScenarioRequirement
    from avsectester.scenarios.carla_provider import CarlaCandidateProvider, CarlaSelectionBackend
    from avsectester.scenarios.requirement import InitialWindow, MinVisibility
    from avsectester.simulators.carla import camera_view, insertion_perturbation

    config = yaml.safe_load(Path(request.config.getoption("--carla-config")).read_text())
    config.pop("patches", None)
    config["ego"].update(vehicle="vehicle.tesla.model3", autopilot=False, destination=None)
    config["ego"]["sensors"] = [
        {
            "type": "CarlaRgbCamera",
            "name": "front",
            "image_size_x": 320,
            "image_size_y": 240,
            "fov": 90,
            "sensor_tick": 0,
        }
    ]
    config["npcs"] = []
    config["lead"] = {"gap": 9.0, "vehicle": "vehicle.tesla.model3"}
    texture = np.full((32, 32, 4), [255, 0, 255, 255], np.uint8)
    insertion = Insertion(
        "patch",
        PlaneAsset(texture, 0.8, 0.5),
        AttachedPlacement("host", "rear_center", (-0.05, 0, 0.2)),
        Orientation("follow_host", (0, 0, 180)),
    )
    previews = []

    def backend_factory(scenario):
        backend = CarlaSelectionBackend(scenario)
        previews.append(backend)
        return backend

    provider = CarlaCandidateProvider(
        initial_frames=2,
        advance_initial=lambda backend, index: backend.step(Control(brake=1)),
        backend_factory=backend_factory,
    )
    source = CarlaScenarioBuilder(config, candidate_provider=provider)
    requirement = ScenarioRequirement(
        "rear-patch",
        roles={"host": RoleSpec(ids=("lead",))},
        insertions=(insertion,),
        constraints=[MinVisibility(0.5)],
        camera="front",
        window=InitialWindow(2),
    )
    cases = list(source.scenarios(requirement, limit=1))
    assert len(cases) == 1, source.selection_log
    assert previews[0].client is None  # Preview released before returning the runnable case.
    case = cases[0]
    assert case.target.binding_ids == {"host": ("lead",)}
    selected_host = next(obj for obj in case.target.scene.objects if obj.track_id == "lead")
    backend = case.make_backend()
    try:
        # Both resets must reconstruct the selected origin, including the same setup ticks.
        for replay_index in range(2):
            observation = backend.reset()
            context = backend.selection_context()
            np.testing.assert_allclose(
                context.scene.ego.pose, case.target.scene.ego.pose, atol=1e-4
            )
            host = next(obj for obj in context.scene.objects if obj.track_id == "lead")
            np.testing.assert_allclose(host.pose, selected_host.pose, atol=1e-4)
            perturb = insertion_perturbation(
                backend,
                case.target.insertions,
                bindings=case.target.binding_ids,
            )
            for step in range(3):
                assert backend.depth_frame == observation.frame
                original = camera_view(observation).copy()
                patched = perturb(observation)
                rendered = camera_view(patched)
                assert patched.frame == observation.frame
                assert np.count_nonzero(np.any(rendered != original, axis=2)) > 10
                np.testing.assert_array_equal(camera_view(observation), original)
                if step == 0:
                    Image.fromarray(original).save(tmp_path / f"clean-{replay_index}.png")
                    Image.fromarray(rendered).save(tmp_path / f"inserted-{replay_index}.png")
                observation = backend.step(Control(brake=1))
    finally:
        backend.close()
    assert backend.client is None
