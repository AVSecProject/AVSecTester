"""Optional real neural/simulator path. Never connects to CARLA without --run-carla."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest


@pytest.mark.carla
def test_clean_spawn_relocation_is_replayed_in_reset_world(request):
    """Exercise real spawning only; no driving steps or neural inference are needed."""
    import yaml
    from avsectester.scenario import prepare_scenario, run_scenario

    config = yaml.safe_load(Path(request.config.getoption("--carla-config")).read_text())
    config["client"]["strict_spawn"] = False
    config["ego"].update(vehicle="vehicle.tesla.model3", spawn=0)
    offset = {"type": "CarlaReferenceFrame", "location": [0, 0, 0.5]}
    config["ego"]["reference_to_spawn"] = offset
    config["ego"]["pipeline"]["perception"] = {"type": "Passthrough3DObjectDetector"}
    # The NPC first tries the ego's occupied position, forcing clean to relocate it.
    config["npcs"] = [
        {
            "type": "CarlaNpc",
            "npc_type": "vehicle.tesla.model3",
            "spawn": 0,
            "reference_to_spawn": offset,
        }
    ]
    prepared = prepare_scenario(config)
    clean = run_scenario(prepared, frames=0)
    replay = clean.replay_scenario
    ego_pose = replay["ego"]["spawn_transform"]
    npc_pose = replay["npcs"][0]["spawn_transform"]
    assert npc_pose["location"] != ego_pose["location"]
    attacked = run_scenario(replay, attacks=prepared.get("attacks", []), frames=0)
    assert attacked.replay_scenario["ego"]["spawn_transform"] == ego_pose
    assert attacked.replay_scenario["npcs"][0]["spawn_transform"] == npc_pose


@pytest.mark.carla
def test_neural_phantom_changes_closed_loop_drive(request, tmp_path, monkeypatch):
    import yaml
    from avcarla.actor import CarlaMobileActor
    from avsectester.metric import impact
    from avsectester.scenario import prepare_scenario, run_scenario, set_perception_gpu

    config = yaml.safe_load(Path(request.config.getoption("--carla-config")).read_text())
    set_perception_gpu(config, request.config.getoption("--carla-gpu"))
    assert config["ego"]["pipeline"]["perception"]["type"] == "MMDetObjectDetector3D"
    assert config.get("attacks"), "The live test needs a configured attack"
    frames = config.get("frames", 40)
    config = prepare_scenario(config)
    initial_states = []
    tick = CarlaMobileActor.tick
    observed_egos = set()

    def record_initial_state(ego, t0, frame0):
        if ego in observed_egos:
            return tick(ego, t0, frame0)
        observed_egos.add(ego)
        vehicles = []
        for vehicle in ego.world.get_actors().filter("vehicle.*"):
            pose = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            vehicles.append(
                [
                    vehicle.type_id,
                    pose.location.x,
                    pose.location.y,
                    pose.location.z,
                    pose.rotation.pitch,
                    pose.rotation.yaw,
                    pose.rotation.roll,
                    velocity.x,
                    velocity.y,
                    velocity.z,
                ]
            )
        initial_states.append(sorted(vehicles))
        return tick(ego, t0, frame0)

    monkeypatch.setattr(CarlaMobileActor, "tick", record_initial_state)
    clean = run_scenario(config, frames=frames)
    repeated_clean = run_scenario(clean.replay_scenario, frames=frames)
    attacked = run_scenario(clean.replay_scenario, attacks=config["attacks"], frames=frames)
    result = impact(clean, attacked)
    # Preserve evidence even when a verdict assertion fails; pytest prints tmp_path on failure.
    (tmp_path / "carla-result.json").write_text(
        json.dumps(
            {
                "config": config,
                "clean": asdict(clean),
                "repeated_clean": asdict(repeated_clean),
                "initial_states": initial_states,
                "attacked": asdict(attacked),
                "impact": asdict(result),
            },
            indent=2,
        )
    )

    assert initial_states[0] == initial_states[1] == initial_states[2]
    assert [r.speed for r in repeated_clean.records] == pytest.approx(
        [r.speed for r in clean.records], abs=0.05
    )
    assert repeated_clean.braking_frames == clean.braking_frames

    assert len(clean.records) == len(attacked.records) == frames
    assert clean.mean_detections > 0, "Baseline neural detector produced no detections"
    assert clean.peak_speed >= 1.0, f"Baseline did not establish driving: {result}"
    assert clean.final_speed > 0.5, f"Baseline ended stopped: {result}"
    assert attacked.final_speed <= 0.5, str(result)
    assert attacked.braking_frames > clean.braking_frames, str(result)
    assert result.attack_succeeded, str(result)
