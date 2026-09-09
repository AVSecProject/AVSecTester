"""Optional real neural/simulator path. Never connects to CARLA without --run-carla."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest


@pytest.mark.carla
def test_neural_phantom_changes_closed_loop_drive(request, tmp_path):
    import yaml
    from avsectester.metric import impact
    from avsectester.scenario import run_scenario, set_perception_gpu

    config = yaml.safe_load(Path(request.config.getoption("--carla-config")).read_text())
    set_perception_gpu(config, request.config.getoption("--carla-gpu"))
    assert config["ego"]["pipeline"]["perception"]["type"] == "MMDetObjectDetector3D"
    assert config.get("attacks"), "The live test needs a configured attack"
    frames = config.get("frames", 40)
    clean = run_scenario(config, frames=frames)
    attacked = run_scenario(config, attacks=config["attacks"], frames=frames)
    result = impact(clean, attacked)
    # Preserve evidence even when a verdict assertion fails; pytest prints tmp_path on failure.
    (tmp_path / "carla-result.json").write_text(
        json.dumps(
            {
                "config": config,
                "clean": asdict(clean),
                "attacked": asdict(attacked),
                "impact": asdict(result),
            },
            indent=2,
        )
    )

    assert len(clean.records) == len(attacked.records) == frames
    assert clean.mean_detections > 0, "Baseline neural detector produced no detections"
    assert clean.peak_speed >= 1.0, f"Baseline did not establish driving: {result}"
    assert clean.final_speed > 0.5, f"Baseline ended stopped: {result}"
    assert attacked.final_speed <= 0.5, str(result)
    assert attacked.braking_frames > clean.braking_frames, str(result)
    assert result.attack_succeeded, str(result)
