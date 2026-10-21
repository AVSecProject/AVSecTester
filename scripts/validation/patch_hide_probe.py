"""Compare patch size and harmonization on one CARLA capture using portable insertion actions."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from avsectester.attack.insertion import Insertion, PlaneAsset
from avsectester.attack.insertion.assets import load_rgba_image
from avsectester.attack.insertion.compositor import PatchCompositor
from avsectester.attack.insertion.action import InsertionAttackAction
from avsectester.attack.insertion.harmonizers import ClassicHarmonizer, PCTNetHarmonizer
from avsectester.attack.placement import AttachedPlacement, Orientation
from avsectester.closed_loop.avstacks.camera_forward_collision import CameraForwardCollisionAVStack
from avsectester.closed_loop.datatypes import Control
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.datasets.carla import CarlaDataset
from avsectester.evaluation.visualization import save_image
from avsectester.experiment import load_configuration
from scripts import REPO_ROOT


def probe_configuration(path: Path, gap: float, port: int, traffic_manager_port: int | None = None):
    """Create a dedicated simulator and recipe without mutating the loaded configuration."""
    configuration = load_configuration(path)
    scenario_id = next(configuration.dataset.scenarios()).id
    recipe = configuration.dataset.native(scenario_id)
    scenario = deepcopy(recipe.scenario)
    scenario["lead"]["gap"] = gap
    dataset = CarlaDataset({scenario_id: replace(recipe, scenario=scenario)})
    simulator = CarlaSimulator({**configuration.simulator.client_config, "connect_port": port,
                                "traffic_manager_port": port + 6000 if traffic_manager_port is None
                                else traffic_manager_port},
                               depth_camera="camera-0-0")
    return dataset, simulator, scenario_id


def patch_action(view, texture, *, width_fraction, height_fraction, harmonizer=None):
    host = view.actors["lead"]
    asset = PlaneAsset(texture, host.extent[1] * width_fraction, host.extent[2] * height_fraction)
    insertion = Insertion("probe", asset, AttachedPlacement("lead", "rear_center", (-.03, 0, 0)),
                          Orientation("follow_host", (0, 0, 180)))
    compositor = PatchCompositor(harmonizer, harmonize=harmonizer is not None)
    return InsertionAttackAction(view.sensor, insertion, compositor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/carla_patch.yaml")
    parser.add_argument("--gap", type=float, default=8.)
    parser.add_argument("--approach", type=int, default=16)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=2300)
    parser.add_argument("--traffic-manager-port", type=int,
                        help="Dedicated TM port; default is RPC port plus 6000")
    parser.add_argument("--output", type=Path, default=Path("tmp/patch-probe"))
    parser.add_argument("--pctnet", action="store_true", help="Also load PCTNet harmonization")
    args = parser.parse_args()
    dataset, simulator, scenario_id = probe_configuration(args.config, args.gap, args.port,
                                                         args.traffic_manager_port)
    frame = dataset.frames(scenario_id)[0]
    texture = load_rgba_image(args.texture)
    avstack = CameraForwardCollisionAVStack(gpu=args.gpu, camera="camera-0-0")
    try:
        with simulator.create_runtime() as simulator_runtime:
            simulator_runtime.reset(dataset.scenario_start(scenario_id, frame))
            for _ in range(args.approach):
                simulator_runtime.step(Control(throttle=.55))
            view = simulator_runtime.sensor_view("camera-0-0")
            save_image(view.data, args.output / "clean.png")
            variants = [("raw", None), ("classic", ClassicHarmonizer())]
            if args.pctnet:
                variants.append(("pctnet", PCTNetHarmonizer(device=args.gpu, strict=True)))
            clean_score = max((score for _, score, _ in avstack.detect(view.data)), default=0.)
            print(f"clean: {clean_score:.3f}")
            for name, harmonizer in variants:
                for size, width, height in (("standard", .85, .6), ("large", 1., .95)):
                    action = patch_action(view, texture, width_fraction=width,
                                          height_fraction=height, harmonizer=harmonizer)
                    image = action.apply(view.data, view)
                    score = max((score for _, score, _ in avstack.detect(image)), default=0.)
                    evidence = action.evidence(view)
                    print(f"{name}/{size}: confidence={score:.3f} visibility={evidence.visibility}")
                    save_image(image, args.output / f"{name}-{size}.png")
    finally:
        avstack.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
