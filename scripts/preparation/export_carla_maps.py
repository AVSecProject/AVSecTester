"""Export OpenDRIVE files using one CARLA client connection."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", nargs="+", required=True, help="Towns to load and export")
    parser.add_argument("--output", type=Path, required=True, help="Directory for <town>.xodr")
    parser.add_argument("--timeout", type=float, default=30.)
    arguments = parser.parse_args()

    import carla

    client = carla.Client(arguments.host, arguments.port)
    client.set_timeout(arguments.timeout)
    arguments.output.mkdir(parents=True, exist_ok=True)
    world = client.get_world()
    for town in arguments.town:
        name = town.rsplit("/", 1)[-1]
        if not name or name in {".", ".."}:
            parser.error("Each town must identify a named CARLA map")
        if world.get_map().name.rsplit("/", 1)[-1] != name:
            world = client.load_world(town)
        destination = arguments.output / f"{name}.xodr"
        destination.write_text(world.get_map().to_opendrive(), encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    main()
