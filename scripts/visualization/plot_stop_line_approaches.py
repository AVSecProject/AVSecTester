"""Plot saved physical speed against distance to the selected stop lines."""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from avsectester.evaluation import signed_stop_line_distances
from avsectester.maps import ScenarioMap

if TYPE_CHECKING:
    from avsectester.experiment import ExperimentResult


def plot_stop_line_approaches(
    result: ExperimentResult, output_directory: str | Path,
) -> tuple[Path, ...]:
    """Write one speed-versus-signed-distance figure per scene from saved evidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for run in result.runs:
        scenario_id = run.segment.scenario_id
        scenario_map = result.scenario_maps.get(scenario_id)
        if not isinstance(scenario_map, ScenarioMap):
            continue
        line_id = run.segment.user_data.get("stop_line_id")
        line = scenario_map.stop_lines.get(line_id)
        if line is None:
            continue
        states = (
            [run.log.steps[0].physical_pre]
            + [step.physical_post for step in run.log.steps]
            if run.log.steps else [event.data.get("physical_state")
                                  for event in run.log.events
                                  if event.kind == "initialization"]
        )
        if not states or any(state is None or state.ego is None for state in states):
            continue
        positions = np.stack([state.ego.pose[:3, 3] for state in states])
        speeds = np.asarray([state.ego_speed for state in states], dtype=float)
        if not np.isfinite(positions).all() or not np.isfinite(speeds).all():
            continue
        distances, _ = signed_stop_line_distances(positions, line.points)
        grouped[scenario_id].append((run, distances, speeds))
    colors = dict(zip(sorted({run.treatment for run in result.runs}),
                      cycle(plt.get_cmap("tab10").colors)))
    paths = []
    for scenario_id, traces in grouped.items():
        figure, axis = plt.subplots(figsize=(9, 5))
        for run, distances, speeds in traces:
            part = f", {run.part}" if run.part is not None else ""
            label = f"{run.treatment} (segment {run.segment_index}, seed {run.seed}{part})"
            axis.plot(distances, speeds, label=label,
                      color=colors[run.treatment], alpha=.75)
        axis.axvline(0., color="black", linestyle=":", label="stop line")
        axis.invert_xaxis()
        axis.set(xlabel="Signed distance before stop line (m)",
                 ylabel="Physical speed (m/s)", title=scenario_id)
        axis.legend(fontsize="small")
        figure.tight_layout()
        path = destination / f"{scenario_id}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)
    return tuple(paths)


def main() -> None:
    """Load CONFIG and plot its saved runs without model inference."""
    from avsectester.experiment import ExperimentResult, load_configuration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    configuration = load_configuration(arguments.configuration)
    source = Path(configuration.runs.output)
    result = ExperimentResult.load(source, configuration)
    destination = arguments.output or source / "stop_line_plots"
    paths = plot_stop_line_approaches(result, destination)
    print(f"Wrote {len(paths)} scene plots into {destination}")


if __name__ == "__main__":
    main()
