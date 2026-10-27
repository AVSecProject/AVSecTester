"""Organize experiment results for evaluation, saving and visualization."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

from avsectester.evaluation.evaluator import RunGroup, evaluate_groups
from avsectester.evaluation.metric import DetectionMetrics, DrivingMetrics, Metric, TrajectoryDeviation
from avsectester.evaluation.report import EvaluationReport
from avsectester.evaluation.visualization import plot_group
from avsectester.data_types import Unavailable

from .configuration_summary import type_name
from .html_report import write_evaluation_html

if TYPE_CHECKING:
    from .config import ExperimentConfiguration
    from .models import ExperimentResult, TreatmentResult


def _default_metrics(result: ExperimentResult) -> tuple[tuple[Metric, ...], list[str]]:
    metrics: tuple[Metric, ...] = (DrivingMetrics(), TrajectoryDeviation())
    complete = [run.log for run in result.runs if run.log.status == "completed"]
    if complete and all(log.steps and all(step.detections is not None for step in log.steps) for log in complete):
        return metrics + (DetectionMetrics(),), []
    reason = ("no completed run provides recorded evidence" if not complete
              else "completed run logs do not all provide detections")
    return metrics, [f"DetectionMetrics omitted: {reason}."]


def evaluate(result: ExperimentResult, configuration: ExperimentConfiguration) -> EvaluationReport:
    """Group and order experiment logs, evaluate them, and save reports and plots.

    Default metrics are selected once for the whole experiment. Evaluation uses only saved
    logs and scenario maps, without loading a dataset or creating a simulator.
    """
    notes: list[str] = []
    if configuration.evaluation.metrics is None:
        metrics, notes = _default_metrics(result)
    else:
        metrics = configuration.evaluation.metrics
    grouped: dict[tuple[int, int], list[TreatmentResult]] = defaultdict(list)
    for run in result.runs:
        grouped[run.group].append(run)
    order = {item.name: index for index, item in enumerate(configuration.runs.treatments)}
    groups = []
    for (segment_index, seed), original in grouped.items():
        runs = sorted(original, key=lambda run: order[run.treatment])
        segment = runs[0].segment
        groups.append(RunGroup(
            segment_index, segment, seed,
            {run.treatment: run.log for run in runs},
            treatment_attacks={run.treatment: run.attack_name for run in runs}, part=runs[0].part,
            errors={run.treatment: run.error for run in runs if run.error is not None},
            scenario_map=result.scenario_maps.get(
                segment.scenario_id, Unavailable("No saved scenario map is available")),
        ))
    report = evaluate_groups(groups, metrics, notes=notes)
    report.selections = [item.as_dict() for item in result.selections]
    avstack_type = getattr(configuration.avstack.factory, "avstack_type", None)
    scenarios = {item.scenario_id for item in result.selections}
    if not scenarios:
        scenarios = {group.segment.scenario_id for group in groups}
    report.experiment = {
        "dataset_type": type_name(configuration.dataset),
        "scenarios_considered": len(scenarios),
        "simulator_type": type_name(configuration.simulator),
        "avstack_type": (f"{avstack_type.__module__}:{avstack_type.__qualname__}"
                         if avstack_type is not None else type_name(configuration.avstack)),
        "seeds": list(configuration.runs.seeds),
        "decision_steps": configuration.runs.decision_steps,
        "executed_parts": list(configuration.split.execute) if configuration.split else ["all (no split)"],
        "comparison_groups": len(groups),
        "treatments": [
            {"treatment": treatment.name, "attack": treatment.attack_name,
             "attack_type": type_name(configuration.attacks[treatment.attack_name])
             if treatment.attack_name is not None else None}
            for treatment in configuration.runs.treatments
        ],
    }
    if configuration.runs.output is not None:
        destination = Path(configuration.runs.output) / "evaluation"
        # Publish statuses before plotting so a plotting failure preserves useful results.
        report.save(destination)
        plots = destination / "plots"
        if plots.exists():
            shutil.rmtree(plots)
        plots.mkdir()
        for group in groups:
            try:
                plot_group({name: log for name, log in group.logs.items() if log.status == "completed"},
                           plots / f"segment-{group.segment_index:05d}-seed-{group.seed}.png",
                           title=f"Segment {group.segment_index}, seed {group.seed}")
            except Exception as error:
                report.record_error("plot_group", error, part=group.part,
                                    segment_index=group.segment_index, seed=group.seed)
        report.save(destination)
        write_evaluation_html(Path(configuration.runs.output))
    return report


__all__ = [
    'evaluate',
]
