"""HTML renderers consume saved reports and public prepared experiment records."""
from dataclasses import replace
from html import unescape
import re

import numpy as np
from PIL import Image

from avsectester.attack import AttackSegment, OfflineReport
from avsectester.closed_loop.avstacks import DrivingAdapter
from avsectester.evaluation import EvaluationReport, StopImpact
from avsectester.experiment import Experiment, ExperimentConfiguration, MetadataConfiguration, RunsConfiguration, SplitConfiguration, save_metadata
from avsectester.experiment.html_report import write_evaluation_html, write_offline_html
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator, MinimalAttack


def saved_preparation(directory, *, load=False):
    attack = MinimalAttack(max_segments=4, assets={
        "optimizer": {"step_size": .0156862745}, "transforms": [{"type": "example:Nested"}],
    })
    configuration = ExperimentConfiguration(
        MemoryDataset(decision_steps=10), MemorySimulator(),
        DrivingAdapter(lambda seed: ConstantAVStack()),
        {"attack": attack, "empty": MinimalAttack(max_segments=4)},
        split=SplitConfiguration(1, validation_segments=1),
        runs=RunsConfiguration(seeds=(2,), output=directory),
    )
    frames = configuration.dataset.frames("scene")
    segments = tuple(AttackSegment("scene", frames[i], frames[i+2]) for i in (0, 2, 4, 6))
    source = directory / "selected<signs>.json"
    save_metadata(segments, source, dataset_type="tests.fakes:MemoryDataset")
    configuration = replace(configuration, metadata=MetadataConfiguration(load=source)) if load else configuration
    return Experiment.prepare(configuration)


def check_links(path):
    text = path.read_text()
    for target in re.findall(r'(?:href|src)="([^"]+)"', text):
        if target.startswith("#"):
            assert f'id="{target[1:]}"' in text
        else:
            assert not target.startswith(("http:", "https:", "data:"))
            assert (path.parent / target).exists()


def without_time(text):
    return re.sub(r"<time.*?</time>", "<time></time>", text)


def test_write_offline_html(tmp_path):
    saved_preparation(tmp_path)
    offline = OfflineReport(None)
    offline.status = "failed"
    offline.set_value("confidence", .12345678, unit="ratio (0–1)", description="Mean confidence.")
    offline.set_value("description", "<script>not executable</script>")
    offline.set_series("objective", [.7654321, .2345678], y_unit="ratio (0–1)",
                       description="Confidence before each update.")
    offline.set_image("texture", np.full((3, 4, 4), 255, dtype=np.uint8),
                      description="Replica face with the initial patch.")
    offline.set_table("samples", [{"frame": 1, "score": .7123456}, {"frame": 2, "score": .6123456}],
                      description="Captured samples.",
                      columns={"frame": "Recorded frame index.", "score": "Detector confidence (ratio)."})
    offline.set_table("wide", [{f"column_{i}": i for i in range(9)}])
    offline.set_table("large", [{"row": index, "marker": "not-inlined"} for index in range(51)])
    offline.save(tmp_path / "attacks" / "attack" / "offline_report")
    EvaluationReport().save(tmp_path / "evaluation")
    write_evaluation_html(tmp_path)
    path = write_offline_html(tmp_path)
    assert path == tmp_path.resolve() / "offline_report.html"
    text = path.read_text()
    assert "Offline report —" in text and "no offline output" in text
    assert "The offline phase failed. Values below are partial." in text
    assert "Mean confidence." in text and "ratio (0–1)" in text
    assert "Captured samples." in text and "Detector confidence (ratio)." in text
    assert "Replica face with the initial patch." in text
    visible = unescape(re.sub(r"<[^>]*>", "", text))
    assert "example:Nested" not in visible
    assert "0.1235" in text and "0.7123" in text and "0.0157" in text
    assert "&lt;script&gt;" in text and "<script>" not in text
    assert "<h4>samples — 2 rows</h4>" in text
    assert "<details><summary>large — 51 rows</summary>" in text
    assert "<details><summary>wide — 1 row</summary>" in text
    assert "not-inlined" in text
    assert '<th>type</th>' in text.lower()
    assert re.search(r'title="tests.fakes:MinimalAttack"[^>]*>MinimalAttack<', text)
    check_links(path)
    path.unlink()
    assert without_time(write_offline_html(tmp_path).read_text()) == without_time(text)


def test_write_evaluation_html(tmp_path):
    saved_preparation(tmp_path)
    label = "example.metrics:ExampleMetric"
    speed, stopped = f"{label}.speed", f"{label}.stopped"
    missing = f"{label}.missing"
    integer, zero = (f"{label}.{name}" for name in ("count", "zero"))
    impact = "StopImpact.a_to_b.reference_drove"
    report = EvaluationReport(groups=[{
        "part": "test", "segment_index": 3, "seed": 2, "scenario_id": "scene",
        "treatment_attacks": {"clean": None, "attack": "attack"},
        "treatments": {
            "clean": {"status": "completed", "error": None,
                      "metrics": {speed: 3.123456, stopped: False,
                                  integer: 7, zero: -1e-6}, "reasons": {}},
            "attack": {"status": "completed", "error": None,
                       "metrics": {speed: .901234, stopped: True, missing: None},
                       "metric_types": {missing: "number"},
                       "reasons": {missing: "no evidence"}},
        },
        "comparisons": [{"treatment_a": "clean", "treatment_b": "attack", "status": "completed",
                         "deltas": {speed: -2.222222, stopped: 1.}, "metrics": {impact: True}}],
    }], notes=["A saved note"], selections=[
        {"attack_name": "attack", "scenario_id": "scene", "status": "selected", "reasons": ["<sign>"]}],
        metrics=[{
            "type": "example.metrics:ExampleMetric", "label": label, "kind": "run",
            "summary": "Example physical measurements.", "options": {"threshold": .1234567},
            "fields": {
                "speed": {"description": "Physical speed after a decision.", "unit": "m/s"},
                "stopped": {"description": "The ego stopped.", "unit": None},
                "count": {"description": "Whole-number count.", "unit": None},
                "zero": {"description": "Signed zero.", "unit": None},
            },
        }, {
            "type": "avsectester.evaluation.metric:StopImpact", "label": "StopImpact", "kind": "pair",
            "summary": StopImpact().describe().summary,
            "fields": {name: {"description": field.description, "unit": field.unit}
                       for name, field in StopImpact().describe().fields.items()},
        }], experiment={
            "dataset_type": "example:Dataset", "scenarios_considered": 1,
            "simulator_type": "example:Simulator", "avstack_type": "example:AVStack",
            "seeds": [2], "decision_steps": None, "executed_parts": ["test"],
            "comparison_groups": 1,
            "treatments": [{"treatment": "clean", "attack": None},
                           {"treatment": "attack", "attack": "attack"}],
        })
    report.record_error("Example", RuntimeError("a saved error"))
    report.summarize()
    evaluation = tmp_path / "evaluation"
    report.save(evaluation)
    plots = evaluation / "plots"
    plots.mkdir()
    Image.fromarray(np.zeros((3, 4, 3), np.uint8)).save(plots / "segment-00003-seed-2.png")

    path = write_evaluation_html(tmp_path)
    assert path == tmp_path.resolve() / "evaluation_report.html"
    text = path.read_text()
    assert "Evaluation report —" in text and "Part: test" in text and label in text
    results = re.search(r'<section id="results">(.*?)</section>', text, re.S).group(1)
    assert "<th>measurement</th><th>unit</th><th>clean</th><th>attack</th>" in results
    assert "3.1235<small>n = 1</small>" in results and "±" not in results
    assert "1 of 1 groups" in text and "—" in text
    differences = re.findall(
        r"<details( open)?><summary>Differences from ([^<]+)</summary>(.*?)</details>", text, re.S)
    assert differences[0][0] == " open" and differences[0][1] == "clean"
    assert "-2.2222<small>n = 1</small>" in differences[0][2]
    assert differences[1][1] == "attack"
    assert "-2.2222" not in differences[1][2]
    assert "2.2222<small>n = 1</small>" in differences[1][2]
    assert 'id="def-StopImpact-a_to_b.reference_drove"' in text
    assert '<summary><span title="avsectester.evaluation.metric:StopImpact">StopImpact</span></summary>' in text
    assert "a saved error" in text
    details = re.search(r'<section id="group-details">(.*?)</section>', text, re.S).group(1)
    assert "<tbody><tr><td>status</td>" in details
    speed_link = ('<a href="#def-example.metrics:ExampleMetric-speed" '
                  'title="Physical speed after a decision.">speed</a>')
    assert speed_link in results
    assert re.search(r">stopped</a></td><td>—</td><td>no</td><td>yes</td>", details)
    assert re.search(r">count</a></td><td>—</td><td>7</td>", details)
    assert re.search(r">zero</a></td><td>—</td><td>0.0000</td>", details)
    assert "-0.0000" not in details
    assert "Physical speed after a decision." in text and "m/s" in text
    assert "frames 6–8" in text and "no evidence" in text and 'class="footnotes"' in text
    assert all(heading in text for heading in (
        "Experiment setup", "Run status", "Metric definitions", "Selection", "Errors and notes"))
    assert all(f'id="{identifier}"' in text for identifier in (
        "experiment-setup", "run-status", "metric-definitions", "results",
        "paired-differences", "group-details", "selection", "errors-and-notes"))
    visible = unescape(re.sub(r"<[^>]*>", "", text))
    assert "example:Dataset" not in visible and label not in visible
    assert '<span title="example:Dataset">Dataset</span>' in text
    assert 'class="metric-group"' in text and '<sup><a href="#reason-3-2-' in text
    assert "No description provided" in text and "full segment" in text
    check_links(path)
    path.unlink()
    assert without_time(write_evaluation_html(tmp_path).read_text()) == without_time(text)

    selection = re.search(r'<section id="selection">(.*?)</section>', text, re.S).group(1)
    assert "<th>attack</th>" in selection and "<th>status</th>" in selection
    assert "<tr><td>attack</td><td>1</td><td>0</td><td>0</td></tr>" in selection
    assert "<td>selected</td>" in selection
    assert "&lt;sign&gt;" in selection and "<sign>" not in selection

    loaded = tmp_path / "loaded"
    saved_preparation(loaded, load=True)
    EvaluationReport().save(loaded / "evaluation")
    loaded_text = write_evaluation_html(loaded).read_text()
    assert (f"Metadata loaded from {loaded.resolve()}/selected&lt;signs&gt;.json "
            "(4 segments). No selection was run.") in loaded_text
