"""HTML reports are rebuilt from saved data without datasets, simulators or models."""
from html import unescape
import json
import re

import numpy as np
from PIL import Image

from avsectester.attack import OfflineReport
from avsectester.evaluation import EvaluationReport, StopImpact
from avsectester.experiment.html_report import write_evaluation_html, write_offline_html


def test_html_reports(tmp_path):
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
    offline.save(tmp_path / "offline" / "attack")
    manifest = {
        "attacks": ["attack", "empty"],
        "configuration": {"attacks": {"attack": {
            "type": "package:Attack", "parameters": {"optimizer": {"step_size": .0156862745},
                                                     "transforms": [{"type": "package:Nested"}]},
        }}},
        "units": [
            {"index": 0, "attack_names": ["attack"], "part": None},
            {"index": 1, "attack_names": ["attack"], "part": "train"},
            {"index": 2, "attack_names": ["attack"], "part": "validation"},
            {"index": 3, "attack_names": ["attack"], "part": "test", "start": 5, "stop": 7},
        ],
    }
    (tmp_path / "preparation-progress.json").write_text(json.dumps(manifest))
    label = "example.metrics:ExampleMetric"
    speed, stopped = f"{label}.speed", f"{label}.stopped"
    missing = f"{label}.missing"
    impact = "StopImpact.a_to_b.reference_drove"
    report = EvaluationReport(groups=[{
        "part": "test", "segment_index": 3, "seed": 2, "scenario_id": "scene",
        "treatment_attacks": {"clean": None, "attack": "attack"},
        "treatments": {
            "clean": {"status": "completed", "error": None,
                      "metrics": {speed: 3.123456, stopped: False}, "reasons": {}},
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

    paths = [write_offline_html(tmp_path), write_evaluation_html(tmp_path)]
    assert paths == [tmp_path / "offline_report.html", tmp_path / "evaluation_report.html"]
    contents = [path.read_text() for path in paths]
    offline_text, evaluation_text = contents
    assert "Offline report —" in offline_text and "Evaluation report —" in evaluation_text
    assert "package:Attack" in offline_text and "no offline output" in offline_text
    assert "The offline phase failed; values below are partial." in offline_text
    assert "Mean confidence." in offline_text and "ratio (0–1)" in offline_text
    assert "Captured samples." in offline_text and "Detector confidence (ratio)." in offline_text
    assert "Replica face with the initial patch." in offline_text
    assert "0.1235" in offline_text and "0.7123" in offline_text and "0.0157" in offline_text
    assert "&lt;script&gt;" in offline_text and "<script>" not in offline_text
    assert "<h4>samples — 2 rows</h4>" in offline_text
    assert "<details><summary>large — 51 rows</summary>" in offline_text
    assert "<details><summary>wide — 1 row</summary>" in offline_text
    assert "not-inlined" in offline_text
    assert "Part: test" in evaluation_text and label in evaluation_text
    results = re.search(r'<section id="results">(.*?)</section>', evaluation_text, re.S).group(1)
    assert "<th>measurement</th><th>unit</th><th>clean</th><th>attack</th>" in results
    assert "3.1235<small>n = 1</small>" in results and "±" not in results
    assert "1 of 1 groups" in evaluation_text and "—" in evaluation_text
    differences = re.findall(r"<details( open)?><summary>Differences from ([^<]+)</summary>(.*?)</details>",
                             evaluation_text, re.S)
    assert differences[0][0] == " open" and differences[0][1] == "clean"
    assert "-2.2222<small>n = 1</small>" in differences[0][2]
    assert differences[1][1] == "attack"
    assert "2.2222<small>n = 1</small>" in differences[1][2] and "-2.2222" not in differences[1][2]
    assert 'id="def-StopImpact-a_to_b.reference_drove"' in evaluation_text
    assert '<summary><span title="avsectester.evaluation.metric:StopImpact">StopImpact</span></summary>' in evaluation_text
    assert 'href="#def-example.metrics:ExampleMetric-speed" title="Physical speed after a decision.">speed</a>' in results
    visible = unescape(re.sub(r"<[^>]*>", "", offline_text + evaluation_text))
    assert "package:Attack" not in visible and "package:Nested" not in visible
    assert "example:Dataset" not in visible and label not in visible
    assert '<span title="example:Dataset">Dataset</span>' in evaluation_text
    group = re.search(r'<section id="group-details">(.*?)</section>', evaluation_text, re.S).group(1)
    assert "<tbody><tr><td>status</td>" in group and 'class="metric-group"' in group
    assert '<sup><a href="#reason-3-2-' in group and 'class="footnotes"' in group
    assert "No description provided" in evaluation_text and "full segment" in evaluation_text
    assert "frames 5–7" in evaluation_text and "a saved error" in evaluation_text
    assert "no evidence" in evaluation_text and "&lt;sign&gt;" in evaluation_text
    assert all(f'id="{identifier}"' in evaluation_text for identifier in (
        "how-to-read", "experiment-setup", "run-status", "metric-definitions", "results",
        "paired-differences", "group-details", "selection", "errors-and-notes",
    ))
    for path, content in zip(paths, contents):
        for target in re.findall(r'(?:href|src)="([^"]+)"', content):
            if target.startswith("#"):
                assert f'id="{target[1:]}"' in content
            else:
                assert not target.startswith(("http:", "https:", "data:"))
                assert (path.parent / target).exists()
    for path in paths:
        path.unlink()

    def strip_time(text):
        return re.sub(r"<time.*?</time>", "<time></time>", text)

    assert strip_time(write_offline_html(tmp_path).read_text()) == strip_time(offline_text)
    assert strip_time(write_evaluation_html(tmp_path).read_text()) == strip_time(evaluation_text)

    manifest["metadata_source"] = {"load": "/data/selected<signs>.json", "sha256": "saved-hash"}
    (tmp_path / "prepared.json").write_text(json.dumps(manifest))
    loaded_text = write_evaluation_html(tmp_path).read_text()
    assert "Metadata loaded from /data/selected&lt;signs&gt;.json (4 segments); no selection was run." in loaded_text
    selection = re.search(r'<section id="selection">(.*?)</section>', loaded_text, re.S).group(1)
    assert "<table>" not in selection
