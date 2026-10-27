"""Build human-readable HTML reports solely from saved experiment files."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any

from avsectester import __version__
from avsectester.evaluation.report import format_number, format_statistic


_STYLE = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{font:15px/1.5 system-ui,sans-serif;color:#243247;background:#f4f6fa;margin:0}
main{max-width:1600px;margin:auto;padding:32px 28px 64px}
h1,h2,h3,h4{line-height:1.25;color:#12243c}h1{font-size:2rem;margin-bottom:8px}
h2{font-size:1.5rem}h3{font-size:1.1rem}p{max-width:100ch}
section,details{background:white;border:1px solid #d9e1ed;border-radius:10px;padding:20px;margin:22px 0}
section section,section details{background:#fbfcfe;margin:16px 0}
nav{background:#eaf0f8;padding:18px 22px;border-radius:10px;margin-top:24px}
nav ul{display:flex;flex-wrap:wrap;gap:8px 26px;padding-left:18px;margin:0}
.table-scroll{max-width:100%;overflow-x:auto}table{border-collapse:collapse;font-size:14px;margin:14px 0;width:100%}
th,td{border:1px solid #d9e1ed;padding:6px 10px;text-align:left;vertical-align:top}
td{overflow-wrap:break-word}td:first-child{white-space:nowrap}.offline-data td:first-child{white-space:normal;min-width:200px;max-width:240px}th{overflow-wrap:break-word;background:#edf2f9;color:#12243c;font-weight:650}tbody tr:nth-child(even){background:#f8faff}
img{display:block;max-width:100%;height:auto}a{color:#175a9a;text-underline-offset:3px}
figure{margin:14px 0}figcaption{color:#536175;margin-top:8px;max-width:100ch}
.image-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,320px));gap:24px;align-items:start}
.image-grid img{max-width:320px;max-height:320px;object-fit:contain;object-position:left}
.wide-data table{width:max-content;min-width:100%}.wide-data td,.wide-data th{white-space:nowrap}.wide-data td:first-child{max-width:none}
#selection td:nth-child(2),#selection th:nth-child(2){white-space:nowrap}
.metric-group th{background:#dfe8f5;text-align:left}.series-image img{max-width:640px}
.footnotes{font-size:13px}.footnotes li{margin:5px 0}sup{margin-left:3px}
summary{cursor:pointer;font-weight:650}details[open]>summary{margin-bottom:16px}
small,.muted{color:#617087}td small{display:block;white-space:nowrap;font-size:12px;margin-top:3px}
.notice{padding:12px 16px;border-left:4px solid #bd3d32;background:#fff0ec;color:#862b24;font-weight:650}
.none{color:#617087}code{font-family:ui-monospace,monospace;overflow-wrap:anywhere}
.header-links{display:flex;gap:10px 22px;flex-wrap:wrap}
caption{text-align:left;font-weight:650;margin-bottom:8px}
@media(max-width:700px){main{padding:20px 12px}section,details{padding:14px}h1{font-size:1.55rem}}
"""


class _Html(str):
    """Markup assembled here from already escaped values."""


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{name}: {_text(item)}" for name, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_text(item) for item in value) + "]"
    return format_number(value)


def _cell(value: Any) -> str:
    return str(value) if isinstance(value, _Html) else escape(_text(value))


def _table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return '<p class="none">None</p>'
    head = "".join(f"<th>{_cell(header)}</th>" for header in headers)
    body = "".join(str(row) if isinstance(row, _Html) else
                   "<tr>" + "".join(f"<td>{_cell(value)}</td>" for value in row) + "</tr>"
                   for row in rows)
    return (f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def _link(path: str, label: str) -> _Html:
    return _Html(f'<a href="{escape(path, quote=True)}">{escape(label)}</a>')


def _image(path: str, description: str) -> str:
    return (f'<figure><img src="{escape(path, quote=True)}" alt="{escape(description, quote=True)}">'
            f"<figcaption>{escape(description)}</figcaption></figure>")


def _document(title: str, body: Sequence[str], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><main><h1>{escape(title)}</h1>{''.join(body)}</main></body></html>\n",
        encoding="utf-8")
    return destination


def _header(links: Sequence[tuple[str, str]]) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return (f'<p class="muted">Generated <time datetime="{now}">{now}</time> UTC · '
            f"avsectester {escape(__version__)}</p>"
            '<p class="header-links">' + " ".join(_link(path, label) for path, label in links) + "</p>")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _manifest(root: Path) -> dict[str, Any]:
    prepared = root / "prepared.json"
    return _load(prepared if prepared.is_file() else root / "preparation-progress.json")


def _counts(manifest: Mapping[str, Any], name: str) -> tuple[int, int]:
    units = [unit for unit in manifest.get("units", []) if name in unit.get("attack_names", [])]
    return (sum(unit.get("part") in (None, "train") for unit in units),
            sum(unit.get("part") == "validation" for unit in units))


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping) and value:
        return [row for name, item in value.items()
                for row in _flatten(item, f"{prefix}.{name}" if prefix else str(name))]
    if isinstance(value, (list, tuple)) and value:
        return [row for index, item in enumerate(value)
                for row in _flatten(item, f"{prefix}.{index}" if prefix else str(index))]
    return [(prefix or "parameters", value)]


def _type_name(value: Any) -> _Html:
    """Display a class name while retaining its exact configured path as a tooltip."""
    if value is None:
        return _Html("—")
    path = str(value)
    name = path.rsplit(":", 1)[-1].rsplit(".", 1)[-1]
    return _Html(f'<span title="{escape(path, quote=True)}">{escape(name)}</span>')


def _configuration_rows(parameters: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return [(name, _type_name(value) if name == "type" or name.endswith(".type") else value)
            for name, value in _flatten(parameters)]


def _label_with_unit(label: str, unit: str | None) -> str:
    return f"{label} ({unit})" if unit else label


def _offline_series(directory: Path, name: str, series: Mapping[str, Any]) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(6.4, 3.5), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(series["x"], series["y"], marker=".")
    axis.set_xlabel(_label_with_unit(series.get("x_label", "iteration"), series.get("x_unit")))
    axis.set_ylabel(_label_with_unit(series.get("y_label") or name, series.get("y_unit")))
    axis.set_title(series.get("y_label") or name)
    if series["x"] and all(float(value).is_integer() for value in series["x"]):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    figure.tight_layout()
    canvas.print_png(directory / f"{name}.png")


def _offline_table(name: str, table: Mapping[str, Any], attack: str, *, folded: bool) -> str:
    rows = table.get("rows", [])
    columns = list(dict.fromkeys(key for row in rows for key in row))
    descriptions = table.get("columns") or {}
    heading = f"{name} — {len(rows)} {'row' if len(rows) == 1 else 'rows'}"
    result = (f"<details><summary>{escape(heading)}</summary>" if folded else
              f"<h4>{escape(heading)}</h4>")
    result += f"<p>{escape(table.get('description') or 'No description provided')}</p>"
    result += _table(["column", "meaning (including units)"], [
        [column, descriptions.get(column, "No description provided")] for column in columns])
    result += "<p>" + _link(f"offline/{attack}/{table['file']}", "CSV") + "</p>"
    data_class = "offline-data wide-data" if folded else "offline-data"
    result += f'<div class="{data_class}">' + _table(
        columns, [[row.get(column) for column in columns] for row in rows]) + "</div>"
    return result + ("</details>" if folded else "")

def write_offline_html(output_root: str | Path) -> Path:
    """Rebuild the offline page and series curves solely from saved entries."""
    root = Path(output_root)
    manifest = _manifest(root)
    offline = root / "offline"
    names = list(dict.fromkeys([
        *manifest.get("attacks", []),
        *[path.parent.name for path in sorted(offline.glob("*/report.json"))],
    ]))
    reports = {name: _load(offline / name / "report.json") for name in names}
    specifications = manifest.get("configuration", {}).get("attacks", {})
    body = [_header([("evaluation_report.html", "Evaluation report")])]
    body.append('<section id="about"><h2>About this page</h2>'
                "<p>The offline phase prepares each attack's artifact before execution, using "
                "its training segments. For example, an attack may optimize a patch; an attack "
                "without optimization records only its initial texture.</p>"
                "<p>Training segments are available to offline preparation. Validation segments "
                "are held aside for optional validation, rather than training updates. When no "
                "split is configured, all selected segments count as training segments here.</p>"
                "<p>Decimal values are shown with four decimal places; whole-number counts are "
                "shown as integers. JSON and CSV keep full precision.</p></section>")
    body.append('<section id="overview"><h2>Overview</h2>')
    body.append(_table(
        ["attack", "type", "status", "train segments", "validation segments", "details"],
        [[name, _type_name(specifications.get(name, {}).get("type")),
          reports[name].get("status", "pending"), *_counts(manifest, name),
          _link(f"#attack-{index}", "View")] for index, name in enumerate(names)]))
    body.append("</section>")
    for index, name in enumerate(names):
        directory, report = offline / name, reports[name]
        specification = specifications.get(name, {})
        body.append(f'<section id="attack-{index}"><h2>{escape(name)}</h2>')
        body.append(f'<p>Status: <strong>{escape(report.get("status", "pending"))}</strong></p>')
        if report.get("status") == "failed":
            body.append('<p class="notice">The offline phase failed; values below are partial.</p>')
        if (directory / "report.json").is_file():
            body.append("<p>" + _link(f"offline/{name}/report.json", "Report JSON") + "</p>")
        if not any(report.get(key) for key in ("values", "series", "tables", "images")):
            body.append("<p>no offline output</p>")
        body.append("<h3>Values</h3>")
        body.append(_table(["name", "value", "unit", "description"], [
            [key, item["value"], item.get("unit"), item.get("description")]
            for key, item in report.get("values", {}).items()]))
        small: list[tuple[str, dict[str, Any]]] = []
        large: list[tuple[str, dict[str, Any]]] = []
        for key, table in report.get("tables", {}).items():
            columns = {column for row in table.get("rows", []) for column in row}
            (small if len(table.get("rows", [])) <= 20 and len(columns) <= 8 else large).append((key, table))
        body.append("<h3>Summary tables</h3>")
        body.extend(_offline_table(key, table, name, folded=False) for key, table in small)
        if not small:
            body.append('<p class="none">None</p>')
        body.append("<h3>Images</h3>")
        images = report.get("images", {})
        if images:
            body.append('<div class="image-grid">')
            for key, image in images.items():
                body.append(_image(f"offline/{name}/{image['file']}", image.get("description") or key))
            body.append("</div>")
        else:
            body.append('<p class="none">None</p>')
        body.append("<h3>Series</h3>")
        series = report.get("series", {})
        if not series:
            body.append('<p class="none">None</p>')
        for key, item in series.items():
            _offline_series(directory, key, item)
            body.append('<div class="series-image">' +
                        _image(f"offline/{name}/{key}.png", item.get("description") or key) + "</div>")
        body.append("<h3>Detail tables</h3>")
        body.extend(_offline_table(key, table, name, folded=True) for key, table in large)
        if not large:
            body.append('<p class="none">None</p>')
        body.append("<details><summary>Configuration</summary>"
                    "<p>Only options written in the configuration are listed; other parameters "
                    "use their defaults.</p>" +
                    _table(["parameter", "value"], _configuration_rows(specification.get("parameters", {}))) +
                    "</details>")
        body.append("<h3>Files</h3>")
        files = directory / "files"
        body.append("<p>" + _link(f"offline/{name}/files/", "Additional offline files") + "</p>"
                    if files.is_dir() and any(files.iterdir()) else '<p class="none">None</p>')
        body.append("</section>")
    return _document(f"Offline report — {root.name}", body, root / "offline_report.html")

def _metric_specs(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    specifications = [dict(spec) for spec in report.get("metrics", [])]
    for specification in specifications:
        specification.setdefault("label", specification.get("type", "Metric").split(":")[-1])
        specification.setdefault("kind", "run")
        specification.setdefault("fields", {})
    return specifications


def _observed_fields(report: Mapping[str, Any], kind: str) -> list[str]:
    key = "deltas" if kind == "delta" else "metrics"
    summaries = report.get("pair_summaries" if kind != "run" else "treatment_summaries", [])
    return list(dict.fromkeys(field for row in summaries for field in (row.get(key) or {})))


def _sections(report: Mapping[str, Any], kind: str) -> list[tuple[dict[str, Any], list[str]]]:
    observed = _observed_fields(report, kind)
    specifications = _metric_specs(report)
    labels = sorted([spec["label"] for spec in specifications], key=len, reverse=True)
    matched = {field: next((label for label in labels if field.startswith(label + ".")), None)
               for field in observed}
    result = []
    for specification in specifications:
        if (kind == "pair") != (specification.get("kind") == "pair"):
            continue
        label = specification["label"]
        fields = list(dict.fromkeys([
            *[f"{label}.{name}" for name in specification.get("fields", {})],
            *[field for field in observed if matched[field] == label],
        ]))
        if fields:
            result.append((specification, fields))
    unknown = [field for field in observed if matched[field] is None]
    if unknown:
        result.append(({"label": "Unspecified metrics", "fields": {}, "kind": kind}, unknown))
    return result


def _field_name(specification: Mapping[str, Any], name: str) -> str:
    prefix = specification["label"] + "."
    return name[len(prefix):] if name.startswith(prefix) else name


def _field_description(specification: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return specification.get("fields", {}).get(_field_name(specification, name), {})


def _definition_id(specification: Mapping[str, Any], name: str) -> str:
    return f"def-{specification['label']}-{_field_name(specification, name)}".replace(" ", "-")


def _field_link(specification: Mapping[str, Any], name: str) -> _Html:
    description = _field_description(specification, name).get("description") or "No description provided"
    return _Html(f'<a href="#{escape(_definition_id(specification, name), quote=True)}" '
                 f'title="{escape(description, quote=True)}">{escape(_field_name(specification, name))}</a>')

def _field_type(report: Mapping[str, Any], name: str) -> str:
    for group in report.get("groups", []):
        rows = [*group.get("treatments", {}).values(), *group.get("comparisons", [])]
        for row in rows:
            field_type = row.get("metric_types", {}).get(name)
            if field_type == "boolean" or isinstance((row.get("metrics") or {}).get(name), bool):
                return "yes-no"
            if field_type == "number":
                return "number"
    for row in [*report.get("treatment_summaries", []), *report.get("pair_summaries", [])]:
        if (row.get("metrics") or {}).get(name, {}).get("kind") == "proportion":
            return "yes-no"
    return "number"


def _statistic(statistic: Mapping[str, Any]) -> _Html:
    lines = format_statistic(statistic).splitlines()
    return _Html(escape(lines[0]) + (f"<small>{escape(lines[1])}</small>" if len(lines) > 1 else ""))


def _part_title(part: str | None) -> str:
    return "All parts (no split)" if part is None else f"Part: {part}"


def _section(identifier: str, title: str, content: Sequence[str]) -> str:
    return (f'<section id="{identifier}"><h2>{escape(title)}</h2>' +
            ("".join(content) if content else '<p class="none">None</p>') + "</section>")


def _experiment_setup(report: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    experiment = report.get("experiment", {})
    if not experiment:
        return []
    decision_steps = experiment.get("decision_steps")
    content = [_table(["setting", "value"], [
        ["dataset type", _type_name(experiment.get("dataset_type"))],
        ["scenarios considered", experiment.get("scenarios_considered")],
        ["simulator type", _type_name(experiment.get("simulator_type"))],
        ["AVStack type", _type_name(experiment.get("avstack_type"))],
        ["seeds", ", ".join(str(seed) for seed in experiment.get("seeds", []))],
        ["decision steps", "full segment" if decision_steps is None else decision_steps],
        ["executed parts", ", ".join(experiment.get("executed_parts", []))],
        ["comparison groups", experiment.get("comparison_groups")],
    ])]
    specifications = manifest.get("configuration", {}).get("attacks", {})
    content.append(_table(["treatment", "attack", "attack type"], [
        [row["treatment"], row.get("attack") or "none",
         _type_name(specifications.get(row.get("attack"), {}).get("type", row.get("attack_type")))
         if row.get("attack") is not None else "none"]
        for row in experiment.get("treatments", [])]))
    return content


def _run_status(report: Mapping[str, Any]) -> list[str]:
    groups = report.get("groups", [])
    parts = list(dict.fromkeys(group.get("part") for group in groups))
    result = []
    for part in parts:
        result.append(f"<h3>{escape(_part_title(part))}</h3>")
        statuses: dict[str, Counter[str]] = {}
        failed = []
        for group in groups:
            if group.get("part") != part:
                continue
            for treatment, row in group.get("treatments", {}).items():
                statuses.setdefault(treatment, Counter())[row["status"]] += 1
                if row["status"] != "completed":
                    failed.append([treatment, group["segment_index"], group["seed"], row.get("error")])
        result.append(_table(["treatment", "completed", "failed", "total"], [
            [treatment, counts["completed"], sum(counts.values()) - counts["completed"], sum(counts.values())]
            for treatment, counts in statuses.items()]))
        if failed:
            result.append("<h4>Failed or incomplete runs</h4>" +
                          _table(["treatment", "segment", "seed", "error"], failed))
    return result


def _definitions(report: Mapping[str, Any]) -> list[str]:
    result = []
    for specification, fields in _sections(report, "run") + _sections(report, "pair"):
        result.append(f"<h3>{_type_name(specification.get('type') or specification['label'])}</h3><p>"
                      f"{escape(specification.get('summary') or 'No description provided')}</p>")
        result.append("<h4>Parameters</h4>" + _table(
            ["name", "value"], _configuration_rows(specification.get("options") or {})
            if specification.get("options") else []))
        rows = []
        for name in fields:
            values = [_field_name(specification, name),
                      _field_description(specification, name).get("description") or "No description provided",
                      _field_description(specification, name).get("unit"), _field_type(report, name)]
            rows.append(_Html(f'<tr id="{escape(_definition_id(specification, name), quote=True)}">' +
                              "".join(f"<td>{_cell(value)}</td>" for value in values) + "</tr>"))
        result.append(_table(["field", "meaning", "unit", "type"], rows))
    return result

def _results(report: Mapping[str, Any]) -> list[str]:
    summaries = report.get("treatment_summaries", [])
    result = []
    for part in dict.fromkeys(row.get("part") for row in summaries):
        rows = [row for row in summaries if row.get("part") == part]
        result.append(f"<h3>{escape(_part_title(part))}</h3>")
        for specification, fields in _sections(report, "run"):
            result.append(f"<h4>{_type_name(specification.get('type') or specification['label'])}</h4>" + _table(
                ["measurement", "unit", *[row["treatment"] for row in rows]],
                [[_field_link(specification, name), _field_description(specification, name).get("unit"),
                  *[_statistic((row.get("metrics") or {}).get(name, {})) for row in rows]] for name in fields]))
    return result

def _pairs(report: Mapping[str, Any]) -> list[str]:
    pairs = report.get("pair_summaries", [])
    result = ["<p>Each paired difference is treatment B − treatment A, computed within a "
              "comparison group before averaging across groups.</p>"]
    if not pairs:
        return result + ['<p class="none">None</p>']
    for part in dict.fromkeys(row.get("part") for row in pairs):
        rows = [row for row in pairs if row.get("part") == part]
        treatments = list(dict.fromkeys(
            [row["treatment"] for row in report.get("treatment_summaries", []) if row.get("part") == part] +
            [name for row in rows for name in (row["treatment_a"], row["treatment_b"])]))
        result.append(f"<h3>{escape(_part_title(part))}</h3>")
        for specification, fields in _sections(report, "delta"):
            result.append(f"<h4>{_type_name(specification.get('type') or specification['label'])}</h4>")
            for index, reference in enumerate(treatments):
                others = [name for name in treatments if name != reference]
                result.append(f"<details{' open' if index == 0 else ''}><summary>"
                              f"Differences from {escape(reference)}</summary><p>"
                              "Each column shows B − A within comparison groups, where A is "
                              f"{escape(reference)}.</p>")
                measurements = []
                for name in fields:
                    values = []
                    for other in others:
                        pair: dict[str, Any] = next((row for row in rows if
                                     {row["treatment_a"], row["treatment_b"]} == {reference, other}), {})
                        statistic = dict((pair.get("deltas") or {}).get(name, {}))
                        if pair.get("treatment_b") == reference and statistic.get("mean") is not None:
                            statistic["mean"] = -float(statistic["mean"])
                        values.append(_statistic(statistic))
                    measurements.append([_field_link(specification, name),
                                         _field_description(specification, name).get("unit"), *values])
                result.append(_table(["measurement", "unit", *others], measurements))
                result.append("</details>")
        result.append("<h4>Pair metrics</h4>")
        pair_sections = _sections(report, "pair")
        if not pair_sections:
            result.append('<p class="none">None</p>')
        for specification, fields in pair_sections:
            result.append(f"<details><summary>{_type_name(specification.get('type') or specification['label'])}"
                          "</summary>" + _table(
                ["measurement", "unit", *[f"{row['treatment_a']} vs {row['treatment_b']}" for row in rows]],
                [[_field_link(specification, name), _field_description(specification, name).get("unit"),
                  *[_statistic((row.get("metrics") or {}).get(name, {})) for row in rows]] for name in fields]))
            result.append("</details>")
    return result

def _group_details(root: Path, report: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    units = {unit["index"]: unit for unit in manifest.get("units", [])}
    result = []
    for group in report.get("groups", []):
        index, seed = group["segment_index"], group["seed"]
        unit = units.get(index, {})
        title = (f"Segment {index} · seed {seed} · {group['scenario_id']} · "
                 f"frames {unit.get('start', '—')}–{unit.get('stop', '—')} · "
                 f"{group.get('part') or 'no split'}")
        result.append(f"<details><summary>{escape(title)}</summary>")
        treatments = group.get("treatments", {})
        rows: list[Any] = [["status", None, *[row["status"] for row in treatments.values()]],
                           ["attack", None, *[group.get("treatment_attacks", {}).get(name) or "none"
                                               for name in treatments]]]
        # One footnote per (field, reason); treatments sharing a reason share its number.
        reasons: dict[tuple[str, str], list[str]] = {}
        for specification, fields in _sections(report, "run"):
            rows.append(_Html(f'<tr class="metric-group"><th colspan="{len(treatments) + 2}">'
                              f"{_type_name(specification.get('type') or specification['label'])}</th></tr>"))
            for name in fields:
                values = []
                for treatment, row in treatments.items():
                    value = (row.get("metrics") or {}).get(name)
                    if value is None:
                        reason = row.get("reasons", {}).get(name) or row.get("error") or (
                            "No measurement was recorded for this run.")
                        key = (_field_name(specification, name), reason)
                        reasons.setdefault(key, []).append(treatment)
                        number = list(reasons).index(key) + 1
                        value = _Html(f'—<sup><a href="#reason-{index}-{seed}-{number}">[{number}]</a></sup>')
                    values.append(value)
                rows.append([_field_link(specification, name),
                             _field_description(specification, name).get("unit"), *values])
        result.append(_table(["measurement", "unit", *treatments], rows))
        if reasons:
            result.append('<ol class="footnotes">' + "".join(
                f'<li id="reason-{index}-{seed}-{number}">{escape(field)} '
                f"({escape(', '.join(names))}): {escape(reason)}</li>"
                for number, ((field, reason), names) in enumerate(reasons.items(), start=1)) + "</ol>")
        filename = f"evaluation/plots/segment-{index:05d}-seed-{seed}.png"
        if (root / filename).is_file():
            result.append(_image(filename, "Ego speed (m/s) after each decision. A second panel with the "
                                 "executed actuator brake appears only when the runs used actuator commands."))
        result.append("</details>")
    return result

def _selection(report: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    source = manifest.get("metadata_source")
    if source and source.get("load"):
        filename = escape(str(source["load"]))
        count = len(manifest.get("units", []))
        return [f"<p>Metadata loaded from {filename} ({count} segments); no selection was run.</p>"]
    rows = report.get("selections", [])
    if not rows:
        return []
    attacks = list(dict.fromkeys(row.get("attack_name", "Unspecified attack") for row in rows))
    statuses = list(dict.fromkeys(["selected", "not_selected", "error"] + [row["status"] for row in rows]))
    result = [_table(["attack", *statuses], [
        [attack, *[sum(row.get("attack_name", "Unspecified attack") == attack and row["status"] == status
                       for row in rows) for status in statuses]] for attack in attacks])]
    result.append("<p>For selected scenarios, notes list other candidates rejected within the same "
                  "scenario (reason × count). For not_selected scenarios, notes explain why no "
                  "candidate passed.</p>")
    for attack in attacks:
        result.append(f"<details><summary>{escape(attack)}</summary>" + _table(
            ["scenario", "status", "notes"],
            [[row.get("scenario_id"), row.get("status"),
              "; ".join([*row.get("reasons", []), *([row["error"]] if row.get("error") else [])]) or None]
             for row in rows if row.get("attack_name", "Unspecified attack") == attack]) + "</details>")
    return result

def write_evaluation_html(output_root: str | Path) -> Path:
    """Rebuild root evaluation_report.html from saved evaluation and preparation files."""
    root = Path(output_root)
    report = _load(root / "evaluation" / "report.json")
    manifest = _manifest(root)
    body = [_header([
        ("offline_report.html", "Offline report"), ("evaluation/report.json", "Evaluation JSON"),
        ("evaluation/runs.csv", "Runs CSV"),
    ])]
    sections = [
        ("how-to-read", "How to read this report"), ("experiment-setup", "Experiment setup"),
        ("run-status", "Run status"), ("metric-definitions", "Metric definitions"),
        ("results", "Results"), ("paired-differences", "Paired differences"),
        ("group-details", "Group details"), ("selection", "Selection"),
        ("errors-and-notes", "Errors and notes"),
    ]
    body.append('<nav aria-label="Contents"><h2>Contents</h2><ul>' +
                "".join(f"<li>{_link('#' + identifier, title)}</li>" for identifier, title in sections) +
                "</ul></nav>")
    how = [
        "<p>A treatment is one configured execution condition; clean means no attack. A comparison "
        "group contains the same segment and seed: all treatments start from the same initial state. "
        "A part is a dataset split: train, validation or test.</p>",
        "<p>Run status is completed when execution finishes successfully. Failed means an execution "
        "error; cancelled means execution was interrupted; incomplete means execution has not "
        "finished. Only completed runs contribute measurements to summaries.</p>",
        "<p>Numerical summaries show mean ± population standard deviation (ddof = 0); n is the "
        "number of groups with an available value. Standard deviation is omitted when n &lt; 2. "
        "Boolean summaries show the fraction that are "
        "true, followed by the true count and total group count.</p>",
        "<p>A paired difference is B − A, calculated within each group before summarizing. Booleans "
        "are subtracted as 1 and 0, so their mean difference is a difference in proportions.</p>",
        "<p>— means unavailable; the reasons are listed in Group details. Decimal values are shown "
        "with four decimal places; whole-number counts are shown as integers. JSON and CSV keep full precision.</p>",
    ]
    body.append(_section("how-to-read", "How to read this report", how))
    body.append(_section("experiment-setup", "Experiment setup", _experiment_setup(report, manifest)))
    body.append(_section("run-status", "Run status", _run_status(report)))
    body.append(_section("metric-definitions", "Metric definitions", _definitions(report)))
    body.append(_section("results", "Results", _results(report)))
    body.append(_section("paired-differences", "Paired differences", _pairs(report)))
    body.append(_section("group-details", "Group details", _group_details(root, report, manifest)))
    body.append(_section("selection", "Selection", _selection(report, manifest)))
    errors = report.get("metric_errors", [])
    error_content = ["<h3>Metric errors</h3>" + _table(
        ["metric", "error", "part", "segment", "seed", "treatment"],
        [[_type_name(row.get("metric")), row.get("error"), row.get("part"), row.get("segment_index"),
          row.get("seed"), row.get("treatment")] for row in errors])]
    notes = report.get("notes", [])
    error_content.append("<h3>Notes</h3>" + (
        "<ul>" + "".join(f"<li>{escape(str(note))}</li>" for note in notes) + "</ul>"
        if notes else '<p class="none">None</p>'))
    body.append(_section("errors-and-notes", "Errors and notes", error_content))
    return _document(f"Evaluation report — {root.name}", body, root / "evaluation_report.html")


__all__ = [
    'write_offline_html',
    'write_evaluation_html',
]
