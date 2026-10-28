"""Documentation contracts: public imports, generated API tables, links and offline runnable examples.

Mark a self-contained example as runnable by placing ``<!-- runnable -->`` on the line directly
before its code fence. Runnable Python examples are executed. Runnable YAML examples are complete
experiment configurations loaded with ``load_configuration``. Other supported blocks are parsed.
Runnable examples must work offline with the core test dependencies, without simulator servers,
GPUs or model weights.
"""

from __future__ import annotations

import ast
import fnmatch
import importlib
import importlib.util
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from scripts.development import write_api_reference as api_reference


ROOT = Path(__file__).resolve().parents[2]


def documentation_paths():
    excluded = {"third_party", "tmp", "results", ".git", ".venv", "node_modules"}
    try:
        names = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).splitlines()
        paths = [ROOT / name for name in names]
    except (OSError, subprocess.CalledProcessError):
        paths = []
        patterns = [line.strip().strip("/") for line in (ROOT / ".gitignore").read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith(("#", "!"))]
        for directory, subdirectories, files in os.walk(ROOT):
            relative = Path(directory).relative_to(ROOT)
            subdirectories[:] = [
                name for name in subdirectories
                if not name.startswith(".") and name not in excluded
                and not any(
                    fnmatch.fnmatch((relative / name).as_posix(), pattern)
                    or ("/" not in pattern and fnmatch.fnmatch(name, pattern))
                    for pattern in patterns)]
            paths.extend(Path(directory) / name for name in files if name.endswith(".md"))
    # The homepage in website/ links to site paths; mkdocs build --strict checks those links.
    return sorted(path for path in paths if path.is_file()
                  and not excluded.intersection(path.relative_to(ROOT).parts)
                  and path.relative_to(ROOT).parts[0] != "website")


DOCS = documentation_paths()

RUNNABLE = "<!-- runnable -->"
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
HTML_IMAGE = re.compile(r"""<img[^>]+src=["']([^"']+)["']""", re.I)
FENCE = re.compile(r"^(```+|~~~+)\s*([\w+-]*)")


def _blocks(text: str):
    """Yield (language, code, start_line, runnable) for each fenced block."""
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = FENCE.match(lines[index])
        if not match:
            index += 1
            continue
        fence, language = match.groups()
        language = "python" if language == "py" else language
        runnable = index > 0 and lines[index - 1].strip() == RUNNABLE
        start = index + 1
        index += 1
        while index < len(lines) and not lines[index].startswith(fence):
            index += 1
        yield language, "\n".join(lines[start:index]), start + 1, runnable
        index += 1


def _prose(text: str, *, inline_code: bool = False) -> str:
    """Remove fenced blocks and inline code so only rendered Markdown remains."""
    kept, fence = [], None
    for line in text.splitlines():
        match = FENCE.match(line)
        if fence is None and match:
            fence = match.group(1)
            continue
        if fence is not None:
            if line.startswith(fence):
                fence = None
            continue
        kept.append(line if inline_code else re.sub(r"`[^`]*`", "", line))
    return "\n".join(kept)


def _slug(heading: str) -> str:
    text = re.sub(r"`", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    anchors = set()
    for line in _prose(path.read_text(), inline_code=True).splitlines():
        match = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not match:
            continue
        slug = _slug(match.group(1))
        count = seen.get(slug, 0)
        anchors.add(slug if count == 0 else f"{slug}-{count}")
        seen[slug] = count + 1
    return anchors


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_links_and_anchors_resolve(path):
    broken = []
    for target in (LINK.findall(_prose(path.read_text()))
                   + HTML_IMAGE.findall(_prose(path.read_text()))):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target):
            continue
        file_part, _, anchor = target.partition("#")
        resolved = (path.parent / file_part).resolve() if file_part else path
        if not resolved.exists():
            broken.append(f"{target}: missing file")
        elif anchor and resolved.suffix == ".md" and anchor not in _anchors(resolved):
            broken.append(f"{target}: missing anchor")
        elif anchor and resolved.suffix == ".md":
            headings = [
                _slug(match[1])
                for line in _prose(resolved.read_text(), inline_code=True).splitlines()
                if (match := re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line))
            ]
            if headings.count(anchor) > 1:
                broken.append(f"{target}: ambiguous repeated heading")
    assert not broken, "\n".join(broken)


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_code_blocks_parse(path):
    errors = []
    for language, code, line, _ in _blocks(path.read_text()):
        try:
            if language == "python":
                compile(code, f"{path.name}:{line}", "exec")
            elif language in {"yaml", "yml"}:
                yaml.safe_load(code)
        except (SyntaxError, yaml.YAMLError) as error:
            errors.append(f"line {line}: {error}")
    assert not errors, "\n".join(errors)


RUNNABLE_BLOCKS = [
    pytest.param(code, id=f"{path.relative_to(ROOT)}:{line}")
    for path in DOCS
    for language, code, line, runnable in _blocks(path.read_text())
    if runnable and language == "python"
]


@pytest.mark.parametrize("code", RUNNABLE_BLOCKS)
def test_runnable_examples(code, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Register the example as a module so dataclasses and pickling resolve its definitions.
    module = types.ModuleType("doc_example")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(code, "<doc example>", "exec"), module.__dict__)


RUNNABLE_YAML_BLOCKS = [
    pytest.param(code, id=f"{path.relative_to(ROOT)}:{line}")
    for path in DOCS
    for language, code, line, runnable in _blocks(path.read_text())
    if runnable and language in {"yaml", "yml"}
]
RUNNABLE_LANGUAGES = {"python", "yaml", "yml"}


def test_runnable_marker_is_only_used_for_supported_languages():
    unsupported = [f"{path.relative_to(ROOT)}:{line} ({language or 'no language'})"
                   for path in DOCS
                   for language, _code, line, runnable in _blocks(path.read_text())
                   if runnable and language not in RUNNABLE_LANGUAGES]
    assert not unsupported, f"Runnable blocks must be Python or YAML: {unsupported}"


@pytest.mark.parametrize("code", RUNNABLE_YAML_BLOCKS)
def test_runnable_yaml_examples_load(code, tmp_path):
    from avsectester.experiment import load_configuration

    document = yaml.safe_load(code)
    assert isinstance(document, dict), "A runnable YAML example must be a mapping"
    path = tmp_path / "experiment.yaml"
    path.write_text(code)
    load_configuration(path)


def _import_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name and not error.name.startswith("avsectester"):
            pytest.skip(f"Optional dependency {error.name} unavailable for {name}")
        raise


def python_imports():
    for path in DOCS:
        for language, code, line, runnable in _blocks(path.read_text()):
            if language != "python" or runnable:
                continue
            for statement in ast.walk(ast.parse(code)):
                if isinstance(statement, ast.Import):
                    for alias in statement.names:
                        if alias.name.startswith("avsectester"):
                            yield pytest.param(
                                alias.name, None,
                                id=f"{path.relative_to(ROOT)}:{line}:{alias.name}")
                elif (isinstance(statement, ast.ImportFrom) and statement.module
                      and statement.module.startswith("avsectester")):
                    for name in statement.names:
                        yield pytest.param(
                            statement.module, name.name,
                            id=f"{path.relative_to(ROOT)}:{line}:{statement.module}.{name.name}")


@pytest.mark.parametrize("module_name,name", list(python_imports()))
def test_python_imports_resolve(module_name, name):
    module = _import_module(module_name)
    assert hasattr(module, "__all__"), f"Internal module {module_name}"
    if name not in (None, "*"):
        assert name in module.__all__, f"{module_name}.{name} is not exported"


def test_public_api_reference():
    generator = api_reference
    text = (ROOT / "docs/reference/python-api.md").read_text()
    expected = generator.render(text)
    blocks = re.compile(r"<!-- api:begin module=(.*?) -->(.*?)<!-- api:end -->", re.S)
    actual_sections = dict(blocks.findall(text))
    for module, block in blocks.findall(expected):
        assert actual_sections.get(module) == block, f"Regenerate API module {module}"
    sections = re.findall(r"^### (avsectester[\w.]*)$", text, re.M)
    assert len(sections) == len(set(sections))
    assert text == expected, "API reference structure or handwritten section order differs"
    assert set(sections) == set(generator.public_modules())
    for module, block in re.findall(r"<!-- api:begin module=(.*?) -->(.*?)<!-- api:end -->", text, re.S):
        if module == "module-map":
            continue
        names = []
        for line in block.splitlines():
            if not line.startswith("|") or line.startswith(("| Name and signature", "|---")):
                continue
            match = re.fullmatch(r"\| `([^`]+)` \| .+ \|", line)
            assert match, f"Unparseable row in {module}: {line}"
            name = re.split(r"[(: =]", match[1], maxsplit=1)[0]
            assert name not in names, f"Duplicate {module}.{name}"
            names.append(name)


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_documentation_format(path):
    prose = _prose(path.read_text(), inline_code=True)
    errors = []
    lines = prose.splitlines()
    for number, line in enumerate(lines, 1):
        if re.search(r"`[^`]+`[ \t]+(?:\.\w|'s\b|=|-\w|[:),]|/(?! )|\((?![A-Za-z`\d][^)]*\)))", line):
            errors.append(f"{number}: separated code fragment: {line}")
        if any(re.search(r"&(?:#\d+|#x[\da-f]+|\w+);", code, re.I)
               for code in re.findall(r"`([^`]+)`", line)):
            errors.append(f"{number}: HTML entity in code: {line}")
        if re.search(r"\b[A-Za-z]{2,}- (?!(?:and|or|to)\b)[A-Za-z]{2,}\b", line):
            errors.append(f"{number}: split word: {line}")
        if line.lstrip().startswith("> >"):
            errors.append(f"{number}: nested task header")
        header_fields = re.findall(r"\*\*(?:Audience|Outcome|Prerequisites):\*\*",
                            re.sub(r"`[^`]*`", "", line))
        if len(header_fields) > 1:
            errors.append(f"{number}: multiple task-header fields")
        if header_fields and not line.startswith(">"):
            errors.append(f"{number}: task header must be quoted")
        if re.search(r"(?:^|:\s+)\d+\.\s+\S.*\s+\d+\.\s+\S", line.strip()):
            errors.append(f"{number}: multiple ordered-list items")
        if (re.fullmatch(r"[A-Za-z][A-Za-z-]*[.,;:]?", line.strip())
                and number < len(lines) and lines[number].strip()
                and not line.rstrip().endswith(":")
                and not lines[number].startswith(("#", "|", ">", "-", "*"))
                and not re.match(r"^\s*\d+\.\s", lines[number])):
            errors.append(f"{number}: isolated word: {line}")
    header = re.search(r"^>.*\*\*Audience:\*\*.*?(?=\n\n|\Z)", prose, re.M | re.S)
    if header and any(not line.startswith(">") for line in header[0].splitlines()):
        errors.append("Every task-header line must start with >")
    for first, second in zip(lines, lines[1:]):
        if re.match(r"^> \*\*(?:Audience|Outcome):", first) and re.match(
                r"^> \*\*(?:Outcome|Prerequisites):", second):
            errors.append("Task-header fields need a separate > line")
    if path.name == "python-api.md":
        assert "lambda:" not in prose and "&#124;" not in prose
    assert not errors, "\n".join(errors)
