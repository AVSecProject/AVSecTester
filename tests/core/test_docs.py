"""Documentation checks: links and anchors resolve, code blocks parse, runnable examples run.

Mark a self-contained Python example as runnable by placing ``<!-- runnable -->`` on the line
directly before its code fence. Runnable examples must work offline with the core test
dependencies, without simulator servers, GPUs or model weights.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = sorted(
    [ROOT / "README.md", ROOT / "scripts" / "README.md", ROOT / "tests" / "README.md",
     ROOT / "benchmarks" / "README.md", ROOT / "models" / "README.md"]
    + list((ROOT / "docs").rglob("*.md"))
)
RUNNABLE = "<!-- runnable -->"
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
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
        runnable = index > 0 and lines[index - 1].strip() == RUNNABLE
        start = index + 1
        index += 1
        while index < len(lines) and not lines[index].startswith(fence):
            index += 1
        yield language, "\n".join(lines[start:index]), start + 1, runnable
        index += 1


def _prose(text: str) -> str:
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
        kept.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(kept)


def _slug(heading: str) -> str:
    text = re.sub(r"`", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    anchors = set()
    for line in _prose(path.read_text()).splitlines():
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
    for target in LINK.findall(_prose(path.read_text())):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target):
            continue
        file_part, _, anchor = target.partition("#")
        resolved = (path.parent / file_part).resolve() if file_part else path
        if not resolved.exists():
            broken.append(f"{target}: missing file")
        elif anchor and resolved.suffix == ".md" and anchor not in _anchors(resolved):
            broken.append(f"{target}: missing anchor")
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
