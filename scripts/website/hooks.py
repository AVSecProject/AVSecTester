"""MkDocs hooks that publish docs/ unchanged beside the homepage in website/.

The repository's docs/ directory stays the only documentation source and keeps rendering on
GitHub. These hooks add its files to the site under the ``docs/`` prefix, point links that leave
docs/ or reach a repository-only page at the repository on GitHub, and undo the table-cell pipe
escapes that only GitHub needs.

Only the standard library is imported at module level, so the pure functions below can be
tested without MkDocs. MkDocs objects are used by duck typing.

Run ``python -m scripts.website.hooks docs/PAGE.md`` to print a page's Markdown as the site
renders it, with repository links and table escapes transformed.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import re
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_URL = "https://github.com/AVSecProject/AVSecTester"
DOCS_PREFIX = "docs/"
NO_EDIT_LINK = {"index.md", "docs/reference/python-api.md"}
# docs/ pages kept on GitHub only; the site does not publish them and links to them go to GitHub.
UNPUBLISHED_DOCS = frozenset({"docs/development/TODO.md"})

_FENCE = re.compile(r"^\s*(```|~~~)")
_CODE_SPAN = re.compile(r"(`+)(.+?)\1")
_INLINE_LINK = re.compile(r"(!?\[[^\]\n]*\]\()([^)\s]+)((?:\s+\"[^\"\n]*\")?\))")
_VERSION = re.compile(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", re.MULTILINE)


def read_version(init_text: str) -> str:
    """Return ``__version__`` from the text of avsectester/__init__.py without importing it."""
    match = _VERSION.search(init_text)
    if match is None:
        raise ValueError("avsectester/__init__.py does not define __version__")
    return match.group(1)


def _outside_code(line: str, transform: Any) -> str:
    """Apply ``transform`` to the parts of one line that are not inline code spans."""
    parts, position = [], 0
    for match in _CODE_SPAN.finditer(line):
        parts.append(transform(line[position:match.start()]))
        parts.append(match.group(0))
        position = match.end()
    parts.append(transform(line[position:]))
    return "".join(parts)


def _prose_lines(markdown: str, transform: Any) -> str:
    """Apply ``transform`` to every line outside fenced code blocks."""
    lines, fence = [], None
    for line in markdown.splitlines(keepends=True):
        match = _FENCE.match(line)
        if fence is None and match:
            fence = match.group(1)
        elif fence is not None and line.lstrip().startswith(fence):
            fence = None
        elif fence is None:
            line = transform(line)
        lines.append(line)
    return "".join(lines)


def repository_url(path: str, ref: str, *, directory: bool) -> str:
    """GitHub URL of a repository path at ``ref``; directories use tree/, files blob/."""
    kind = "tree" if directory else "blob"
    return f"{REPOSITORY_URL}/{kind}/{ref}/{path}"


def rewrite_repository_links(
    markdown: str, page_path: str, ref: str, root: Path = REPOSITORY_ROOT,
) -> str:
    """Point relative links that leave docs/ or reach UNPUBLISHED_DOCS at GitHub.

    ``page_path`` is the page's repository path, such as ``docs/guides/attacks.md``. Links to
    published docs/ files, absolute URLs, pure anchors, fenced code and inline code are unchanged.
    The fragment is preserved.
    """
    base = posixpath.dirname(page_path)

    def rewrite(match: re.Match[str]) -> str:
        target = match.group(2)
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith(("#", "/")):
            return match.group(0)
        path, hash_mark, fragment = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(base, path))
        published = ((resolved == "docs" or resolved.startswith(DOCS_PREFIX))
                     and resolved not in UNPUBLISHED_DOCS)
        if published or resolved.startswith(".."):
            return match.group(0)
        url = repository_url(resolved, ref, directory=(root / resolved).is_dir())
        return f"{match.group(1)}{url}{hash_mark}{fragment}{match.group(3)}"

    return _prose_lines(markdown, lambda line: _outside_code(line, lambda text: _INLINE_LINK.sub(rewrite, text)))


def unescape_table_code_pipes(markdown: str) -> str:
    """Replace ``\\|`` with ``|`` inside code spans of table rows only.

    GitHub needs the escape to keep a pipe in a table cell's code; Python-Markdown shows the
    backslash literally and does not split cells at pipes inside code.
    """
    def row(line: str) -> str:
        if not line.lstrip().startswith("|"):
            return line
        return _CODE_SPAN.sub(lambda match: match.group(0).replace("\\|", "|"), line)

    return _prose_lines(markdown, row)


def documentation_files(root: Path = REPOSITORY_ROOT) -> list[tuple[str, Path]]:
    """Every published file below docs/ as (site source URI, absolute path), in a stable order."""
    docs = root / "docs"
    found = []
    for path in sorted(docs.rglob("*")):
        relative = path.relative_to(docs)
        src_uri = DOCS_PREFIX + relative.as_posix()
        if (path.is_file() and not any(part.startswith(".") for part in relative.parts)
                and src_uri not in UNPUBLISHED_DOCS):
            found.append((src_uri, path))
    return found


# MkDocs event hooks --------------------------------------------------------------------------

def on_config(config: Any) -> Any:
    init_text = (REPOSITORY_ROOT / "avsectester" / "__init__.py").read_text(encoding="utf-8")
    config.extra["version"] = read_version(init_text)
    return config


def on_files(files: Any, config: Any) -> Any:
    from mkdocs.structure.files import File

    for src_uri, path in documentation_files(REPOSITORY_ROOT):
        file = File.generated(config, src_uri, abs_src_path=str(path))
        # Edit links point at the real source under docs/ on GitHub.
        file.edit_uri = src_uri
        files.append(file)
    return files


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    src_uri = page.file.src_uri
    if src_uri in NO_EDIT_LINK:
        page.edit_url = None
    if src_uri.startswith(DOCS_PREFIX):
        ref = os.environ.get("GITHUB_SHA") or "main"
        markdown = rewrite_repository_links(markdown, src_uri, ref, REPOSITORY_ROOT)
        markdown = unescape_table_code_pipes(markdown)
    return markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a docs/ page's Markdown as the project site renders it.")
    parser.add_argument("page", type=Path, help="Markdown file below docs/")
    parser.add_argument("--ref", default=os.environ.get("GITHUB_SHA") or "main",
                        help="Git ref used in GitHub links (default: GITHUB_SHA or main)")
    arguments = parser.parse_args()
    page = arguments.page.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    if not page.startswith(DOCS_PREFIX):
        parser.error("page must be a Markdown file below docs/")
    markdown = (REPOSITORY_ROOT / page).read_text(encoding="utf-8")
    print(unescape_table_code_pipes(rewrite_repository_links(markdown, page, arguments.ref)), end="")


if __name__ == "__main__":
    main()
