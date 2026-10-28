"""Check a built site against the repository's documentation.

Usage: python -m scripts.website.check_site SITE_DIR

Run after ``mkdocs build --strict -f website/mkdocs.yml``. The checks are:

* every published Markdown file below docs/ has a page at its /docs/ URL, and repository-only
  pages have none;
* code in the Python API reference shows no table-escape backslash before ``|``;
* every link to a repository file or directory on GitHub names a path that exists here.
"""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

from scripts.website.hooks import UNPUBLISHED_DOCS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_LINK = re.compile(
    r"^https://github\.com/AVSecProject/AVSecTester/(?:blob|tree)/[^/]+/([^#?]+)")


def page_path(markdown: Path, docs: Path) -> Path:
    """Built HTML path of one docs/ Markdown file, with directory URLs."""
    relative = markdown.relative_to(docs).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "README":
        parts = parts[:-1]
    return Path("docs", *parts, "index.html")


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.code: list[str] = []
        self._code_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "code":
            self._code_depth += 1
            self.code.append("")
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.links.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag == "code" and self._code_depth:
            self._code_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._code_depth:
            self.code[-1] += data


def collect(path: Path) -> _Collector:
    collector = _Collector()
    collector.feed(path.read_text(encoding="utf-8"))
    return collector


def check(site: Path, root: Path = REPOSITORY_ROOT) -> list[str]:
    errors = []
    docs = root / "docs"
    for markdown in sorted(docs.rglob("*.md")):
        published = (site / page_path(markdown, docs)).is_file()
        if markdown.relative_to(root).as_posix() in UNPUBLISHED_DOCS:
            if published:
                errors.append(f"Repository-only page is published: {markdown.relative_to(root)}")
        elif not published:
            errors.append(f"No page for {markdown.relative_to(root)}")

    api = site / "docs" / "reference" / "python-api" / "index.html"
    if api.is_file():
        escaped = [code for code in collect(api).code if "\\|" in code]
        errors.extend(f"Escaped pipe in Python API code: {code}" for code in escaped)
    else:
        errors.append("No Python API reference page")

    for page in sorted(site.rglob("*.html")):
        for link in collect(page).links:
            match = REPOSITORY_LINK.match(link)
            if match and not (root / match.group(1)).exists():
                errors.append(f"{page.relative_to(site)}: GitHub link to a missing path: {link}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("site", type=Path, help="Built site directory")
    errors = check(parser.parse_args().site)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Site check: {'failed' if errors else 'passed'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
