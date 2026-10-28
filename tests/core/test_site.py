"""Project website: documentation link rewriting, table escapes and the homepage media budget."""

import re
from pathlib import Path

from scripts.website.hooks import rewrite_repository_links, unescape_table_code_pipes

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "website" / "content" / "assets" / "media"
VIDEO_LIMIT = 4 * 1024 * 1024
MEDIA_LIMIT = 15 * 1024 * 1024
BLOB = "https://github.com/AVSecProject/AVSecTester/blob/v1"
TREE = "https://github.com/AVSecProject/AVSecTester/tree/v1"


def test_links_leaving_docs_point_at_the_repository(tmp_path):
    (tmp_path / "configs").mkdir()
    markdown = "\n".join([
        "[guide](../guides/attacks.md#online-phase) and [top](#overview)",
        "[config](../../configs/x.yaml#L3) and [folder](../../configs)",
        "![figure](../assets/figure.png) and [site](https://example.org/a.md)",
        "[todo](../development/TODO.md#roadmap)",
        "`[code](../../configs/x.yaml)`",
        "```text",
        "[fenced](../../configs/x.yaml)",
        "```",
    ])
    rewritten = rewrite_repository_links(markdown, "docs/examples/README.md", "v1", tmp_path)
    assert rewritten.splitlines() == [
        "[guide](../guides/attacks.md#online-phase) and [top](#overview)",
        f"[config]({BLOB}/configs/x.yaml#L3) and [folder]({TREE}/configs)",
        "![figure](../assets/figure.png) and [site](https://example.org/a.md)",
        f"[todo]({BLOB}/docs/development/TODO.md#roadmap)",
        "`[code](../../configs/x.yaml)`",
        "```text",
        "[fenced](../../configs/x.yaml)",
        "```",
    ]


def test_table_code_pipes_are_unescaped_only_in_table_rows():
    markdown = "\n".join([
        r"| `a \| b` | text \| kept |",
        r"Prose `a \| b` keeps its escape.",
        "```text",
        r"| `fenced \| code` |",
        "```",
    ])
    assert unescape_table_code_pipes(markdown).splitlines() == [
        r"| `a | b` | text \| kept |",
        r"Prose `a \| b` keeps its escape.",
        "```text",
        r"| `fenced \| code` |",
        "```",
    ]


def test_homepage_media_exist_are_referenced_and_fit_the_budget():
    homepage = (ROOT / "website" / "content" / "index.md").read_text(encoding="utf-8")
    referenced = set(re.findall(r"assets/media/([^\s\"')]+)", homepage))
    stored = {path.name for path in MEDIA.iterdir() if path.is_file()}
    assert referenced and referenced == stored
    videos = [path for path in MEDIA.iterdir() if path.suffix == ".mp4"]
    assert videos and all(path.stat().st_size <= VIDEO_LIMIT for path in videos)
    assert sum(path.stat().st_size for path in MEDIA.iterdir()) <= MEDIA_LIMIT
